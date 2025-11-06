//! Module for the io_uring device's buf_ring feature.
//!
//! This implementation provides a safe Rust wrapper around io_uring buffer rings,
//! which allow the kernel to select buffers automatically for I/O operations.
//!
//! ## Design
//!
//! - **Ring entries**: Page-aligned array of `BufRingEntry` structures that the kernel reads from
//! - **Buffers**: Caller-provided buffer memory that we take ownership of
//! - **Reference counting**: Track which buffers are in use to know when to return them to the ring
//!
//! ## Memory Layout
//!
//! The ring entries are allocated separately from the buffer data. This allows callers to use
//! custom allocation strategies for buffers (e.g., DMA memory, hugepages) while we manage
//! the ring structure that io_uring requires.

// Developer's note about io_uring return codes when a buf_ring is used:
//
// While a buf_ring pool is exhausted, new calls to read that are, or are not, ready to read will
// fail with the 105 error, "no buffers", while existing calls that were waiting to become ready to
// read will not fail. Only when the data becomes ready to read will they fail, if the buffer ring
// is still empty at that time. This makes sense when thinking about it from how the kernel
// implements the start of a read command; it can be confusing when first working with these
// commands from the userland perspective.

use io_uring::types;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::{Cell, RefCell};
use std::io;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr::NonNull;
use std::rc::Rc;

use super::tail::RingTail;
use super::provided::ProvidedBuffer;
use super::{Bgid, Bid};

/// A page-aligned vector for io_uring ring entries.
///
/// io_uring requires the buffer ring structure to be page-aligned. This type provides
/// a safe wrapper around a manually-allocated, page-aligned array of entries.
struct PageAlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
    _marker: PhantomData<T>,
}

impl<T> PageAlignedVec<T> {
    /// Create a new page-aligned vector with the given length.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The layout is invalid
    /// - Allocation fails
    fn new(len: usize) -> io::Result<Self> {
        const PAGE_SIZE: usize = 4096;

        let layout = Layout::from_size_align(len * size_of::<T>(), PAGE_SIZE)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid layout"))?;

        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr as *mut T).ok_or_else(|| {
            io::Error::new(io::ErrorKind::OutOfMemory, "allocation failed")
        })?;

        Ok(Self {
            ptr,
            len,
            layout,
            _marker: PhantomData,
        })
    }

    /// Get a pointer to the start of the allocation.
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get a mutable pointer to the start of the allocation.
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get the length of the vector.
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for PageAlignedVec<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

// Safety: PageAlignedVec is not Send/Sync because it's part of the single-threaded io_uring design

/// A `BufRing` represents the ring and the buffers used with the kernel's io_uring buf_ring
/// feature.
///
/// This type provides a safe Rust interface to io_uring's buffer ring functionality, which allows
/// the kernel to automatically select buffers for I/O operations without per-operation syscalls.
///
/// ## Usage
///
/// 1. Create buffers (can use custom allocators, DMA memory, etc.)
/// 2. Create a `BufRing` and transfer ownership of buffers
/// 3. Register with io_uring
/// 4. Submit I/O operations with `IOSQE_BUFFER_SELECT` flag
/// 5. Kernel returns `ProvidedBuffer` through CQE
///
/// ## Reference Counting
///
/// BufRings are reference counted via `Rc<RawBufRing>`. The ring stays alive as long as any
/// `ProvidedBuffer` exists. Individual buffers are tracked by ID and automatically returned to
/// the ring when all references are dropped.
///
/// ## Memory Management
///
/// The ring takes ownership of buffer memory but doesn't allocate it. This allows callers to:
/// - Use custom allocators (jemalloc, mimalloc, etc.)
/// - Allocate DMA-capable memory
/// - Use hugepages for performance
/// - Pre-initialize buffer contents
///
/// The ring structure itself is allocated internally with page alignment as required by io_uring.
#[derive(Clone)]
pub struct BufRing {
    pub (crate) raw: Rc<RawBufRing>,
}

/// Internal buffer ring implementation.
///
/// This type owns all memory and implements the actual buffer ring logic. It's wrapped in
/// `Rc` by `BufRing` to allow `ProvidedBuffer` instances to keep the ring alive.
pub(crate) struct RawBufRing {
    /// Buffer group ID
    bgid: Bgid,

    /// Ring size mask (ring_entries - 1, used for fast modulo via bitwise AND)
    /// Example: if ring has 16 entries, mask is 15 (0b1111)
    ring_entries_mask: u16,

    /// Number of buffers
    buf_cnt: u16,

    /// Size of each buffer in bytes
    buf_len: usize,

    /// Page-aligned ring entries that io_uring reads from
    ring_entries: PageAlignedVec<types::BufRingEntry>,

    /// Reference count for each buffer (indexed by bid)
    /// When count reaches 0, buffer is returned to ring
    ref_counts: RefCell<Vec<u32>>,

    /// Buffer memory (owned, provided by caller)
    buffers: Box<[u8]>,

    /// Ring tail management (wraps C union complexity)
    tail: RefCell<RingTail>,

    /// Whether this ring is registered with io_uring
    registered: Cell<bool>,
}

impl BufRing {
    /// Create a new buffer ring.
    ///
    /// # Arguments
    ///
    /// * `bgid` - Buffer group ID (must be unique within the io_uring instance)
    /// * `buffers` - Buffer memory to take ownership of
    /// * `buf_len` - Size of each individual buffer in bytes
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `buffers.len()` is not a multiple of `buf_len`
    /// - Buffer count is 0 or exceeds maximum (32768)
    /// - Buffer count is not a power of 2
    /// - Ring entries allocation fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use tokio_uring::buf::ring::BufRing;
    ///
    /// // Create 16 buffers of 4KB each
    /// let buffers = vec![0u8; 16 * 4096];
    /// let ring = BufRing::new(0, buffers, 4096)?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn new(bgid: Bgid, buffers: Vec<u8>, buf_len: usize) -> io::Result<Self> {
        // Validate buf_len
        if buf_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "buf_len must be > 0",
            ));
        }

        // Validate buffers length
        if buffers.len() % buf_len != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "buffers.len() must be a multiple of buf_len",
            ));
        }

        let buf_cnt = buffers.len() / buf_len;

        // Validate buffer count
        if buf_cnt == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "must have at least one buffer",
            ));
        }

        if buf_cnt > 32768 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "buffer count exceeds maximum (32768)",
            ));
        }

        // Buffer count must be power of 2
        if !buf_cnt.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "buffer count must be a power of 2",
            ));
        }

        let buf_cnt = buf_cnt as u16;
        let ring_entries_mask = buf_cnt - 1;

        // Allocate page-aligned ring entries
        let mut ring_entries = PageAlignedVec::<types::BufRingEntry>::new(buf_cnt as usize)?;

        // Initialize ring tail
        let tail = unsafe { RingTail::new(ring_entries.as_ptr()) };

        // Initialize reference counts (all 0)
        let ref_counts = vec![0u32; buf_cnt as usize];

        // Initialize all ring entries and add buffers to ring
        let buffers_ptr = buffers.as_ptr();
        for bid in 0..buf_cnt {
            let buffer_addr = unsafe { buffers_ptr.add(bid as usize * buf_len) };

            unsafe {
                let entry = ring_entries.as_mut_ptr().add(bid as usize);
                types::BufRingEntry::set(
                    entry,
                    buffer_addr as u64,
                    buf_len as u32,
                    bid,
                );
            }

            // Don't increment ref count yet - buffers start in the ring
        }

        // Set initial tail to indicate all buffers are available
        let mut tail_mut = tail;
        for _ in 0..buf_cnt {
            tail_mut.increment();
        }
        tail_mut.sync();

        let raw = RawBufRing {
            bgid,
            ring_entries_mask,
            buf_cnt,
            buf_len,
            ring_entries,
            buffers: buffers.into_boxed_slice(),
            tail: RefCell::new(tail_mut),
            ref_counts: RefCell::new(ref_counts),
            registered: Cell::new(false),
        };

        Ok(Self {
            raw: Rc::new(raw),
        })
    }

    /// Get the buffer group ID.
    #[inline]
    pub fn bgid(&self) -> Bgid {
        self.raw.bgid()
    }
}

impl RawBufRing {
    /// Get the buffer group ID.
    #[inline]
    pub(crate) fn bgid(&self) -> Bgid {
        self.bgid
    }

    /// Get the number of buffers in this ring.
    #[inline]
    pub(crate) fn buffer_count(&self) -> u16 {
        self.buf_cnt
    }

    /// Get the size of each buffer.
    #[inline]
    pub(crate) fn buffer_len(&self) -> usize {
        self.buf_len
    }

    /// Get a pointer to a specific buffer's memory.
    ///
    /// # Safety
    ///
    /// - `bid` must be < `buffer_count()`
    #[inline]
    pub(crate) unsafe fn buffer_ptr(&self, bid: Bid) -> *const u8 {
        debug_assert!((bid as usize) < self.buf_cnt as usize);
        self.buffers.as_ptr().add(bid as usize * self.buf_len)
    }

    /// Increment the reference count for a buffer.
    ///
    /// Called when:
    /// - Buffer is returned from CQE
    /// - ProvidedBuffer is cloned
    pub(crate) fn increment_ref(&self, bid: Bid) {
        let mut ref_counts = self.ref_counts.borrow_mut();
        ref_counts[bid as usize] += 1;
    }

    /// Decrement the reference count for a buffer.
    ///
    /// If count reaches 0, the buffer is returned to the ring.
    /// Called when ProvidedBuffer is dropped.
    pub(crate) fn decrement_ref(&self, bid: Bid) {
        let mut ref_counts = self.ref_counts.borrow_mut();
        let count = &mut ref_counts[bid as usize];
        *count -= 1;

        if *count == 0 {
            drop(ref_counts); // Release borrow before calling return_buffer
            self.return_buffer(bid);
        }
    }

    /// Get the current reference count for a buffer (for debugging).
    pub(crate) fn ref_count(&self, bid: Bid) -> Option<u32> {
        self.ref_counts.borrow().get(bid as usize).copied()
    }

    /// Return a buffer to the ring for kernel reuse.
    fn return_buffer(&self, bid: Bid) {
        let mut tail = self.tail.borrow_mut();

        // Calculate ring entry index
        let idx = tail.local() & self.ring_entries_mask;

        // Get the ring entry
        unsafe {
            let entry = self.ring_entries.as_ptr().add(idx as usize) as *mut types::BufRingEntry;
            let buffer_addr = self.buffer_ptr(bid);

            // Fill in the entry
            types::BufRingEntry::set(entry, buffer_addr as u64, self.buf_len as u32, bid);
        }

        // Increment tail and sync to kernel
        tail.increment();
        tail.sync();
    }

    /// Extract a ProvidedBuffer from a CQE.
    ///
    /// This is called when the kernel returns a buffer through a completion.
    pub(crate) fn get_buffer_from_cqe(
        self: &Rc<Self>,
        cqe_flags: u32,
        len: u32,
    ) -> Option<super::provided::ProvidedBuffer> {
        // Extract buffer ID from CQE flags (upper 16 bits)
        const BUFFER_ID_SHIFT: u32 = 16;
        let bid = (cqe_flags >> BUFFER_ID_SHIFT) as Bid;

        // Increment ref count
        self.increment_ref(bid);

        // Get buffer pointer
        let ptr = unsafe { self.buffer_ptr(bid) };

        // Create ProvidedBuffer
        Some(ProvidedBuffer::new(BufRing{raw: self.clone()}, bid, ptr, len))
    }
}

impl Drop for RawBufRing {
    fn drop(&mut self) {
        // TODO: Unregister with io_uring if registered
        if self.registered.get() {
            // Will implement when we add registration support
        }

        // PageAlignedVec and Box<[u8]> will clean up automatically
    }
}

// Safety: RawBufRing is not Send/Sync because:
// 1. Uses Rc (not Arc)
// 2. RefCell is not thread-safe
// 3. io_uring is single-threaded by design
