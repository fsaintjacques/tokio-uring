//! Provided buffer type for io_uring buffer rings.
//!
//! This module implements the buffer type returned from I/O operations when using
//! io_uring buffer rings with the `IOSQE_BUFFER_SELECT` flag.
//!
//! ## Reference Counting
//!
//! The buffer ring tracks reference counts for each buffer (by bid). When a buffer
//! is returned from an I/O operation, its reference count is incremented. When the
//! user clones a `ProvidedBuffer`, the count is incremented. When a `ProvidedBuffer`
//! is dropped, the count is decremented. When the count reaches zero, the buffer
//! is returned to the ring for reuse by the kernel.
//!
//! This design supports the `IOU_PBUF_RING_INC` feature (kernel 6.12+) where multiple
//! I/O operations can consume different portions of the same buffer (same bid). All
//! portions must be dropped before the buffer is returned to the ring.

use super::ring::BufRing;
use super::{Bid, Bgid};

/// A buffer (or portion of a buffer) provided by the kernel via an io_uring buffer ring.
///
/// This type represents data returned from an I/O operation using provided buffers.
/// It may represent the full buffer or a portion of it (with kernel 6.12+ and
/// `IOU_PBUF_RING_INC`, multiple operations can consume different portions of the
/// same buffer).
///
/// The buffer is reference-counted by the ring. When all references are dropped
/// (all clones of this buffer with the same `bid`), the buffer is automatically
/// returned to the ring for reuse.
///
/// # Memory Safety
///
/// - Buffer memory is owned by the buffer ring
/// - The buffer holds a strong reference (`Rc`) to the ring, keeping it alive
/// - Buffer memory is guaranteed valid as long as the `ProvidedBuffer` exists
/// - Users can safely sub-slice the result of `as_slice()` using standard Rust slice operations
pub struct ProvidedBuffer {
    /// Strong reference to the buffer ring.
    /// This keeps the ring (and all buffer memory) alive as long as any buffer exists.
    ring: BufRing,

    /// Pointer to the start of this buffer's data.
    /// Points to the base address for full buffers, or to a specific offset
    /// for incremental consumption (IOU_PBUF_RING_INC).
    ptr: *const u8,

    /// Length of data in this buffer/slice.
    len: u32,

    /// Buffer ID within the ring (0 to N-1 where N is buffer count).
    bid: Bid,
}

impl ProvidedBuffer {
    pub(crate) fn new(ring: BufRing, bid: Bid, ptr: *const u8, len: u32) -> Self {
        Self {
            ring,
            bid,
            ptr,
            len,
        }
    }

    /// Get the buffer group ID of the ring this buffer came from.
    pub fn bgid(&self) -> Bgid {
        self.ring.bgid()
    }

    /// Get the buffer ID within its ring.
    pub fn bid(&self) -> Bid {
        self.bid
    }

    /// Get the length of data in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns true if this buffer contains no data.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a byte slice of the data.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len as usize) }
    }
}

impl Clone for ProvidedBuffer {
    fn clone(&self) -> Self {
        // Increment reference count for this bid
        self.ring.raw.increment_ref(self.bid);

        Self {
            ring: self.ring.clone(),
            bid: self.bid,
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl Drop for ProvidedBuffer {
    fn drop(&mut self) {
        // Decrement reference count
        // When it reaches 0, the ring will return the buffer
        self.ring.raw.decrement_ref(self.bid);
    }
}

impl From<ProvidedBuffer> for Vec<u8> {
    fn from(buf: ProvidedBuffer) -> Self {
        buf.as_slice().to_vec()
    }
}

impl AsRef<[u8]> for ProvidedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}