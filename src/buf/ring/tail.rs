//! Safe wrapper around the io_uring buffer ring's tail mechanism.
//!
//! In io_uring, the buffer ring doesn't have a separate header. Instead, the tail counter
//! is embedded in the `resv` field of the first ring entry. This module provides a safe,
//! ergonomic Rust interface that hides this C union complexity.

use std::sync::atomic::{AtomicU16, Ordering};

/// A safe wrapper around the buffer ring's tail counter.
///
/// This type provides atomic access to the tail index, which tracks where userspace
/// should add the next buffer. The kernel reads this atomically to know how many
/// buffers are available.
///
/// # Implementation Note
///
/// The tail is physically stored in the `resv` field of the first ring entry, but
/// this detail is completely hidden from users of this type.
pub(super) struct RingTail {
    /// Pointer to the tail field within the ring structure.
    ///
    /// This points to the resv field of the first BufRingEntry, but callers
    /// don't need to know that detail.
    tail: *const AtomicU16,

    /// Local copy of the tail for batched updates.
    ///
    /// We maintain a local counter and only sync to shared memory when needed,
    /// reducing atomic operations.
    local_tail: u16,
}

impl RingTail {
    /// Create a new RingTail from the ring base address.
    ///
    /// # Safety
    ///
    /// - `ring_base` must point to a valid, page-aligned io_uring buffer ring
    /// - The ring must remain valid for the lifetime of this RingTail
    /// - Only one RingTail should exist per ring (exclusive access)
    pub(super) unsafe fn new(ring_base: *const io_uring::types::BufRingEntry) -> Self {
        let tail = io_uring::types::BufRingEntry::tail(ring_base) as *const AtomicU16;

        Self {
            tail,
            local_tail: 0,
        }
    }

    /// Get the current local tail value.
    ///
    /// This returns the local counter, which may be ahead of what the kernel sees
    /// until `sync()` is called.
    #[inline]
    pub(super) fn local(&self) -> u16 {
        self.local_tail
    }

    /// Increment the local tail counter.
    ///
    /// This is a cheap operation that doesn't touch shared memory.
    /// Call `sync()` afterward to make the update visible to the kernel.
    #[inline]
    pub(super) fn increment(&mut self) {
        self.local_tail = self.local_tail.wrapping_add(1);
    }

    /// Synchronize the local tail to the kernel-visible tail.
    ///
    /// This performs an atomic store with Release ordering, making all previous
    /// buffer additions visible to the kernel.
    #[inline]
    pub(super) fn sync(&self) {
        // SAFETY: tail pointer is valid for the lifetime of this RingTail,
        // and we have exclusive access via &self (enforced by RawBufRing)
        unsafe {
            (*self.tail).store(self.local_tail, Ordering::Release);
        }
    }

    /// Read the current kernel-visible tail value.
    ///
    /// This is rarely needed, but can be useful for debugging or verification.
    #[inline]
    pub(super) fn read_shared(&self) -> u16 {
        // SAFETY: Same as sync()
        unsafe { (*self.tail).load(Ordering::Acquire) }
    }
}

// Not Send/Sync because:
// 1. Contains a raw pointer
// 2. Part of single-threaded io_uring design
// 3. Exclusive access enforced by RawBufRing ownership

impl std::fmt::Debug for RingTail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingTail")
            .field("local_tail", &self.local_tail)
            .field("shared_tail", &self.read_shared())
            .finish()
    }
}