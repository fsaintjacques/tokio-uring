/// The Buffer Group ID.
///
/// A buffer group ID (bgid) is a 16-bit identifier used by io_uring to distinguish between
/// different pools of provided buffers. When submitting an operation that uses provided buffers
/// (by setting the `IOSQE_BUFFER_SELECT` flag), the bgid in the submission queue entry (SQE)
/// tells the kernel which buffer pool to select from.
///
/// Multiple buffer groups can be registered simultaneously, each with a unique bgid. This allows
/// different types of operations or different sockets to use separate buffer pools with different
/// characteristics (e.g., different buffer sizes, different pool capacities).
///
/// When the kernel completes an operation using a provided buffer, it returns the bgid along with
/// the specific buffer ID (bid) in the completion queue entry (CQE), allowing userspace to identify
/// which buffer was used and return it to the appropriate pool.
///
/// The creator of a buffer group is responsible for choosing a bgid that does not conflict with
/// other buffer groups registered with the same io_uring instance.
pub(crate) type Bgid = u16;

/// The Buffer ID.
///
/// A buffer ID (bid) is a 16-bit identifier used within a specific buffer group to identify
/// individual buffers. Unlike the buffer group ID (bgid) which identifies the pool, the bid
/// identifies a specific buffer within that pool.
///
/// When the kernel selects a buffer from a buffer ring to fulfill an I/O operation, it returns
/// the bid in the upper 16 bits of the completion queue entry's (CQE) flags field. Userspace
/// extracts this bid to determine which specific buffer was used, allowing it to:
///
/// - Access the buffer's data at the correct memory location
/// - Track which buffers are currently in use vs. available in the pool
/// - Return the buffer to the pool when the operation completes
///
/// Buffer IDs within a buffer group typically range from 0 to N-1, where N is the number of
/// buffers in the pool. The buffer ring implementation is responsible for managing the mapping
/// between bids and their corresponding memory locations.
///
/// The bid is returned to userspace via the `IORING_CQE_F_BUFFER` flag bit being set in the CQE,
/// with the actual bid value encoded in bits 16-31 of the flags field.
pub(crate) type Bid = u16;

mod ring;
pub use ring::BufRing;

mod provided;
pub use provided::ProvidedBuffer;

mod tail;