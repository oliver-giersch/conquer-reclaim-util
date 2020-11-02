//! TODO: mod-lvl docs

use core::{
    iter::FusedIterator,
    mem::MaybeUninit,
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};

use alloc::boxed::Box;

// *************************************************************************************************
// AtomicToken (trait)
// *************************************************************************************************

/// A token that can be stored inside a [`TokenQueue`][TokenQueue].
///
/// # Safety
///
/// Implementing this trait is unsafe, because implementing it incorrectly can
/// lead to undefined behavior in the safe parts of the provided API of, e.g.,
/// [`TokenQueue`][TokenQueue].
///
/// Incorrect implementations could, e.g., entail not using unique values for
/// the required placeholder constants or not forwarding the traits methods to
/// the appropriate atomic synchronization primitives.
///
/// # Example
///
/// An correct and sound implementation for generic atomic pointers:
///
/// ```
/// # use std::sync::atomic::{AtomicPtr, Ordering};
/// unsafe impl<T> AtomicToken for AtomicPtr<T> {
///     type Token = *mut T;
///
///     const NOT_YET_USED: *mut T = 0x0 as *mut T;
///     const FREE: *mut T = 0x1 as *mut T;
///     const RESERVED: *mut T = 0x2 as *mut T;
///
///     fn new(token: *mut T) -> Self {
///         AtomicPtr::new(token)
///     }
///
///     fn load(&self, order: Ordering) -> *mut T {
///         AtomicPtr::load(self, order)
///     }
///
///     fn store(&self, token: *mut T, order: Ordering) {
///         AtomicPtr::store(&self, token, order)
///     }
///
///     fn compare_exchange(
///         &self,
///         current: *mut T,
///         new: *mut T,
///         success: Ordering,
///         failure: Ordering,
///     ) -> Result<*mut T, *mut > {
///         AtomicPtr::compare_exchange(self, current, new, success, failure)
///     }
/// }
/// ```
pub unsafe trait AtomicToken: Send + Sync + Sized {
    /// The associate (non-atomic) token type.
    type Token: Copy + PartialEq + Sized;

    /// A placeholder value indicating a token has never been used before.
    const NOT_YET_USED: Self::Token;
    /// A placeholder value indicating a token is currently free to be acquired.
    const FREE: Self::Token;
    /// A placeholder value indicating a token is currently reserved.
    const RESERVED: Self::Token;

    /// Creates a new [`AtomicToken`].
    fn new(token: Self::Token) -> Self;

    /// Atomically loads the current token value.
    ///
    /// This method should forward to the type's native `load` method (e.g.,
    /// [`AtomicPtr::load`]).
    fn load(&self, order: Ordering) -> Self::Token;

    /// Atomically stores `token` into `self`.
    ///
    /// This method should forward to the type's native `store` method (e.g.,
    /// [`AtomicPtr::store`]).
    fn store(&self, token: Self::Token, order: Ordering);

    /// Performs an atomic *compare-and-swap*.
    ///
    /// This method should forward to the type's native `compare_exchange`
    /// method (e.g., [`AtomicPtr::compare_exchange`]).
    fn compare_exchange(
        &self,
        current: Self::Token,
        new: Self::Token,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::Token, Self::Token>;
}

// *************************************************************************************************
// TokenQueue
// *************************************************************************************************

/// An efficient concurrent lock-free token queue.
///
/// The queue can store (atomic) representations of simple (copyable) token
/// types such as pointers or integers, which fit into a single memory word
/// (i.e., `usize` and smaller).
/// It never de-allocates any memory before it is dropped, meaning it grows
/// monotonically.
/// This has the advantage, that the queue does not require a specialized memory
/// reclamation mechanism without giving up on its overall lock-freedom
/// guarantees (N.B. the queue still does memory allocation, which is usually
/// not guaranteed to be lock-free).
///
/// Any token, that has been inserted into the queue, can be re-used arbitrarily
/// often, as long as it is explicitly set to [`FREE`][AtomicToken::FREE] once
/// it is no longer needed.
///
/// Internally, the queue is constructed as a linked list of nodes, each of
/// which contains an array containing up to `NODE_SIZE` atomic tokens (see also
/// [`AtomicToken`]).
#[derive(Debug, Default)]
pub struct TokenQueue<T, const NODE_SIZE: usize> {
    /// The pointer to the first node in the queue.
    head: AtomicPtr<Node<T, NODE_SIZE>>,
}

/********** impl inherent (const) *****************************************************************/

impl<T, const NODE_SIZE: usize> TokenQueue<T, NODE_SIZE> {
    /// Creates an empty queue.
    #[inline]
    pub const fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }
}

/********** impl inherent *************************************************************************/

impl<T: AtomicToken, const NODE_SIZE: usize> TokenQueue<T, NODE_SIZE> {
    /// Returns an unique token reference set to [`RESERVED`][AtomicToken::RESERVED].
    ///
    /// The queue first attempts to acquire a token that already exists but is
    /// currently not in use.
    /// If no such token is found, a new node with `NODE_SIZE` elements is
    /// appended, from which the first token is returned.
    /// The returned token is unique, in that it can only be reserved by one
    /// thread, but shared access to the token is still possible, e.g., during
    /// iteration of the queue.
    ///
    /// The returned token must be manually set to [`FREE`][AtomicToken::FREE],
    /// once it is no longer needed.
    /// Failing to do so means that the token can never be re-used again,
    /// effectively leaking it.
    #[must_use = "failing to set a token free results in a leak"]
    #[inline]
    pub fn get_reserved(&self) -> &T {
        self.get_token(T::RESERVED, Ordering::Release)
    }

    /// Gets or inserts a unique token from the queue, setting its value to
    /// `token`.
    ///
    /// The `order` argument is applied to the CAS constituting this operation's
    /// *linearization point*.
    ///
    /// The returned token is unique, in that it can only be reserved by one
    /// thread, but shared access to the token is still possible, e.g., during
    /// iteration of the queue.
    ///
    /// The returned token must be manually set either
    /// [`Free`][AtomicToken::FREE], once it is no longer needed.
    /// Otherwise it can never be re-used again, effectively leaking the token.
    ///
    /// # Panics
    ///
    /// Panics if `order` is [`Relaxed`][rlx] or [`Acquire`][acq].
    ///
    /// [rlx]: Ordering::Relaxed
    /// [acq]: Ordering::Acquire
    #[inline]
    pub fn get_token(&self, token: T::Token, order: Ordering) -> &T {
        assert_order(order);
        let mut prev = &self.head;
        // iterate over the nodes in the linked list from the head
        loop {
            // (lst:1) this acq load syncs-with the rel-acq CAS (lst:2)
            let curr = prev.load(Ordering::Acquire);
            if curr.is_null() {
                break;
            }

            // try to acquire a token in the current node
            if let Some(atomic) = unsafe { self.try_insert_in(curr as *const _, token, order) } {
                return atomic;
            }

            // ...if no token was currently available, advance to the next node
            prev = unsafe { &(*curr).next };
        }

        // if no token could be acquired from any already allocated node, a new node must be
        // inserted at the back of the queue.
        unsafe { self.insert_back(prev, token, order) }
    }

    #[inline]
    unsafe fn insert_back(
        &self,
        mut tail: *const AtomicPtr<Node<T, NODE_SIZE>>,
        token: T::Token,
        order: Ordering,
    ) -> &T {
        use Ordering::{AcqRel, Acquire};
        // allocate a new node with `token` stored in the first slot.
        let node = Box::into_raw(Box::new(Node::new(token)));
        // attempt to append the new node after the current tail node
        // (lst:2) this rel/acq CAS syncs with the acq load (lst:1)
        // note: the most precise ordering would be rel-acq, since on success it must only be
        // guaranteed that the previous node allocation happens-before the CAS
        while let Err(read) = (*tail).compare_exchange(ptr::null_mut(), node, AcqRel, Acquire) {
            // the previous CAS failed, so another thread must have succeeded in appending a
            // different node, so this thread attempts to insert its token into that node, first
            if let Some(atomic) = self.try_insert_in(read, token, order) {
                // if the insertion succeeded, the previously allocated node can be dropped again.
                Box::from_raw(node);
                return atomic;
            }

            // ...otherwise, update the tail pointer and re-try
            tail = &(*read).next;
        }

        // on successful CAS, return the pre-reserved slot containing `token`.
        &(*node).tokens[0]
    }

    #[inline]
    unsafe fn try_insert_in(
        &self,
        node: *const Node<T, NODE_SIZE>,
        token: T::Token,
        order: Ordering,
    ) -> Option<&T> {
        // this loop attempts to acquire one token from the given node's array, trying every token
        // exactly once (in the worst case)
        // although the first token is pre-reserved on allocation, it may already be free again and
        // must hence also be checked
        for atomic in &(*node).tokens[..] {
            // read the token's current value
            let curr = atomic.load(Ordering::Relaxed);

            // if the token is either FREE or NOT_YET_USED, attempt to acquire it using a CAS by
            // setting its value to `token`.
            let success = (curr == T::FREE || curr == T::NOT_YET_USED)
                && atomic
                    .compare_exchange(curr, token, order, Ordering::Relaxed)
                    .is_ok();

            // if the CAS succeeded, return the acquired token reference
            if success {
                return Some(atomic);
            }
        }

        None
    }
}

/********** impl Drop *****************************************************************************/

impl<T, const NODE_SIZE: usize> Drop for TokenQueue<T, NODE_SIZE> {
    #[inline(never)]
    fn drop(&mut self) {
        // drop all nodes
        let mut curr = self.head.load(Ordering::Relaxed);
        while !curr.is_null() {
            let node = unsafe { Box::from_raw(curr) };
            curr = node.next.load(Ordering::Relaxed);
        }
    }
}

// *************************************************************************************************
// Iter
// *************************************************************************************************

/// An iterator over all tokens in a [`TokenQueue`].
#[derive(Debug)]
pub struct Iter<'a, T, const NODE_SIZE: usize> {
    idx: usize,
    curr: Option<&'a Node<T, NODE_SIZE>>,
}

/********** impl Iterator *************************************************************************/

impl<'a, T: AtomicToken, const NODE_SIZE: usize> Iterator for Iter<'a, T, NODE_SIZE> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // this loop is executed at most twice
        while let Some(node) = self.curr {
            if self.idx < NODE_SIZE {
                // current iteration is at some token in the current node
                let idx = self.idx;
                self.idx += 1;
                return Some(&node.tokens[idx]);
            } else {
                // try to advance to next node
                // (lst:4) this acq load syncs-with the rel-acq CAS (lst:3)
                self.curr = unsafe { node.next.load(Ordering::Acquire).as_ref() };
                self.idx = 0;
            }
        }

        None
    }
}

/********** impl FusedIterator ********************************************************************/

impl<'a, T: AtomicToken, const NODE_SIZE: usize> FusedIterator for Iter<'a, T, NODE_SIZE> {}

// *************************************************************************************************
// Node
// *************************************************************************************************

#[repr(C)]
#[derive(Debug)]
struct Node<T, const NODE_SIZE: usize> {
    /// The array of atomic tokens.
    tokens: [T; NODE_SIZE],
    /// The pointer to the next node.
    next: AtomicPtr<Self>,
}

/********** impl inherent *************************************************************************/

impl<T: AtomicToken, const NODE_SIZE: usize> Node<T, NODE_SIZE> {
    #[inline]
    fn new(first: T::Token) -> Self {
        // initialize array of atomic tokens
        let tokens = unsafe {
            // create uninitialized array
            let mut tokens: MaybeUninit<[T; NODE_SIZE]> = MaybeUninit::uninit();
            // the pointer to first element
            let ptr = tokens.as_mut_ptr() as *mut T;

            // initialize first token
            ptr.write(T::new(first));
            // initialize remaining tokens
            for offset in 1..NODE_SIZE {
                ptr.add(offset).write(T::new(T::NOT_YET_USED));
            }

            tokens.assume_init()
        };

        Self {
            tokens,
            next: AtomicPtr::default(),
        }
    }
}

#[inline(always)]
fn assert_order(order: Ordering) {
    assert!(
        order == Ordering::Release || order == Ordering::AcqRel || order == Ordering::SeqCst,
        "the `order` must be `Release` or stronger"
    )
}
