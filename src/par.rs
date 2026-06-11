// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Parallelism abstraction shared by the preprocessing and postprocessing code.
//!
//! On native targets this module simply re-exports the [`rayon`] prelude, so the
//! hot loops run across all available cores exactly as before. On
//! `wasm32-unknown-unknown` there are no OS threads and `rayon` cannot be built,
//! so the module instead provides drop-in sequential shims with the same method
//! names (`into_par_iter`, `par_chunks_mut`, `for_each_with`). This lets the
//! shared image pipeline compile for the browser without sprinkling `cfg`
//! attributes across every call site.

#[cfg(not(target_arch = "wasm32"))]
pub use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator, ParallelSliceMut,
};

#[cfg(target_arch = "wasm32")]
pub use shim::{IntoParallelIterator, ParallelIterator, ParallelSliceMut};

/// Sequential stand-ins for the small slice of the rayon prelude used by the
/// shared pipeline. Each method delegates to the equivalent `std` iterator, so
/// behaviour is identical (just single-threaded).
#[cfg(target_arch = "wasm32")]
mod shim {
    /// Sequential analogue of `rayon::iter::IntoParallelIterator`.
    pub trait IntoParallelIterator {
        /// Element type yielded by the iterator.
        type Item;
        /// Concrete sequential iterator returned by [`into_par_iter`].
        type Iter: Iterator<Item = Self::Item>;
        /// Convert into a sequential iterator (mirrors rayon's `into_par_iter`).
        fn into_par_iter(self) -> Self::Iter;
    }

    impl<T: IntoIterator> IntoParallelIterator for T {
        type Item = T::Item;
        type Iter = T::IntoIter;
        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    /// Sequential analogue of `rayon::slice::ParallelSliceMut` (the subset used
    /// here: chunked mutable iteration).
    pub trait ParallelSliceMut<T> {
        /// Mutable, non-overlapping chunks of length `chunk_size` (mirrors
        /// rayon's `par_chunks_mut`).
        fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T>;
    }

    impl<T> ParallelSliceMut<T> for [T] {
        fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T> {
            self.chunks_mut(chunk_size)
        }
    }

    /// Adds the rayon `for_each_with` adaptor to ordinary iterators. The seed is
    /// threaded through every call rather than cloned per worker, which is the
    /// correct single-threaded equivalent.
    pub trait ParallelIterator: Iterator + Sized {
        /// Run `op` for each item, passing a shared mutable seed value.
        fn for_each_with<S, F>(self, init: S, mut op: F)
        where
            F: FnMut(&mut S, Self::Item),
        {
            let mut init = init;
            for item in self {
                op(&mut init, item);
            }
        }
    }

    impl<I: Iterator> ParallelIterator for I {}
}
