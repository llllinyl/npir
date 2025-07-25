use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    mem::size_of,
    ops::{Index, IndexMut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

const ALIGN_SIMD: usize = 64; // enough to support AVX-512
pub type AlignedMemory64<T> = AlignedMemory<T, ALIGN_SIMD>;

pub struct AlignedMemory<T, const ALIGN: usize> {
    p: *mut T,
    len: usize,
    layout: Layout,
}

impl<T, const ALIGN: usize> AlignedMemory<T, ALIGN> {
    pub fn new(len: usize) -> Self {
        let sz_bytes = len * size_of::<T>();
        let layout = Layout::from_size_align(sz_bytes, ALIGN).unwrap();

        let ptr;
        unsafe {
            ptr = alloc_zeroed(layout);
        }

        Self {
            p: ptr as *mut T,
            len,
            layout,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { from_raw_parts(self.p, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { from_raw_parts_mut(self.p, self.len) }
    }

    pub unsafe fn as_ptr(&self) -> *const T {
        self.p
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.p
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

unsafe impl<T, const ALIGN: usize> Send for AlignedMemory<T, ALIGN> {}
unsafe impl<T, const ALIGN: usize> Sync for AlignedMemory<T, ALIGN> {}

impl<T, const ALIGN: usize> Drop for AlignedMemory<T, ALIGN> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.p as *mut u8, self.layout);
        }
    }
}

impl<T, const ALIGN: usize> Index<usize> for AlignedMemory<T, ALIGN> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T, const ALIGN: usize> IndexMut<usize> for AlignedMemory<T, ALIGN> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<T: Clone, const ALIGN: usize> Clone for AlignedMemory<T, ALIGN> {
    fn clone(&self) -> Self {
        let mut out = Self::new(self.len);
        out.as_mut_slice().clone_from_slice(self.as_slice());
        out
    }
}
