#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;

use rand::distributions::Standard;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use std::cell::RefCell;
use std::ops::{Add, Mul, Neg};
use crate::{aligned_memory::*, arith::*, discrete_gaussian::*, ntt::*, params::*, util::*};

// thread_local!(static SCRATCH: RefCell<AlignedMemory64<u16>> = RefCell::new(AlignedMemory64::<u16>::new(4096 as usize)));
thread_local!(static SCRATCH: RefCell<AlignedMemory64<u64>> = RefCell::new(AlignedMemory64::<u64>::new(16384 as usize)));

pub trait PolyMatrix<'a> {
    fn is_ntt(&self) -> bool;
    fn get_rows(&self) -> usize;
    fn get_cols(&self) -> usize;
    fn get_params(&self) -> &Params;
    fn num_words(&self) -> usize;
    fn zero(params: &'a Params, rows: usize, cols: usize) -> Self;
    fn random(params: &'a Params, rows: usize, cols: usize) -> Self;
    fn random_ternary_rng<T: Rng>(params: &'a Params, rows: usize, cols: usize, rng: &mut T) -> Self;
    fn random_rng<T: Rng>(params: &'a Params, rows: usize, cols: usize, rng: &mut T) -> Self;
    fn as_slice(&self) -> &[u64];
    fn as_mut_slice(&mut self) -> &mut [u64];
    fn zero_out(&mut self) {
        for item in self.as_mut_slice() {
            *item = 0;
        }
    }
    fn get_poly(&self, row: usize, col: usize) -> &[u64] {
        let num_words = self.num_words();
        let start = (row * self.get_cols() + col) * num_words;
        // &self.as_slice()[start..start + num_words]
        unsafe { self.as_slice().get_unchecked(start..start + num_words) }
    }
    fn get_poly_mut(&mut self, row: usize, col: usize) -> &mut [u64] {
        let num_words = self.num_words();
        let start = (row * self.get_cols() + col) * num_words;
        // &mut self.as_mut_slice()[start..start + num_words]
        unsafe {
            self.as_mut_slice()
                .get_unchecked_mut(start..start + num_words)
        }
    }
    fn copy_into(&mut self, p: &Self, target_row: usize, target_col: usize) {
        assert!(target_row < self.get_rows());
        assert!(target_col < self.get_cols());
        assert!(target_row + p.get_rows() <= self.get_rows());
        assert!(target_col + p.get_cols() <= self.get_cols());
        for r in 0..p.get_rows() {
            for c in 0..p.get_cols() {
                let pol_src = p.get_poly(r, c);
                let pol_dst = self.get_poly_mut(target_row + r, target_col + c);
                pol_dst.copy_from_slice(pol_src);
            }
        }
    }

    fn submatrix(&self, target_row: usize, target_col: usize, rows: usize, cols: usize) -> Self;
    fn pad_top(&self, pad_rows: usize) -> Self;
}

pub struct PolyMatrixSmall<'a> {
    pub params: &'a Params,
    pub rows: usize,
    pub cols: usize,
    pub data: AlignedMemory64<u16>,
}

pub struct PolyMatrixRaw<'a> {
    pub params: &'a Params,
    pub rows: usize,
    pub cols: usize,
    pub data: AlignedMemory64<u64>,
}

pub struct PolyMatrixNTT<'a> {
    pub params: &'a Params,
    pub rows: usize,
    pub cols: usize,
    pub data: AlignedMemory64<u64>,
}


impl<'a> PolyMatrixSmall<'a> {
    fn get_cols(&self) -> usize {
        self.cols
    }
    fn as_slice(&self) -> &[u16] {
        self.data.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [u16] {
        self.data.as_mut_slice()
    }
    pub fn get_poly(&self, row: usize, col: usize) -> &[u16] {
        let num_words = self.num_words();
        let start = (row * self.get_cols() + col) * num_words;
        unsafe { self.as_slice().get_unchecked(start..start + num_words) }
    }
    pub fn get_poly_mut(&mut self, row: usize, col: usize) -> &mut [u16] {
        let num_words = self.num_words();
        let start = (row * self.get_cols() + col) * num_words;
        unsafe {
            self.as_mut_slice()
                .get_unchecked_mut(start..start + num_words)
        }
    }
    fn num_words(&self) -> usize {
        self.params.poly_len
    }
    pub fn zero(params: &'a Params, rows: usize, cols: usize) -> PolyMatrixSmall<'a> {
        let num_coeffs = rows * cols * params.poly_len;
        let data = AlignedMemory64::<u16>::new(num_coeffs);
        PolyMatrixSmall {
            params,
            rows,
            cols,
            data,
        }
    }
}

impl<'a> Clone for PolyMatrixSmall<'a> {
    fn clone(&self) -> Self {
        let mut data_clone = AlignedMemory64::<u16>::new(self.data.len());
        data_clone
            .as_mut_slice()
            .copy_from_slice(self.data.as_slice());
        PolyMatrixSmall {
            params: self.params,
            rows: self.rows,
            cols: self.cols,
            data: data_clone,
        }
    }
}

impl<'a> PolyMatrix<'a> for PolyMatrixRaw<'a> {
    fn is_ntt(&self) -> bool {
        false
    }
    fn get_rows(&self) -> usize {
        self.rows
    }
    fn get_cols(&self) -> usize {
        self.cols
    }
    fn get_params(&self) -> &Params {
        &self.params
    }
    fn as_slice(&self) -> &[u64] {
        self.data.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [u64] {
        self.data.as_mut_slice()
    }
    fn num_words(&self) -> usize {
        self.params.poly_len
    }
    fn zero(params: &'a Params, rows: usize, cols: usize) -> PolyMatrixRaw<'a> {
        let num_coeffs = rows * cols * params.poly_len;
        let data = AlignedMemory64::<u64>::new(num_coeffs);
        PolyMatrixRaw {
            params,
            rows,
            cols,
            data,
        }
    }
    fn random_ternary_rng<T: Rng>(params: &'a Params, rows: usize, cols: usize, rng: &mut T) -> Self {
        let mut iter = rng.sample_iter(&Standard);
        let mut out = PolyMatrixRaw::zero(params, rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                for i in 0..params.poly_len {
                    let val: u64 = iter.next().unwrap();
                    let result = match val % 3 {
                        0 => 0,
                        1 => 1,
                        2 => params.modulus - 1,
                        _ => unreachable!(),
                    };
                    out.get_poly_mut(r, c)[i] = result;
                }
            }
        }
        out
    }
    fn random_rng<T: Rng>(params: &'a Params, rows: usize, cols: usize, rng: &mut T) -> Self {
        let mut iter = rng.sample_iter(&Standard);
        let mut out = PolyMatrixRaw::zero(params, rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                for i in 0..params.poly_len {
                    let val: u64 = iter.next().unwrap();
                    out.get_poly_mut(r, c)[i] = val % params.modulus;
                }
            }
        }
        out
    }
    fn random(params: &'a Params, rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::random_rng(params, rows, cols, &mut rng)
    }
    fn pad_top(&self, pad_rows: usize) -> Self {
        let mut padded = Self::zero(self.params, self.rows + pad_rows, self.cols);
        padded.copy_into(&self, pad_rows, 0);
        padded
    }
    fn submatrix(&self, target_row: usize, target_col: usize, rows: usize, cols: usize) -> Self {
        let mut m = Self::zero(self.params, rows, cols);
        assert!(target_row < self.rows);
        assert!(target_col < self.cols);
        assert!(target_row + rows <= self.rows);
        assert!(target_col + cols <= self.cols);
        for r in 0..rows {
            for c in 0..cols {
                let pol_src = self.get_poly(target_row + r, target_col + c);
                let pol_dst = m.get_poly_mut(r, c);
                pol_dst.copy_from_slice(pol_src);
            }
        }
        m
    }
}

impl<'a> Clone for PolyMatrixRaw<'a> {
    fn clone(&self) -> Self {
        let mut data_clone = AlignedMemory64::<u64>::new(self.data.len());
        data_clone
            .as_mut_slice()
            .copy_from_slice(self.data.as_slice());
        PolyMatrixRaw {
            params: self.params,
            rows: self.rows,
            cols: self.cols,
            data: data_clone,
        }
    }
}

impl<'a> PolyMatrixRaw<'a> {
    pub fn identity(params: &'a Params, rows: usize, cols: usize) -> PolyMatrixRaw<'a> {
        let num_coeffs = rows * cols * params.poly_len;
        let mut data = AlignedMemory::new(num_coeffs);
        for r in 0..rows {
            let c = r;
            let idx = r * cols * params.poly_len + c * params.poly_len;
            data[idx] = 1;
        }
        PolyMatrixRaw {
            params,
            rows,
            cols,
            data,
        }
    }

    pub fn noise(
        params: &'a Params,
        rows: usize,
        cols: usize,
        dg: &DiscreteGaussian,
        rng: &mut ChaCha20Rng,
    ) -> Self {
        let mut out = PolyMatrixRaw::zero(params, rows, cols);
        dg.sample_matrix(&mut out, rng);
        out
    }

    pub fn fast_noise(
        params: &'a Params,
        rows: usize,
        cols: usize,
        dg: &DiscreteGaussian,
        rng: &mut ChaCha20Rng,
    ) -> Self {
        let mut out = PolyMatrixRaw::zero(params, rows, cols);
        let modulus = params.modulus;
        for r in 0..out.rows {
            for c in 0..out.cols {
                let poly = out.get_poly_mut(r, c);
                for z in 0..poly.len() {
                    let s = dg.fast_sample(modulus, rng);
                    poly[z] = s;
                }
            }
        }
        out
    }

    pub fn ntt(&self) -> PolyMatrixNTT<'a> {
        to_ntt_alloc(&self)
    }

    pub fn reduce_mod(&mut self, modulus: u64) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                for z in 0..self.params.poly_len {
                    self.get_poly_mut(r, c)[z] %= modulus;
                }
            }
        }
    }

    pub fn apply_func<F: Fn(u64) -> u64>(&mut self, func: F) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                let pol_mut = self.get_poly_mut(r, c);
                for el in pol_mut {
                    *el = func(*el);
                }
            }
        }
    }

    pub fn to_vec(&self, modulus_bits: usize, num_coeffs: usize) -> Vec<u8> {
        let sz_bits = self.rows * self.cols * num_coeffs * modulus_bits;
        let sz_bytes = f64::ceil((sz_bits as f64) / 8f64) as usize + 32;
        let sz_bytes_roundup_16 = ((sz_bytes + 15) / 16) * 16;
        let mut data = vec![0u8; sz_bytes_roundup_16];
        let mut bit_offs = 0;
        for r in 0..self.rows {
            for c in 0..self.cols {
                for z in 0..num_coeffs {
                    let val = self.get_poly(r, c)[z];
                    write_arbitrary_bits(data.as_mut_slice(), val, bit_offs, modulus_bits);
                    bit_offs += modulus_bits;
                }
                bit_offs = (bit_offs / 8) * 8
            }
        }
        data
    }

    pub fn single_value(params: &'a Params, value: u64) -> PolyMatrixRaw<'a> {
        let mut out = Self::zero(params, 1, 1);
        out.data[0] = value;
        out
    }
}

impl<'a> PolyMatrix<'a> for PolyMatrixNTT<'a> {
    fn is_ntt(&self) -> bool {
        true
    }
    fn get_rows(&self) -> usize {
        self.rows
    }
    fn get_cols(&self) -> usize {
        self.cols
    }
    fn get_params(&self) -> &Params {
        &self.params
    }
    fn as_slice(&self) -> &[u64] {
        self.data.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [u64] {
        self.data.as_mut_slice()
    }
    fn num_words(&self) -> usize {
        self.params.poly_len * self.params.crt_count
    }
    fn zero(params: &'a Params, rows: usize, cols: usize) -> PolyMatrixNTT<'a> {
        let num_coeffs = rows * cols * params.poly_len * params.crt_count;
        let data = AlignedMemory::new(num_coeffs);
        PolyMatrixNTT {
            params,
            rows,
            cols,
            data,
        }
    }
    fn random_ternary_rng<T: Rng>(params: &'a Params, rows: usize, cols: usize, rng: &mut T) -> Self {
        let mut iter = rng.sample_iter(&Standard);
        let mut out = PolyMatrixNTT::zero(params, rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                for i in 0..params.crt_count {
                    for j in 0..params.poly_len {
                        let idx = calc_index(&[i, j], &[params.crt_count, params.poly_len]);
                        let val: u64 = iter.next().unwrap();
                        let result = match val % 3 {
                            0 => 0,
                            1 => 1,
                            2 => params.modulus - 1,
                            _ => unreachable!(),
                        };
                        out.get_poly_mut(r, c)[idx] = result;
                    }
                }
            }
        }
        out
    }
    fn random_rng<T: Rng>(params: &'a Params, rows: usize, cols: usize, rng: &mut T) -> Self {
        let mut iter = rng.sample_iter(&Standard);
        let mut out = PolyMatrixNTT::zero(params, rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                for i in 0..params.crt_count {
                    for j in 0..params.poly_len {
                        let idx = calc_index(&[i, j], &[params.crt_count, params.poly_len]);
                        let val: u64 = iter.next().unwrap();
                        out.get_poly_mut(r, c)[idx] = val % params.moduli[i];
                    }
                }
            }
        }
        out
    }
    fn random(params: &'a Params, rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::random_rng(params, rows, cols, &mut rng)
    }
    fn pad_top(&self, pad_rows: usize) -> Self {
        let mut padded = Self::zero(self.params, self.rows + pad_rows, self.cols);
        padded.copy_into(&self, pad_rows, 0);
        padded
    }

    fn submatrix(&self, target_row: usize, target_col: usize, rows: usize, cols: usize) -> Self {
        let mut m = Self::zero(self.params, rows, cols);
        assert!(target_row < self.rows);
        assert!(target_col < self.cols);
        assert!(target_row + rows <= self.rows);
        assert!(target_col + cols <= self.cols);
        for r in 0..rows {
            for c in 0..cols {
                let pol_src = self.get_poly(target_row + r, target_col + c);
                let pol_dst = m.get_poly_mut(r, c);
                pol_dst.copy_from_slice(pol_src);
            }
        }
        m
    }
}

impl<'a> Clone for PolyMatrixNTT<'a> {
    fn clone(&self) -> Self {
        let mut data_clone = AlignedMemory64::<u64>::new(self.data.len());
        data_clone
            .as_mut_slice()
            .copy_from_slice(self.data.as_slice());
        PolyMatrixNTT {
            params: self.params,
            rows: self.rows,
            cols: self.cols,
            data: data_clone,
        }
    }
}

impl<'a> PolyMatrixNTT<'a> {
    pub fn raw(&self) -> PolyMatrixRaw<'a> {
        from_ntt_alloc(&self)
    }
}

pub fn shift_rows_by_one<'a>(inp: &PolyMatrixNTT<'a>) -> PolyMatrixNTT<'a> {
    if inp.rows == 1 {
        return inp.clone();
    }

    let all_but_last_row = inp.submatrix(0, 0, inp.rows - 1, inp.cols);
    let last_row = inp.submatrix(inp.rows - 1, 0, 1, inp.cols);
    let out = stack_ntt(&last_row, &all_but_last_row);
    out
}

pub fn multiply_poly(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = multiply_modular(params, a[idx], b[idx], c);
        }
    }
}

pub fn multiply_add_poly(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = multiply_add_modular(params, a[idx], b[idx], res[idx], c);
        }
    }
}

pub fn add_poly(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = add_modular(params, a[idx], b[idx], c);
        }
    }
}

pub fn add_poly_raw(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for i in 0..params.poly_len {
        res[i] = a[i] + b[i]  % params.modulus;
    }
}

pub fn add_poly_into(params: &Params, res: &mut [u64], a: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = add_modular(params, res[idx], a[idx], c);
        }
    }
}

pub fn sub_poly_into(params: &Params, res: &mut [u64], a: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = sub_modular(params, res[idx], a[idx], c);
        }
    }
}

pub fn invert_poly(params: &Params, res: &mut [u64], a: &[u64]) {
    for i in 0..params.poly_len {
        res[i] = params.modulus - a[i];
    }
}

pub fn invert_poly_ntt(params: &Params, res: &mut [u64], a: &[u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = invert_modular(params, a[idx], c);
        }
    }
}

pub fn modular_reduce(params: &Params, res: &mut [u64]) {
    for c in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = c * params.poly_len + i;
            res[idx] = barrett_coeff_u64(params, res[idx], c);
        }
    }
}

#[cfg(target_feature = "avx2")]
pub fn multiply(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT, b: &PolyMatrixNTT) {
    assert_eq!(res.rows, a.rows);
    assert_eq!(res.cols, b.cols);
    assert_eq!(a.cols, b.rows);

    let params = res.params;
    let poly_len = params.poly_len;
    let crt_count = params.crt_count;

    let barrett_consts = unsafe {
        [
            _mm256_set1_epi64x(params.barrett_cr_1[0] as i64),
            _mm256_set1_epi64x(params.barrett_cr_1[1] as i64)
        ]
    };
    let moduli = unsafe {
        [
            _mm256_set1_epi64x(params.moduli[0] as i64),
            _mm256_set1_epi64x(params.moduli[1] as i64)
        ]
    };

    for i in 0..a.rows {
        for j in 0..b.cols {
            let res_poly = res.get_poly_mut(i, j);
            unsafe {
                for z in (0..poly_len * crt_count).step_by(4) {
                    _mm256_store_si256(
                        res_poly.as_mut_ptr().add(z) as *mut __m256i,
                        _mm256_setzero_si256()
                    );
                }
            }

            for k in 0..a.cols {
                let pol1 = a.get_poly(i, k);
                let pol2 = b.get_poly(k, j);
                unsafe {
                        for c in 0..crt_count {
                        let c_offset = c * poly_len;
                        let reduce = 1 << (64 - 2 * params.moduli[c].ilog2() as usize - 3);
                        for i in (0..poly_len).step_by(4) {
                            let idx = c_offset + i;
                            let p_x = pol1.as_ptr().add(idx);
                            let p_y = pol2.as_ptr().add(idx);
                            let p_z = res_poly.as_mut_ptr().add(idx);

                            let x = _mm256_loadu_si256(p_x as *const __m256i);
                            let y = _mm256_loadu_si256(p_y as *const __m256i);
                            let z = _mm256_loadu_si256(p_z as *const __m256i);

                            let product = _mm256_mul_epu32(x, y);
                            let mut sum = _mm256_add_epi64(z, product);
                            
                            if k % reduce == 0 || k == a.cols - 1 {
                                sum = avx2_barrett_reduction(sum, barrett_consts[c], moduli[c]);
                            }
                            _mm256_storeu_si256(p_z as *mut __m256i, sum);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_feature = "avx2")]
pub unsafe fn avx2_barrett_reduction(
    input: __m256i,
    barrett_const: __m256i,
    modulus: __m256i,
) -> __m256i {
    let input_hi = _mm256_srli_epi64(input, 32);
    let const_hi = _mm256_srli_epi64(barrett_const, 32);
    
    let a = _mm256_mul_epu32(input_hi, barrett_const);
    let b = _mm256_mul_epu32(input, const_hi);
    let hi = _mm256_mul_epu32(input_hi, const_hi);
    
    let cross_sum = _mm256_add_epi64(a, b);
    let tmp = _mm256_add_epi64(hi, _mm256_srli_epi64(cross_sum, 32));
    
    let mod_hi = _mm256_srli_epi64(modulus, 32);
    
    let mod_lo = _mm256_mul_epu32(tmp, modulus);
    let mod_a = _mm256_mul_epu32(_mm256_srli_epi64(tmp, 32), modulus);
    let mod_b = _mm256_mul_epu32(tmp, mod_hi);
    let mod_hi = _mm256_mul_epu32(_mm256_srli_epi64(tmp, 32), mod_hi);
    
    let mod_cross = _mm256_add_epi64(mod_a, mod_b);
    let mod_product = _mm256_add_epi64(
        _mm256_slli_epi64(mod_hi, 32),
        _mm256_add_epi64(
            _mm256_slli_epi64(mod_cross, 32),
            mod_lo
        )
    );
    
    let res = _mm256_sub_epi64(input, mod_product);
    
    let ge_mask = _mm256_or_si256(
        _mm256_cmpgt_epi64(res, modulus),
        _mm256_cmpeq_epi64(res, modulus)
    );
    let correction = _mm256_and_si256(modulus, ge_mask);
    _mm256_sub_epi64(res, correction)
}

#[cfg(target_feature = "avx2")]
pub fn multiply_add_poly_avx(params: &Params, res: &mut [u64], a: &[u64], b: &[u64]) {
    for c in 0..params.crt_count {
        for i in (0..params.poly_len).step_by(4) {
            unsafe {
                let p_x = &a[c * params.poly_len + i] as *const u64;
                let p_y = &b[c * params.poly_len + i] as *const u64;
                let p_z = &mut res[c * params.poly_len + i] as *mut u64;
                let x = _mm256_load_si256(p_x as *const __m256i);
                let y = _mm256_load_si256(p_y as *const __m256i);
                let z = _mm256_load_si256(p_z as *const __m256i);

                let product = _mm256_mul_epu32(x, y);
                let out = _mm256_add_epi64(z, product);

                _mm256_store_si256(p_z as *mut __m256i, out);
            }
        }
    }
}

#[cfg(target_feature = "avx2")]
pub fn multiply_no_reduce(
    res: &mut PolyMatrixNTT,
    a: &PolyMatrixNTT,
    b: &PolyMatrixNTT,
    start_inner_dim: usize,
) {
    assert_eq!(res.rows, a.rows);
    assert_eq!(res.cols, b.cols);
    assert_eq!(a.cols, b.rows);

    let params = res.params;
    for i in 0..a.rows {
        for j in 0..b.cols {
            let res_poly = res.get_poly_mut(i, j);
            for k in start_inner_dim..a.cols {
                let pol1 = a.get_poly(i, k);
                let pol2 = b.get_poly(k, j);
                multiply_add_poly_avx(params, res_poly, pol1, pol2);
            }
        }
    }
}

#[cfg(not(target_feature = "avx2"))]
pub fn multiply(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT, b: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == b.cols);
    assert!(a.cols == b.rows);

    let params = res.params;
    for i in 0..a.rows {
        for j in 0..b.cols {
            for z in 0..params.poly_len * params.crt_count {
                res.get_poly_mut(i, j)[z] = 0;
            }
            for k in 0..a.cols {
                let res_poly = res.get_poly_mut(i, j);
                let pol1 = a.get_poly(i, k);
                let pol2 = b.get_poly(k, j);
                multiply_add_poly(params, res_poly, pol1, pol2);
            }
        }
    }
}

pub fn add(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT, b: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);
    assert!(a.rows == b.rows);
    assert!(a.cols == b.cols);

    let params = res.params;
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = a.get_poly(i, j);
            let pol2 = b.get_poly(i, j);
            add_poly(params, res_poly, pol1, pol2);
        }
    }
}

pub fn add_raw(res: &mut PolyMatrixRaw, a: &PolyMatrixRaw, b: &PolyMatrixRaw) {
    assert_eq!(res.rows, a.rows);
    assert_eq!(res.cols, a.cols);
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);

    let params = res.params;
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = a.get_poly(i, j);
            let pol2 = b.get_poly(i, j);
            add_poly_raw(params, res_poly, pol1, pol2);
        }
    }
}

pub fn add_into(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let params = res.params;
    for i in 0..res.rows {
        for j in 0..res.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol2 = a.get_poly(i, j);
            add_poly_into(params, res_poly, pol2);
        }
    }
}

pub fn sub_into(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let params = res.params;
    for i in 0..res.rows {
        for j in 0..res.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol2 = a.get_poly(i, j);
            sub_poly_into(params, res_poly, pol2);
        }
    }
}

pub fn add_into_at(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT, t_row: usize, t_col: usize) {
    let params = res.params;
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(t_row + i, t_col + j);
            let pol2 = a.get_poly(i, j);
            add_poly_into(params, res_poly, pol2);
        }
    }
}

pub fn invert(res: &mut PolyMatrixRaw, a: &PolyMatrixRaw) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let params = res.params;
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = a.get_poly(i, j);
            invert_poly(params, res_poly, pol1);
        }
    }
}

pub fn invert_ntt(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let params = res.params;
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = a.get_poly(i, j);
            invert_poly_ntt(params, res_poly, pol1);
        }
    }
}

pub fn stack<'a>(a: &PolyMatrixRaw<'a>, b: &PolyMatrixRaw<'a>) -> PolyMatrixRaw<'a> {
    assert_eq!(a.cols, b.cols);
    let mut c = PolyMatrixRaw::zero(a.params, a.rows + b.rows, a.cols);
    c.copy_into(a, 0, 0);
    c.copy_into(b, a.rows, 0);
    c
}

pub fn stack_ntt<'a>(a: &PolyMatrixNTT<'a>, b: &PolyMatrixNTT<'a>) -> PolyMatrixNTT<'a> {
    assert_eq!(a.cols, b.cols);
    let mut c = PolyMatrixNTT::zero(a.params, a.rows + b.rows, a.cols);
    c.copy_into(a, 0, 0);
    c.copy_into(b, a.rows, 0);
    c
}

pub fn scalar_multiply(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT, b: &PolyMatrixNTT) {
    assert_eq!(a.rows, 1);
    assert_eq!(a.cols, 1);

    let params = res.params;
    let pol2 = a.get_poly(0, 0);
    for i in 0..b.rows {
        for j in 0..b.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = b.get_poly(i, j);
            multiply_poly(params, res_poly, pol1, pol2);
        }
    }
}

pub fn scalar_multiply_alloc<'a>(
    a: &PolyMatrixNTT<'a>,
    b: &PolyMatrixNTT<'a>,
) -> PolyMatrixNTT<'a> {
    let mut res = PolyMatrixNTT::zero(b.params, b.rows, b.cols);
    scalar_multiply(&mut res, a, b);
    res
}

pub fn single_poly<'a>(params: &'a Params, val: u64) -> PolyMatrixRaw<'a> {
    let mut res = PolyMatrixRaw::zero(params, 1, 1);
    res.get_poly_mut(0, 0)[0] = val;
    res
}

pub fn reduce_copy_small(params: &Params, out: &mut [u64], inp: &[u16]) {
    for n in 0..params.crt_count {
        for z in 0..params.poly_len {
            out[n * params.poly_len + z] = barrett_coeff_u64(params, inp[z] as u64, n);
        }
    }
}

pub fn reduce_copy(params: &Params, out: &mut [u64], inp: &[u64]) {
    for n in 0..params.crt_count {
        for z in 0..params.poly_len {
            out[n * params.poly_len + z] = barrett_coeff_u64(params, inp[z], n);
        }
    }
}

pub fn to_ntt_small(a: &mut PolyMatrixNTT, b: &PolyMatrixSmall) {
    let params = a.params;
    for r in 0..a.rows {
        for c in 0..a.cols {
            let pol_src = b.get_poly(r, c);
            let pol_dst = a.get_poly_mut(r, c);
            reduce_copy_small(params, pol_dst, pol_src);
            ntt_forward(params, pol_dst);
        }
    }
}

pub fn to_ntt(a: &mut PolyMatrixNTT, b: &PolyMatrixRaw) {
    let params = a.params;
    for r in 0..a.rows {
        for c in 0..a.cols {
            let pol_src = b.get_poly(r, c);
            let pol_dst = a.get_poly_mut(r, c);
            reduce_copy(params, pol_dst, pol_src);
            ntt_forward(params, pol_dst);
        }
    }
}

pub fn to_ntt_no_reduce(a: &mut PolyMatrixNTT, b: &PolyMatrixRaw) {
    let params = a.params;
    for r in 0..a.rows {
        for c in 0..a.cols {
            let pol_src = b.get_poly(r, c);
            let pol_dst = a.get_poly_mut(r, c);
            for n in 0..params.crt_count {
                let idx = n * params.poly_len;
                pol_dst[idx..idx + params.poly_len].copy_from_slice(pol_src);
            }
            ntt_forward(params, pol_dst);
        }
    }
}

pub fn to_ntt_alloc_small<'a>(b: &PolyMatrixSmall<'a>) -> PolyMatrixNTT<'a> {
    let mut a = PolyMatrixNTT::zero(b.params, b.rows, b.cols);
    to_ntt_small(&mut a, b);
    a
}

pub fn to_ntt_alloc<'a>(b: &PolyMatrixRaw<'a>) -> PolyMatrixNTT<'a> {
    let mut a = PolyMatrixNTT::zero(b.params, b.rows, b.cols);
    to_ntt(&mut a, b);
    a
}

pub fn from_ntt(a: &mut PolyMatrixRaw, b: &PolyMatrixNTT) {
    let params = a.params;
    SCRATCH.with(|scratch_cell| {
        let scratch_vec = &mut *scratch_cell.borrow_mut();
        let scratch = scratch_vec.as_mut_slice();
        for r in 0..a.rows {
            for c in 0..a.cols {
                let pol_src = b.get_poly(r, c);
                let pol_dst = a.get_poly_mut(r, c);
                scratch[0..pol_src.len()].copy_from_slice(pol_src);
                ntt_inverse(params, scratch);
                for z in 0..params.poly_len {
                    pol_dst[z] = params.crt_compose(scratch, z);
                }
            }
        }
    });
}

pub fn from_ntt_scratch(a: &mut PolyMatrixRaw, scratch: &mut [u64], b: &PolyMatrixNTT) {
    assert_eq!(b.rows, 2);
    assert_eq!(b.cols, 1);

    let params = b.params;
    for r in 0..b.rows {
        for c in 0..b.cols {
            let pol_src = b.get_poly(r, c);
            scratch[0..pol_src.len()].copy_from_slice(pol_src);
            ntt_inverse(params, scratch);
            if r == 0 {
                let pol_dst = a.get_poly_mut(r, c);
                for z in 0..params.poly_len {
                    pol_dst[z] = params.crt_compose(scratch, z);
                }
            }
        }
    }
}

pub fn from_ntt_alloc<'a>(b: &PolyMatrixNTT<'a>) -> PolyMatrixRaw<'a> {
    let mut a = PolyMatrixRaw::zero(b.params, b.rows, b.cols);
    from_ntt(&mut a, b);
    a
}

impl<'a, 'b> Neg for &'b PolyMatrixRaw<'a> {
    type Output = PolyMatrixRaw<'a>;

    fn neg(self) -> Self::Output {
        let mut out = PolyMatrixRaw::zero(self.params, self.rows, self.cols);
        invert(&mut out, self);
        out
    }
}

impl<'a, 'b> Neg for &'b PolyMatrixNTT<'a> {
    type Output = PolyMatrixNTT<'a>;

    fn neg(self) -> Self::Output {
        let mut out = PolyMatrixNTT::zero(self.params, self.rows, self.cols);
        invert_ntt(&mut out, self);
        out
    }
}

impl<'a, 'b> Mul for &'b PolyMatrixNTT<'a> {
    type Output = PolyMatrixNTT<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = PolyMatrixNTT::zero(self.params, self.rows, rhs.cols);
        multiply(&mut out, self, rhs);
        out
    }
}

impl<'a, 'b> Add for &'b PolyMatrixNTT<'a> {
    type Output = PolyMatrixNTT<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = PolyMatrixNTT::zero(self.params, self.rows, self.cols);
        add(&mut out, self, rhs);
        out
    }
}

impl<'a, 'b> Add for &'b PolyMatrixRaw<'a> {
    type Output = PolyMatrixRaw<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = PolyMatrixRaw::zero(self.params, self.rows, self.cols);
        add_raw(&mut out, self, rhs);
        out
    }
}

