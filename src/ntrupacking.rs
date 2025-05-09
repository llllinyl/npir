#![allow(improper_ctypes)]
#![allow(unused_imports)]
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::{discrete_gaussian::*, poly::*, params::*, number_theory::*, arith::*, ntt::*};
use libc::{c_long, c_int};
use rayon::prelude::*;
use std::time::Instant;

extern "C"{
    pub fn get_sk(sk: *mut i64, sk_inv: *mut i64, Q: c_long, N: c_int);
}

pub fn generate_y_constants<'a>(params: &'a Params) -> Vec<PolyMatrixNTT<'a>> {
    let mut y_constants = Vec::with_capacity(params.poly_len_log2);
    
    for level in 1..=params.poly_len_log2 {
        let num_components = 1 << level;
        let mut y_raw = PolyMatrixRaw::zero(params, 1, 1);
        
        y_raw.data[params.poly_len / num_components] = 1;
        
        y_constants.push(to_ntt_alloc(&y_raw));
    }

    y_constants
}

pub fn tau<'a>(params: &'a Params, rotation_index: usize, input: PolyMatrixRaw<'a>) -> PolyMatrixRaw<'a> {
    let dimension = params.poly_len;
    let modulus = params.modulus;
    let mut output = PolyMatrixRaw::zero(params, 1, 1);
    
    for coeff_idx in 0..dimension {
        let rotation = coeff_idx * rotation_index;
        let group = rotation / dimension;
        let new_position = rotation % dimension;
        let value = input.get_poly(0, 0)[coeff_idx] % params.modulus;
        
        output.get_poly_mut(0, 0)[new_position] = if group % 2 == 1 {
            modulus - value
        } else {
            value
        };
    }
    output
}

pub fn gadgetrp<'a>(params: &'a Params, input: PolyMatrixRaw<'a>, bits_per_component: usize, num_components: usize) -> PolyMatrixRaw<'a> {
    let dimension = params.poly_len;
    let bit_mask = (1 << bits_per_component) - 1;
    let mut output = PolyMatrixRaw::zero(params, 1, num_components);
    
    for coeff_idx in 0..dimension {
        let value = input.get_poly(0, 0)[coeff_idx] % params.modulus;
        
        for component_idx in 0..num_components {
            let bit_offset = component_idx * bits_per_component;
            let component_value = if bit_offset >= num_components * bits_per_component {
                0
            } else {
                (value >> bit_offset) & bit_mask
            };
            
            output.get_poly_mut(0, component_idx)[coeff_idx] = component_value;
        }
    }
    output
}

pub fn homomorphic_automorph<'a>(params: &'a Params, rotation_shift: usize, ciphertext: &PolyMatrixNTT<'a>,
    rpk: &PolyMatrixNTT<'a>, bits_per_component: usize, num_components: usize) -> PolyMatrixNTT<'a> {
    let mut res = PolyMatrixNTT::zero(params, 1, 1);
    let rotated_raw = tau(params, rotation_shift, ciphertext.raw());
    let rotated_gadget = gadgetrp(
        params, 
        rotated_raw, 
        bits_per_component, 
        num_components
    );
    
    let mut rotated_ntt = PolyMatrixNTT::zero(params, 1, num_components);
    for component_idx in 0..num_components {
        let src_poly = rotated_gadget.get_poly(0, component_idx);
        let dst_poly = rotated_ntt.get_poly_mut(0, component_idx);
        
        reduce_copy(params, dst_poly, src_poly);
        ntt_forward(params, dst_poly);
    }
    multiply_no_reduce(&mut res, &rotated_ntt, rpk, 0);
    res
}

pub fn fast_add_into_no_reduce(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT) {
    let a_slc = a.as_slice();
    let res_slc = res.as_mut_slice();
    for (res_chunk, a_chunk) in res_slc.chunks_exact_mut(8).zip(a_slc.chunks_exact(8)) {
        for i in 0..8 {
            res_chunk[i] += a_chunk[i];
        }
    }
}

pub fn fast_add_into(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT) {
    let params = res.params;
    for i in 0..res.rows {
        for j in 0..res.cols {
            let res_poly = res.get_poly_mut(i, j);
            let a_poly = a.get_poly(i, j);
            for c in 0..params.crt_count {
                for i in 0..params.poly_len {
                    let idx = c * params.poly_len + i;
                    unsafe {
                        let p_res = res_poly.as_mut_ptr().add(idx);
                        let p_a = a_poly.as_ptr().add(idx);
                        let val = *p_res + *p_a;
                        let reduced =
                            barrett_raw_u64(val, params.barrett_cr_1[c], params.moduli[c]);
                        *p_res = reduced;
                    }
                }
            }
        }
    }
}

pub struct NtruRp<'a> {
    ntru_params: &'a Params,
    pub sk: PolyMatrixNTT<'a>,
    pub sk_inv: PolyMatrixNTT<'a>,
    pub rpk: Vec<PolyMatrixNTT<'a>>,
    pub tpk: usize,
    pub bpk: usize,
}

impl<'a> NtruRp<'a> {
    pub fn new(ntru_params: &'a Params, tpk: usize) -> NtruRp<'a> {
        let dimension = ntru_params.poly_len;
        let modulus = ntru_params.modulus;
        let mut tem_sk = PolyMatrixRaw::zero(ntru_params, 1, 1);
        let mut tem_skinv = PolyMatrixRaw::zero(ntru_params, 1, 1);
        let mut vec = vec![0i64; dimension as usize];
        let mut vec_inv = vec![0i64; dimension as usize];
        unsafe {
            get_sk(vec.as_mut_ptr(), vec_inv.as_mut_ptr(), modulus as c_long, dimension as c_int);
        }
        for i in 0..dimension as usize {
            tem_sk.get_poly_mut(0, 0)[i] = (vec[i] as u64).rem_euclid(modulus);
            tem_skinv.get_poly_mut(0, 0)[i] = (vec_inv[i] as u64).rem_euclid(modulus);
        }
        let sk = to_ntt_alloc(&tem_sk);
        let sk_inv = to_ntt_alloc(&tem_skinv);

        let bpk = ((ntru_params.modulus as f64).log2() / tpk as f64).ceil() as usize;
        let mut rpk = Vec::new();
        let mut rng = ChaCha20Rng::from_entropy();
        let dg = DiscreteGaussian::init(ntru_params.noise_width);
        let mut g = PolyMatrixRaw::zero(ntru_params, 1, 1);
        for i in 0..ntru_params.poly_len_log2 {
            let mut btem = 1u64;
            let mut tem_rpk = PolyMatrixNTT::zero(ntru_params, tpk, 1);
            for j in 0..tpk {
                let mut tauf = tau(&ntru_params, (1 << (i + 1)) as usize + 1, tem_sk.clone());
  
                for k in 0..dimension{
                    g.get_poly_mut(0, 0)[k] = dg.sample(modulus, &mut rng) as u64;
                    let data = tauf.get_poly(0, 0)[k];
                    tauf.get_poly_mut(0, 0)[k] = multiply_uint_mod(data, btem, modulus);
                }
                
                g = &g + &tauf;
                let g_ntt = to_ntt_alloc(&g);
                tem_rpk.copy_into(&(&g_ntt * &sk_inv), j, 0);
                
                btem *= (1 << bpk) as u64;
            }
            rpk.push(tem_rpk);
        }
        
        NtruRp {
            ntru_params, sk, sk_inv, rpk, tpk, bpk,
        }
    }

    pub fn delta_q(&self) -> u64 {
        self.ntru_params.modulus / self.ntru_params.pt_modulus
    }

    pub fn delta_q1(&self) -> u64 {
        self.ntru_params.moduli[0] / self.ntru_params.pt_modulus
    }

    pub fn get_sk(&self) -> &PolyMatrixNTT<'_> {
        &self.sk
    }

    pub fn get_sk_inv(&self) -> &PolyMatrixNTT<'_> {
        &self.sk_inv
    }

    pub fn encrypt(&self, plaintext: u64, scaling_factor: u64) -> PolyMatrixNTT<'_> {
        let mut rng = ChaCha20Rng::from_entropy();
        let dg = DiscreteGaussian::init(self.ntru_params.noise_width);
        let modulus = self.ntru_params.modulus;

        let mut noise_poly = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
        
        for coeff in 0..self.ntru_params.poly_len {
            noise_poly.get_poly_mut(0, 0)[coeff] = dg.sample(modulus, &mut rng) as u64;
        }
        
        noise_poly.get_poly_mut(0, 0)[0] += plaintext * self.delta_q();
        
        for coeff in 0..self.ntru_params.poly_len {
            let value = noise_poly.get_poly(0, 0)[coeff].rem_euclid(modulus);
            noise_poly.get_poly_mut(0, 0)[coeff] = 
                multiply_uint_mod(value, scaling_factor, modulus);
        }
        
        let ntt_noise = to_ntt_alloc(&noise_poly);
        &ntt_noise * self.get_sk_inv()
    }

    pub fn encryptpolywoscale(&self, plaintext_poly: PolyMatrixRaw<'a>, scaling_factor: u64) -> PolyMatrixNTT<'_> {
        let mut rng = ChaCha20Rng::from_entropy();
        let dg = DiscreteGaussian::init(self.ntru_params.noise_width);
        let modulus = self.ntru_params.modulus;
        
        let mut noise_poly = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
        
        for coeff in 0..self.ntru_params.poly_len {
            noise_poly.get_poly_mut(0, 0)[coeff] = dg.sample(modulus, &mut rng) as u64;
        }
        
        noise_poly = &noise_poly + &plaintext_poly;
        
        for coeff in 0..self.ntru_params.poly_len {
            let value = noise_poly.get_poly(0, 0)[coeff].rem_euclid(modulus);
            noise_poly.get_poly_mut(0, 0)[coeff] = 
                multiply_uint_mod(value, scaling_factor, modulus);
        }
        
        let ntt_noise = to_ntt_alloc(&noise_poly);
        &ntt_noise * self.get_sk_inv()
    }

    pub fn decrypt(&self, ciphertext: PolyMatrixNTT<'_>) -> u64 {
        let raw_poly_ntt = &ciphertext * self.get_sk();
        let raw_poly = from_ntt_alloc(&raw_poly_ntt);
        
        let first_coeff = raw_poly.get_poly(0, 0)[0].rem_euclid(self.ntru_params.modulus);
        let quotient = first_coeff.div_euclid(self.delta_q());
        let remainder = first_coeff.rem_euclid(self.delta_q());
        
        if remainder * 2 >= self.delta_q() {
            (quotient + 1) % self.ntru_params.pt_modulus
        } else {
            quotient
        }
    }

    pub fn decryptrp(&self, ciphertext: PolyMatrixNTT<'_>) -> PolyMatrixRaw<'_> {
        let raw_poly_ntt = &ciphertext * self.get_sk();
        let raw_poly = from_ntt_alloc(&raw_poly_ntt);
        
        let mut result = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
        let delta = self.delta_q1();
        let pt_modulus = self.ntru_params.pt_modulus;
        let moduli = self.ntru_params.moduli[0];
        
        for coeff in 0..self.ntru_params.poly_len {
            let value = raw_poly.get_poly(0, 0)[coeff].rem_euclid(moduli);
            let quotient = value.div_euclid(delta);
            let remainder = value.rem_euclid(delta);
            
            let rounded = if remainder * 2 >= delta {
                quotient + 1
            } else {
                quotient
            };
            result.get_poly_mut(0, 0)[coeff] = rounded.rem_euclid(pt_modulus);
        }
        
        result
    }

    pub fn decryptpoly(&self, ciphertext: PolyMatrixNTT<'_>) -> PolyMatrixRaw<'_> {
        let raw_poly_ntt = &ciphertext * self.get_sk();
        let raw_poly = from_ntt_alloc(&raw_poly_ntt);
        
        let mut result = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
        let delta = self.delta_q();
        let pt_modulus = self.ntru_params.pt_modulus;
        let modulus = self.ntru_params.modulus;
        
        for coeff in 0..self.ntru_params.poly_len {
            let value = raw_poly.get_poly(0, 0)[coeff].rem_euclid(modulus);
            let quotient = value.div_euclid(delta);
            let remainder = value.rem_euclid(delta);
            
            let rounded = if remainder * 2 >= delta {
                (quotient + 1).rem_euclid(pt_modulus)
            } else {
                quotient
            };
            result.get_poly_mut(0, 0)[coeff] = rounded;
        }
        
        result
    }

    pub fn modreduction(&self, ciphertext: PolyMatrixRaw<'_>) -> PolyMatrixRaw<'_> {
        let mut result = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
        let moduli = self.ntru_params.moduli[1];
        
        for coeff in 0..self.ntru_params.poly_len {
            result.get_poly_mut(0, 0)[coeff] = 
                ciphertext.get_poly(0, 0)[coeff].div_euclid(moduli);
        }
        
        result
    }

    pub fn packing(&self, recursion_level: usize, index: usize, ciphertexts: &[PolyMatrixNTT<'a>], y_constants: &[PolyMatrixNTT<'a>]) -> PolyMatrixNTT<'a> {
        if recursion_level == 0 {
            return ciphertexts[index].clone();
        }
    
        let step_size = 1 << (self.ntru_params.poly_len_log2 - recursion_level);
        
        let even_index = index;
        let odd_index = index + step_size;
    
        let mut even_ciphertext = self.packing(
            recursion_level - 1, 
            even_index, 
            ciphertexts, 
            y_constants,
        );
        let odd_ciphertext = self.packing(
            recursion_level - 1, 
            odd_index, 
            ciphertexts, 
            y_constants,
        );
    
        let y_constant = &y_constants[recursion_level - 1];
    
        let y_times_odd = scalar_multiply_alloc(y_constant, &odd_ciphertext);
    
        let mut diff = even_ciphertext.clone();
        sub_into(&mut diff, &y_times_odd);
        fast_add_into_no_reduce(&mut even_ciphertext, &y_times_odd);
    
        let automorphed_diff = homomorphic_automorph(
            self.ntru_params,
            (1 << recursion_level) + 1,
            &diff,
            &self.rpk[recursion_level - 1],
            self.bpk,
            self.tpk,
        );
        fast_add_into(&mut even_ciphertext, &automorphed_diff);
        even_ciphertext
    }

    pub fn packing_for(&self, recursion_depth: usize, ciphertexts: &[PolyMatrixNTT<'a>], y_constants: &Vec<PolyMatrixNTT<'a>>) -> PolyMatrixNTT<'a> {
        let mut working_ciphertexts = Vec::new();
        for number in 0..self.ntru_params.poly_len{
            working_ciphertexts.push(ciphertexts[number].clone());
        }
        for level in 0..recursion_depth{
            let step = 1 << (self.ntru_params.poly_len_log2 - level - 1);
            for j in 0..step{
                let mut left_ciphertext = working_ciphertexts[j].clone();
                let right_ciphertext = working_ciphertexts[j + step].clone();
                let y_constant = &y_constants[level];

                let y_times_right = scalar_multiply_alloc(&y_constant, &right_ciphertext);
                let mut difference = left_ciphertext.clone();
                sub_into(&mut difference, &y_times_right);
                fast_add_into_no_reduce(&mut left_ciphertext, &y_times_right);
                let automorphed_difference = homomorphic_automorph(
                    self.ntru_params,
                    (1 << (level + 1)) + 1,
                    &difference,
                    &self.rpk[level],
                    self.bpk,
                    self.tpk,
                );
                fast_add_into(&mut left_ciphertext, &automorphed_difference);
                working_ciphertexts[j] = left_ciphertext.clone();
            }
        }
        working_ciphertexts[0].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn sk_correctness() {
        let ntru_params = Params::init(2048, 
            &[2013265921, 5767169], 
            2.05, 
            1,
            64, 
            30,);
        let ntrurp = NtruRp::new(&ntru_params, 3);
        let res_ntt = ntrurp.get_sk() * ntrurp.get_sk_inv();
        let mut res = from_ntt_alloc(&res_ntt);
        assert_eq!(res.rows, 1);
        assert_eq!(res.cols, 1);
        assert_eq!(res.get_poly_mut(0, 0)[0], 1);
        for i in 1..ntru_params.poly_len {
            assert_eq!(res.get_poly_mut(0, 0)[i], 0);
        }
    }

    #[test]
    #[ignore]
    fn enc_dec_correctness() {
        let ntru_params = Params::init(2048, 
            &[2013265921, 5767169], 
            2.05, 
            1,
            64, 
            30,);
        let ntrurp = NtruRp::new(&ntru_params, 3);
        let mut rng = ChaCha20Rng::from_entropy();
        let plaintext: u64 = rng.gen::<u64>().rem_euclid(ntrurp.ntru_params.pt_modulus);
        let ct = ntrurp.encrypt(plaintext, 1);
        let b = ntrurp.decrypt(ct);
        assert_eq!(plaintext, b, "The Enc/Decryption is OK!");
    }

    #[test]
    //#[ignore]
    fn packing_correctness() {
        let ntru_params = Params::init(2048, 
            // &[5767169, 2013265921], 
            &[23068673, 1004535809], 
            2.05, 
            1,
            256, 
            31,);
        let ntrurp = NtruRp::new(&ntru_params, 3);
        let y_constants = generate_y_constants(&ntru_params);
        let dimension = ntru_params.poly_len;
        let modulus = ntru_params.modulus;
        let pt = ntru_params.pt_modulus;
        let n1 = invert_uint_mod(dimension.try_into().unwrap(), modulus).unwrap();
        let mut rng = ChaCha20Rng::from_entropy();
        let mut micros1 = 0;
        for _t in 0..6{
            let mut vec: Vec<u64> = vec![0; dimension as usize];
            let mut ctrp = Vec::new();
            for i in 0..dimension {
                vec[i] = rng.gen::<u64>().rem_euclid(pt);
                let ct = ntrurp.encrypt(vec[i].try_into().unwrap(), n1);
                ctrp.push(ct);
            }

            println!("Begin to packing...");
            let start1 = Instant::now();
            let rho = ntrurp.packing(ntru_params.poly_len_log2, 0, &ctrp, &y_constants); 
            if _t != 0 {
                micros1 +=  start1.elapsed().as_micros();
            }
            let rho_raw = from_ntt_alloc(&rho);
            let modrho = ntrurp.modreduction(rho_raw);
            let b = ntrurp.decryptrp(to_ntt_alloc(&modrho));
            (0..dimension).into_par_iter().for_each(|i| {
                assert_eq!(vec[i], b.get_poly(0, 0)[i]);
            });
        }
        println!("packing time: {} microseconds", micros1 / 5);
    }

    #[test]
    // #[ignore]
    fn packing_for_correctness() {
        let ntru_params = Params::init(2048, 
            &[23068673, 1004535809], 
            2.05, 
            1,
            64, 
            31,);
        let ntrurp = NtruRp::new(&ntru_params, 3);
        let y_constants = generate_y_constants(&ntru_params);
        let dimension = ntru_params.poly_len;
        let modulus = ntru_params.modulus;
        let pt = ntru_params.pt_modulus;
        let n1 = invert_uint_mod(dimension.try_into().unwrap(), modulus).unwrap();
        let mut rng = ChaCha20Rng::from_entropy();
        let mut micros1 = 0;
        for _t in 0..6{
            let mut vec: Vec<u64> = vec![0; dimension as usize];
            let mut ctrp = Vec::new();
            for i in 0..dimension {
                vec[i] = rng.gen::<u64>().rem_euclid(pt);
                let ct = ntrurp.encrypt(vec[i].try_into().unwrap(), n1);
                ctrp.push(ct);
            }
            println!("Begin to packing...");
            let start1 = Instant::now();
            let rho = ntrurp.packing_for(ntru_params.poly_len_log2, &ctrp, &y_constants); 
            if _t != 0 {
                micros1 +=  start1.elapsed().as_micros();
            }
            let rho_raw = from_ntt_alloc(&rho);
            let modrho = ntrurp.modreduction(rho_raw);
            let b = ntrurp.decryptrp(to_ntt_alloc(&modrho));
            (0..dimension).into_par_iter().for_each(|i| {
                assert_eq!(vec[i], b.get_poly(0, 0)[i]);
            });
        }
        println!("packing_for time: {} microseconds", micros1 / 5);
    }
}
