#![allow(improper_ctypes)]
#![allow(unused_imports)]
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::{discrete_gaussian::*, poly::*, params::*, number_theory::*, arith::*};
use libc::{c_long, c_int};
use rayon::prelude::*;
use std::time::Instant;

extern "C"{
    pub fn get_sk(sk: &mut Vec<i64>, sk_inv: &mut Vec<i64>, Q: c_long, N: c_int);
}

pub fn rightrot<'a>(params: &'a Params, offset: usize, input: PolyMatrixRaw<'a>) -> PolyMatrixRaw<'a> {
    let dimension = params.poly_len;
    let modulus = params.modulus;
    let mut output = PolyMatrixRaw::zero(params, 1, 1);
    for i in 0..dimension {
        let new_index = (i + offset) % dimension;
        let data: u64 = input.get_poly(0, 0)[i] % modulus;
        if i + offset >= dimension {
            output.get_poly_mut(0, 0)[new_index] = (modulus - data) % modulus;
        } else {
            output.get_poly_mut(0, 0)[new_index] = data;
        };
    }

    output
}

pub fn tau<'a>(params: &'a Params, index: usize, input: PolyMatrixRaw<'a>) -> PolyMatrixRaw<'a> {
    let dimension = params.poly_len;
    let modulus = params.modulus;
    let mut output = PolyMatrixRaw::zero(params, 1, 1);
    for i in 0..dimension {
        let new_index = (i * index) % dimension;
        let data: u64 = input.get_poly(0, 0)[i] % modulus;
        if ((i * index) / dimension) % 2 == 1 {
            output.get_poly_mut(0, 0)[new_index] = (modulus - data) % modulus;
        } else {
            output.get_poly_mut(0, 0)[new_index] = data;
        };
    }
    output
}

pub fn gadgetrp<'a>(params: &'a Params, input: PolyMatrixRaw<'a>) -> PolyMatrixRaw<'a> {
    let dimension = params.poly_len;
    let t = params.t_rp;
    let b = 65536u64;
    let mut output = PolyMatrixRaw::zero(params, 1, t);
    for i in 0..dimension {
        let mut data: u64 = input.get_poly(0, 0)[i];
        for j in 0..t {
            output.get_poly_mut(0, j)[i] = data % b;
            data = data / b;
        }
    }
    output
}

pub struct NtruRp<'a> {
    ntru_params: &'a Params,
    sk: PolyMatrixNTT<'a>,
    sk_inv: PolyMatrixNTT<'a>,
    rpk: PolyMatrixNTT<'a>,
}

impl<'a> NtruRp<'a> {
    pub fn new(ntru_params: &'a Params) -> NtruRp<'a> {
        let dimension = ntru_params.poly_len;
        let modulus = ntru_params.modulus;
        let mut tem_sk = PolyMatrixRaw::zero(ntru_params, 1, 1);
        let mut tem_skinv = PolyMatrixRaw::zero(ntru_params, 1, 1);
        let mut vec: Vec<i64> = vec![0; dimension as usize];
        let mut vec_inv: Vec<i64> = vec![0; dimension as usize];
        unsafe {    
            get_sk(&mut vec, &mut vec_inv, modulus as c_long, dimension as c_int);
        }
        for i in 0..dimension as usize {
            tem_sk.get_poly_mut(0, 0)[i] = vec[i] as u64 % modulus;
            tem_skinv.get_poly_mut(0, 0)[i] = vec_inv[i] as u64 % modulus;
        }
        let sk = to_ntt_alloc(&tem_sk);
        let sk_inv = to_ntt_alloc(&tem_skinv);

        let mut tem_rpk = PolyMatrixNTT::zero(ntru_params, ntru_params.t_rp, ntru_params.poly_len_log2);
        let mut rng = ChaCha20Rng::from_entropy();
        let dg = DiscreteGaussian::init(ntru_params.noise_width);
        let mut g = PolyMatrixRaw::zero(ntru_params, 1, 1);
        for i in 0..ntru_params.poly_len_log2 {
            let mut btem = 1u64;
            for j in 0..ntru_params.t_rp {
                let mut tauf = tau(&ntru_params, (1 << (i + 1)) as usize + 1, tem_sk.clone());
                
                for k in 0..dimension{
                    g.get_poly_mut(0, 0)[k] = dg.sample(modulus, &mut rng) as u64;
                    let data = tauf.get_poly(0, 0)[k];
                    tauf.get_poly_mut(0, 0)[k] = multiply_uint_mod(data, btem, modulus);
                }
                
                g = &g + &tauf;
                let g_ntt = to_ntt_alloc(&g);
                tem_rpk.copy_into(&(&g_ntt * &sk_inv), j, i);
                
                btem *= 65536u64;
            }
        }
        let rpk = tem_rpk.clone();
        
        NtruRp {
            ntru_params, sk, sk_inv, rpk,
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

    pub fn encrypt(&self, pt: u64) -> PolyMatrixRaw<'_> {
        let mut rng = ChaCha20Rng::from_entropy();
        let dg = DiscreteGaussian::init(self.ntru_params.noise_width);
        let mut g = PolyMatrixRaw::zero(self.ntru_params, 1, 1);

        for i in 0..self.ntru_params.poly_len{
            g.get_poly_mut(0, 0)[i] = dg.sample(self.ntru_params.modulus, &mut rng) as u64;
        }
        g.get_poly_mut(0, 0)[0] += pt * self.delta_q();
        let g_ntt = to_ntt_alloc(&g);
        let res_ntt = &g_ntt * self.get_sk_inv();
        let res = from_ntt_alloc(&res_ntt);
        res
    }

    pub fn decrypt(&self, ct: PolyMatrixRaw<'_>) -> u64 {
        let ct_ntt = to_ntt_alloc(&ct);
        let res_ntt = &ct_ntt * self.get_sk();
        let res = from_ntt_alloc(&res_ntt);
        let quotient = res.get_poly(0, 0)[0].div_euclid(self.delta_q());
        let remainder = res.get_poly(0, 0)[0].rem_euclid(self.delta_q());

        let b = if remainder * 2 >= self.delta_q() {
            (quotient + 1) % self.ntru_params.pt_modulus
        } else {
            quotient % self.ntru_params.pt_modulus
        };
        b
    }

    pub fn encryptpoly(&self, pt: PolyMatrixRaw<'a>) -> PolyMatrixRaw<'_> {
        let mut rng = ChaCha20Rng::from_entropy();
        let dg = DiscreteGaussian::init(self.ntru_params.noise_width);
        let mut g = PolyMatrixRaw::zero(self.ntru_params, 1, 1);

        for i in 0..self.ntru_params.poly_len{
            g.get_poly_mut(0, 0)[i] = dg.sample(self.ntru_params.modulus, &mut rng) as u64;
        }
        g = &g + &pt;
        let g_ntt = to_ntt_alloc(&g);
        let res_ntt = &g_ntt * self.get_sk_inv();
        let res = from_ntt_alloc(&res_ntt);
        res
    }

    pub fn decryptrp(&self, ct: PolyMatrixRaw<'_>) -> PolyMatrixRaw<'_> {
        let ct_ntt = to_ntt_alloc(&ct);
        let res_ntt = &ct_ntt * self.get_sk();
        let res = from_ntt_alloc(&res_ntt);

        let mut pt_x = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
        let delta = self.delta_q1();
        let pt = self.ntru_params.pt_modulus;
        let moduli = self.ntru_params.moduli[0];
        for i in 0..self.ntru_params.poly_len {
            let data: u64 = res.get_poly(0, 0)[i] % moduli;
            let quotient = data.div_euclid(delta);
            let remainder = data.rem_euclid(delta);
            let b = if remainder * 2 >= delta {
                (quotient + 1) % pt
            } else {
                quotient % pt
            };
            pt_x.get_poly_mut(0, 0)[i] = b;
        }
        pt_x
    }

    pub fn modreduction(&self, ct: PolyMatrixRaw<'_>) -> PolyMatrixRaw<'_> {
        assert_eq!(ct.rows, 1); assert_eq!(ct.cols, 1);
        let dimension = self.ntru_params.poly_len;
        let mut output = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
        let moduli = self.ntru_params.moduli[1];
        for i in 0..dimension {
            let quotient = ct.get_poly(0, 0)[i].div_euclid(moduli);
            output.get_poly_mut(0, 0)[i] = quotient;
        }
        output
    }

    pub fn ringpack(&self, log_num: usize, index: usize, ct: &PolyMatrixRaw<'a>) -> PolyMatrixRaw<'a> {
        if log_num == 0 {
            return ct.submatrix(index, 0, 1, 1).clone()
        } else {
            let num: usize = 1 << log_num;
            let n2t: usize = self.ntru_params.poly_len / num;
            let even = index;
            let odd = index + n2t;
            // let (cte, mut cto) = rayon::join(
            //     || {
            //         let cte = self.ringpack(log_num - 1, even, ct);
            //         cte
            //     },
            //     || {
            //         let cto = self.ringpack(log_num - 1, odd, ct);
            //         cto
            //     }
            // );
            let cte = self.ringpack(log_num - 1, even, ct);
            let mut cto = self.ringpack(log_num - 1, odd, ct);
 
                
            cto = rightrot(self.ntru_params, n2t, cto);
            let alpha = &cte + &cto;
            // let (rho_ntt, rpk_ntt) = rayon::join(
            //     || {
            //         cto = &cte + &(-&cto);
            //         let rho = tau(self.ntru_params, num + 1, cto);
            //         let rhogadget = gadgetrp(self.ntru_params, rho);
            //         let rho_ntt = to_ntt_alloc(&rhogadget);
            //         rho_ntt
            //     },
            //     || {
            //         let rpk_ntt = self.rpk.submatrix(0, log_num - 1, self.ntru_params.t_rp, 1);
            //         rpk_ntt
            //     }
            // );
            cto = &cte + &(-&cto);
            let rho = tau(self.ntru_params, num + 1, cto);
            let rhogadget = gadgetrp(self.ntru_params, rho);
            let rho_ntt = to_ntt_alloc(&rhogadget);
            let rpk_ntt = self.rpk.submatrix(0, log_num - 1, self.ntru_params.t_rp, 1);
            let res_ntt = &rho_ntt * &rpk_ntt;
            
            let mut res = from_ntt_alloc(&res_ntt);
            res = &alpha + &res;
            return res
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_factor_correctness() {
        let ntru_params = Params::init(2048, 
            &[65537, 1004535809], 
            2.05, 
            1, 
            64, 
            12,
            33,);
        let ntrurp = NtruRp::new(&ntru_params);
        assert_eq!(ntrurp.delta_q(), ntrurp.ntru_params.modulus / ntrurp.ntru_params.pt_modulus, "The ciphertext scaler is right!");
        assert_eq!(ntrurp.delta_q1(), ntrurp.ntru_params.moduli[0] / ntrurp.ntru_params.pt_modulus, "The packing scaler is right!");
    }

    #[test]
    fn sk_correctness() {
        let ntru_params = Params::init(2048, 
            &[65537, 1004535809], 
            2.05, 
            1,
            64, 
            12,
            30,);
        let ntrurp = NtruRp::new(&ntru_params);
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
    fn enc_dec_correctness() {
        let ntru_params = Params::init(2048, 
            &[65537, 1004535809], 
            2.05, 
            1,
            64, 
            12,
            30,);
        let ntrurp = NtruRp::new(&ntru_params);
        let mut rng = ChaCha20Rng::from_entropy();
        let plaintext: u64 = rng.gen::<u64>() % ntrurp.ntru_params.pt_modulus;
        let ct = ntrurp.encrypt(plaintext);
        let b = ntrurp.decrypt(ct);
        assert_eq!(plaintext, b, "The Enc/Decryption is OK!");
    }

    #[test]
    fn ringpack_correctness() {
        let ntru_params = Params::init(4096, 
            &[65537, 1004535809], 
            2.05, 
            1,
            64, 
            3,
            31,);
        let ntrurp = NtruRp::new(&ntru_params);
        let dimension = ntru_params.poly_len;
        let modulus = ntru_params.modulus;
        let pt = ntru_params.pt_modulus;
        let n1 = invert_uint_mod(dimension.try_into().unwrap(), modulus).unwrap();
        let mut rng = ChaCha20Rng::from_entropy();
        let mut vec: Vec<u64> = vec![0; dimension as usize];
        let mut ctrp = PolyMatrixRaw::zero(&ntru_params, dimension, 1);
        let mut micros1 = 0;
        for _t in 0..5{
            for i in 0..dimension {
                vec[i] = rng.gen::<u64>() % pt;
                let ct = ntrurp.encrypt(vec[i].try_into().unwrap());
                for j in 0..dimension {
                    let data = ct.get_poly(0, 0)[j] as u64;
                    ctrp.get_poly_mut(i, 0)[j] = multiply_uint_mod(data, n1, modulus);
                }       
            }

            let start1 = Instant::now();
            let rho = ntrurp.ringpack(ntrurp.ntru_params.poly_len_log2, 0, &ctrp); 
            let duration1 = start1.elapsed();
            micros1 += duration1.as_micros();
            let modrho = ntrurp.modreduction(rho);
            let b = ntrurp.decryptrp(modrho);
            (0..dimension).into_par_iter().for_each(|i| {
                assert_eq!(vec[i], b.get_poly(0, 0)[i]);
            });
        }

        println!("Ringpacking time: {} microseconds", micros1 / 20);

    }
}