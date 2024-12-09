#![allow(improper_ctypes)]
#![allow(unused_imports)]
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::{discrete_gaussian::*, poly::*, params::*, number_theory::*, arith::*, ntt::*};
use libc::{c_long, c_int};
use rayon::prelude::*;
use std::time::Instant;

extern "C"{
    pub fn get_sk(sk: &mut Vec<i64>, sk_inv: &mut Vec<i64>, Q: c_long, N: c_int);
}

pub fn generate_y_constants<'a>(
    params: &'a Params,
) -> Vec<PolyMatrixNTT<'a>> {
    let mut y_constants = Vec::new();
    for num_cts_log2 in 1..params.poly_len_log2 + 1 {
        let num_cts = 1 << num_cts_log2;
        
        let mut y_raw = PolyMatrixRaw::zero(params, 1, 1);
        y_raw.data[params.poly_len / num_cts] = 1;
        let y = to_ntt_alloc(&y_raw);

        y_constants.push(y);
    }

    y_constants
}

pub fn tau<'a>(
    params: &'a Params,
    index: usize,
    input: PolyMatrixRaw<'a>
) -> PolyMatrixRaw<'a> {
    let dimension = params.poly_len;
    let modulus = params.modulus;
    let mut output = PolyMatrixRaw::zero(params, 1, 1);
    for i in 0..dimension {
        let num = (i * index) / dimension;
        let rem = (i * index) % dimension;
        let data: u64 = input.get_poly(0, 0)[i] % modulus;
        if num % 2 == 1 {
            output.get_poly_mut(0, 0)[rem] = (modulus - data) % modulus;
        } else {
            output.get_poly_mut(0, 0)[rem] = data;
        };
    }
    output
}

pub fn gadgetrp<'a>(
    params: &'a Params,
    input: PolyMatrixRaw<'a>
) -> PolyMatrixRaw<'a> {
    let dimension = params.poly_len;
    let t = params.t_rp;
    let mask = 65535u64;
    let mut output = PolyMatrixRaw::zero(params, 1, t);
    for i in 0..dimension {
        let data: u64 = input.get_poly(0, 0)[i];
        for j in 0..t {
            let bit_offs = j * 16usize;
            let piece = if bit_offs >= 48 {
                0
            } else {
                (data >> bit_offs) & mask
            };
            output.get_poly_mut(0, j)[i] = piece;
        }
    }
    output
}

pub fn homomorphic_automorph<'a>(
    params: &'a Params,
    shift: usize,
    ct: &PolyMatrixNTT<'a>,
    rpk: &PolyMatrixNTT<'a>,
) -> PolyMatrixNTT<'a> {
    let ct_raw = ct.raw();
    let rho = tau(params, shift, ct_raw);
    let rhogadget = gadgetrp(params, rho);
    let mut rho_ntt = PolyMatrixNTT::zero(params, 1 , params.t_rp);
    for i in 1..params.t_rp {
        let pol_src = rhogadget.get_poly(0, i);
        let pol_dst = rho_ntt.get_poly_mut(0, i);
        reduce_copy(params, pol_dst, pol_src);
        ntt_forward(params, pol_dst);
    }
    &rho_ntt * rpk
}

pub struct NtruRp<'a> {
    ntru_params: &'a Params,
    sk: PolyMatrixNTT<'a>,
    sk_inv: PolyMatrixNTT<'a>,
    rpk: Vec<PolyMatrixNTT<'a>>,
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

        let mut rpk = Vec::new();
        let mut rng = ChaCha20Rng::from_entropy();
        let dg = DiscreteGaussian::init(ntru_params.noise_width);
        let mut g = PolyMatrixRaw::zero(ntru_params, 1, 1);
        for i in 0..ntru_params.poly_len_log2 {
            let mut btem = 1u64;
            let mut tem_rpk = PolyMatrixNTT::zero(ntru_params, ntru_params.t_rp, 1);
            for j in 0..ntru_params.t_rp {
                let mut tauf = tau(&ntru_params, (1 << (i + 1)) as usize + 1, tem_sk.clone());
  
                for k in 0..dimension{
                    g.get_poly_mut(0, 0)[k] = dg.sample(modulus, &mut rng) as u64;
                    let data = tauf.get_poly(0, 0)[k];
                    tauf.get_poly_mut(0, 0)[k] = multiply_uint_mod(data, btem, modulus);
                }
                
                g = &g + &tauf;
                let g_ntt = to_ntt_alloc(&g);
                tem_rpk.copy_into(&(&g_ntt * &sk_inv), j, 0);
                
                btem *= 65536u64;
            }
            rpk.push(tem_rpk);
        }
        
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

    pub fn encrypt(&self, pt: u64, factor: u64) -> PolyMatrixNTT<'_> {
        let mut rng = ChaCha20Rng::from_entropy();
        let dg = DiscreteGaussian::init(self.ntru_params.noise_width);
        let mut g = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
        let modulus = self.ntru_params.modulus;

        for i in 0..self.ntru_params.poly_len{
            g.get_poly_mut(0, 0)[i] = dg.sample(modulus, &mut rng) as u64;
        }
        g.get_poly_mut(0, 0)[0] += pt * self.delta_q();
        for i in 0..self.ntru_params.poly_len{
            let data = g.get_poly(0, 0)[i] as u64;
            g.get_poly_mut(0, 0)[i] = multiply_uint_mod(data, factor, modulus);
        }
        let g_ntt = to_ntt_alloc(&g);
        let res_ntt = &g_ntt * self.get_sk_inv();
        res_ntt
    }

    pub fn decrypt(&self, ct: PolyMatrixNTT<'_>) -> u64 {
        let res_ntt = &ct * self.get_sk();
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

    pub fn encryptpoly(&self, pt: PolyMatrixRaw<'a>, factor: u64) -> PolyMatrixNTT<'_> {
        let mut rng = ChaCha20Rng::from_entropy();
        let dg = DiscreteGaussian::init(self.ntru_params.noise_width);
        let mut g = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
        let modulus = self.ntru_params.modulus;

        for i in 0..self.ntru_params.poly_len{
            g.get_poly_mut(0, 0)[i] = dg.sample(modulus, &mut rng) as u64;
        }
        g = &g + &pt;
        for i in 0..self.ntru_params.poly_len{
            let data = g.get_poly(0, 0)[i] as u64;
            g.get_poly_mut(0, 0)[i] = multiply_uint_mod(data, factor, modulus);
        }
        let g_ntt = to_ntt_alloc(&g);
        let res_ntt = &g_ntt * self.get_sk_inv();
        res_ntt
    }

    pub fn decryptrp(&self, ct: PolyMatrixNTT<'_>) -> PolyMatrixRaw<'_> {
        let res_ntt = &ct * self.get_sk();
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

    pub fn ringpack(&self, 
        log_num: usize, 
        index: usize, 
        ct: &[PolyMatrixNTT<'a>], 
        y_constants: &Vec<PolyMatrixNTT<'a>>
    ) -> PolyMatrixNTT<'a> {
        if log_num == 0 {
            return ct[index].clone();
        }

        let step = 1 << (self.ntru_params.poly_len_log2 - log_num);
        let even = index;
        let odd = index + step;
        let mut cte = self.ringpack(log_num - 1, even, ct, y_constants);
        let cto = self.ringpack(log_num - 1, odd, ct, y_constants);
        let y = &y_constants[log_num - 1];

        let y_times_cto = scalar_multiply_alloc(&y, &cto);
        let mut ct_sum_1 = cte.clone();
        sub_into(&mut ct_sum_1, &y_times_cto);
        add_into(&mut cte, &y_times_cto);

        let ct_sum_1_automorphed = homomorphic_automorph(
            self.ntru_params,
            (1 << log_num) + 1,
            &ct_sum_1,
            &self.rpk[log_num - 1]
        );
        &cte + &ct_sum_1_automorphed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
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
    #[ignore]
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
    #[ignore]
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
        let ct = ntrurp.encrypt(plaintext, 1);
        let b = ntrurp.decrypt(ct);
        assert_eq!(plaintext, b, "The Enc/Decryption is OK!");
    }

    #[test]
    fn ringpack_correctness() {
        let ntru_params = Params::init(2048, 
            &[65537, 1004535809], 
            2.05, 
            1,
            64, 
            3,
            31,);
        let ntrurp = NtruRp::new(&ntru_params);
        let y_constants = generate_y_constants(&ntru_params);
        let dimension = ntru_params.poly_len;
        let modulus = ntru_params.modulus;
        let pt = ntru_params.pt_modulus;
        let n1 = invert_uint_mod(dimension.try_into().unwrap(), modulus).unwrap();
        let mut rng = ChaCha20Rng::from_entropy();
        let mut micros1 = 0;
        for _t in 0..5{
            let mut vec: Vec<u64> = vec![0; dimension as usize];
            let mut ctrp = Vec::new();
            for i in 0..dimension {
                vec[i] = rng.gen::<u64>() % pt;
                let ct = ntrurp.encrypt(vec[i].try_into().unwrap(), 1);
                let mut ct_raw = from_ntt_alloc(&ct);
                for j in 0..dimension {
                    let data = ct_raw.get_poly(0, 0)[j] as u64;
                    ct_raw.get_poly_mut(0, 0)[j] = multiply_uint_mod(data, n1, modulus);
                }
                let new_ct = to_ntt_alloc(&ct_raw);
                ctrp.push(new_ct);
            }
            println!("Begin to packing...");
            let start1 = Instant::now();
            let rho = ntrurp.ringpack(ntrurp.ntru_params.poly_len_log2, 0, &ctrp, &y_constants); 
            micros1 +=  start1.elapsed().as_micros();
            let rho_raw = from_ntt_alloc(&rho);
            let modrho = ntrurp.modreduction(rho_raw);
            let b = ntrurp.decryptrp(to_ntt_alloc(&modrho));
            (0..dimension).into_par_iter().for_each(|i| {
                assert_eq!(vec[i], b.get_poly(0, 0)[i]);
            });
        }
        println!("Ringpacking time: {} microseconds", micros1 / 5);
    }
}