use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use subtle::ConditionallySelectable;
use subtle::ConstantTimeGreater;

use crate::poly::*;
use std::f64::consts::PI;

pub const NUM_WIDTHS: usize = 4;


pub struct DiscreteGaussian {
    pub weighted_index: WeightedIndex<f64>,
    pub cdf_table: Vec<u64>,
    pub max_val: i64,
}

impl DiscreteGaussian {
    pub fn init(noise_width: f64) -> Self {
        let max_val = (noise_width * (NUM_WIDTHS as f64)).ceil() as i64;
        let mut table = Vec::new();
        let mut total = 0.0;

        for i in -max_val..max_val + 1 {
            let p_val = f64::exp(-PI * f64::powi(i as f64, 2) / f64::powi(noise_width, 2));
            table.push(p_val);
            total += p_val;
        }

        let mut cdf_table = Vec::new();
        let mut cum_prob = 0.0;

        for p_val in &table {
            cum_prob += p_val / total;
            let cum_prob_u64 = (cum_prob * (u64::MAX as f64)).round() as u64;
            cdf_table.push(cum_prob_u64);
        }

        Self {
            weighted_index: WeightedIndex::new(table).unwrap(),
            cdf_table,
            max_val,
        }
    }

    pub fn sample(&self, modulus: u64, rng: &mut ChaCha20Rng) -> u64 {
        let sampled_val = rng.gen::<u64>();
        let len = (2 * self.max_val + 1) as usize;
        let mut to_output = 0;

        for i in (0..len).rev() {
            let mut out_val = (i as i64) - self.max_val;
            if out_val < 0 {
                out_val += modulus as i64;
            }
            let out_val_u64 = out_val as u64;

            let point = self.cdf_table[i];

            let cmp = !(sampled_val.ct_gt(&point));
            to_output.conditional_assign(&out_val_u64, cmp);
        }
        to_output
    }

    pub fn fast_sample(&self, modulus: u64, rng: &mut ChaCha20Rng) -> u64 {
        let sampled_val = self.weighted_index.sample(rng);
        let mut val = (sampled_val as i64) - self.max_val;
        if val < 0 {
            val += modulus as i64;
        }
        val as u64
    }

    pub fn sample_matrix(&self, p: &mut PolyMatrixRaw, rng: &mut ChaCha20Rng) {
        let modulus = p.get_params().modulus;
        for r in 0..p.rows {
            for c in 0..p.cols {
                let poly = p.get_poly_mut(r, c);
                for z in 0..poly.len() {
                    let s = self.sample(modulus, rng);
                    poly[z] = s;
                }
            }
        }
    }
}
