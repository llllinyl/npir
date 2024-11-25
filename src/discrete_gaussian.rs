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

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::*;

    #[test]
    fn dg_seems_okay() {
        let params = get_test_params();
        let dg = DiscreteGaussian::init(params.noise_width);
        let mut rng = get_chacha_rng();
        let mut v = Vec::new();
        let trials = 10000;
        let mut sum = 0;
        for _ in 0..trials {
            let val = dg.sample(params.modulus, &mut rng);
            let mut val_i64 = val as i64;
            if val_i64 >= (params.modulus as i64) / 2 {
                val_i64 -= params.modulus as i64;
            }
            v.push(val_i64);
            sum += val_i64;
        }
        let expected_mean = 0;
        let computed_mean = sum as f64 / trials as f64;
        let expected_std_dev = params.noise_width / f64::sqrt(2f64 * std::f64::consts::PI);
        let std_dev_of_mean = expected_std_dev / f64::sqrt(trials as f64);
        println!("mean:: expected: {}, got: {}", expected_mean, computed_mean);
        assert!(f64::abs(computed_mean) < std_dev_of_mean * 5f64);

        let computed_variance: f64 = v
            .iter()
            .map(|x| (computed_mean - (*x as f64)).powi(2))
            .sum::<f64>()
            / (v.len() as f64);
        let computed_std_dev = computed_variance.sqrt();
        println!(
            "std_dev:: expected: {}, got: {}",
            expected_std_dev, computed_std_dev
        );
        assert!((computed_std_dev - expected_std_dev).abs() < (expected_std_dev * 0.1));
    }

    #[test]
    fn cdf_table_seems_okay() {
        let dg = DiscreteGaussian::init(6.4);
        println!("{:?}", dg.cdf_table);
    }
}
