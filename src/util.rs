use crate::{params::*};
use rand::{prelude::SmallRng, thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
pub type Seed = <ChaCha20Rng as SeedableRng>::Seed;

pub fn calc_index(indices: &[usize], lengths: &[usize]) -> usize {
    let mut idx = 0usize;
    let mut prod = 1usize;
    for i in (0..indices.len()).rev() {
        idx += indices[i] * prod;
        prod *= lengths[i];
    }
    idx
}

pub fn get_test_params() -> Params {
    Params::init(2048, 
        &[65537, 1004535809], 
        2.05, 
        1,
        64, 
        12,
        33,)
}

pub fn get_seed() -> u64 {
    thread_rng().gen::<u64>()
}

pub fn get_seeded_rng() -> SmallRng {
    SmallRng::seed_from_u64(get_seed())
}

pub fn get_chacha_seed() -> Seed {
    thread_rng().gen::<[u8; 32]>()
}

pub fn get_chacha_rng() -> ChaCha20Rng {
    ChaCha20Rng::from_seed(get_chacha_seed())
}

pub fn read_arbitrary_bits(data: &[u8], bit_offs: usize, num_bits: usize) -> u64 {
    let word_off = bit_offs / 64;
    let bit_off_within_word = bit_offs % 64;
    if (bit_off_within_word + num_bits) <= 64 {
        let idx = word_off * 8;
        let val = u64::from_ne_bytes(data[idx..idx + 8].try_into().unwrap());
        (val >> bit_off_within_word) & ((1u64 << num_bits) - 1)
    } else {
        let idx = word_off * 8;
        let val = u128::from_ne_bytes(data[idx..idx + 16].try_into().unwrap());
        ((val >> bit_off_within_word) & ((1u128 << num_bits) - 1)) as u64
    }
}

pub fn write_arbitrary_bits(data: &mut [u8], mut val: u64, bit_offs: usize, num_bits: usize) {
    let word_off = bit_offs / 64;
    let bit_off_within_word = bit_offs % 64;
    val = val & ((1u64 << num_bits) - 1);
    if (bit_off_within_word + num_bits) <= 64 {
        let idx = word_off * 8;
        let mut cur_val = u64::from_ne_bytes(data[idx..idx + 8].try_into().unwrap());
        cur_val &= !(((1u64 << num_bits) - 1) << bit_off_within_word);
        cur_val |= val << bit_off_within_word;
        data[idx..idx + 8].copy_from_slice(&u64::to_ne_bytes(cur_val));
    } else {
        let idx = word_off * 8;
        let mut cur_val = u128::from_ne_bytes(data[idx..idx + 16].try_into().unwrap());
        let mask = !(((1u128 << num_bits) - 1) << bit_off_within_word);
        cur_val &= mask;
        cur_val |= (val as u128) << bit_off_within_word;
        data[idx..idx + 16].copy_from_slice(&u128::to_ne_bytes(cur_val));
    }
}
