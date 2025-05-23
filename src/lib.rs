#![feature(stdarch_x86_avx512)]

pub mod aligned_memory;
pub mod arith;
pub mod discrete_gaussian;
pub mod number_theory;
pub mod util;
pub mod ntt;
pub mod params;
pub mod poly;

pub mod ntrupacking;
pub mod npirstandard;
pub mod npirbatch;