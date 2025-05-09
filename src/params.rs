use crate::{arith::*, ntt::*, number_theory::*};

pub const MAX_MODULI: usize = 4;

pub static MIN_Q2_BITS: u64 = 14;
pub static Q2_VALUES: [u64; 37] = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    12289,
    12289,
    61441,
    65537,
    65537,
    520193,
    786433,
    786433,
    3604481,
    7340033,
    16515073,
    33292289,
    67043329,
    132120577,
    268369921,
    469762049,
    1073479681,
    2013265921,
    4293918721,
    8588886017,
    17175674881,
    34359214081,
    68718428161,
];

#[derive(Debug, PartialEq, Clone)]
pub struct Params {
    pub poly_len: usize,
    pub poly_len_log2: usize,
    pub ntt_tables: Vec<Vec<Vec<u64>>>,
    pub scratch: Vec<u64>,

    pub crt_count: usize,
    pub barrett_cr_0: [u64; MAX_MODULI],
    pub barrett_cr_1: [u64; MAX_MODULI],
    pub barrett_cr_0_modulus: u64,
    pub barrett_cr_1_modulus: u64,
    pub mod0_inv_mod1: u64,
    pub mod1_inv_mod0: u64,
    pub moduli: [u64; MAX_MODULI],
    pub modulus: u64,
    pub modulus_log2: u64,
    pub noise_width: f64,

    pub n: usize,
    pub pt_modulus: u64,

    pub db_size_log: usize,
}

impl Params {
    pub fn get_ntt_forward_table(&self, i: usize) -> &[u64] {
        self.ntt_tables[i][0].as_slice()
    }
    pub fn get_ntt_forward_prime_table(&self, i: usize) -> &[u64] {
        self.ntt_tables[i][1].as_slice()
    }
    pub fn get_ntt_inverse_table(&self, i: usize) -> &[u64] {
        self.ntt_tables[i][2].as_slice()
    }
    pub fn get_ntt_inverse_prime_table(&self, i: usize) -> &[u64] {
        self.ntt_tables[i][3].as_slice()
    }

    pub fn crt_compose_1(&self, x: u64) -> u64 {
        assert_eq!(self.crt_count, 1);
        x
    }

    pub fn crt_compose_2(&self, x: u64, y: u64) -> u64 {
        assert_eq!(self.crt_count, 2);
        let mut val = (x as u128) * (self.mod1_inv_mod0 as u128);
        val += (y as u128) * (self.mod0_inv_mod1 as u128);
        barrett_reduction_u128(self, val)
    }

    pub fn crt_compose(&self, a: &[u64], idx: usize) -> u64 {
        if self.crt_count == 1 {
            self.crt_compose_1(a[idx])
        } else {
            self.crt_compose_2(a[idx], a[idx + self.poly_len])
        }
    }

    pub fn init(
        poly_len: usize,
        moduli: &[u64],
        noise_width: f64,
        n: usize,
        pt_modulus: u64,
        db_size_log: usize,
    ) -> Self {
        let poly_len_log2 = log2(poly_len as u64) as usize;
        let crt_count = moduli.len();
        assert!(crt_count <= MAX_MODULI);
        let mut moduli_array = [0; MAX_MODULI];
        for i in 0..crt_count {
            moduli_array[i] = moduli[i];
        }
        let ntt_tables = if crt_count > 1 {
            build_ntt_tables(poly_len, moduli, None)
        } else {
            build_ntt_tables_alt(poly_len, moduli, None)
        };
        let scratch = vec![0u64; crt_count * poly_len];
        let mut modulus = 1;
        for m in moduli {
            modulus *= m;
        }
        let modulus_log2 = log2_ceil(modulus);
        let (barrett_cr_0, barrett_cr_1) = get_barrett(moduli);
        let (barrett_cr_0_modulus, barrett_cr_1_modulus) = get_barrett_crs(modulus);
        let mut mod0_inv_mod1 = 0;
        let mut mod1_inv_mod0 = 0;
        if crt_count == 2 {
            mod0_inv_mod1 = moduli[0] * invert_uint_mod(moduli[0], moduli[1]).unwrap();
            mod1_inv_mod0 = moduli[1] * invert_uint_mod(moduli[1], moduli[0]).unwrap();
        }
        Self {
            poly_len,
            poly_len_log2,
            ntt_tables,
            scratch,
            crt_count,
            barrett_cr_0,
            barrett_cr_1,
            barrett_cr_0_modulus,
            barrett_cr_1_modulus,
            mod0_inv_mod1,
            mod1_inv_mod0,
            moduli: moduli_array,
            modulus,
            modulus_log2,
            noise_width,
            n,
            pt_modulus,
            db_size_log,
        }
    }
}

