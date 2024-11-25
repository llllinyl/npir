use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::{ntrurp::*, poly::*, params::*, number_theory::*, arith::*};
use std::time::Instant;


pub struct Npir<'a> {
    ntru_params: &'a Params,
    ntrurp: NtruRp<'a>,
    db: PolyMatrixNTT<'a>,
    n1: u64,
    drows: usize,
    dcols: usize,
}
pub fn randomdb<'a>(params: &'a Params, db: &mut PolyMatrixNTT<'a>) {
    let mut rng = ChaCha20Rng::from_entropy();
    let dimension = params.poly_len;
    let pt = params.pt_modulus;
    let mut dbraw = PolyMatrixRaw::zero(params, db.rows, db.cols);
    for i in 0..db.rows {
        for j in 0..db.cols {
            for k in 0..dimension{
                let data = rng.gen::<u64>() % pt;
                dbraw.get_poly_mut(i, j)[k] = data;
            }
        }
    }
    *db = to_ntt_alloc(&dbraw);
}

impl<'a> Npir<'a> {
    pub fn new(ntru_params: &'a Params) -> Npir<'a> {
        let ntrurp = NtruRp::new(ntru_params);
        let pt = ntru_params.pt_modulus as f64;
        let log_p = pt.log2() as usize;
        let cols = ntru_params.db_size_log - log_p - 2 * ntru_params.poly_len_log2;
        let dcols = 1 << cols as usize;
        let drows = ntru_params.poly_len;
        let mut db = PolyMatrixNTT::zero(ntru_params, ntru_params.poly_len, dcols);
        randomdb(ntru_params, &mut db);
        let n1 = invert_uint_mod(ntru_params.poly_len as u64, ntru_params.modulus).unwrap();

        Npir {
            ntru_params, ntrurp, db, n1, drows, dcols,
        }
    }

    pub fn query(&self, index_c: usize) -> PolyMatrixNTT<'_> {
        let dimension = self.ntru_params.poly_len;
        let alpha = index_c / dimension;
        let beta = index_c % dimension;
        let mut query_raw = PolyMatrixRaw::zero(self.ntru_params, self.dcols, 1);
        let delta = self.ntrurp.delta_q();
        let modulus_delta = self.ntru_params.modulus - delta;

        for i in 0..self.dcols {
            let ct = if i == alpha {
                let mut pt = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
                let val = if beta == 0 { delta } else { modulus_delta };
                let beta = if beta == 0 { beta } else { dimension - beta };
                pt.get_poly_mut(0, 0)[beta] = val;
                self.ntrurp.encryptpoly(pt)
            } else {
                self.ntrurp.encrypt(0)
            };
            query_raw.get_poly_mut(i, 0).copy_from_slice(ct.get_poly(0, 0));
        }
        let query = to_ntt_alloc(&query_raw);
        query
    }

    pub fn answer(&self, query: PolyMatrixNTT<'_>) -> PolyMatrixRaw<'_> {
        // let start1 = Instant::now();
        let modulus = self.ntru_params.modulus;
        let dimension = self.ntru_params.poly_len;
        let mut db_processed = PolyMatrixNTT::zero(&self.ntru_params, dimension, 1);
        let n1 = self.n1;

        multiply(&mut db_processed, &self.db, &query);
        let mut db_raw = from_ntt_alloc(&db_processed);
        // let duration1 = start1.elapsed();
        // let micros1 = duration1.as_micros();
        
        // let start2 = Instant::now();
        for i in 0..dimension {
            for j in 0..dimension {
                let data = db_raw.get_poly(i, 0)[j] as u64;
                db_raw.get_poly_mut(i, 0)[j] = multiply_uint_mod(data, n1, modulus);
            }
        }
        // let duration2 = start2.elapsed();
        // let micros2 = duration2.as_micros();
        
        // let start3 = Instant::now();
        let rho = self.ntrurp.ringpack(self.ntru_params.poly_len_log2, 0, &db_raw); 
        let ans = self.ntrurp.modreduction(rho);
        // let duration3 = start3.elapsed();
        // let micros3 = duration3.as_micros();
        // let total = (micros1 + micros2 + micros3) as f64;
        // println!("Database-query multiplication costs time: {} %", micros1 as f64/ total);
        // println!("Factorial multiplication costs time: {} %", micros2 as f64/ total);
        // println!("Packing and modreduction cost time: {} %", micros3 as f64/ total);
        ans
    }

    pub fn recovery(&self, index_r: usize, ans: PolyMatrixRaw<'_>) -> u64 {
        let b = self.ntrurp.decryptrp(ans);
        return b.get_poly(0, 0)[index_r]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    #[test]
    fn pir_correctness() {
        let ntru_params = Params::init(2048, 
            &[65537, 1004535809], 
            2.05, 
            1,
            512, 
            3,
            33,);
        println!("Generate the database with size 2^{} ...", ntru_params.db_size_log); 
        let npir = Npir::new(&ntru_params);
        println!("The database has {} rows and {} cols.", ntru_params.poly_len, npir.dcols); 
        let dimension = ntru_params.poly_len;
        let mut rng = ChaCha20Rng::from_entropy();
        let mut micro_total = 0;
        for _t in 0..5 {
            let index_r = rng.gen::<usize>() % ntru_params.poly_len;
            let index_c = rng.gen::<usize>() % (ntru_params.poly_len * npir.dcols);
            println!("Query the data at {}, {} ...", index_r, index_c);
            let query = npir.query(index_c);

            println!("Server computes the answer ...");
            let start1 = Instant::now();
            let ans = npir.answer(query);
            let duration1 = start1.elapsed();
            let micros1 = duration1.as_micros();
            micro_total += micros1;
            println!("Server time: {} microseconds", micros1);

            let b = npir.recovery(index_r, ans);
            let db_raw = from_ntt_alloc(&npir.db);
            assert_eq!(b, db_raw.get_poly(index_r, index_c / dimension)[index_c % dimension]);
            println!("Extract the data {} from the database!", b);
        }
        println!("Server ave time: {} microseconds", micro_total / 20);
    }
}