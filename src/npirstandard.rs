use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::{discrete_gaussian::*, aligned_memory::*, ntt::*, ntrupacking::*, poly::*, params::*, number_theory::*, arith::*};
use std::time::Instant;


pub struct Npir<'a> {
    pub ntru_params: &'a Params,
    pub ntrurp: NtruRp<'a>,
    pub db: PolyMatrixNTT<'a>,
    pub pack: bool,
    pub db_pack: Vec<AlignedMemory64<u64>>,
    pub db_raw: PolyMatrixSmall<'a>,
    pub n1: u64,
    pub r1: u64,
    pub drows: usize,
    pub phi: usize,
    pub ell: usize,
    pub tg: usize,
    pub bg: usize,
    pub tce: usize,
    pub bce: usize,
    pub cek: Vec<PolyMatrixNTT<'a>>,
    y_constants: Vec<PolyMatrixNTT<'a>>,
}

pub fn multiply_x_inverse<'a>(params: &'a Params, k: usize, input: PolyMatrixRaw<'a>) -> PolyMatrixRaw<'a> {
    let dimension = params.poly_len;
    let modulus = params.modulus;
    let mut output = PolyMatrixRaw::zero(params, 1, 1);
    let input_poly = input.get_poly(0, 0);
    let output_poly = output.get_poly_mut(0, 0);
    
    for i in 0..dimension {
        let rem = (i + dimension - k) % dimension;
        let data = input_poly[i] % modulus;
        output_poly[rem] = if i < k { modulus - data } else { data };
    }
    output
}

pub fn randomdb<'a>(params: &'a Params,  db_raw: &mut PolyMatrixSmall<'a>) {
    let mut rng = ChaCha20Rng::from_entropy();
    let dimension = params.poly_len;
    let pt = params.pt_modulus;

    for i in 0..db_raw.rows {
        for j in 0..db_raw.cols {
            for k in 0..dimension{
                db_raw.get_poly_mut(i, j)[k] = rng.gen::<u16>().rem_euclid(pt as u16);
            }
        }
    }
    println!("Finish the database!");
    println!("========================================================================================");
}

pub fn pack_db<'a>(params: &'a Params, db_raw: PolyMatrixSmall<'a>) -> Vec<AlignedMemory64<u64>> {
    let mut db_vec = Vec::with_capacity(db_raw.rows);
    let cols = db_raw.cols;
    let dimension = params.poly_len;
    
    for r in 0..db_raw.rows {
        let mut packed_row = AlignedMemory64::<u64>::new(cols * dimension);
        
        for c in 0..cols {
            let pol_src = db_raw.get_poly(r, c);
            let mut pol_ntt: Vec<u64> = vec![0; dimension * 2];
            
            reduce_copy_small(params, &mut pol_ntt, pol_src);
            ntt_forward(params, &mut pol_ntt);

            for i in 0..dimension {
                let idx_packed = c * dimension + i;
                packed_row[idx_packed] = pol_ntt[i] | (pol_ntt[i + dimension] << 32);
            }
        }
        db_vec.push(packed_row);
    }
    db_vec
}

pub fn db_multiply_query<'a>(params: &'a Params, 
        db: &[AlignedMemory64<u64>], query: &PolyMatrixNTT) -> Vec<PolyMatrixNTT<'a>> {
    let mut res = Vec::with_capacity(db.len());
    let poly_len = params.poly_len;

    for db_row in db {
        let mut res_poly = PolyMatrixNTT::zero(params, 1, 1);
        let result_poly = res_poly.get_poly_mut(0, 0);

        for query_row in 0..query.rows {
            let query_poly = query.get_poly(query_row, 0);

            for z in 0..poly_len {
                let idx_packed = query_row * poly_len + z;
                let packed = db_row[idx_packed];
                
                let db_lo = packed as u32 as u64;
                let db_hi = (packed >> 32) as u32 as u64;

                let idx_res = z;
                result_poly[idx_res] = multiply_add_modular(
                    params,
                    query_poly[idx_res],
                    db_lo,
                    result_poly[idx_res],
                    0,
                );

                let idx_res = poly_len + z;
                result_poly[idx_res] = multiply_add_modular(
                    params,
                    query_poly[idx_res],
                    db_hi,
                    result_poly[idx_res],
                    1,
                );
            }
        }
        res.push(res_poly);
    }
    res
}

impl<'a> Npir<'a> {
    pub fn new(ntru_params: &'a Params, pack: bool, packing_number: usize, tpk: usize, tg: usize, tce: usize) -> Npir<'a> {
        let ntrurp = NtruRp::new(ntru_params,tpk);
        let plaintext_modulus = ntru_params.pt_modulus as f64;
        let plaintext_bit = plaintext_modulus.log2() as usize;
        let dimension = ntru_params.poly_len;
        let modulus = ntru_params.modulus;
        let sk_raw = from_ntt_alloc(&ntrurp.sk.clone());
        let sk_inv = ntrurp.sk_inv.clone();
        
        let phi = packing_number;
        let drows = ntru_params.poly_len * phi;
        let ell = (1 << (ntru_params.db_size_log - 3)) 
            / ((ntru_params.poly_len * drows * plaintext_bit) / 8 as usize);
        
        let bg = ((ntru_params.modulus as f64).log2() / tg as f64).ceil() as usize;
        let bce = ((ntru_params.modulus as f64).log2() / tce as f64).ceil() as usize;

        let n1 = invert_uint_mod(ntru_params.poly_len as u64, ntru_params.modulus).unwrap();
        let r1 = invert_uint_mod(ell as u64, ntru_params.modulus).unwrap();
        
        let y_constants = generate_y_constants(&ntru_params);

        let mut cek = Vec::new();
        let mut rng = ChaCha20Rng::from_entropy();
        let dg = DiscreteGaussian::init(ntru_params.noise_width);
        let mut g = PolyMatrixRaw::zero(&ntru_params, 1, 1);
        for i in 0..ntru_params.poly_len_log2 {
            let mut btem = 1u64;
            let mut tem_cek = PolyMatrixNTT::zero(&ntru_params, tce, 1);
            for j in 0..tce {
                let mut tauf = tau(&ntru_params, (1 << (i + 1)) as usize + 1, sk_raw.clone());
  
                for k in 0..dimension{
                    g.get_poly_mut(0, 0)[k] = dg.fast_sample(modulus, &mut rng) as u64;
                    let data = tauf.get_poly(0, 0)[k];
                    tauf.get_poly_mut(0, 0)[k] = multiply_uint_mod(data, btem, modulus);
                }
                
                g = &g + &tauf;
                let g_ntt = to_ntt_alloc(&g);
                tem_cek.copy_into(&(&g_ntt * &sk_inv), j, 0);
                
                btem *= (1 << bce) as u64;
            }
            cek.push(tem_cek);
        }

        println!("Generate and preprocess the database!");
        let mut db_raw = PolyMatrixSmall::zero(ntru_params, drows, ell);
        randomdb(ntru_params, &mut db_raw);
        let init = Instant::now();
        let (db, db_pack) = if !pack {
            let db = to_ntt_alloc_small(&db_raw);
            let db_pack = vec![AlignedMemory64::<u64>::new(1)];
            (db, db_pack)
        } else {
            let db = PolyMatrixNTT::zero(ntru_params, 1, 1);
            let db_pack = pack_db(ntru_params, db_raw.clone());
            (db, db_pack)
        };
        println!("Server prep. time: {} μs", init.elapsed().as_micros());
        println!("========================================================================================");
        
        let modbit = ((ntru_params.modulus as f64).log2() / 8.00 as f64).ceil() as usize;
        let mod0bit = ((ntru_params.moduli[0] as f64).log2() / 8.00 as f64).ceil() as usize;
        println!("Public parameters size: {:.2} KB", (modbit * dimension * (ell.ilog2() as usize * tce + ntru_params.poly_len_log2 * tpk)) as f64 / 1024.0 as f64);
        println!("Query size: {:.2} KB", (modbit * dimension * ((ell as f64 / dimension as f64).ceil() as usize + tg)) as f64 / 1024.0 as f64);
        println!("Response size: {:.2} KB", (mod0bit * dimension * phi) as f64 / 1024.0 as f64);
        println!("========================================================================================");
        
        Npir {
            ntru_params,
            ntrurp,
            db,
            pack,
            db_pack,
            db_raw,
            n1,
            r1,
            drows,
            phi,
            ell,
            tg,
            bg,
            tce,
            bce,
            cek,
            y_constants,
        }
    }
    
    pub fn compress_query(&self, index_c: usize) -> (PolyMatrixNTT<'_>, PolyMatrixNTT<'_>) {
        let dimension = self.ntru_params.poly_len;
        let row_index = index_c / dimension;
        let col_index = index_c % dimension;
        let modulus = self.ntru_params.modulus;
        let noise_scale = self.n1;
        
        let start_time = Instant::now();
        let mut rotation_cipher = PolyMatrixNTT::zero(self.ntru_params, self.tg, 1);
        let mut temp_poly = PolyMatrixRaw::zero(self.ntru_params, 1, 1);

        // let rotation_value = if col_index == 0 { 1 } else { modulus - 1 }; // test correctness of compression
        let rotation_value = if col_index == 0 { noise_scale } else { modulus - noise_scale };
        let rotation_offset = if col_index == 0 { col_index } else { dimension - col_index };
        
        let mut bg_scale = 1u64;
        for i in 0..self.tg {
            // let mut rotated_ntt = self.ntrurp.encrypt(0, 1); // test correctness of compression
            let mut rotated_ntt = self.ntrurp.encrypt(0, noise_scale.try_into().unwrap());
            
            temp_poly.get_poly_mut(0, 0)[rotation_offset] = multiply_uint_mod(rotation_value, bg_scale, modulus);
            let temp_ntt = to_ntt_alloc(&temp_poly);
            fast_add_into(&mut rotated_ntt, &temp_ntt);

            rotation_cipher.get_poly_mut(i, 0).copy_from_slice(rotated_ntt.get_poly(0, 0));
            
            bg_scale *= (1 << self.bg) as u64;
        }
    
        let mut column_poly = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
        column_poly.get_poly_mut(0, 0)[row_index] = self.ntrurp.delta_q();
        let column_cipher = self.ntrurp.encryptpolywoscale(
            column_poly, 
            self.r1.try_into().unwrap()
        );
        println!("Compressed query generation time: {} μs", start_time.elapsed().as_micros());

        (column_cipher, rotation_cipher)
    }

    pub fn uncompress(&self, column_cipher: PolyMatrixNTT<'a>) -> Vec<PolyMatrixNTT<'a>> {
        let recursion_depth = self.ell.ilog2() as usize;
        let degree = self.ntru_params.poly_len_log2;
        let dimension = self.ntru_params.poly_len;
        
        let mut ciphertexts = vec![PolyMatrixNTT::zero(self.ntru_params, 1, 1); self.ell as usize];
        ciphertexts[0] = column_cipher;
    
        for level in 0..recursion_depth {
            let automorph_factor = (dimension >> level) + 1;
            let level_size = 1 << level;
            
            for idx in 0..level_size {
                let current_ct = from_ntt_alloc(&ciphertexts[idx]);
                let negated_ct = multiply_x_inverse(&self.ntru_params, level_size, current_ct);
                let mut negated_ct_ntt = to_ntt_alloc(&negated_ct);
                
                let automorphed_ct = homomorphic_automorph(
                    self.ntru_params,
                    automorph_factor,
                    &ciphertexts[idx],
                    &self.cek[degree - level - 1],
                    self.bce,
                    self.tce,
                );
                fast_add_into(&mut ciphertexts[idx], &automorphed_ct);
                
                let automorphed_neg_ct = homomorphic_automorph(
                    self.ntru_params,
                    automorph_factor,
                    &negated_ct_ntt,
                    &self.cek[degree - level - 1],
                    self.bce,
                    self.tce,
                );
                fast_add_into(&mut negated_ct_ntt, &automorphed_neg_ct);
                
                let target_idx = idx + level_size;
                ciphertexts[target_idx] = negated_ct_ntt.clone();
            }
        }
        ciphertexts
    }

    pub fn queryrecovery(&self, rotation_cipher: PolyMatrixNTT<'_>, uncompressed_ciphertexts: &[PolyMatrixNTT<'_>]) -> PolyMatrixNTT<'_> {
        let mut recovered_query = PolyMatrixNTT::zero(self.ntru_params, self.ell, 1);
        let mut res = PolyMatrixNTT::zero(self.ntru_params, 1, 1);

        for i in 0..self.ell {
            let current_ct_ntt = uncompressed_ciphertexts[i].clone();
            let current_ct = from_ntt_alloc(&current_ct_ntt);

            let decomposed_gadget = gadgetrp(
                &self.ntru_params,
                current_ct,
                self.bg,
                self.tg,
            );
            let decomposed_gadget_ntt = to_ntt_alloc(&decomposed_gadget);

            multiply(&mut res, &decomposed_gadget_ntt, &rotation_cipher);
            recovered_query.get_poly_mut(i, 0).copy_from_slice(res.get_poly(0, 0));
        }

        recovered_query
    }

    pub fn answercompressed(&self, column_cipher: PolyMatrixNTT<'a>, rotation_cipher: PolyMatrixNTT<'_>) -> Vec<PolyMatrixRaw<'_>> {
        let dimension = self.ntru_params.poly_len;
        let mut total_time = 0;
        let uncompress_time = Instant::now();
        let uncom_cipher = self.uncompress(column_cipher);
        let query = self.queryrecovery(rotation_cipher, &uncom_cipher);
        println!("query recovery time: {} μs", uncompress_time.elapsed().as_micros());
        total_time += uncompress_time.elapsed().as_micros();
        
        let start_time = Instant::now();
        let db_res = if !self.pack {
            let mut processed_db_vec: Vec<PolyMatrixNTT> = (0..self.drows)
                .map(|_| PolyMatrixNTT::zero(self.ntru_params, 1, 1))
                .collect();
            multiply_vec(&mut processed_db_vec, &self.db, &query);
            processed_db_vec
        } else {
            let db_res = db_multiply_query(self.ntru_params, &self.db_pack, &query);
            db_res
        };
        println!("simplePIR processing time: {} μs", start_time.elapsed().as_micros());
        total_time += start_time.elapsed().as_micros();

        let mut answer_blocks = Vec::with_capacity(self.phi);
        
        for block_idx in 0..self.phi {
            let block_start_time = Instant::now();
            
            let block_data = &db_res[(block_idx * dimension)..=(block_idx * dimension + dimension - 1)];
    
            let packed_block = self.ntrurp.packing_for(
                self.ntru_params.poly_len_log2,
                &block_data,
                &self.y_constants
            );
            
            let raw_block = from_ntt_alloc(&packed_block);
            let reduced_block = self.ntrurp.modreduction(raw_block);
            
            answer_blocks.push(reduced_block);
            println!("Block {} packing time: {} μs", block_idx, block_start_time.elapsed().as_micros());
            total_time += block_start_time.elapsed().as_micros();
        }
        println!("Answer total time: {} μs", total_time);

        answer_blocks
    }

    pub fn query(&self, index: usize) -> PolyMatrixNTT<'_> {
        let dimension = self.ntru_params.poly_len;
        let row_index = index / dimension;
        let col_index = index % dimension;
        let delta = self.ntrurp.delta_q();
        let modulus_delta = self.ntru_params.modulus - delta;
        let noise_scale = self.n1;
        let mut query_matrix = PolyMatrixNTT::zero(self.ntru_params, self.ell, 1);

        let start_time = Instant::now();
        for i in 0..self.ell {
            let ciphertext = if i == row_index {
                let mut plaintext = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
                let value = if col_index == 0 { delta } else { modulus_delta };
                let rotation_offset = if col_index == 0 { col_index } else { dimension - col_index };
                
                plaintext.get_poly_mut(0, 0)[rotation_offset] = value;
                self.ntrurp.encryptpolywoscale(plaintext, noise_scale.try_into().unwrap())
            } else {
                self.ntrurp.encrypt(0, noise_scale.try_into().unwrap())
            };
            
            query_matrix.get_poly_mut(i, 0).copy_from_slice(ciphertext.get_poly(0, 0));
        }
        println!("Standard query generation time: {} μs", start_time.elapsed().as_micros());
        query_matrix
    }

    pub fn answer(&self, query: PolyMatrixNTT<'_>) -> Vec<PolyMatrixRaw<'_>> {
        let mut answer_blocks = Vec::with_capacity(self.phi);

        let start_time = Instant::now();
        let mut processed_db = PolyMatrixNTT::zero(self.ntru_params, self.drows, 1);
        multiply(&mut processed_db, &self.db, &query);
        println!("simplePIR processing time: {} μs", start_time.elapsed().as_micros());
    
        for block_idx in 0..self.phi {
            let block_start_time = Instant::now();
            
            let mut block_data = Vec::with_capacity(self.ntru_params.poly_len);
            for i in 0..self.ntru_params.poly_len {
                block_data.push(processed_db.submatrix(
                    self.ntru_params.poly_len * block_idx + i, 
                    0, 
                    1, 
                    1)
                );
            }
    
            let packed_block = self.ntrurp.packing_for(
                self.ntru_params.poly_len_log2,
                &block_data,
                &self.y_constants
            );
            
            let raw_block = from_ntt_alloc(&packed_block);
            let reduced_block = self.ntrurp.modreduction(raw_block);
            
            answer_blocks.push(reduced_block);
            println!("Block {} packing time: {} μs", block_idx, block_start_time.elapsed().as_micros());
        }
    
        answer_blocks
    }
    
    pub fn recovery(&self, answer_blocks: &[PolyMatrixRaw<'_>]) -> Vec<PolyMatrixRaw<'_>> {
        let mut data = Vec::with_capacity(self.phi);

        let start_time = Instant::now();
        for i in 0..self.phi{
            let target_block = answer_blocks[i].clone();
            let target_block_ntt = to_ntt_alloc(&target_block);
            let decrypted_block = self.ntrurp.decryptrp(target_block_ntt);
            data.push(decrypted_block);        
        }
        println!("Data recovery time: {} μs", start_time.elapsed().as_micros());
        data
    }
}

/*
pub fn npirfree_test(databaselog: usize) {
    let ntru_params = Params::init(2048, 
        &[23068673, 1004535809], 
        2.05, 
        1,
        256, 
        databaselog,);
    println!("Generate the database with size 2^{} ...", ntru_params.db_size_log); 
    let npir = Npir::new(&ntru_params, 1, 3, 35, 8); // The second input is \phi; the third is t_pk; the fourth is t_g; the fifth is t_ce

    println!("The database has {} rows and {} cols.", npir.drows, npir.ell); 
    let dimension = ntru_params.poly_len;
    let mut rng = ChaCha20Rng::from_entropy();
    let mut micro_total = 0;
    for _t in 0..6 {
        let index_c = rng.gen::<usize>() % (ntru_params.poly_len * npir.ell);
        println!("Query the data at {}-th column ...", index_c);
        let query = npir.query(index_c);

        println!("Server computes the answer ...");
        let start1 = Instant::now();
        let ans: Vec<PolyMatrixRaw> = npir.answer(query);
        let duration1 = start1.elapsed();
        let micros1 = duration1.as_micros();
        if _t != 0 {
            micro_total += micros1;
        }
        println!("Server time: {} μs", micros1);

        let b = npir.recovery(&ans);
        for i in 0..1 {
            for j in 0..dimension{
                assert_eq!(b[i].get_poly(0, 0)[j], npir.db_raw.get_poly(i * dimension + j, index_c / dimension)[index_c % dimension]);
            }
        }
        println!("Extract correctly!");
        println!("########################################################################################");
    }
    println!("Server ave time: {} μs", micro_total / 5);
}
*/

pub fn npir_test(databaselog: usize) {
    let ntru_params = Params::init(2048, 
        &[23068673, 1004535809],
        2.05,
        1,
        256, 
        databaselog,);
    println!("Generate the database with size 2^{} ...", ntru_params.db_size_log); 
    let npir = Npir::new(&ntru_params, false, 1, 3, 5, 8);

    println!("The database has {} rows and {} cols.", npir.drows, npir.ell); 
    let dimension = ntru_params.poly_len;
    let mut rng = ChaCha20Rng::from_entropy();
    let mut micro_total = 0;
    for _t in 0..6 {
        let index_c = rng.gen::<usize>() % (ntru_params.poly_len * npir.ell);
        println!("Query the data at {} column ...", index_c);
        let (column, rotate) = npir.compress_query(index_c);

        println!("Server computes the answer ...");
        let start1 = Instant::now();
        let ans = npir.answercompressed(column, rotate);
        let duration1 = start1.elapsed();
        let micros1 = duration1.as_micros();
        if _t != 0 {
            micro_total += micros1;
        }
        println!("Server time: {} μs", micros1);

        let b = npir.recovery(&ans);
        for i in 0..1 {
            for j in 0..dimension{
                assert_eq!(b[i].get_poly(0, 0)[j], npir.db_raw.get_poly(i * dimension + j, index_c / dimension)[index_c % dimension] as u64);
            }
        }
        println!("Extract correctly!");
        println!("########################################################################################");
    }
    println!("Server ave time: {} μs", micro_total / 5);
}

pub fn npir_large_test(databaselog: usize, phi: usize) {
    let ntru_params = Params::init(2048,
        &[23068673, 1004535809],
        2.05,
        1,
        256,
        databaselog,);
    println!("Generate the database with size 2^{} ...", ntru_params.db_size_log);
    let npir = Npir::new(&ntru_params, false, phi, 3, 5, 8);

    println!("The database has {} rows and {} cols.", npir.drows, npir.ell);
    let dimension = ntru_params.poly_len;
    let mut rng = ChaCha20Rng::from_entropy();
    let mut micro_total = 0;
    for _t in 0..6 {
        let index_c = rng.gen::<usize>() % (ntru_params.poly_len * npir.ell);
        println!("Query the data at {} column ...", index_c);
        let (column, rotate) = npir.compress_query(index_c);


        println!("Server computes the answer ...");
        let start1 = Instant::now();
        let ans = npir.answercompressed(column, rotate);
        let duration1 = start1.elapsed();
        let micros1 = duration1.as_micros();
        if _t != 0 {
            micro_total += micros1;
        }
        println!("Server time: {} μs", micros1);

        let b = npir.recovery(&ans);
        for i in 0..phi {
            for j in 0..dimension{
                assert_eq!(b[i].get_poly(0, 0)[j], npir.db_raw.get_poly(i * dimension + j, index_c / dimension)[index_c % dimension] as u64);
            }
        }
        println!("Extract correctly!");
        println!("########################################################################################");
    }
    println!("Server ave time: {} μs", micro_total / 5);
}

pub fn npir_pack_test(databaselog: usize, phi: usize) {
    let ntru_params = Params::init(2048,
        &[23068673, 1004535809],
        2.05,
        1,
        256,
        databaselog,);
    println!("Generate the database with size 2^{} ...", ntru_params.db_size_log);
    let npir = Npir::new(&ntru_params, true, phi, 3, 5, 8);

    println!("The database has {} rows and {} cols.", npir.drows, npir.ell);
    let dimension = ntru_params.poly_len;
    let mut rng = ChaCha20Rng::from_entropy();
    let mut micro_total = 0;
    for _t in 0..6 {
        let index_c = rng.gen::<usize>() % (ntru_params.poly_len * npir.ell);
        println!("Query the data at {} column ...", index_c);
        let (column, rotate) = npir.compress_query(index_c);


        println!("Server computes the answer ...");
        let start1 = Instant::now();
        let ans = npir.answercompressed(column, rotate);
        let duration1 = start1.elapsed();
        let micros1 = duration1.as_micros();
        if _t != 0 {
            micro_total += micros1;
        }
        println!("Server time: {} μs", micros1);

        let b = npir.recovery(&ans);
        for i in 0..phi {
            for j in 0..dimension{
                assert_eq!(b[i].get_poly(0, 0)[j], npir.db_raw.get_poly(i * dimension + j, index_c / dimension)[index_c % dimension] as u64);
            }
        }
        println!("Extract correctly!");
        println!("########################################################################################");
    }
    println!("Server ave time: {} μs", micro_total / 5);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn npir_64mb_correctness() {
        npir_test(29);
    }

    #[test]
    #[ignore]
    fn npir_128mb_correctness() {
        npir_test(30);
    }

    #[test]
    #[ignore]
    fn npir_256mb_correctness() {
        npir_test(31);
    }

    #[test]
    //#[ignore]
    fn npir_256mb_pack_correctness() {
        npir_pack_test(31,1);
    }

    #[test]
    #[ignore]
    fn npir_512mb_correctness() {
        npir_test(32);
    }

    #[test]
    #[ignore]
    fn npir_1gb_correctness() {
        npir_test(33);
    }

    #[test]
    #[ignore]
    fn npir_2gb_correctness() {
        npir_test(34);
    }

    #[test]
    #[ignore]
    fn npir_4gb_correctness() {
        npir_test(35);
    }

    #[test]
    #[ignore]
    fn npir_8gb_correctness() {
        npir_test(36);
    }

    #[test]
    #[ignore]
    fn npir_1gb_32kb_correctness() {
        npir_large_test(33,16);
    }

    #[test]
    #[ignore]
    fn npir_2gb_32kb_correctness() {
        npir_large_test(34,16);
    }

    #[test]
    #[ignore]
    fn npir_4gb_32kb_correctness() {
        npir_large_test(35,16);
    }

    #[test]
    #[ignore]
    fn npir_8gb_32kb_correctness() {
        npir_large_test(36,16);
    }

    #[test]
    #[ignore]
    fn npir_16gb_32kb_correctness() {
        npir_large_test(37,16);
    }

    #[test]
    #[ignore]
    fn npir_32gb_32kb_correctness() {
        npir_large_test(38,16);
    }
}
