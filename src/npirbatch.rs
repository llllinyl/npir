#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;


use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::{discrete_gaussian::*, aligned_memory::*, ntt::*, ntrupacking::*, poly::*, params::*, number_theory::*, arith::*};
use std::time::Instant;


pub struct BatchNpir<'a> {
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

pub fn multiply_x_inverse<'a>(params: &'a Params, k: usize,input: PolyMatrixRaw<'a>,) -> PolyMatrixRaw<'a> {
    let dimension = params.poly_len;
    let modulus = params.modulus;
    let mut output = PolyMatrixRaw::zero(params, 1, 1);
    for i in 0..dimension {
        let rem = (i + dimension - k) % dimension;
        let data = input.get_poly(0, 0)[i] % modulus;
        output.get_poly_mut(0, 0)[rem] = if i < k { (modulus - data) % modulus } else { data };
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

#[cfg(target_feature = "avx2")]
pub fn db_multiply_query<'a>(params: &'a Params, 
        db: &[AlignedMemory64<u64>], query: &PolyMatrixNTT) -> Vec<PolyMatrixNTT<'a>> {
    let dimension = params.poly_len;
    let crt_count = params.crt_count;
    let mut res = Vec::with_capacity(db.len());
    let rows = query.rows;

    let barrett_consts = unsafe {
        [
            _mm256_set1_epi64x(params.barrett_cr_1[0] as i64),
            _mm256_set1_epi64x(params.barrett_cr_1[1] as i64)
        ]
    };
    let moduli = unsafe {
        [
            _mm256_set1_epi64x(params.moduli[0] as i64),
            _mm256_set1_epi64x(params.moduli[1] as i64)
        ]
    };

    for db_row in db {
        let mut res_poly = PolyMatrixNTT::zero(params, 1, 1);
        let result_poly = res_poly.get_poly_mut(0, 0);

        unsafe {
            for z in (0..dimension * crt_count).step_by(4) {
                _mm256_storeu_si256(
                    result_poly.as_mut_ptr().add(z) as *mut __m256i,
                _mm256_setzero_si256(),
                );
            }
        }

        for query_row in 0..rows {
            let query_poly = query.get_poly(query_row, 0);

            unsafe {
                for z in (0..dimension).step_by(4) {
                    let idx_packed_base = query_row * dimension + z;
                    let packed0 = db_row[idx_packed_base];
                    let packed1 = db_row[idx_packed_base + 1];
                    let packed2 = db_row[idx_packed_base + 2];
                    let packed3 = db_row[idx_packed_base + 3];

                    let db_lo = _mm256_set_epi64x(
                        (packed3 >> 32) as i64,
                        (packed2 >> 32) as i64,
                        (packed1 >> 32) as i64,
                        (packed0 >> 32) as i64,
                    );
                    let db_hi = _mm256_set_epi64x(
                        (packed3 as u32) as i64,
                        (packed2 as u32) as i64,
                        (packed1 as u32) as i64,
                        (packed0 as u32) as i64,
                    );

                    for c in 0..crt_count {
                        let reduce = 1 << (64 - 2 * params.moduli[c].ilog2() as usize - 3);
                        let c_offset = c * dimension;
                        let idx_res = c_offset + z;

                        let query_vec = _mm256_loadu_si256(query_poly.as_ptr().add(idx_res) as *const __m256i);
                        let mut res_vec = _mm256_loadu_si256(result_poly.as_mut_ptr().add(idx_res) as *const __m256i);

                        let db_part = if c == 0 { db_hi } else { db_lo };
                        let product = _mm256_mul_epu32(query_vec, db_part);
                        res_vec = _mm256_add_epi64(res_vec, product);

                        if query_row % reduce == 0 || query_row == rows - 1 {
                            res_vec = avx2_barrett_reduction(res_vec, barrett_consts[c], moduli[c]);
                        }
                        _mm256_storeu_si256(result_poly.as_mut_ptr().add(idx_res) as *mut __m256i, res_vec);
                    }
                }
            }
        }
        res.push(res_poly);
    }
    res
}

#[cfg(not(target_feature = "avx2"))]
pub fn db_multiply_query<'a>(params: &'a Params, 
        db: &[AlignedMemory64<u64>], query: &PolyMatrixNTT) -> Vec<PolyMatrixNTT<'a>> {
    let mut res = Vec::with_capacity(db.len());
    let dimension = params.poly_len;

    for db_row in db {
        let mut res_poly = PolyMatrixNTT::zero(params, 1, 1);
        let result_poly = res_poly.get_poly_mut(0, 0);

        for query_row in 0..query.rows {
            let query_poly = query.get_poly(query_row, 0);

            for z in 0..dimension {
                let idx_packed = query_row * dimension + z;
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

                let idx_res = idx_res + dimension;
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

impl<'a> BatchNpir<'a> {
    pub fn new(ntru_params: &'a Params, pack: bool, packing_number: usize, tpk: usize, tg: usize, tce: usize) -> BatchNpir<'a> {
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

        BatchNpir {
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
    
    pub fn compress_query(&self, index_c: &[usize], batchsize: usize) -> (Vec<PolyMatrixNTT<'_>>, Vec<PolyMatrixNTT<'_>>) {
        let dimension = self.ntru_params.poly_len;
        let modulus = self.ntru_params.modulus;
        let noise_scale = self.n1;
        let ell = self.ell;

        let start_time = Instant::now();
        let mut rotation_cipher = vec![PolyMatrixNTT::zero(self.ntru_params, self.tg, 1); batchsize as usize];
        
        for idx in 0..batchsize {
            let mut temp_poly = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
            let col_index = index_c[idx] % dimension;
            let rotation_value = if col_index == 0 { noise_scale } else { modulus - noise_scale };
            let rotation_offset = if col_index == 0 { col_index } else { dimension - col_index };
            
            let mut bg_scale = 1u64;
            for i in 0..self.tg {
                let mut rotated_ntt = self.ntrurp.encrypt(0, noise_scale.try_into().unwrap());
                
                temp_poly.get_poly_mut(0, 0)[rotation_offset] = multiply_uint_mod(
                    rotation_value, bg_scale, modulus
                );
                let temp_ntt = to_ntt_alloc(&temp_poly);
                fast_add_into(&mut rotated_ntt, &temp_ntt);

                rotation_cipher[idx].get_poly_mut(i, 0).copy_from_slice(rotated_ntt.get_poly(0, 0));
                
                bg_scale *= (1 << self.bg) as u64;
            }
        }
        let iter = (batchsize * ell + dimension - 1) / dimension;
        let mut column_cipher = vec![PolyMatrixNTT::zero(self.ntru_params, 1, 1); iter as usize];
        let num = dimension / ell;

        for time in 0..iter {
            let mut tem = 0;
            let mut column_poly = PolyMatrixRaw::zero(self.ntru_params, 1, 1);
            'outer: for idx in 0..num {
                tem = (idx + 1) * ell;
                if time * num + idx >= batchsize {
                    tem = idx * ell;
                    break 'outer;
                }
                let index = time * num + idx;
                let row_index = index_c[index] / dimension;
                column_poly.get_poly_mut(0, 0)[idx * ell + row_index] = self.ntrurp.delta_q();
            }
            let invert = invert_uint_mod(tem.next_power_of_two() as u64, modulus).unwrap();
            column_cipher[time].copy_into(&self.ntrurp.encryptpolywoscale(
                column_poly.clone(), 
                invert.try_into().unwrap()), 0, 0
            );
        }
        println!("Compressed query generation time: {} μs", start_time.elapsed().as_micros());

        (column_cipher, rotation_cipher)
    }

    pub fn uncompress(&self, column_cipher: &[PolyMatrixNTT<'a>], batchsize: usize) -> Vec<PolyMatrixNTT<'a>> {
        let degree = self.ntru_params.poly_len_log2 as usize;
        let dimension = self.ntru_params.poly_len;
        let ell = self.ell;
        let iter = (batchsize * ell + dimension - 1) / dimension;
        let number = ell * batchsize;

        let mut ciphertexts = vec![PolyMatrixNTT::zero(self.ntru_params, 1, 1); number as usize];
        
        for time in 0..iter {
            let offset = time * dimension;
            ciphertexts[offset] = column_cipher[time].clone();
    
            let recursion_depth = if (number - offset) < dimension 
                {(number - offset).next_power_of_two().ilog2() as usize}
                else {degree};
            
            for level in 0..recursion_depth {
                let automorph_factor = (dimension >> level) + 1;
                let level_size = 1 << level;
                
                for idx in 0..level_size {
                    let current_ct = from_ntt_alloc(&ciphertexts[offset + idx]);
                    let negated_ct = multiply_x_inverse(&self.ntru_params, level_size, current_ct);
                    let mut negated_ct_ntt = to_ntt_alloc(&negated_ct);
                    
                    let automorphed_ct = homomorphic_automorph(
                        self.ntru_params,
                        automorph_factor,
                        &ciphertexts[offset + idx],
                        &self.cek[degree - level - 1],
                        self.bce,
                        self.tce,
                    );
                    fast_add_into(&mut ciphertexts[offset + idx], &automorphed_ct);
                    
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
                    ciphertexts[offset + target_idx] = negated_ct_ntt.clone();
                }
            }
        }
 
        ciphertexts
    }

    pub fn queryrecovery(&self, rotation_cipher: &[PolyMatrixNTT<'_>], 
    uncompressed_ciphertexts: &[PolyMatrixNTT<'_>], batchsize: usize) -> Vec<PolyMatrixNTT<'_>> {
        let mut recovered_query = vec![PolyMatrixNTT::zero(self.ntru_params, self.ell, 1); batchsize as usize];
        let ell = self.ell;

        for idx in 0..batchsize {
            let mut res = PolyMatrixNTT::zero(self.ntru_params, 1, 1);
            for i in 0..ell {
                let current_ct_ntt = uncompressed_ciphertexts[idx * ell + i].clone();
                let current_ct = from_ntt_alloc(&current_ct_ntt);

                let decomposed_gadget = gadgetrp(
                    &self.ntru_params,
                    current_ct,
                    self.bg,
                    self.tg,
                );
                let decomposed_gadget_ntt = to_ntt_alloc(&decomposed_gadget);

                multiply(&mut res, &decomposed_gadget_ntt, &rotation_cipher[idx]);
                recovered_query[idx].get_poly_mut(i, 0).copy_from_slice(res.get_poly(0, 0));
            }            
        }

        recovered_query
    }

    pub fn answercompressed(&self, column_cipher: &[PolyMatrixNTT<'_>], 
    rotation_cipher: &[PolyMatrixNTT<'_>], batchsize: usize) -> Vec<PolyMatrixRaw<'_>> {
        let dimension = self.ntru_params.poly_len;
        let mut total_time = 0;
        let uncompress_time = Instant::now();
        let uncom_cipher = self.uncompress(column_cipher, batchsize);
        let query = self.queryrecovery(rotation_cipher, &uncom_cipher, batchsize);
        //println!("query recovery time: {} μs", uncompress_time.elapsed().as_micros());
        total_time += uncompress_time.elapsed().as_micros();

        let phi = self.phi;
        let mut answer_blocks = Vec::with_capacity(phi * batchsize);
        for idx in 0..batchsize {
            let start_time = Instant::now();
            let db_res = if !self.pack {
                let mut processed_db_vec: Vec<PolyMatrixNTT> = (0..self.drows)
                    .map(|_| PolyMatrixNTT::zero(self.ntru_params, 1, 1))
                    .collect();
                multiply_vec(&mut processed_db_vec, &self.db, &query[idx]);
                processed_db_vec
            } else {
                let db_res = db_multiply_query(self.ntru_params, &self.db_pack, &query[idx]);
                db_res
            };
            //println!("simplePIR processing time: {} μs", start_time.elapsed().as_micros());
            total_time += start_time.elapsed().as_micros();

            for block_idx in 0..phi {
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
                //println!("Batch {} Block {} packing time: {} μs", idx, block_idx, block_start_time.elapsed().as_micros());
                total_time += block_start_time.elapsed().as_micros();
            }
        }
        println!("Answer total time: {} μs", total_time);
        answer_blocks
    }
    
    pub fn recovery(&self, answer_blocks: &[PolyMatrixRaw<'_>], batchsize: usize) -> Vec<PolyMatrixRaw<'_>> {
        let phi = self.phi;
        let mut result = Vec::with_capacity(batchsize * phi);

        let start_time = Instant::now();
        for idx in 0..batchsize {
            for i in 0..phi {
                let target_block = answer_blocks[idx * phi + i].clone();
                let target_block_ntt = to_ntt_alloc(&target_block);
                let decrypted_block = self.ntrurp.decryptrp(target_block_ntt);  
                result.push(decrypted_block);                 
            }
         
        }
        println!("Data recovery time: {} μs", start_time.elapsed().as_micros());

        result
    }
}

pub fn batch_npir_test(databaselog: usize, batchsize: usize) {
    let ntru_params = Params::init(2048, 
        &[23068673, 1004535809], 
        2.05,
        1,
        256, 
        databaselog,);
    println!("Generate the database with size 2^{} ...", ntru_params.db_size_log); 
    let batchnpir = BatchNpir::new(&ntru_params, false, 1, 3, 5, 8);

    let dimension = ntru_params.poly_len;
    let totalcolumn = batchnpir.ell * batchsize;
    let ctnum = (totalcolumn as f64 / dimension as f64).ceil() as usize;
    let modbit = ((ntru_params.modulus as f64).log2() / 8.00 as f64).ceil() as usize;
    let mod0bit = ((ntru_params.moduli[0] as f64).log2() / 8.00 as f64).ceil() as usize;
    let pbsize = if ctnum == 0 { 
        (modbit * dimension * ((totalcolumn as f64).log2().ceil() as usize * batchnpir.tce + ntru_params.poly_len_log2 * batchnpir.ntrurp.tpk)) as f64 / 8192.0 as f64 } 
    else { 
        (modbit * dimension * ntru_params.poly_len_log2 * (batchnpir.tce + batchnpir.ntrurp.tpk)) as f64 / 1024.0 as f64 };
    println!("Public parameters size: {:.2} KB", pbsize);
    println!("Query size: {:.2} KB", (modbit * dimension * (ctnum + batchnpir.tg * batchsize)) as f64 / 1024.0 as f64);
    println!("Response size: {:.2} KB", (mod0bit * dimension * batchnpir.phi * batchsize) as f64 / 1024.0 as f64);
    println!("========================================================================================");
    println!("The database has {} rows and {} cols.", batchnpir.drows, batchnpir.ell); 
    let mut rng = ChaCha20Rng::from_entropy();
    for _t in 0..6 {
        let mut index_c = Vec::with_capacity(batchsize);
        for idx in 0..batchsize{
            index_c.push(rng.gen::<usize>() % (ntru_params.poly_len * batchnpir.ell));
            println!("Query the data at {}-th column ...", index_c[idx]);
        }

        let (column, rotate) = batchnpir.compress_query(&index_c, batchsize);

        println!("Server computes the answer ...");
        let ans = batchnpir.answercompressed(&column, &rotate, batchsize);

        let b = batchnpir.recovery(&ans, batchsize);
        for idx in 0..batchsize {
            for i in 0..1 {
                for j in 0..dimension{
                    assert_eq!(b[idx * 1 + i].get_poly(0, 0)[j] as u16, batchnpir.db_raw.get_poly(i * dimension + j, index_c[idx] / dimension)[index_c[idx] % dimension]);
                }
            }
            println!("Batch {}: Extract correctly!", idx);           
        }
        println!("########################################################################################");
    }
}

pub fn batch_npir_pack_test(databaselog: usize, batchsize: usize) {
    let ntru_params = Params::init(2048, 
        &[23068673, 1004535809], 
        2.05,
        1,
        256, 
        databaselog,);
    println!("Generate the database with size 2^{} ...", ntru_params.db_size_log); 
    let batchnpir = BatchNpir::new(&ntru_params, true, 1, 3, 5, 8);

    let dimension = ntru_params.poly_len;
    let totalcolumn = batchnpir.ell * batchsize;
    let ctnum = (totalcolumn as f64 / dimension as f64).ceil() as usize;
    let modbit = ((ntru_params.modulus as f64).log2() / 8.00 as f64).ceil() as usize;
    let mod0bit = ((ntru_params.moduli[0] as f64).log2() / 8.00 as f64).ceil() as usize;
    let pbsize = if ctnum == 0 { 
        (modbit * dimension * ((totalcolumn as f64).log2().ceil() as usize * batchnpir.tce + ntru_params.poly_len_log2 * batchnpir.ntrurp.tpk)) as f64 / 8192.0 as f64 } 
    else { 
        (modbit * dimension * ntru_params.poly_len_log2 * (batchnpir.tce + batchnpir.ntrurp.tpk)) as f64 / 1024.0 as f64 };
    println!("Public parameters size: {:.2} KB", pbsize);
    println!("Query size: {:.2} KB", (modbit * dimension * (ctnum + batchnpir.tg * batchsize)) as f64 / 1024.0 as f64);
    println!("Response size: {:.2} KB", (mod0bit * dimension * batchnpir.phi * batchsize) as f64 / 1024.0 as f64);
    println!("========================================================================================");
    println!("The database has {} rows and {} cols.", batchnpir.drows, batchnpir.ell); 
    let mut rng = ChaCha20Rng::from_entropy();
    for _t in 0..6 {
        let mut index_c = Vec::with_capacity(batchsize);
        for idx in 0..batchsize{
            index_c.push(rng.gen::<usize>() % (ntru_params.poly_len * batchnpir.ell));
            println!("Query the data at {}-th column ...", index_c[idx]);
        }

        let (column, rotate) = batchnpir.compress_query(&index_c, batchsize);

        println!("Server computes the answer ...");
        let ans = batchnpir.answercompressed(&column, &rotate, batchsize);

        let b = batchnpir.recovery(&ans, batchsize);
        for idx in 0..batchsize {
            for i in 0..1 {
                for j in 0..dimension{
                    assert_eq!(b[idx * 1 + i].get_poly(0, 0)[j] as u16, batchnpir.db_raw.get_poly(i * dimension + j, index_c[idx] / dimension)[index_c[idx] % dimension]);
                }
            }
            println!("Batch {}: Extract correctly!", idx);           
        }
        println!("########################################################################################");
    }
}

pub fn batch_npir_pack_large_test(databaselog: usize, batchsize: usize, phi: usize) {
    let ntru_params = Params::init(2048,
        &[23068673, 1004535809],
        2.05,
        1,
        256,
        databaselog,);
    println!("Generate the database with size 2^{} ...", ntru_params.db_size_log);
    let batchnpir = BatchNpir::new(&ntru_params, true, phi, 3, 5, 8);

    let dimension = ntru_params.poly_len;
    let totalcolumn = batchnpir.ell * batchsize;
    let ctnum = (totalcolumn as f64 / dimension as f64).ceil() as usize;
    let modbit = ((ntru_params.modulus as f64).log2() / 8.00 as f64).ceil() as usize;
    let mod0bit = ((ntru_params.moduli[0] as f64).log2() / 8.00 as f64).ceil() as usize;
    let pbsize = if ctnum == 0 {
        (modbit * dimension * ((totalcolumn as f64).log2().ceil() as usize * batchnpir.tce + ntru_params.poly_len_log2 * batchnpir.ntrurp.tpk)) as f64 / 8192.0 as f64 }
    else {
        (modbit * dimension * ntru_params.poly_len_log2 * (batchnpir.tce + batchnpir.ntrurp.tpk)) as f64 / 1024.0 as f64 };
    println!("Public parameters size: {:.2} KB", pbsize);
    println!("Query size: {:.2} KB", (modbit * dimension * (ctnum + batchnpir.tg * batchsize)) as f64 / 1024.0 as f64);
    println!("Response size: {:.2} KB", (mod0bit * dimension * batchnpir.phi * batchsize) as f64 / 1024.0 as f64);
    println!("========================================================================================");
    println!("The database has {} rows and {} cols.", batchnpir.drows, batchnpir.ell);
    let mut rng = ChaCha20Rng::from_entropy();
    for _t in 0..6 {
        let mut index_c = Vec::with_capacity(batchsize);
        for idx in 0..batchsize{
            index_c.push(rng.gen::<usize>() % (ntru_params.poly_len * batchnpir.ell));
            println!("Query the data at {}-th column ...", index_c[idx]);
        }
        let (column, rotate) = batchnpir.compress_query(&index_c, batchsize);

        println!("Server computes the answer ...");
        let ans = batchnpir.answercompressed(&column, &rotate, batchsize);

        let b = batchnpir.recovery(&ans, batchsize);
        for idx in 0..batchsize {
            for i in 0..phi {
                for j in 0..dimension{
                    assert_eq!(b[idx * phi + i].get_poly(0, 0)[j] as u16, batchnpir.db_raw.get_poly(i * dimension + j, index_c[idx] / dimension)[index_c[idx] % dimension]);
                }
            }
            //println!("Batch {}: Extract correctly!", idx);
        }
        println!("Extract correctly for all batch!");
        println!("########################################################################################");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
  
    #[test]
    //#[ignore]
    fn testbatch_1gb_8() {
        batch_npir_pack_large_test(33, 8, 16);
    }

    #[test]
    //#[ignore]
    fn testbatch_1gb_32() {
        batch_npir_pack_large_test(33, 32, 16);
    }

    #[test]
    //#[ignore]
    fn testbatch_8gb_8() {
        batch_npir_pack_large_test(36, 8, 16);
    }

    #[test]
    //#[ignore]
    fn testbatch_8gb_32() {
        batch_npir_pack_large_test(36, 32, 16);
    }

}
