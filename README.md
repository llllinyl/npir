# NPIR

This is an Rust implementation of the NPIR scheme, introduiced in "\textsc{Npir}: High-Rate PIR for Databases with Moderate-Size Records".

## Build
To build and run this code, we need some environmental needs and adjustments.

1. Run `sudo apt-get update && sudo apt-get install -y build-essential` to ensure an up-to-date environment.
2. Our code is under C++ and Rust, so we need to install C++ compiler and Rust compiler. In particular, we will use to `sudo apt install g++` install g++, and `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` and `sudo apt install cargo` to install Rust.
3. We need to install `ntl-11.5.1` and `libgmp3-dev` to compile the C++ code. 
    * First, we can run `sudo apt install libgmp3-dev` to install `gmp-6.2.1`. If it fails, go to the official website [gmp-6.2.1](https://gmplib.org) to download `gmp-6.2.1`. Unzip the file and open it. Then enter `./configure`, `make`, `make check` and `sudo make install`.
    + Then we can go to the official website [ntl-11.5.1](https://libntl.org/download.html) to download `ntl-11.5.1`. Unzip the file and open it. Then enter `./configure NTL_GMP_LIP=on CXXFLAGS='-fPIC -O2'`, `make`, `make check` and `sudo make install`.
4. We need to adjust the path in `build.rs` and change it to the local path of the current host.
5. Our implementation requires the environment to support AVX2.

## Running Tests
After building the code, we can run the following command to run the test:
1. If the `libinvmod.so` file does, you need to call `g++ -shared -fPIC -o libinvmod.so invmod.cpp -lntl -lgmp` to generate it.
2. Files can be compiled via `cargo build --release`.
3. The module `libinvmod.so` needs to be imported before running to invoke `export LD_LIBRARY_PATH={your_path}/npir/src:$LD_LIBRARY_PATH`. It is worth noting that there is a path that needs to be modified here.
4. If you want to test packing, invoke ```RUSTFLAGS="-C target-cpu=native" cargo test --release -- --nocapture ntrupacking::tests::[module]```. You can choose between a recursive implementation or a non-recursive implementation in your tests (use module= `test_packing` or `test_for_packing`).
5. If you want to test the NPIR scheme, invoke ```RUSTFLAGS="-C target-cpu=native" cargo test --release -- --nocapture npirstandard::tests::[module]```. You can test different data sets in the `npirstandard.rs` file `test` module for both small and moderate-size records:
      * use module= `test_[x]_2kb` where the database size `x=64mb/256mb/512mb/1gb/2gb/4gb/8gb` for small records;
      + use module= `test_[x]_32kb` where the database size `x=1gb/2gb/4gb/8gb/16gb/32gb` for moderate-size records;
      + use module= `test_[x]_128kb` where the database size `x=1gb/8gb` for large records;
      + use module= `test_8gb_[y]` where the record size `y=2kb/4kb/8kb/16kb/32kb/64kb/128kb` for varying record sizes.
7. If you want to test the batch version, invoke ```RUSTFLAGS="-C target-cpu=native" cargo test --release -- --nocapture npirbatch::tests::[module]```. You can test different data sets in the `npirbatch.rs` file `test` module (use module= `testbatch_[x]gb_[y]` where the database size `x=1/8` and the batch size `y=8/32`).

## Note
During testing, we found that some CPUs occasionally reported invalid memory errors due to the function `pack_db` in files `npirstandard.rs` and `npirbatch.rs`, while others did not. In such cases, simply re-running the test resolves the issue. However, we have not yet determined the reason why some CPUs are affected and others are not.

## Acknowledgements
We use the NTT module of [menonsamir/spiral-rs](https://github.com/menonsamir/spiral-rs) library for Spiral.

## License
This implementation is provided for research and educational purposes only.
Commercial use requires explicit permission from the authors.

## Citation
To cite Qiyana in academic papers, please use the following entry:
```bibtex
@misc{cryptoeprint:2025/2257,
      author = {Yuliang Lin and Baosheng Wang and Yi Wang and Rongmao Chen},
      title = {\textsc{Npir}: High-Rate {PIR} for Databases with Moderate-Size Records},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/2257},
      year = {2025},
      url = {https://eprint.iacr.org/2025/2257}
}
```














