# NPIR

This is an Rust implementation of the NPIR scheme, a high-throughput NTRU-based single-server PIR scheme, introduiced in "".

## Build
To build and run this code, we need some environmental needs and adjustments.

1. Run `sudo apt-get update && sudo apt-get install -y build-essential` to ensure an up-to-date environment.
2. Our code is under C++ and Rust, so we need to install C++ compiler and Rust compiler. In particular, we will use `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` Install Rust.
3. We need to install `libntl-dev` and `libgmp-dev` to compile the C++ code. In particular, we can run `sudo apt-get install -y libntl-dev libgmp-dev` to install these libraries.
4. We need to adjust the path in `build.rs` and change it to the local path of the current host.

## Run
After building the code, we can run the following command to run the test:
1. If the `libinvmod.so` file does, you need to call `g++ -shared -fPIC -o libinvmod.so invmod.cpp -lntl -lgmp` to generate it.
2. Files can be compiled via `cargo build --release`.
3. The module `libinvmod.so` needs to be imported before running to invoke `export LD_LIBRARY_PATH=/home/lyl/Desktop/PIR-experiment/npir/src:$LD_LIBRARY_PATH`. It is worth noting that there is a path that needs to be modified here.
4. If you want to test packing, invoke `RUSTFLAGS="-C target-cpu=native" cargo test --release -- --nocapture ntrurp`.
5. If you want to test the NPIR scheme, invoke `RUSTFLAGS="-C target-cpu=native" cargo test --release -- --nocapture npir`. You can manually tweak the parameters and test different data sets in the `npir.rs` file `test` module.

## Acknowledgements
We use the [menonsamir/spiral-rs](https://github.com/menonsamir/spiral-rs) library for Spiral as our starting point.