# npir
The paths in build.rs need to be checked.


g++ -shared -fPIC -o libinvmod.so invmod.cpp -lntl -lgmp

cargo build --release

export LD_LIBRARY_PATH=/home/lyl/Desktop/PIR-experiment/NPIR_rust/src:$LD_LIBRARY_PATH

RUSTFLAGS="-C target-cpu=native" cargo test --release -- --nocapture ntrurp

RUSTFLAGS="-C target-cpu=native" cargo test --release -- --nocapture npir