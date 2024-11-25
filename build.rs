fn main() {
    println!("cargo:rustc-link-search=native=/home/lyl/Desktop/ntl-11.5.1/lib");
    println!("cargo:rustc-link-lib=ntl");
    println!("cargo:rustc-link-lib=gmp");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-search=native=/home/lyl/Desktop/PIR-experiment/npir_rs/src");
    println!("cargo:rustc-link-lib=invmod");
}
