fn main() {
    println!("cargo:rustc-link-search=native={your_path}/ntl-11.5.1/lib");
    println!("cargo:rustc-link-lib=ntl");
    println!("cargo:rustc-link-lib=gmp");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-search=native={your_path}/npir/src");
    println!("cargo:rustc-link-lib=invmod");
}
