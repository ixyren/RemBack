use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use flate2::write::ZlibEncoder;
use flate2::Compression;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let profile = env::var("PROFILE").unwrap_or_else(|_| "release".to_owned());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let target_root = manifest_dir.join("..").join("..").join("target");

    let candidates = [
        target_root.join(&profile).join("remback.exe"),
        target_root.join("release").join("remback.exe"),
        target_root.join("debug").join("remback.exe"),
    ];

    let found = candidates.iter().find(|p| p.is_file()).cloned();
    let payload_out = out_dir.join("remback.exe.zlib");

    if let Some(exe_path) = found {
        println!("cargo:rerun-if-changed={}", exe_path.display());
        let mut raw = Vec::new();
        File::open(&exe_path)
            .expect("open remback.exe")
            .read_to_end(&mut raw)
            .expect("read remback.exe");

        let out_file = File::create(&payload_out).expect("create remback.exe.zlib");
        let mut encoder = ZlibEncoder::new(out_file, Compression::best());
        encoder.write_all(&raw).expect("compress remback.exe");
        encoder.finish().expect("finish compression");
    } else {
        // Fallback placeholder so `cargo check` succeeds before `remback-app` is built.
        println!(
            "cargo:warning=remback.exe not built yet; embedding empty placeholder in remback-installer"
        );
        let out_file = File::create(&payload_out).expect("create placeholder");
        let mut encoder = ZlibEncoder::new(out_file, Compression::fast());
        encoder.write_all(&[]).expect("write placeholder");
        encoder.finish().expect("finish placeholder");
    }

    println!("cargo:rerun-if-changed=../../assets/remback.ico");
    #[cfg(target_os = "windows")]
    if Path::new("../../assets/remback.ico").is_file() {
        let mut res = winres::WindowsResource::new();
        res.set_icon("../../assets/remback.ico");
        res.set("ProductName", "RemBack Setup");
        res.set("FileDescription", "Instalador oficial de RemBack");
        res.set("CompanyName", "ixyren");
        if let Err(e) = res.compile() {
            println!("cargo:warning=winres compile failed: {e}");
        }
    }
}
