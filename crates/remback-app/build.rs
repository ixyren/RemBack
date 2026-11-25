fn main() {
    println!("cargo:rerun-if-changed=../../assets/remback.ico");
    #[cfg(target_os = "windows")]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("../../assets/remback.ico");
        res.set("ProductName", "RemBack");
        res.set("FileDescription", "RemBack — GUI para BackgroundMattingV2");
        res.set("CompanyName", "ixyren");
        if let Err(e) = res.compile() {
            println!("cargo:warning=winres compile failed: {e}");
        }
    }
}
