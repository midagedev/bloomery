fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_OXIDE_TARGET");
    println!("cargo:rustc-check-cfg=cfg(bloomery_nvfp4_mma)");
    println!("cargo:rustc-check-cfg=cfg(bloomery_e4m3_cvt)");
    let target = std::env::var("CUDA_OXIDE_TARGET").unwrap_or_default();
    // The crate's `BUILD_TARGET`: absent when the build was given no target.
    if !target.is_empty() {
        println!("cargo:rustc-env=GPU_BUILD_TARGET={target}");
    }
    if matches!(
        target.as_str(),
        "sm_120a" | "sm_120f" | "sm_121a" | "sm_121f"
    ) {
        println!("cargo:rustc-cfg=bloomery_nvfp4_mma");
    }
    // The E4M3 pair convert to f16 (`cvt.rn.f16x2.e4m3x2`) exists from sm_89 on.
    let sm: String = target
        .strip_prefix("sm_")
        .unwrap_or_default()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    if sm.parse::<u32>().is_ok_and(|n| n >= 89) {
        println!("cargo:rustc-cfg=bloomery_e4m3_cvt");
    }
}
