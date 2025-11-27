use std::process::Command;
use std::env;

fn main() {
    // Set CUDA architecture based on target
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "linux" {
        println!("cargo:rustc-env=CUDA_ARCH=sm_60");

        // Try to find CUDA installation
        if let Ok(cuda_path) = env::var("CUDA_HOME") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        } else {
            // Common CUDA installation paths
            let cuda_paths = ["/usr/local/cuda/lib64", "/opt/cuda/lib64"];
            for path in &cuda_paths {
                if std::path::Path::new(path).exists() {
                    println!("cargo:rustc-link-search=native={}", path);
                    break;
                }
            }
        }
    }

    // Generate version info
    let output = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();

    let git_hash = output.trim();
    println!("cargo:rustc-env=GIT_HASH={}", git_hash);

    // Generate build info
    let build_time = chrono::Utc::now().to_rfc3339();
    println!("cargo:rustc-env=BUILD_TIME={}", build_time);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
