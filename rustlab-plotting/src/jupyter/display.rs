//! Jupyter notebook display integration for rustlab-plotting
//! 
//! This module provides cross-platform display functionality for Jupyter notebooks using evcxr.
//! It enables plots to be displayed inline in notebook cells on Windows, Linux, and macOS.

use crate::error::Result;
use base64::{engine::general_purpose, Engine as _};

#[cfg(target_os = "windows")]
use std::path::Path;

/// Display trait for Jupyter notebook integration
pub trait JupyterDisplay {
    /// Display the content in a Jupyter notebook cell
    fn display(&self) -> Result<()>;
}

/// Display SVG content in a Jupyter notebook
pub fn display_svg(svg_content: &str) -> Result<()> {
    #[cfg(feature = "evcxr-support")]
    {
        // Use evcxr's proper display mechanism with HTML wrapper
        println!("EVCXR_BEGIN_CONTENT text/html");
        println!("<div style=\"text-align: center; margin: 10px;\">");
        println!("{}", svg_content);
        println!("</div>");
        println!("EVCXR_END_CONTENT");
    }
    
    #[cfg(not(feature = "evcxr-support"))]
    {
        // For non-evcxr environments, try HTML wrapper for better rendering
        println!("%%html");
        println!("<div style=\"text-align: center; margin: 10px;\">");
        println!("{}", svg_content);
        println!("</div>");
    }
    
    Ok(())
}

/// Display SVG as base64 data URL (fallback method)
pub fn display_svg_fallback(svg_content: &str) -> Result<()> {
    let encoded = general_purpose::STANDARD.encode(svg_content.as_bytes());
    let data_url = format!("data:image/svg+xml;base64,{}", encoded);
    
    // Print HTML that browsers/notebooks can render
    println!(
        r#"<img src="{}" style="max-width: 100%; height: auto;" alt="Plot" />"#,
        data_url
    );
    
    Ok(())
}

/// Display PNG content in a Jupyter notebook
pub fn display_png(png_data: &[u8]) -> Result<()> {
    #[cfg(feature = "evcxr-support")]
    {
        let encoded = general_purpose::STANDARD.encode(png_data);
        let html = format!(
            r#"<div style="text-align: center;">
                <img src="data:image/png;base64,{}" style="max-width: 100%; height: auto;" alt="Plot" />
            </div>"#,
            encoded
        );
        
        // Direct output for evcxr Jupyter kernel
        println!("EVCXR_BEGIN_CONTENT text/html");
        println!("{}", html);
        println!("EVCXR_END_CONTENT");
    }
    
    #[cfg(not(feature = "evcxr-support"))]
    {
        let encoded = general_purpose::STANDARD.encode(png_data);
        println!(
            r#"<img src="data:image/png;base64,{}" style="max-width: 100%; height: auto;" alt="Plot" />"#,
            encoded
        );
    }
    
    Ok(())
}

/// Check if we're running in a Jupyter environment (cross-platform)
pub fn is_jupyter_environment() -> bool {
    #[cfg(feature = "evcxr-support")]
    {
        // Check if evcxr runtime is available
        std::env::var("EVCXR_JUPYTER").is_ok() || 
        std::env::var("JUPYTER_KERNEL_SOCKET").is_ok()
    }
    
    #[cfg(not(feature = "evcxr-support"))]
    {
        // Check common Jupyter environment variables (works on all platforms)
        std::env::var("JUPYTER_KERNEL_SOCKET").is_ok() ||
        std::env::var("KERNEL_ID").is_ok() ||
        std::env::var("JPY_SESSION_NAME").is_ok() ||
        std::env::var("JUPYTER_SERVER_ROOT").is_ok() ||
        // Additional Windows-specific checks
        std::env::var("JUPYTER_RUNTIME_DIR").is_ok() ||
        // Check for common notebook server indicators
        std::env::var("JPY_PARENT_PID").is_ok()
    }
}

/// Auto-detect display method and display content appropriately
pub fn auto_display_svg(svg_content: &str) -> Result<()> {
    if is_jupyter_environment() {
        display_svg(svg_content)
    } else {
        // In regular terminal, just indicate the plot was created
        println!("📊 Plot created (use .save() to save to file)");
        Ok(())
    }
}

/// Windows-specific helper to detect Jupyter installation
#[cfg(target_os = "windows")]
pub fn detect_jupyter_windows() -> bool {
    // Check for Jupyter in common Windows installation paths
    let common_paths = [
        r"C:\Python*\Scripts\jupyter.exe",
        r"C:\Users\*\AppData\Local\Programs\Python\Python*\Scripts\jupyter.exe",
        r"C:\ProgramData\Anaconda*\Scripts\jupyter.exe",
        r"C:\Users\*\Anaconda*\Scripts\jupyter.exe",
        r"C:\Users\*\Miniconda*\Scripts\jupyter.exe",
    ];
    
    // Also check if jupyter is in PATH
    if std::process::Command::new("jupyter")
        .arg("--version")
        .output()
        .is_ok()
    {
        return true;
    }
    
    // Check Windows registry for Python installations (simplified)
    std::env::var("JUPYTER_CONFIG_DIR").is_ok() ||
    std::env::var("JUPYTER_DATA_DIR").is_ok()
}

/// macOS-specific helper to detect Jupyter installation
#[cfg(target_os = "macos")]
pub fn detect_jupyter_macos() -> bool {
    // Check common macOS paths
    std::process::Command::new("jupyter")
        .arg("--version")
        .output()
        .is_ok() ||
    std::env::var("JUPYTER_CONFIG_DIR").is_ok()
}

/// Linux-specific helper to detect Jupyter installation
#[cfg(target_os = "linux")]
pub fn detect_jupyter_linux() -> bool {
    // Check if jupyter is available in PATH
    std::process::Command::new("jupyter")
        .arg("--version")
        .output()
        .is_ok() ||
    std::env::var("JUPYTER_CONFIG_DIR").is_ok()
}

/// Platform-agnostic Jupyter environment detection
pub fn detect_jupyter_platform() -> bool {
    #[cfg(target_os = "windows")]
    {
        detect_jupyter_windows()
    }
    
    #[cfg(target_os = "macos")]
    {
        detect_jupyter_macos()
    }
    
    #[cfg(target_os = "linux")]
    {
        detect_jupyter_linux()
    }
    
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        // Fallback for other platforms
        std::process::Command::new("jupyter")
            .arg("--version")
            .output()
            .is_ok()
    }
}

/// Enhanced cross-platform display method that falls back gracefully
pub fn display_with_fallback(svg_content: &str) -> Result<()> {
    // Check if we're in a Jupyter environment
    if is_jupyter_environment() {
        // Try evcxr display first
        #[cfg(feature = "evcxr-support")]
        {
            return display_svg(svg_content);
        }
        
        // Fallback to direct SVG output for Jupyter without evcxr
        #[cfg(not(feature = "evcxr-support"))]
        {
            println!("{}", svg_content);
            return Ok(());
        }
    }
    
    // For non-Jupyter environments, just confirm plot creation
    println!("📊 Plot created successfully!");
    println!("💡 In Jupyter: plots display inline");
    println!("📁 Use .save(\"filename.svg\") to save the plot to file");
    
    Ok(())
}

/// Simple SVG display for evcxr Jupyter kernel
pub fn evcxr_display_svg(svg_content: &str) {
    println!("EVCXR_BEGIN_CONTENT text/html");
    println!("<div style=\"text-align: center; margin: 10px;\">");
    println!("{}", svg_content);
    println!("</div>");
    println!("EVCXR_END_CONTENT");
}

/// Debug function to check environment variables
pub fn debug_jupyter_environment() {
    println!("🔍 Jupyter Environment Debug:");
    println!("EVCXR_JUPYTER: {:?}", std::env::var("EVCXR_JUPYTER"));
    println!("EVCXR: {:?}", std::env::var("EVCXR"));
    println!("JUPYTER_KERNEL_SOCKET: {:?}", std::env::var("JUPYTER_KERNEL_SOCKET"));
    println!("KERNEL_ID: {:?}", std::env::var("KERNEL_ID"));
    println!("KERNEL_NAME: {:?}", std::env::var("KERNEL_NAME"));
    println!("JPY_SESSION_NAME: {:?}", std::env::var("JPY_SESSION_NAME"));
    println!("JUPYTER_SERVER_ROOT: {:?}", std::env::var("JUPYTER_SERVER_ROOT"));
    println!("JUPYTER_RUNTIME_DIR: {:?}", std::env::var("JUPYTER_RUNTIME_DIR"));
    println!("JPY_PARENT_PID: {:?}", std::env::var("JPY_PARENT_PID"));
    println!("is_evcxr_environment(): {}", is_evcxr_environment());
    println!("is_jupyter_environment(): {}", is_jupyter_environment());
    
    // Since you're in evcxr, let's assume any Jupyter environment with Rust is evcxr
    println!("🔧 Assuming evcxr since you can run Rust code in Jupyter");
}

/// Check if we're running in evcxr Jupyter kernel
pub fn is_evcxr_environment() -> bool {
    // Multiple ways to detect evcxr
    std::env::var("EVCXR_JUPYTER").is_ok() ||
    std::env::var("EVCXR").is_ok() ||
    // Check if evcxr crate is available (compile-time check won't work, so try runtime)
    // If we can import evcxr_runtime, we're probably in evcxr
    std::env::var("JUPYTER_KERNEL_SOCKET").is_ok() && 
        (std::env::var("KERNEL_ID").unwrap_or_default().contains("rust") ||
         std::env::var("KERNEL_NAME").unwrap_or_default().contains("rust") ||
         std::env::var("KERNEL_ID").unwrap_or_default().contains("evcxr")) ||
    // Pragmatic fallback: if we're in a Jupyter environment but can't detect specific kernel info,
    // and we're running Rust code, it's most likely evcxr
    (is_jupyter_environment() && 
     std::env::var("KERNEL_ID").is_err() && 
     std::env::var("KERNEL_NAME").is_err())
}

/// Display helper that tries multiple methods
pub fn smart_display_svg(svg_content: &str) -> Result<()> {
    // Method 1: Try evcxr if we detect it - improved detection
    if is_evcxr_environment() {
        evcxr_display_svg(svg_content);
        return Ok(());
    }
    
    // Method 2: For non-evcxr Jupyter environments, save file and show instructions
    if is_jupyter_environment() {
        // Save to a temporary file that can be viewed
        use std::fs::File;
        use std::io::Write;
        
        let filename = format!("plot_{}.svg", std::process::id());
        if let Ok(mut file) = File::create(&filename) {
            let _ = file.write_all(svg_content.as_bytes());
            println!("📊 Plot saved as '{}' - open this file in your browser to view the plot.", filename);
            println!("💡 For inline display, use the evcxr Jupyter kernel: cargo install evcxr_jupyter && evcxr_jupyter --install");
        } else {
            println!("📊 Plot created but couldn't save to file. Use .save(\"filename.svg\") method instead.");
        }
        return Ok(());
    }
    
    // Method 3: Terminal fallback
    println!("📊 Plot generated! Save with .save(\"plot.svg\") to view.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svg_fallback_display() {
        let svg = r#"<svg width="100" height="100"><circle cx="50" cy="50" r="40" fill="red" /></svg>"#;
        assert!(display_svg_fallback(svg).is_ok());
    }

    #[test]
    fn test_environment_detection() {
        // This will depend on the actual environment
        let _is_jupyter = is_jupyter_environment();
    }
}