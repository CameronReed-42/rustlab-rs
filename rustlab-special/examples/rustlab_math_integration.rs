//! Example demonstrating rustlab-special integration with rustlab-math
//! 
//! This example shows how to use special functions with rustlab-math arrays and vectors.
//! 
//! Run with: cargo run --example rustlab_math_integration --features integration

#[cfg(feature = "integration")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustlab_math::{vec64, array64};
    use rustlab_special::integration::*;
    use rustlab_special::integration::convenience::*;
    
    println!("=== RustLab Special Functions Integration Demo ===\n");
    
    // ========== VECTOR OPERATIONS ==========
    println!("=== Vector Operations ===");
    
    let x = vec64![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    println!("Input vector x: {:?}", 
        (0..x.len()).map(|i| x.get(i).unwrap()).collect::<Vec<_>>());
    
    // Bessel functions
    println!("\n--- Bessel Functions ---");
    let j0_vals = x.bessel_j0();
    let j1_vals = x.bessel_j1();
    let y0_vals = x.bessel_y0();
    
    println!("J_0(x):  {:?}", 
        (0..j0_vals.len()).map(|i| format!("{:.6}", j0_vals.get(i).unwrap())).collect::<Vec<_>>());
    println!("J_1(x):  {:?}", 
        (0..j1_vals.len()).map(|i| format!("{:.6}", j1_vals.get(i).unwrap())).collect::<Vec<_>>());
    println!("Y_0(x):  {:?}", 
        (0..y0_vals.len()).map(|i| format!("{:.6}", y0_vals.get(i).unwrap())).collect::<Vec<_>>());
    
    // Modified Bessel functions
    println!("\n--- Modified Bessel Functions ---");
    let i0_vals = x.bessel_i0();
    let k0_vals = x.bessel_k0();
    
    println!("I_0(x):  {:?}", 
        (0..i0_vals.len()).map(|i| format!("{:.6}", i0_vals.get(i).unwrap())).collect::<Vec<_>>());
    println!("K_0(x):  {:?}", 
        (0..k0_vals.len()).map(|i| format!("{:.6}", k0_vals.get(i).unwrap())).collect::<Vec<_>>());
    
    // Error functions
    println!("\n--- Error Functions ---");
    let erf_vals = x.erf();
    let erfc_vals = x.erfc();
    
    println!("erf(x):  {:?}", 
        (0..erf_vals.len()).map(|i| format!("{:.6}", erf_vals.get(i).unwrap())).collect::<Vec<_>>());
    println!("erfc(x): {:?}", 
        (0..erfc_vals.len()).map(|i| format!("{:.6}", erfc_vals.get(i).unwrap())).collect::<Vec<_>>());
    
    // Gamma functions
    println!("\n--- Gamma Functions ---");
    let gamma_vals = x.gamma();
    let lgamma_vals = x.lgamma();
    let digamma_vals = x.digamma();
    
    println!("Γ(x):    {:?}", 
        (0..gamma_vals.len()).map(|i| format!("{:.6}", gamma_vals.get(i).unwrap())).collect::<Vec<_>>());
    println!("ln(Γ(x)):{:?}", 
        (0..lgamma_vals.len()).map(|i| format!("{:.6}", lgamma_vals.get(i).unwrap())).collect::<Vec<_>>());
    println!("ψ(x):    {:?}", 
        (0..digamma_vals.len()).map(|i| format!("{:.6}", digamma_vals.get(i).unwrap())).collect::<Vec<_>>());
    
    // ========== ARRAY OPERATIONS ==========
    println!("\n=== Array Operations ===");
    
    let matrix = array64![
        [1.0, 1.5, 2.0],
        [2.5, 3.0, 3.5]
    ];
    
    println!("Input matrix:");
    println!("{}", matrix);
    
    // Apply Bessel functions to the entire matrix
    let bessel_matrix = matrix.bessel_j0();
    println!("\nJ_0 applied element-wise:");
    println!("{}", bessel_matrix);
    
    let gamma_matrix = matrix.gamma();
    println!("Γ applied element-wise:");
    println!("{}", gamma_matrix);
    
    // ========== STATISTICAL CONVENIENCE FUNCTIONS ==========
    println!("\n=== Statistical Applications ===");
    
    let z_scores = vec64![-2.0, -1.0, 0.0, 1.0, 2.0];
    println!("Z-scores: {:?}", 
        (0..z_scores.len()).map(|i| z_scores.get(i).unwrap()).collect::<Vec<_>>());
    
    // Normal CDF (cumulative distribution function)
    let normal_cdf_vals = normal_cdf(&z_scores);
    println!("Normal CDF Φ(z): {:?}", 
        (0..normal_cdf_vals.len()).map(|i| format!("{:.6}", normal_cdf_vals.get(i).unwrap())).collect::<Vec<_>>());
    
    // Normal PDF (probability density function)
    let normal_pdf_vals = normal_pdf(&z_scores);
    println!("Normal PDF φ(z): {:?}", 
        (0..normal_pdf_vals.len()).map(|i| format!("{:.6}", normal_pdf_vals.get(i).unwrap())).collect::<Vec<_>>());
    
    // Beta function between two vectors
    let a_vals = vec64![1.0, 2.0, 3.0];
    let b_vals = vec64![1.0, 1.0, 2.0];
    let beta_vals = beta_vectors(&a_vals, &b_vals);
    
    println!("\nBeta function B(a,b):");
    for i in 0..a_vals.len() {
        println!("B({:.1}, {:.1}) = {:.6}", 
            a_vals.get(i).unwrap(), 
            b_vals.get(i).unwrap(),
            beta_vals.get(i).unwrap()
        );
    }
    
    // ========== SPHERICAL BESSEL FUNCTIONS ==========
    println!("\n=== Spherical Bessel Functions ===");
    
    let spherical_x = vec64![0.5, 1.0, 2.0, 3.0];
    let spherical_j0 = spherical_x.spherical_bessel_j(0);
    let spherical_j1 = spherical_x.spherical_bessel_j(1);
    
    println!("x:    {:?}", 
        (0..spherical_x.len()).map(|i| spherical_x.get(i).unwrap()).collect::<Vec<_>>());
    println!("j_0(x): {:?}", 
        (0..spherical_j0.len()).map(|i| format!("{:.6}", spherical_j0.get(i).unwrap())).collect::<Vec<_>>());
    println!("j_1(x): {:?}", 
        (0..spherical_j1.len()).map(|i| format!("{:.6}", spherical_j1.get(i).unwrap())).collect::<Vec<_>>());
    
    // Verify spherical Bessel identity: j_0(x) = sin(x)/x
    println!("\nVerifying j_0(x) = sin(x)/x:");
    for i in 0..spherical_x.len() {
        let x_val = spherical_x.get(i).unwrap();
        let j0_computed = spherical_j0.get(i).unwrap();
        let j0_identity = x_val.sin() / x_val;
        println!("x={:.1}: j_0(x)={:.6}, sin(x)/x={:.6}, diff={:.2e}", 
            x_val, j0_computed, j0_identity, (j0_computed - j0_identity).abs());
    }
    
    // ========== PERFORMANCE DEMONSTRATION ==========
    println!("\n=== Performance Demonstration ===");
    
    use std::time::Instant;
    
    // Large vector for performance testing
    let large_x = VectorF64::linspace(0.1, 10.0, 10000);
    
    let start = Instant::now();
    let _large_bessel = large_x.bessel_j0();
    let bessel_time = start.elapsed();
    
    let start = Instant::now();
    let _large_erf = large_x.erf();
    let erf_time = start.elapsed();
    
    let start = Instant::now();
    let _large_gamma = large_x.gamma();
    let gamma_time = start.elapsed();
    
    println!("Performance on 10,000 element vector:");
    println!("  J_0(x): {:?}", bessel_time);
    println!("  erf(x): {:?}", erf_time);
    println!("  Γ(x):   {:?}", gamma_time);
    
    println!("\n=== Integration Demo Complete ===");
    
    Ok(())
}

#[cfg(not(feature = "integration"))]
fn main() {
    println!("This example requires the 'integration' feature to be enabled.");
    println!("Run with: cargo run --example rustlab_math_integration --features integration");
}