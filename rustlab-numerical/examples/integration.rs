//! Numerical integration examples using rustlab-numerical
//! 
//! Demonstrates various quadrature methods with a math-first approach

use rustlab_numerical::integration::*;
use rustlab_numerical::Result;
use std::f64::consts::PI;

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║   Numerical Integration with RustLab-Numerical        ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    // Example 1: Basic quadrature methods
    example_basic_quadrature()?;
    
    // Example 2: Adaptive integration
    example_adaptive_integration()?;
    
    // Example 3: Oscillatory integrals
    example_oscillatory_integrals()?;
    
    // Example 4: Improper integrals
    example_improper_integrals()?;

    Ok(())
}

fn example_basic_quadrature() -> Result<()> {
    println!("1. Basic Quadrature Methods");
    println!("═══════════════════════════\n");
    
    // Integrate f(x) = x³ - 2x² + x + 1 on [0, 3]
    let f = |x: f64| x.powi(3) - 2.0 * x.powi(2) + x + 1.0;
    let a = 0.0;
    let b = 3.0;
    let exact = 13.5; // Analytical result
    
    println!("Integrating f(x) = x³ - 2x² + x + 1 on [0, 3]");
    println!("Exact value: {:.10}", exact);
    
    // Test with different number of intervals (using multiples of 6 for all methods)
    for n in [12, 24, 48, 96] {
        let trap = trapz(f, a, b, n)?;
        let simp = simpson(f, a, b, n)?;
        let simp38 = simpson38(f, a, b, n)?;
        
        println!("\nWith {} intervals:", n);
        println!("  Trapezoidal:  {:.10} (error: {:.2e})", trap, (trap - exact).abs());
        println!("  Simpson 1/3:  {:.10} (error: {:.2e})", simp, (simp - exact).abs());
        println!("  Simpson 3/8:  {:.10} (error: {:.2e})", simp38, (simp38 - exact).abs());
    }
    
    Ok(())
}

fn example_adaptive_integration() -> Result<()> {
    println!("\n2. Adaptive Integration");
    println!("══════════════════════\n");
    
    // Integrate a function with varying behavior
    let f = |x: f64| {
        if x < 1.0 {
            x.sqrt()
        } else {
            2.0 - x.ln()
        }
    };
    
    println!("Integrating piecewise function:");
    println!("  f(x) = √x        for x < 1");
    println!("  f(x) = 2 - ln(x) for x ≥ 1");
    println!("on [0, 2]");
    
    // Romberg integration with different tolerances
    for tol in [1e-4, 1e-6, 1e-8, 1e-10] {
        let result = romberg(f, 0.0, 2.0, tol, 10)?;
        println!("\nTolerance {:.0e}: {:.12}", tol, result);
    }
    
    // Analytical result for comparison
    let exact = 2.0/3.0 + 2.0 - 2.0 * 2f64.ln() + 1.0;
    println!("\nExact value: {:.12}", exact);
    
    Ok(())
}

fn example_oscillatory_integrals() -> Result<()> {
    println!("\n3. Oscillatory Integrals");
    println!("════════════════════════\n");
    
    // Integrate oscillatory functions
    println!("Integrating ∫₀^π sin(10x) × exp(-x/2) dx");
    
    let f = |x: f64| (10.0 * x).sin() * (-x / 2.0).exp();
    
    // Analytical solution using integration by parts
    let exact = 400.0 / 401.0 * (1.0 - (-PI / 2.0).exp());
    
    println!("Exact value: {:.10}", exact);
    
    // Test different methods with increasing subdivisions
    println!("\nConvergence study:");
    for n in [50, 100, 200, 400, 800] {
        let trap = trapz(f, 0.0, PI, n)?;
        let simp = simpson(f, 0.0, PI, n)?;
        
        println!("\nn = {}:", n);
        println!("  Trapezoidal: {:.10} (rel. error: {:.2e})", 
                trap, ((trap - exact) / exact).abs());
        println!("  Simpson:     {:.10} (rel. error: {:.2e})", 
                simp, ((simp - exact) / exact).abs());
    }
    
    Ok(())
}

fn example_improper_integrals() -> Result<()> {
    println!("\n4. Improper Integrals");
    println!("════════════════════\n");
    
    // Example 1: Integral with singularity at endpoint
    println!("Example 1: ∫₀¹ 1/√x dx = 2");
    
    // Transform to remove singularity: x = t²
    let f_transformed = |t: f64| 2.0; // After substitution
    
    let result = simpson(f_transformed, 0.0, 1.0, 100)?;
    println!("  Result: {:.10} (error: {:.2e})", result, (result - 2.0).abs());
    
    // Example 2: Gaussian integral approximation
    println!("\nExample 2: ∫₋ₐᵃ exp(-x²) dx ≈ √π for large a");
    
    let gaussian = |x: f64| (-x * x).exp();
    
    for a in [3.0, 4.0, 5.0, 6.0] {
        let result = romberg(gaussian, -a, a, 1e-10, 15)?;
        let exact = PI.sqrt();
        println!("  a = {}: {:.10} (rel. error: {:.2e})", 
                a, result, ((result - exact) / exact).abs());
    }
    
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Integration examples completed!                       ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    
    Ok(())
}