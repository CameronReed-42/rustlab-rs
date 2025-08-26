//! Numerical differentiation examples using rustlab-numerical
//! 
//! Demonstrates finite difference methods with mathematical rigor

use rustlab_numerical::differentiation::*;
use rustlab_numerical::Result;
use num_complex::Complex64;
use std::f64::consts::{PI, E};

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║   Numerical Differentiation with RustLab-Numerical    ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    // Example 1: Basic finite differences
    example_basic_differences()?;
    
    // Example 2: Complex step differentiation
    example_complex_step()?;
    
    // Example 3: Richardson extrapolation
    example_richardson_extrapolation()?;
    
    // Example 4: Higher order derivatives
    example_higher_derivatives()?;

    Ok(())
}

fn example_basic_differences() -> Result<()> {
    println!("1. Basic Finite Differences");
    println!("═══════════════════════════\n");
    
    // Test function: f(x) = x * exp(x) * sin(x)
    let f = |x: f64| x * x.exp() * x.sin();
    let df = |x: f64| x.exp() * (x.sin() + x * x.sin() + x * x.cos());
    
    let x0 = 1.0;
    let exact = df(x0);
    
    println!("Function: f(x) = x × eˣ × sin(x)");
    println!("Point: x₀ = {}", x0);
    println!("Exact derivative: {:.10}", exact);
    
    // Test different step sizes
    println!("\nConvergence study:");
    println!("h           Forward         Central         Backward");
    println!("─────────────────────────────────────────────────────");
    
    for exp in 1..=8 {
        let h = 10.0_f64.powi(-exp as i32);
        
        let forward = forward_diff(f, x0, h, 1)?;
        let central = central_diff(f, x0, h, 2)?;  // Central diff needs order 2, 4, or 6
        let backward = backward_diff(f, x0, h, 1)?;
        
        println!("10⁻{}  {:13.10}  {:13.10}  {:13.10}", 
                exp, forward, central, backward);
        println!("      err: {:.2e}   err: {:.2e}   err: {:.2e}",
                (forward - exact).abs(), 
                (central - exact).abs(),
                (backward - exact).abs());
    }
    
    Ok(())
}

fn example_complex_step() -> Result<()> {
    println!("\n2. Complex Step Differentiation");
    println!("═══════════════════════════════\n");
    
    // Function that's difficult for finite differences
    let f = |x: f64| (x * x.cos()).exp();
    let f_complex = |z: Complex64| (z * z.cos()).exp();
    let df = |x: f64| f(x) * (2.0 * x.cos() - x * x * x.sin());
    
    let x0 = PI / 4.0;
    let exact = df(x0);
    
    println!("Function: f(x) = exp(x² × cos(x))");
    println!("Point: x₀ = π/4");
    println!("Exact derivative: {:.15}", exact);
    
    // Compare methods
    println!("\nMethod comparison:");
    
    let h = 1e-8;
    let central = central_diff(f, x0, h, 1)?;
    let complex = complex_step_diff(f_complex, x0, h)?;
    
    println!("Central difference (h={}): {:.15}", h, central);
    println!("  Relative error: {:.2e}", ((central - exact) / exact).abs());
    
    println!("\nComplex step (h={}): {:.15}", h, complex);
    println!("  Relative error: {:.2e}", ((complex - exact) / exact).abs());
    
    // Show complex step works with tiny h
    let h_tiny = 1e-100;
    let complex_tiny = complex_step_diff(f_complex, x0, h_tiny)?;
    println!("\nComplex step (h=1e-100): {:.15}", complex_tiny);
    println!("  Relative error: {:.2e}", ((complex_tiny - exact) / exact).abs());
    
    Ok(())
}

fn example_richardson_extrapolation() -> Result<()> {
    println!("\n3. Richardson Extrapolation");
    println!("═══════════════════════════\n");
    
    // Test on a polynomial
    let f = |x: f64| x.powi(5) - 3.0 * x.powi(3) + 2.0 * x;
    let df = |x: f64| 5.0 * x.powi(4) - 9.0 * x.powi(2) + 2.0;
    
    let x0 = 1.5;
    let exact = df(x0);
    
    println!("Function: f(x) = x⁵ - 3x³ + 2x");
    println!("Point: x₀ = {}", x0);
    println!("Exact f'(x₀) = {:.15}", exact);
    
    // Richardson extrapolation with different orders
    println!("\nRichardson extrapolation results:");
    
    let h0 = 0.1;
    for n_terms in 1..=5 {
        let method = |func, x, h, _order| central_diff(func, x, h, 1);
        let result = richardson_extrapolation(f, x0, h0, method, n_terms)?;
        println!("  Terms = {}: {:.15} (error: {:.2e})", 
                n_terms, result, (result - exact).abs());
    }
    
    Ok(())
}

fn example_higher_derivatives() -> Result<()> {
    println!("\n4. Higher Order Derivatives");
    println!("═══════════════════════════\n");
    
    // Test function with known derivatives
    let f = |x: f64| x.sin() * x.exp();
    
    let x0 = PI / 6.0;
    let h = 0.01;
    
    println!("Function: f(x) = sin(x) × eˣ");
    println!("Point: x₀ = π/6");
    
    // Compute first and second derivatives 
    println!("\n1st derivative:");
    let numerical1 = central_diff(f, x0, h, 2)?;  // 2nd order central diff
    let exact1 = x0.exp() * (x0.sin() + x0.cos());
    println!("  f'({:.4}) = {:.10}", x0, numerical1);
    println!("  Exact:     {:.10}", exact1);
    println!("  Rel. error: {:.2e}", ((numerical1 - exact1) / exact1).abs());
    
    println!("\n2nd derivative (finite difference of finite difference):");
    let eps = h;
    let f_plus = central_diff(f, x0 + eps, h, 2)?;
    let f_minus = central_diff(f, x0 - eps, h, 2)?;
    let numerical2 = (f_plus - f_minus) / (2.0 * eps);
    let exact2 = x0.exp() * (2.0 * x0.cos());
    println!("  f''({:.4}) = {:.10}", x0, numerical2);
    println!("  Exact:      {:.10}", exact2);
    println!("  Rel. error:  {:.2e}", ((numerical2 - exact2) / exact2).abs());
    
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Differentiation examples completed!                   ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    
    Ok(())
}