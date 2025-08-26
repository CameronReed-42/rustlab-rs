//! Root finding examples using rustlab-numerical
//! 
//! Demonstrates various methods for finding zeros of functions

use rustlab_numerical::roots::*;
use rustlab_numerical::Result;
use std::f64::consts::{PI, E};

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║      Root Finding with RustLab-Numerical              ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    // Example 1: Bracketing methods
    example_bracketing_methods()?;
    
    // Example 2: Newton-Raphson method
    example_newton_raphson()?;
    
    // Example 3: Hybrid methods
    example_hybrid_methods()?;
    
    // Example 4: Multiple roots
    example_multiple_roots()?;

    Ok(())
}

fn example_bracketing_methods() -> Result<()> {
    println!("1. Bracketing Methods");
    println!("════════════════════\n");
    
    // Find root of f(x) = x³ - 2x - 5
    let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
    
    println!("Finding root of f(x) = x³ - 2x - 5");
    println!("Known to have a root near x = 2.0945...");
    
    let a = 2.0;
    let b = 3.0;
    let tol = 1e-10;
    
    // Bisection method
    let bisect_result = bisection(f, a, b, tol, 100)?;
    println!("\nBisection method:");
    println!("  Root: {:.12}", bisect_result.root);
    println!("  f(root): {:.2e}", f(bisect_result.root));
    println!("  Iterations: {}", bisect_result.iterations);
    
    // Illinois method
    let illinois_result = illinois(f, a, b, tol, 100)?;
    println!("\nIllinois method:");
    println!("  Root: {:.12}", illinois_result.root);
    println!("  f(root): {:.2e}", f(illinois_result.root));
    println!("  Iterations: {}", illinois_result.iterations);
    
    // Ridders' method
    let ridders_result = ridders(f, a, b, tol, 100)?;
    println!("\nRidders' method:");
    println!("  Root: {:.12}", ridders_result.root);
    println!("  f(root): {:.2e}", f(ridders_result.root));
    println!("  Iterations: {}", ridders_result.iterations);
    
    Ok(())
}

fn example_newton_raphson() -> Result<()> {
    println!("\n2. Newton-Raphson Method");
    println!("════════════════════════\n");
    
    // Find root of f(x) = cos(x) - x
    let f = |x: f64| x.cos() - x;
    let df = |x: f64| -x.sin() - 1.0;
    
    println!("Finding root of f(x) = cos(x) - x");
    println!("(Transcendental equation)");
    
    let x0 = 0.5;  // Initial guess
    let tol = 1e-12;
    
    let newton_result = newton_raphson(f, df, x0, tol, 100)?;
    
    println!("\nStarting from x₀ = {}", x0);
    println!("Root: {:.15}", newton_result.root);
    println!("f(root): {:.2e}", f(newton_result.root));
    println!("Iterations: {}", newton_result.iterations);
    
    // Compare convergence rates
    println!("\nConvergence comparison (iterations to reach tolerance {}):", tol);
    
    // Try different methods with same initial bracket
    let a = 0.0;
    let b = 1.0;
    
    // Count iterations (simplified - actual implementation may vary)
    println!("  Bisection: ~{} iterations", ((b - a) / tol).log2().ceil() as i32);
    println!("  Newton-Raphson: typically 4-6 iterations (quadratic convergence)");
    
    Ok(())
}

fn example_hybrid_methods() -> Result<()> {
    println!("\n3. Hybrid Methods");
    println!("═════════════════\n");
    
    // Test on a difficult function
    let f = |x: f64| x.exp() - 3.0 * x.powi(2);
    let df = |x: f64| x.exp() - 6.0 * x;
    
    println!("Finding roots of f(x) = eˣ - 3x²");
    println!("This function has three roots");
    
    // First root near -0.5
    println!("\nFirst root (near x = -0.5):");
    let root1_result = brent(f, -1.0, 0.0, 1e-10, 100)?;
    println!("  Brent's method: {:.12}", root1_result.root);
    println!("  f(root): {:.2e}", f(root1_result.root));
    
    // Second root near 0.9
    println!("\nSecond root (near x = 0.9):");
    let root2_result = brent(f, 0.5, 1.5, 1e-10, 100)?;
    println!("  Brent's method: {:.12}", root2_result.root);
    println!("  f(root): {:.2e}", f(root2_result.root));
    
    // Third root near 3.7
    println!("\nThird root (near x = 3.7):");
    let root3_result = brent(f, 3.0, 4.0, 1e-10, 100)?;
    println!("  Brent's method: {:.12}", root3_result.root);
    println!("  f(root): {:.2e}", f(root3_result.root));
    
    // Secant method for comparison
    println!("\nSecant method for the third root:");
    let secant_result = secant(f, 3.5, 3.8, 1e-10, 100)?;
    println!("  Root: {:.12}", secant_result.root);
    println!("  Difference from Brent: {:.2e}", (secant_result.root - root3_result.root).abs());
    
    Ok(())
}

fn example_multiple_roots() -> Result<()> {
    println!("\n4. Finding Multiple Roots");
    println!("═════════════════════════\n");
    
    // Polynomial with known roots
    let f = |x: f64| (x - 1.0) * (x - 2.0) * (x - 3.0) * (x + 1.0);
    
    println!("Finding roots of f(x) = (x-1)(x-2)(x-3)(x+1)");
    println!("Known roots: -1, 1, 2, 3");
    
    // Search intervals
    let intervals = [(-2.0, 0.0), (0.0, 1.5), (1.5, 2.5), (2.5, 3.5)];
    let tol = 1e-12;
    
    println!("\nFound roots:");
    for (i, (a, b)) in intervals.iter().enumerate() {
        let root_result = brent(f, *a, *b, tol, 100)?;
        println!("  Root {}: {:.15} (f(x) = {:.2e})", i + 1, root_result.root, f(root_result.root));
    }
    
    // Demonstrate behavior near multiple roots
    println!("\nTesting function with double root:");
    let g = |x: f64| (x - 2.0).powi(2) * (x + 1.0);
    let dg = |x: f64| 2.0 * (x - 2.0) * (x + 1.0) + (x - 2.0).powi(2);
    
    println!("g(x) = (x-2)²(x+1) has a double root at x=2");
    
    // Newton's method struggles with multiple roots
    let newton_result = newton_raphson(g, dg, 2.1, 1e-6, 100)?;
    println!("  Newton from x=2.1: {:.10} (slower convergence)", newton_result.root);
    
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Root finding examples completed!                      ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    
    Ok(())
}