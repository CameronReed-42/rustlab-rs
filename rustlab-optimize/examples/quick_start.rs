//! Quick start example demonstrating the math-first API

use rustlab_optimize::prelude::*;

fn main() -> Result<()> {
    println!("🚀 RustLab-Optimize Quick Start Examples\n");

    // ========================================================================
    // 1D OPTIMIZATION - Natural mathematical notation
    // ========================================================================
    println!("1️⃣  1D Optimization: minimize (x - 2)²");
    
    let x_min = minimize_1d(|x| (x - 2.0).powi(2))
        .solve()?;
    
    println!("   Minimum found at x = {:.6}", x_min);
    println!("   Expected: x = 2.000000");
    println!("   Error: {:.2e}\n", (x_min - 2.0).abs());

    // ========================================================================
    // 2D OPTIMIZATION - Rosenbrock function
    // ========================================================================
    println!("2️⃣  2D Optimization: minimize Rosenbrock function");
    
    let rosenbrock = |x: f64, y: f64| (1.0 - x).powi(2) + 100.0 * (y - x*x).powi(2);
    
    let (x, y) = minimize_2d(rosenbrock)
        .from(-1.2, 1.0)  // Classic starting point
        .tolerance(1e-8)
        .solve()?;
    
    println!("   Minimum found at ({:.6}, {:.6})", x, y);
    println!("   Expected: (1.000000, 1.000000)");
    println!("   Error: {:.2e}\n", ((x-1.0).powi(2) + (y-1.0).powi(2)).sqrt());

    // ========================================================================
    // N-D OPTIMIZATION - Sphere function
    // ========================================================================
    println!("3️⃣  N-D Optimization: minimize sphere function Σx_i²");
    
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
    
    let result = minimize(sphere)
        .from(&[5.0, -3.0, 2.0, -1.0])  // Starting point
        .using_bfgs()  // Explicit algorithm choice
        .solve()?;
    
    println!("   Minimum found at: {:?}", result.solution);
    println!("   Objective value: {:.2e}", result.objective_value);
    println!("   Algorithm used: {}", result.algorithm_used.name());
    println!("   Iterations: {}", result.iterations);
    println!("   Success: {}\n", result.success);

    // ========================================================================
    // CURVE FITTING - Exponential decay (showcases automatic LM)
    // ========================================================================
    println!("4️⃣  Curve Fitting: exponential decay y = A*exp(-k*t)");
    
    // Generate synthetic data: y = 10*exp(-0.5*t) + noise
    let t = linspace(0.0, 5.0, 20);
    let y_true: Vec<f64> = t.iter()
        .map(|&ti| 10.0 * (-0.5 * ti).exp())
        .collect();
    
    // Add small amount of noise
    let y_noisy: Vec<f64> = y_true.iter()
        .enumerate()
        .map(|(i, &yi)| yi + 0.1 * (i as f64 * 0.3).sin())
        .collect();
    
    let y_data = vec64(&y_noisy);
    
    // This automatically uses Levenberg-Marquardt!
    let fit = fit_exponential(&t, &y_data)?;
    
    println!("   Fitted model: y = {:.3} * exp(-{:.3}*t)", fit.amplitude, fit.decay_rate);
    println!("   True values:  y = 10.000 * exp(-0.500*t)");
    println!("   Half-life: {:.3} (true: {:.3})", fit.half_life, (2.0_f64).ln() / 0.5);
    println!("   R²: {:.6}", fit.r_squared);
    println!("   💡 Used Levenberg-Marquardt automatically for nonlinear least squares\n");

    // ========================================================================
    // COMPARISON: Algorithm selection
    // ========================================================================
    println!("5️⃣  Algorithm Comparison on same problem");
    
    let difficult_function = |x: &[f64]| {
        // Himmelblau's function: (x² + y - 11)² + (x + y² - 7)²  
        let x = x[0];
        let y = x[1];
        (x*x + y - 11.0).powi(2) + (x + y*y - 7.0).powi(2)
    };
    
    let algorithms = [
        ("Auto-select", None),
        ("BFGS", Some("bfgs")),
        ("Nelder-Mead", Some("nelder_mead")),
        ("Gradient Descent", Some("gradient_descent")),
    ];
    
    for (name, alg) in &algorithms {
        let mut optimizer = minimize(difficult_function)
            .from(&[0.0, 0.0]);
            
        optimizer = match alg {
            Some("bfgs") => optimizer.using_bfgs(),
            Some("nelder_mead") => optimizer.using_nelder_mead(), 
            Some("gradient_descent") => optimizer.using_gradient_descent(),
            _ => optimizer, // Auto-select
        };
        
        match optimizer.solve() {
            Ok(result) => {
                println!("   {}: f = {:.6}, iterations = {}, algorithm = {}", 
                         name, result.objective_value, result.iterations, 
                         result.algorithm_used.name());
            }
            Err(e) => {
                println!("   {}: Failed - {}", name, e);
            }
        }
    }

    println!("\n✅ All examples completed successfully!");
    println!("💡 Notice how the API matches mathematical notation and requires minimal setup");
    
    Ok(())
}

// Helper function to create VectorF64 from slice (would be in prelude)
fn vec64(data: &[f64]) -> VectorF64 {
    rustlab_math::VectorF64::from_slice(data)
}

// Helper function to create linspace (would be in rustlab-math)
fn linspace(start: f64, stop: f64, num: usize) -> VectorF64 {
    let step = (stop - start) / (num - 1) as f64;
    let data: Vec<f64> = (0..num)
        .map(|i| start + i as f64 * step)
        .collect();
    vec64(&data)
}