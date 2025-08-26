//! Mathematical interpolation examples using rustlab-numerical
//! 
//! This example demonstrates interpolation with rustlab's math-first approach,
//! showcasing functional programming and natural mathematical expressions.

use rustlab_math::{vec64, ArrayF64};
use rustlab_numerical::interpolation::*;
use rustlab_numerical::Result;
use std::f64::consts::PI;

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║   Mathematical Interpolation with RustLab-Numerical  ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // Example 1: Functional approach to sampling and interpolation
    example_functional_interpolation()?;
    
    // Example 2: Mathematical function composition
    example_function_composition()?;
    
    // Example 3: Vector operations with interpolation
    example_vector_interpolation()?;
    
    // Example 4: 2D functional surface interpolation
    example_functional_surfaces()?;
    
    // Example 5: Error norms and analysis
    example_error_norms()?;

    Ok(())
}

fn example_functional_interpolation() -> Result<()> {
    println!("1. Functional Approach to Interpolation");
    println!("═══════════════════════════════════════\n");
    
    // Create sample points using functional methods
    let n = 7;
    let x = linspace(0.0, 2.0 * PI, n);
    
    // Map x values to y = sin(x) + 0.1*cos(3x)
    let y = vec64_from_fn(n, |i| {
        let xi = x.get(i).unwrap();
        xi.sin() + 0.1 * (3.0 * xi).cos()
    });
    
    println!("Sampling f(x) = sin(x) + 0.1·cos(3x) on [0, 2π]");
    
    // Create different interpolators
    let linear = LinearInterpolator::new(x.clone(), y.clone())?;
    let spline = CubicSpline::new(x.clone(), y.clone(), BoundaryCondition::Natural)?;
    
    // Evaluate on a dense grid using functional approach
    let n_eval = 50;
    let x_dense = linspace(0.0, 2.0 * PI, n_eval);
    
    // Map through interpolators
    let y_true = vec64_from_fn(n_eval, |i| {
        let xi = x_dense.get(i).unwrap();
        xi.sin() + 0.1 * (3.0 * xi).cos()
    });
    
    let y_linear = vec64_from_fn(n_eval, |i| {
        linear.eval(x_dense.get(i).unwrap()).unwrap()
    });
    
    let y_spline = vec64_from_fn(n_eval, |i| {
        spline.eval(x_dense.get(i).unwrap()).unwrap()
    });
    
    // Compute L2 norms of errors using vector operations
    let linear_error = compute_rms_error(&y_linear, &y_true);
    let spline_error = compute_rms_error(&y_spline, &y_true);
    
    println!("\nRMS Errors:");
    println!("  Linear interpolation: {:.6}", linear_error);
    println!("  Cubic spline:         {:.6}", spline_error);
    println!("  Improvement factor:   {:.2}x", linear_error / spline_error);
    
    Ok(())
}

fn example_function_composition() -> Result<()> {
    println!("\n2. Mathematical Function Composition");
    println!("════════════════════════════════════\n");
    
    // Demonstrate interpolation of composed functions
    let n = 8;
    let x = vec64_from_fn(n, |i| -1.0 + 2.0 * i as f64 / (n - 1) as f64);
    
    // g(x) = exp(-x²/2) (Gaussian)
    let g = |x: f64| (-x * x / 2.0).exp();
    
    // h(x) = x³ - x (polynomial)
    let h = |x: f64| x.powi(3) - x;
    
    // f(x) = g(h(x)) - composition
    let f = |x: f64| g(h(x));
    
    // Sample the composed function
    let y = vec64_from_fn(n, |i| f(x.get(i).unwrap()));
    
    println!("Interpolating f(x) = exp(-(x³-x)²/2)");
    println!("This is the composition g∘h where:");
    println!("  g(x) = exp(-x²/2)");
    println!("  h(x) = x³ - x");
    
    let poly = NewtonInterpolator::new(x.clone(), y.clone())?;
    let spline = CubicSpline::new(x, y, BoundaryCondition::Natural)?;
    
    // Test at critical points
    let test_points = vec![0.0, 0.5, -0.5, 0.577, -0.577]; // Including ±1/√3
    
    println!("\nEvaluation at test points:");
    for x_val in test_points {
        let true_val = f(x_val);
        let poly_val = poly.eval(x_val)?;
        let spline_val = spline.eval(x_val)?;
        
        println!("\nx = {:.3}:", x_val);
        println!("  True f(x):    {:.6}", true_val);
        println!("  Polynomial:   {:.6} (error: {:.2e})", 
                 poly_val, (poly_val - true_val).abs());
        println!("  Spline:       {:.6} (error: {:.2e})", 
                 spline_val, (spline_val - true_val).abs());
    }
    
    Ok(())
}

fn example_vector_interpolation() -> Result<()> {
    println!("\n3. Vector Operations with Interpolation");
    println!("═══════════════════════════════════════\n");
    
    // Create parametric curve: (x(t), y(t))
    let n = 10;
    let t = linspace(0.0, 2.0 * PI, n);
    
    // Lissajous curve
    let x_curve = vec64_from_fn(n, |i| {
        let ti = t.get(i).unwrap();
        (2.0 * ti).sin()
    });
    
    let y_curve = vec64_from_fn(n, |i| {
        let ti = t.get(i).unwrap();
        (3.0 * ti).sin()
    });
    
    println!("Parametric Lissajous curve:");
    println!("  x(t) = sin(2t)");
    println!("  y(t) = sin(3t)");
    
    // Create interpolators for both components (using Natural BC)
    let x_interp = CubicSpline::new(t.clone(), x_curve, BoundaryCondition::Natural)?;
    let y_interp = CubicSpline::new(t.clone(), y_curve, BoundaryCondition::Natural)?;
    
    // Compute arc length using interpolated derivatives
    let n_segments = 100;
    let dt = 2.0 * PI / n_segments as f64;
    let mut arc_length = 0.0;
    
    for i in 0..n_segments {
        let t1 = i as f64 * dt;
        let t2 = (i + 1) as f64 * dt;
        
        let dx_dt1 = x_interp.eval_derivative(t1)?;
        let dy_dt1 = y_interp.eval_derivative(t1)?;
        let dx_dt2 = x_interp.eval_derivative(t2)?;
        let dy_dt2 = y_interp.eval_derivative(t2)?;
        
        // Trapezoidal rule for arc length
        let speed1 = (dx_dt1 * dx_dt1 + dy_dt1 * dy_dt1).sqrt();
        let speed2 = (dx_dt2 * dx_dt2 + dy_dt2 * dy_dt2).sqrt();
        arc_length += 0.5 * (speed1 + speed2) * dt;
    }
    
    println!("\nApproximate arc length: {:.4}", arc_length);
    
    // Sample curvature at a few points
    println!("\nCurvature κ(t) = |x'y'' - x''y'| / (x'² + y'²)^(3/2):");
    for t_val in [0.0, PI/4.0, PI/2.0, PI] {
        let dx = x_interp.eval_derivative(t_val)?;
        let dy = y_interp.eval_derivative(t_val)?;
        let ddx = x_interp.eval_second_derivative(t_val)?;
        let ddy = y_interp.eval_second_derivative(t_val)?;
        
        let curvature = (dx * ddy - ddx * dy).abs() / (dx * dx + dy * dy).powf(1.5_f64);
        println!("  κ({:.3}) = {:.4}", t_val, curvature);
    }
    
    Ok(())
}

fn example_functional_surfaces() -> Result<()> {
    println!("\n4. Functional 2D Surface Interpolation");
    println!("══════════════════════════════════════\n");
    
    // Create a mathematical surface using functional operations
    let nx = 6;
    let ny = 6;
    
    let x = linspace(-1.0, 1.0, nx);
    let y = linspace(-1.0, 1.0, ny);
    
    // Build surface: f(x,y) = exp(-(x²+y²)) * cos(2π√(x²+y²))
    let mut z_data = vec![vec![0.0; ny]; nx];
    for i in 0..nx {
        for j in 0..ny {
            let xi = x.get(i).unwrap();
            let yj = y.get(j).unwrap();
            let r = (xi * xi + yj * yj).sqrt();
            z_data[i][j] = (-r * r).exp() * (2.0 * PI * r).cos();
        }
    }
    let z = ArrayF64::from_vec2d(z_data)?;
    
    println!("Surface: f(x,y) = exp(-(x²+y²)) · cos(2π√(x²+y²))");
    println!("(2D radial wave function)");
    
    // For comparison, create both interpolators
    let bilinear = BilinearInterpolator::new(x.clone(), y.clone(), z.clone())?;
    
    // Evaluate on a test grid and compute error norms
    let test_points = vec![
        (0.0, 0.0),    // Center (maximum)
        (0.3, 0.4),    // Inside first ring
        (-0.5, 0.0),   // On axis
        (0.35, 0.35),  // Diagonal
    ];
    
    println!("\nInterpolation results:");
    for (x_test, y_test) in test_points {
        let r_squared: f64 = x_test * x_test + y_test * y_test;
        let r = r_squared.sqrt();
        let true_val = (-r * r).exp() * (2.0 * PI * r).cos();
        let interp_val = bilinear.eval(x_test, y_test)?;
        
        println!("\n  f({:.2}, {:.2}):", x_test, y_test);
        println!("    True value:  {:.6}", true_val);
        println!("    Interpolated: {:.6}", interp_val);
        println!("    Rel. error:   {:.2e}", ((interp_val - true_val) / true_val.abs().max(1e-10)).abs());
    }
    
    Ok(())
}

fn example_error_norms() -> Result<()> {
    println!("\n5. Error Analysis with Functional Norms");
    println!("═══════════════════════════════════════\n");
    
    // Analyze convergence rates for different interpolation methods
    println!("Convergence study for f(x) = 1/(1 + x²) on [-5, 5]");
    
    let f = |x: f64| 1.0 / (1.0 + x * x);
    
    // Test with increasing number of points
    for n in [5, 9, 17] {
        let x_nodes = linspace(-5.0, 5.0, n);
        let y_nodes = vec64_from_fn(n, |i| f(x_nodes.get(i).unwrap()));
        
        // Create interpolators
        let linear = LinearInterpolator::new(x_nodes.clone(), y_nodes.clone())?;
        let spline = CubicSpline::new(x_nodes, y_nodes, BoundaryCondition::Natural)?;
        
        // Evaluate on dense test grid
        let n_test = 1000;
        let x_test = linspace(-5.0, 5.0, n_test);
        let y_true = vec64_from_fn(n_test, |i| f(x_test.get(i).unwrap()));
        
        // Compute interpolated values
        let y_linear = vec64_from_fn(n_test, |i| {
            linear.eval(x_test.get(i).unwrap()).unwrap()
        });
        
        let y_spline = vec64_from_fn(n_test, |i| {
            spline.eval(x_test.get(i).unwrap()).unwrap()
        });
        
        // Compute different error norms
        let linear_l2 = compute_rms_error(&y_linear, &y_true);
        let spline_l2 = compute_rms_error(&y_spline, &y_true);
        
        let linear_linf = compute_max_error(&y_linear, &y_true);
        let spline_linf = compute_max_error(&y_spline, &y_true);
        
        println!("\nWith {} interpolation nodes:", n);
        println!("  L² errors:  Linear = {:.2e}, Spline = {:.2e}", linear_l2, spline_l2);
        println!("  L∞ errors:  Linear = {:.2e}, Spline = {:.2e}", linear_linf, spline_linf);
    }
    
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Functional interpolation examples completed!          ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    
    Ok(())
}

// Helper functions for error computation
fn compute_rms_error(computed: &VectorF64, true_vals: &VectorF64) -> f64 {
    let n = computed.len();
    let mut sum = 0.0;
    for i in 0..n {
        let diff = computed.get(i).unwrap() - true_vals.get(i).unwrap();
        sum += diff * diff;
    }
    (sum / n as f64).sqrt()
}

fn compute_max_error(computed: &VectorF64, true_vals: &VectorF64) -> f64 {
    let n = computed.len();
    let mut max_err = 0.0_f64;
    for i in 0..n {
        let diff = (computed.get(i).unwrap() - true_vals.get(i).unwrap()).abs();
        max_err = max_err.max(diff);
    }
    max_err
}

// Helper function for creating vectors from functions
fn vec64_from_fn<F>(n: usize, f: F) -> vec64 
where 
    F: Fn(usize) -> f64
{
    vec64::from_slice(&(0..n).map(f).collect::<Vec<f64>>())
}

// Helper function for creating linearly spaced vectors
fn linspace(start: f64, end: f64, n: usize) -> vec64 {
    vec64::from_slice(
        &(0..n).map(|i| {
            if i == n - 1 {
                end  // Ensure last point is exactly the endpoint
            } else {
                start + (end - start) * i as f64 / (n - 1) as f64
            }
        }).collect::<Vec<f64>>()
    )
}