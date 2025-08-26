use rustlab_optimize::prelude::*;
use rustlab_optimize::core::ConvergenceStatus;

#[test]
fn test_notebook_01_examples() {
    // Test 1D optimization
    let f = |x: f64| (x - 3.0).powi(2) + 1.0;
    let x_min = minimize_1d(f).solve().unwrap();
    assert!((x_min - 3.0).abs() < 1e-6);
    
    // Test 2D optimization
    let rosenbrock = |x: f64, y: f64| {
        (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
    };
    
    let (x, y) = minimize_2d(rosenbrock)
        .from(0.0, 0.0)
        .tolerance(1e-6)
        .solve()
        .unwrap();
    
    assert!((x - 1.0).abs() < 1e-3);
    assert!((y - 1.0).abs() < 1e-3);
    
    // Test N-D optimization
    let sphere = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum::<f64>();
    
    let result = minimize(sphere)
        .from(&[1.0, 2.0, 3.0])
        .solve()
        .unwrap();
    
    assert!(result.objective_value < 1e-6);
    assert!(matches!(result.convergence, ConvergenceStatus::Success));
}

#[test]
fn test_algorithm_selection() {
    let quadratic = |x: &[f64]| {
        (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
    };
    
    // Test automatic selection
    let auto_result = minimize(quadratic)
        .from(&[0.0, 0.0])
        .solve()
        .unwrap();
    
    assert!((auto_result.solution[0] - 1.0).abs() < 1e-6);
    assert!((auto_result.solution[1] - 2.0).abs() < 1e-6);
    
    // Test explicit BFGS
    let bfgs_result = minimize(quadratic)
        .from(&[0.0, 0.0])
        .using_bfgs()
        .solve()
        .unwrap();
    
    assert!((bfgs_result.solution[0] - 1.0).abs() < 1e-6);
    assert!((bfgs_result.solution[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_convergence_tolerance() {
    let f = |x: &[f64]| {
        100.0 * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2)
    };
    
    // Loose tolerance - should converge quickly
    let result_loose = minimize(f)
        .from(&[-1.2, 1.0])
        .tolerance(1e-3)
        .solve()
        .unwrap();
    
    // Tight tolerance - should take more iterations
    let result_tight = minimize(f)
        .from(&[-1.2, 1.0])
        .tolerance(1e-9)
        .solve()
        .unwrap();
    
    assert!(result_loose.iterations < result_tight.iterations);
    assert!(result_tight.objective_value <= result_loose.objective_value);
}

#[test]
fn test_bounds() {
    // Test with bounds that restrict the minimum
    let f = |x: f64| (x - 3.0).powi(2);
    
    // Without bounds, minimum at x=3
    let unbounded = minimize_1d(f).solve().unwrap();
    assert!((unbounded - 3.0).abs() < 1e-6);
    
    // With bounds [0, 2], minimum at x=2
    let bounded = minimize_1d(f)
        .bounds(0.0, 2.0)
        .solve()
        .unwrap();
    assert!((bounded - 2.0).abs() < 1e-6);
}