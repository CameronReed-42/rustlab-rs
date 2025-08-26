//! Shared utilities for notebook examples
//! 
//! Common functions and helpers used across multiple notebook examples

use rustlab_optimize::prelude::*;
use rustlab_math::prelude::*;
use std::f64::consts::PI;

/// Generate synthetic data with noise for testing curve fitting
pub fn generate_noisy_data(
    x: &[f64], 
    true_fn: impl Fn(f64) -> f64, 
    noise_level: f64
) -> Vec<f64> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    
    x.iter()
        .map(|&xi| {
            let true_val = true_fn(xi);
            let noise = noise_level * (rng.gen::<f64>() - 0.5) * 2.0;
            true_val + noise
        })
        .collect()
}

/// Calculate R-squared (coefficient of determination)
pub fn calculate_r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len();
    assert_eq!(n, y_pred.len(), "Arrays must have same length");
    
    let y_mean = y_true.iter().sum::<f64>() / n as f64;
    
    let ss_tot: f64 = y_true.iter()
        .map(|&y| (y - y_mean).powi(2))
        .sum();
    
    let ss_res: f64 = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
        .sum();
    
    1.0 - (ss_res / ss_tot)
}

/// Calculate residuals
pub fn calculate_residuals(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    y_true.iter()
        .zip(y_pred.iter())
        .map(|(&y_t, &y_p)| y_t - y_p)
        .collect()
}

/// Calculate root mean square error (RMSE)
pub fn calculate_rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    let mse: f64 = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
        .sum::<f64>() / n;
    
    mse.sqrt()
}

/// Print optimization result summary
pub fn print_result_summary(result: &OptimizationResult) {
    println!("Optimization Result Summary:");
    println!("  Algorithm: {}", result.algorithm_used.name());
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.function_evals);
    println!("  Final value: {:.6e}", result.objective_value);
    
    if result.solution.len() <= 5 {
        println!("  Solution: {:?}", result.solution);
    } else {
        println!("  Solution (first 5): [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, ...]",
                 result.solution[0], result.solution[1], result.solution[2],
                 result.solution[3], result.solution[4]);
    }
    
    match &result.convergence_status {
        ConvergenceStatus::Converged { gradient_norm, .. } => {
            println!("  Gradient norm: {:.2e}", gradient_norm);
        }
        ConvergenceStatus::MaxIterations => {
            println!("  Status: Maximum iterations reached");
        }
        _ => {}
    }
}

/// Generate grid points for 2D function visualization
pub fn generate_2d_grid(
    x_range: (f64, f64), 
    y_range: (f64, f64), 
    n_points: usize
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let x_step = (x_range.1 - x_range.0) / (n_points - 1) as f64;
    let y_step = (y_range.1 - y_range.0) / (n_points - 1) as f64;
    
    let mut x_grid = vec![vec![0.0; n_points]; n_points];
    let mut y_grid = vec![vec![0.0; n_points]; n_points];
    
    for i in 0..n_points {
        for j in 0..n_points {
            x_grid[i][j] = x_range.0 + j as f64 * x_step;
            y_grid[i][j] = y_range.0 + i as f64 * y_step;
        }
    }
    
    (x_grid, y_grid)
}

/// Common test functions for optimization benchmarks

/// Rosenbrock function (2D)
pub fn rosenbrock(x: f64, y: f64) -> f64 {
    let a = 1.0;
    let b = 100.0;
    (a - x).powi(2) + b * (y - x.powi(2)).powi(2)
}

/// Himmelblau's function (2D) - has 4 equal minima
pub fn himmelblau(x: f64, y: f64) -> f64 {
    (x.powi(2) + y - 11.0).powi(2) + (x + y.powi(2) - 7.0).powi(2)
}

/// Beale function (2D)
pub fn beale(x: f64, y: f64) -> f64 {
    (1.5 - x + x*y).powi(2) + 
    (2.25 - x + x*y.powi(2)).powi(2) + 
    (2.625 - x + x*y.powi(3)).powi(2)
}

/// Ackley function (N-D)
pub fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * PI;
    
    let sum1 = x.iter().map(|&xi| xi.powi(2)).sum::<f64>() / n;
    let sum2 = x.iter().map(|&xi| (c * xi).cos()).sum::<f64>() / n;
    
    -a * (-b * sum1.sqrt()).exp() - sum2.exp() + a + 1_f64.exp()
}

/// Rastrigin function (N-D) - highly multimodal
pub fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter()
        .map(|&xi| xi.powi(2) - 10.0 * (2.0 * PI * xi).cos())
        .sum::<f64>()
}

/// Sphere function (N-D) - simple convex
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi.powi(2)).sum()
}

/// Format vector for display (truncate if too long)
pub fn format_vector(v: &[f64], max_display: usize) -> String {
    if v.len() <= max_display {
        format!("{:?}", v)
    } else {
        let displayed: Vec<String> = v.iter()
            .take(max_display)
            .map(|x| format!("{:.3}", x))
            .collect();
        format!("[{}... ({} total)]", displayed.join(", "), v.len())
    }
}

/// Compare optimization algorithms on the same problem
pub fn compare_algorithms(
    objective: impl Fn(&[f64]) -> f64 + Clone + Send + Sync + 'static,
    initial: &[f64],
    algorithms: Vec<(&str, Box<dyn Solver>)>
) -> Result<()> {
    println!("Algorithm Comparison:");
    println!("{:<20} {:>10} {:>15} {:>15}", "Algorithm", "Iterations", "Final Value", "Time (ms)");
    println!("{}", "-".repeat(60));
    
    for (name, solver) in algorithms {
        let start = std::time::Instant::now();
        
        let problem = OptimizationProblem::new(
            ObjectiveFunction::Scalar(Box::new(objective.clone())),
            initial,
            None
        );
        
        let result = solver.solve(problem)?;
        let elapsed = start.elapsed().as_millis();
        
        println!("{:<20} {:>10} {:>15.6e} {:>15}", 
                 name, result.iterations, result.objective_value, elapsed);
    }
    
    Ok(())
}

/// Generate initial points for multi-start optimization
pub fn generate_random_starts(bounds: &[(f64, f64)], n_starts: usize) -> Vec<Vec<f64>> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    
    (0..n_starts)
        .map(|_| {
            bounds.iter()
                .map(|(low, high)| rng.gen::<f64>() * (high - low) + low)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_r_squared() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 1.9, 3.1, 3.9, 5.1];
        let r2 = calculate_r_squared(&y_true, &y_pred);
        assert!(r2 > 0.98 && r2 < 1.0);
    }
    
    #[test]
    fn test_rmse() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 3.0];
        let rmse = calculate_rmse(&y_true, &y_pred);
        assert_eq!(rmse, 0.0);
    }
    
    #[test]
    fn test_sphere_minimum() {
        let x = vec![0.0, 0.0, 0.0];
        assert_eq!(sphere(&x), 0.0);
    }
}