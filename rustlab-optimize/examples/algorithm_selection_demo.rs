//! Demonstration of bounds-aware algorithm selection
//!
//! This example shows how the algorithm selection logic considers bounds,
//! parameter fixing, and problem characteristics to choose optimal solvers.

use rustlab_optimize::*;
use rustlab_optimize::algorithms::{recommend_algorithm, OptimizationProblem, ProblemCharacteristics};
use rustlab_math::vec64;

fn main() -> Result<()> {
    println!("🔍 Algorithm Selection Demo");
    println!("============================\n");
    
    // Example 1: Unconstrained optimization
    println!("📍 Example 1: Unconstrained Quadratic Problem");
    let objective1 = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    let initial1 = vec![1.0, 1.0];
    let problem1 = minimize(objective1).from(&initial1).build_problem()?;
    
    let (solver1, reason1) = recommend_algorithm(&problem1);
    println!("   Selected: {} Algorithm", solver1.algorithm().name());
    println!("   Reason: {}\n", reason1);
    
    // Example 2: Constrained optimization  
    println!("📍 Example 2: Bounded Quadratic Problem");
    let objective2 = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 2.0).powi(2);
    let initial2 = vec![0.0, 0.0];
    let problem2 = minimize(objective2)
        .from(&initial2)
        .bounds(&[0.0, 0.0], &[5.0, 4.0])
        .build_problem()?;
        
    let (solver2, reason2) = recommend_algorithm(&problem2);
    println!("   Selected: {} Algorithm", solver2.algorithm().name());
    println!("   Reason: {}\n", reason2);
    
    // Example 3: Least squares curve fitting
    println!("📍 Example 3: Nonlinear Least Squares");
    let x_data = vec64![0.0, 1.0, 2.0, 3.0, 4.0];
    let y_data = vec64![1.0, 2.7, 7.4, 20.1, 54.6];
    
    let residual = move |params: &rustlab_math::VectorF64| -> rustlab_math::VectorF64 {
        let a = params[0];
        let b = params[1];
        let mut residuals = rustlab_math::VectorF64::zeros(5);
        for i in 0..5 {
            let x = x_data[i];
            let predicted = a * (b * x).exp();
            let observed = y_data[i];
            residuals[i] = predicted - observed;
        }
        residuals
    };
    
    let initial3 = vec64![1.0, 0.5];
    let problem3 = rustlab_optimize::algorithms::OptimizationProblem::least_squares(residual, &initial3, None)
        .with_characteristics(rustlab_optimize::algorithms::ProblemCharacteristics::CurveFitting);
        
    let (solver3, reason3) = recommend_algorithm(&problem3);
    println!("   Selected: {} Algorithm", solver3.algorithm().name());
    println!("   Reason: {}\n", reason3);
    
    // Example 4: Large-scale unconstrained
    println!("📍 Example 4: Large-Scale Problem");
    let large_objective = |x: &[f64]| x.iter().map(|xi| xi.powi(2)).sum::<f64>();
    let initial4 = vec![1.0; 500];  // 500-dimensional problem
    let problem4 = minimize(large_objective)
        .from(&initial4)
        .with_characteristics(ProblemCharacteristics::LargeScale)
        .build_problem()?;
        
    let (solver4, reason4) = recommend_algorithm(&problem4);
    println!("   Selected: {} Algorithm", solver4.algorithm().name());
    println!("   Reason: {}\n", reason4);
    
    // Example 5: Large-scale with bounds
    println!("📍 Example 5: Large-Scale with Bounds");
    let lower5 = vec![-1.0; 500];
    let upper5 = vec![1.0; 500];
    let problem5 = minimize(large_objective)
        .from(&initial4)
        .bounds(&lower5, &upper5)
        .with_characteristics(ProblemCharacteristics::LargeScale)
        .build_problem()?;
        
    let (solver5, reason5) = recommend_algorithm(&problem5);
    println!("   Selected: {} Algorithm", solver5.algorithm().name());
    println!("   Reason: {}\n", reason5);
    
    // Example 6: Parameter fixing
    println!("📍 Example 6: Problem with Fixed Parameters");
    let objective6 = |x: &[f64]| x[0].powi(2) + x[1].powi(2) + x[2].powi(2);
    let initial6 = vec![1.0, 2.0, 3.0];
    let problem6 = minimize(objective6)
        .from(&initial6)
        .fix_parameters(&[(1, 2.0)])  // Fix second parameter
        .build_problem()?;
        
    let (solver6, reason6) = recommend_algorithm(&problem6);
    println!("   Selected: {} Algorithm", solver6.algorithm().name());
    println!("   Reason: {}\n", reason6);
    
    // Example 7: Noisy problem
    println!("📍 Example 7: Noisy/Non-smooth Problem");
    let noisy_objective = |x: &[f64]| {
        // Add noise to make it non-smooth
        let base = x[0].powi(2) + x[1].powi(2);
        base + 0.1 * (100.0 * x[0]).sin() + 0.1 * (100.0 * x[1]).cos()
    };
    let initial7 = vec![1.0, 1.0];
    let problem7 = minimize(noisy_objective)
        .from(&initial7)
        .with_characteristics(ProblemCharacteristics::NoisyNonSmooth)
        .build_problem()?;
        
    let (solver7, reason7) = recommend_algorithm(&problem7);
    println!("   Selected: {} Algorithm", solver7.algorithm().name());
    println!("   Reason: {}\n", reason7);
    
    // Example 8: Noisy problem with bounds
    println!("📍 Example 8: Noisy Problem with Bounds");
    let problem8 = minimize(noisy_objective)
        .from(&initial7)
        .bounds(&[-2.0, -2.0], &[2.0, 2.0])
        .with_characteristics(ProblemCharacteristics::NoisyNonSmooth)
        .build_problem()?;
        
    let (solver8, reason8) = recommend_algorithm(&problem8);
    println!("   Selected: {} Algorithm", solver8.algorithm().name());
    println!("   Reason: {}\n", reason8);
    
    println!("✅ Algorithm Selection Summary:");
    println!("   • Bounds-aware selection prioritizes BFGS and L-M for constrained problems");
    println!("   • Problem characteristics guide primary algorithm choice");
    println!("   • Dimension scaling considered for large problems");
    println!("   • Parameter fixing support checked");
    println!("   • Fallback to BFGS ensures robustness");
    
    Ok(())
}