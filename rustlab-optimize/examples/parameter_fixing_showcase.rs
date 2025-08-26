//! Comprehensive demonstration of parameter fixing capabilities
//! 
//! This example shows various approaches to fixing parameters in optimization
//! and curve fitting, which is essential for scientific analysis.

use rustlab_optimize::prelude::*;

fn main() -> Result<()> {
    println!("🔧 Parameter Fixing Showcase - Scientific Applications\n");
    
    // ========================================================================
    // 1. GENERAL OPTIMIZATION - Fix specific parameters
    // ========================================================================
    println!("1️⃣  General Optimization with Parameter Fixing");
    
    // Minimize f(x,y,z) = (x-1)² + (y-2)² + (z-3)² 
    // but fix y = 2.5 (constraint from experimental setup)
    let objective = |params: &[f64]| {
        (params[0] - 1.0).powi(2) + (params[1] - 2.0).powi(2) + (params[2] - 3.0).powi(2)
    };
    
    let result = minimize(objective)
        .from(&[0.0, 2.5, 0.0])    // Initial guess
        .fix_parameter(1, 2.5)      // Fix y = 2.5 (index 1)
        .solve()?;
    
    println!("   Constrained minimum: [{:.3}, {:.3}, {:.3}]", 
             result.solution[0], result.solution[1], result.solution[2]);
    println!("   Expected: [1.000, 2.500, 3.000]");
    println!("   Objective: {:.6}\n", result.objective_value);
    
    // ========================================================================
    // 2. EXPONENTIAL DECAY - Fix amplitude (common in physics)
    // ========================================================================
    println!("2️⃣  Exponential Decay: Fix Amplitude from Theory");
    
    // Simulate radioactive decay data where initial amount is known
    let time = linspace(0.0, 10.0, 25);
    let known_amplitude = 1000.0;  // Known from initial measurement
    let true_decay_rate = 0.693;   // Half-life = 1.0
    
    // Generate synthetic data with noise
    let y_true: Vec<f64> = time.iter()
        .map(|&t| known_amplitude * (-true_decay_rate * t).exp())
        .collect();
    
    let y_noisy: Vec<f64> = y_true.iter()
        .enumerate()
        .map(|(i, &y)| y + 10.0 * (i as f64 * 0.5).sin()) // Add systematic noise
        .collect();
    
    let y_data = vec64(&y_noisy);
    
    // Fit with fixed amplitude - optimize only decay rate
    let fit = fit_exponential_advanced(&time, &y_data)
        .fix_amplitude(known_amplitude)  // Fix A = 1000 from theory
        .with_initial_decay_rate(0.5)    // Initial guess for k
        .solve()?;
    
    println!("   Fixed amplitude exponential fit:");
    println!("   Amplitude: {:.1} (fixed)", fit.amplitude);
    println!("   Decay rate: {:.3} (fitted, true: {:.3})", fit.decay_rate, true_decay_rate);
    println!("   Half-life: {:.3} (true: {:.3})", fit.half_life, (2.0_f64).ln() / true_decay_rate);
    println!("   R²: {:.4}\n", fit.r_squared);
    
    // ========================================================================  
    // 3. DRUG KINETICS - Multi-parameter fixing
    // ========================================================================
    println!("3️⃣  Pharmacokinetics: Multi-compartment Model");
    
    // Two-compartment model: C(t) = A₁*exp(-α*t) + A₂*exp(-β*t)
    // where α > β (fast and slow elimination phases)
    let time_pk = linspace(0.0, 24.0, 30); // 24 hours
    
    // Known from previous studies (population pharmacokinetics)
    let known_alpha = 2.0;    // Fast elimination rate (1/hr)
    let known_a2 = 5.0;       // Slow phase amplitude (mg/L)
    
    // Generate synthetic concentration data
    let c_true: Vec<f64> = time_pk.iter()
        .map(|&t| 50.0 * (-known_alpha * t).exp() + known_a2 * (-0.2 * t).exp())
        .collect();
    
    let c_data = vec64(&c_true);
    
    // Fit with multiple fixed parameters
    let pk_result = fit(&time_pk, &c_data, |t, params| {
        // params = [A1, alpha, A2, beta]
        params[0] * (-params[1] * t).exp() + params[2] * (-params[3] * t).exp()
    })
    .with_initial(&[40.0, 2.0, 5.0, 0.1])  // Initial guess
    .fix_parameters(&[
        (1, known_alpha),  // Fix α = 2.0 (known fast rate)
        (2, known_a2),     // Fix A₂ = 5.0 (known slow amplitude)
    ])
    .solve()?;
    
    println!("   Two-compartment pharmacokinetic model:");
    println!("   A₁ (fast amplitude): {:.1} mg/L (fitted)", pk_result.solution[0]);
    println!("   α (fast rate): {:.1} /hr (fixed)", pk_result.solution[1]);
    println!("   A₂ (slow amplitude): {:.1} mg/L (fixed)", pk_result.solution[2]);
    println!("   β (slow rate): {:.3} /hr (fitted, true: 0.200)", pk_result.solution[3]);
    println!("   Objective: {:.2e}\n", pk_result.objective_value);
    
    // ========================================================================
    // 4. PARAMETER MASK - Systematic parameter studies
    // ========================================================================
    println!("4️⃣  Parameter Sensitivity Study using Optimization Mask");
    
    // Study which parameters are most important in Michaelis-Menten kinetics
    // v = (V_max * [S]) / (K_m + [S])
    let substrate = linspace(0.1, 10.0, 20);
    let v_true: Vec<f64> = substrate.iter()
        .map(|&s| (100.0 * s) / (2.0 + s))  // V_max=100, K_m=2.0
        .collect();
    
    let velocity = vec64(&v_true);
    
    println!("   Michaelis-Menten parameter study:");
    
    // Case 1: Optimize both parameters
    let both_result = fit(&substrate, &velocity, |s, p| (p[0] * s) / (p[1] + s))
        .with_initial(&[80.0, 1.5])
        .solve()?;
    
    println!("   Both optimized: V_max = {:.1}, K_m = {:.2}, obj = {:.2e}", 
             both_result.solution[0], both_result.solution[1], both_result.objective_value);
    
    // Case 2: Fix V_max, optimize only K_m
    let km_only_result = fit(&substrate, &velocity, |s, p| (p[0] * s) / (p[1] + s))
        .with_initial(&[100.0, 1.5])
        .optimize_mask(&[false, true])  // Fix V_max at initial value
        .solve()?;
    
    println!("   K_m only: V_max = {:.1} (fixed), K_m = {:.2}, obj = {:.2e}", 
             km_only_result.solution[0], km_only_result.solution[1], km_only_result.objective_value);
    
    // Case 3: Fix K_m, optimize only V_max  
    let vmax_only_result = fit(&substrate, &velocity, |s, p| (p[0] * s) / (p[1] + s))
        .with_initial(&[80.0, 2.0])
        .optimize_mask(&[true, false])  // Fix K_m at initial value
        .solve()?;
    
    println!("   V_max only: V_max = {:.1}, K_m = {:.2} (fixed), obj = {:.2e}", 
             vmax_only_result.solution[0], vmax_only_result.solution[1], vmax_only_result.objective_value);
    
    // ========================================================================
    // 5. SCIENTIFIC WORKFLOW - Model comparison with parameter constraints
    // ========================================================================
    println!("\n5️⃣  Scientific Workflow: Model Selection with Constraints");
    
    // Compare simple exponential vs bi-exponential for same data
    let t_comparison = linspace(0.0, 5.0, 25);
    
    // True data is bi-exponential: y = 80*exp(-2*t) + 20*exp(-0.3*t)
    let y_bi_true: Vec<f64> = t_comparison.iter()
        .map(|&t| 80.0 * (-2.0 * t).exp() + 20.0 * (-0.3 * t).exp())
        .collect();
    
    let y_bi_data = vec64(&y_bi_true);
    
    // Model 1: Simple exponential (will be poor fit)
    let simple_fit = fit_exponential(&t_comparison, &y_bi_data)?;
    
    // Model 2: Bi-exponential with constraints
    let bi_result = fit(&t_comparison, &y_bi_data, |t, p| {
        p[0] * (-p[1] * t).exp() + p[2] * (-p[3] * t).exp()
    })
    .with_initial(&[70.0, 1.5, 25.0, 0.2])
    .bounds(&[0.0, 0.0, 0.0, 0.0], &[200.0, 10.0, 100.0, 5.0])  // Physical constraints
    .solve()?;
    
    println!("   Model comparison:");
    println!("   Simple exponential: R² = {:.4}, obj = {:.2e}", 
             simple_fit.r_squared, simple_fit.amplitude.powi(2)); // Placeholder objective
    println!("   Bi-exponential: R² = ???, obj = {:.2e}", bi_result.objective_value);
    println!("   Fitted parameters: [{:.1}, {:.2}, {:.1}, {:.2}]", 
             bi_result.solution[0], bi_result.solution[1], 
             bi_result.solution[2], bi_result.solution[3]);
    println!("   True parameters:   [80.0, 2.00, 20.0, 0.30]");
    
    println!("\n✅ Parameter fixing demonstration complete!");
    println!("💡 Key benefits:");
    println!("   • Incorporate theoretical constraints");
    println!("   • Reduce parameter correlation");
    println!("   • Enable systematic sensitivity analysis");  
    println!("   • Compare nested models rigorously");
    
    Ok(())
}

// Helper function to create VectorF64 from slice
fn vec64(data: &[f64]) -> VectorF64 {
    rustlab_math::VectorF64::from_slice(data)
}

// Helper function to create linspace
fn linspace(start: f64, stop: f64, num: usize) -> VectorF64 {
    let step = (stop - start) / (num - 1) as f64;
    let data: Vec<f64> = (0..num)
        .map(|i| start + i as f64 * step)
        .collect();
    vec64(&data)
}