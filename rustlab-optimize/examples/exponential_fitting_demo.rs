//! Exponential fitting demonstration using the new math-first API
//!
//! This example showcases the completed Levenberg-Marquardt implementation
//! with parameter fixing for exponential decay fitting.

use rustlab_optimize::prelude::*;

fn main() -> Result<()> {
    println!("🧮 Exponential Fitting Proof of Concept\n");
    
    // ========================================================================
    // Generate synthetic exponential decay data: y = 10 * exp(-0.5 * x)
    // ========================================================================
    
    let true_amplitude = 10.0;
    let true_decay_rate = 0.5;
    
    // Create time points using math-first operations
    let time = linspace(0.0, 6.0, 25);
    
    // Generate true exponential decay
    let mut y_true = Vec::with_capacity(time.len());
    for &t in time.iter() {
        y_true.push(true_amplitude * (-true_decay_rate * t).exp());
    }
    
    // Add some noise to make it realistic
    let mut y_noisy = Vec::with_capacity(y_true.len());
    for (i, &y) in y_true.iter().enumerate() {
        let noise = 0.2 * (i as f64 * 0.7).sin(); // Systematic noise
        y_noisy.push(y + noise);
    }
    
    let y_data = VectorF64::from_slice(&y_noisy);
    
    println!("Data generated:");
    println!("  True parameters: A = {:.1}, k = {:.2}", true_amplitude, true_decay_rate);
    println!("  Data points: {}", time.len());
    println!("  Time range: [{:.1}, {:.1}]", time[0], time[time.len()-1]);
    
    // ========================================================================
    // 1. Basic exponential fitting (automatic parameter estimation)
    // ========================================================================
    
    println!("\n1️⃣  Basic Exponential Fitting (Auto LM Selection)");
    
    match fit_exponential(&time, &y_data) {
        Ok(fit) => {
            println!("   ✅ Fitting successful!");
            println!("   Amplitude: {:.3} (true: {:.3}, error: {:.1}%)", 
                     fit.amplitude, true_amplitude, 
                     100.0 * (fit.amplitude - true_amplitude).abs() / true_amplitude);
            println!("   Decay rate: {:.3} (true: {:.3}, error: {:.1}%)", 
                     fit.decay_rate, true_decay_rate,
                     100.0 * (fit.decay_rate - true_decay_rate).abs() / true_decay_rate);
            println!("   Half-life: {:.3} (true: {:.3})", 
                     fit.half_life, (2.0_f64).ln() / true_decay_rate);
            println!("   R²: {:.6}", fit.r_squared);
        }
        Err(e) => {
            println!("   ❌ Fitting failed: {}", e);
        }
    }
    
    // ========================================================================
    // 2. Advanced fitting with parameter fixing
    // ========================================================================
    
    println!("\n2️⃣  Parameter Fixing: Fix Amplitude (Known from Theory)");
    
    match fit_exponential_advanced(&time, &y_data)
        .fix_amplitude(true_amplitude)  // Fix A = 10.0
        .with_initial_decay_rate(0.3)   // Initial guess for k
        .solve()
    {
        Ok(fit) => {
            println!("   ✅ Constrained fitting successful!");
            println!("   Amplitude: {:.3} (fixed)", fit.amplitude);
            println!("   Decay rate: {:.3} (fitted, true: {:.3}, error: {:.1}%)", 
                     fit.decay_rate, true_decay_rate,
                     100.0 * (fit.decay_rate - true_decay_rate).abs() / true_decay_rate);
            println!("   Half-life: {:.3}", fit.half_life);
            println!("   R²: {:.6}", fit.r_squared);
        }
        Err(e) => {
            println!("   ❌ Constrained fitting failed: {}", e);
        }
    }
    
    // ========================================================================
    // 3. Fitting with bounds (physical constraints)
    // ========================================================================
    
    println!("\n3️⃣  Bounded Fitting: A > 0, k ∈ [0.1, 2.0]");
    
    match fit_exponential_advanced(&time, &y_data)
        .amplitude_bounds(0.01, 50.0)      // Physical: amplitude must be positive
        .decay_rate_bounds(0.1, 2.0)       // Physical: reasonable decay rates
        .with_initial(8.0, 0.6)            // Initial guess
        .solve()
    {
        Ok(fit) => {
            println!("   ✅ Bounded fitting successful!");
            println!("   Amplitude: {:.3} (bounds: [0.01, 50.0])", fit.amplitude);
            println!("   Decay rate: {:.3} (bounds: [0.1, 2.0])", fit.decay_rate);
            println!("   Half-life: {:.3}", fit.half_life);
            println!("   R²: {:.6}", fit.r_squared);
        }
        Err(e) => {
            println!("   ❌ Bounded fitting failed: {}", e);
        }
    }
    
    // ========================================================================
    // 4. Show algorithm details
    // ========================================================================
    
    println!("\n4️⃣  Algorithm Information");
    
    // Test direct minimization approach for comparison
    let objective = |params: &[f64]| -> f64 {
        let amplitude = params[0];
        let decay_rate = params[1];
        
        let mut sse = 0.0;
        for (i, &t) in time.iter().enumerate() {
            let predicted = amplitude * (-decay_rate * t).exp();
            let observed = y_data[i];
            let residual = predicted - observed;
            sse += residual * residual;
        }
        sse
    };
    
    match minimize(objective)
        .from(&[8.0, 0.4])
        .using_levenberg_marquardt()
        .solve()
    {
        Ok(result) => {
            println!("   Direct minimization (explicit LM):");
            println!("   Algorithm used: {}", result.algorithm_used.name());
            println!("   Iterations: {}", result.iterations);
            println!("   Function evaluations: {}", result.function_evaluations);
            println!("   Final objective: {:.2e}", result.objective_value);
            println!("   Success: {}", result.success);
            
            if let Some(grad_norm) = result.info.gradient_norm {
                println!("   Final gradient norm: {:.2e}", grad_norm);
            }
            
            // Show that parameters match curve fitting approach
            let amplitude = result.solution[0];
            let decay_rate = result.solution[1];
            println!("   Parameters: A = {:.3}, k = {:.3}", amplitude, decay_rate);
        }
        Err(e) => {
            println!("   Direct minimization failed: {}", e);
        }
    }
    
    println!("\n✅ Exponential Fitting Proof of Concept Complete!");
    println!("\n🎯 Key Achievements:");
    println!("   • Levenberg-Marquardt algorithm working with math-first operations");
    println!("   • Parameter fixing infrastructure functional");  
    println!("   • Curve fitting API automatically selects LM");
    println!("   • Advanced fitting with bounds and constraints");
    println!("   • Integration with rustlab-math VectorF64 operations");
    
    Ok(())
}

// Helper function to create linspace (would be in rustlab-math)
fn linspace(start: f64, stop: f64, num: usize) -> VectorF64 {
    let step = (stop - start) / (num - 1) as f64;
    let data: Vec<f64> = (0..num)
        .map(|i| start + i as f64 * step)
        .collect();
    VectorF64::from_slice(&data)
}