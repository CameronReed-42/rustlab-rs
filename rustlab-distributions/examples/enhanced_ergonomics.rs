//! Example demonstrating the enhanced ergonomic API for rustlab-distributions
//!
//! This example showcases:
//! - Ergonomic distribution creation without Result types
//! - Builder patterns for complex distributions
//! - Direct vector sampling methods
//! - Type-driven distribution fitting
//! - Automatic distribution selection

use rustlab_distributions::{
    EnhancedNormal, NormalBuilder, 
    fitting::{FitDistribution, FittingMethod, fitting_convenience},
    enhanced_api::convenience
};
use rand::thread_rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Enhanced RustLab Distributions API Demo ===\n");
    
    // 1. Ergonomic Distribution Creation
    println!("1. Ergonomic Distribution Creation:");
    
    // Old way (still available): Normal::new(0.0, 1.0).unwrap()
    // New way: No Result wrapper for valid inputs
    let normal = EnhancedNormal::new(5.0, 2.0);
    println!("   Normal distribution: μ={}, σ={}", normal.mean(), normal.std_dev());
    
    // Convenience constructors
    let standard = EnhancedNormal::standard();
    let unit_variance = EnhancedNormal::with_mean(10.0);
    let zero_mean = EnhancedNormal::with_variance(4.0);
    
    println!("   Standard normal: μ={}, σ={}", standard.mean(), standard.std_dev());
    println!("   Unit variance: μ={}, σ={}", unit_variance.mean(), unit_variance.std_dev());
    println!("   Zero mean: μ={}, σ={}", zero_mean.mean(), zero_mean.std_dev());
    
    // 2. Builder Pattern
    println!("\n2. Builder Pattern for Complex Distributions:");
    
    let complex_normal = NormalBuilder::new()
        .mean(100.0)
        .variance(25.0)  // σ² = 25, so σ = 5
        .build();
    
    println!("   Built distribution: μ={}, σ={}", 
             complex_normal.mean(), complex_normal.std_dev());
    
    // Builder with defaults
    let default_normal = NormalBuilder::new().build();  // μ=0, σ=1
    println!("   Default builder: μ={}, σ={}", 
             default_normal.mean(), default_normal.std_dev());
    
    // 3. Direct Vector Sampling
    println!("\n3. Direct Vector Sampling:");
    
    let mut rng = thread_rng();
    
    // Direct sampling into Vec
    let samples = normal.samples(1000, &mut rng);
    println!("   Generated {} samples directly into Vec", samples.len());
    
    // Sample statistics
    let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let sample_var = samples.iter()
        .map(|&x| (x - sample_mean).powi(2))
        .sum::<f64>() / (samples.len() - 1) as f64;
    
    println!("   Sample mean: {:.3} (expected: {:.3})", sample_mean, normal.mean());
    println!("   Sample std: {:.3} (expected: {:.3})", sample_var.sqrt(), normal.std_dev());
    
    // 4. Convenience Functions
    println!("\n4. Convenience Functions:");
    
    let quick_samples = convenience::standard_normal_samples(100);
    let custom_samples = convenience::normal_samples(50, 3.0, 1.5);
    
    println!("   Quick standard normal samples: {} values", quick_samples.len());
    println!("   Custom normal samples: {} values", custom_samples.len());
    
    // PDF/CDF for arrays
    let x_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let pdf_values = convenience::standard_normal_pdf(&x_values);
    let cdf_values = convenience::standard_normal_cdf(&x_values);
    
    println!("   PDF values at x={:?}: {:?}", x_values, 
             pdf_values.iter().map(|&v| format!("{:.3}", v)).collect::<Vec<_>>());
    println!("   CDF values at x={:?}: {:?}", x_values,
             cdf_values.iter().map(|&v| format!("{:.3}", v)).collect::<Vec<_>>());
    
    // 5. Type-Driven Distribution Fitting
    println!("\n5. Type-Driven Distribution Fitting:");
    
    // Generate some test data
    let test_data = convenience::normal_samples(500, 8.0, 3.0);
    
    // Type-driven fitting - the type system knows we want a Normal distribution
    let fit_result = test_data.fit::<EnhancedNormal>()?;
    
    println!("   Fitted distribution: {}", fit_result.distribution);
    println!("   Log-likelihood: {:.3}", fit_result.log_likelihood);
    println!("   AIC: {:.3}", fit_result.aic);
    println!("   BIC: {:.3}", fit_result.bic);
    
    // Compare with true parameters
    println!("   True parameters: μ=8.0, σ=3.0");
    println!("   Fitted parameters: μ={:.3}, σ={:.3}", 
             fit_result.distribution.mean(), fit_result.distribution.std_dev());
    
    // 6. Method Selection for Fitting
    println!("\n6. Fitting Method Selection:");
    
    let moments_fit = test_data.fit_with_method::<EnhancedNormal>(FittingMethod::Moments)?;
    let mle_fit = test_data.fit_with_method::<EnhancedNormal>(FittingMethod::MLE)?;
    
    println!("   Method of Moments: μ={:.3}, σ={:.3}", 
             moments_fit.distribution.mean(), moments_fit.distribution.std_dev());
    println!("   Maximum Likelihood: μ={:.3}, σ={:.3}", 
             mle_fit.distribution.mean(), mle_fit.distribution.std_dev());
    
    // 7. Automatic Distribution Selection
    println!("\n7. Automatic Distribution Selection:");
    
    let best_fit = test_data.fit_best()?;
    println!("   Best fitting distribution: {}", best_fit.distribution_type);
    println!("   Best AIC: {:.3}", best_fit.best_aic);
    println!("   Best BIC: {:.3}", best_fit.best_bic);
    
    if let Some(ref normal_fit) = best_fit.normal {
        println!("   Normal fit: {}", normal_fit.distribution);
    }
    
    // 8. Convenience Fitting Functions
    println!("\n8. Convenience Fitting Functions:");
    
    let simple_fit = fitting_convenience::fit_normal(&test_data)?;
    let moments_simple = fitting_convenience::fit_normal_moments(&test_data)?;
    let mle_simple = fitting_convenience::fit_normal_mle(&test_data)?;
    
    println!("   Simple fit: {}", simple_fit.distribution);
    println!("   Moments fit: {}", moments_simple.distribution);
    println!("   MLE fit: {}", mle_simple.distribution);
    
    // 9. Error-Free Operations (Panic vs Result)
    println!("\n9. Error Handling Philosophy:");
    
    // Math-first philosophy: panic on clearly invalid inputs
    println!("   Using panic-on-invalid for clean mathematical code:");
    let clean_normal = EnhancedNormal::new(0.0, 1.0);
    let quantile = clean_normal.quantile(0.95);  // No Result wrapper
    println!("   95th percentile: {:.3}", quantile);
    
    // Still available for programmatic use
    println!("   Using Result for programmatic error handling:");
    match EnhancedNormal::try_new(0.0, -1.0) {
        Ok(_) => println!("   This shouldn't happen"),
        Err(e) => println!("   Caught error: {}", e),
    }
    
    match clean_normal.try_quantile(1.5) {
        Ok(_) => println!("   This shouldn't happen"),
        Err(e) => println!("   Caught error: {}", e),
    }
    
    println!("\n=== Demo Complete ===");
    println!("The enhanced API provides:");
    println!("• Clean, math-first distribution creation");
    println!("• Builder patterns for complex configurations");
    println!("• Direct vector sampling without trait complexity");
    println!("• Type-driven distribution fitting");
    println!("• Automatic distribution selection");
    println!("• Comprehensive convenience functions");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_example_runs() {
        // Just verify the example compiles and runs without panicking
        main().unwrap();
    }
}