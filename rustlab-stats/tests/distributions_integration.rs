//! Integration tests with rustlab-distributions
//!
//! Tests that rustlab-stats works seamlessly with distribution sampling
//! and fitting capabilities.

#[cfg(test)]
mod distributions_integration {
    use rustlab_stats::prelude::*;
    use rustlab_math::{vec64, VectorF64};
    
    // Note: These tests are commented out until rustlab-distributions is available
    // They show the intended integration patterns
    
    /*
    use rustlab_distributions::{Normal, Exponential, Distribution};
    
    #[test]
    fn test_distribution_sampling_and_analysis() {
        // Sample from a normal distribution
        let normal = Normal::new(10.0, 2.0).unwrap();
        let samples: Vec<f64> = (0..1000).map(|_| normal.sample()).collect();
        let data = VectorF64::from_slice(&samples);
        
        // Analyze the samples
        let mean = data.mean();
        let std = data.std(None);
        let skew = data.skewness();
        let kurt = data.kurtosis(true);  // Excess kurtosis
        
        // Should be close to theoretical values
        assert!((mean - 10.0).abs() < 0.2);  // Close to μ=10
        assert!((std - 2.0).abs() < 0.2);    // Close to σ=2
        assert!(skew.abs() < 0.2);           // Normal is symmetric
        assert!(kurt.abs() < 0.5);           // Normal has excess kurtosis ≈ 0
        
        // Test normality
        let normalized = data.zscore();
        let ks_test = normalized.kolmogorov_smirnov_test(&Normal::standard());
        assert!(ks_test.p_value > 0.05);  // Fail to reject normality
    }
    
    #[test]
    fn test_distribution_fitting() {
        // Generate exponential data
        let exp = Exponential::new(2.0).unwrap();
        let samples: Vec<f64> = (0..1000).map(|_| exp.sample()).collect();
        let data = VectorF64::from_slice(&samples);
        
        // Fit distributions using type-driven API
        let fitted_normal = data.fit::<Normal>();
        let fitted_exp = data.fit::<Exponential>();
        
        // Exponential should fit better
        assert!(fitted_exp.log_likelihood > fitted_normal.log_likelihood);
        assert!(fitted_exp.aic < fitted_normal.aic);
        
        // Parameter should be close to true value
        assert!((fitted_exp.params.rate - 2.0).abs() < 0.2);
    }
    
    #[test]
    fn test_distribution_comparison() {
        // Sample from two different distributions
        let normal1 = Normal::new(0.0, 1.0).unwrap();
        let normal2 = Normal::new(0.5, 1.0).unwrap();
        
        let samples1: Vec<f64> = (0..500).map(|_| normal1.sample()).collect();
        let samples2: Vec<f64> = (0..500).map(|_| normal2.sample()).collect();
        
        let data1 = VectorF64::from_slice(&samples1);
        let data2 = VectorF64::from_slice(&samples2);
        
        // Statistical tests
        let t_test = data1.t_test_two_sample(&data2, false, Alternative::TwoSided);
        let mw_test = data1.mann_whitney_u(&data2, Alternative::TwoSided);
        
        // Should detect the difference
        assert!(t_test.p_value < 0.05);
        assert!(mw_test.p_value < 0.05);
    }
    */
    
    #[test]
    fn test_goodness_of_fit_preparation() {
        // Test chi-square goodness of fit with manual bins
        let data = vec64![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        
        // Create observed frequencies
        let observed = vec![1.0, 2.0, 3.0, 2.0, 1.0];  // Counts for bins 1,2,3,4,5
        let expected = vec![1.8, 1.8, 1.8, 1.8, 1.8];  // Uniform expectation
        
        let observed_vec = VectorF64::from_slice(&observed);
        let expected_vec = VectorF64::from_slice(&expected);
        
        let chi2_test = observed_vec.chi_square_goodness_of_fit(&expected_vec);
        
        // Should detect non-uniformity
        assert!(chi2_test.statistic > 0.0);
        assert_eq!(chi2_test.degrees_of_freedom, 4);  // 5 bins - 1
    }
    
    #[test]
    fn test_statistical_moments_for_distribution_fitting() {
        // Calculate moments that would be used for method of moments fitting
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        // First four moments
        let m1 = data.mean();
        let m2 = data.moment(2);
        let m3 = data.moment(3);
        let m4 = data.moment(4);
        
        // Central moments
        let var = data.var(None);
        let skew = data.skewness();
        let kurt = data.kurtosis(true);
        
        println!("Raw moments: m1={}, m2={}, m3={}, m4={}", m1, m2, m3, m4);
        println!("Central: var={}, skew={}, kurt={}", var, skew, kurt);
        
        // These would be used for distribution parameter estimation
        assert_eq!(m1, 5.5);
        assert!(var > 0.0);
        assert_eq!(skew, 0.0);  // Symmetric
    }
    
    #[test]
    fn test_sample_generation_validation() {
        // Test that we can validate if samples come from expected distribution
        // This prepares for future distribution integration
        
        // Generate pseudo-normal data using Box-Muller transform
        let n = 1000;
        let mut normal_samples = Vec::with_capacity(n);
        
        // Simple pseudo-random for demonstration
        let mut seed = 12345u64;
        for _ in 0..n/2 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let u1 = (seed as f64 / u64::MAX as f64).min(0.999).max(0.001);
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let u2 = (seed as f64 / u64::MAX as f64).min(0.999).max(0.001);
            
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            
            normal_samples.push(r * theta.cos());
            normal_samples.push(r * theta.sin());
        }
        
        let data = VectorF64::from_slice(&normal_samples);
        
        // Test properties of normal distribution
        let mean = data.mean();
        let std = data.std(None);
        let skew = data.skewness();
        let kurt = data.kurtosis(true);
        
        println!("Generated data: mean={:.3}, std={:.3}, skew={:.3}, kurt={:.3}", 
                 mean, std, skew, kurt);
        
        // Should be approximately standard normal
        assert!(mean.abs() < 0.1);
        assert!((std - 1.0).abs() < 0.1);
        assert!(skew.abs() < 0.2);
        assert!(kurt.abs() < 0.5);
    }
}