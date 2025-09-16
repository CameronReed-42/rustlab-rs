//! Comprehensive unit tests for numerical integration methods
//! 
//! Tests all integration methods against known analytical solutions
//! to verify correctness and accuracy

use rustlab_numerical::integration::*;
use rustlab_numerical::Result;
use approx::assert_relative_eq;
use std::f64::consts::PI;

/// Test trapezoidal rule with polynomial functions
#[test]
fn test_trapz_polynomial() -> Result<()> {
    // Test with linear function: ∫(2x + 1)dx from 0 to 2 = [x² + x] = 6
    let linear = |x: f64| 2.0 * x + 1.0;
    let result = trapz(linear, 0.0, 2.0, 100)?;
    assert_relative_eq!(result, 6.0, epsilon = 1e-12);
    
    // Test with quadratic: ∫x²dx from 0 to 3 = [x³/3] = 9
    let quadratic = |x: f64| x * x;
    let result = trapz(quadratic, 0.0, 3.0, 1000)?;
    assert_relative_eq!(result, 9.0, epsilon = 1e-5);
    
    // Test with cubic: ∫x³dx from -1 to 1 = [x⁴/4] = 0
    let cubic = |x: f64| x.powi(3);
    let result = trapz(cubic, -1.0, 1.0, 500)?;
    assert_relative_eq!(result, 0.0, epsilon = 1e-12);
    
    Ok(())
}

/// Test Simpson's rule with polynomial functions
#[test]
fn test_simpson_polynomial() -> Result<()> {
    // Simpson's rule is exact for polynomials up to degree 3
    
    // Linear: ∫(3x + 2)dx from 1 to 4 = [3x²/2 + 2x] = 28.5
    let linear = |x: f64| 3.0 * x + 2.0;
    let result = simpson(linear, 1.0, 4.0, 100)?; // Must be even for Simpson's
    assert_relative_eq!(result, 28.5, epsilon = 1e-12);
    
    // Quadratic: ∫(x² - 2x + 1)dx from 0 to 2 = [x³/3 - x² + x] = 2/3
    let quadratic = |x: f64| x * x - 2.0 * x + 1.0;
    let result = simpson(quadratic, 0.0, 2.0, 10)?;
    assert_relative_eq!(result, 2.0/3.0, epsilon = 1e-15);
    
    // Cubic: ∫x³dx from 0 to 2 = [x⁴/4] = 4
    let cubic = |x: f64| x.powi(3);
    let result = simpson(cubic, 0.0, 2.0, 10)?;
    assert_relative_eq!(result, 4.0, epsilon = 1e-15);
    
    Ok(())
}

/// Test Simpson's 3/8 rule with polynomial functions
#[test]
fn test_simpson38_polynomial() -> Result<()> {
    // Simpson's 3/8 rule is also exact for polynomials up to degree 3
    
    // Cubic: ∫(x³ - x²)dx from 0 to 3 = [x⁴/4 - x³/3] = 81/4 - 9 = 11.25
    let cubic = |x: f64| x.powi(3) - x * x;
    let result = simpson38(cubic, 0.0, 3.0, 99)?; // Must be multiple of 3
    assert_relative_eq!(result, 11.25, epsilon = 1e-15);
    
    // Quartic: ∫x⁴dx from -1 to 1 = [x⁵/5] = 0 (odd function)
    let quartic = |x: f64| x.powi(4);
    let odd_result = simpson38(|x: f64| x * quartic(x), -1.0, 1.0, 99)?;
    assert_relative_eq!(odd_result, 0.0, epsilon = 1e-14);
    
    Ok(())
}

/// Test Romberg integration with smooth functions
#[test]
fn test_romberg_smooth_functions() -> Result<()> {
    // Test with exponential: ∫e^x dx from 0 to 1 = [e^x] = e - 1
    let exponential = |x: f64| x.exp();
    let result = romberg(exponential, 0.0, 1.0, 1e-12, 10)?;
    let expected = 1.0_f64.exp() - 1.0;
    assert_relative_eq!(result, expected, epsilon = 1e-12);
    
    // Test with sine: ∫sin(x)dx from 0 to π = [-cos(x)] = 2
    let sine = |x: f64| x.sin();
    let result = romberg(sine, 0.0, PI, 1e-10, 10)?;
    assert_relative_eq!(result, 2.0, epsilon = 1e-10);
    
    // Test with cosine: ∫cos(x)dx from 0 to π/2 = [sin(x)] = 1
    let cosine = |x: f64| x.cos();
    let result = romberg(cosine, 0.0, PI/2.0, 1e-10, 10)?;
    assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    
    Ok(())
}

/// Test integration accuracy comparison
#[test]
fn test_integration_accuracy_comparison() -> Result<()> {
    // Test function: f(x) = x * sin(x) from 0 to π
    // Analytical: ∫x*sin(x)dx = sin(x) - x*cos(x) + C
    // From 0 to π: [sin(π) - π*cos(π)] - [sin(0) - 0*cos(0)] = 0 + π - 0 = π
    
    let f = |x: f64| x * x.sin();
    let expected = PI;
    let n = 1000;
    
    let trapz_result = trapz(f, 0.0, PI, n)?;
    let simpson_result = simpson(f, 0.0, PI, n)?;
    let simpson38_result = simpson38(f, 0.0, PI, n - (n % 3))?; // Adjust for multiple of 3
    let romberg_result = romberg(f, 0.0, PI, 1e-10, 15)?;
    
    // All methods should converge to π
    assert_relative_eq!(trapz_result, expected, epsilon = 1e-5);
    assert_relative_eq!(simpson_result, expected, epsilon = 1e-10);
    assert_relative_eq!(simpson38_result, expected, epsilon = 1e-10);
    assert_relative_eq!(romberg_result, expected, epsilon = 1e-12);
    
    // Higher-order methods should be more accurate
    let trapz_error = (trapz_result - expected).abs();
    let simpson_error = (simpson_result - expected).abs();
    let romberg_error = (romberg_result - expected).abs();
    
    assert!(simpson_error < trapz_error);
    assert!(romberg_error < simpson_error);
    
    Ok(())
}

/// Test integration with oscillatory functions
#[test]
fn test_oscillatory_functions() -> Result<()> {
    // High-frequency sine: ∫sin(10x)dx from 0 to 2π = [-cos(10x)/10] = 0
    let high_freq_sine = |x: f64| (10.0 * x).sin();
    let result = romberg(high_freq_sine, 0.0, 2.0 * PI, 1e-8, 15)?;
    assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    
    // Modulated sine: ∫sin(x)*cos(x)dx = ∫sin(2x)/2 dx from 0 to π = 0
    let modulated = |x: f64| x.sin() * x.cos();
    let result = romberg(modulated, 0.0, PI, 1e-8, 12)?;
    assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    
    Ok(())
}

/// Test integration with logarithmic and inverse functions
#[test]
fn test_special_functions() -> Result<()> {
    // ∫ln(x)dx from 1 to e = [x*ln(x) - x] = e*1 - e - (1*0 - 1) = 1
    let ln_func = |x: f64| x.ln();
    let result = romberg(ln_func, 1.0, 1.0_f64.exp(), 1e-10, 12)?;
    assert_relative_eq!(result, 1.0, epsilon = 1e-8);
    
    // ∫1/x dx from 1 to 2 = [ln(x)] = ln(2)
    let inverse = |x: f64| 1.0 / x;
    let result = romberg(inverse, 1.0, 2.0, 1e-10, 12)?;
    assert_relative_eq!(result, 2.0_f64.ln(), epsilon = 1e-8);
    
    // ∫sqrt(x)dx from 0 to 4 = [2x^(3/2)/3] = 16/3
    let sqrt_func = |x: f64| x.sqrt();
    let result = romberg(sqrt_func, 0.0, 4.0, 1e-10, 12)?;
    assert_relative_eq!(result, 16.0/3.0, epsilon = 1e-5);
    
    Ok(())
}

/// Test integration edge cases and error conditions
#[test]
fn test_integration_edge_cases() -> Result<()> {
    let f = |x: f64| x * x;
    
    // Test with zero interval
    let result = trapz(f, 1.0, 1.0, 10)?;
    assert_relative_eq!(result, 0.0, epsilon = 1e-15);
    
    // Test with reversed limits (should be negative)
    let forward = trapz(f, 0.0, 2.0, 100)?;
    let backward = trapz(f, 2.0, 0.0, 100)?;
    assert_relative_eq!(forward, -backward, epsilon = 1e-12);
    
    // Test with insufficient intervals
    assert!(simpson(f, 0.0, 1.0, 1).is_err()); // Need at least 2 intervals
    assert!(simpson38(f, 0.0, 1.0, 2).is_err()); // Need multiple of 3
    
    // Test Romberg convergence failure
    let discontinuous = |x: f64| if x < 0.5 { 0.0 } else { 1.0 };
    // This should either converge slowly or fail to meet tolerance
    let result = romberg(discontinuous, 0.0, 1.0, 1e-12, 5);
    // We expect this to either work with reduced accuracy or return an error
    match result {
        Ok(val) => assert!((val - 0.5).abs() < 1e-1), // Should be approximately 0.5
        Err(_) => (), // Convergence failure is acceptable for discontinuous functions
    }
    
    Ok(())
}

/// Test integration with negative functions and mixed signs
#[test]
fn test_negative_and_mixed_functions() -> Result<()> {
    // ∫(-x²)dx from 0 to 2 = [-x³/3] = -8/3
    let negative_quad = |x: f64| -x * x;
    let result = simpson(negative_quad, 0.0, 2.0, 100)?;
    assert_relative_eq!(result, -8.0/3.0, epsilon = 1e-12);
    
    // ∫sin(x)dx from 0 to 2π = [-cos(x)] = 0 (full period)
    let sine_full_period = |x: f64| x.sin();
    let result = romberg(sine_full_period, 0.0, 2.0 * PI, 1e-10, 12)?;
    assert_relative_eq!(result, 0.0, epsilon = 1e-8);
    
    // ∫(x² - 1)dx from -1 to 1 = [x³/3 - x] = (1/3 - 1) - (-1/3 + 1) = -4/3
    let mixed = |x: f64| x * x - 1.0;
    let result = simpson(mixed, -1.0, 1.0, 100)?;
    assert_relative_eq!(result, -4.0/3.0, epsilon = 1e-12);
    
    Ok(())
}

/// Test integration with rational functions
#[test]
fn test_rational_functions() -> Result<()> {
    // ∫1/(1+x²)dx from 0 to 1 = [arctan(x)] = π/4
    let arctan_derivative = |x: f64| 1.0 / (1.0 + x * x);
    let result = romberg(arctan_derivative, 0.0, 1.0, 1e-10, 12)?;
    assert_relative_eq!(result, PI/4.0, epsilon = 1e-8);
    
    // ∫x/(1+x²)dx from 0 to 1 = [ln(1+x²)/2] = ln(2)/2
    let log_derivative = |x: f64| x / (1.0 + x * x);
    let result = romberg(log_derivative, 0.0, 1.0, 1e-10, 12)?;
    assert_relative_eq!(result, 2.0_f64.ln() / 2.0, epsilon = 1e-8);
    
    Ok(())
}

/// Test integration with exponential and hyperbolic functions
#[test]
fn test_exponential_hyperbolic() -> Result<()> {
    // ∫e^(-x)dx from 0 to ∞ ≈ ∫e^(-x)dx from 0 to 10 ≈ 1
    let exp_decay = |x: f64| (-x).exp();
    let result = romberg(exp_decay, 0.0, 10.0, 1e-8, 12)?;
    assert_relative_eq!(result, 1.0, epsilon = 1e-4);
    
    // ∫sinh(x)dx from 0 to 1 = [cosh(x)] = cosh(1) - cosh(0) = cosh(1) - 1
    let sinh_func = |x: f64| x.sinh();
    let result = romberg(sinh_func, 0.0, 1.0, 1e-10, 12)?;
    let expected = 1.0_f64.cosh() - 1.0;
    assert_relative_eq!(result, expected, epsilon = 1e-8);
    
    // ∫cosh(x)dx from 0 to 1 = [sinh(x)] = sinh(1)
    let cosh_func = |x: f64| x.cosh();
    let result = romberg(cosh_func, 0.0, 1.0, 1e-10, 12)?;
    let expected = 1.0_f64.sinh();
    assert_relative_eq!(result, expected, epsilon = 1e-8);
    
    Ok(())
}

/// Test integration convergence properties
#[test]
fn test_integration_convergence() -> Result<()> {
    // Test that higher subdivisions improve accuracy
    let f = |x: f64| x.exp();
    let exact = 1.0_f64.exp() - 1.0;
    
    let n_values = [10, 100, 1000];
    let mut prev_error = f64::INFINITY;
    
    for &n in n_values.iter() {
        let result = trapz(f, 0.0, 1.0, n)?;
        let error = (result - exact).abs();
        
        // Error should decrease with more subdivisions
        if n > 10 {
            assert!(error < prev_error);
        }
        prev_error = error;
    }
    
    // Test Simpson's rule convergence
    let mut prev_error = f64::INFINITY;
    for &n in n_values.iter() {
        let result = simpson(f, 0.0, 1.0, n)?;
        let error = (result - exact).abs();
        
        if n > 10 {
            assert!(error < prev_error);
        }
        prev_error = error;
    }
    
    Ok(())
}

/// Test real-world integration applications
#[test]
fn test_real_world_applications() -> Result<()> {
    // Physics: Work done by variable force F(x) = kx² from x=0 to x=L
    let k = 2.0;
    let l = 3.0;
    let force = |x: f64| k * x * x;
    let work = simpson(force, 0.0, l, 100)?;
    let expected_work = k * l.powi(3) / 3.0; // ∫kx²dx = kx³/3
    assert_relative_eq!(work, expected_work, epsilon = 1e-12);
    
    // Statistics: Normal distribution CDF approximation
    // ∫(1/√(2π))e^(-x²/2)dx from -1 to 1 (should be ≈ 0.6827)
    let normal_pdf = |x: f64| (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp();
    let prob = romberg(normal_pdf, -1.0, 1.0, 1e-8, 12)?;
    assert!((prob - 0.6827).abs() < 0.001); // Within 1% of true value
    
    // Engineering: RMS value of AC signal
    // RMS = √(1/T ∫₀ᵀ f²(t)dt) for f(t) = A*sin(ωt)
    let amplitude = 10.0;
    let frequency = 2.0; // ω = 2 rad/s
    let period = 2.0 * PI / frequency;
    
    let signal_squared = |t: f64| {
        let signal = amplitude * (frequency * t).sin();
        signal * signal
    };
    
    let mean_square = simpson(signal_squared, 0.0, period, 1000)? / period;
    let rms = mean_square.sqrt();
    let expected_rms = amplitude / 2.0_f64.sqrt(); // A/√2 for sinusoidal
    
    assert_relative_eq!(rms, expected_rms, epsilon = 1e-3);
    
    Ok(())
}