//! Comprehensive unit tests for numerical differentiation methods
//! 
//! Tests all differentiation methods against known analytical derivatives
//! to verify correctness and accuracy

use rustlab_numerical::differentiation::*;
use rustlab_numerical::Result;
use approx::assert_relative_eq;
use std::f64::consts::{PI, E};
use num_complex::Complex64;

/// Test forward differences with polynomial functions
#[test]
fn test_forward_diff_polynomial() -> Result<()> {
    // Test with linear function: f(x) = 2x + 3, f'(x) = 2
    let linear = |x: f64| 2.0 * x + 3.0;
    let result = forward_diff(linear, 1.0, 1e-5, 1)?;
    assert_relative_eq!(result, 2.0, epsilon = 1e-10);
    
    // Test with quadratic: f(x) = x², f'(x) = 2x
    let quadratic = |x: f64| x * x;
    let result = forward_diff(quadratic, 3.0, 1e-6, 2)?;
    assert_relative_eq!(result, 6.0, epsilon = 1e-8);
    
    // Test with cubic: f(x) = x³, f'(x) = 3x²
    let cubic = |x: f64| x.powi(3);
    let result = forward_diff(cubic, 2.0, 1e-6, 3)?;
    assert_relative_eq!(result, 12.0, epsilon = 1e-6);
    
    // Test with quartic: f(x) = x⁴, f'(x) = 4x³
    let quartic = |x: f64| x.powi(4);
    let result = forward_diff(quartic, 1.5, 1e-6, 4)?;
    assert_relative_eq!(result, 4.0 * 1.5_f64.powi(3), epsilon = 1e-4);
    
    Ok(())
}

/// Test backward differences with polynomial functions
#[test]
fn test_backward_diff_polynomial() -> Result<()> {
    // Test with linear function: f(x) = 3x - 1, f'(x) = 3
    let linear = |x: f64| 3.0 * x - 1.0;
    let result = backward_diff(linear, 2.0, 1e-5, 1)?;
    assert_relative_eq!(result, 3.0, epsilon = 1e-10);
    
    // Test with quadratic: f(x) = x² + x, f'(x) = 2x + 1
    let quadratic = |x: f64| x * x + x;
    let result = backward_diff(quadratic, 4.0, 1e-6, 2)?;
    assert_relative_eq!(result, 9.0, epsilon = 1e-8);
    
    // Test with cubic: f(x) = 2x³ - x, f'(x) = 6x² - 1
    let cubic = |x: f64| 2.0 * x.powi(3) - x;
    let result = backward_diff(cubic, 1.0, 1e-6, 3)?;
    assert_relative_eq!(result, 5.0, epsilon = 1e-6);
    
    Ok(())
}

/// Test central differences with polynomial functions
#[test]
fn test_central_diff_polynomial() -> Result<()> {
    // Central differences should be more accurate for smooth functions
    
    // Test with quadratic: f(x) = x² - 2x + 1, f'(x) = 2x - 2
    let quadratic = |x: f64| x * x - 2.0 * x + 1.0;
    let result = central_diff(quadratic, 3.0, 1e-6, 2)?;
    assert_relative_eq!(result, 4.0, epsilon = 1e-8);
    
    // Test with cubic: f(x) = x³ + x² - x, f'(x) = 3x² + 2x - 1
    let cubic = |x: f64| x.powi(3) + x * x - x;
    let result = central_diff(cubic, 2.0, 1e-6, 4)?;
    let expected = 3.0 * 4.0 + 2.0 * 2.0 - 1.0; // = 15
    assert_relative_eq!(result, expected, epsilon = 1e-6);
    
    // Test with quartic: f(x) = x⁴ - 3x², f'(x) = 4x³ - 6x
    let quartic = |x: f64| x.powi(4) - 3.0 * x * x;
    let result = central_diff(quartic, 1.5, 1e-6, 6)?;
    let expected = 4.0 * 1.5_f64.powi(3) - 6.0 * 1.5; // = 13.5 - 9 = 4.5
    assert_relative_eq!(result, expected, epsilon = 1e-8);
    
    Ok(())
}

/// Test Richardson extrapolation with smooth functions
#[test]
fn test_richardson_extrapolation() -> Result<()> {
    // Test with exponential: f(x) = e^x, f'(x) = e^x
    let exponential = |x: f64| x.exp();
    let x = 1.0;
    let result = richardson_extrapolation(exponential, x, 1e-4, central_diff, 4)?;
    assert_relative_eq!(result, E, epsilon = 1e-10);
    
    // Test with sine: f(x) = sin(x), f'(x) = cos(x)
    let sine = |x: f64| x.sin();
    let x = PI / 4.0;
    let result = richardson_extrapolation(sine, x, 1e-4, central_diff, 4)?;
    let expected = x.cos(); // cos(π/4) = √2/2
    assert_relative_eq!(result, expected, epsilon = 1e-6);
    
    // Test with cosine: f(x) = cos(x), f'(x) = -sin(x)
    let cosine = |x: f64| x.cos();
    let x = PI / 3.0;
    let result = richardson_extrapolation(cosine, x, 1e-4, central_diff, 4)?;
    let expected = -x.sin(); // -sin(π/3) = -√3/2
    assert_relative_eq!(result, expected, epsilon = 1e-6);
    
    Ok(())
}

/// Test complex-step differentiation
#[test]
fn test_complex_step_diff() -> Result<()> {
    // Complex-step should give machine precision for analytic functions
    
    // Test with exponential: f(x) = e^x, f'(x) = e^x
    let exponential = |z: Complex64| z.exp();
    let x = 0.5;
    let result = complex_step_diff(exponential, x, 1e-10)?;
    assert_relative_eq!(result, x.exp(), epsilon = 1e-14);
    
    // Test with polynomial: f(x) = x⁴ + 2x³ - x + 1, f'(x) = 4x³ + 6x² - 1
    let polynomial = |z: Complex64| z.powi(4) + 2.0 * z.powi(3) - z + Complex64::new(1.0, 0.0);
    let x = 1.0;
    let result = complex_step_diff(polynomial, x, 1e-10)?;
    let expected = 4.0 + 6.0 - 1.0; // = 9
    assert_relative_eq!(result, expected, epsilon = 1e-14);
    
    // Test with trigonometric: f(x) = sin(x) * cos(x), f'(x) = cos²(x) - sin²(x) = cos(2x)
    let trig = |z: Complex64| z.sin() * z.cos();
    let x = PI / 6.0;
    let result = complex_step_diff(trig, x, 1e-10)?;
    let expected = (2.0 * x).cos();
    assert_relative_eq!(result, expected, epsilon = 1e-14);
    
    Ok(())
}

/// Test differentiation accuracy comparison
#[test]
fn test_differentiation_accuracy_comparison() -> Result<()> {
    // Test function: f(x) = x * e^x, f'(x) = e^x * (1 + x)
    let f = |x: f64| x * x.exp();
    let x: f64 = 0.5;
    let expected = x.exp() * (1.0 + x);
    let h = 1e-6;
    
    let forward_result = forward_diff(f, x, h, 2)?;
    let backward_result = backward_diff(f, x, h, 2)?;
    let central_result = central_diff(f, x, h, 2)?;
    let richardson_result = richardson_extrapolation(f, x, h, central_diff, 4)?;
    
    // For complex step, need Complex64 function
    let f_complex = |z: num_complex::Complex64| z * z.exp();
    let complex_result = complex_step_diff(f_complex, x, 1e-10)?;
    
    // All methods should converge to the true derivative
    assert_relative_eq!(forward_result, expected, epsilon = 1e-6);
    assert_relative_eq!(backward_result, expected, epsilon = 1e-6);
    assert_relative_eq!(central_result, expected, epsilon = 1e-10);
    assert_relative_eq!(richardson_result, expected, epsilon = 1e-8);
    assert_relative_eq!(complex_result, expected, epsilon = 1e-14);
    
    // Calculate errors
    let forward_error = (forward_result - expected).abs();
    let backward_error = (backward_result - expected).abs();
    let central_error = (central_result - expected).abs();
    let richardson_error = (richardson_result - expected).abs();
    let complex_error = (complex_result - expected).abs();
    
    // Higher-order methods should generally be more accurate
    // Note: Richardson extrapolation may not always be better due to round-off errors
    assert!(central_error < forward_error);
    assert!(central_error < backward_error);
    assert!(complex_error < 1e-10); // Complex step should be very accurate
    
    Ok(())
}

/// Test differentiation with trigonometric functions
#[test]
fn test_trigonometric_functions() -> Result<()> {
    let h = 1e-8;
    
    // Test sin(x): f'(x) = cos(x)
    let sine = |x: f64| x.sin();
    let x = PI / 4.0;
    let result = central_diff(sine, x, h, 4)?;
    assert_relative_eq!(result, x.cos(), epsilon = 1e-8);
    
    // Test cos(x): f'(x) = -sin(x)
    let cosine = |x: f64| x.cos();
    let x = PI / 6.0;
    let result = central_diff(cosine, x, h, 4)?;
    assert_relative_eq!(result, -x.sin(), epsilon = 1e-8);
    
    // Test tan(x): f'(x) = sec²(x) = 1/cos²(x)
    let tangent = |x: f64| x.tan();
    let x = PI / 8.0;
    let result = central_diff(tangent, x, h, 4)?;
    let expected = 1.0 / (x.cos() * x.cos());
    assert_relative_eq!(result, expected, epsilon = 1e-8);
    
    Ok(())
}

/// Test differentiation with logarithmic and exponential functions
#[test]
fn test_logarithmic_exponential() -> Result<()> {
    let h = 1e-8;
    
    // Test ln(x): f'(x) = 1/x
    let log_func = |x: f64| x.ln();
    let x = 2.0;
    let result = central_diff(log_func, x, h, 4)?;
    assert_relative_eq!(result, 1.0 / x, epsilon = 1e-8);
    
    // Test e^x: f'(x) = e^x
    let exp_func = |x: f64| x.exp();
    let x = 0.5;
    let result = central_diff(exp_func, x, h, 4)?;
    assert_relative_eq!(result, x.exp(), epsilon = 1e-6);
    
    // Test x^n: f'(x) = n*x^(n-1)
    let power_func = |x: f64| x.powf(2.5);
    let x = 3.0;
    let result = central_diff(power_func, x, h, 4)?;
    let expected = 2.5 * x.powf(1.5);
    assert_relative_eq!(result, expected, epsilon = 1e-8);
    
    Ok(())
}

/// Test differentiation with inverse trigonometric functions
#[test]
fn test_inverse_trigonometric() -> Result<()> {
    let h = 1e-8;
    
    // Test arcsin(x): f'(x) = 1/√(1-x²)
    let arcsin_func = |x: f64| x.asin();
    let x = 0.5;
    let result = central_diff(arcsin_func, x, h, 4)?;
    let expected = 1.0 / (1.0 - x * x).sqrt();
    assert_relative_eq!(result, expected, epsilon = 1e-8);
    
    // Test arctan(x): f'(x) = 1/(1+x²)
    let arctan_func = |x: f64| x.atan();
    let x = 1.0;
    let result = central_diff(arctan_func, x, h, 4)?;
    let expected = 1.0 / (1.0 + x * x);
    assert_relative_eq!(result, expected, epsilon = 1e-8);
    
    Ok(())
}

/// Test differentiation edge cases and error conditions
#[test]
fn test_differentiation_edge_cases() -> Result<()> {
    let f = |x: f64| x * x;
    
    // Test with very small step size (should work but may have precision issues)
    let result = central_diff(f, 1.0, 1e-10, 2)?;
    assert_relative_eq!(result, 2.0, epsilon = 1e-4);
    
    // Test with zero step size (should error)
    assert!(forward_diff(f, 1.0, 0.0, 1).is_err());
    assert!(backward_diff(f, 1.0, 0.0, 1).is_err());
    assert!(central_diff(f, 1.0, 0.0, 2).is_err());
    
    // Test with invalid order (should error)
    assert!(forward_diff(f, 1.0, 1e-6, 0).is_err());
    assert!(central_diff(f, 1.0, 1e-6, 1).is_err()); // Central diff needs even order >= 2
    assert!(central_diff(f, 1.0, 1e-6, 7).is_err()); // Order too high
    
    Ok(())
}

/// Test differentiation with step size optimization
#[test]
fn test_step_size_optimization() -> Result<()> {
    // Test that appropriate step sizes give good results
    let f = |x: f64| (x * x * x).sin(); // f'(x) = 3x² * cos(x³)
    let x: f64 = 1.0;
    let expected = 3.0 * x * x * (x * x * x).cos();
    
    // Test different step sizes
    let step_sizes = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12];
    let mut best_error = f64::INFINITY;
    
    for &h in step_sizes.iter() {
        let result = central_diff(f, x, h, 4)?;
        let error = (result - expected).abs();
        
        // For well-conditioned problems, smaller h should generally be better
        // until we hit machine precision limits
        if h >= 1e-10 {
            best_error = best_error.min(error);
        }
        
        // All should be reasonably accurate
        assert!(error < 1e-4);
    }
    
    // Best error should be quite small
    assert!(best_error < 1e-10);
    
    Ok(())
}

/// Test differentiation with composite functions
#[test]
fn test_composite_functions() -> Result<()> {
    let h = 1e-8;
    
    // Test chain rule: f(x) = sin(x²), f'(x) = 2x * cos(x²)
    let composite1 = |x: f64| (x * x).sin();
    let x = 0.5;
    let result = central_diff(composite1, x, h, 4)?;
    let expected = 2.0 * x * (x * x).cos();
    assert_relative_eq!(result, expected, epsilon = 1e-8);
    
    // Test product rule: f(x) = x * sin(x), f'(x) = sin(x) + x * cos(x)
    let product = |x: f64| x * x.sin();
    let x: f64 = PI / 3.0;
    let result = central_diff(product, x, h, 4)?;
    let expected = x.sin() + x * x.cos();
    assert_relative_eq!(result, expected, epsilon = 1e-6);
    
    // Test quotient rule: f(x) = sin(x)/x, f'(x) = (x*cos(x) - sin(x))/x²
    let quotient = |x: f64| x.sin() / x;
    let x: f64 = 1.0;
    let result = central_diff(quotient, x, h, 4)?;
    let expected = (x * x.cos() - x.sin()) / (x * x);
    assert_relative_eq!(result, expected, epsilon = 1e-8);
    
    Ok(())
}

/// Test differentiation with discontinuous functions
#[test]
fn test_discontinuous_functions() -> Result<()> {
    // Test step function (derivative should be very large or infinite at discontinuity)
    let step = |x: f64| if x < 0.0 { 0.0 } else { 1.0 };
    
    // Away from discontinuity, derivative should be near zero
    let result_left = central_diff(step, -0.1, 1e-6, 2)?;
    let result_right = central_diff(step, 0.1, 1e-6, 2)?;
    
    assert!(result_left.abs() < 1e-6);
    assert!(result_right.abs() < 1e-6);
    
    // At discontinuity, result will depend on step size but should be large
    let result_at_disc = central_diff(step, 0.0, 1e-6, 2)?;
    assert!(result_at_disc.abs() > 1e4);
    
    Ok(())
}

/// Test real-world differentiation applications
#[test]
fn test_real_world_applications() -> Result<()> {
    // Physics: Velocity from position
    // Position: s(t) = 5t² + 2t + 1, Velocity: v(t) = ds/dt = 10t + 2
    let position = |t: f64| 5.0 * t * t + 2.0 * t + 1.0;
    let t = 2.0;
    let velocity = central_diff(position, t, 1e-8, 4)?;
    let expected_velocity = 10.0 * t + 2.0;
    assert_relative_eq!(velocity, expected_velocity, epsilon = 1e-6);
    
    // Economics: Marginal cost from total cost
    // Total cost: C(q) = 100 + 50q + 0.1q², Marginal cost: MC(q) = dC/dq = 50 + 0.2q
    let total_cost = |q: f64| 100.0 + 50.0 * q + 0.1 * q * q;
    let q = 10.0;
    let marginal_cost = central_diff(total_cost, q, 1e-8, 4)?;
    let expected_mc = 50.0 + 0.2 * q;
    assert_relative_eq!(marginal_cost, expected_mc, epsilon = 1e-4);
    
    // Engineering: Heat transfer rate
    // Temperature: T(x) = T₀ * e^(-kx), Heat flux: q = -k * dT/dx = k² * T₀ * e^(-kx)
    let t0 = 100.0;
    let k = 0.1;
    let temperature = |x: f64| t0 * (-k * x).exp();
    let x = 5.0;
    let temp_gradient = central_diff(temperature, x, 1e-8, 4)?;
    let expected_gradient = -k * t0 * (-k * x).exp();
    assert_relative_eq!(temp_gradient, expected_gradient, epsilon = 1e-6);
    
    Ok(())
}