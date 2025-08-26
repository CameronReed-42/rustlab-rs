//! Comprehensive unit tests for interpolation methods
//! 
//! Tests all interpolation methods against known analytical functions
//! to verify correctness and accuracy

use rustlab_math::{vec64, ArrayF64};
use rustlab_numerical::interpolation::*;
use rustlab_numerical::Result;
use approx::assert_relative_eq;
use std::f64::consts::PI;

/// Test linear interpolation with exact linear functions
#[test]
fn test_linear_interpolation_exact() -> Result<()> {
    // Linear function: y = 2x + 3
    let x = vec64::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let y = vec64::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0]);
    
    let interp = LinearInterpolator::new(x, y)?;
    
    // Test at grid points (should be exact)
    assert_relative_eq!(interp.eval(0.0)?, 3.0, epsilon = 1e-15);
    assert_relative_eq!(interp.eval(2.0)?, 7.0, epsilon = 1e-15);
    assert_relative_eq!(interp.eval(4.0)?, 11.0, epsilon = 1e-15);
    
    // Test at intermediate points (should be exact for linear function)
    assert_relative_eq!(interp.eval(0.5)?, 4.0, epsilon = 1e-15);
    assert_relative_eq!(interp.eval(1.5)?, 6.0, epsilon = 1e-15);
    assert_relative_eq!(interp.eval(2.5)?, 8.0, epsilon = 1e-15);
    
    Ok(())
}

/// Test polynomial interpolation with known polynomial functions
#[test]
fn test_lagrange_polynomial_exact() -> Result<()> {
    // Quadratic function: y = x² - 3x + 2
    let x = vec64::from_slice(&[-1.0, 0.0, 1.0, 2.0]);
    let y = vec64::from_slice(&[6.0, 2.0, 0.0, 0.0]); // Values of x² - 3x + 2
    
    let interp = LagrangeInterpolator::new(x, y)?;
    
    // Test at grid points
    for i in 0..4 {
        let xi = [-1.0, 0.0, 1.0, 2.0][i];
        let yi = [6.0, 2.0, 0.0, 0.0][i];
        assert_relative_eq!(interp.eval(xi)?, yi, epsilon = 1e-14);
    }
    
    // Test at intermediate points (polynomial should be exact)
    assert_relative_eq!(interp.eval(0.5)?, 0.25 - 1.5 + 2.0, epsilon = 1e-14); // 0.75
    assert_relative_eq!(interp.eval(1.5)?, 2.25 - 4.5 + 2.0, epsilon = 1e-14); // -0.25
    
    Ok(())
}

/// Test Newton interpolation consistency with Lagrange
#[test]
fn test_newton_lagrange_consistency() -> Result<()> {
    // Cubic function: y = x³ - 2x² + x + 1
    let x = vec64::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let mut y_vals = Vec::new();
    for i in 0..x.len() {
        let xi = x.get(i)?;
        y_vals.push(xi.powi(3) - 2.0 * xi.powi(2) + xi + 1.0);
    }
    let y = vec(&y_vals);
    
    let lagrange = LagrangeInterpolator::new(x.clone(), y.clone())?;
    let newton = NewtonInterpolator::new(x, y)?;
    
    // Both should give identical results
    let test_points = vec64::from_slice(&[0.5, 1.5, 2.5, 3.5]);
    for i in 0..test_points.len() {
        let xi = test_points.get(i)?;
        let lag_val = lagrange.eval(xi)?;
        let new_val = newton.eval(xi)?;
        assert_relative_eq!(lag_val, new_val, epsilon = 1e-12);
    }
    
    Ok(())
}

/// Test cubic spline interpolation properties
#[test]
fn test_cubic_spline_properties() -> Result<()> {
    // Test with a known cubic polynomial that splines should reproduce exactly
    let x = vec64::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let mut y_vals = Vec::new();
    for i in 0..x.len() {
        let xi = x.get(i)?;
        y_vals.push(xi.powi(3) - xi.powi(2) + 2.0 * xi + 1.0);
    }
    let y = vec(&y_vals);
    
    let spline = CubicSpline::new(x.clone(), y.clone(), BoundaryCondition::Natural)?;
    
    // Test at grid points (should be exact)
    for i in 0..x.len() {
        let xi = x.get(i)?;
        let yi = y.get(i)?;
        assert_relative_eq!(spline.eval(xi)?, yi, epsilon = 1e-14);
    }
    
    // Test derivatives at known points
    // For f(x) = x³ - x² + 2x + 1, f'(x) = 3x² - 2x + 2
    let test_x: f64 = 2.0;
    let expected_deriv = 3.0 * test_x.powi(2) - 2.0 * test_x + 2.0; // = 10
    let computed_deriv = spline.eval_derivative(test_x)?;
    assert_relative_eq!(computed_deriv, expected_deriv, epsilon = 1e-10);
    
    Ok(())
}

/// Test cubic spline boundary conditions
#[test]
fn test_cubic_spline_boundary_conditions() -> Result<()> {
    let x = vec64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
    let y = vec64::from_slice(&[1.0, 2.0, 5.0, 4.0]);
    
    // Test clamped boundary conditions
    let start_deriv = 1.5;
    let end_deriv = -2.0;
    let clamped = CubicSpline::new(
        x.clone(), 
        y.clone(), 
        BoundaryCondition::Clamped { 
            start_derivative: start_deriv, 
            end_derivative: end_deriv 
        }
    )?;
    
    // Check that boundary derivatives are satisfied
    assert_relative_eq!(clamped.eval_derivative(0.0)?, start_deriv, epsilon = 1e-12);
    assert_relative_eq!(clamped.eval_derivative(3.0)?, end_deriv, epsilon = 1e-12);
    
    // Test natural boundary conditions (second derivative = 0 at ends)
    let natural = CubicSpline::new(x, y, BoundaryCondition::Natural)?;
    assert_relative_eq!(natural.eval_second_derivative(0.0)?, 0.0, epsilon = 1e-12);
    assert_relative_eq!(natural.eval_second_derivative(3.0)?, 0.0, epsilon = 1e-12);
    
    Ok(())
}

/// Test 2D bilinear interpolation
#[test]
fn test_bilinear_interpolation() -> Result<()> {
    // Create a simple bilinear surface: f(x,y) = x + 2y
    let x = vec64::from_slice(&[0.0, 1.0, 2.0]);
    let y = vec64::from_slice(&[0.0, 1.0]);
    
    let z = ArrayF64::from_vec2d(vec![
        vec![0.0, 2.0],  // f(0,0)=0, f(0,1)=2
        vec![1.0, 3.0],  // f(1,0)=1, f(1,1)=3
        vec![2.0, 4.0]   // f(2,0)=2, f(2,1)=4
    ]).unwrap();
    
    let interp = BilinearInterpolator::new(x, y, z)?;
    
    // Test at grid points
    assert_relative_eq!(interp.eval(0.0, 0.0)?, 0.0, epsilon = 1e-15);
    assert_relative_eq!(interp.eval(1.0, 1.0)?, 3.0, epsilon = 1e-15);
    assert_relative_eq!(interp.eval(2.0, 0.0)?, 2.0, epsilon = 1e-15);
    
    // Test at intermediate points (should be exact for bilinear function)
    assert_relative_eq!(interp.eval(0.5, 0.5)?, 1.5, epsilon = 1e-15); // 0.5 + 2*0.5 = 1.5
    assert_relative_eq!(interp.eval(1.5, 0.5)?, 2.5, epsilon = 1e-15); // 1.5 + 2*0.5 = 2.5
    
    Ok(())
}

/// Test 2D bicubic interpolation
#[test]
fn test_bicubic_interpolation() -> Result<()> {
    // Create a 4x4 grid for bicubic interpolation
    let x = vec64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
    let y = vec64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
    
    // Use a polynomial surface: f(x,y) = x² + y²
    let mut z_data = vec![vec![0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let xi = i as f64;
            let yj = j as f64;
            z_data[i][j] = xi * xi + yj * yj;
        }
    }
    
    let z = ArrayF64::from_vec2d(vec![
        vec![z_data[0][0], z_data[0][1], z_data[0][2], z_data[0][3]],
        vec![z_data[1][0], z_data[1][1], z_data[1][2], z_data[1][3]],
        vec![z_data[2][0], z_data[2][1], z_data[2][2], z_data[2][3]],
        vec![z_data[3][0], z_data[3][1], z_data[3][2], z_data[3][3]]
    ]).unwrap();
    
    let interp = BicubicInterpolator::new(x, y, z)?;
    
    // Test at grid points (within valid domain for bicubic)
    assert_relative_eq!(interp.eval(1.0, 1.0)?, 2.0, epsilon = 1e-12); // 1² + 1² = 2
    assert_relative_eq!(interp.eval(2.0, 2.0)?, 8.0, epsilon = 1e-12); // 2² + 2² = 8
    
    // Test at intermediate point
    let test_val = interp.eval(1.5, 1.5)?;
    let expected = 1.5 * 1.5 + 1.5 * 1.5; // 4.5
    assert_relative_eq!(test_val, expected, epsilon = 1e-2); // Bicubic won't be exact for all polynomials
    
    Ok(())
}

/// Test extrapolation modes
#[test]
fn test_extrapolation_modes() -> Result<()> {
    let x = vec64::from_slice(&[1.0, 2.0, 3.0]);
    let y = vec64::from_slice(&[2.0, 4.0, 6.0]); // y = 2x
    
    // Test different extrapolation modes
    let linear_error = LinearInterpolator::new(x.clone(), y.clone())?;
    let linear_constant = linear_error.clone().with_extrapolation(ExtrapolationMode::Constant);
    let linear_linear = linear_error.clone().with_extrapolation(ExtrapolationMode::Linear);
    let linear_nan = linear_error.clone().with_extrapolation(ExtrapolationMode::NaN);
    
    // Error mode should fail outside domain
    assert!(linear_error.eval(0.5).is_err());
    assert!(linear_error.eval(3.5).is_err());
    
    // Constant mode should return boundary values
    assert_relative_eq!(linear_constant.eval(0.5)?, 2.0, epsilon = 1e-15); // First value
    assert_relative_eq!(linear_constant.eval(3.5)?, 6.0, epsilon = 1e-15); // Last value
    
    // Linear mode should extrapolate linearly
    assert_relative_eq!(linear_linear.eval(0.0)?, 0.0, epsilon = 1e-15); // 2*0 = 0
    assert_relative_eq!(linear_linear.eval(4.0)?, 8.0, epsilon = 1e-15); // 2*4 = 8
    
    // NaN mode should return NaN
    assert!(linear_nan.eval(0.5)?.is_nan());
    assert!(linear_nan.eval(3.5)?.is_nan());
    
    Ok(())
}

/// Test error conditions and edge cases
#[test]
fn test_interpolation_edge_cases() -> Result<()> {
    // Test insufficient data
    let x_short = vec64::from_slice(&[1.0]);
    let y_short = vec64::from_slice(&[2.0]);
    assert!(LinearInterpolator::new(x_short, y_short).is_err());
    
    // Test dimension mismatch
    let x = vec64::from_slice(&[1.0, 2.0, 3.0]);
    let y = vec64::from_slice(&[1.0, 2.0]); // Wrong size
    assert!(LinearInterpolator::new(x, y).is_err());
    
    // Test non-monotonic data for splines
    let x_non_mono = vec64::from_slice(&[1.0, 3.0, 2.0, 4.0]);
    let y_mono = vec64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert!(CubicSpline::new(x_non_mono, y_mono, BoundaryCondition::Natural).is_err());
    
    // Test duplicate points for polynomial interpolation
    let x_dup = vec64::from_slice(&[1.0, 2.0, 2.0, 3.0]);
    let y_dup = vec64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert!(LagrangeInterpolator::new(x_dup.clone(), y_dup.clone()).is_err());
    assert!(NewtonInterpolator::new(x_dup, y_dup).is_err());
    
    Ok(())
}

/// Test interpolation with real-world data characteristics
#[test]
fn test_interpolation_with_noisy_data() -> Result<()> {
    // Simulate noisy measurement data
    let x = linspace(0.0, 2.0 * PI, 20);
    let mut y_vals = Vec::new();
    
    for i in 0..x.len() {
        let xi = x.get(i)?;
        let clean = xi.sin();
        let noise = 0.01 * (2.0 * (i as f64 * 0.123).cos()); // Small deterministic "noise"
        y_vals.push(clean + noise);
    }
    let y = vec(&y_vals);
    
    // Test that interpolators handle noisy data gracefully
    let linear = LinearInterpolator::new(x.clone(), y.clone())?;
    let spline = CubicSpline::new(x.clone(), y.clone(), BoundaryCondition::Natural)?;
    
    // Evaluate at a test point
    let test_x = PI / 2.0;
    let linear_val = linear.eval(test_x)?;
    let spline_val = spline.eval(test_x)?;
    
    // Should be roughly close to sin(π/2) = 1
    assert!((linear_val - 1.0).abs() < 0.1);
    assert!((spline_val - 1.0).abs() < 0.1);
    
    // Spline should be differentiable
    let deriv = spline.eval_derivative(test_x)?;
    assert!(deriv.is_finite());
    
    Ok(())
}

/// Test interpolation accuracy for smooth functions
#[test]
fn test_interpolation_accuracy() -> Result<()> {
    // Test with exp(x) function over [0, 1]
    let n_points = 10;
    let x = linspace(0.0, 1.0, n_points);
    let mut y_vals = Vec::new();
    
    for i in 0..x.len() {
        y_vals.push(x.get(i)?.exp());
    }
    let y = vec(&y_vals);
    
    let linear = LinearInterpolator::new(x.clone(), y.clone())?;
    let spline = CubicSpline::new(x, y, BoundaryCondition::Natural)?;
    
    // Test accuracy at intermediate points
    let test_points = vec64::from_slice(&[0.15, 0.35, 0.65, 0.85]);
    
    for i in 0..test_points.len() {
        let xi = test_points.get(i)?;
        let exact = xi.exp();
        let linear_val = linear.eval(xi)?;
        let spline_val = spline.eval(xi)?;
        
        let linear_error = (linear_val - exact).abs() / exact;
        let spline_error = (spline_val - exact).abs() / exact;
        
        // Spline should be more accurate than linear for smooth functions
        assert!(spline_error < linear_error || spline_error < 1e-3);
        
        // Both should have reasonable accuracy
        assert!(linear_error < 0.01); // 1% error
        assert!(spline_error < 0.001); // 0.1% error
    }
    
    Ok(())
}

/// Test convenience functions
#[test]
fn test_convenience_functions() -> Result<()> {
    let x = vec64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
    let y = vec64::from_slice(&[0.0, 1.0, 4.0, 9.0]); // y = x²
    let xi = vec64::from_slice(&[0.5, 1.5, 2.5]);
    
    // Test convenience functions for spline interpolation
    let yi = interp1d_cubic_spline(&x, &y, &xi)?;
    
    // Verify results
    for i in 0..xi.len() {
        let x_val = xi.get(i)?;
        let y_val = yi.get(i)?;
        let expected = x_val * x_val;
        
        // Natural spline won't be exact for quadratic, but should be close
        assert!((y_val - expected).abs() < 0.1);
    }
    
    Ok(())
}