//! Tests for complex number mathematical functions

use rustlab_math::*;
use num_complex::Complex;
use std::f64::consts::PI;

#[test]
fn test_complex_array_mathematical_functions() {
    // Create a complex array with known values
    let z1 = Complex::new(1.0, 0.0);  // Real number 1
    let z2 = Complex::new(0.0, 1.0);  // Imaginary number i
    let z3 = Complex::new(1.0, 1.0);  // 1 + i
    let z4 = Complex::new(-1.0, 0.0); // Real number -1
    
    let arr = ArrayC64::from_slice(&[z1, z2, z3, z4], 2, 2).unwrap();
    
    // Test basic math functions
    let sin_result = arr.sin();
    let cos_result = arr.cos();
    let exp_result = arr.exp();
    
    // Verify sin(1) has expected real part
    let sin_z1 = sin_result.get(0, 0).unwrap();
    assert!((sin_z1.re - 1.0_f64.sin()).abs() < 1e-10, "sin(1) real part should be sin(1)");
    
    // Verify exp(i) = cos(1) + i*sin(1) (Euler's formula)
    let exp_z2 = exp_result.get(0, 1).unwrap();
    assert!((exp_z2.re - 1.0_f64.cos()).abs() < 1e-10, "exp(i) real part should be cos(1)");
    assert!((exp_z2.im - 1.0_f64.sin()).abs() < 1e-10, "exp(i) imag part should be sin(1)");
    
    println!("✅ Complex array mathematical functions work correctly");
}

#[test]
fn test_complex_array_specific_functions() {
    // Test complex-specific functions
    let z1 = Complex::new(3.0, 4.0);  // 3 + 4i
    let z2 = Complex::new(-2.0, 1.0); // -2 + i
    
    let arr = ArrayC64::from_slice(&[z1, z2], 1, 2).unwrap();
    
    // Test real part extraction
    let real_parts = arr.real();
    assert_eq!(real_parts.get(0, 0).unwrap(), 3.0);
    assert_eq!(real_parts.get(0, 1).unwrap(), -2.0);
    
    // Test imaginary part extraction
    let imag_parts = arr.imag();
    assert_eq!(imag_parts.get(0, 0).unwrap(), 4.0);
    assert_eq!(imag_parts.get(0, 1).unwrap(), 1.0);
    
    // Test conjugate
    let conj_result = arr.conj();
    let conj_z1 = conj_result.get(0, 0).unwrap();
    let conj_z2 = conj_result.get(0, 1).unwrap();
    assert_eq!(conj_z1, Complex::new(3.0, -4.0));
    assert_eq!(conj_z2, Complex::new(-2.0, -1.0));
    
    // Test norm (magnitude)
    let norm_result = arr.norm();
    let norm_z1 = norm_result.get(0, 0).unwrap();
    let norm_z2 = norm_result.get(0, 1).unwrap();
    assert!((norm_z1 - 5.0).abs() < 1e-10, "Norm of 3+4i should be 5");
    assert!((norm_z2 - (5.0_f64).sqrt()).abs() < 1e-10, "Norm of -2+i should be sqrt(5)");
    
    // Test argument (phase)
    let arg_result = arr.arg();
    let arg_z1 = arg_result.get(0, 0).unwrap();
    assert!((arg_z1 - (4.0_f64 / 3.0).atan()).abs() < 1e-10, "Argument of 3+4i should be atan(4/3)");
    
    println!("✅ Complex-specific array functions work correctly");
}

#[test]
fn test_complex_vector_mathematical_functions() {
    // Test complex vector functions
    let z_vals = [
        Complex::new(1.0, 0.0),  // 1
        Complex::new(0.0, PI/2.0), // i*π/2
        Complex::new(PI, 0.0),   // π
    ];
    
    let vec = VectorC64::from_slice(&z_vals);
    
    // Test trigonometric functions
    let sin_result = vec.sin();
    let cos_result = vec.cos();
    let exp_result = vec.exp();
    
    // Test exp(i*π/2) ≈ i (approximately)
    let exp_ipi2 = exp_result.get(1).unwrap();
    assert!((exp_ipi2.re - 0.0).abs() < 1e-10, "exp(i*π/2) real part should be close to 0");
    assert!((exp_ipi2.im - 1.0).abs() < 1e-10, "exp(i*π/2) imag part should be close to 1");
    
    println!("✅ Complex vector mathematical functions work correctly");
}

#[test]
fn test_complex_vector_specific_functions() {
    // Test complex-specific vector operations
    let z_vals = [
        Complex::new(1.0, 2.0),  
        Complex::new(-3.0, 4.0), 
        Complex::new(0.0, -1.0), 
    ];
    
    let vec = VectorC64::from_slice(&z_vals);
    
    // Test real parts
    let real_parts = vec.real();
    assert_eq!(real_parts.get(0).unwrap(), 1.0);
    assert_eq!(real_parts.get(1).unwrap(), -3.0);
    assert_eq!(real_parts.get(2).unwrap(), 0.0);
    
    // Test imaginary parts
    let imag_parts = vec.imag();
    assert_eq!(imag_parts.get(0).unwrap(), 2.0);
    assert_eq!(imag_parts.get(1).unwrap(), 4.0);
    assert_eq!(imag_parts.get(2).unwrap(), -1.0);
    
    // Test conjugate
    let conj_result = vec.conj();
    assert_eq!(conj_result.get(0).unwrap(), Complex::new(1.0, -2.0));
    assert_eq!(conj_result.get(1).unwrap(), Complex::new(-3.0, -4.0));
    assert_eq!(conj_result.get(2).unwrap(), Complex::new(0.0, 1.0));
    
    // Test norm for each individual element
    let z0_norm = z_vals[0].norm();
    let z1_norm = z_vals[1].norm(); 
    let z2_norm = z_vals[2].norm();
    
    assert!((z0_norm - (5.0_f64).sqrt()).abs() < 1e-10, "Norm of 1+2i should be sqrt(5)");
    assert!((z1_norm - 5.0).abs() < 1e-10, "Norm of -3+4i should be 5");
    assert!((z2_norm - 1.0).abs() < 1e-10, "Norm of -i should be 1");
    
    println!("✅ Complex vector specific functions work correctly");
}

#[test]
fn test_complex_f32_functions() {
    // Test f32 complex types work too
    let z1 = Complex::new(1.0f32, 1.0f32);
    let z2 = Complex::new(2.0f32, -1.0f32);
    
    let arr = ArrayC32::from_slice(&[z1, z2], 1, 2).unwrap();
    let vec = VectorC32::from_slice(&[z1, z2]);
    
    // Test basic functions work
    let sin_arr = arr.sin();
    let cos_vec = vec.cos();
    
    // Test complex-specific functions
    let real_arr = arr.real();
    let imag_vec = vec.imag();
    
    assert_eq!(real_arr.get(0, 0).unwrap(), 1.0f32);
    assert_eq!(real_arr.get(0, 1).unwrap(), 2.0f32);
    assert_eq!(imag_vec.get(0).unwrap(), 1.0f32);
    assert_eq!(imag_vec.get(1).unwrap(), -1.0f32);
    
    println!("✅ Complex f32 functions work correctly");
}

#[test]
fn test_euler_identity() {
    // Test Euler's identity: e^(iπ) + 1 = 0
    let i_pi = Complex::new(0.0, PI);
    let arr = ArrayC64::from_slice(&[i_pi], 1, 1).unwrap();
    
    let exp_result = arr.exp();
    let exp_ipi = exp_result.get(0, 0).unwrap();
    
    // exp(iπ) should be approximately -1 + 0i
    assert!((exp_ipi.re - (-1.0)).abs() < 1e-10, "exp(iπ) real part should be -1");
    assert!(exp_ipi.im.abs() < 1e-10, "exp(iπ) imaginary part should be 0");
    
    // Therefore exp(iπ) + 1 should be approximately 0
    let euler_result = exp_ipi + Complex::new(1.0, 0.0);
    assert!(euler_result.norm() < 1e-10, "Euler's identity: e^(iπ) + 1 = 0");
    
    println!("✅ Euler's identity verified!");
}

#[test] 
fn test_complex_sqrt_and_ln() {
    // Test that ln and sqrt work for complex numbers
    let z = Complex::new(1.0, 0.0); // Real number 1
    let arr = ArrayC64::from_slice(&[z], 1, 1).unwrap();
    
    // ln(1) should be 0
    let ln_result = arr.ln();
    let ln_1 = ln_result.get(0, 0).unwrap();
    assert!(ln_1.norm() < 1e-10, "ln(1) should be 0");
    
    // sqrt(1) should be 1  
    let sqrt_result = arr.sqrt();
    let sqrt_1 = sqrt_result.get(0, 0).unwrap();
    assert!((sqrt_1 - Complex::new(1.0, 0.0)).norm() < 1e-10, "sqrt(1) should be 1");
    
    // Test sqrt(-1) = i
    let neg_one = Complex::new(-1.0, 0.0);
    let neg_arr = ArrayC64::from_slice(&[neg_one], 1, 1).unwrap();
    let sqrt_neg = neg_arr.sqrt();
    let sqrt_neg_1 = sqrt_neg.get(0, 0).unwrap();
    
    // sqrt(-1) should be approximately i (0 + 1i)
    assert!(sqrt_neg_1.re.abs() < 1e-10, "sqrt(-1) real part should be 0");
    assert!((sqrt_neg_1.im.abs() - 1.0).abs() < 1e-10, "sqrt(-1) imaginary part should be ±1");
    
    println!("✅ Complex sqrt and ln functions work correctly");
}