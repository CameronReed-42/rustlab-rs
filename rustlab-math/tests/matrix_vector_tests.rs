//! Tests for matrix-vector and vector-matrix operations with views

use rustlab_math::*;

#[test]
fn test_matrix_vector_multiplication() {
    let a = ArrayF64::from_slice(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ], 3, 3).unwrap();
    
    let v = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    
    // Matrix ^ Vector
    let result = &a ^ &v;
    assert_eq!(result.len(), 3);
    assert_eq!(result.get(0), Some(14.0)); // 1*1 + 2*2 + 3*3 = 14
    assert_eq!(result.get(1), Some(32.0)); // 4*1 + 5*2 + 6*3 = 32
    assert_eq!(result.get(2), Some(50.0)); // 7*1 + 8*2 + 9*3 = 50
}

#[test]
fn test_vector_matrix_multiplication() {
    let a = ArrayF64::from_slice(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ], 3, 3).unwrap();
    
    let v = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    
    // Vector ^ Matrix (row vector × matrix)
    let result = &v ^ &a;
    assert_eq!(result.len(), 3);
    assert_eq!(result.get(0), Some(30.0)); // 1*1 + 2*4 + 3*7 = 30
    assert_eq!(result.get(1), Some(36.0)); // 1*2 + 2*5 + 3*8 = 36
    assert_eq!(result.get(2), Some(42.0)); // 1*3 + 2*6 + 3*9 = 42
}

#[test]
fn test_matrix_vector_with_views() {
    let a = ArrayF64::from_slice(&[
        2.0, 3.0,
        4.0, 5.0
    ], 2, 2).unwrap();
    
    let v = VectorF64::from_slice(&[1.0, 2.0]);
    
    let view_a = a.view();
    let view_v = v.view();
    
    // ArrayView ^ VectorView
    let result = view_a ^ view_v;
    assert_eq!(result.get(0), Some(8.0));  // 2*1 + 3*2 = 8
    assert_eq!(result.get(1), Some(14.0)); // 4*1 + 5*2 = 14
    
    // ArrayView ^ Vector
    let result = view_a ^ &v;
    assert_eq!(result.get(0), Some(8.0));
    assert_eq!(result.get(1), Some(14.0));
    
    // Array ^ VectorView
    let result = &a ^ view_v;
    assert_eq!(result.get(0), Some(8.0));
    assert_eq!(result.get(1), Some(14.0));
}

#[test]
fn test_vector_matrix_with_views() {
    let a = ArrayF64::from_slice(&[
        2.0, 3.0,
        4.0, 5.0
    ], 2, 2).unwrap();
    
    let v = VectorF64::from_slice(&[1.0, 2.0]);
    
    let view_a = a.view();
    let view_v = v.view();
    
    // VectorView ^ ArrayView
    let result = view_v ^ view_a;
    assert_eq!(result.get(0), Some(10.0)); // 1*2 + 2*4 = 10
    assert_eq!(result.get(1), Some(13.0)); // 1*3 + 2*5 = 13
    
    // VectorView ^ Array
    let result = view_v ^ &a;
    assert_eq!(result.get(0), Some(10.0));
    assert_eq!(result.get(1), Some(13.0));
    
    // Vector ^ ArrayView
    let result = &v ^ view_a;
    assert_eq!(result.get(0), Some(10.0));
    assert_eq!(result.get(1), Some(13.0));
}

#[test]
fn test_complex_expressions_with_matrix_vector() {
    let a = ArrayF64::from_slice(&[
        1.0, 2.0,
        3.0, 4.0
    ], 2, 2).unwrap();
    
    let b = ArrayF64::from_slice(&[
        2.0, 0.0,
        0.0, 2.0
    ], 2, 2).unwrap();
    
    let v = VectorF64::from_slice(&[1.0, 1.0]);
    
    // Complex expression: (A + B) ^ v
    let result = (&a + &b) ^ &v;
    // (A+B) = [[3,2],[3,6]], then [[3,2],[3,6]] ^ [1,1] = [3+2, 3+6] = [5, 9]
    assert_eq!(result.get(0), Some(5.0));  // 3*1 + 2*1 = 5
    assert_eq!(result.get(1), Some(9.0));  // 3*1 + 6*1 = 9
    
    // Another complex expression: v ^ (A * 2.0)
    let result = &v ^ (&a * 2.0);
    assert_eq!(result.get(0), Some(8.0));  // 1*2 + 1*6 = 8
    assert_eq!(result.get(1), Some(12.0)); // 1*4 + 1*8 = 12
}

#[test]
fn test_ergonomic_math_first_syntax() {
    let a = ArrayF64::ones(3, 3);
    let v = VectorF64::ones(3);
    
    // Direct math-first syntax without explicit references
    let result1 = a.view() ^ v.view();
    assert_eq!(result1.len(), 3);
    assert_eq!(result1.get(0), Some(3.0)); // Sum of 3 ones
    
    // Mixed operations
    let result2 = &a ^ v.view();
    assert_eq!(result2.get(0), Some(3.0));
    
    let result3 = a.view() ^ &v;
    assert_eq!(result3.get(0), Some(3.0));
    
    // Chained operations
    let b = ArrayF64::ones(3, 3);
    let result4 = (a.view() + b.view()) ^ v.view();
    assert_eq!(result4.get(0), Some(6.0)); // (1+1)*1 + (1+1)*1 + (1+1)*1 = 6
}

#[test]
fn test_dimension_mismatch_errors() {
    let a = ArrayF64::ones(3, 4);
    let v = VectorF64::ones(3); // Wrong size for A ^ v
    
    // This should panic with dimension mismatch
    let result = std::panic::catch_unwind(|| {
        &a ^ &v
    });
    assert!(result.is_err());
    
    // Vector-matrix also checks dimensions
    let result = std::panic::catch_unwind(|| {
        &v ^ &a // v is length 3, A has 3 rows, but returns 4 columns
    });
    assert!(result.is_ok()); // This should work: 3-vector × 3x4 matrix = 4-vector
}

#[test]
fn test_vector_set_get_methods() {
    // Test the new set() method for vectors
    let mut v = VectorF64::zeros(5);
    
    // Test setting values
    assert!(v.set(0, 1.0).is_ok());
    assert!(v.set(2, 3.14).is_ok());
    assert!(v.set(4, -2.5).is_ok());
    
    // Test getting values
    assert_eq!(v.get(0), Some(1.0));
    assert_eq!(v.get(1), Some(0.0)); // Default zero
    assert_eq!(v.get(2), Some(3.14));
    assert_eq!(v.get(3), Some(0.0)); // Default zero
    assert_eq!(v.get(4), Some(-2.5));
    
    // Test bounds checking
    assert!(v.set(5, 99.0).is_err()); // Out of bounds
    assert!(v.set(100, 99.0).is_err()); // Way out of bounds
    assert_eq!(v.get(5), None); // Out of bounds
    
    // Test that the error message is reasonable
    let err = v.set(10, 1.0).unwrap_err();
    assert!(err.to_string().contains("Index out of bounds"));
    assert!(err.to_string().contains("10"));
    assert!(err.to_string().contains("5"));
}

#[test]
fn test_vector_set_consistency_with_arrays() {
    // Test that vector set() behaves similarly to array set()
    let mut v = VectorF64::zeros(3);
    let mut a = ArrayF64::zeros(1, 3);
    
    // Set same values
    v.set(0, 1.0).unwrap();
    v.set(1, 2.0).unwrap();
    v.set(2, 3.0).unwrap();
    
    a.set(0, 0, 1.0).unwrap();
    a.set(0, 1, 2.0).unwrap();
    a.set(0, 2, 3.0).unwrap();
    
    // Both should have same values
    assert_eq!(v.get(0), Some(1.0));
    assert_eq!(v.get(1), Some(2.0));
    assert_eq!(v.get(2), Some(3.0));
    
    assert_eq!(a.get(0, 0), Some(1.0));
    assert_eq!(a.get(0, 1), Some(2.0));
    assert_eq!(a.get(0, 2), Some(3.0));
    
    // Both should reject out-of-bounds
    assert!(v.set(3, 99.0).is_err());
    assert!(a.set(0, 3, 99.0).is_err());
}