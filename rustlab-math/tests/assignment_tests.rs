//! Comprehensive tests for assignment operators with automatic SIMD

use rustlab_math::*;

#[test]
fn test_array_assignment_operators() {
    let mut a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let b = ArrayF64::from_slice(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
    
    // Test += with reference
    a += &b;
    assert_eq!(a.get(0, 0), Some(3.0)); // 1 + 2 = 3
    assert_eq!(a.get(0, 1), Some(5.0)); // 2 + 3 = 5
    assert_eq!(a.get(1, 0), Some(7.0)); // 3 + 4 = 7
    assert_eq!(a.get(1, 1), Some(9.0)); // 4 + 5 = 9
    
    // Test -= with owned
    let c = ArrayF64::from_slice(&[1.0, 1.0, 1.0, 1.0], 2, 2).unwrap();
    a -= c;
    assert_eq!(a.get(0, 0), Some(2.0)); // 3 - 1 = 2
    assert_eq!(a.get(0, 1), Some(4.0)); // 5 - 1 = 4
    assert_eq!(a.get(1, 0), Some(6.0)); // 7 - 1 = 6
    assert_eq!(a.get(1, 1), Some(8.0)); // 9 - 1 = 8
    
    // Test *= element-wise
    let d = ArrayF64::from_slice(&[2.0, 2.0, 2.0, 2.0], 2, 2).unwrap();
    a *= &d;
    assert_eq!(a.get(0, 0), Some(4.0));  // 2 * 2 = 4
    assert_eq!(a.get(0, 1), Some(8.0));  // 4 * 2 = 8
    assert_eq!(a.get(1, 0), Some(12.0)); // 6 * 2 = 12
    assert_eq!(a.get(1, 1), Some(16.0)); // 8 * 2 = 16
    
    // Test /= element-wise
    a /= &d;
    assert_eq!(a.get(0, 0), Some(2.0)); // 4 / 2 = 2
    assert_eq!(a.get(0, 1), Some(4.0)); // 8 / 2 = 4
    assert_eq!(a.get(1, 0), Some(6.0)); // 12 / 2 = 6
    assert_eq!(a.get(1, 1), Some(8.0)); // 16 / 2 = 8
}

#[test]
fn test_array_scalar_assignment() {
    let mut a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    
    // Test += scalar
    a += 2.0;
    assert_eq!(a.get(0, 0), Some(3.0)); // 1 + 2 = 3
    assert_eq!(a.get(0, 1), Some(4.0)); // 2 + 2 = 4
    assert_eq!(a.get(1, 0), Some(5.0)); // 3 + 2 = 5
    assert_eq!(a.get(1, 1), Some(6.0)); // 4 + 2 = 6
    
    // Test -= scalar
    a -= 1.0;
    assert_eq!(a.get(0, 0), Some(2.0)); // 3 - 1 = 2
    assert_eq!(a.get(0, 1), Some(3.0)); // 4 - 1 = 3
    assert_eq!(a.get(1, 0), Some(4.0)); // 5 - 1 = 4
    assert_eq!(a.get(1, 1), Some(5.0)); // 6 - 1 = 5
    
    // Test *= scalar (automatic SIMD optimization)
    a *= 3.0;
    assert_eq!(a.get(0, 0), Some(6.0));  // 2 * 3 = 6
    assert_eq!(a.get(0, 1), Some(9.0));  // 3 * 3 = 9
    assert_eq!(a.get(1, 0), Some(12.0)); // 4 * 3 = 12
    assert_eq!(a.get(1, 1), Some(15.0)); // 5 * 3 = 15
    
    // Test /= scalar (automatic SIMD optimization via reciprocal)
    a /= 3.0;
    assert_eq!(a.get(0, 0), Some(2.0)); // 6 / 3 = 2
    assert_eq!(a.get(0, 1), Some(3.0)); // 9 / 3 = 3
    assert_eq!(a.get(1, 0), Some(4.0)); // 12 / 3 = 4
    assert_eq!(a.get(1, 1), Some(5.0)); // 15 / 3 = 5
}

#[test]
fn test_vector_assignment_operators() {
    let mut v = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let u = VectorF64::from_slice(&[2.0, 3.0, 4.0, 5.0]);
    
    // Test += with reference
    v += &u;
    assert_eq!(v.get(0), Some(3.0)); // 1 + 2 = 3
    assert_eq!(v.get(1), Some(5.0)); // 2 + 3 = 5
    assert_eq!(v.get(2), Some(7.0)); // 3 + 4 = 7
    assert_eq!(v.get(3), Some(9.0)); // 4 + 5 = 9
    
    // Test -= with owned
    let w = VectorF64::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    v -= w;
    assert_eq!(v.get(0), Some(2.0)); // 3 - 1 = 2
    assert_eq!(v.get(1), Some(4.0)); // 5 - 1 = 4
    assert_eq!(v.get(2), Some(6.0)); // 7 - 1 = 6
    assert_eq!(v.get(3), Some(8.0)); // 9 - 1 = 8
    
    // Test *= element-wise
    let multiplier = VectorF64::from_slice(&[2.0, 2.0, 2.0, 2.0]);
    v *= &multiplier;
    assert_eq!(v.get(0), Some(4.0));  // 2 * 2 = 4
    assert_eq!(v.get(1), Some(8.0));  // 4 * 2 = 8
    assert_eq!(v.get(2), Some(12.0)); // 6 * 2 = 12
    assert_eq!(v.get(3), Some(16.0)); // 8 * 2 = 16
    
    // Test /= element-wise
    v /= &multiplier;
    assert_eq!(v.get(0), Some(2.0)); // 4 / 2 = 2
    assert_eq!(v.get(1), Some(4.0)); // 8 / 2 = 4
    assert_eq!(v.get(2), Some(6.0)); // 12 / 2 = 6
    assert_eq!(v.get(3), Some(8.0)); // 16 / 2 = 8
}

#[test]
fn test_vector_scalar_assignment() {
    let mut v = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    
    // Test += scalar
    v += 5.0;
    assert_eq!(v.get(0), Some(6.0));  // 1 + 5 = 6
    assert_eq!(v.get(1), Some(7.0));  // 2 + 5 = 7
    assert_eq!(v.get(2), Some(8.0));  // 3 + 5 = 8
    assert_eq!(v.get(3), Some(9.0));  // 4 + 5 = 9
    
    // Test -= scalar
    v -= 2.0;
    assert_eq!(v.get(0), Some(4.0)); // 6 - 2 = 4
    assert_eq!(v.get(1), Some(5.0)); // 7 - 2 = 5
    assert_eq!(v.get(2), Some(6.0)); // 8 - 2 = 6
    assert_eq!(v.get(3), Some(7.0)); // 9 - 2 = 7
    
    // Test *= scalar (automatic SIMD optimization)
    v *= 2.5;
    assert_eq!(v.get(0), Some(10.0)); // 4 * 2.5 = 10
    assert_eq!(v.get(1), Some(12.5)); // 5 * 2.5 = 12.5
    assert_eq!(v.get(2), Some(15.0)); // 6 * 2.5 = 15
    assert_eq!(v.get(3), Some(17.5)); // 7 * 2.5 = 17.5
    
    // Test /= scalar (automatic SIMD optimization via reciprocal)
    v /= 2.5;
    assert_eq!(v.get(0), Some(4.0)); // 10 / 2.5 = 4
    assert_eq!(v.get(1), Some(5.0)); // 12.5 / 2.5 = 5
    assert_eq!(v.get(2), Some(6.0)); // 15 / 2.5 = 6
    assert_eq!(v.get(3), Some(7.0)); // 17.5 / 2.5 = 7
}

#[test]
fn test_array_with_views_assignment() {
    let mut a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let b = ArrayF64::from_slice(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
    let view_b = b.view();
    
    // Test Array += ArrayView
    a += view_b;
    assert_eq!(a.get(0, 0), Some(3.0)); // 1 + 2 = 3
    assert_eq!(a.get(0, 1), Some(5.0)); // 2 + 3 = 5
    assert_eq!(a.get(1, 0), Some(7.0)); // 3 + 4 = 7
    assert_eq!(a.get(1, 1), Some(9.0)); // 4 + 5 = 9
    
    // Test Array -= ArrayView
    a -= view_b;
    assert_eq!(a.get(0, 0), Some(1.0)); // 3 - 2 = 1
    assert_eq!(a.get(0, 1), Some(2.0)); // 5 - 3 = 2
    assert_eq!(a.get(1, 0), Some(3.0)); // 7 - 4 = 3
    assert_eq!(a.get(1, 1), Some(4.0)); // 9 - 5 = 4
    
    // Test Array *= ArrayView (element-wise)
    a *= view_b;
    assert_eq!(a.get(0, 0), Some(2.0));  // 1 * 2 = 2
    assert_eq!(a.get(0, 1), Some(6.0));  // 2 * 3 = 6
    assert_eq!(a.get(1, 0), Some(12.0)); // 3 * 4 = 12
    assert_eq!(a.get(1, 1), Some(20.0)); // 4 * 5 = 20
    
    // Test Array /= ArrayView (element-wise)
    a /= view_b;
    assert_eq!(a.get(0, 0), Some(1.0)); // 2 / 2 = 1
    assert_eq!(a.get(0, 1), Some(2.0)); // 6 / 3 = 2
    assert_eq!(a.get(1, 0), Some(3.0)); // 12 / 4 = 3
    assert_eq!(a.get(1, 1), Some(4.0)); // 20 / 5 = 4
}

#[test]
fn test_vector_with_views_assignment() {
    let mut v = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let u = VectorF64::from_slice(&[2.0, 3.0, 4.0, 5.0]);
    let view_u = u.view();
    
    // Test Vector += VectorView
    v += view_u;
    assert_eq!(v.get(0), Some(3.0)); // 1 + 2 = 3
    assert_eq!(v.get(1), Some(5.0)); // 2 + 3 = 5
    assert_eq!(v.get(2), Some(7.0)); // 3 + 4 = 7
    assert_eq!(v.get(3), Some(9.0)); // 4 + 5 = 9
    
    // Test Vector -= VectorView
    v -= view_u;
    assert_eq!(v.get(0), Some(1.0)); // 3 - 2 = 1
    assert_eq!(v.get(1), Some(2.0)); // 5 - 3 = 2
    assert_eq!(v.get(2), Some(3.0)); // 7 - 4 = 3
    assert_eq!(v.get(3), Some(4.0)); // 9 - 5 = 4
    
    // Test Vector *= VectorView (element-wise)
    v *= view_u;
    assert_eq!(v.get(0), Some(2.0));  // 1 * 2 = 2
    assert_eq!(v.get(1), Some(6.0));  // 2 * 3 = 6
    assert_eq!(v.get(2), Some(12.0)); // 3 * 4 = 12
    assert_eq!(v.get(3), Some(20.0)); // 4 * 5 = 20
    
    // Test Vector /= VectorView (element-wise)
    v /= view_u;
    assert_eq!(v.get(0), Some(1.0)); // 2 / 2 = 1
    assert_eq!(v.get(1), Some(2.0)); // 6 / 3 = 2
    assert_eq!(v.get(2), Some(3.0)); // 12 / 4 = 3
    assert_eq!(v.get(3), Some(4.0)); // 20 / 5 = 4
}

#[test]
fn test_assignment_ergonomic_chaining() {
    let mut a = ArrayF64::ones(3, 3);
    let b = ArrayF64::ones(3, 3);
    let view_b = b.view();
    
    // Test ergonomic chaining with views and scalars
    a += view_b;  // a = 2 * ones
    a *= 3.0;     // a = 6 * ones (automatic SIMD)
    a -= &b;      // a = 5 * ones
    a /= 5.0;     // a = ones (automatic SIMD via reciprocal)
    
    // Verify all elements are 1.0
    for row in 0..3 {
        for col in 0..3 {
            assert_eq!(a.get(row, col), Some(1.0));
        }
    }
}

#[test]
fn test_assignment_simd_large_arrays() {
    // Test with large arrays to trigger SIMD optimizations
    let mut a = ArrayF64::ones(100, 100);  // 10,000 elements
    let b = ArrayF64::ones(100, 100);
    
    // These operations should use automatic SIMD
    a += &b;      // Faer's optimized addition
    a *= 2.0;     // Faer's optimized scalar multiplication
    a /= 4.0;     // Optimized via reciprocal multiplication
    
    // Verify correctness
    assert_eq!(a.get(0, 0), Some(1.0));    // (1 + 1) * 2 / 4 = 1
    assert_eq!(a.get(50, 50), Some(1.0));  // Same calculation
    assert_eq!(a.get(99, 99), Some(1.0));  // Same calculation
}

#[test]
fn test_assignment_complex_numbers() {
    use num_complex::Complex;
    
    let mut a = ArrayC64::from_slice(&[
        Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)
    ], 2, 2).unwrap();
    
    let b = ArrayC64::from_slice(&[
        Complex::new(1.0, 1.0), Complex::new(2.0, 2.0),
        Complex::new(3.0, 3.0), Complex::new(4.0, 4.0)
    ], 2, 2).unwrap();
    
    // Test += with complex numbers
    a += &b;
    assert_eq!(a.get(0, 0), Some(Complex::new(2.0, 3.0))); // (1+2i) + (1+1i) = 2+3i
    assert_eq!(a.get(0, 1), Some(Complex::new(5.0, 6.0))); // (3+4i) + (2+2i) = 5+6i
    
    // Test *= scalar with complex numbers (automatic SIMD)
    a *= Complex::new(2.0, 0.0);
    assert_eq!(a.get(0, 0), Some(Complex::new(4.0, 6.0))); // 2 * (2+3i) = 4+6i
    assert_eq!(a.get(0, 1), Some(Complex::new(10.0, 12.0))); // 2 * (5+6i) = 10+12i
}

#[test]
fn test_assignment_dimension_errors() {
    let mut a = ArrayF64::ones(2, 2);
    let b = ArrayF64::ones(3, 3);  // Wrong dimensions
    
    // This should panic with dimension mismatch
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        a += &b;
    }));
    assert!(result.is_err());
    
    // Same for vectors
    let mut v = VectorF64::ones(3);
    let u = VectorF64::ones(4);  // Wrong length
    
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        v += &u;
    }));
    assert!(result.is_err());
}