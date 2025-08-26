//! Tests for zero-copy view functionality

use rustlab_math::*;

#[test]
fn test_array_view_basic_operations() {
    let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let b = ArrayF64::from_slice(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
    
    // Create views
    let view_a = a.view();
    let view_b = b.view();
    
    // Test view addition
    let result = view_a + view_b;
    assert_eq!(result.get(0, 0), Some(6.0));  // 1 + 5
    assert_eq!(result.get(0, 1), Some(8.0));  // 2 + 6
    assert_eq!(result.get(1, 0), Some(10.0)); // 3 + 7
    assert_eq!(result.get(1, 1), Some(12.0)); // 4 + 8
    
    // Test view subtraction
    let result = view_b - view_a;
    assert_eq!(result.get(0, 0), Some(4.0));  // 5 - 1
    assert_eq!(result.get(0, 1), Some(4.0));  // 6 - 2
    assert_eq!(result.get(1, 0), Some(4.0));  // 7 - 3
    assert_eq!(result.get(1, 1), Some(4.0));  // 8 - 4
    
    // Test view element-wise multiplication
    let result = view_a * view_b;
    assert_eq!(result.get(0, 0), Some(5.0));  // 1 * 5
    assert_eq!(result.get(0, 1), Some(12.0)); // 2 * 6
    assert_eq!(result.get(1, 0), Some(21.0)); // 3 * 7
    assert_eq!(result.get(1, 1), Some(32.0)); // 4 * 8
}

#[test]
fn test_array_view_matrix_multiplication() {
    let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let b = ArrayF64::from_slice(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
    
    let view_a = a.view();
    let view_b = b.view();
    
    // Test view ^ view matrix multiplication
    let result = view_a ^ view_b;
    assert_eq!(result.get(0, 0), Some(19.0)); // 1*5 + 2*7 = 5 + 14 = 19
    assert_eq!(result.get(0, 1), Some(22.0)); // 1*6 + 2*8 = 6 + 16 = 22
    assert_eq!(result.get(1, 0), Some(43.0)); // 3*5 + 4*7 = 15 + 28 = 43
    assert_eq!(result.get(1, 1), Some(50.0)); // 3*6 + 4*8 = 18 + 32 = 50
}

#[test]
fn test_array_view_scalar_multiplication() {
    let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let view_a = a.view();
    
    // Test view * scalar
    let result = view_a * 3.0;
    assert_eq!(result.get(0, 0), Some(3.0));  // 1 * 3
    assert_eq!(result.get(0, 1), Some(6.0));  // 2 * 3
    assert_eq!(result.get(1, 0), Some(9.0));  // 3 * 3
    assert_eq!(result.get(1, 1), Some(12.0)); // 4 * 3
}

#[test]
fn test_vector_view_basic_operations() {
    let v1 = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    let v2 = VectorF64::from_slice(&[4.0, 5.0, 6.0]);
    
    let view1 = v1.view();
    let view2 = v2.view();
    
    // Test view addition
    let result = view1 + view2;
    assert_eq!(result.get(0), Some(5.0));  // 1 + 4
    assert_eq!(result.get(1), Some(7.0));  // 2 + 5
    assert_eq!(result.get(2), Some(9.0));  // 3 + 6
    
    // Test view subtraction
    let result = view2 - view1;
    assert_eq!(result.get(0), Some(3.0));  // 4 - 1
    assert_eq!(result.get(1), Some(3.0));  // 5 - 2
    assert_eq!(result.get(2), Some(3.0));  // 6 - 3
    
    // Test view element-wise multiplication
    let result = view1 * view2;
    assert_eq!(result.get(0), Some(4.0));  // 1 * 4
    assert_eq!(result.get(1), Some(10.0)); // 2 * 5
    assert_eq!(result.get(2), Some(18.0)); // 3 * 6
}

#[test]
fn test_vector_view_dot_product() {
    let v1 = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    let v2 = VectorF64::from_slice(&[4.0, 5.0, 6.0]);
    
    let view1 = v1.view();
    let view2 = v2.view();
    
    // Test view ^ view dot product
    let result = view1 ^ view2;
    assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

#[test]
fn test_vector_view_scalar_multiplication() {
    let v = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    let view = v.view();
    
    // Test view * scalar
    let result = view * 2.5;
    assert_eq!(result.get(0), Some(2.5));  // 1 * 2.5
    assert_eq!(result.get(1), Some(5.0));  // 2 * 2.5
    assert_eq!(result.get(2), Some(7.5));  // 3 * 2.5
}

#[test]
fn test_mixed_operations_array_and_view() {
    let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let b = ArrayF64::from_slice(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
    
    let view_a = a.view();
    
    // Test Array ^ ArrayView
    let result = &b ^ view_a;
    assert_eq!(result.get(0, 0), Some(23.0)); // 5*1 + 6*3 = 5 + 18 = 23
    assert_eq!(result.get(0, 1), Some(34.0)); // 5*2 + 6*4 = 10 + 24 = 34
    assert_eq!(result.get(1, 0), Some(31.0)); // 7*1 + 8*3 = 7 + 24 = 31
    assert_eq!(result.get(1, 1), Some(46.0)); // 7*2 + 8*4 = 14 + 32 = 46
    
    // Test ArrayView ^ Array 
    let result = view_a ^ &b;
    assert_eq!(result.get(0, 0), Some(19.0)); // 1*5 + 2*7 = 5 + 14 = 19
    assert_eq!(result.get(0, 1), Some(22.0)); // 1*6 + 2*8 = 6 + 16 = 22
    assert_eq!(result.get(1, 0), Some(43.0)); // 3*5 + 4*7 = 15 + 28 = 43
    assert_eq!(result.get(1, 1), Some(50.0)); // 3*6 + 4*8 = 18 + 32 = 50
}

#[test]
fn test_mixed_operations_vector_and_view() {
    let v1 = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    let v2 = VectorF64::from_slice(&[4.0, 5.0, 6.0]);
    
    let view1 = v1.view();
    
    // Test Vector ^ VectorView (dot product)
    let result = &v2 ^ view1;
    assert_eq!(result, 32.0); // 4*1 + 5*2 + 6*3 = 4 + 10 + 18 = 32
    
    // Test VectorView ^ Vector (dot product)
    let result = view1 ^ &v2;
    assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

#[test]
fn test_view_to_owned() {
    let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let v = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    
    let array_view = a.view();
    let vector_view = v.view();
    
    // Test converting views back to owned types
    let owned_array = array_view.to_owned();
    assert_eq!(owned_array.get(0, 0), Some(1.0));
    assert_eq!(owned_array.get(1, 1), Some(4.0));
    
    let owned_vector = vector_view.to_owned();
    assert_eq!(owned_vector.get(0), Some(1.0));
    assert_eq!(owned_vector.get(2), Some(3.0));
}