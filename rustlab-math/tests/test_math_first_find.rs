// Test the new math-first find and any methods

use rustlab_math::{VectorF64, comparison::*};

#[test]
fn test_math_first_find_methods() {
    let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0, 9.0]);
    
    // Test math-first approach
    let large_mask = data.gt(5.0);
    assert_eq!(large_mask.as_slice(), &[false, false, true, false, true]);
    
    let large_values = data.find_where(&large_mask).unwrap();
    assert_eq!(large_values.to_slice(), &[7.0, 9.0]);
    
    let large_indices = data.find_indices(&large_mask).unwrap();
    assert_eq!(large_indices.to_slice(), &[2.0, 4.0]);
}

#[test]
fn test_ergonomic_any_methods() {
    let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0, 9.0]);
    
    assert!(data.any_gt(5.0));
    assert!(!data.any_lt(1.0));
    assert!(data.all_gt(0.0));
    assert!(!data.all_gt(10.0));
}

#[test]
fn test_find_value_methods() {
    let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0, 9.0]);
    
    assert_eq!(data.find_gt(5.0), Some(7.0));
    assert_eq!(data.find_lt(2.0), Some(1.0));
    assert_eq!(data.find_gt(10.0), None);
}

#[test]  
fn test_find_index_methods() {
    let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0, 9.0]);
    
    assert_eq!(data.find_index_gt(5.0), Some(2));
    assert_eq!(data.find_index_lt(2.0), Some(0));
    assert_eq!(data.find_index_gt(10.0), None);
}