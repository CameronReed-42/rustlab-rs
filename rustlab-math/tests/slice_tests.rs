//! Tests for as_slice() and as_mut_slice() methods

use rustlab_math::*;

#[test]
fn test_vector_slice_access() {
    let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Test immutable slice access
    match vec.as_slice() {
        Some(slice) => {
            assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0]);
            assert_eq!(slice.len(), 5);
        }
        None => panic!("❌ Slice access failed - vector should be contiguous"),
    }
    
    // Test contiguity check
    assert!(vec.is_contiguous(), "Vector should be contiguous");
    
    // Test unchecked access
    let slice_unchecked = vec.as_slice_unchecked();
    assert_eq!(slice_unchecked, &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_vector_mut_slice_access() {
    let mut vec = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    
    // Test mutable slice access
    match vec.as_mut_slice() {
        Some(slice) => {
            assert_eq!(slice.len(), 3);
            slice[0] = 10.0;
            slice[2] = 30.0;
        }
        None => panic!("❌ Mutable slice access failed"),
    }
    
    // Verify the changes
    if let Some(slice) = vec.as_slice() {
        assert_eq!(slice, &[10.0, 2.0, 30.0]);
    }
}

#[test]
fn test_vector_contiguity_for_various_constructors() {
    let zeros = VectorF64::zeros(5);
    let ones = VectorF64::ones(3);
    let from_slice = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    
    // All standard constructors should create contiguous vectors
    assert!(zeros.is_contiguous(), "zeros() should create contiguous vector");
    assert!(ones.is_contiguous(), "ones() should create contiguous vector");
    assert!(from_slice.is_contiguous(), "from_slice() should create contiguous vector");
    
    // Test slice access
    if let Some(zeros_slice) = zeros.as_slice() {
        assert_eq!(zeros_slice.len(), 5);
        assert!(zeros_slice.iter().all(|&x| x == 0.0));
    }
    
    if let Some(ones_slice) = ones.as_slice() {
        assert_eq!(ones_slice.len(), 3);
        assert!(ones_slice.iter().all(|&x| x == 1.0));
    }
}

#[test]
fn test_empty_vector() {
    let empty = VectorF64::zeros(0);
    
    assert!(empty.is_contiguous(), "Empty vector should be contiguous");
    
    if let Some(slice) = empty.as_slice() {
        assert_eq!(slice.len(), 0);
        assert_eq!(slice, &[] as &[f64]);
    }
}

#[test]
fn test_array_slice_access() {
    let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    
    // Test immutable slice access - faer arrays are NOT contiguous
    match arr.as_slice() {
        Some(_) => panic!("❌ Array slice access should return None - faer arrays are not contiguous"),
        None => {
            // This is expected - faer matrices store columns separately
            println!("✅ Array correctly returns None for slice access (not contiguous)");
        }
    }
    
    // Test contiguity check - faer arrays are not contiguous
    assert!(!arr.is_contiguous(), "Array should NOT be contiguous (faer behavior)");
    
    // Verify individual element access still works
    assert_eq!(arr.get(0, 0), Some(1.0));
    assert_eq!(arr.get(0, 1), Some(2.0));
    assert_eq!(arr.get(1, 0), Some(3.0));
    assert_eq!(arr.get(1, 1), Some(4.0));
}

#[test]
fn test_array_mut_slice_access() {
    let mut arr = ArrayF64::ones(2, 3); // 2x3 array of ones
    
    // Test mutable slice access - should return None for faer arrays
    match arr.as_mut_slice() {
        Some(_) => panic!("❌ Mutable array slice access should return None - faer arrays are not contiguous"),
        None => {
            println!("✅ Array correctly returns None for mutable slice access (not contiguous)");
        }
    }
    
    // Instead, verify we can use to_vec() to get the data
    let vec = arr.to_vec();
    assert_eq!(vec.len(), 6); // 2 * 3 = 6 elements
    assert!(vec.iter().all(|&x| x == 1.0)); // All elements are 1.0
    
    // And verify individual element access for modifications
    // Note: Individual element modification would need to be done element by element
    assert_eq!(arr.get(0, 0), Some(1.0));
    assert_eq!(arr.get(1, 2), Some(1.0));
}

#[test]
fn test_array_contiguity_for_various_constructors() {
    let zeros = ArrayF64::zeros(3, 4);
    let ones = ArrayF64::ones(2, 2);
    let from_slice = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    
    // faer arrays are NOT contiguous (only empty arrays are)
    assert!(!zeros.is_contiguous(), "zeros() creates non-contiguous faer array");
    assert!(!ones.is_contiguous(), "ones() creates non-contiguous faer array");
    assert!(!from_slice.is_contiguous(), "from_slice() creates non-contiguous faer array");
    
    // Test alternative access methods - to_vec() and iter()
    let zeros_vec = zeros.to_vec();
    assert_eq!(zeros_vec.len(), 12); // 3 * 4 = 12
    assert!(zeros_vec.iter().all(|&x| x == 0.0));
    
    let ones_vec = ones.to_vec();
    assert_eq!(ones_vec.len(), 4); // 2 * 2 = 4
    assert!(ones_vec.iter().all(|&x| x == 1.0));
    
    // Test iterator access
    let sum: f64 = ones.iter().sum();
    assert_eq!(sum, 4.0); // 4 ones
    
    let count = zeros.iter().count();
    assert_eq!(count, 12); // 12 elements
}

#[test]
fn test_empty_array() {
    let empty_rows = ArrayF64::zeros(0, 5);
    let empty_cols = ArrayF64::zeros(3, 0);
    let empty_both = ArrayF64::zeros(0, 0);
    
    assert!(empty_rows.is_contiguous(), "Empty rows array should be contiguous");
    assert!(empty_cols.is_contiguous(), "Empty cols array should be contiguous");  
    assert!(empty_both.is_contiguous(), "Empty array should be contiguous");
    
    if let Some(slice) = empty_rows.as_slice() {
        assert_eq!(slice.len(), 0);
    }
    
    if let Some(slice) = empty_cols.as_slice() {
        assert_eq!(slice.len(), 0);
    }
    
    if let Some(slice) = empty_both.as_slice() {
        assert_eq!(slice.len(), 0);
    }
}

#[test]
fn test_slice_operations_with_different_types() {
    // Test with f32 vectors (still contiguous)
    let vec_f32 = VectorF32::from_slice(&[1.0f32, 2.0f32, 3.0f32]);
    assert!(vec_f32.is_contiguous());
    if let Some(slice) = vec_f32.as_slice() {
        assert_eq!(slice, &[1.0f32, 2.0f32, 3.0f32]);
    }
    
    // Test with f32 arrays (not contiguous)
    let arr_f32 = ArrayF32::ones(2, 2);
    assert!(!arr_f32.is_contiguous()); // faer arrays are not contiguous
    
    // Use to_vec() instead
    let arr_f32_vec = arr_f32.to_vec();
    assert_eq!(arr_f32_vec.len(), 4);
    assert!(arr_f32_vec.iter().all(|&x| x == 1.0f32));
    
    // Test with complex vectors (still contiguous)
    use num_complex::Complex;
    let vec_c64 = VectorC64::from_slice(&[
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
    ]);
    assert!(vec_c64.is_contiguous());
    if let Some(slice) = vec_c64.as_slice() {
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], Complex::new(1.0, 2.0));
        assert_eq!(slice[1], Complex::new(3.0, 4.0));
    }
    
    // Test complex arrays with to_vec()
    let arr_c64 = ArrayC64::from_slice(&[
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ], 2, 2).unwrap();
    
    let arr_c64_vec = arr_c64.to_vec();
    assert_eq!(arr_c64_vec.len(), 4);
    // Column-major order: [1+0i, 3+0i, 2+0i, 4+0i]
    assert_eq!(arr_c64_vec[0], Complex::new(1.0, 0.0));
    assert_eq!(arr_c64_vec[1], Complex::new(3.0, 0.0));
}

#[test]
fn test_slice_performance_characteristics() {
    // Test with large data
    let large_vec = VectorF64::zeros(10000);
    let large_arr = ArrayF64::zeros(100, 100);
    
    // Vectors are contiguous, arrays are not
    assert!(large_vec.is_contiguous());
    assert!(!large_arr.is_contiguous()); // faer arrays are not contiguous
    
    // Vector slice access (zero-copy)
    if let Some(vec_slice) = large_vec.as_slice() {
        assert_eq!(vec_slice.len(), 10000);
        // Ensure we can access without copying
        assert_eq!(vec_slice[0], 0.0);
        assert_eq!(vec_slice[9999], 0.0);
    }
    
    // Array iterator access (no intermediate vector allocation)
    let count = large_arr.iter().count();
    assert_eq!(count, 10000); // 100 * 100
    
    // Array sum using iterator (efficient)
    let sum: f64 = large_arr.iter().sum();
    assert_eq!(sum, 0.0); // All zeros
    
    // Test that to_vec() works but creates a copy
    let arr_vec = large_arr.to_vec();
    assert_eq!(arr_vec.len(), 10000);
    assert!(arr_vec.iter().all(|&x| x == 0.0));
}

#[test]
fn test_slice_interoperability_with_std() {
    let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    
    if let Some(slice) = vec.as_slice() {
        // Test standard slice operations
        assert_eq!(slice.len(), 5);
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice.first(), Some(&1.0));
        assert_eq!(slice.last(), Some(&5.0));
        
        // Test iterator functionality
        let sum: f64 = slice.iter().sum();
        assert_eq!(sum, 15.0); // 1 + 2 + 3 + 4 + 5
        
        let max = slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert_eq!(max, 5.0);
        
        // Test slice splitting
        let (left, right) = slice.split_at(2);
        assert_eq!(left, &[1.0, 2.0]);
        assert_eq!(right, &[3.0, 4.0, 5.0]);
    }
}

#[test]
fn test_array_to_vec_column_major_order() {
    // Test that to_vec() returns data in column-major order
    let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    
    // Matrix should look like:
    // [1.0  2.0]
    // [3.0  4.0]
    
    // Verify individual elements
    assert_eq!(arr.get(0, 0), Some(1.0));
    assert_eq!(arr.get(0, 1), Some(2.0));
    assert_eq!(arr.get(1, 0), Some(3.0));
    assert_eq!(arr.get(1, 1), Some(4.0));
    
    // to_vec() should return column-major order: [col0_row0, col0_row1, col1_row0, col1_row1]
    let vec = arr.to_vec();
    assert_eq!(vec, vec![1.0, 3.0, 2.0, 4.0]); // Column-major: [1.0, 3.0, 2.0, 4.0]
    
    // Test with a 3x2 matrix for more clarity
    let arr2 = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
    
    // Matrix should look like:
    // [1.0  2.0]
    // [3.0  4.0] 
    // [5.0  6.0]
    
    let vec2 = arr2.to_vec();
    // Column-major: [col0: 1,3,5, col1: 2,4,6]
    assert_eq!(vec2, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    
    // Test iterator produces same order
    let iter_vec: Vec<_> = arr2.iter().cloned().collect();
    assert_eq!(iter_vec, vec2);
}