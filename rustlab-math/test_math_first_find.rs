// Test the new math-first find and any methods

use rustlab_math::{VectorF64, comparison::*};

fn main() {
    println!("=== Testing Math-First Find and Any Methods ===");
    
    let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0, 9.0]);
    println!("Data: {:?}", data.to_slice());
    
    // Test math-first approach
    println!("\n=== Math-First Approach ===");
    let large_mask = data.gt(5.0);
    println!("Mask (> 5.0): {:?}", large_mask.as_slice());
    
    let large_values = data.find_where(&large_mask).unwrap();
    println!("Values where > 5.0: {:?}", large_values.to_slice());
    
    let large_indices = data.find_indices(&large_mask).unwrap();
    println!("Indices where > 5.0: {:?}", large_indices.to_slice());
    
    // Test ergonomic shortcuts
    println!("\n=== Ergonomic Shortcuts ===");
    println!("Any > 5.0: {}", data.any_gt(5.0));
    println!("Any < 1.0: {}", data.any_lt(1.0));
    println!("All > 0.0: {}", data.all_gt(0.0));
    println!("All > 10.0: {}", data.all_gt(10.0));
    
    // Test find methods
    println!("\n=== Find Methods ===");
    println!("First > 5.0: {:?}", data.find_gt(5.0));
    println!("First < 2.0: {:?}", data.find_lt(2.0));
    println!("First > 10.0: {:?}", data.find_gt(10.0));
    
    // Test find index methods
    println!("\n=== Find Index Methods ===");
    println!("Index of first > 5.0: {:?}", data.find_index_gt(5.0));
    println!("Index of first < 2.0: {:?}", data.find_index_lt(2.0));
    println!("Index of first > 10.0: {:?}", data.find_index_gt(10.0));
    
    println!("\n✅ All tests completed successfully!");
}