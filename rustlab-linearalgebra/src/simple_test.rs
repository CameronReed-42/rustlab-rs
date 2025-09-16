//! Simple test to understand faer 0.22 API

use faer::{Mat, Entity};
use rustlab_math::ArrayF64;

/// Test basic faer functionality
pub fn test_faer_api() {
    // Create a simple 2x2 matrix
    let matrix = Mat::<f64>::from_fn(2, 2, |i, j| (i + j) as f64);
    println!("Matrix: {:?}", matrix);
    
    // Test basic operations
    let det = matrix.determinant();
    println!("Determinant: {:?}", det);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_faer() {
        test_faer_api();
    }
}