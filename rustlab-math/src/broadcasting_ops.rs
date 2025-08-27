//! Math-first broadcasting operator overloading with AI-optimized documentation
//!
//! This module implements automatic broadcasting for natural mathematical syntax,
//! building on the explicit functions in [`broadcasting`]. All operations use
//! zero-cost abstractions and integrate with RustLab's ^ operator for matrix multiplication.
//!
//! # Automatic Broadcasting Rules
//! - `&matrix + &vector` automatically broadcasts based on dimensions
//! - Row-wise: vector length matches matrix columns
//! - Column-wise: vector length matches matrix rows
//! - **Element-wise only**: Use ^ for matrix multiplication, not *
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
//! 
//! let data = array64![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];  // 2×3 matrix
//! let means = vec64![10.0, 20.0, 30.0];                   // Column means (length 3)
//! let row_bias = vec64![100.0, 200.0];                    // Row bias (length 2)
//! 
//! // Automatic row-wise broadcasting (vector length = columns)
//! let centered = &data - &means;     // Center each feature
//! 
//! // Automatic column-wise broadcasting (vector length = rows)
//! let biased = &data + &row_bias;    // Add bias to each sample
//! 
//! // IMPORTANT: Use ^ for matrix multiplication
//! let weights = vec64![0.5, 0.3, 0.2];
//! let predictions = &data ^ &weights;  // Matrix × vector = vector
//! ```
//!
//! # Cross-Module Integration
//! - Uses explicit functions from [`broadcasting`] module
//! - Integrates with [`Array`] and [`Vector`] core operations
//! - Compatible with [`statistics`] for feature normalization
//! - Complements [`operators`] module (^ for matrix multiplication)

use crate::{ArrayF64, VectorF64};
use crate::broadcasting::{Shape, broadcast_compatibility, BroadcastType};
use std::ops::{Add, Sub, Mul, Div};

// ========== ADDITION (Matrix + Vector) ==========

/// &ArrayF64 + &VectorF64 with automatic broadcasting
/// 
/// # For AI Code Generation
/// - Automatically detects row-wise vs column-wise broadcasting
/// - Row-wise: vector.len() == matrix.ncols() (most common in ML)
/// - Column-wise: vector.len() == matrix.nrows() (less common)
/// - Element-wise addition only (NOT matrix multiplication)
/// - Use ^ operator for matrix-vector multiplication instead
/// - Panics on incompatible dimensions
/// 
/// # Broadcasting Logic
/// 1. Check if vector length matches matrix columns → row-wise broadcast
/// 2. Check if vector length matches matrix rows → column-wise broadcast
/// 3. Panic if neither matches
/// 
/// # Example
/// ```rust
/// use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
/// 
/// let data = array64![[1.0, 2.0], [3.0, 4.0]];     // 2×2 matrix
/// let col_means = vec64![1.5, 3.0];               // Column means (length 2)
/// let row_bias = vec64![10.0, 20.0];              // Row bias (length 2)
/// 
/// let centered = &data - &col_means;    // Row-wise: subtract mean from each column
/// let biased = &data + &row_bias;       // Column-wise: add bias to each row
/// ```
/// 
/// # See Also
/// - [`broadcasting::ArrayF64::broadcast_add_row_vector`]: Explicit row-wise function
/// - [`broadcasting::ArrayF64::broadcast_add_col_vector`]: Explicit column-wise function
impl Add<&VectorF64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn add(self, vector: &VectorF64) -> ArrayF64 {
        let matrix_shape = Shape::Matrix(self.nrows(), self.ncols());
        let vector_shape = Shape::Vector(vector.len());
        
        match broadcast_compatibility(&matrix_shape, &vector_shape) {
            BroadcastType::VectorToMatrixRows => {
                // Vector length matches columns - broadcast across rows
                self.broadcast_add_row_vector(vector)
                    .expect("Broadcasting failed: dimensions should be compatible")
            }
            BroadcastType::VectorToMatrixCols => {
                // Vector length matches rows - broadcast across columns
                self.broadcast_add_col_vector(vector)
                    .expect("Broadcasting failed: dimensions should be compatible")
            }
            _ => panic!(
                "Cannot broadcast add: Matrix {}×{} with Vector length {}",
                self.nrows(), self.ncols(), vector.len()
            ),
        }
    }
}

/// &VectorF64 + &ArrayF64 with automatic broadcasting (commutative)
impl Add<&ArrayF64> for &VectorF64 {
    type Output = ArrayF64;
    
    fn add(self, matrix: &ArrayF64) -> ArrayF64 {
        matrix + self  // Addition is commutative
    }
}

/// ArrayF64 + &VectorF64 with automatic broadcasting
impl Add<&VectorF64> for ArrayF64 {
    type Output = ArrayF64;
    
    fn add(self, vector: &VectorF64) -> ArrayF64 {
        &self + vector
    }
}

/// &ArrayF64 + VectorF64 with automatic broadcasting
impl Add<VectorF64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn add(self, vector: VectorF64) -> ArrayF64 {
        self + &vector
    }
}

/// ArrayF64 + VectorF64 with automatic broadcasting
impl Add<VectorF64> for ArrayF64 {
    type Output = ArrayF64;
    
    fn add(self, vector: VectorF64) -> ArrayF64 {
        &self + &vector
    }
}

/// VectorF64 + &ArrayF64 with automatic broadcasting (commutative)
impl Add<&ArrayF64> for VectorF64 {
    type Output = ArrayF64;
    
    fn add(self, matrix: &ArrayF64) -> ArrayF64 {
        matrix + &self
    }
}

/// &VectorF64 + ArrayF64 with automatic broadcasting (commutative)
impl Add<ArrayF64> for &VectorF64 {
    type Output = ArrayF64;
    
    fn add(self, matrix: ArrayF64) -> ArrayF64 {
        &matrix + self
    }
}

/// VectorF64 + ArrayF64 with automatic broadcasting (commutative)
impl Add<ArrayF64> for VectorF64 {
    type Output = ArrayF64;
    
    fn add(self, matrix: ArrayF64) -> ArrayF64 {
        &matrix + &self
    }
}

// ========== SUBTRACTION (Matrix - Vector) ==========

/// &ArrayF64 - &VectorF64 with automatic broadcasting
impl Sub<&VectorF64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn sub(self, vector: &VectorF64) -> ArrayF64 {
        let matrix_shape = Shape::Matrix(self.nrows(), self.ncols());
        let vector_shape = Shape::Vector(vector.len());
        
        match broadcast_compatibility(&matrix_shape, &vector_shape) {
            BroadcastType::VectorToMatrixRows => {
                self.broadcast_sub_row_vector(vector)
                    .expect("Broadcasting failed: dimensions should be compatible")
            }
            BroadcastType::VectorToMatrixCols => {
                self.broadcast_sub_col_vector(vector)
                    .expect("Broadcasting failed: dimensions should be compatible")
            }
            _ => panic!(
                "Cannot broadcast subtract: Matrix {}×{} with Vector length {}",
                self.nrows(), self.ncols(), vector.len()
            ),
        }
    }
}

/// &VectorF64 - &ArrayF64 with automatic broadcasting
impl Sub<&ArrayF64> for &VectorF64 {
    type Output = ArrayF64;
    
    fn sub(self, matrix: &ArrayF64) -> ArrayF64 {
        // vector - matrix: Need to negate the matrix result
        let matrix_shape = Shape::Matrix(matrix.nrows(), matrix.ncols());
        let vector_shape = Shape::Vector(self.len());
        
        match broadcast_compatibility(&vector_shape, &matrix_shape) {
            BroadcastType::VectorToMatrixRows => {
                // Create a matrix where each row is the vector
                let result = matrix.broadcast_sub_row_vector(self)
                    .expect("Broadcasting failed: dimensions should be compatible");
                // Negate the result since we want vector - matrix, not matrix - vector
                &result * (-1.0)
            }
            BroadcastType::VectorToMatrixCols => {
                let result = matrix.broadcast_sub_col_vector(self)
                    .expect("Broadcasting failed: dimensions should be compatible");
                &result * (-1.0)
            }
            _ => panic!(
                "Cannot broadcast subtract: Vector length {} with Matrix {}×{}",
                self.len(), matrix.nrows(), matrix.ncols()
            ),
        }
    }
}

// Additional owned variants for subtraction
impl Sub<&VectorF64> for ArrayF64 {
    type Output = ArrayF64;
    fn sub(self, vector: &VectorF64) -> ArrayF64 { &self - vector }
}

impl Sub<VectorF64> for &ArrayF64 {
    type Output = ArrayF64;
    fn sub(self, vector: VectorF64) -> ArrayF64 { self - &vector }
}

impl Sub<VectorF64> for ArrayF64 {
    type Output = ArrayF64;
    fn sub(self, vector: VectorF64) -> ArrayF64 { &self - &vector }
}

impl Sub<&ArrayF64> for VectorF64 {
    type Output = ArrayF64;
    fn sub(self, matrix: &ArrayF64) -> ArrayF64 { &self - matrix }
}

impl Sub<ArrayF64> for &VectorF64 {
    type Output = ArrayF64;
    fn sub(self, matrix: ArrayF64) -> ArrayF64 { self - &matrix }
}

impl Sub<ArrayF64> for VectorF64 {
    type Output = ArrayF64;
    fn sub(self, matrix: ArrayF64) -> ArrayF64 { &self - &matrix }
}

// ========== MULTIPLICATION (Matrix * Vector) ==========

/// &ArrayF64 * &VectorF64 with automatic broadcasting (ELEMENT-WISE)
/// 
/// # ⚠️ CRITICAL FOR AI: Element-wise multiplication only!
/// - This is Hadamard product (element-wise), NOT matrix multiplication
/// - For matrix-vector multiplication, use ^ operator: `matrix ^ vector`
/// - Broadcasting rules same as addition: row-wise or column-wise
/// 
/// # For AI Code Generation
/// - Element-wise multiplication with broadcasting
/// - Common uses: feature scaling, masking, element-wise products
/// - NOT for linear algebra matrix multiplication
/// - Use ^ for matrix multiplication: produces vector output
/// - Use * for element-wise: produces matrix output (same shape as input)
/// 
/// # Example
/// ```rust
/// use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
/// 
/// let data = array64![[1.0, 2.0], [3.0, 4.0]];     // 2×2 matrix
/// let scales = vec64![2.0, 0.5];                   // Feature scales
/// 
/// // Element-wise multiplication (broadcasting)
/// let scaled = &data * &scales;     // Each column scaled by corresponding factor
/// // Result: [[1*2, 2*0.5], [3*2, 4*0.5]] = [[2, 1], [6, 2]]
/// 
/// // Compare with matrix multiplication:
/// let weights = vec64![0.1, 0.9];
/// let linear = &data ^ &weights;    // Matrix × vector = [0.3, 3.9] (vector!)
/// ```
/// 
/// # See Also
/// - [`operators`]: Use ^ for matrix multiplication
/// - [`broadcasting::ArrayF64::broadcast_mul_row_vector`]: Explicit row-wise function
impl Mul<&VectorF64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn mul(self, vector: &VectorF64) -> ArrayF64 {
        let matrix_shape = Shape::Matrix(self.nrows(), self.ncols());
        let vector_shape = Shape::Vector(vector.len());
        
        match broadcast_compatibility(&matrix_shape, &vector_shape) {
            BroadcastType::VectorToMatrixRows => {
                self.broadcast_mul_row_vector(vector)
                    .expect("Broadcasting failed: dimensions should be compatible")
            }
            BroadcastType::VectorToMatrixCols => {
                self.broadcast_mul_col_vector(vector)
                    .expect("Broadcasting failed: dimensions should be compatible")
            }
            _ => panic!(
                "Cannot broadcast multiply: Matrix {}×{} with Vector length {}",
                self.nrows(), self.ncols(), vector.len()
            ),
        }
    }
}

/// &VectorF64 * &ArrayF64 with automatic broadcasting (commutative)
impl Mul<&ArrayF64> for &VectorF64 {
    type Output = ArrayF64;
    
    fn mul(self, matrix: &ArrayF64) -> ArrayF64 {
        matrix * self  // Multiplication is commutative for element-wise
    }
}

// Additional owned variants for multiplication
impl Mul<&VectorF64> for ArrayF64 {
    type Output = ArrayF64;
    fn mul(self, vector: &VectorF64) -> ArrayF64 { &self * vector }
}

impl Mul<VectorF64> for &ArrayF64 {
    type Output = ArrayF64;
    fn mul(self, vector: VectorF64) -> ArrayF64 { self * &vector }
}

impl Mul<VectorF64> for ArrayF64 {
    type Output = ArrayF64;
    fn mul(self, vector: VectorF64) -> ArrayF64 { &self * &vector }
}

impl Mul<&ArrayF64> for VectorF64 {
    type Output = ArrayF64;
    fn mul(self, matrix: &ArrayF64) -> ArrayF64 { &self * matrix }
}

impl Mul<ArrayF64> for &VectorF64 {
    type Output = ArrayF64;
    fn mul(self, matrix: ArrayF64) -> ArrayF64 { self * &matrix }
}

impl Mul<ArrayF64> for VectorF64 {
    type Output = ArrayF64;
    fn mul(self, matrix: ArrayF64) -> ArrayF64 { &self * &matrix }
}

// ========== DIVISION (Matrix / Vector) ==========

/// &ArrayF64 / &VectorF64 with automatic broadcasting
impl Div<&VectorF64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn div(self, vector: &VectorF64) -> ArrayF64 {
        let matrix_shape = Shape::Matrix(self.nrows(), self.ncols());
        let vector_shape = Shape::Vector(vector.len());
        
        match broadcast_compatibility(&matrix_shape, &vector_shape) {
            BroadcastType::VectorToMatrixRows => {
                self.broadcast_div_row_vector(vector)
                    .expect("Broadcasting failed: dimensions should be compatible")
            }
            BroadcastType::VectorToMatrixCols => {
                self.broadcast_div_col_vector(vector)
                    .expect("Broadcasting failed: dimensions should be compatible")
            }
            _ => panic!(
                "Cannot broadcast divide: Matrix {}×{} with Vector length {}",
                self.nrows(), self.ncols(), vector.len()
            ),
        }
    }
}

/// &VectorF64 / &ArrayF64 with automatic broadcasting
impl Div<&ArrayF64> for &VectorF64 {
    type Output = ArrayF64;
    
    fn div(self, matrix: &ArrayF64) -> ArrayF64 {
        // vector / matrix: Need to create a broadcast result then divide
        let matrix_shape = Shape::Matrix(matrix.nrows(), matrix.ncols());
        let vector_shape = Shape::Vector(self.len());
        
        match broadcast_compatibility(&vector_shape, &matrix_shape) {
            BroadcastType::VectorToMatrixRows => {
                // Broadcast vector to match matrix shape, then divide
                let broadcast_matrix = ArrayF64::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
                    self.get(j).unwrap_or(1.0) / matrix.get(i, j).unwrap_or(1.0)
                });
                broadcast_matrix
            }
            BroadcastType::VectorToMatrixCols => {
                let broadcast_matrix = ArrayF64::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
                    self.get(i).unwrap_or(1.0) / matrix.get(i, j).unwrap_or(1.0)
                });
                broadcast_matrix
            }
            _ => panic!(
                "Cannot broadcast divide: Vector length {} with Matrix {}×{}",
                self.len(), matrix.nrows(), matrix.ncols()
            ),
        }
    }
}

// Additional owned variants for division
impl Div<&VectorF64> for ArrayF64 {
    type Output = ArrayF64;
    fn div(self, vector: &VectorF64) -> ArrayF64 { &self / vector }
}

impl Div<VectorF64> for &ArrayF64 {
    type Output = ArrayF64;
    fn div(self, vector: VectorF64) -> ArrayF64 { self / &vector }
}

impl Div<VectorF64> for ArrayF64 {
    type Output = ArrayF64;
    fn div(self, vector: VectorF64) -> ArrayF64 { &self / &vector }
}

impl Div<&ArrayF64> for VectorF64 {
    type Output = ArrayF64;
    fn div(self, matrix: &ArrayF64) -> ArrayF64 { &self / matrix }
}

impl Div<ArrayF64> for &VectorF64 {
    type Output = ArrayF64;
    fn div(self, matrix: ArrayF64) -> ArrayF64 { self / &matrix }
}

impl Div<ArrayF64> for VectorF64 {
    type Output = ArrayF64;
    fn div(self, matrix: ArrayF64) -> ArrayF64 { &self / &matrix }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_automatic_broadcasting_addition() {
        // Test row-wise broadcasting (vector length matches columns)
        let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let row_vec = VectorF64::from_slice(&[10.0, 20.0, 30.0]);
        
        let result = &matrix + &row_vec;
        assert_relative_eq!(result.get(0, 0).unwrap(), 11.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 1).unwrap(), 22.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 2).unwrap(), 33.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 0).unwrap(), 14.0, epsilon = 1e-10);
        
        // Test column-wise broadcasting (vector length matches rows)
        let col_vec = VectorF64::from_slice(&[100.0, 200.0]);
        let result2 = &matrix + &col_vec;
        assert_relative_eq!(result2.get(0, 0).unwrap(), 101.0, epsilon = 1e-10);
        assert_relative_eq!(result2.get(1, 0).unwrap(), 204.0, epsilon = 1e-10);
        
        // Test commutative
        let result3 = &row_vec + &matrix;
        assert_eq!(result.shape(), result3.shape());
    }
    
    #[test]
    fn test_automatic_broadcasting_subtraction() {
        let matrix = ArrayF64::from_slice(&[10.0, 20.0, 30.0, 40.0], 2, 2).unwrap();
        let vec = VectorF64::from_slice(&[1.0, 2.0]);
        
        // Matrix - vector broadcasting: vector broadcasts as column vector
        // Matrix:        Vector:      Result:
        // [10  20]   -   [1]     =   [9  19]
        // [30  40]       [2]         [28 38]  
        let result = &matrix - &vec;
        
        assert_relative_eq!(result.get(0, 0).unwrap(), 9.0, epsilon = 1e-10);   // 10 - 1
        assert_relative_eq!(result.get(1, 0).unwrap(), 28.0, epsilon = 1e-10);  // 30 - 2 (row broadcast)
        assert_relative_eq!(result.get(0, 1).unwrap(), 19.0, epsilon = 1e-10);  // 20 - 1  
        assert_relative_eq!(result.get(1, 1).unwrap(), 38.0, epsilon = 1e-10);  // 40 - 2
        
        // Vector - matrix
        let result2 = &vec - &matrix;
        assert_relative_eq!(result2.get(0, 0).unwrap(), -9.0, epsilon = 1e-10);  // 1 - 10
        assert_relative_eq!(result2.get(1, 0).unwrap(), -28.0, epsilon = 1e-10); // 2 - 30
    }
    
    #[test]
    fn test_automatic_broadcasting_multiplication() {
        let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let row_vec = VectorF64::from_slice(&[10.0, 10.0, 10.0]);
        
        let result = &matrix * &row_vec;
        assert_relative_eq!(result.get(0, 0).unwrap(), 10.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 1).unwrap(), 20.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 2).unwrap(), 30.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_automatic_broadcasting_division() {
        let matrix = ArrayF64::from_slice(&[10.0, 20.0, 30.0, 40.0], 2, 2).unwrap();
        let vec = VectorF64::from_slice(&[10.0, 10.0]);
        
        let result = &matrix / &vec;
        assert_relative_eq!(result.get(0, 0).unwrap(), 1.0, epsilon = 1e-10);   // 10/10
        assert_relative_eq!(result.get(0, 1).unwrap(), 2.0, epsilon = 1e-10);   // 20/10
        assert_relative_eq!(result.get(1, 0).unwrap(), 3.0, epsilon = 1e-10);   // 30/10
        assert_relative_eq!(result.get(1, 1).unwrap(), 4.0, epsilon = 1e-10);   // 40/10
    }
    
    #[test]
    #[should_panic(expected = "Cannot broadcast")]
    fn test_incompatible_broadcasting() {
        let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0]); // Wrong size
        
        let _ = &matrix + &vec;
    }
}