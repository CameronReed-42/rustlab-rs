//! Broadcasting operations for rustlab-math-v2 with AI-optimized documentation
//!
//! This module provides comprehensive broadcasting support following NumPy-style
//! broadcasting rules. All operations use zero-cost abstractions and integrate
//! with RustLab's mathematical operators (^ for matrix multiplication, * for element-wise).
//!
//! # Broadcasting Rules
//! - **Scalar-Matrix**: scalar applied to every matrix element
//! - **Vector-Matrix (row)**: vector length must match matrix columns
//! - **Vector-Matrix (col)**: vector length must match matrix rows
//! - **Matrix-Matrix**: dimensions must be identical or one dimension is 1
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::{ArrayF64, VectorF64, broadcasting::*};
//! 
//! let data = ArrayF64::ones(100, 5);         // 100 samples, 5 features
//! let mean_vals = VectorF64::ones(5);        // Feature means
//! let centered = data.broadcast_sub_row_vector(&mean_vals)?;  // Center data
//! 
//! // Equivalent using automatic broadcasting operators (see broadcasting_ops)
//! let centered2 = &data - &mean_vals;        // Automatic broadcasting
//! ```
//!
//! # Cross-Module Integration
//! - See [`broadcasting_ops`] for operator overloading with automatic broadcasting
//! - Compatible with [`Array`] and [`Vector`] core operations
//! - Integrates with [`statistics`] for feature normalization patterns

use crate::{ArrayF64, VectorF64};
use faer::Mat;

/// Broadcasting rules for determining compatible shapes
#[derive(Debug, Clone, PartialEq)]
pub enum BroadcastType {
    /// Identical shapes - no broadcasting needed
    Identical,
    /// Scalar to matrix/vector broadcasting
    ScalarToMatrix,
    /// Vector to matrix broadcasting (row-wise)
    VectorToMatrixRows,
    /// Vector to matrix broadcasting (column-wise)  
    VectorToMatrixCols,
    /// Matrix to vector broadcasting (reduction)
    MatrixToVector,
    /// Incompatible shapes
    Incompatible,
}

/// Broadcasting operation types for explicit broadcasting functions
/// 
/// # For AI Code Generation
/// - Represents the four basic arithmetic operations for broadcasting
/// - Used with [`ArrayF64::broadcast_with_matrix`] for explicit control
/// - Element-wise operations only (NOT matrix multiplication)
/// - For matrix multiplication, use the ^ operator instead
/// 
/// # Operations
/// - `Add`: Element-wise addition (A + B)
/// - `Sub`: Element-wise subtraction (A - B)  
/// - `Mul`: Element-wise multiplication (A * B, Hadamard product)
/// - `Div`: Element-wise division (A / B)
#[derive(Clone, Copy)]
pub enum BroadcastOp {
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
}

/// Shape representation for broadcasting compatibility checks
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// 1D vector shape with length
    Vector(usize),
    /// 2D matrix shape with (rows, columns)
    Matrix(usize, usize),
}

/// Determine broadcasting compatibility between two shapes
/// 
/// # For AI Code Generation
/// - Implements NumPy-style broadcasting rules
/// - Returns specific broadcast type for optimization
/// - Use this to validate operations before execution
/// - Common pattern: check compatibility before calling broadcast functions
/// 
/// # Broadcasting Rules
/// 1. **Identical**: Same dimensions, direct element-wise operation
/// 2. **ScalarToMatrix**: One operand is 1×1, broadcast to larger shape
/// 3. **VectorToMatrixRows**: Vector length matches matrix columns (broadcast across rows)
/// 4. **VectorToMatrixCols**: Vector length matches matrix rows (broadcast across columns)
/// 5. **Incompatible**: No valid broadcasting rule applies
/// 
/// # Example
/// ```rust
/// use rustlab_math::broadcasting::{broadcast_compatibility, Shape, BroadcastType};
/// 
/// let matrix_shape = Shape::Matrix(100, 5);   // 100×5 data matrix
/// let vector_shape = Shape::Vector(5);        // 5-element feature vector
/// 
/// let compat = broadcast_compatibility(&matrix_shape, &vector_shape);
/// assert_eq!(compat, BroadcastType::VectorToMatrixRows);  // Vector broadcasts to each row
/// ```
pub fn broadcast_compatibility(left_shape: &Shape, right_shape: &Shape) -> BroadcastType {
    match (left_shape, right_shape) {
        // Identical shapes
        (Shape::Matrix(r1, c1), Shape::Matrix(r2, c2)) if r1 == r2 && c1 == c2 => BroadcastType::Identical,
        (Shape::Vector(l1), Shape::Vector(l2)) if l1 == l2 => BroadcastType::Identical,
        
        // Scalar broadcasting (represented as 1x1 matrix)
        (Shape::Matrix(1, 1), Shape::Matrix(_, _)) => BroadcastType::ScalarToMatrix,
        (Shape::Matrix(_, _), Shape::Matrix(1, 1)) => BroadcastType::ScalarToMatrix,
        
        // Vector to matrix broadcasting
        (Shape::Vector(len), Shape::Matrix(rows, cols)) => {
            if *len == *rows {
                BroadcastType::VectorToMatrixCols
            } else if *len == *cols {
                BroadcastType::VectorToMatrixRows
            } else {
                BroadcastType::Incompatible
            }
        }
        
        (Shape::Matrix(rows, cols), Shape::Vector(len)) => {
            if *len == *rows {
                BroadcastType::VectorToMatrixCols
            } else if *len == *cols {
                BroadcastType::VectorToMatrixRows
            } else {
                BroadcastType::Incompatible
            }
        }
        
        _ => BroadcastType::Incompatible,
    }
}

/// Broadcasting implementation for ArrayF64
impl ArrayF64 {
    /// Broadcast add this matrix with a vector row-wise
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ and vector v ∈ ℝⁿ:
    /// result[i,j] = A[i,j] + v[j] for all i ∈ [0,m), j ∈ [0,n)
    /// Vector is broadcast across rows (each row gets the same vector added)
    /// 
    /// # For AI Code Generation
    /// - Vector length MUST equal matrix columns: v.len() == A.ncols()
    /// - Common uses: feature bias addition, mean centering, offset application
    /// - Equivalent to NumPy: `A + v` where v is shape (n,)
    /// - More explicit than automatic broadcasting operators
    /// - Result has same shape as input matrix
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, VectorF64};
    /// 
    /// // Feature normalization: subtract mean from each feature
    /// let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
    /// let feature_means = VectorF64::from_slice(&[2.0, 5.0]);  // Mean of each column
    /// 
    /// let centered = data.broadcast_sub_row_vector(&feature_means).unwrap();
    /// // Result: each row has column means subtracted
    /// // [[1-2, 2-5], [3-2, 4-5], [5-2, 6-5]] = [[-1, -3], [1, -1], [3, 1]]
    /// ```
    /// 
    /// # Errors
    /// Returns error if vector.len() ≠ matrix.ncols()
    /// 
    /// # See Also
    /// - [`broadcast_add_col_vector`]: Column-wise broadcasting
    /// - [`broadcasting_ops`]: Automatic broadcasting with operators
    /// - [`statistics::BasicStatistics::mean`]: Calculate means for normalization
    pub fn broadcast_add_row_vector(&self, vector: &VectorF64) -> Result<ArrayF64, String> {
        if vector.len() != self.ncols() {
            return Err(format!(
                "Vector length {} does not match matrix columns {}",
                vector.len(), self.ncols()
            ));
        }
        
        let result = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            unsafe {
                self.inner.get_unchecked(i, j) + vector.inner.get_unchecked(j)
            }
        });
        
        Ok(ArrayF64 { inner: result })
    }
    
    /// Broadcast subtract this matrix with a vector row-wise
    /// Vector is subtracted from each row of the matrix
    pub fn broadcast_sub_row_vector(&self, vector: &VectorF64) -> Result<ArrayF64, String> {
        if vector.len() != self.ncols() {
            return Err(format!(
                "Vector length {} does not match matrix columns {}",
                vector.len(), self.ncols()
            ));
        }
        
        let result = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            unsafe {
                self.inner.get_unchecked(i, j) - vector.inner.get_unchecked(j)
            }
        });
        
        Ok(ArrayF64 { inner: result })
    }
    
    /// Broadcast multiply this matrix with a vector row-wise
    /// Vector is multiplied with each row of the matrix
    pub fn broadcast_mul_row_vector(&self, vector: &VectorF64) -> Result<ArrayF64, String> {
        if vector.len() != self.ncols() {
            return Err(format!(
                "Vector length {} does not match matrix columns {}",
                vector.len(), self.ncols()
            ));
        }
        
        let result = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            unsafe {
                self.inner.get_unchecked(i, j) * vector.inner.get_unchecked(j)
            }
        });
        
        Ok(ArrayF64 { inner: result })
    }
    
    /// Broadcast divide this matrix with a vector row-wise
    /// Each row of the matrix is divided by the vector
    pub fn broadcast_div_row_vector(&self, vector: &VectorF64) -> Result<ArrayF64, String> {
        if vector.len() != self.ncols() {
            return Err(format!(
                "Vector length {} does not match matrix columns {}",
                vector.len(), self.ncols()
            ));
        }
        
        let result = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            unsafe {
                self.inner.get_unchecked(i, j) / vector.inner.get_unchecked(j)
            }
        });
        
        Ok(ArrayF64 { inner: result })
    }
    
    /// Broadcast add this matrix with a vector column-wise
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ and vector v ∈ ℝᵐ:
    /// result[i,j] = A[i,j] + v[i] for all i ∈ [0,m), j ∈ [0,n)
    /// Vector is broadcast across columns (each column gets the same vector added)
    /// 
    /// # For AI Code Generation
    /// - Vector length MUST equal matrix rows: v.len() == A.nrows()
    /// - Common uses: sample-wise bias addition, per-sample normalization
    /// - Equivalent to NumPy: `A + v.reshape(-1, 1)` where v is shape (m,)
    /// - Less common than row-wise broadcasting in ML applications
    /// - Result has same shape as input matrix
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, VectorF64};
    /// 
    /// // Sample-wise adjustment: add different offset to each sample
    /// let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let sample_offsets = VectorF64::from_slice(&[10.0, 20.0]);  // Offset for each row
    /// 
    /// let adjusted = data.broadcast_add_col_vector(&sample_offsets).unwrap();
    /// // Result: [[1+10, 2+10], [3+20, 4+20]] = [[11, 12], [23, 24]]
    /// ```
    /// 
    /// # Errors
    /// Returns error if vector.len() ≠ matrix.nrows()
    /// 
    /// # See Also
    /// - [`broadcast_add_row_vector`]: Row-wise broadcasting (more common)
    /// - [`broadcasting_ops`]: Automatic broadcasting with operators
    pub fn broadcast_add_col_vector(&self, vector: &VectorF64) -> Result<ArrayF64, String> {
        if vector.len() != self.nrows() {
            return Err(format!(
                "Vector length {} does not match matrix rows {}",
                vector.len(), self.nrows()
            ));
        }
        
        let result = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            unsafe {
                self.inner.get_unchecked(i, j) + vector.inner.get_unchecked(i)
            }
        });
        
        Ok(ArrayF64 { inner: result })
    }
    
    /// Broadcast subtract this matrix with a vector column-wise
    /// Vector is subtracted from each column of the matrix
    pub fn broadcast_sub_col_vector(&self, vector: &VectorF64) -> Result<ArrayF64, String> {
        if vector.len() != self.nrows() {
            return Err(format!(
                "Vector length {} does not match matrix rows {}",
                vector.len(), self.nrows()
            ));
        }
        
        let result = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            unsafe {
                self.inner.get_unchecked(i, j) - vector.inner.get_unchecked(i)
            }
        });
        
        Ok(ArrayF64 { inner: result })
    }
    
    /// Broadcast multiply this matrix with a vector column-wise
    /// Vector is multiplied with each column of the matrix
    pub fn broadcast_mul_col_vector(&self, vector: &VectorF64) -> Result<ArrayF64, String> {
        if vector.len() != self.nrows() {
            return Err(format!(
                "Vector length {} does not match matrix rows {}",
                vector.len(), self.nrows()
            ));
        }
        
        let result = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            unsafe {
                self.inner.get_unchecked(i, j) * vector.inner.get_unchecked(i)
            }
        });
        
        Ok(ArrayF64 { inner: result })
    }
    
    /// Broadcast divide this matrix with a vector column-wise
    /// Each column of the matrix is divided by the vector
    pub fn broadcast_div_col_vector(&self, vector: &VectorF64) -> Result<ArrayF64, String> {
        if vector.len() != self.nrows() {
            return Err(format!(
                "Vector length {} does not match matrix rows {}",
                vector.len(), self.nrows()
            ));
        }
        
        let result = Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            unsafe {
                self.inner.get_unchecked(i, j) / vector.inner.get_unchecked(i)
            }
        });
        
        Ok(ArrayF64 { inner: result })
    }
    
    /// Broadcast with another matrix using element-wise operations
    /// 
    /// # Mathematical Specification
    /// Applies element-wise operation between two matrices following broadcasting rules:
    /// - Identical shapes: direct element-wise operation
    /// - Scalar (1×1) matrix: broadcast scalar to all elements of larger matrix
    /// 
    /// # For AI Code Generation
    /// - Handles both identical shapes and scalar broadcasting
    /// - Operations: Add, Sub, Mul, Div (element-wise, NOT matrix multiplication)
    /// - For matrix multiplication use ^ operator, not this function
    /// - Returns Result for dimension compatibility checking
    /// - More explicit than automatic broadcasting operators
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, broadcasting::BroadcastOp};
    /// 
    /// let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let scalar = ArrayF64::from_slice(&[2.0], 1, 1).unwrap();  // 1×1 "scalar" matrix
    /// 
    /// let doubled = data.broadcast_with_matrix(&scalar, BroadcastOp::Mul).unwrap();
    /// // Result: [[1*2, 2*2], [3*2, 4*2]] = [[2, 4], [6, 8]]
    /// 
    /// // Feature scaling with different scales per feature
    /// let scales = ArrayF64::from_slice(&[0.1, 10.0], 1, 2).unwrap();  // Row vector
    /// // This would need identical shape broadcasting (not implemented in this function)
    /// ```
    /// 
    /// # Errors
    /// Returns error for incompatible broadcasting shapes
    /// 
    /// # See Also
    /// - [`broadcast_add_row_vector`]: Vector-matrix row broadcasting
    /// - [`broadcast_add_col_vector`]: Vector-matrix column broadcasting
    /// - [`broadcasting_ops`]: Automatic broadcasting with standard operators
    pub fn broadcast_with_matrix(&self, other: &ArrayF64, op: BroadcastOp) -> Result<ArrayF64, String> {
        let self_shape = Shape::Matrix(self.nrows(), self.ncols());
        let other_shape = Shape::Matrix(other.nrows(), other.ncols());
        
        match broadcast_compatibility(&self_shape, &other_shape) {
            BroadcastType::Identical => {
                // Same shape - direct element-wise operation
                let result = match op {
                    BroadcastOp::Add => Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
                        unsafe { self.inner.get_unchecked(i, j) + other.inner.get_unchecked(i, j) }
                    }),
                    BroadcastOp::Sub => Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
                        unsafe { self.inner.get_unchecked(i, j) - other.inner.get_unchecked(i, j) }
                    }),
                    BroadcastOp::Mul => Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
                        unsafe { self.inner.get_unchecked(i, j) * other.inner.get_unchecked(i, j) }
                    }),
                    BroadcastOp::Div => Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
                        unsafe { self.inner.get_unchecked(i, j) / other.inner.get_unchecked(i, j) }
                    }),
                };
                Ok(ArrayF64 { inner: result })
            }
            BroadcastType::ScalarToMatrix => {
                // One matrix is 1x1 (scalar)
                if other.nrows() == 1 && other.ncols() == 1 {
                    let scalar = unsafe { other.inner.get_unchecked(0, 0) };
                    let result = match op {
                        BroadcastOp::Add => Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
                            unsafe { self.inner.get_unchecked(i, j) + scalar }
                        }),
                        BroadcastOp::Sub => Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
                            unsafe { self.inner.get_unchecked(i, j) - scalar }
                        }),
                        BroadcastOp::Mul => Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
                            unsafe { self.inner.get_unchecked(i, j) * scalar }
                        }),
                        BroadcastOp::Div => Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
                            unsafe { self.inner.get_unchecked(i, j) / scalar }
                        }),
                    };
                    Ok(ArrayF64 { inner: result })
                } else if self.nrows() == 1 && self.ncols() == 1 {
                    let scalar = unsafe { self.inner.get_unchecked(0, 0) };
                    let result = match op {
                        BroadcastOp::Add => Mat::from_fn(other.nrows(), other.ncols(), |i, j| {
                            unsafe { scalar + other.inner.get_unchecked(i, j) }
                        }),
                        BroadcastOp::Sub => Mat::from_fn(other.nrows(), other.ncols(), |i, j| {
                            unsafe { scalar - other.inner.get_unchecked(i, j) }
                        }),
                        BroadcastOp::Mul => Mat::from_fn(other.nrows(), other.ncols(), |i, j| {
                            unsafe { scalar * other.inner.get_unchecked(i, j) }
                        }),
                        BroadcastOp::Div => Mat::from_fn(other.nrows(), other.ncols(), |i, j| {
                            unsafe { scalar / other.inner.get_unchecked(i, j) }
                        }),
                    };
                    Ok(ArrayF64 { inner: result })
                } else {
                    Err("Invalid scalar broadcasting case".to_string())
                }
            }
            _ => Err(format!(
                "Incompatible broadcasting: Matrix({}, {}) with Matrix({}, {})",
                self.nrows(), self.ncols(), other.nrows(), other.ncols()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_broadcast_compatibility() {
        // Identical shapes
        assert_eq!(
            broadcast_compatibility(&Shape::Matrix(3, 3), &Shape::Matrix(3, 3)),
            BroadcastType::Identical
        );
        
        // Scalar broadcasting
        assert_eq!(
            broadcast_compatibility(&Shape::Matrix(1, 1), &Shape::Matrix(3, 3)),
            BroadcastType::ScalarToMatrix
        );
        
        // Vector to matrix broadcasting
        assert_eq!(
            broadcast_compatibility(&Shape::Vector(3), &Shape::Matrix(3, 4)),
            BroadcastType::VectorToMatrixCols
        );
        
        assert_eq!(
            broadcast_compatibility(&Shape::Vector(4), &Shape::Matrix(3, 4)),
            BroadcastType::VectorToMatrixRows
        );
    }
    
    #[test]
    fn test_matrix_vector_broadcasting() {
        let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let row_vector = VectorF64::from_slice(&[10.0, 20.0, 30.0]);
        
        let result = matrix.broadcast_add_row_vector(&row_vector).unwrap();
        
        assert_relative_eq!(result.get(0, 0).unwrap(), 11.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 1).unwrap(), 22.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 2).unwrap(), 33.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 0).unwrap(), 14.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 1).unwrap(), 25.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 2).unwrap(), 36.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_matrix_col_vector_broadcasting() {
        let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let col_vector = VectorF64::from_slice(&[100.0, 200.0]);
        
        
        let result = matrix.broadcast_mul_col_vector(&col_vector).unwrap();
        
        // Expected:
        // Matrix is [[1, 2], [3, 4]]
        // Col vector is [100, 200]
        // Result should be [[1*100, 2*100], [3*200, 4*200]] = [[100, 200], [600, 800]]
        assert_relative_eq!(result.get(0, 0).unwrap(), 100.0, epsilon = 1e-10);  // 1 * 100
        assert_relative_eq!(result.get(0, 1).unwrap(), 200.0, epsilon = 1e-10);  // 2 * 100
        assert_relative_eq!(result.get(1, 0).unwrap(), 600.0, epsilon = 1e-10);  // 3 * 200
        assert_relative_eq!(result.get(1, 1).unwrap(), 800.0, epsilon = 1e-10);  // 4 * 200
    }
    
    #[test]
    fn test_matrix_scalar_broadcasting() {
        let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let scalar_matrix = ArrayF64::from_slice(&[5.0], 1, 1).unwrap();
        
        let result = matrix.broadcast_with_matrix(&scalar_matrix, BroadcastOp::Add).unwrap();
        
        assert_relative_eq!(result.get(0, 0).unwrap(), 6.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 1).unwrap(), 7.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 0).unwrap(), 8.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 1).unwrap(), 9.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_broadcast_dimension_mismatch() {
        let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let wrong_vector = VectorF64::from_slice(&[1.0, 2.0, 3.0]); // length 3, should be 2
        
        assert!(matrix.broadcast_add_row_vector(&wrong_vector).is_err());
        assert!(matrix.broadcast_add_col_vector(&wrong_vector).is_err());
    }
}