//! Essential matrix operations with mathematical notation and high performance
//! 
//! This module extends rustlab-math Array types with fundamental linear algebra
//! operations using concise mathematical notation. All operations are optimized
//! using the faer backend for maximum performance.
//!
//! # For AI Code Generation
//! - Extension trait pattern: methods added to existing Array<T> types
//! - Mathematical notation: A.T(), A.det(), A.inv() mirror textbook usage
//! - Error handling: All operations return Result<T> for dimension/singularity checks
//! - Performance: O(n³) algorithms with optimized BLAS-like implementations

use faer::Mat;
use faer::prelude::Solve;
use rustlab_math::Array;
use crate::{LinearAlgebraError, Result};
use faer_entity::{Entity, ComplexField as FaerComplexField};
use faer_traits::ComplexField;

/// Essential linear algebra operations with mathematical notation for matrices
/// 
/// # Mathematical Specification
/// Provides fundamental matrix operations:
/// - Transpose: A.T() computes A^T where (A^T)_{ij} = A_{ji}
/// - Determinant: A.det() computes det(A) for square matrices
/// - Inverse: A.inv() computes A^{-1} such that A·A^{-1} = I
/// 
/// # Dimensions
/// - Transpose: (m×n) → (n×m)
/// - Determinant: (n×n) → scalar, requires square matrix
/// - Inverse: (n×n) → (n×n), requires square invertible matrix
/// 
/// # Complexity
/// - Transpose: O(mn) time, O(mn) space
/// - Determinant: O(n³) via LU decomposition
/// - Inverse: O(n³) via LU decomposition with back substitution
/// 
/// # For AI Code Generation
/// - Use .T() instead of .transpose() for concise mathematical notation
/// - Check .det() ≠ 0 before computing .inv() to avoid singular matrices
/// - All operations return Result<T> - use ? operator for error propagation
/// - Methods work on any Array<T> where T implements required traits
/// - Common uses: solving Ax=b via x = A.inv() * b (though A.solve(b) is preferred)
/// 
/// # Example
/// ```
/// use rustlab_linearalgebra::{BasicLinearAlgebra, ArrayF64};
/// use rustlab_math::array64;
/// 
/// let A = array64![[1.0, 2.0], [3.0, 4.0]];
/// 
/// // Matrix transpose: A.T()
/// let At = A.T();
/// assert_eq!(At.get(0, 1), Some(3.0));  // (A.T())[0,1] = A[1,0]
/// 
/// // Determinant: det(A) = ad - bc
/// let det = A.det()?;
/// assert!((det - (-2.0)).abs() < 1e-10);  // 1*4 - 2*3 = -2
/// 
/// // Matrix inverse: A^{-1}
/// let A_inv = A.inv()?;
/// 
/// // Verify: A * A^{-1} = I (approximately)
/// let identity_check = &A ^ &A_inv;  // Matrix multiplication
/// assert!((identity_check.get(0,0).unwrap() - 1.0).abs() < 1e-10);
/// assert!((identity_check.get(1,1).unwrap() - 1.0).abs() < 1e-10);
/// ```
/// 
/// # Errors
/// - `DimensionMismatch`: Matrix not square for det() or inv().
///   Fix: Ensure matrix.nrows() == matrix.ncols()
/// - `Singular`: Matrix not invertible for inv().
///   Fix: Check det() ≠ 0 or use pseudoinverse for rank-deficient matrices
/// 
/// # See Also
/// - [`LinearSystemOps::solve_system`]: Preferred over inv() for solving Ax=b
/// - [`DecompositionMethods::lu`]: More efficient for multiple operations
pub trait BasicLinearAlgebra<T: Entity + ComplexField> {
    /// Compute matrix transpose using concise mathematical notation A.T()
    /// 
    /// # Mathematical Specification
    /// Given matrix A ∈ ℝ^{m×n}, computes A^T ∈ ℝ^{n×m} where (A^T)_{ij} = A_{ji}
    /// 
    /// # Dimensions
    /// - Input: A (m×n)
    /// - Output: A^T (n×m)
    /// 
    /// # Complexity
    /// - Time: O(mn) for copying elements
    /// - Space: O(mn) for result matrix
    /// 
    /// # For AI Code Generation
    /// - Returns new Array<T>, does not modify original
    /// - Use A.T() instead of A.transpose() for mathematical clarity
    /// - Works for any dimensions (not restricted to square matrices)
    /// - Common uses: normal equations A^T A, matrix multiplication preparation
    fn T(&self) -> Array<T>;
    
    /// Compute determinant of square matrix using LU decomposition
    /// 
    /// # Mathematical Specification
    /// For square matrix A ∈ ℝ^{n×n}, computes det(A) ∈ ℝ
    /// det(A) = product of diagonal elements of U in PA = LU decomposition
    /// 
    /// # Dimensions
    /// - Input: A (n×n) - must be square
    /// - Output: scalar det(A)
    /// 
    /// # Complexity
    /// - Time: O(n³) via LU decomposition
    /// - Space: O(n²) for decomposition storage
    /// 
    /// # For AI Code Generation
    /// - Returns Result<T> - matrix must be square or error
    /// - det(A) = 0 indicates singular (non-invertible) matrix
    /// - Use for checking invertibility before computing inverse
    /// - Common uses: volume scaling, invertibility test, characteristic polynomial
    /// - For 2×2: det = ad - bc, for larger matrices use this method
    fn det(&self) -> Result<T>;
    
    /// Compute matrix inverse A^{-1} using LU decomposition with partial pivoting
    /// 
    /// # Mathematical Specification
    /// For invertible matrix A ∈ ℝ^{n×n}, computes A^{-1} such that:
    /// A·A^{-1} = A^{-1}·A = I (identity matrix)
    /// 
    /// # Dimensions
    /// - Input: A (n×n) - must be square and invertible
    /// - Output: A^{-1} (n×n)
    /// 
    /// # Complexity
    /// - Time: O(n³) for LU decomposition + O(n³) for back substitution
    /// - Space: O(n²) for result matrix
    /// 
    /// # For AI Code Generation
    /// - Returns Result<Array<T>> - fails if matrix singular or not square
    /// - Prefer A.solve_system(b) over A.inv() * b for efficiency
    /// - Check det(A) ≠ 0 before calling to avoid singular matrix errors
    /// - Use for analytical derivations, not numerical computation
    /// - Common uses: solving multiple systems, theoretical calculations
    fn inv(&self) -> Result<Array<T>>;
}

impl<T: Entity + ComplexField + FaerComplexField> BasicLinearAlgebra<T> for Array<T> {
    fn T(&self) -> Array<T> {
        self.transpose()
    }
    
    fn det(&self) -> Result<T> {
        if self.nrows() != self.ncols() {
            return Err(LinearAlgebraError::DimensionMismatch {
                expected: format!("square matrix (n×n)"),
                actual: format!("{}×{}", self.nrows(), self.ncols()),
            });
        }

        // Use faer's built-in determinant method
        let det = self.as_faer().determinant();
        Ok(det)
    }
    
    fn inv(&self) -> Result<Array<T>> {
        if self.nrows() != self.ncols() {
            return Err(LinearAlgebraError::DimensionMismatch {
                expected: format!("square matrix (n×n)"),
                actual: format!("{}×{}", self.nrows(), self.ncols()),
            });
        }

        // Check if matrix is singular by computing determinant
        let det = self.as_faer().determinant();
        let tolerance = T::faer_from_f64(1e-12);
        if det.faer_abs() < tolerance.faer_abs() {
            return Err(LinearAlgebraError::Singular);
        }

        // Use faer's LU decomposition to compute inverse
        let lu = self.as_faer().partial_piv_lu();
        
        // Create identity matrix
        let n = self.nrows();
        let identity = Mat::from_fn(n, n, |i, j| {
            if i == j { T::faer_one() } else { T::faer_zero() }
        });
        
        // Solve A * X = I to get X = A^(-1)
        let inverse_mat = lu.solve(&identity);
        Ok(Array::from_faer(inverse_mat))
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::ArrayF64;
    use approx::assert_relative_eq;

    #[test]
    fn test_transpose() {
        let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let transposed = matrix.T();
        
        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed.get(0, 0), Some(1.0));
        assert_eq!(transposed.get(1, 0), Some(2.0));
        assert_eq!(transposed.get(2, 0), Some(3.0));
        assert_eq!(transposed.get(0, 1), Some(4.0));
    }

    #[test]
    fn test_determinant_2x2() {
        let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let det = matrix.det().unwrap();
        
        // det = 1*4 - 2*3 = 4 - 6 = -2
        assert_relative_eq!(det, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant_identity() {
        let identity = ArrayF64::from_slice(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let det = identity.det().unwrap();
        
        assert_relative_eq!(det, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_2x2() {
        let matrix = ArrayF64::from_slice(&[2.0, 0.0, 0.0, 2.0], 2, 2).unwrap();
        let inverse = matrix.inv().unwrap();
        
        assert_relative_eq!(inverse.get(0, 0).unwrap(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(inverse.get(1, 1).unwrap(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(inverse.get(0, 1).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(inverse.get(1, 0).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_identity() {
        let identity = ArrayF64::from_slice(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let inverse = identity.inv().unwrap();
        
        // Identity matrix should be its own inverse
        assert_relative_eq!(inverse.get(0, 0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(inverse.get(1, 1).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(inverse.get(0, 1).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(inverse.get(1, 0).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_non_square_matrix_error() {
        let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        
        // Determinant should fail for non-square matrix
        assert!(matrix.det().is_err());
        
        // Inverse should fail for non-square matrix
        assert!(matrix.inv().is_err());
    }

    #[test]
    fn test_singular_matrix_error() {
        // Create singular matrix (zero determinant)
        let singular = ArrayF64::from_slice(&[1.0, 2.0, 2.0, 4.0], 2, 2).unwrap();
        
        // Determinant should be zero
        let det = singular.det().unwrap();
        assert_relative_eq!(det, 0.0, epsilon = 1e-10);
        
        // Inverse should fail for singular matrix
        assert!(singular.inv().is_err());
    }
}