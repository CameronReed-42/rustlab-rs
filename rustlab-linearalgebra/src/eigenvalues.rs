//! Eigenvalue and eigenvector computation with mathematical notation for AI code generation
//!
//! Provides high-performance eigenvalue decomposition for real matrices using faer 0.22
//! backend. Handles both general matrices (potentially complex eigenvalues) and
//! self-adjoint matrices (guaranteed real eigenvalues).
//!
//! # For AI Code Generation
//! - Extension trait pattern: methods added to ArrayF64 via EigenvalueOps
//! - Mathematical notation: A.eigenvalues(), A.eigenvectors(), A.eig()
//! - Complex eigenvalues: Use Complex<f64> for general matrices
//! - Real eigenvalues: Use specialized self-adjoint methods for symmetric matrices
//! - Error handling: All operations return Result<T> for robustness
//!
//! # Mathematical Specifications
//! - **General eigenvalue problem**: A·v = λ·v where λ may be complex
//! - **Self-adjoint eigenvalue problem**: A·v = λ·v where A = A^T, λ ∈ ℝ
//! - **Eigendecomposition**: A = V·Λ·V^{-1} where Λ is diagonal matrix of eigenvalues
//!
//! # Usage Patterns
//! ```
//! use rustlab_linearalgebra::{EigenvalueOps, ArrayF64, array64};
//! use num_complex::Complex;
//!
//! let A = array64![[1.0, 2.0], [0.0, 3.0]];
//!
//! // General matrix (may have complex eigenvalues)
//! let eigenvals: Vec<Complex<f64>> = A.eigenvalues()?;
//! let eig_decomp = A.eigenvectors()?;
//!
//! // Symmetric matrix (guaranteed real eigenvalues)
//! let symmetric = array64![[4.0, 2.0], [2.0, 3.0]];
//! let real_eigenvals: Vec<f64> = symmetric.eigenvalues_self_adjoint()?;
//! ```

use crate::{LinearAlgebraError, Result};
use num_complex::Complex;
use rustlab_math::ArrayF64;
use faer::Mat;

/// Eigenvalue decomposition result containing eigenvalues and eigenvectors
/// 
/// # Mathematical Specification
/// Represents eigendecomposition A = V·Λ·V^{-1} where:
/// - Λ: diagonal matrix with eigenvalues λ₁, λ₂, ..., λₙ
/// - V: matrix with corresponding eigenvectors as columns
/// 
/// # For AI Code Generation
/// - Access eigenvalues: decomp.eigenvalues (Vec<Complex<f64>>)
/// - Access eigenvectors: decomp.eigenvectors (Mat<Complex<f64>>)
/// - Handle complex eigenvalues properly with Complex<f64> type
/// - Eigenvector i corresponds to eigenvalue i
/// 
/// # Complex Eigenvalues
/// Real matrices can have complex eigenvalues in conjugate pairs.
/// Use .re and .im fields to access real and imaginary parts.
/// 
/// # Example
/// ```
/// let eig = A.eigenvectors()?;
/// 
/// for (i, eigenval) in eig.eigenvalues.iter().enumerate() {
///     if eigenval.im.abs() < 1e-10 {
///         println!("Real eigenvalue {}: {}", i, eigenval.re);
///     } else {
///         println!("Complex eigenvalue {}: {} + {}i", i, eigenval.re, eigenval.im);
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    /// Eigenvalues (may be complex for real input matrices)
    /// 
    /// Ordered by computational algorithm (not necessarily by magnitude).
    /// For real matrices, complex eigenvalues appear in conjugate pairs.
    pub eigenvalues: Vec<Complex<f64>>,
    
    /// Eigenvectors matrix with eigenvectors as columns
    /// 
    /// Column i is the eigenvector corresponding to eigenvalues[i].
    /// May be complex even for real input matrices if eigenvalues are complex.
    pub eigenvectors: Mat<Complex<f64>>,
}

/// Compute eigenvalues of a general square matrix
/// 
/// # Mathematical Specification
/// Solves characteristic equation det(A - λI) = 0 to find eigenvalues λ.
/// For real matrix A, eigenvalues may be complex and appear in conjugate pairs.
/// 
/// # Algorithm
/// Uses QR algorithm with implicit shifts for numerical stability.
/// Handles general real matrices (not restricted to symmetric).
/// 
/// # Complexity
/// - Time: O(n³) for n×n matrix
/// - Space: O(n) for eigenvalue storage
/// 
/// # For AI Code Generation
/// - Returns Complex<f64> to handle potentially complex eigenvalues
/// - Use for general matrices where eigenvalue structure unknown
/// - Check eigenval.im.abs() < tolerance to identify real eigenvalues
/// - For symmetric matrices, use eigenvalues_self_adjoint() for real output
/// 
/// # Mathematical Properties
/// - Trace: tr(A) = Σλᵢ (sum of eigenvalues)
/// - Determinant: det(A) = Πλᵢ (product of eigenvalues)
/// - Characteristic polynomial: p(λ) = det(A - λI)
/// 
/// # Example
/// ```
/// let A = array64![[1.0, 2.0], [0.0, 3.0]];  // Upper triangular
/// let eigenvals = eigenvalues(&A)?;
/// 
/// // For triangular matrices, eigenvalues are diagonal entries
/// assert!((eigenvals[0].re - 1.0).abs() < 1e-10);
/// assert!((eigenvals[1].re - 3.0).abs() < 1e-10);
/// 
/// // Check if eigenvalues are real
/// for eigenval in eigenvals {
///     if eigenval.im.abs() < 1e-12 {
///         println!("Real eigenvalue: {}", eigenval.re);
///     } else {
///         println!("Complex eigenvalue: {} + {}i", eigenval.re, eigenval.im);
///     }
/// }
/// ```
pub fn eigenvalues(matrix: &ArrayF64) -> Result<Vec<Complex<f64>>> {
    // Check if matrix is square
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    if rows != cols {
        return Err(LinearAlgebraError::NotSquare { rows, cols });
    }
    
    // Use faer 0.22's eigenvalue computation
    let eigenvals = matrix.as_faer().eigenvalues()
        .map_err(|_| LinearAlgebraError::decomposition_failed("Eigenvalue computation failed"))?;
    
    Ok(eigenvals)
}

/// Compute eigenvalues and eigenvectors of a general square matrix
/// 
/// # Mathematical Specification
/// Computes eigendecomposition A = V·Λ·V^{-1} where:
/// - Λ: diagonal matrix with eigenvalues λ₁, λ₂, ..., λₙ
/// - V: matrix with eigenvectors as columns (V[:,i] corresponds to λᵢ)
/// 
/// # Algorithm
/// Uses Schur decomposition followed by eigenvector computation.
/// Handles general real matrices with potentially complex eigenvalues/eigenvectors.
/// 
/// # Complexity
/// - Time: O(n³) for n×n matrix
/// - Space: O(n²) for eigenvector matrix storage
/// 
/// # For AI Code Generation
/// - Returns complete eigendecomposition for analysis
/// - Access: decomp.eigenvalues and decomp.eigenvectors
/// - Eigenvector i corresponds to eigenvalue i
/// - Handle complex eigenvectors properly for real matrices
/// 
/// # Numerical Properties
/// - Eigenvalue/eigenvector pairs satisfy: A·vᵢ = λᵢ·vᵢ
/// - For real matrices: complex eigenvalues/eigenvectors appear in conjugate pairs
/// - Eigenvectors may not be orthogonal for general matrices
/// 
/// # Verification
/// ```
/// let eig = eigenvectors(&A)?;
/// 
/// // Verify eigenvalue equation: A·v = λ·v
/// for i in 0..eig.eigenvalues.len() {
///     let lambda = eig.eigenvalues[i];
///     let v = eig.eigenvectors.col(i);
///     let Av = A.as_faer() * v;
///     let lambda_v = v * lambda;
///     // Should be approximately equal: ||Av - λv|| ≈ 0
/// }
/// ```
/// 
/// # Example
/// ```
/// let A = array64![[4.0, 1.0], [0.0, 2.0]];
/// let eig = eigenvectors(&A)?;
/// 
/// println!("Eigenvalues:");
/// for (i, eigenval) in eig.eigenvalues.iter().enumerate() {
///     println!("  λ_{} = {} + {}i", i, eigenval.re, eigenval.im);
/// }
/// 
/// println!("Eigenvectors (as columns):");
/// // eig.eigenvectors contains eigenvectors as columns
/// ```
pub fn eigenvectors(matrix: &ArrayF64) -> Result<EigenDecomposition> {
    // Check if matrix is square
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    if rows != cols {
        return Err(LinearAlgebraError::NotSquare { rows, cols });
    }
    
    // Use faer 0.22's eigenvalue computation with eigenvectors
    let evd = matrix.as_faer().eigen()
        .map_err(|_| LinearAlgebraError::decomposition_failed("Eigen decomposition failed"))?;
    
    // Extract eigenvalues
    let eigenvals = evd.S();
    let mut eigenvalues = Vec::with_capacity(rows);
    for i in 0..rows {
        eigenvalues.push(eigenvals[i]);
    }
    
    // Extract eigenvectors
    let eigenvecs = evd.U().to_owned();
    
    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors: eigenvecs,
    })
}

/// Compute eigenvalues for symmetric (self-adjoint) matrices with guaranteed real eigenvalues
/// 
/// # Mathematical Specification
/// For symmetric matrix A = A^T, computes real eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λₙ.
/// Exploits symmetry for improved performance and numerical stability.
/// 
/// # Requirements
/// - Matrix must be symmetric: A = A^T
/// - Algorithm assumes symmetry; only uses lower triangular part
/// 
/// # Algorithm
/// Uses specialized symmetric eigenvalue algorithm (typically divide-and-conquer).
/// More efficient and numerically stable than general eigenvalue computation.
/// 
/// # Complexity
/// - Time: O(n³) but with better constants than general case
/// - Space: O(n) for eigenvalue storage
/// - More efficient than general eigenvalues() for symmetric matrices
/// 
/// # For AI Code Generation
/// - Use when matrix is known to be symmetric
/// - Returns Vec<f64> (real eigenvalues) instead of Complex<f64>
/// - Eigenvalues returned in ascending order
/// - More efficient and stable than general eigenvalue computation
/// 
/// # Properties of Symmetric Matrices
/// - All eigenvalues are real
/// - Eigenvectors are orthogonal: V^T V = I
/// - Spectral decomposition: A = V·Λ·V^T where V is orthogonal
/// 
/// # Common Symmetric Matrices
/// - Covariance matrices
/// - Correlation matrices  
/// - Gram matrices A^T A
/// - Hessian matrices in optimization
/// 
/// # Example
/// ```
/// // Symmetric matrix (covariance-like)
/// let A = array64![[4.0, 2.0], [2.0, 3.0]];
/// let eigenvals = eigenvalues_self_adjoint(&A)?;
/// 
/// // Eigenvalues are real and in ascending order
/// assert!(eigenvals[0] <= eigenvals[1]);
/// 
/// // All eigenvalues are real (no imaginary part)
/// for lambda in eigenvals {
///     println!("Real eigenvalue: {}", lambda);
/// }
/// 
/// // For positive definite matrices, all eigenvalues > 0
/// if eigenvals.iter().all(|&x| x > 0.0) {
///     println!("Matrix is positive definite");
/// }
/// ```
pub fn eigenvalues_self_adjoint(matrix: &ArrayF64) -> Result<Vec<f64>> {
    // Check if matrix is square
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    if rows != cols {
        return Err(LinearAlgebraError::NotSquare { rows, cols });
    }
    
    // Use faer 0.22's self-adjoint eigenvalue computation
    let eigenvals = matrix.as_faer().self_adjoint_eigenvalues(faer::Side::Lower)
        .map_err(|_| LinearAlgebraError::decomposition_failed("Self-adjoint eigenvalue computation failed"))?;
    
    Ok(eigenvals)
}

/// Compute eigenvalues and orthogonal eigenvectors for symmetric matrices
/// 
/// # Mathematical Specification
/// Computes spectral decomposition A = V·Λ·V^T for symmetric matrix A where:
/// - Λ: diagonal matrix with real eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λₙ
/// - V: orthogonal matrix with eigenvectors as columns (V^T V = I)
/// 
/// # Requirements
/// - Matrix must be symmetric: A = A^T
/// - Algorithm uses only lower triangular part
/// 
/// # Algorithm
/// Uses specialized symmetric eigendecomposition (typically divide-and-conquer).
/// Guarantees orthogonal eigenvectors and real eigenvalues.
/// 
/// # Complexity
/// - Time: O(n³) with better constants than general case
/// - Space: O(n²) for orthogonal eigenvector matrix
/// 
/// # For AI Code Generation
/// - Use when matrix is symmetric and you need eigenvectors
/// - Returns (eigenvalues: Vec<f64>, eigenvectors: Mat<f64>)
/// - Eigenvectors are orthogonal: V^T V = I
/// - More efficient than general eigenvectors() for symmetric matrices
/// 
/// # Orthogonality Properties
/// - Eigenvectors are orthonormal: ||vᵢ|| = 1, vᵢ · vⱼ = 0 for i ≠ j
/// - Matrix V is orthogonal: V^T V = V V^T = I
/// - Spectral decomposition: A = Σᵢ λᵢ vᵢ vᵢ^T
/// 
/// # Applications
/// - Principal Component Analysis (PCA)
/// - Quadratic form diagonalization
/// - Optimization (Hessian analysis)
/// - Covariance matrix eigendecomposition
/// 
/// # Example
/// ```
/// // Symmetric positive definite matrix
/// let A = array64![[4.0, 2.0], [2.0, 3.0]];
/// let (eigenvals, eigenvecs) = eigenvectors_self_adjoint(&A)?;
/// 
/// // Eigenvalues in ascending order
/// assert!(eigenvals[0] <= eigenvals[1]);
/// 
/// // Verify spectral decomposition: A = V·Λ·V^T
/// let V = &eigenvecs;
/// let Lambda = Mat::from_fn(2, 2, |i, j| {
///     if i == j { eigenvals[i] } else { 0.0 }
/// });
/// let reconstructed = V * &Lambda * V.transpose();
/// // reconstructed should equal original A
/// 
/// // Verify orthogonality: V^T V = I
/// let identity_check = V.transpose() * V;
/// // Should be approximately identity matrix
/// ```
pub fn eigenvectors_self_adjoint(matrix: &ArrayF64) -> Result<(Vec<f64>, Mat<f64>)> {
    // Check if matrix is square
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    if rows != cols {
        return Err(LinearAlgebraError::NotSquare { rows, cols });
    }
    
    // Use faer 0.22's self-adjoint eigenvalue computation
    let evd = matrix.as_faer().self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| LinearAlgebraError::decomposition_failed("Self-adjoint eigen decomposition failed"))?;
    
    // Extract eigenvalues (real, in nondecreasing order)
    let eigenvals = evd.S();
    let mut eigenvalues = Vec::with_capacity(rows);
    for i in 0..rows {
        eigenvalues.push(eigenvals[i]);
    }
    
    // Extract eigenvectors (real, orthogonal)
    let eigenvecs = evd.U().to_owned();
    
    Ok((eigenvalues, eigenvecs))
}

/// Convenience alias for complete eigendecomposition (eigenvalues + eigenvectors)
/// 
/// # Mathematical Specification
/// Identical to eigenvectors() - computes A = V·Λ·V^{-1} for general matrices.
/// Provided for compatibility with NumPy/MATLAB naming conventions.
/// 
/// # For AI Code Generation
/// - Use eig() for concise notation matching NumPy: np.linalg.eig(A)
/// - Returns same EigenDecomposition as eigenvectors()
/// - Handles general matrices (potentially complex eigenvalues)
/// 
/// # Example
/// ```
/// let A = array64![[1.0, 2.0], [0.0, 3.0]];
/// 
/// // These are equivalent:
/// let eig1 = eig(&A)?;
/// let eig2 = eigenvectors(&A)?;
/// 
/// // Access eigenvalues and eigenvectors
/// let eigenvals = eig1.eigenvalues;
/// let eigenvecs = eig1.eigenvectors;
/// ```
pub fn eig(matrix: &ArrayF64) -> Result<EigenDecomposition> {
    eigenvectors(matrix)
}

/// Extension trait providing eigenvalue computation methods with mathematical notation
/// 
/// # For AI Code Generation
/// This trait adds eigenvalue methods directly to ArrayF64 matrices:
/// - Use concise notation: A.eigenvalues(), A.eigenvectors(), A.eig()
/// - Choose appropriate method based on matrix properties
/// - All methods return Result<T> for error handling
/// 
/// # Method Selection Guide
/// | Method | Best For | Output Type | Requirements |
/// |--------|----------|-------------|-------------|
/// | `eigenvalues()` | General matrices | Complex<f64> | Square matrix |
/// | `eigenvalues_self_adjoint()` | Symmetric matrices | f64 | Symmetric matrix |
/// | `eigenvectors()` | Need eigenvectors | EigenDecomposition | Square matrix |
/// | `eigenvectors_self_adjoint()` | Symmetric + eigenvectors | (Vec<f64>, Mat<f64>) | Symmetric matrix |
/// | `eig()` | NumPy compatibility | EigenDecomposition | Square matrix |
/// 
/// # Performance Hierarchy
/// 1. `eigenvalues_self_adjoint()` - fastest for symmetric matrices
/// 2. `eigenvalues()` - general case, eigenvalues only
/// 3. `eigenvectors_self_adjoint()` - symmetric matrices with eigenvectors
/// 4. `eigenvectors()` / `eig()` - general case with eigenvectors
/// 
/// # Example
/// ```
/// use rustlab_linearalgebra::{EigenvalueOps, ArrayF64, array64};
/// 
/// let A = array64![[4.0, 2.0], [2.0, 3.0]];
/// 
/// // For symmetric matrices, use specialized methods
/// if is_symmetric(&A) {
///     let eigenvals = A.eigenvalues_self_adjoint()?;  // Vec<f64>
///     let (vals, vecs) = A.eigenvectors_self_adjoint()?;
/// } else {
///     let eigenvals = A.eigenvalues()?;  // Vec<Complex<f64>>
///     let eig_decomp = A.eig()?;
/// }
/// ```
pub trait EigenvalueOps {
    /// Compute eigenvalues for general square matrices (potentially complex)
    /// 
    /// # Output: Vec<Complex<f64>>
    /// Complex eigenvalues to handle general real matrices.
    /// Use .re and .im to access real and imaginary parts.
    /// 
    /// # Use when:
    /// - Matrix structure unknown or non-symmetric
    /// - Only eigenvalues needed (more efficient than eigenvectors)
    /// - OK with complex output type
    fn eigenvalues(&self) -> Result<Vec<Complex<f64>>>;
    
    /// Compute complete eigendecomposition (eigenvalues + eigenvectors)
    /// 
    /// # Output: EigenDecomposition
    /// Contains both eigenvalues and eigenvectors in single structure.
    /// Eigenvectors may be complex even for real input matrices.
    /// 
    /// # Use when:
    /// - Need both eigenvalues and eigenvectors
    /// - General matrices (potentially non-symmetric)
    /// - Can handle complex eigenvectors
    fn eigenvectors(&self) -> Result<EigenDecomposition>;
    
    /// Compute real eigenvalues for symmetric matrices (more efficient)
    /// 
    /// # Output: Vec<f64>
    /// Guaranteed real eigenvalues in ascending order.
    /// More efficient than general eigenvalue computation.
    /// 
    /// # Use when:
    /// - Matrix is symmetric: A = A^T
    /// - Want real eigenvalues without complex arithmetic
    /// - Only eigenvalues needed (not eigenvectors)
    fn eigenvalues_self_adjoint(&self) -> Result<Vec<f64>>;
    
    /// Compute real eigenvalues and orthogonal eigenvectors for symmetric matrices
    /// 
    /// # Output: (Vec<f64>, Mat<f64>)
    /// Tuple of (eigenvalues, orthogonal_eigenvectors).
    /// Eigenvectors are guaranteed orthonormal.
    /// 
    /// # Use when:
    /// - Matrix is symmetric: A = A^T
    /// - Need both eigenvalues and eigenvectors
    /// - Want real arithmetic and orthogonal eigenvectors
    fn eigenvectors_self_adjoint(&self) -> Result<(Vec<f64>, Mat<f64>)>;
    
    /// Convenience alias for eigenvectors() (NumPy/MATLAB compatibility)
    /// 
    /// # Output: EigenDecomposition
    /// Identical to eigenvectors() - provided for familiar naming.
    /// 
    /// # Use when:
    /// - Porting from NumPy: np.linalg.eig(A) → A.eig()
    /// - Prefer concise notation
    fn eig(&self) -> Result<EigenDecomposition>;
}

impl EigenvalueOps for ArrayF64 {
    fn eigenvalues(&self) -> Result<Vec<Complex<f64>>> {
        eigenvalues(self)
    }
    
    fn eigenvectors(&self) -> Result<EigenDecomposition> {
        eigenvectors(self)
    }
    
    fn eigenvalues_self_adjoint(&self) -> Result<Vec<f64>> {
        eigenvalues_self_adjoint(self)
    }
    
    fn eigenvectors_self_adjoint(&self) -> Result<(Vec<f64>, Mat<f64>)> {
        eigenvectors_self_adjoint(self)
    }
    
    fn eig(&self) -> Result<EigenDecomposition> {
        eig(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::array64;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_eigenvalues_2x2() {
        // Identity matrix should have eigenvalues [1, 1]
        let matrix = array64![
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        
        let eigenvals = matrix.eigenvalues().unwrap();
        
        // Check dimensions
        assert_eq!(eigenvals.len(), 2);
        
        // For identity matrix, eigenvalues should be close to 1
        for eigenval in eigenvals {
            assert_relative_eq!(eigenval.re, 1.0, epsilon = 1e-10);
            assert_relative_eq!(eigenval.im, 0.0, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_eigenvalues_self_adjoint() {
        // Simple diagonal matrix (symmetric)
        let matrix = array64![
            [2.0, 0.0],
            [0.0, 3.0]
        ];
        
        let eigenvals = matrix.eigenvalues_self_adjoint().unwrap();
        
        // Check dimensions
        assert_eq!(eigenvals.len(), 2);
        
        // For diagonal matrix, eigenvalues should be the diagonal elements
        // (in nondecreasing order for self-adjoint)
        assert_relative_eq!(eigenvals[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(eigenvals[1], 3.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_eigenvectors_computation() {
        // Simple diagonal matrix
        let matrix = array64![
            [2.0, 0.0],
            [0.0, 3.0]
        ];
        
        let eig = matrix.eigenvectors().unwrap();
        
        // Check dimensions
        assert_eq!(eig.eigenvalues.len(), 2);
        assert_eq!(eig.eigenvectors.nrows(), 2);
        assert_eq!(eig.eigenvectors.ncols(), 2);
    }
    
    #[test]
    fn test_eigenvectors_self_adjoint() {
        // Simple symmetric matrix
        let matrix = array64![
            [4.0, 2.0],
            [2.0, 1.0]
        ];
        
        let (eigenvals, eigenvecs) = matrix.eigenvectors_self_adjoint().unwrap();
        
        // Check dimensions
        assert_eq!(eigenvals.len(), 2);
        assert_eq!(eigenvecs.nrows(), 2);
        assert_eq!(eigenvecs.ncols(), 2);
        
        // Eigenvalues should be real and in nondecreasing order
        assert!(eigenvals[0] <= eigenvals[1]);
    }
}