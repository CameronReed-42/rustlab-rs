//! High-performance matrix decompositions with mathematical notation for AI code generation
//!
//! Provides optimized implementations of essential matrix decompositions using faer 0.22
//! backend. All decompositions follow mathematical conventions and return reusable
//! decomposition objects for efficient multiple solves.
//!
//! # For AI Code Generation
//! - Extension trait pattern: methods added to ArrayF64 via DecompositionMethods
//! - Mathematical notation: A.lu(), A.qr(), A.cholesky(), A.svd()
//! - Reusable decompositions: factor once, solve many times
//! - Error handling: All operations return Result<T> for robustness
//! - Performance hierarchy: Cholesky < LU < QR < SVD (increasing cost)
//!
//! # Mathematical Specifications
//! - **LU**: A = PLU where P is permutation, L lower triangular, U upper triangular
//! - **QR**: A = QR where Q is orthogonal, R is upper triangular  
//! - **Cholesky**: A = L·L^T where L is lower triangular (SPD matrices only)
//! - **SVD**: A = U·Σ·V^T where U,V orthogonal, Σ diagonal with singular values
//!
//! # Usage Patterns
//! ```
//! use rustlab_linearalgebra::{DecompositionMethods, ArrayF64, array64};
//!
//! let A = array64![[4.0, 2.0], [2.0, 3.0]];
//! let b = array64![[8.0], [7.0]];
//!
//! // Factor once, solve many
//! let chol = A.cholesky()?;  // O(n³/3) for SPD matrices
//! let x1 = chol.solve(&b)?;  // O(n²) per solve
//! let x2 = chol.solve(&other_b)?;
//!
//! // Use appropriate decomposition for problem type
//! let lu = A.lu()?;         // General square matrices
//! let qr = A.qr()?;         // Overdetermined systems (m > n)
//! let svd = A.svd()?;       // Rank-deficient, pseudoinverse
//! ```

use crate::{LinearAlgebraError, Result};
use faer::prelude::*;
use faer::linalg::solvers::*;
use rustlab_math::ArrayF64;

/// LU decomposition result for efficient multiple solves with same matrix
/// 
/// # Mathematical Specification
/// Stores PA = LU factorization where:
/// - P: permutation matrix for numerical stability
/// - L: lower triangular matrix with unit diagonal
/// - U: upper triangular matrix
/// 
/// # For AI Code Generation
/// - Reusable handle: factor once with A.lu(), solve many times
/// - Use solve() method for each right-hand side
/// - More efficient than A.inv() * b for solving linear systems
/// - Handles general square matrices (not restricted to SPD)
/// 
/// # Complexity
/// - Factorization: O(n³) - done once during A.lu()
/// - Each solve: O(n²) via forward/back substitution
/// 
/// # Example
/// ```
/// let A = array64![[2.0, 1.0], [1.0, 1.0]];
/// let lu = A.lu()?;  // Factor once
/// 
/// // Solve multiple systems efficiently
/// let x1 = lu.solve(&b1)?;  // Ax1 = b1
/// let x2 = lu.solve(&b2)?;  // Ax2 = b2
/// ```
#[derive(Debug)]
pub struct LuDecomposition {
    decomposition: PartialPivLu<f64>,
}

/// QR decomposition result for least squares and overdetermined systems
/// 
/// # Mathematical Specification
/// Stores A = QR factorization where:
/// - Q: orthogonal matrix (Q^T Q = I)
/// - R: upper triangular matrix
/// 
/// # For AI Code Generation
/// - Best for overdetermined systems (m > n) where A is m×n with m > n
/// - Provides least squares solution: min ||Ax - b||₂
/// - More stable than normal equations A^T A x = A^T b
/// - Use solve() method for minimum norm solutions
/// 
/// # Complexity
/// - Factorization: O(mn²) for m×n matrix
/// - Each solve: O(mn) for least squares solution
/// 
/// # Example
/// ```
/// // Overdetermined system: more equations than unknowns
/// let A = array64![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];  // 3×2 matrix
/// let b = array64![[2.0], [3.0], [5.0]];
/// 
/// let qr = A.qr()?;  // Factor once
/// let x = qr.solve(&b)?;  // Least squares solution
/// ```
#[derive(Debug)]
pub struct QrDecomposition {
    decomposition: Qr<f64>,
}

/// Cholesky decomposition for symmetric positive definite matrices
/// 
/// # Mathematical Specification
/// Stores A = L·L^T factorization where:
/// - L: lower triangular matrix with positive diagonal entries
/// - Only valid for symmetric positive definite (SPD) matrices
/// 
/// # For AI Code Generation
/// - Fastest decomposition: O(n³/3) vs O(n³) for LU
/// - More numerically stable than LU for SPD matrices
/// - Fails gracefully: returns NotPositiveDefinite error if matrix not SPD
/// - Use when you know matrix is SPD (covariance, Gram matrices, etc.)
/// 
/// # Complexity
/// - Factorization: O(n³/3) - roughly 2x faster than LU
/// - Each solve: O(n²) via forward/back substitution
/// - Memory: O(n²) but can overwrite input matrix
/// 
/// # Requirements
/// Matrix A must be:
/// - Symmetric: A = A^T
/// - Positive definite: x^T A x > 0 for all x ≠ 0
/// 
/// # Example
/// ```
/// // Positive definite matrix (e.g., from A^T A)
/// let A = array64![[4.0, 2.0], [2.0, 3.0]];
/// 
/// match A.cholesky() {
///     Ok(chol) => {
///         let x = chol.solve(&b)?;  // Fast, stable solve
///     },
///     Err(NotPositiveDefinite) => {
///         // Fall back to LU for general matrices
///         let lu = A.lu()?;
///         let x = lu.solve(&b)?;
///     }
/// }
/// ```
#[derive(Debug)]
pub struct CholeskyDecomposition {
    decomposition: Llt<f64>,
}

/// Singular Value Decomposition for rank analysis and pseudoinverse
/// 
/// # Mathematical Specification
/// Stores A = U·Σ·V^T factorization where:
/// - U: m×m orthogonal matrix (left singular vectors)
/// - Σ: m×n diagonal matrix with non-negative singular values
/// - V^T: n×n orthogonal matrix (right singular vectors, transposed)
/// 
/// # For AI Code Generation
/// - Most general decomposition: works for any m×n matrix
/// - Use for rank-deficient matrices, pseudoinverse, PCA
/// - Access components: svd.u(), svd.singular_values(), svd.vt()
/// - Solve systems: use solve() for minimum norm least squares
/// 
/// # Complexity
/// - Factorization: O(min(mn², m²n)) - most expensive decomposition
/// - Memory: O(mn + m² + n²) for storing U, Σ, V^T
/// - Use only when other decompositions insufficient
/// 
/// # Applications
/// - Rank determination: count non-zero singular values
/// - Pseudoinverse: A⁺ = V·Σ⁺·U^T where Σ⁺ inverts non-zero values
/// - Principal Component Analysis (PCA)
/// - Matrix approximation: keep largest singular values
/// 
/// # Example
/// ```
/// let A = array64![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];  // 3×2 matrix
/// 
/// let svd = A.svd()?;
/// let u = svd.u();                    // 3×3 left singular vectors
/// let s = svd.singular_values();      // 2 singular values (σ₁ ≥ σ₂ ≥ 0)
/// let vt = svd.vt();                  // 2×2 right singular vectors^T
/// 
/// // Check rank: count σᵢ > tolerance
/// let rank = s.iter().filter(|&&x| x > 1e-12).count();
/// ```
#[derive(Debug)]
pub struct SvdDecomposition {
    decomposition: Svd<f64>,
}

impl LuDecomposition {
    /// Solve linear system Ax = b using stored LU factorization
    /// 
    /// # Mathematical Specification
    /// Solves Ax = b where A was previously factored as PA = LU.
    /// Uses forward substitution (Ly = Pb) then back substitution (Ux = y).
    /// 
    /// # Dimensions
    /// - A: n×n (must be square, from original factorization)
    /// - b: n×k (can solve multiple right-hand sides)
    /// - x: n×k (solution matrix)
    /// 
    /// # Complexity
    /// - Time: O(n²k) for n×k right-hand side
    /// - Space: O(nk) for solution storage
    /// 
    /// # For AI Code Generation
    /// - Reuse factorization: call this method multiple times with different b
    /// - More efficient than A.inv() * b for solving systems
    /// - Handles general square matrices (not restricted to special structure)
    /// - Use when you have multiple right-hand sides for same A
    /// 
    /// # Example
    /// ```
    /// let A = array64![[2.0, 1.0], [1.0, 1.0]];
    /// let lu = A.lu()?;  // Factor once: O(n³)
    /// 
    /// // Solve multiple systems efficiently
    /// let b1 = array64![[3.0], [2.0]];
    /// let x1 = lu.solve(&b1)?;  // O(n²)
    /// 
    /// let b2 = array64![[5.0], [3.0]];
    /// let x2 = lu.solve(&b2)?;  // O(n²)
    /// ```
    pub fn solve(&self, b: &ArrayF64) -> Result<ArrayF64> {
        let solution = self.decomposition.solve(b.as_faer());
        Ok(ArrayF64::from_faer(solution))
    }
    
    // Note: determinant method not available in faer 0.22 PartialPivLu
    // Future enhancement: add determinant computation from LU factors
    // det(A) = det(P) * ∏ᵢ U[i,i] where det(P) = ±1 from permutation
}

impl QrDecomposition {
    /// Solve least squares problem min ||Ax - b||₂ using QR factorization
    /// 
    /// # Mathematical Specification
    /// For overdetermined system Ax = b where A is m×n with m ≥ n:
    /// - If m = n: exact solution (if A invertible)
    /// - If m > n: least squares solution minimizing ||Ax - b||₂
    /// 
    /// Uses QR factorization A = QR to solve R^T Q^T Ax = R^T Q^T b
    /// 
    /// # Dimensions
    /// - A: m×n (from original factorization, m ≥ n recommended)
    /// - b: m×k (compatible with A's row dimension)
    /// - x: n×k (solution in column space of A)
    /// 
    /// # Complexity
    /// - Time: O(mnk) for m×n matrix and n×k right-hand side
    /// - Space: O(nk) for solution storage
    /// 
    /// # For AI Code Generation
    /// - Best for overdetermined systems (more equations than unknowns)
    /// - More stable than normal equations A^T A x = A^T b
    /// - Automatically provides least squares solution
    /// - Use when A is tall-skinny (m >> n)
    /// 
    /// # Example
    /// ```
    /// // Overdetermined: fit line y = ax + b to 3 points
    /// let A = array64![[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]];  // [x, 1] design matrix
    /// let y = array64![[1.0], [3.0], [4.0]];  // y values
    /// 
    /// let qr = A.qr()?;
    /// let coeffs = qr.solve(&y)?;  // [slope, intercept] least squares fit
    /// ```
    pub fn solve(&self, b: &ArrayF64) -> Result<ArrayF64> {
        let solution = self.decomposition.solve(b.as_faer());
        Ok(ArrayF64::from_faer(solution))
    }
}

impl CholeskyDecomposition {
    /// Solve symmetric positive definite system Ax = b using Cholesky factorization
    /// 
    /// # Mathematical Specification
    /// Solves Ax = b where A = L·L^T was factored via Cholesky decomposition.
    /// Uses forward substitution (Ly = b) then back substitution (L^T x = y).
    /// 
    /// # Dimensions
    /// - A: n×n symmetric positive definite (from original factorization)
    /// - b: n×k (can solve multiple right-hand sides)
    /// - x: n×k (solution matrix)
    /// 
    /// # Complexity
    /// - Time: O(n²k) for n×k right-hand side
    /// - Space: O(nk) for solution storage
    /// - Roughly 2x faster than LU solve due to triangular structure
    /// 
    /// # For AI Code Generation
    /// - Fastest option for SPD matrices (covariance, Gram matrices)
    /// - More numerically stable than LU for well-conditioned SPD systems
    /// - Use when matrix structure guarantees positive definiteness
    /// - Graceful fallback: if cholesky() fails, use lu() instead
    /// 
    /// # Numerical Properties
    /// - No pivoting needed (inherently stable for SPD matrices)
    /// - Condition number: κ(A) = λmax/λmin where λ are eigenvalues
    /// - All eigenvalues positive for SPD matrices
    /// 
    /// # Example
    /// ```
    /// // Covariance matrix (always SPD when full rank)
    /// let data = array64![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let cov = &data.T() ^ &data;  // A^T A is always positive semi-definite
    /// 
    /// let chol = cov.cholesky()?;  // Succeeds for SPD matrices
    /// let x = chol.solve(&b)?;     // Fast, stable solve
    /// ```
    pub fn solve(&self, b: &ArrayF64) -> Result<ArrayF64> {
        let solution = self.decomposition.solve(b.as_faer());
        Ok(ArrayF64::from_faer(solution))
    }
}

impl SvdDecomposition {
    /// Get left singular vectors matrix U
    /// 
    /// # Mathematical Specification
    /// Returns U from A = U·Σ·V^T where U is m×m orthogonal matrix.
    /// Columns of U are left singular vectors, spanning column space of A.
    /// 
    /// # Dimensions
    /// - For m×n input matrix A: returns m×m matrix U
    /// - U^T U = I (orthogonal)
    /// 
    /// # For AI Code Generation
    /// - Use for PCA: U contains principal components
    /// - Column i corresponds to singular value σᵢ
    /// - First r columns span column space of A (r = rank)
    /// 
    /// # Example
    /// ```
    /// let svd = A.svd()?;
    /// let u = svd.u();  // m×m left singular vectors
    /// // u.column(0) is first principal component
    /// ```
    pub fn u(&self) -> ArrayF64 {
        ArrayF64::from_faer(self.decomposition.U().to_owned())
    }
    
    /// Get singular values in descending order
    /// 
    /// # Mathematical Specification
    /// Returns diagonal elements σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0 from Σ matrix.
    /// Number of non-zero values indicates numerical rank of matrix.
    /// 
    /// # For AI Code Generation
    /// - Use for rank determination: count values > tolerance
    /// - Condition number: κ(A) = σ₁/σᵣ (largest/smallest non-zero)
    /// - Matrix approximation: keep largest k values for rank-k approximation
    /// - Relative importance: σᵢ²/Σσⱼ² gives energy in component i
    /// 
    /// # Example
    /// ```
    /// let svd = A.svd()?;
    /// let s = svd.singular_values();
    /// 
    /// // Determine numerical rank
    /// let tolerance = 1e-12;
    /// let rank = s.iter().filter(|&&x| x > tolerance).count();
    /// 
    /// // Condition number
    /// let condition = s[0] / s.last().unwrap();
    /// ```
    pub fn singular_values(&self) -> Vec<f64> {
        let s_diag = self.decomposition.S();
        let dim = s_diag.dim();
        let mut values = Vec::with_capacity(dim);
        for i in 0..dim {
            values.push(s_diag[i]);
        }
        values
    }
    
    /// Get right singular vectors matrix V^T (transposed)
    /// 
    /// # Mathematical Specification
    /// Returns V^T from A = U·Σ·V^T where V^T is n×n orthogonal matrix.
    /// Rows of V^T are right singular vectors, spanning row space of A.
    /// 
    /// # Dimensions
    /// - For m×n input matrix A: returns n×n matrix V^T
    /// - V^T V = I (orthogonal)
    /// 
    /// # For AI Code Generation
    /// - Note: returns V^T (transposed), not V itself
    /// - Row i corresponds to singular value σᵢ
    /// - Use V^T for pseudoinverse: A⁺ = V·Σ⁺·U^T
    /// 
    /// # Example
    /// ```
    /// let svd = A.svd()?;
    /// let vt = svd.vt();  // n×n right singular vectors (transposed)
    /// // vt.row(0) is first right singular vector
    /// ```
    pub fn vt(&self) -> ArrayF64 {
        ArrayF64::from_faer(self.decomposition.V().transpose().to_owned())
    }
    
    /// Solve linear system using SVD (minimum norm least squares solution)
    /// 
    /// # Mathematical Specification
    /// Computes minimum norm solution to Ax = b using pseudoinverse:
    /// x = A⁺b = V·Σ⁺·U^T·b where Σ⁺ inverts non-zero singular values
    /// 
    /// # Solution Properties
    /// - If A is full rank: unique least squares solution
    /// - If A is rank-deficient: minimum norm solution among all least squares solutions
    /// - Always exists, even for singular or non-square matrices
    /// 
    /// # Complexity
    /// - Time: O(mn) using precomputed SVD
    /// - Space: O(n) for solution vector
    /// 
    /// # For AI Code Generation
    /// - Most robust solver: handles any matrix (square, rectangular, singular)
    /// - Use when other decompositions fail or matrix is known rank-deficient
    /// - Automatic regularization: small singular values effectively ignored
    /// 
    /// # Example
    /// ```
    /// // Rank-deficient or overdetermined system
    /// let A = array64![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];  // rank 1
    /// let b = array64![[1.0], [2.0], [3.0]];
    /// 
    /// let svd = A.svd()?;
    /// let x = svd.solve(&b)?;  // Minimum norm least squares solution
    /// ```
    pub fn solve(&self, b: &ArrayF64) -> Result<ArrayF64> {
        let solution = self.decomposition.solve(b.as_faer());
        Ok(ArrayF64::from_faer(solution))
    }
}

/// Compute LU decomposition with partial pivoting for general square matrices
/// 
/// # Mathematical Specification
/// Computes PA = LU factorization where:
/// - P: permutation matrix for numerical stability (row swapping)
/// - L: lower triangular matrix with unit diagonal (L[i,i] = 1)
/// - U: upper triangular matrix
/// 
/// # Algorithm
/// Uses Gaussian elimination with partial pivoting (row swapping for stability).
/// Pivot selection: choose largest absolute value in column for numerical stability.
/// 
/// # Complexity
/// - Time: O(n³) floating point operations
/// - Space: O(n²) for storing L and U factors
/// 
/// # For AI Code Generation
/// - Most general decomposition for square matrices
/// - Use when matrix is not known to have special structure
/// - Handles nearly all invertible matrices robustly
/// - Factor once, solve many times with different right-hand sides
/// 
/// # Error Conditions
/// - Returns NotSquare if matrix is not square
/// - Factorization should succeed for all invertible matrices
/// 
/// # Example
/// ```
/// let A = array64![[2.0, 1.0], [1.0, 1.0]];
/// let lu = lu_decompose(&A)?;
/// 
/// // Solve multiple systems with same A
/// let x1 = lu.solve(&b1)?;
/// let x2 = lu.solve(&b2)?;
/// ```
pub fn lu_decompose(matrix: &ArrayF64) -> Result<LuDecomposition> {
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    if rows != cols {
        return Err(LinearAlgebraError::NotSquare { rows, cols });
    }
    
    let lu_decomp = matrix.as_faer().partial_piv_lu();
    Ok(LuDecomposition {
        decomposition: lu_decomp,
    })
}

/// Compute QR decomposition for least squares and overdetermined systems
/// 
/// # Mathematical Specification
/// Computes A = QR factorization where:
/// - Q: m×m orthogonal matrix (Q^T Q = I)
/// - R: m×n upper triangular matrix (R[i,j] = 0 for i > j)
/// 
/// # Algorithm
/// Uses Householder reflections for numerical stability.
/// More stable than Gram-Schmidt orthogonalization.
/// 
/// # Complexity
/// - Time: O(mn²) for m×n matrix (m ≥ n recommended)
/// - Space: O(mn) for storing Q and R factors
/// 
/// # For AI Code Generation
/// - Best choice for overdetermined systems (m > n)
/// - More stable than normal equations A^T A x = A^T b
/// - Naturally handles least squares: min ||Ax - b||₂
/// - Works for any m×n matrix (not restricted to square)
/// 
/// # Applications
/// - Linear regression and curve fitting
/// - Overdetermined systems from experimental data
/// - Orthogonalization processes
/// 
/// # Example
/// ```
/// // Overdetermined: 3 equations, 2 unknowns
/// let A = array64![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
/// let qr = qr_decompose(&A)?;
/// 
/// let x = qr.solve(&b)?;  // Least squares solution
/// ```
pub fn qr_decompose(matrix: &ArrayF64) -> Result<QrDecomposition> {
    let qr_decomp = matrix.as_faer().qr();
    Ok(QrDecomposition {
        decomposition: qr_decomp,
    })
}

/// Compute Cholesky decomposition for symmetric positive definite matrices
/// 
/// # Mathematical Specification
/// Computes A = L·L^T factorization where:
/// - L: lower triangular matrix with positive diagonal entries
/// - Only succeeds if A is symmetric positive definite (SPD)
/// 
/// # Requirements
/// Matrix A must satisfy:
/// - Symmetric: A = A^T
/// - Positive definite: x^T A x > 0 for all x ≠ 0
/// 
/// # Algorithm
/// Uses modified Cholesky algorithm without pivoting.
/// Fails gracefully if matrix is not positive definite.
/// 
/// # Complexity
/// - Time: O(n³/3) ≈ half the cost of LU decomposition
/// - Space: O(n²) but can overwrite input matrix
/// - Most efficient decomposition when applicable
/// 
/// # For AI Code Generation
/// - Fastest option for SPD matrices (covariance, Gram matrices)
/// - More numerically stable than LU for well-conditioned SPD
/// - Use when matrix structure guarantees positive definiteness
/// - Always try this first for SPD systems
/// 
/// # Common SPD Matrices
/// - Covariance matrices: Σ = E[(X-μ)(X-μ)^T]
/// - Gram matrices: A^T A (always positive semi-definite)
/// - Regularized normal equations: A^T A + λI
/// 
/// # Error Handling
/// Returns NotPositiveDefinite if:
/// - Matrix has negative eigenvalues
/// - Matrix is singular or ill-conditioned
/// - Matrix is not symmetric (within numerical tolerance)
/// 
/// # Example
/// ```
/// // Covariance matrix (SPD by construction)
/// let data = array64![[1.0, 2.0], [3.0, 4.0]];
/// let cov = &data.T() ^ &data;  // A^T A is positive semi-definite
/// 
/// match cholesky_decompose(&cov) {
///     Ok(chol) => {
///         let x = chol.solve(&b)?;  // Fast solve
///     },
///     Err(NotPositiveDefinite) => {
///         // Fall back to LU for general matrices
///         let lu = lu_decompose(&cov)?;
///         let x = lu.solve(&b)?;
///     }
/// }
/// ```
pub fn cholesky_decompose(matrix: &ArrayF64) -> Result<CholeskyDecomposition> {
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    if rows != cols {
        return Err(LinearAlgebraError::NotSquare { rows, cols });
    }
    
    let chol_decomp = matrix.as_faer().llt(faer::Side::Lower);
    match chol_decomp {
        Ok(chol) => Ok(CholeskyDecomposition {
            decomposition: chol,
        }),
        Err(_) => Err(LinearAlgebraError::NotPositiveDefinite),
    }
}

/// Compute Singular Value Decomposition for any matrix
/// 
/// # Mathematical Specification
/// Computes A = U·Σ·V^T factorization where:
/// - U: m×m orthogonal matrix (left singular vectors)
/// - Σ: m×n diagonal matrix with σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
/// - V^T: n×n orthogonal matrix (right singular vectors, transposed)
/// 
/// # Algorithm
/// Uses iterative algorithm (typically bidiagonalization + QR iterations).
/// Most robust decomposition - works for any matrix.
/// 
/// # Complexity
/// - Time: O(min(mn², m²n)) - most expensive decomposition
/// - Space: O(mn + m² + n²) for storing all components
/// - Use only when other decompositions are insufficient
/// 
/// # For AI Code Generation
/// - Most general decomposition: works for any m×n matrix
/// - Use for rank-deficient matrices where LU/QR fail
/// - Essential for pseudoinverse, PCA, matrix approximation
/// - Provides complete information about matrix structure
/// 
/// # Applications
/// - **Rank determination**: count σᵢ > tolerance
/// - **Pseudoinverse**: A⁺ = V·Σ⁺·U^T where Σ⁺ inverts non-zero values
/// - **Principal Component Analysis**: U contains principal components
/// - **Matrix approximation**: Aₖ = Σᵢ₌₁ᵏ σᵢuᵢvᵢ^T (rank-k approximation)
/// - **Condition number**: κ(A) = σ₁/σᵣ
/// 
/// # Numerical Properties
/// - Always succeeds for finite matrices
/// - Singular values are inherently non-negative
/// - Provides complete spectral information
/// 
/// # Example
/// ```
/// let A = array64![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];  // 3×2 matrix
/// let svd = svd_decompose(&A)?;
/// 
/// let u = svd.u();                    // 3×3 left singular vectors
/// let s = svd.singular_values();      // 2 singular values
/// let vt = svd.vt();                  // 2×2 right singular vectors^T
/// 
/// // Determine rank
/// let rank = s.iter().filter(|&&x| x > 1e-12).count();
/// ```
pub fn svd_decompose(matrix: &ArrayF64) -> Result<SvdDecomposition> {
    let svd_decomp = matrix.as_faer().svd()
        .map_err(|_| LinearAlgebraError::decomposition_failed("SVD decomposition failed"))?;
    
    Ok(SvdDecomposition {
        decomposition: svd_decomp,
    })
}

/// Compute only singular values (more efficient when vectors not needed)
/// 
/// # Mathematical Specification
/// Computes σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0 from A = U·Σ·V^T without computing U,V.
/// Provides essential spectral information at reduced computational cost.
/// 
/// # Complexity
/// - Time: O(min(mn log n, m²n)) - faster than full SVD
/// - Space: O(min(m,n)) for singular values only
/// - Use when you only need rank, condition number, or spectral properties
/// 
/// # For AI Code Generation
/// - More efficient than svd() when vectors not needed
/// - Use for rank determination, condition estimation
/// - Sufficient for many matrix analysis tasks
/// 
/// # Applications
/// - **Rank determination**: count values > tolerance
/// - **Condition number**: κ(A) = σ₁/σᵣ (stability analysis)
/// - **Spectral norm**: ||A||₂ = σ₁ (largest singular value)
/// - **Numerical rank**: effective rank considering numerical precision
/// 
/// # Example
/// ```
/// let A = array64![[1.0, 2.0], [3.0, 4.0]];
/// let s = singular_values(&A)?;
/// 
/// // Quick rank check
/// let tolerance = 1e-12;
/// let rank = s.iter().filter(|&&x| x > tolerance).count();
/// 
/// // Condition number
/// let condition = s[0] / s.last().unwrap();
/// println!("Matrix condition number: {:.2e}", condition);
/// ```
pub fn singular_values(matrix: &ArrayF64) -> Result<Vec<f64>> {
    let s_vec = matrix.as_faer().singular_values()
        .map_err(|_| LinearAlgebraError::decomposition_failed("Singular values computation failed"))?;
    
    Ok(s_vec)
}

/// Extension trait providing matrix decomposition methods with mathematical notation
/// 
/// # For AI Code Generation
/// This trait adds decomposition methods directly to ArrayF64 matrices:
/// - Use concise notation: A.lu(), A.qr(), A.cholesky(), A.svd()
/// - All methods return Result<T> for error handling
/// - Factor once, solve many: decompositions are reusable
/// - Choose appropriate method based on matrix properties and problem type
/// 
/// # Method Selection Guide
/// | Method | Best For | Requirements | Complexity |
/// |--------|----------|--------------|------------|
/// | `cholesky()` | SPD systems | Symmetric positive definite | O(n³/3) |
/// | `lu()` | General systems | Square matrix | O(n³) |
/// | `qr()` | Least squares | Overdetermined (m≥n) | O(mn²) |
/// | `svd()` | Rank analysis | Any matrix | O(min(mn²,m²n)) |
/// 
/// # Example
/// ```
/// use rustlab_linearalgebra::{DecompositionMethods, ArrayF64, array64};
/// 
/// let A = array64![[4.0, 2.0], [2.0, 3.0]];
/// 
/// // Try Cholesky first (fastest for SPD)
/// if let Ok(chol) = A.cholesky() {
///     let x = chol.solve(&b)?;
/// } else {
///     // Fallback to LU for general matrices
///     let lu = A.lu()?;
///     let x = lu.solve(&b)?;
/// }
/// ```
pub trait DecompositionMethods {
    /// Compute LU decomposition with partial pivoting for general square matrices
    /// 
    /// # Mathematical Form
    /// PA = LU where P is permutation, L lower triangular, U upper triangular
    /// 
    /// # Use Cases
    /// - General linear systems Ax = b
    /// - Multiple solves with same coefficient matrix
    /// - When matrix is not known to be SPD
    /// 
    /// # Complexity: O(n³) factorization + O(n²) per solve
    fn lu(&self) -> Result<LuDecomposition>;
    
    /// Compute QR decomposition for least squares and overdetermined systems
    /// 
    /// # Mathematical Form
    /// A = QR where Q is orthogonal, R is upper triangular
    /// 
    /// # Use Cases
    /// - Overdetermined systems (m > n)
    /// - Least squares fitting: min ||Ax - b||₂
    /// - More stable than normal equations
    /// 
    /// # Complexity: O(mn²) factorization + O(mn) per solve
    fn qr(&self) -> Result<QrDecomposition>;
    
    /// Compute Cholesky decomposition for symmetric positive definite matrices
    /// 
    /// # Mathematical Form
    /// A = L·L^T where L is lower triangular with positive diagonal
    /// 
    /// # Use Cases
    /// - Covariance matrices, Gram matrices (A^T A)
    /// - When you know matrix is SPD
    /// - Fastest and most stable for SPD systems
    /// 
    /// # Complexity: O(n³/3) factorization + O(n²) per solve
    fn cholesky(&self) -> Result<CholeskyDecomposition>;
    
    /// Compute Singular Value Decomposition for rank analysis and pseudoinverse
    /// 
    /// # Mathematical Form
    /// A = U·Σ·V^T where U,V orthogonal, Σ diagonal with σ₁ ≥ σ₂ ≥ ... ≥ 0
    /// 
    /// # Use Cases
    /// - Rank-deficient matrices
    /// - Pseudoinverse computation
    /// - Principal Component Analysis (PCA)
    /// - Matrix approximation
    /// 
    /// # Complexity: O(min(mn²,m²n)) factorization
    fn svd(&self) -> Result<SvdDecomposition>;
}

// Implementation for ArrayF64
impl DecompositionMethods for ArrayF64 {
    fn lu(&self) -> Result<LuDecomposition> {
        lu_decompose(self)
    }
    
    fn qr(&self) -> Result<QrDecomposition> {
        qr_decompose(self)
    }
    
    fn cholesky(&self) -> Result<CholeskyDecomposition> {
        cholesky_decompose(self)
    }
    
    fn svd(&self) -> Result<SvdDecomposition> {
        svd_decompose(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::array64;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_lu_solve() {
        let matrix = array64![
            [4.0, 3.0],
            [6.0, 3.0]
        ];
        
        let b = array64![[10.0], [12.0]];
        
        let lu = matrix.lu().unwrap();
        let x = lu.solve(&b).unwrap();
        
        // Check that we got a solution
        assert_eq!(x.nrows(), 2);
        assert_eq!(x.ncols(), 1);
    }
    
    #[test]
    fn test_qr_solve() {
        let matrix = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        
        let b = array64![[5.0], [6.0]];
        
        let qr = matrix.qr().unwrap();
        let x = qr.solve(&b).unwrap();
        
        // Check that we got a solution
        assert_eq!(x.nrows(), 2);
        assert_eq!(x.ncols(), 1);
    }
    
    #[test]
    fn test_cholesky_solve() {
        // Create a positive definite matrix
        let matrix = array64![
            [4.0, 2.0],
            [2.0, 3.0]
        ];
        
        let b = array64![[8.0], [7.0]];
        
        let chol = matrix.cholesky().unwrap();
        let x = chol.solve(&b).unwrap();
        
        // Check that we got a solution
        assert_eq!(x.nrows(), 2);
        assert_eq!(x.ncols(), 1);
    }
    
    #[test]
    fn test_svd_components() {
        let matrix = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        
        let svd = matrix.svd().unwrap();
        
        // Check dimensions of U, S, V^T
        let u = svd.u();
        let s = svd.singular_values();
        let vt = svd.vt();
        
        assert_eq!(u.nrows(), 2);
        assert_eq!(u.ncols(), 2);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.nrows(), 2);
        assert_eq!(vt.ncols(), 2);
        
        // Singular values should be positive and in descending order
        assert!(s[0] > 0.0);
        assert!(s[1] > 0.0);
        assert!(s[0] >= s[1]);
    }
    
    #[test]
    fn test_singular_values_only() {
        let matrix = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        
        let singular_vals = singular_values(&matrix).unwrap();
        
        // Check dimensions
        assert_eq!(singular_vals.len(), 2);
        
        // Singular values should be positive and in descending order
        assert!(singular_vals[0] > 0.0);
        assert!(singular_vals[1] > 0.0);
        assert!(singular_vals[0] >= singular_vals[1]);
    }
}