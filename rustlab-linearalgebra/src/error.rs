//! Comprehensive error handling for linear algebra operations with AI guidance
//!
//! Provides specialized error types for mathematical operations, enabling robust
//! error handling and user-friendly diagnostics. All errors include actionable
//! guidance for resolution.
//!
//! # For AI Code Generation
//! - Use Result<T> for all fallible operations
//! - Match on specific error variants for intelligent error handling
//! - Provide fallback strategies for common numerical issues
//! - Include dimensional information for debugging
//!
//! # Error Categories
//! - **Structural**: Matrix dimensions, square requirements
//! - **Mathematical**: Singularity, positive definiteness
//! - **Numerical**: Convergence, stability issues
//! - **Computational**: Algorithm failures, decomposition errors
//!
//! # Example Error Handling
//! ```
//! use rustlab_linearalgebra::{LinearAlgebraError, Result};
//!
//! match matrix.cholesky() {
//!     Ok(decomp) => decomp.solve(&b)?,
//!     Err(LinearAlgebraError::NotPositiveDefinite) => {
//!         // Fallback to LU for non-SPD matrices
//!         matrix.lu()?.solve(&b)?
//!     },
//!     Err(e) => return Err(e),
//! }
//! ```

use thiserror::Error;

/// Standard Result type for linear algebra operations with comprehensive error information
/// 
/// # For AI Code Generation
/// - Use this type for all fallible linear algebra operations
/// - Enables robust error handling with ? operator
/// - Provides detailed error diagnostics for debugging
/// 
/// # Example
/// ```
/// fn solve_system(A: &ArrayF64, b: &ArrayF64) -> Result<ArrayF64> {
///     let lu = A.lu()?;  // Propagates errors automatically
///     lu.solve(b)        // Returns Result<ArrayF64>
/// }
/// ```
pub type Result<T> = std::result::Result<T, LinearAlgebraError>;

/// Comprehensive error types for linear algebra operations with actionable guidance
/// 
/// # For AI Code Generation
/// Each error variant provides:
/// - Clear mathematical context
/// - Specific failure conditions
/// - Suggested resolution strategies
/// - Dimensional information where relevant
/// 
/// # Error Handling Strategy
/// ```
/// match result {
///     Ok(value) => value,
///     Err(LinearAlgebraError::NotSquare { rows, cols }) => {
///         return Err(format!("Expected square matrix, got {}×{}", rows, cols));
///     },
///     Err(LinearAlgebraError::Singular) => {
///         // Try regularization: A + εI
///         let regularized = A + &(epsilon * &eye(A.nrows()));
///         regularized.lu()?.solve(b)?
///     },
///     Err(LinearAlgebraError::NotPositiveDefinite) => {
///         // Fallback to general LU decomposition
///         A.lu()?.solve(b)?
///     },
///     // ... handle other cases
/// }
/// ```
#[derive(Error, Debug)]
pub enum LinearAlgebraError {
    /// Matrix is not square when square matrix is required
    /// 
    /// # When This Occurs
    /// - Determinant computation: det() requires n×n matrix
    /// - Matrix inverse: inv() requires n×n matrix  
    /// - Eigenvalue computation: eigenvalues() requires n×n matrix
    /// - LU/Cholesky decomposition: requires n×n matrix
    /// 
    /// # Resolution Strategies
    /// - Check input dimensions before operation
    /// - Use QR decomposition for rectangular matrices (least squares)
    /// - Extract square submatrix if appropriate
    /// - Transpose if dimensions are swapped
    /// 
    /// # Example Fix
    /// ```
    /// if matrix.nrows() != matrix.ncols() {
    ///     // Use QR for overdetermined systems instead
    ///     let qr = matrix.qr()?;
    ///     return qr.solve(&b);
    /// }
    /// let det = matrix.det()?;  // Now safe
    /// ```
    #[error("Matrix must be square for this operation, got shape ({rows}, {cols})")]
    NotSquare { 
        /// Number of rows in the matrix
        rows: usize, 
        /// Number of columns in the matrix
        cols: usize 
    },
    
    /// Matrix is singular (not invertible)
    /// 
    /// # Mathematical Meaning
    /// - Determinant is zero (or numerically close to zero)
    /// - Matrix has no inverse: A⁻¹ does not exist
    /// - System Ax = b may have no solution or infinitely many solutions
    /// - Matrix is rank-deficient: rank(A) < n for n×n matrix
    /// 
    /// # When This Occurs
    /// - Rows/columns are linearly dependent
    /// - Matrix is ill-conditioned (condition number very large)
    /// - Numerical round-off causes apparent singularity
    /// 
    /// # Resolution Strategies
    /// 1. **Check condition number**: Use SVD to assess numerical rank
    /// 2. **Regularization**: Add small diagonal: A + εI where ε ≈ 1e-12
    /// 3. **Pseudoinverse**: Use SVD-based pseudoinverse for rank-deficient systems
    /// 4. **Different algorithm**: Try QR decomposition for overdetermined systems
    /// 
    /// # Example Recovery
    /// ```
    /// match A.inv() {
    ///     Ok(inv) => inv,
    ///     Err(LinearAlgebraError::Singular) => {
    ///         // Try SVD-based pseudoinverse
    ///         let svd = A.svd()?;
    ///         // Use SVD to solve minimum norm least squares
    ///         svd.solve(&b)?
    ///     }
    /// }
    /// ```
    #[error("Matrix is singular and cannot be inverted")]
    Singular,
    
    /// Matrix is singular (alternative name for consistency)
    #[error("Matrix is singular and cannot be inverted")]
    SingularMatrix,
    
    /// Matrix is not positive definite (required for Cholesky decomposition)
    /// 
    /// # Mathematical Requirements for Positive Definiteness
    /// - Symmetric: A = Aᵀ
    /// - All eigenvalues positive: λᵢ > 0
    /// - All leading principal minors positive
    /// - xᵀAx > 0 for all x ≠ 0
    /// 
    /// # When This Occurs
    /// - Matrix has negative or zero eigenvalues
    /// - Matrix is not symmetric (violates SPD requirement)
    /// - Numerical errors cause apparent non-positive-definiteness
    /// - Input matrix is positive semi-definite (rank-deficient)
    /// 
    /// # Common Causes
    /// - Correlation matrices with perfect correlations
    /// - Gram matrices AᵀA from rank-deficient A
    /// - Regularization parameter too small
    /// - Numerical precision issues
    /// 
    /// # Resolution Strategies
    /// 1. **Fallback to LU**: Use general LU decomposition instead
    /// 2. **Regularization**: Add λI where λ > 0 to make SPD
    /// 3. **Check symmetry**: Ensure A = Aᵀ within tolerance
    /// 4. **Eigenvalue analysis**: Check if eigenvalues are positive
    /// 
    /// # Example Recovery
    /// ```
    /// match A.cholesky() {
    ///     Ok(chol) => chol.solve(&b)?,
    ///     Err(LinearAlgebraError::NotPositiveDefinite) => {
    ///         // Fallback to general LU decomposition
    ///         A.lu()?.solve(&b)?
    ///     }
    /// }
    /// ```
    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,
    
    /// Dimension mismatch in matrix operations
    /// 
    /// # Common Causes
    /// - Matrix multiplication: A(m×n) * B(p×q) requires n = p
    /// - Linear systems: A(n×n) * x = b requires b.len() = n
    /// - Element-wise operations: A(m×n) + B(p×q) requires m=p, n=q
    /// - Broadcasting incompatibility
    /// 
    /// # Resolution Strategies
    /// 1. **Check dimensions**: Verify compatibility before operations
    /// 2. **Transpose if needed**: Use A.T() to fix dimension mismatches
    /// 3. **Reshape vectors**: Convert between row/column vectors
    /// 4. **Extract submatrices**: Use slicing to match dimensions
    /// 
    /// # Example Prevention
    /// ```
    /// // Check compatibility before matrix multiplication
    /// if A.ncols() != B.nrows() {
    ///     return Err(LinearAlgebraError::dimension_mismatch(
    ///         format!("{}×{}", A.nrows(), B.nrows()),
    ///         format!("{}×{}", A.nrows(), A.ncols())
    ///     ));
    /// }
    /// let result = &A ^ &B;  // Safe multiplication
    /// ```
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { 
        /// Expected dimensions
        expected: String, 
        /// Actual dimensions
        actual: String 
    },
    
    /// Invalid matrix properties for the requested operation
    /// 
    /// # Common Invalid Conditions
    /// - Empty matrix (0×0 dimensions)
    /// - Infinite or NaN values in matrix elements
    /// - Matrix violates algorithm assumptions
    /// - Incompatible data types or precision
    /// 
    /// # Resolution Strategies
    /// 1. **Validate input**: Check for NaN/Inf before operations
    /// 2. **Sanitize data**: Replace invalid values with reasonable defaults
    /// 3. **Check assumptions**: Verify matrix satisfies algorithm requirements
    /// 4. **Use robust algorithms**: Choose methods that handle edge cases
    /// 
    /// # Example Validation
    /// ```
    /// fn validate_matrix(A: &ArrayF64) -> Result<()> {
    ///     if A.nrows() == 0 || A.ncols() == 0 {
    ///         return Err(LinearAlgebraError::invalid_matrix("Empty matrix"));
    ///     }
    ///     // Check for NaN/Inf values
    ///     for &val in A.iter() {
    ///         if !val.is_finite() {
    ///             return Err(LinearAlgebraError::invalid_matrix("Matrix contains NaN/Inf"));
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[error("Invalid matrix for operation: {reason}")]
    InvalidMatrix { 
        /// Reason why the matrix is invalid
        reason: String 
    },
    
    /// Convergence failure in iterative algorithms
    /// 
    /// # When This Occurs
    /// - Iterative eigenvalue algorithms (power method, QR algorithm)
    /// - Iterative linear solvers (CG, GMRES)
    /// - Matrix function computations (matrix exponential, logarithm)
    /// - Optimization-based matrix operations
    /// 
    /// # Common Causes
    /// - Poor conditioning (large condition number)
    /// - Inappropriate tolerance settings
    /// - Matrix near singularity
    /// - Algorithm not suitable for matrix structure
    /// 
    /// # Resolution Strategies
    /// 1. **Increase iterations**: Higher iteration limit
    /// 2. **Adjust tolerance**: Relax convergence criteria
    /// 3. **Preconditioning**: Use preconditioned algorithms
    /// 4. **Different algorithm**: Switch to direct methods
    /// 5. **Regularization**: Add λI to improve conditioning
    /// 
    /// # Example Recovery
    /// ```
    /// match A.eigenvalues() {
    ///     Ok(eigenvals) => eigenvals,
    ///     Err(LinearAlgebraError::ConvergenceFailure { iterations }) => {
    ///         eprintln!("Eigenvalue computation didn't converge in {} iterations", iterations);
    ///         // Try with regularization
    ///         let regularized = A + &(1e-12 * &eye(A.nrows()));
    ///         regularized.eigenvalues()?
    ///     }
    /// }
    /// ```
    #[error("Algorithm failed to converge after {iterations} iterations")]
    ConvergenceFailure { 
        /// Number of iterations attempted before giving up
        iterations: usize 
    },
    
    /// Numerical instability detected during computation
    /// 
    /// # Signs of Numerical Instability
    /// - Loss of significant digits during computation
    /// - Results vary dramatically with small input changes
    /// - Computed results violate mathematical properties
    /// - Residuals much larger than expected
    /// 
    /// # Common Causes
    /// - Ill-conditioned matrices (large condition number)
    /// - Subtractive cancellation in floating-point arithmetic
    /// - Accumulation of round-off errors
    /// - Algorithm poorly suited for problem structure
    /// 
    /// # Resolution Strategies
    /// 1. **Higher precision**: Use f64 instead of f32, or arbitrary precision
    /// 2. **Stable algorithms**: Choose numerically stable variants
    /// 3. **Pivoting**: Use algorithms with pivoting for stability
    /// 4. **Iterative refinement**: Improve solution accuracy
    /// 5. **Regularization**: Add small diagonal term for stability
    /// 
    /// # Example Handling
    /// ```
    /// match A.lu() {
    ///     Ok(decomp) => decomp,
    ///     Err(LinearAlgebraError::NumericalInstability { details }) => {
    ///         eprintln!("Numerical issues: {}", details);
    ///         // Try more stable algorithm
    ///         A.svd()?  // SVD is more numerically stable
    ///     }
    /// }
    /// ```
    #[error("Numerical instability detected: {details}")]
    NumericalInstability { 
        /// Details about the numerical instability
        details: String 
    },
    
    /// Error propagated from underlying rustlab-math library
    /// 
    /// # Common Underlying Errors
    /// - Array creation failures (memory allocation)
    /// - Index out of bounds in array operations
    /// - Type conversion errors
    /// - Broadcasting failures
    /// 
    /// # For AI Code Generation
    /// - Automatically converts rustlab_math::MathError to LinearAlgebraError
    /// - Preserves original error context and stack trace
    /// - Use ? operator for seamless error propagation
    /// 
    /// # Example
    /// ```
    /// fn matrix_operation(data: &[f64]) -> Result<ArrayF64> {
    ///     // MathError automatically converts to LinearAlgebraError
    ///     let matrix = ArrayF64::from_slice(data, 3, 3)?;
    ///     Ok(matrix)
    /// }
    /// ```
    #[error("Math error: {0}")]
    MathError(#[from] rustlab_math::MathError),
    
    /// General computation error for unexpected failures
    /// 
    /// # When This Occurs
    /// - Unexpected algorithm failures
    /// - Memory allocation failures during computation
    /// - Hardware-specific computation errors
    /// - Unhandled edge cases in algorithms
    /// 
    /// # For AI Code Generation
    /// - Catch-all for unexpected computational issues
    /// - Include detailed error message for debugging
    /// - Often indicates need for algorithm robustness improvement
    /// 
    /// # Resolution Approach
    /// ```
    /// match computation_result {
    ///     Ok(result) => result,
    ///     Err(LinearAlgebraError::ComputationError { message }) => {
    ///         eprintln!("Unexpected computation error: {}", message);
    ///         // Log error for debugging
    ///         // Try alternative approach or report to user
    ///         return Err("Computation failed - please try different parameters".into());
    ///     }
    /// }
    /// ```
    #[error("Computation error: {message}")]
    ComputationError { 
        /// Error message describing the computation failure
        message: String 
    },
    
    /// Matrix decomposition algorithm failed
    /// 
    /// # Common Decomposition Failures
    /// - SVD algorithm convergence issues
    /// - QR decomposition numerical problems
    /// - Eigenvalue computation failures
    /// - Memory allocation during factorization
    /// 
    /// # When This Occurs
    /// - Matrix properties incompatible with chosen decomposition
    /// - Numerical precision issues in algorithm
    /// - Extreme matrix conditioning
    /// - Resource constraints (memory, computation time)
    /// 
    /// # Resolution Strategies
    /// 1. **Try different decomposition**: LU → QR → SVD (increasing robustness)
    /// 2. **Adjust parameters**: Change tolerance, iteration limits
    /// 3. **Preprocess matrix**: Scale, regularize, or condition
    /// 4. **Use backup algorithm**: Fall back to more robust method
    /// 
    /// # Example Cascading
    /// ```
    /// // Try decompositions in order of preference
    /// fn robust_solve(A: &ArrayF64, b: &ArrayF64) -> Result<ArrayF64> {
    ///     // Try Cholesky first (fastest for SPD)
    ///     if let Ok(chol) = A.cholesky() {
    ///         return chol.solve(b);
    ///     }
    ///     
    ///     // Fall back to LU
    ///     if let Ok(lu) = A.lu() {
    ///         return lu.solve(b);
    ///     }
    ///     
    ///     // Last resort: SVD (most robust)
    ///     let svd = A.svd()?;
    ///     svd.solve(b)
    /// }
    /// ```
    #[error("Matrix decomposition failed: {message}")]
    DecompositionFailed { 
        /// Details about why the decomposition failed
        message: String 
    },
}

impl LinearAlgebraError {
    /// Create a dimension mismatch error with detailed context
    /// 
    /// # For AI Code Generation
    /// - Use this constructor for clear dimensional error reporting
    /// - Provide specific expected vs actual dimensions
    /// - Include operation context in error message
    /// 
    /// # Example
    /// ```
    /// if A.ncols() != b.len() {
    ///     return Err(LinearAlgebraError::dimension_mismatch(
    ///         format!("vector length {}", A.nrows()),
    ///         format!("vector length {}", b.len())
    ///     ));
    /// }
    /// ```
    pub fn dimension_mismatch<E: ToString, A: ToString>(expected: E, actual: A) -> Self {
        Self::DimensionMismatch {
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }
    
    /// Create an invalid matrix error with specific reason
    /// 
    /// # For AI Code Generation
    /// - Use for input validation errors
    /// - Provide clear reason for rejection
    /// - Help users understand matrix requirements
    /// 
    /// # Example
    /// ```
    /// if matrix.nrows() == 0 {
    ///     return Err(LinearAlgebraError::invalid_matrix(
    ///         "Cannot compute eigenvalues of empty matrix"
    ///     ));
    /// }
    /// ```
    pub fn invalid_matrix<R: ToString>(reason: R) -> Self {
        Self::InvalidMatrix {
            reason: reason.to_string(),
        }
    }
    
    /// Create a convergence failure error with iteration count
    /// 
    /// # For AI Code Generation
    /// - Use when iterative algorithms fail to converge
    /// - Include iteration count for debugging
    /// - Suggests algorithm tuning or alternative methods
    /// 
    /// # Example
    /// ```
    /// if iteration > max_iterations {
    ///     return Err(LinearAlgebraError::convergence_failure(iteration));
    /// }
    /// ```
    pub fn convergence_failure(iterations: usize) -> Self {
        Self::ConvergenceFailure { iterations }
    }
    
    /// Create a numerical instability error with diagnostic details
    /// 
    /// # For AI Code Generation
    /// - Use when detecting numerical precision issues
    /// - Include specific instability indicators
    /// - Suggest more stable algorithms or parameters
    /// 
    /// # Example
    /// ```
    /// if condition_number > 1e12 {
    ///     return Err(LinearAlgebraError::numerical_instability(
    ///         format!("Matrix condition number {:.2e} indicates ill-conditioning", condition_number)
    ///     ));
    /// }
    /// ```
    pub fn numerical_instability<D: ToString>(details: D) -> Self {
        Self::NumericalInstability {
            details: details.to_string(),
        }
    }
    
    /// Create a general computation error with descriptive message
    /// 
    /// # For AI Code Generation
    /// - Use for unexpected computational failures
    /// - Provide clear error description for debugging
    /// - Include context about what operation failed
    /// 
    /// # Example
    /// ```
    /// if memory_allocation_failed {
    ///     return Err(LinearAlgebraError::computation_error(
    ///         "Failed to allocate memory for matrix decomposition"
    ///     ));
    /// }
    /// ```
    pub fn computation_error<M: ToString>(message: M) -> Self {
        Self::ComputationError {
            message: message.to_string(),
        }
    }
    
    /// Create a decomposition failure error with algorithm details
    /// 
    /// # For AI Code Generation
    /// - Use when matrix decomposition algorithms fail
    /// - Include which decomposition failed and why
    /// - Suggest alternative decomposition methods
    /// 
    /// # Example
    /// ```
    /// if svd_computation_failed {
    ///     return Err(LinearAlgebraError::decomposition_failed(
    ///         "SVD failed to converge - matrix may be ill-conditioned"
    ///     ));
    /// }
    /// ```
    pub fn decomposition_failed<M: ToString>(message: M) -> Self {
        Self::DecompositionFailed {
            message: message.to_string(),
        }
    }
}