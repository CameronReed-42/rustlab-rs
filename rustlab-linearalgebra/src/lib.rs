//! RustLab Linear Algebra - High-performance matrix operations with AI-optimized documentation
//!
//! Provides comprehensive linear algebra functionality built on rustlab-math and powered
//! by the faer 0.22 backend. Designed for scientific computing, machine learning, and
//! numerical analysis with mathematical notation and robust error handling.
//!

#![allow(non_snake_case)]  // Allow mathematical notation (X for matrices, T for transpose)
//! # For AI Code Generation
//!
//! This crate uses extension traits to add mathematical methods directly to ArrayF64:
//! - **Mathematical notation**: A.T(), A.det(), A.inv(), A.eigenvalues()
//! - **Decomposition methods**: A.lu(), A.qr(), A.cholesky(), A.svd()
//! - **Error handling**: All operations return Result<T> with detailed error types
//! - **Performance hierarchy**: Choose optimal algorithm for matrix structure
//!
//! # Core Features
//!
//! ## Matrix Decompositions
//! - **LU**: General square matrices (O(n³))
//! - **QR**: Overdetermined systems, least squares (O(mn²))
//! - **Cholesky**: Symmetric positive definite (O(n³/3))
//! - **SVD**: Any matrix, pseudoinverse, rank analysis (O(min(mn²,m²n)))
//!
//! ## Eigenvalue Computation
//! - **General matrices**: Complex eigenvalues via QR algorithm
//! - **Symmetric matrices**: Real eigenvalues, orthogonal eigenvectors
//! - **Spectral analysis**: Condition numbers, matrix norms
//!
//! ## Basic Operations
//! - **Transpose**: A.T() for mathematical clarity
//! - **Determinant**: A.det() via LU decomposition
//! - **Inverse**: A.inv() with singularity detection
//!
//! # Quick Start Guide
//!
//! ```rust
//! use rustlab_linearalgebra::*;
//! use rustlab_math::{ArrayF64, array64};
//!
//! // Create matrices using rustlab-math
//! let A = array64![[4.0, 2.0], [2.0, 3.0]];
//! let b = array64![[8.0], [7.0]];
//!
//! // Basic operations with mathematical notation
//! let det = A.det()?;               // Determinant
//! let At = A.T();                   // Transpose  
//! let A_inv = A.inv()?;             // Inverse (checks singularity)
//!
//! // Efficient solving: factor once, solve many
//! let chol = A.cholesky()?;         // O(n³/3) for SPD matrices
//! let x = chol.solve(&b)?;          // O(n²) per solve
//!
//! // Eigenvalue analysis
//! let eigenvals = A.eigenvalues_self_adjoint()?;  // Real eigenvalues
//! let (vals, vecs) = A.eigenvectors_self_adjoint()?;
//!
//! // Robust solving with fallback strategy
//! let solution = match A.cholesky() {
//!     Ok(chol) => chol.solve(&b)?,   // Fast path for SPD
//!     Err(_) => A.lu()?.solve(&b)?,  // General fallback
//! };
//! ```
//!
//! # Algorithm Selection Guide
//!
//! | Problem Type | Recommended Method | Complexity | Properties |
//! |--------------|-------------------|------------|------------|
//! | **SPD Systems** | `A.cholesky()?.solve(b)` | O(n³/3) | Fastest, most stable for SPD |
//! | **General Systems** | `A.lu()?.solve(b)` | O(n³) | General square matrices |
//! | **Overdetermined** | `A.qr()?.solve(b)` | O(mn²) | Least squares solution |
//! | **Rank Deficient** | `A.svd()?.solve(b)` | O(min(mn²,m²n)) | Pseudoinverse, most robust |
//! | **Eigenvalues** | `A.eigenvalues_self_adjoint()` | O(n³) | Symmetric matrices only |
//! | **General Eigen** | `A.eigenvalues()` | O(n³) | May return complex values |
//!
//! # Error Handling Patterns
//!
//! ```rust
//! use rustlab_linearalgebra::{LinearAlgebraError, Result};
//!
//! fn robust_solve(A: &ArrayF64, b: &ArrayF64) -> Result<ArrayF64> {
//!     match A.cholesky() {
//!         Ok(chol) => chol.solve(b),
//!         Err(LinearAlgebraError::NotPositiveDefinite) => {
//!             // Fall back to LU for non-SPD matrices
//!             A.lu()?.solve(b)
//!         },
//!         Err(e) => Err(e),
//!     }
//! }
//!
//! // Handle dimension mismatches
//! match A.solve_system(&b) {
//!     Ok(x) => x,
//!     Err(LinearAlgebraError::DimensionMismatch { expected, actual }) => {
//!         eprintln!("Dimension error: expected {}, got {}", expected, actual);
//!         return Err("Incompatible dimensions".into());
//!     },
//!     Err(LinearAlgebraError::Singular) => {
//!         // Try regularization or pseudoinverse
//!         A.svd()?.solve(b)?
//!     },
//!     Err(e) => return Err(e),
//! }
//! ```
//!
//! # Performance Guidelines
//!
//! ## Memory Layout
//! - All matrices use column-major storage (compatible with BLAS/LAPACK)
//! - Avoid unnecessary copying with reference operators: `&A ^ &B`
//! - Reuse decompositions for multiple solves
//!
//! ## Algorithm Efficiency
//! ```rust
//! // ✅ Efficient: Factor once, solve many
//! let lu = A.lu()?;
//! let x1 = lu.solve(&b1)?;
//! let x2 = lu.solve(&b2)?;
//!
//! // ❌ Inefficient: Repeated factorization
//! let x1 = A.lu()?.solve(&b1)?;
//! let x2 = A.lu()?.solve(&b2)?;
//!
//! // ✅ Optimal: Use fastest decomposition for matrix type
//! let solver = if is_symmetric_positive_definite(&A) {
//!     A.cholesky()?.into()  // O(n³/3)
//! } else {
//!     A.lu()?.into()        // O(n³)
//! };
//! ```

#![warn(missing_docs)]

// Re-export rustlab-math-v2 types for convenience
pub use rustlab_math::{
    Array, ArrayF64, ArrayF32, ArrayC64, ArrayC32,
    Vector, VectorF64, VectorF32, VectorC64, VectorC32,
    Result as MathResult, MathError,
};

// Note: Entity trait moved in faer 0.22 - temporarily disable
// pub use faer::Entity;

// Core modules
pub mod decompositions;
pub mod eigenvalues;
// pub mod solvers;
pub mod error;
// pub mod simple_test;
pub mod minimal;
pub mod basic_operations;

// Re-exports for convenience
pub use decompositions::{
    LuDecomposition, QrDecomposition, CholeskyDecomposition, SvdDecomposition,
    lu_decompose, qr_decompose, cholesky_decompose, svd_decompose,
    singular_values,
    DecompositionMethods,
};
pub use eigenvalues::{EigenDecomposition, eigenvalues, eigenvectors, eig, EigenvalueOps};
// pub use solvers::{LinearSolver, solve, solve_triangular, LinearSystemOps};
pub use error::{LinearAlgebraError, Result};
pub use minimal::{MinimalLinearAlgebra, simple_lu_decompose};
pub use basic_operations::BasicLinearAlgebra;

/// Main linear algebra trait for future extensibility
/// 
/// # For AI Code Generation
/// This trait is currently a placeholder for future generic linear algebra operations.
/// Use the specific extension traits (BasicLinearAlgebra, DecompositionMethods, etc.)
/// for current functionality.
/// 
/// # Future Extensions
/// - Generic element types beyond f64
/// - Sparse matrix operations  
/// - Batch operations for multiple matrices
/// - Advanced matrix functions (exponential, logarithm)
/// 
/// # Current Alternatives
/// ```rust
/// // Use specific traits instead:
/// use rustlab_linearalgebra::{BasicLinearAlgebra, DecompositionMethods, EigenvalueOps};
/// 
/// let det = matrix.det()?;           // BasicLinearAlgebra
/// let lu = matrix.lu()?;             // DecompositionMethods  
/// let eigenvals = matrix.eigenvalues()?; // EigenvalueOps
/// ```
pub trait LinearAlgebra<T> {
    /// Compute matrix determinant (placeholder implementation)
    /// 
    /// # Note
    /// Use `BasicLinearAlgebra::det()` for current determinant computation.
    /// This method exists for future generic trait unification.
    fn determinant(&self) -> Result<T>;
}

