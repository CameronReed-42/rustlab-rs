//! Comprehensive error handling for numerical methods
//!
//! This module provides a robust error type system designed specifically for numerical
//! computing applications. All errors are carefully categorized to provide maximum
//! information for debugging and error recovery while maintaining performance and
//! ergonomics.
//!
//! ## Error Categories
//!
//! ### Input Validation Errors
//! - **InsufficientData**: Not enough data points for the requested operation
//! - **DimensionMismatch**: Array dimensions don't match requirements
//! - **NotMonotonic**: Required monotonicity constraint violated
//! - **InvalidParameter**: Parameter values outside valid ranges
//!
//! ### Numerical Computation Errors
//! - **OutOfBounds**: Values outside valid computation domain
//! - **NumericalInstability**: Numerical instability detected during computation
//! - **DivisionByZero**: Division by zero in specific computational context
//! - **ConvergenceFailure**: Iterative algorithm failed to converge
//!
//! ### Function Evaluation Errors
//! - **FunctionEvaluation**: Error during user-provided function evaluation
//! - **IntegrationError**: Specific errors during numerical integration
//!
//! ### External Dependencies
//! - **CoreError**: Errors from underlying RustLab-Math operations
//!
//! ## Error Handling Philosophy
//!
//! ### Fail-Fast Principle
//! All methods validate inputs early and return meaningful error messages
//! to help users identify and correct problems quickly.
//!
//! ### Contextual Information
//! Errors include specific context about what operation failed and why,
//! making debugging more efficient and educational.
//!
//! ### Composability
//! The error type implements standard traits allowing easy integration
//! with the broader Rust error handling ecosystem.
//!
//! ## Usage Guidelines
//!
//! ### Error Propagation
//! ```rust
//! use rustlab_numerical::{Result, NumericalError};
//!
//! fn compute_something(x: &[f64]) -> Result<f64> {
//!     if x.len() < 2 {
//!         return Err(NumericalError::InsufficientData {
//!             got: x.len(),
//!             need: 2,
//!         });
//!     }
//!     // ... computation
//!     Ok(42.0)
//! }
//! ```
//!
//! ### Error Matching
//! ```rust
//! use rustlab_numerical::{simpson, NumericalError};
//!
//! match simpson(|x| x.sin(), 0.0, 1.0, 0) {
//!     Ok(result) => println!("Integral: {}", result),
//!     Err(NumericalError::InvalidParameter(msg)) => {
//!         eprintln!("Parameter error: {}", msg);
//!     }
//!     Err(e) => eprintln!("Other error: {}", e),
//! }
//! ```
//!
//! ### Recovery Strategies
//! ```rust,ignore
//! use rustlab_numerical::{brent, bisection, NumericalError};
//!
//! // Try fast method first, fall back to reliable method
//! fn robust_root_find(f: impl Fn(f64) -> f64, a: f64, b: f64) -> Result<f64> {
//!     match brent(f, a, b, 1e-12, 50) {
//!         Ok(result) => Ok(result.root),
//!         Err(NumericalError::ConvergenceFailure { .. }) => {
//!             // Fall back to guaranteed method
//!             bisection(f, a, b, 1e-8, 100).map(|r| r.root)
//!         }
//!         Err(e) => Err(e),
//!     }
//! }
//! ```

use thiserror::Error;

/// Comprehensive error type for numerical computing operations
///
/// This enum covers all possible error conditions that can occur during
/// numerical computations. Each variant provides detailed context about
/// the specific failure to aid in debugging and error recovery.
///
/// # Design Principles
///
/// - **Specific Context**: Each error includes relevant parameters and context
/// - **User-Friendly Messages**: Error messages are descriptive and actionable
/// - **Structured Data**: Errors include structured data for programmatic handling
/// - **Performance**: Zero-cost when no errors occur, minimal overhead when errors happen
///
/// # Error Categories
///
/// Errors are logically grouped by their nature and typical recovery strategies:
///
/// 1. **Input Validation**: Catch invalid inputs before computation
/// 2. **Numerical Issues**: Handle mathematical edge cases and instabilities  
/// 3. **Convergence Problems**: Manage iterative algorithm convergence failures
/// 4. **External Dependencies**: Propagate errors from underlying libraries
#[derive(Error, Debug)]
pub enum NumericalError {
    /// Insufficient data points for the requested numerical operation
    ///
    /// Many numerical methods require a minimum number of data points to function
    /// correctly. This error occurs when the provided data doesn't meet these
    /// requirements.
    ///
    /// # Common Causes
    /// - Interpolation methods requiring multiple points
    /// - Statistical operations needing minimum sample sizes
    /// - Integration methods requiring minimum subdivisions
    ///
    /// # Recovery Strategies
    /// - Collect more data points if possible
    /// - Choose methods with lower data requirements
    /// - Use default/fallback values for missing data
    #[error("Insufficient data points: got {got}, need at least {need}")]
    InsufficientData { got: usize, need: usize },
    
    /// Mismatched dimensions between input arrays or matrices
    ///
    /// Numerical operations often require arrays of the same length or matrices
    /// with compatible dimensions. This error indicates a dimension mismatch.
    ///
    /// # Common Causes
    /// - X and Y arrays of different lengths in interpolation
    /// - Matrix operations with incompatible dimensions
    /// - Vector operations on arrays of different sizes
    ///
    /// # Recovery Strategies
    /// - Verify input data consistency before computation
    /// - Resize arrays to match (with appropriate padding/truncation)
    /// - Use element-wise operations that handle different sizes
    #[error("Input arrays have different lengths: {0} vs {1}")]
    DimensionMismatch(usize, usize),
    
    /// Value outside the valid domain for computation
    ///
    /// Many numerical methods have restricted domains where they can safely
    /// and accurately compute results. This error occurs when input values
    /// fall outside these valid ranges.
    ///
    /// # Common Causes
    /// - Interpolation outside the data range
    /// - Function evaluation at domain boundaries
    /// - Root finding outside bracketing intervals
    ///
    /// # Recovery Strategies
    /// - Use extrapolation methods for out-of-bounds values
    /// - Extend the computation domain if mathematically valid
    /// - Return special values (NaN, boundary values) for out-of-bounds cases
    #[error("Value {value} is outside the valid range [{min}, {max}]")]
    OutOfBounds { value: f64, min: f64, max: f64 },
    
    /// Required monotonicity constraint violated in input data
    ///
    /// Some numerical methods require monotonic (strictly increasing or
    /// decreasing) input data to function correctly. This error occurs when
    /// the monotonicity constraint is violated.
    ///
    /// # Common Causes
    /// - Interpolation with non-monotonic x-values
    /// - Integration with improperly ordered bounds
    /// - Optimization with non-monotonic objective functions
    ///
    /// # Recovery Strategies
    /// - Sort input data before processing
    /// - Use methods that don't require monotonicity
    /// - Apply data preprocessing to ensure monotonicity
    #[error("Input values must be strictly monotonic for {method}")]
    NotMonotonic { method: &'static str },
    
    /// Iterative algorithm failed to converge within specified limits
    ///
    /// Many numerical algorithms are iterative and may fail to converge to
    /// the desired accuracy within the maximum number of allowed iterations.
    /// This error indicates such a convergence failure.
    ///
    /// # Common Causes
    /// - Poor initial guesses for iterative methods
    /// - Ill-conditioned problems with slow convergence
    /// - Insufficient iteration limits for problem complexity
    /// - Numerical instabilities preventing convergence
    ///
    /// # Recovery Strategies
    /// - Increase maximum iteration count
    /// - Improve initial guess quality
    /// - Switch to more robust (but potentially slower) algorithms
    /// - Relax convergence tolerance if appropriate
    #[error("Failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },
    
    /// Invalid parameter value provided to numerical method
    ///
    /// This error occurs when parameter values are outside their valid ranges
    /// or violate method requirements. It includes a descriptive message
    /// explaining the specific parameter issue.
    ///
    /// # Common Causes
    /// - Negative tolerances for convergence criteria
    /// - Zero step sizes for finite difference methods
    /// - Invalid order specifications for approximation methods
    /// - Out-of-range options for algorithm variants
    ///
    /// # Recovery Strategies
    /// - Validate parameters before calling numerical methods
    /// - Use default parameter values when uncertain
    /// - Implement parameter range checking in calling code
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// Numerical instability detected during computation
    ///
    /// This error indicates that numerical instabilities (such as overflow,
    /// underflow, or loss of precision) have been detected during computation.
    /// It includes context about where the instability occurred.
    ///
    /// # Common Causes
    /// - Overflow in exponential computations
    /// - Loss of precision in subtractive cancellation
    /// - Division by very small numbers causing instability
    /// - Ill-conditioned matrix operations
    ///
    /// # Recovery Strategies
    /// - Use alternative algorithms with better numerical properties
    /// - Apply scaling or preconditioning to improve stability
    /// - Increase precision (e.g., use higher precision types)
    /// - Modify problem formulation to avoid instabilities
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
    
    /// Division by zero encountered in specific computational context
    ///
    /// This error occurs when a division by zero is encountered during
    /// numerical computation. It provides context about where the division
    /// by zero occurred to aid in debugging.
    ///
    /// # Common Causes
    /// - Zero derivatives in Newton-Raphson method
    /// - Zero denominators in rational interpolation
    /// - Singular matrices in linear system solving
    /// - Zero intervals in integration or differentiation
    ///
    /// # Recovery Strategies
    /// - Check for zero values before division
    /// - Use regularization techniques to avoid exact zeros
    /// - Switch to alternative methods that handle zero cases
    /// - Apply perturbation to avoid exact zero conditions
    #[error("Division by zero in {context}")]
    DivisionByZero { context: &'static str },
    
    /// Error during evaluation of user-provided function
    ///
    /// This error occurs when a user-provided function (such as those passed
    /// to integration or root-finding methods) fails to evaluate correctly.
    /// It wraps the underlying error for propagation.
    ///
    /// # Common Causes
    /// - Function evaluation at invalid domain points
    /// - Panics or errors within user-provided closures
    /// - Mathematical domain errors (sqrt of negative, etc.)
    /// - Resource exhaustion during function evaluation
    ///
    /// # Recovery Strategies
    /// - Add domain checking to user functions
    /// - Use safe mathematical operations (checked arithmetic)
    /// - Implement graceful error handling in function closures
    /// - Validate function domains before numerical operations
    #[error("Function evaluation failed: {0}")]
    FunctionEvaluation(String),
    
    /// Specific error during numerical integration operations
    ///
    /// This error covers integration-specific problems that don't fit into
    /// other error categories. It provides detailed context about the
    /// integration failure.
    ///
    /// # Common Causes
    /// - Divergent integrals that don't converge
    /// - Improper handling of infinite integration bounds
    /// - Oscillatory functions causing quadrature difficulties
    /// - Singularities within integration domain
    ///
    /// # Recovery Strategies
    /// - Use adaptive quadrature methods for difficult integrands
    /// - Apply coordinate transformations for infinite domains
    /// - Split integration domain around known singularities
    /// - Use specialized methods for oscillatory integrands
    #[error("Integration error: {0}")]
    IntegrationError(String),
    
    /// Error from underlying RustLab-Math operations
    ///
    /// This variant wraps errors from the RustLab-Math crate, allowing
    /// seamless error propagation from lower-level mathematical operations.
    /// The original error is preserved for full context.
    ///
    /// # Common Causes
    /// - Matrix operations failing in underlying linear algebra
    /// - Vector operations with dimension mismatches
    /// - Memory allocation failures in large computations
    /// - Hardware-specific numerical errors
    ///
    /// # Recovery Strategies
    /// - Check the wrapped error for specific failure details
    /// - Apply appropriate recovery based on the underlying error type
    /// - Consider alternative data structures or algorithms
    /// - Reduce problem size if memory-related errors occur
    #[error(transparent)]
    CoreError(#[from] rustlab_math::error::MathError),
}

/// Convenience type alias for numerical computation results
///
/// This type alias simplifies function signatures throughout the crate by
/// providing a standard Result type specialized for numerical errors.
///
/// # Usage
///
/// ```rust
/// use rustlab_numerical::Result;
///
/// fn my_numerical_function(x: f64) -> Result<f64> {
///     if x < 0.0 {
///         Err(rustlab_numerical::NumericalError::InvalidParameter(
///             "x must be non-negative".to_string()
///         ))
///     } else {
///         Ok(x.sqrt())
///     }
/// }
/// ```
///
/// # Error Propagation
///
/// The `?` operator can be used naturally with this type:
///
/// ```rust,ignore
/// fn complex_computation(data: &[f64]) -> Result<f64> {
///     let interpolated = interpolate_data(data)?;
///     let integrated = integrate_function(interpolated)?;
///     let root = find_root(integrated)?
///     Ok(root)
/// }
/// ```
pub type Result<T> = std::result::Result<T, NumericalError>;