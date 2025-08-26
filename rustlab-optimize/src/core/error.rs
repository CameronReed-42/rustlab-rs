//! Error handling for optimization operations

use thiserror::Error;

/// Result type for optimization operations
pub type Result<T> = std::result::Result<T, Error>;

/// Comprehensive error types for optimization and curve fitting operations
/// 
/// # For AI Code Generation
/// - All optimization functions return Result<T, Error>
/// - Use ? operator for error propagation: let result = minimize(f).solve()?;
/// - Common errors: ConvergenceFailed (bad initial guess), DimensionMismatch (data size)
/// - Handle errors with match or use .unwrap() only for known-good cases
/// - Error messages include diagnostic information for debugging
#[derive(Error, Debug)]
pub enum Error {
    /// Invalid input parameters or function arguments
    /// 
    /// # For AI Code Generation
    /// - Common causes: missing initial point, empty data arrays, invalid model function
    /// - Fix: Check function arguments, ensure all required parameters provided
    /// - Example fix: Always use .from(&initial) for minimize() functions
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Optimization algorithm failed to find a solution within iteration limit
    /// 
    /// # For AI Code Generation
    /// - Common causes: poor initial guess, wrong algorithm, pathological function
    /// - Fix strategies: try different starting point, increase tolerance, change algorithm
    /// - For curve fitting: check if model form matches data pattern
    /// - Use .max_iterations(n) to increase limit if making progress
    #[error("Failed to converge after {iterations} iterations: {reason}")]
    ConvergenceFailed {
        /// Number of iterations performed
        iterations: usize,
        /// Reason for failure
        reason: String,
    },

    /// Array dimensions don't match between x_data, y_data, or parameter arrays
    /// 
    /// # For AI Code Generation
    /// - Common causes: x_data.len() != y_data.len(), bounds arrays wrong size
    /// - Fix: Ensure all data arrays have same length, bounds match parameter count
    /// - For curve fitting: x and y data must have identical length
    /// - For bounds: lower.len() == upper.len() == n_parameters
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Numerical instability: NaN, infinity, or severe ill-conditioning
    /// 
    /// # For AI Code Generation
    /// - Common causes: objective function returns NaN/inf, singular matrices
    /// - Fix: add parameter bounds, check for division by zero, scale variables
    /// - For curve fitting: ensure positive data where required (exponential models)
    /// - Consider using different algorithm or regularization
    #[error("Numerical error: {message}")]
    NumericalError {
        /// Error message
        message: String,
    },

    /// Parameter value exceeds specified bounds during optimization
    /// 
    /// # For AI Code Generation
    /// - Indicates bounds are too restrictive or initial point is infeasible
    /// - Fix: check initial point is within bounds, relax bounds if necessary
    /// - Modern algorithms handle bounds automatically - this error is rare
    /// - If persistent: bounds may be inconsistent with problem physics
    #[error("Parameter bounds violated: {parameter} = {value} not in [{lower}, {upper}]")]
    BoundsViolation {
        /// Parameter index
        parameter: usize,
        /// Parameter value
        value: f64,
        /// Lower bound
        lower: f64,
        /// Upper bound
        upper: f64,
    },

    /// Internal algorithm failure or unsupported operation
    /// 
    /// # For AI Code Generation
    /// - Rare error indicating algorithm limitation or bug
    /// - Fix: try different algorithm or report issue if reproducible
    /// - May indicate unsupported problem type for chosen algorithm
    /// - Consider automatic algorithm selection instead of explicit choice
    #[error("Algorithm error: {0}")]
    AlgorithmError(String),
}

impl Error {
    /// Create a convergence failure error
    pub fn convergence_failed(iterations: usize, reason: impl Into<String>) -> Self {
        Self::ConvergenceFailed {
            iterations,
            reason: reason.into(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create a numerical error
    pub fn numerical_error(message: impl Into<String>) -> Self {
        Self::NumericalError {
            message: message.into(),
        }
    }

    /// Create a bounds violation error
    pub fn bounds_violation(parameter: usize, value: f64, lower: f64, upper: f64) -> Self {
        Self::BoundsViolation {
            parameter,
            value,
            lower,
            upper,
        }
    }

    /// Create an algorithm-specific error
    pub fn algorithm_error(message: impl Into<String>) -> Self {
        Self::AlgorithmError(message.into())
    }
}