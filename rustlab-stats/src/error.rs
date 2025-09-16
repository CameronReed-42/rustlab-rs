//! Error types for rustlab-stats
//!
//! Following the math-first philosophy, most operations should panic on invalid
//! input rather than return Results. However, some operations may legitimately
//! fail and should return Results.

use std::fmt;

/// Result type for rustlab-stats operations
pub type Result<T> = std::result::Result<T, StatsError>;

/// Error types for statistical operations
#[derive(Debug, Clone, PartialEq)]
pub enum StatsError {
    /// Insufficient data for the requested operation
    InsufficientData {
        /// Operation that was attempted
        operation: String,
        /// Required minimum sample size
        required: usize,
        /// Actual sample size
        actual: usize,
    },
    
    /// Convergence failure in iterative algorithms
    ConvergenceFailure {
        /// Algorithm that failed to converge
        algorithm: String,
        /// Maximum iterations reached
        max_iterations: usize,
        /// Final error/tolerance achieved
        final_error: f64,
    },
    
    /// Invalid parameters for statistical operations
    InvalidParameters {
        /// Parameter name
        parameter: String,
        /// Error message
        message: String,
    },
    
    /// Numerical instability detected
    NumericalInstability {
        /// Operation that became unstable
        operation: String,
        /// Additional context
        context: String,
    },
    
    /// Invalid input data or parameters
    InvalidInput(String),
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsError::InsufficientData { operation, required, actual } => {
                write!(f, "Insufficient data for {}: requires {} samples, got {}", 
                       operation, required, actual)
            },
            StatsError::ConvergenceFailure { algorithm, max_iterations, final_error } => {
                write!(f, "{} failed to converge after {} iterations (final error: {})", 
                       algorithm, max_iterations, final_error)
            },
            StatsError::InvalidParameters { parameter, message } => {
                write!(f, "Invalid parameter '{}': {}", parameter, message)
            },
            StatsError::NumericalInstability { operation, context } => {
                write!(f, "Numerical instability in {}: {}", operation, context)
            },
            StatsError::InvalidInput(message) => {
                write!(f, "Invalid input: {}", message)
            },
        }
    }
}

impl std::error::Error for StatsError {}