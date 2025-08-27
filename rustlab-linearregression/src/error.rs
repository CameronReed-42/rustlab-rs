//! Error handling for linear regression operations

use thiserror::Error;

/// Errors that can occur during linear regression operations
#[derive(Error, Debug)]
pub enum LinearRegressionError {
    #[error("Dimension mismatch: X has {x_rows} rows but y has {y_len} elements")]
    /// Input matrices have incompatible dimensions
    DimensionMismatch { 
        /// Number of rows in X matrix
        x_rows: usize, 
        /// Length of y vector
        y_len: usize 
    },
    
    #[error("Invalid input: {0}")]
    /// Invalid input data provided
    InvalidInput(String),
    
    #[error("Singular matrix: cannot invert matrix with condition number {cond}")]
    /// Matrix is singular and cannot be inverted
    SingularMatrix { 
        /// Condition number of the matrix
        cond: f64 
    },
    
    #[error("Convergence failed after {iterations} iterations")]
    /// Iterative algorithm failed to converge
    ConvergenceFailed { 
        /// Number of iterations performed
        iterations: usize 
    },
    
    #[error("Numerical instability detected: {0}")]
    /// Numerical instability detected during computation
    NumericalInstability(String),
    
    #[error("Not fitted: model must be fitted before {operation}")]
    /// Model has not been fitted yet
    NotFitted { 
        /// Operation that was attempted
        operation: String 
    },
    
    #[error("Invalid parameter: {name} = {value} (must be {constraint})")]
    /// Invalid parameter value provided
    InvalidParameter {
        /// Name of the parameter
        name: String,
        /// Actual value provided
        value: f64,
        /// Constraint description
        constraint: String,
    },
    
    #[error("Linear algebra error: {0}")]
    /// Error from underlying linear algebra operations
    LinearAlgebra(String),
    
    #[error("Statistics error: {0}")]
    /// Error from underlying statistics operations
    Statistics(String),
    
    #[error("Cross-validation error: {0}")]
    /// Error during cross-validation
    CrossValidation(String),
}

/// Convenient type alias for Results with LinearRegressionError
pub type Result<T> = std::result::Result<T, LinearRegressionError>;

impl From<rustlab_linearalgebra::LinearAlgebraError> for LinearRegressionError {
    fn from(err: rustlab_linearalgebra::LinearAlgebraError) -> Self {
        LinearRegressionError::LinearAlgebra(err.to_string())
    }
}

impl From<rustlab_stats::StatsError> for LinearRegressionError {
    fn from(err: rustlab_stats::StatsError) -> Self {
        LinearRegressionError::Statistics(err.to_string())
    }
}