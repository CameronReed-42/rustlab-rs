//! Error handling for linear regression operations

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LinearRegressionError {
    #[error("Dimension mismatch: X has {x_rows} rows but y has {y_len} elements")]
    DimensionMismatch { x_rows: usize, y_len: usize },
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Singular matrix: cannot invert matrix with condition number {cond}")]
    SingularMatrix { cond: f64 },
    
    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },
    
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
    
    #[error("Not fitted: model must be fitted before {operation}")]
    NotFitted { operation: String },
    
    #[error("Invalid parameter: {name} = {value} (must be {constraint})")]
    InvalidParameter {
        name: String,
        value: f64,
        constraint: String,
    },
    
    #[error("Linear algebra error: {0}")]
    LinearAlgebra(String),
    
    #[error("Statistics error: {0}")]
    Statistics(String),
    
    #[error("Cross-validation error: {0}")]
    CrossValidation(String),
}

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