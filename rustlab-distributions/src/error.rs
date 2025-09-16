//! Error types for the distributions library

use thiserror::Error;

/// Distribution-specific error type
#[derive(Error, Debug)]
pub enum DistributionError {
    /// Invalid parameter error
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// Invalid operation error
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    /// Numerical error
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    /// Sampling error
    #[error("Sampling error: {0}")]
    SamplingError(String),
    
    /// Math library error (when integration feature is enabled)
    #[cfg(feature = "integration")]
    #[error("RustLab math error: {0}")]
    MathError(String),
}

/// Result type for distribution operations
pub type Result<T> = std::result::Result<T, DistributionError>;

impl DistributionError {
    /// Create an invalid parameter error
    pub fn invalid_parameter(msg: impl Into<String>) -> Self {
        DistributionError::InvalidParameter(msg.into())
    }
    
    /// Create an invalid operation error
    pub fn invalid_operation(msg: impl Into<String>) -> Self {
        DistributionError::InvalidOperation(msg.into())
    }
    
    /// Create a numerical error
    pub fn numerical_error(msg: impl Into<String>) -> Self {
        DistributionError::NumericalError(msg.into())
    }
    
    /// Create a sampling error
    pub fn sampling_error(msg: impl Into<String>) -> Self {
        DistributionError::SamplingError(msg.into())
    }
}