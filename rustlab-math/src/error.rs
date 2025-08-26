//! Error types for RustLab Math v2

use thiserror::Error;

/// Result type alias for RustLab Math operations
pub type Result<T> = std::result::Result<T, MathError>;

/// Error types for mathematical operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MathError {
    /// Dimension mismatch in operations
    #[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        /// Expected dimensions
        expected: (usize, usize),
        /// Actual dimensions
        actual: (usize, usize),
    },
    
    /// Invalid array dimensions
    #[error("Invalid dimensions: rows={rows}, cols={cols}")]
    InvalidDimensions {
        /// Number of rows
        rows: usize,
        /// Number of columns
        cols: usize,
    },
    
    /// Index out of bounds
    #[error("Index out of bounds: index={index}, size={size}")]
    IndexOutOfBounds {
        /// Requested index
        index: usize,
        /// Array size
        size: usize,
    },
    
    /// Invalid slice length for array creation
    #[error("Invalid slice length: expected {expected}, got {actual}")]
    InvalidSliceLength {
        /// Expected length
        expected: usize,
        /// Actual length
        actual: usize,
    },
    
    /// Slicing operation error
    #[error("Slicing error: {message}")]
    SlicingError {
        /// Error message
        message: String,
    },
}

/// Convenience functions for creating specific errors
impl MathError {
    /// Create a dimension mismatch error
    /// 
    /// # For AI Code Generation
    /// - Helper function for creating dimension mismatch errors
    /// - Use when implementing operations that require compatible dimensions
    /// - More convenient than constructing the enum variant manually
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::MathError;
    /// 
    /// fn check_compatible_dimensions(a_shape: (usize, usize), b_shape: (usize, usize)) -> Result<(), MathError> {
    ///     if a_shape.1 != b_shape.0 {
    ///         return Err(MathError::dimension_mismatch((a_shape.1, b_shape.0), b_shape));
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn dimension_mismatch(expected: (usize, usize), actual: (usize, usize)) -> Self {
        MathError::DimensionMismatch { expected, actual }
    }
    
    /// Create an invalid dimensions error
    /// 
    /// # For AI Code Generation
    /// - Helper function for creating invalid dimensions errors
    /// - Use when validating array creation parameters
    /// - More convenient than constructing the enum variant manually
    pub fn invalid_dimensions(rows: usize, cols: usize) -> Self {
        MathError::InvalidDimensions { rows, cols }
    }
    
    /// Create an index out of bounds error
    /// 
    /// # For AI Code Generation
    /// - Helper function for creating index out of bounds errors
    /// - Use when implementing array/vector access methods
    /// - More convenient than constructing the enum variant manually
    pub fn index_out_of_bounds(index: usize, size: usize) -> Self {
        MathError::IndexOutOfBounds { index, size }
    }
    
    /// Create an invalid slice length error
    /// 
    /// # For AI Code Generation
    /// - Helper function for creating slice length errors
    /// - Use when validating data for array construction
    /// - More convenient than constructing the enum variant manually
    pub fn invalid_slice_length(expected: usize, actual: usize) -> Self {
        MathError::InvalidSliceLength { expected, actual }
    }
}