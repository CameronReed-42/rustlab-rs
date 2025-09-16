//! Seamless integration with RustLab-Math for advanced statistical computing
//!
//! This module provides comprehensive integration between `rustlab-distributions` and
//! `rustlab-math`, extending arrays and vectors with powerful distribution-based
//! functionality. It enables high-performance statistical computing workflows
//! by bridging probability theory with linear algebra operations.
//!
//! ## Module Overview
//!
//! - **array_ext**: Extension traits for 2D arrays with distribution operations
//! - **vector_ext**: Extension traits for 1D vectors with statistical methods
//! - **convenience**: High-level convenience functions for common statistical tasks
//! - **random_arrays**: Specialized functions for generating random matrices
//!
//! ## Key Features
//!
//! ### Distribution-Array Integration
//! - **Fill Operations**: Populate arrays with samples from any distribution
//! - **Elementwise PDF/CDF**: Apply distribution functions to array elements
//! - **Statistical Analysis**: Built-in descriptive statistics and hypothesis tests
//! - **Correlation Analysis**: Generate correlation matrices from data arrays
//!
//! ### High-Performance Sampling
//! - **Vectorized Operations**: Optimized batch sampling for large datasets
//! - **Memory Efficient**: In-place operations to minimize allocations
//! - **Zero-Copy Integration**: Seamless data flow between distributions and linear algebra
//!
//! ### Advanced Statistical Methods
//! - **Monte Carlo Integration**: Tools for numerical integration using random sampling
//! - **Bootstrap Methods**: Resampling techniques for statistical inference
//! - **Multivariate Analysis**: Support for high-dimensional statistical operations
//!
//! ## Usage
//!
//! Enable this module by including the "integration" feature in your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rustlab-distributions = { version = "0.1", features = ["integration"] }
//! rustlab-math = "0.1"
//! ```
//!
//! ## Examples
//!
//! ```rust
//! use rustlab_distributions::integration::*;
//! use rustlab_math::{ArrayF64, VectorF64};
//! use rand::thread_rng;
//!
//! let mut rng = thread_rng();
//!
//! // Create and fill 2D array with normal samples
//! let mut data = ArrayF64::zeros(100, 50);
//! data.fill_normal(0.0, 1.0, &mut rng).unwrap();
//!
//! // Apply statistical analysis
//! let (row_means, row_stds) = data.empirical_stats_rows().unwrap();
//! let correlation_matrix = data.correlation_matrix().unwrap();
//!
//! // Generate random vector with specific distribution
//! let exponential_samples = VectorF64::exponential(1000, 2.5, &mut rng).unwrap();
//! 
//! // Apply PDF to existing data
//! let pdf_values = exponential_samples.normal_pdf(0.0, 1.0).unwrap();
//! ```
//!
//! ## Performance Considerations
//!
//! - **Batch Operations**: Use array/vector methods for better cache locality
//! - **In-Place Updates**: Prefer `fill_*` methods over creating new arrays
//! - **Memory Layout**: Arrays use row-major order for optimal performance
//! - **SIMD Optimization**: Future versions will leverage vectorized operations
//!
//! ## Mathematical Foundation
//!
//! The integration leverages the mathematical relationship between probability
//! distributions and linear algebra:
//! - **Sampling**: X ~ F ⇒ X ∈ ℝ^n (vectorization of random variables)
//! - **Transformations**: Y = g(X) where g: ℝ^n → ℝ^m (distribution of functions)
//! - **Statistics**: E[X], Var[X], Cov[X,Y] computed via array operations
//! - **Hypothesis Testing**: Statistical inference through matrix computations

/// Extension traits for 2D arrays with comprehensive distribution functionality
#[cfg(feature = "integration")]
pub mod array_ext;

/// Extension traits for 1D vectors with statistical computing methods
#[cfg(feature = "integration")]
pub mod vector_ext;

/// High-level convenience functions for common statistical computing workflows
#[cfg(feature = "integration")]
pub mod convenience;

/// Specialized functions for generating random matrices with various structures
#[cfg(feature = "integration")]
pub mod random_arrays;

/// Re-export all array extension functionality
#[cfg(feature = "integration")]
pub use array_ext::*;

/// Re-export all vector extension functionality  
#[cfg(feature = "integration")]
pub use vector_ext::*;

/// Re-export all convenience functions
#[cfg(feature = "integration")]
pub use convenience::*;

/// Re-export all random array generation functions
#[cfg(feature = "integration")]
pub use random_arrays::*;

/// Re-export core continuous distributions for integrated workflows
#[cfg(feature = "integration")]
pub use crate::continuous::*;

/// Re-export core discrete distributions for integrated workflows
#[cfg(feature = "integration")]
pub use crate::discrete::*;

/// Re-export distribution traits for seamless integration
#[cfg(feature = "integration")]
pub use crate::traits::*;