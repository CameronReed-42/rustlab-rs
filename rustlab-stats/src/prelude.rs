//! Prelude module for convenient and comprehensive imports
//!
//! The prelude provides a curated selection of the most commonly used types, traits,
//! and functions from rustlab-stats, enabling productive statistical analysis with
//! a single import statement.
//!
//! # Usage Pattern
//!
//! ```rust
//! use rustlab_stats::prelude::*;
//! use rustlab_math::vec64;
//!
//! // Now you have access to all statistical functionality
//! let data = vec64![1, 2, 3, 4, 5];
//! let median = data.median();          // From quantiles module
//! let skew = data.skewness();          // From shape module  
//! let scaled = data.zscore(None);      // From normalization module
//! let test = data.ttest_1samp(0.0, Alternative::TwoSided); // From hypothesis module
//! ```
//!
//! # What's Included
//!
//! This prelude re-exports:
//!
//! ## Core Data Types
//! - `VectorF64`, `VectorF32`: From rustlab-math for seamless integration
//! - `ArrayF64`, `ArrayF32`: For multidimensional statistical operations
//!
//! ## Statistical Traits
//! - `Quantiles`: Median, IQR, percentiles, quartiles
//! - `Shape`: Skewness, kurtosis, distribution shape analysis  
//! - `AdvancedDescriptive`: Mode, MAD, geometric mean, robust statistics
//! - `Correlation`, `Covariance`: Relationship analysis between variables
//! - `ParametricTests`, `NonParametricTests`: Hypothesis testing
//! - `Normalization`: Data preprocessing and scaling
//! - `AdvancedArrayStatistics`: Axis-wise operations on multidimensional data
//!
//! ## Utility Types
//! - `Alternative`: Hypothesis test direction specification
//! - `TestResult`: Structured hypothesis test results
//! - `CorrelationMethod`: Different correlation approaches
//! - `QuantileMethod`: Quantile interpolation methods
//! - `StatsError`, `Result`: Error handling types
//!
//! ## Performance Features (when enabled)
//! - `AdaptiveStats`: Automatically choose fastest implementation
//! - `ParallelStats`: Parallel processing capabilities
//! - `ZeroCopyStats`: Memory-efficient operations
//! - `StreamingStats`: Online/streaming statistical algorithms
//!
//! # Design Rationale
//!
//! The prelude is carefully curated to:
//!
//! 1. **Maximize Productivity**: Include all commonly used functionality
//! 2. **Minimize Conflicts**: Avoid name collisions with standard library
//! 3. **Maintain Clarity**: Don't hide important type information
//! 4. **Enable Discovery**: Make statistical methods easily discoverable
//!
//! # Alternative Import Strategies
//!
//! If you prefer more explicit imports or want to avoid potential naming conflicts:
//!
//! ```rust
//! // Selective imports
//! use rustlab_stats::advanced::{Quantiles, Shape};
//! use rustlab_stats::hypothesis::{ParametricTests, Alternative};
//! use rustlab_stats::normalization::Normalization;
//!
//! // Module-level imports
//! use rustlab_stats::{advanced, hypothesis, correlation};
//! 
//! // Qualified usage
//! use rustlab_stats as stats;
//! let median = stats::advanced::Quantiles::median(&data);
//! ```
//!
//! # Performance Considerations
//!
//! The prelude import itself has zero runtime cost - all trait methods are
//! resolved at compile time. However, some imported functionality may have
//! different performance characteristics:
//!
//! - Basic operations (mean, median): O(n) time complexity
//! - Quantile operations: O(n log n) due to sorting requirements
//! - Hypothesis tests: Varies by test type and sample size
//! - SIMD operations: Require CPU feature detection at runtime

// Re-export rustlab-math core types for convenience
pub use rustlab_math::{VectorF64, VectorF32, ArrayF64, ArrayF32};

// Advanced descriptive statistics
pub use crate::advanced::*;

// Correlation and covariance analysis
pub use crate::correlation::*;

// Hypothesis testing
pub use crate::hypothesis::*;

// Normalization and scaling
pub use crate::normalization::*;

// Performance optimization
pub use crate::performance::*;

// Error types
pub use crate::{StatsError, Result};