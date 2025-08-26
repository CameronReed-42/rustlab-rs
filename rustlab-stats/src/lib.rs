//! RustLab Stats - Advanced Statistical Analysis and Data Science Toolkit
//!
//! A comprehensive, high-performance statistical analysis library for the RustLab ecosystem,
//! designed to complement rustlab-math with advanced statistical operations, hypothesis testing,
//! and data preprocessing capabilities. This crate bridges the gap between basic mathematical
//! operations and sophisticated statistical analysis.
//!
//! # Design Philosophy and Architecture
//!
//! ## Core Principles
//!
//! - **Math-First Design**: Clean, mathematical APIs without verbose prefixes
//! - **Type-Driven Safety**: Leverage Rust's type system for statistical correctness
//! - **Zero-Copy Efficiency**: Direct operations on existing data structures
//! - **Composable Operations**: Seamless integration with rustlab-math ecosystem
//! - **Performance-Oriented**: SIMD optimizations and parallel processing support
//! - **Statistical Rigor**: Mathematically sound implementations with proper bias corrections
//!
//! ## Ecosystem Integration
//!
//! RustLab-Stats builds upon rustlab-math's foundation:
//! 
//! ```text
//! rustlab-math (Basic Operations)
//!     ↓
//! rustlab-stats (Advanced Analysis)
//!     ↓
//! Your Application (Domain-Specific Analysis)
//! ```
//!
//! # Module Organization
//!
//! ## Core Statistical Modules
//!
//! ### Advanced Descriptive Statistics (`advanced`)
//! - **Purpose**: Beyond mean/variance - quantiles, shape measures, robust statistics
//! - **Key Features**: Median, IQR, skewness, kurtosis, MAD, percentiles
//! - **Use Cases**: Exploratory data analysis, outlier detection, distribution characterization
//!
//! ### Correlation and Covariance Analysis (`correlation`) 
//! - **Purpose**: Relationship analysis between variables
//! - **Key Features**: Pearson, Spearman, Kendall correlations; covariance matrices
//! - **Use Cases**: Feature selection, dependency analysis, multivariate statistics
//!
//! ### Hypothesis Testing (`hypothesis`)
//! - **Purpose**: Statistical inference and significance testing
//! - **Key Features**: t-tests, Mann-Whitney U, Wilcoxon, chi-square tests
//! - **Use Cases**: A/B testing, scientific research, quality control
//!
//! ### Data Normalization (`normalization`)
//! - **Purpose**: Data preprocessing and scaling for ML/analysis
//! - **Key Features**: Z-score, robust scaling, min-max, unit vector normalization
//! - **Use Cases**: Machine learning preprocessing, feature scaling, data standardization
//!
//! ### Performance Optimization (`performance`)
//! - **Purpose**: High-performance implementations and benchmarking
//! - **Key Features**: SIMD acceleration, parallel processing, streaming algorithms
//! - **Use Cases**: Large-scale data analysis, real-time processing, performance tuning
//!
//! # Quick Start Guide
//!
//! ## Basic Usage Pattern
//!
//! ```rust
//! use rustlab_stats::prelude::*;
//! use rustlab_math::vec64;
//!
//! // Create data using rustlab-math
//! let data = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
//!
//! // Basic statistics (rustlab-math)
//! let mean = data.mean();     // 5.5
//! let std = data.std(None);   // ~3.03
//!
//! // Advanced statistics (rustlab-stats)
//! let median = data.median();         // 5.5
//! let iqr = data.iqr();              // 5.0 (Q3 - Q1)
//! let skewness = data.skewness();    // ~0.0 (symmetric)
//! ```
//!
//! ## Hypothesis Testing Example
//!
//! ```rust
//! use rustlab_stats::prelude::*;
//! use rustlab_math::vec64;
//!
//! // A/B test scenario
//! let control_group = vec64![23, 25, 21, 24, 26, 22, 28];
//! let treatment_group = vec64![28, 30, 26, 32, 29, 31, 33];
//!
//! // Parametric test (assumes normality)
//! let t_result = control_group.ttest_ind(&treatment_group, Alternative::TwoSided);
//! println!("t-test p-value: {:.4}", t_result.p_value);
//!
//! // Non-parametric alternative (no normality assumption)
//! let u_result = control_group.mannwhitneyu(&treatment_group, Alternative::TwoSided);
//! println!("Mann-Whitney p-value: {:.4}", u_result.p_value);
//! ```
//!
//! ## Data Preprocessing Pipeline
//!
//! ```rust
//! use rustlab_stats::prelude::*;
//! use rustlab_math::vec64;
//!
//! // Raw data with potential outliers
//! let raw_data = vec64![10, 12, 11, 13, 12, 14, 100]; // 100 is outlier
//!
//! // Robust preprocessing pipeline
//! let median_centered = raw_data.robust_scale();  // Less sensitive to outlier
//! let standardized = raw_data.zscore(None);       // Traditional standardization
//! let bounded = raw_data.minmax_scale();          // Scale to [0,1]
//! ```
//!
//! ## Multivariate Analysis
//!
//! ```rust
//! use rustlab_stats::prelude::*;
//! use rustlab_math::{vec64, ArrayF64};
//!
//! // Create sample dataset
//! let height = vec64![165, 170, 175, 180, 185];
//! let weight = vec64![55, 65, 70, 80, 85];
//!
//! // Correlation analysis
//! let pearson_r = height.pearson_correlation(&weight);
//! let spearman_rho = height.spearman_correlation(&weight);
//!
//! // Array operations for multivariate data
//! let data_matrix = ArrayF64::from_slice(
//!     &[165.0, 55.0, 170.0, 65.0, 175.0, 70.0], 3, 2
//! ).unwrap();
//!
//! let col_medians = data_matrix.median_axis(Axis::Rows);
//! ```
//!
//! # Advanced Features
//!
//! ## Performance Optimization
//!
//! ```rust,ignore
//! use rustlab_stats::performance::*;
//!
//! // SIMD-accelerated operations (when available)
//! let data = vec64![/* large dataset */];
//! let fast_mean = data.mean_adaptive();  // Chooses best implementation
//!
//! // Parallel processing (with 'parallel' feature)
//! #[cfg(feature = "parallel")]
//! let parallel_var = data.var_parallel(None);
//!
//! // Streaming statistics for large datasets
//! let mut streaming = StreamingStats::<f64>::new();
//! for value in data_stream {
//!     streaming.update(value);
//! }
//! let running_mean = streaming.mean();
//! ```
//!
//! ## Robust Statistics for Real-World Data
//!
//! ```rust
//! use rustlab_stats::prelude::*;
//! use rustlab_math::vec64;
//!
//! // Financial returns with extreme events
//! let daily_returns = vec64![0.01, -0.02, 0.015, -0.008, -0.15]; // -15% crash day
//!
//! // Robust measures handle outliers better
//! let robust_center = daily_returns.median();     // Less affected by crash
//! let robust_spread = daily_returns.mad();        // Median absolute deviation
//! let distribution_shape = daily_returns.skewness(); // Detect asymmetry
//! ```
//!
//! # Feature Flags
//!
//! Enable additional functionality through Cargo features:
//!
//! ```toml
//! [dependencies]
//! rustlab-stats = { version = "*", features = ["parallel", "simd"] }
//! ```
//!
//! - **`parallel`**: Enable parallel processing with Rayon
//! - **`simd`**: Enable SIMD optimizations (x86_64 only)
//!
//! # Statistical Guarantees and Correctness
//!
//! ## Numerical Accuracy
//! - Bias-corrected sample statistics (proper degrees of freedom)
//! - Numerically stable algorithms (Welford's method for variance)
//! - Appropriate handling of edge cases and boundary conditions
//!
//! ## Mathematical Rigor
//! - Implementations based on established statistical literature
//! - Comprehensive test suites with known statistical properties
//! - Clear documentation of assumptions and limitations
//!
//! ## Error Handling Philosophy
//! - **Panics**: For mathematically invalid operations (division by zero, empty data)
//! - **Results**: For operations that may legitimately fail (convergence, insufficient data)
//! - **Type Safety**: Prevent common statistical errors at compile time
//!
//! # Integration with Scientific Computing Ecosystem
//!
//! RustLab-Stats is designed to integrate seamlessly with:
//! - **Data Loading**: CSV, JSON, database connectors
//! - **Visualization**: Plotting libraries for statistical graphics
//! - **Machine Learning**: scikit-learn-style preprocessing
//! - **Parallel Computing**: Distributed statistical computing
//!
//! # Performance Characteristics
//!
//! | Operation | Time Complexity | Space Complexity | Notes |
//! |-----------|----------------|------------------|-------|
//! | Basic descriptive | O(n) | O(1) | Single pass |
//! | Quantiles | O(n log n) | O(n) | Sorting required |
//! | Hypothesis tests | O(n) to O(n²) | O(1) to O(n) | Depends on test |
//! | Correlation | O(n) | O(1) | After preprocessing |
//! | SIMD operations | O(n/k) | O(1) | k = SIMD width |
//!
//! # Getting Help
//!
//! - **Documentation**: Comprehensive API docs with mathematical foundations
//! - **Examples**: Real-world usage patterns and best practices  
//! - **Tests**: Extensive test suite demonstrating expected behavior
//! - **Community**: GitHub discussions for questions and contributions

#![warn(missing_docs)]
#![allow(unsafe_code)] // Required for SIMD optimizations

// Core modules
pub mod advanced;
pub mod correlation;
pub mod hypothesis;
pub mod normalization;
pub mod performance;
pub mod prelude;

// Error handling
mod error;
pub use error::{StatsError, Result};

// Re-export rustlab-math for convenience
pub use rustlab_math;
