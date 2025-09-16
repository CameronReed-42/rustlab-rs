//! Advanced descriptive statistics beyond basic measures
//!
//! This module extends the basic statistical operations in rustlab-math with sophisticated
//! descriptive statistics, robust measures, and distribution analysis tools. It provides
//! the foundation for exploratory data analysis, outlier detection, and comprehensive
//! data characterization.
//!
//! # Module Organization
//!
//! ## Core Submodules
//!
//! ### `descriptive` - Advanced Central Tendency and Spread
//! - **Robust measures**: Geometric mean, harmonic mean, trimmed mean
//! - **Variability measures**: Median Absolute Deviation (MAD), coefficient of variation
//! - **Distribution summaries**: Mode detection, range calculations
//! - **Use cases**: Outlier-resistant analysis, financial statistics, quality metrics
//!
//! ### `quantiles` - Percentile and Quantile Analysis  
//! - **Quantile methods**: Multiple interpolation approaches (linear, nearest, etc.)
//! - **Percentile operations**: Any percentile from 0-100%
//! - **Quartile analysis**: Q1, Q2 (median), Q3, and IQR calculations
//! - **Use cases**: Risk analysis, performance benchmarking, data distribution assessment
//!
//! ### `shape` - Distribution Shape Characterization
//! - **Asymmetry measures**: Skewness analysis for distribution tail behavior
//! - **Tail behavior**: Kurtosis (excess and raw) for heavy/light tail detection
//! - **Moment analysis**: Higher-order central moments for theoretical work
//! - **Use cases**: Normality testing, risk modeling, distribution fitting
//!
//! ### `arrays` - Multidimensional Statistical Operations
//! - **Axis-wise operations**: Apply any statistic along rows or columns
//! - **Matrix statistics**: Comprehensive statistics for 2D data
//! - **Broadcasting support**: Efficient operations on large arrays
//! - **Use cases**: Feature analysis, time series matrices, image processing
//!
//! # Mathematical Foundations
//!
//! ## Robust Statistics Theory
//!
//! Many statistics in this module are designed to be robust - meaning they are
//! less sensitive to outliers and violations of distributional assumptions:
//!
//! **Breakdown Point**: The proportion of contaminated data a statistic can handle
//! - Median: 50% breakdown point
//! - MAD: 50% breakdown point  
//! - Mean: 0% breakdown point (single outlier affects result)
//!
//! **Influence Function**: How much a single observation can affect the statistic
//! - Median: Bounded influence
//! - MAD: Bounded influence
//! - Mean: Unbounded influence
//!
//! ## Distribution Analysis Framework
//!
//! The shape analysis tools provide comprehensive distribution characterization:
//!
//! ```text
//! Location: mean, median, mode, trimmed mean
//!     ↓
//! Scale: std dev, MAD, IQR, range  
//!     ↓
//! Shape: skewness (asymmetry), kurtosis (tails)
//!     ↓
//! Complete distributional picture
//! ```
//!
//! # Integration Patterns
//!
//! ## With Basic Statistics
//!
//! This module builds upon rustlab-math foundations:
//!
//! ```rust
//! use rustlab_stats::prelude::*;
//! use rustlab_math::vec64;
//!
//! let data = vec64![1, 2, 3, 4, 5, 100]; // Contains outlier
//!
//! // Basic statistics (rustlab-math)
//! let mean = data.mean();        // Affected by outlier
//! let std = data.std(None);      // Inflated by outlier
//!
//! // Advanced robust statistics (rustlab-stats)
//! let median = data.median();    // Robust to outlier  
//! let mad = data.mad();          // Robust spread measure
//! let iqr = data.iqr();          // Quartile-based spread
//! ```
//!
//! ## Cross-Module Synergy
//!
//! Advanced descriptive statistics work seamlessly with other rustlab-stats modules:
//!
//! ```rust
//! // Distribution analysis pipeline
//! let data = vec64![/* sample data */];
//!
//! // 1. Descriptive analysis
//! let skew = data.skewness();
//! let kurt = data.kurtosis();
//!
//! // 2. Choose appropriate tests based on distribution shape
//! let test_result = if skew.abs() < 0.5 && kurt.abs() < 1.0 {
//!     // Approximately normal - use parametric test
//!     data.ttest_1samp(0.0, Alternative::TwoSided)
//! } else {
//!     // Non-normal - use non-parametric test
//!     data.wilcoxon(None, Alternative::TwoSided).unwrap()
//! };
//!
//! // 3. Apply appropriate normalization
//! let normalized = if /* has outliers */ {
//!     data.robust_scale()  // Robust to outliers
//! } else {
//!     data.zscore(None)    // Standard normalization
//! };
//! ```
//!
//! # Performance and Efficiency
//!
//! ## Computational Complexity
//!
//! | Operation Category | Typical Complexity | Notes |
//! |-------------------|-------------------|-------|
//! | Basic descriptive | O(n) | Single pass |
//! | Quantiles | O(n log n) | Sorting required |
//! | Mode detection | O(n log n) | Frequency analysis |
//! | Shape measures | O(n) | After basic stats |
//! | Array operations | O(n × m) | For n×m arrays |
//!
//! ## Memory Efficiency
//!
//! - **Zero-copy design**: Operations work directly on existing data
//! - **Minimal allocation**: Only temporary arrays for sorting when necessary
//! - **Streaming capable**: Many operations can be computed incrementally
//!
//! # Applications and Use Cases
//!
//! ## Exploratory Data Analysis (EDA)
//! ```rust
//! // Comprehensive data summary
//! let summary = DataSummary {
//!     location: (data.mean(), data.median(), data.mode()),
//!     spread: (data.std(None), data.mad(), data.iqr()),
//!     shape: (data.skewness(), data.kurtosis()),
//!     range: (data.min(), data.max(), data.range())
//! };
//! ```
//!
//! ## Quality Control and Manufacturing
//! ```rust
//! // Process monitoring with robust statistics
//! let measurements = vec64![/* sensor readings */];
//! let robust_center = measurements.median();
//! let robust_spread = measurements.mad();
//! let outlier_threshold = robust_center + 3.0 * robust_spread;
//! ```
//!
//! ## Financial Risk Analysis
//! ```rust
//! // Portfolio return analysis
//! let returns = vec64![/* daily returns */];
//! let downside_risk = returns.quantile(0.05, None);  // 5% VaR
//! let tail_risk = returns.kurtosis();                // Tail heaviness
//! let asymmetry = returns.skewness();                // Return asymmetry
//! ```
//!
//! # Best Practices
//!
//! ## Robust vs Classical Statistics
//! - Use robust statistics (median, MAD) when outliers are present
//! - Use classical statistics (mean, std) for theoretical work with normal data
//! - Compare both approaches to understand data characteristics
//!
//! ## Distribution Assessment
//! - Always examine skewness and kurtosis before choosing statistical methods
//! - Use quantile-quantile plots alongside numerical shape measures
//! - Consider transformation if distribution is highly non-normal
//!
//! ## Performance Optimization
//! - For repeated operations on same data, consider caching sorted versions
//! - Use axis-wise operations for multidimensional analysis
//! - Enable parallel processing features for large datasets

pub mod arrays;
pub mod descriptive;
pub mod quantiles;
pub mod shape;

pub use arrays::*;
pub use descriptive::*;
pub use quantiles::*;
pub use shape::*;