//! Quantile and percentile calculations with multiple interpolation methods
//!
//! This module provides comprehensive functionality for computing quantiles, percentiles,
//! quartiles, and related order statistics. It supports multiple interpolation methods
//! to handle the ambiguity that arises when the desired quantile falls between two
//! data points.
//!
//! # Mathematical Foundation
//!
//! ## Quantiles and Percentiles
//!
//! A quantile q (where 0 ≤ q ≤ 1) is a value that divides a probability distribution
//! such that a proportion q of the distribution lies below that value. Percentiles are
//! quantiles expressed as percentages (0-100%).
//!
//! For a sample of size n, the theoretical position of the q-th quantile is:
//! ```text
//! position = q × (n - 1) + 1
//! ```
//!
//! ## Interpolation Methods
//!
//! When the position falls between two data points, different methods handle interpolation:
//!
//! ### Linear Interpolation (R Type 7 - Default)
//! Most commonly used, provides smooth estimates:
//! ```text
//! Q(q) = Xₗ + (position - k) × (Xₗ₊₁ - Xₗ)
//! ```
//! where k = floor(position) and X values are sorted observations.
//!
//! ### Nearest Methods
//! - **Lower**: Always use the lower adjacent value
//! - **Higher**: Always use the higher adjacent value  
//! - **Nearest**: Use the closest value (round to nearest)
//! - **Midpoint**: Average of lower and higher values
//!
//! ## Special Cases and Edge Behavior
//!
//! - **q = 0**: Always returns minimum value
//! - **q = 1**: Always returns maximum value
//! - **Single Value**: All quantiles return that single value
//! - **Two Values**: Linear interpolation becomes simple average for q = 0.5
//!
//! # Statistical Applications
//!
//! ## Quartiles and Box Plots
//! - **Q1 (25th percentile)**: Lower quartile, bottom 25% of data
//! - **Q2 (50th percentile)**: Median, middle value
//! - **Q3 (75th percentile)**: Upper quartile, top 25% of data
//! - **IQR = Q3 - Q1**: Interquartile range, measure of spread
//!
//! ## Robust Statistics
//! Quantiles are robust to outliers and provide distribution-free measures:
//! - **Median**: More robust central tendency than mean
//! - **IQR**: More robust spread measure than standard deviation
//! - **Quantile Range**: Trimmed range excluding extreme values
//!
//! # Performance Characteristics
//!
//! - **Time Complexity**: O(n log n) due to sorting requirement
//! - **Space Complexity**: O(n) for temporary sorted copy
//! - **Numerical Stability**: Exact for discrete methods, stable interpolation
//! - **Memory Efficiency**: Single sorting pass, minimal additional allocation
//!
//! # Usage Guidelines
//!
//! ## Method Selection
//! - **Linear**: Most common, good for continuous distributions
//! - **Nearest methods**: When exact data values are meaningful
//! - **Lower/Higher**: For conservative estimates or discrete data
//! - **Midpoint**: Simple average, good for symmetric data
//!
//! ## Sample Size Considerations
//! - **Small samples (n < 10)**: Method choice can significantly affect results
//! - **Large samples (n > 100)**: Methods converge to similar values
//! - **Very large samples**: Consider approximate algorithms for efficiency

use rustlab_math::{VectorF64, VectorF32};

/// Methods for computing quantiles when the desired position falls between data points
///
/// Different fields and applications prefer different approaches to handling the ambiguity
/// that arises when a quantile position doesn't correspond exactly to an observed data point.
/// Each method represents a different strategy for interpolation or value selection.
///
/// # Method Comparison
///
/// | Method | Continuous? | Matches SAS | Matches R | Common Use |
/// |--------|-------------|-------------|-----------|------------|
/// | Linear | Yes | PCTLDEF=5 | type=7 (default) | General purpose |
/// | Lower | No | PCTLDEF=3 | type=1 | Conservative estimates |
/// | Higher | No | PCTLDEF=2 | type=3 | Pessimistic estimates |
/// | Nearest | No | PCTLDEF=4 | type=2 | Discrete data |
/// | Midpoint | Yes | PCTLDEF=1 | type=4 | Simple average |
///
/// # Detailed Descriptions
///
/// ## Linear Interpolation
/// **Most commonly used method** that provides smooth, continuous quantile estimates.
/// When the quantile position falls between ranks k and k+1, it linearly interpolates:
/// ```text
/// Q = Xₖ + fraction × (Xₖ₊₁ - Xₖ)
/// ```
/// - **Advantages**: Smooth estimates, widely accepted standard
/// - **Disadvantages**: May produce values not present in original data
/// - **Use cases**: Continuous data, standard statistical reporting
///
/// ## Lower Value Method
/// Always selects the **lower adjacent data point** when position is between ranks.
/// Provides conservative estimates that never exceed what was actually observed.
/// - **Advantages**: Always returns actual data values, conservative
/// - **Disadvantages**: Can underestimate true population quantiles
/// - **Use cases**: Risk analysis, regulatory reporting, discrete data
///
/// ## Higher Value Method  
/// Always selects the **higher adjacent data point** when position is between ranks.
/// Provides pessimistic estimates useful for worst-case scenario planning.
/// - **Advantages**: Always returns actual data values, pessimistic
/// - **Disadvantages**: Can overestimate true population quantiles
/// - **Use cases**: Safety margins, quality control, upper bounds
///
/// ## Nearest Value Method
/// Selects the **closest data point** by rounding the position to the nearest integer.
/// Provides a natural discrete approximation to continuous quantiles.
/// - **Advantages**: Returns actual data values, intuitive for discrete data
/// - **Disadvantages**: Discontinuous, can be unstable for small samples
/// - **Use cases**: Survey data, discrete measurements, ordinal scales
///
/// ## Midpoint Method
/// Computes the **arithmetic mean** of the two adjacent data points when position
/// falls between ranks. Provides a simple averaging approach.
/// - **Advantages**: Smooth for large samples, easy to understand
/// - **Disadvantages**: Less sophisticated than linear interpolation
/// - **Use cases**: Simple applications, educational contexts
///
/// # Example Comparison
///
/// For data [1, 2, 3, 4, 5] and q = 0.3 (position = 2.2):
/// - **Linear**: 1.8 + 0.2 × (2 - 1.8) = 2.2
/// - **Lower**: 2 (floor of position 2.2)
/// - **Higher**: 3 (ceil of position 2.2)
/// - **Nearest**: 2 (round position 2.2 to 2)
/// - **Midpoint**: (2 + 3) / 2 = 2.5
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantileMethod {
    /// Linear interpolation (most common, equivalent to R's type 7)
    Linear,
    /// Lower value (nearest rank method)
    Lower,
    /// Higher value  
    Higher,
    /// Midpoint between lower and higher
    Midpoint,
    /// Nearest value
    Nearest,
}

impl Default for QuantileMethod {
    fn default() -> Self {
        QuantileMethod::Linear
    }
}

/// Trait for quantile operations on vectors
///
/// This trait provides a comprehensive interface for computing quantiles, percentiles,
/// and related order statistics on numerical data. All methods handle edge cases
/// gracefully and provide mathematically sound results.
///
/// # Mathematical Guarantees
///
/// - **Monotonicity**: quantile(q1) ≤ quantile(q2) when q1 ≤ q2
/// - **Boundary Conditions**: quantile(0) = min, quantile(1) = max
/// - **Consistency**: Same results for equivalent data orderings
/// - **Interpolation Accuracy**: Linear method provides optimal continuous estimates
///
/// # Error Handling Philosophy
///
/// This trait follows the "math-first" philosophy of panicking on invalid inputs
/// rather than returning Result types, as quantile computation is a fundamental
/// mathematical operation that should not fail during normal usage.
///
/// # Performance Notes
///
/// All quantile operations require sorting the data, resulting in O(n log n) complexity.
/// For repeated quantile calculations on the same data, consider sorting once and
/// using the sorted array with direct quantile functions.
pub trait Quantiles<T> {
    /// Compute the specified quantile using configurable interpolation method
    ///
    /// This is the core quantile computation function that supports all interpolation
    /// methods. It handles the mathematical complexity of determining exact positions
    /// and applying appropriate interpolation when positions fall between data points.
    ///
    /// # Mathematical Details
    ///
    /// The quantile computation follows these steps:
    /// 1. **Validation**: Ensure 0 ≤ q ≤ 1 and data is non-empty
    /// 2. **Sorting**: Create sorted copy of data (O(n log n))
    /// 3. **Position Calculation**: position = q × (n - 1)
    /// 4. **Interpolation**: Apply selected method based on position
    ///
    /// # Method Selection Guide
    ///
    /// - **None (default)**: Uses Linear interpolation (R type 7 standard)
    /// - **Some(Linear)**: Explicit linear interpolation for continuous estimates
    /// - **Some(Lower)**: Conservative estimates, actual data values only
    /// - **Some(Higher)**: Pessimistic estimates, actual data values only
    /// - **Some(Nearest)**: Discrete approximation, rounds to nearest data point
    /// - **Some(Midpoint)**: Simple average of adjacent points
    ///
    /// # Edge Cases
    ///
    /// - **Empty data**: Panics (undefined mathematical operation)
    /// - **Single value**: Returns that value for any quantile
    /// - **q = 0.0**: Returns minimum value (all methods agree)
    /// - **q = 1.0**: Returns maximum value (all methods agree)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// let data = vec64![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
    ///
    /// // Default linear interpolation
    /// let q1 = data.quantile(0.25, None);  // First quartile
    /// 
    /// // Conservative estimate (always actual data point)
    /// let q1_lower = data.quantile(0.25, Some(QuantileMethod::Lower));
    /// 
    /// // Compare different methods for same quantile
    /// let median_linear = data.quantile(0.5, Some(QuantileMethod::Linear));
    /// let median_nearest = data.quantile(0.5, Some(QuantileMethod::Nearest));
    /// ```
    ///
    /// # Arguments
    /// * `q` - Quantile value between 0.0 and 1.0 (inclusive)
    /// * `method` - Interpolation method (None defaults to Linear)
    ///
    /// # Returns
    /// The computed quantile value using the specified method
    ///
    /// # Panics
    /// - Panics if the vector is empty (undefined operation)
    /// - Panics if q is not in the range [0.0, 1.0]
    /// - Panics if data contains NaN values (undefined ordering)
    fn quantile(&self, q: T, method: Option<QuantileMethod>) -> T;
    
    /// Compute the median (50th percentile, second quartile)
    ///
    /// The median is the middle value that separates the higher half from the lower
    /// half of a data sample. It is a robust measure of central tendency that is
    /// less sensitive to outliers than the arithmetic mean.
    ///
    /// # Mathematical Properties
    ///
    /// - **Robust Statistic**: 50% breakdown point (unaffected by up to 50% outliers)
    /// - **Location Parameter**: Minimizes sum of absolute deviations
    /// - **Quantile**: Exactly the 0.5 quantile or 50th percentile
    /// - **Ordinal Invariant**: Unchanged by monotonic transformations
    ///
    /// # Computation Details
    ///
    /// Uses linear interpolation (default quantile method) to handle even-length samples:
    /// - **Odd length n**: Median = X₍ₙ₊₁₎⁄₂₎ (middle value)
    /// - **Even length n**: Median = (Xₙ⁄₂ + Xₙ⁄₂₊₁) / 2 (average of two middle values)
    ///
    /// # Relationship to Mean
    ///
    /// - **Symmetric distributions**: Mean ≈ Median
    /// - **Right-skewed distributions**: Mean > Median  
    /// - **Left-skewed distributions**: Mean < Median
    /// - **Robust comparison**: Median less affected by extreme values
    ///
    /// # Applications
    ///
    /// - **Income analysis**: Typical income (less affected by billionaires)
    /// - **Performance metrics**: Typical response time (ignores outliers)
    /// - **Quality control**: Central value when distribution is unknown
    /// - **Non-parametric statistics**: Distribution-free central tendency
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// // Odd length: middle value
    /// let odd_data = vec64![1, 3, 5, 7, 9];
    /// assert_eq!(odd_data.median(), 5.0);
    ///
    /// // Even length: average of two middle values
    /// let even_data = vec64![1, 3, 5, 7];
    /// assert_eq!(even_data.median(), 4.0);  // (3 + 5) / 2
    ///
    /// // Robustness to outliers
    /// let with_outlier = vec64![1, 2, 3, 4, 1000];
    /// assert_eq!(with_outlier.median(), 3.0);  // Unaffected by 1000
    /// ```
    ///
    /// # Returns
    /// The median value of the dataset
    ///
    /// # Panics
    /// Panics if the vector is empty (median undefined for empty set)
    fn median(&self) -> T;
    
    /// Compute the specified percentile (0.0 to 100.0)
    ///
    /// Percentiles express quantiles as percentages, making them more intuitive for
    /// many applications. The p-th percentile is the value below which p percent
    /// of the data falls.
    ///
    /// # Mathematical Relationship
    ///
    /// ```text
    /// percentile(p) = quantile(p / 100.0)
    /// ```
    ///
    /// This function provides a convenient interface that automatically converts
    /// percentile notation to the underlying quantile calculation.
    ///
    /// # Common Percentiles
    ///
    /// - **P0 (0th percentile)**: Minimum value
    /// - **P25 (25th percentile)**: First quartile (Q1)
    /// - **P50 (50th percentile)**: Median (Q2)
    /// - **P75 (75th percentile)**: Third quartile (Q3)  
    /// - **P90 (90th percentile)**: 90% of data below this value
    /// - **P95, P99**: Common thresholds in performance analysis
    /// - **P100 (100th percentile)**: Maximum value
    ///
    /// # Applications by Domain
    ///
    /// ## Educational Testing
    /// - **Standardized scores**: "Scored better than 85% of test takers"
    /// - **Grade boundaries**: Top 10% receive A grades
    /// - **Performance comparison**: Relative standing in peer group
    ///
    /// ## Performance Analysis
    /// - **Response times**: P95 latency (95% of requests faster than this)
    /// - **Service level agreements**: P99 uptime guarantees
    /// - **Resource planning**: P90 capacity requirements
    ///
    /// ## Medical Statistics
    /// - **Growth charts**: Height/weight percentiles for children
    /// - **Reference ranges**: Normal values between P5 and P95
    /// - **Risk assessment**: Percentile rankings for biomarkers
    ///
    /// ## Financial Analysis
    /// - **Risk metrics**: Value at Risk (VaR) at P1 or P5 percentile
    /// - **Salary bands**: P25, P50, P75 compensation levels
    /// - **Portfolio analysis**: Performance relative to benchmark
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// let scores = vec64![65, 70, 75, 80, 82, 85, 88, 90, 92, 95];
    ///
    /// // Common percentiles
    /// let p25 = scores.percentile(25.0, None);   // First quartile
    /// let p50 = scores.percentile(50.0, None);   // Median
    /// let p75 = scores.percentile(75.0, None);   // Third quartile
    /// let p90 = scores.percentile(90.0, None);   // 90th percentile
    ///
    /// // Performance analysis
    /// let response_times = vec64![45, 52, 48, 61, 58, 47, 44, 49, 53, 150]; // ms
    /// let p95_latency = response_times.percentile(95.0, None);
    /// println!("95% of requests complete within {} ms", p95_latency);
    /// ```
    ///
    /// # Arguments
    /// * `p` - Percentile value between 0.0 and 100.0 (inclusive)
    /// * `method` - Interpolation method (None defaults to Linear)
    ///
    /// # Returns
    /// The computed percentile value using the specified method
    ///
    /// # Panics
    /// - Panics if the vector is empty (undefined operation)
    /// - Panics if p is not in the range [0.0, 100.0]
    /// - Panics if data contains NaN values (undefined ordering)
    fn percentile(&self, p: T, method: Option<QuantileMethod>) -> T;
    
    /// Compute quartiles (Q1, Q2, Q3) for comprehensive distribution summary
    ///
    /// Quartiles divide a ranked dataset into four equal parts, providing a robust
    /// summary of the data distribution. They form the basis for box plots and
    /// provide insights into distribution shape, central tendency, and spread.
    ///
    /// # Mathematical Definition
    ///
    /// - **Q1 (First Quartile)**: 25th percentile, separates bottom 25% from top 75%
    /// - **Q2 (Second Quartile)**: 50th percentile (median), separates lower and upper halves
    /// - **Q3 (Third Quartile)**: 75th percentile, separates bottom 75% from top 25%
    ///
    /// ```text
    /// Q1 = quantile(0.25)    ← 25% of data below this value
    /// Q2 = quantile(0.50)    ← 50% of data below this value (median)
    /// Q3 = quantile(0.75)    ← 75% of data below this value
    /// ```
    ///
    /// # Distribution Analysis
    ///
    /// Quartiles enable analysis of distribution characteristics:
    ///
    /// ## Symmetry Assessment
    /// - **Symmetric**: (Q2 - Q1) ≈ (Q3 - Q2)
    /// - **Right-skewed**: (Q3 - Q2) > (Q2 - Q1)
    /// - **Left-skewed**: (Q2 - Q1) > (Q3 - Q2)
    ///
    /// ## Spread Measures
    /// - **Interquartile Range (IQR)**: Q3 - Q1 (middle 50% spread)
    /// - **Semi-interquartile Range**: (Q3 - Q1) / 2
    /// - **Quartile Coefficient of Dispersion**: (Q3 - Q1) / (Q3 + Q1)
    ///
    /// # Box Plot Construction
    ///
    /// Quartiles are essential components of box plots:
    /// - **Box bounds**: Q1 and Q3 form the box edges
    /// - **Median line**: Q2 shown as line inside box
    /// - **Whiskers**: Extend to data within 1.5 × IQR from box edges
    /// - **Outliers**: Points beyond whiskers are potential outliers
    ///
    /// # Robust Statistics Properties
    ///
    /// - **Breakdown Point**: 25% (unaffected by up to 25% outliers in tails)
    /// - **Influence Function**: Bounded, less sensitive than mean-based measures
    /// - **Distribution-Free**: Valid for any continuous distribution
    ///
    /// # Applications
    ///
    /// ## Quality Control
    /// - **Process capability**: Check if process stays within quartile bounds
    /// - **Control charts**: Quartile-based control limits
    /// - **Defect analysis**: Identify unusual patterns in quartile shifts
    ///
    /// ## Financial Analysis
    /// - **Risk assessment**: VaR calculations using quartiles
    /// - **Performance attribution**: Quartile rankings of fund performance
    /// - **Salary analysis**: Compensation quartiles by role/location
    ///
    /// ## Medical Statistics
    /// - **Reference intervals**: Normal ranges between Q1 and Q3
    /// - **Growth percentiles**: Tracking development over time
    /// - **Biomarker analysis**: Quartile-based risk stratification
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// let exam_scores = vec64![65, 68, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92];
    /// let (q1, q2, q3) = exam_scores.quartiles();
    ///
    /// println!("Q1: {:.1} (25% scored below this)", q1);
    /// println!("Q2: {:.1} (median score)", q2);
    /// println!("Q3: {:.1} (75% scored below this)", q3);
    ///
    /// let iqr = q3 - q1;
    /// println!("IQR: {:.1} (middle 50% spread)", iqr);
    ///
    /// // Symmetry analysis
    /// let lower_spread = q2 - q1;
    /// let upper_spread = q3 - q2;
    /// if upper_spread > lower_spread * 1.2 {
    ///     println!("Distribution appears right-skewed");
    /// } else if lower_spread > upper_spread * 1.2 {
    ///     println!("Distribution appears left-skewed");
    /// } else {
    ///     println!("Distribution appears approximately symmetric");
    /// }
    /// ```
    ///
    /// # Returns
    /// Tuple (Q1, Q2, Q3) representing the 25th, 50th, and 75th percentiles
    ///
    /// # Panics
    /// Panics if the vector is empty (quartiles undefined for empty dataset)
    fn quartiles(&self) -> (T, T, T);
    
    /// Compute the interquartile range (IQR = Q3 - Q1)
    ///
    /// The IQR measures the spread of the middle 50% of the data, providing a robust
    /// alternative to the standard deviation that is less sensitive to outliers.
    /// It represents the range containing the central half of the distribution.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// IQR = Q3 - Q1 = 75th percentile - 25th percentile
    /// ```
    ///
    /// The IQR captures the variability of the "typical" data points while excluding
    /// the most extreme 25% from each tail of the distribution.
    ///
    /// # Statistical Properties
    ///
    /// ## Robustness
    /// - **Breakdown Point**: 25% (unaffected by up to 25% outliers in either tail)
    /// - **Outlier Resistance**: Much less sensitive than standard deviation
    /// - **Distribution-Free**: Meaningful for any distribution shape
    ///
    /// ## Scale Properties
    /// - **Linear Transformation**: IQR(aX + b) = |a| × IQR(X)
    /// - **Units**: Same units as the original data
    /// - **Non-negative**: Always ≥ 0, equals 0 only when Q1 = Q3
    ///
    /// # Comparison with Standard Deviation
    ///
    /// | Property | IQR | Standard Deviation |
    /// |----------|-----|--------------------|
    /// | Outlier sensitivity | Low | High |
    /// | Breakdown point | 25% | 0% |
    /// | Distribution assumption | None | Works best for normal |
    /// | Computational cost | O(n log n) | O(n) |
    /// | Interpretation | Middle 50% spread | Average squared deviation |
    ///
    /// For normal distributions: IQR ≈ 1.35 × σ (standard deviation)
    ///
    /// # Applications
    ///
    /// ## Outlier Detection
    /// **1.5 × IQR Rule**: Values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR are potential outliers
    /// ```rust
    /// let lower_fence = q1 - 1.5 * iqr;
    /// let upper_fence = q3 + 1.5 * iqr;
    /// ```
    ///
    /// ## Data Quality Assessment
    /// - **Consistency checks**: Compare IQR across time periods
    /// - **Process monitoring**: Track IQR as measure of process stability
    /// - **Data validation**: Detect periods of unusual variability
    ///
    /// ## Robust Statistics
    /// - **Scale estimation**: IQR/1.35 approximates σ for normal data
    /// - **Standardization**: (X - median) / IQR for robust z-scores
    /// - **Confidence intervals**: Bootstrap confidence intervals using IQR
    ///
    /// ## Performance Analysis
    /// - **Response time analysis**: IQR of latencies (ignoring extreme outliers)
    /// - **Load testing**: Measure typical performance variability
    /// - **Service quality**: Assess consistency of service delivery
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// // Response times in milliseconds
    /// let response_times = vec64![45, 47, 48, 50, 52, 53, 55, 58, 62, 180]; // One outlier
    /// 
    /// let iqr = response_times.iqr();
    /// let std_dev = response_times.std(None);
    /// 
    /// println!("IQR: {:.1} ms (robust spread measure)", iqr);
    /// println!("Std Dev: {:.1} ms (affected by outlier)", std_dev);
    /// 
    /// // Outlier detection using IQR rule
    /// let (q1, _, q3) = response_times.quartiles();
    /// let lower_fence = q1 - 1.5 * iqr;
    /// let upper_fence = q3 + 1.5 * iqr;
    /// 
    /// for &time in response_times.as_slice_unchecked() {
    ///     if time < lower_fence || time > upper_fence {
    ///         println!("Potential outlier: {} ms", time);
    ///     }
    /// }
    /// 
    /// // Robust coefficient of variation
    /// let median = response_times.median();
    /// let robust_cv = iqr / median;
    /// println!("Robust CV: {:.3} (relative variability)", robust_cv);
    /// ```
    ///
    /// # Returns
    /// The interquartile range (Q3 - Q1) as a measure of spread
    ///
    /// # Panics
    /// Panics if the vector is empty (IQR undefined for empty dataset)
    fn iqr(&self) -> T;
}

// Helper function to compute quantile with specified method
/// 
/// Core quantile computation logic that handles all interpolation methods.
/// Assumes data is already sorted in ascending order.
///
/// # Implementation Details
///
/// This function implements the mathematical algorithms for each quantile method:
/// - Calculates the theoretical position: q × (n - 1)
/// - Determines adjacent data points when position falls between indices
/// - Applies appropriate interpolation or selection rule
///
/// # Performance
/// 
/// - **Time**: O(1) after sorting (constant time index calculations)
/// - **Space**: O(1) (in-place computation on sorted data)
/// - **Numerical Stability**: Uses stable floating-point arithmetic
///
/// # Arguments
/// * `sorted_data` - Slice of data sorted in ascending order
/// * `q` - Quantile value in [0.0, 1.0]
/// * `method` - Interpolation method to use
///
/// # Returns
/// Computed quantile value
fn compute_quantile_f64(sorted_data: &[f64], q: f64, method: QuantileMethod) -> f64 {
    let n = sorted_data.len();
    
    match method {
        QuantileMethod::Linear => {
            let index = q * (n - 1) as f64;
            let lower_index = index.floor() as usize;
            let upper_index = index.ceil() as usize;
            
            if lower_index == upper_index {
                sorted_data[lower_index]
            } else {
                let fraction = index - lower_index as f64;
                sorted_data[lower_index] * (1.0 - fraction) + sorted_data[upper_index] * fraction
            }
        },
        QuantileMethod::Lower => {
            let index = (q * (n - 1) as f64).floor() as usize;
            sorted_data[index]
        },
        QuantileMethod::Higher => {
            let index = (q * (n - 1) as f64).ceil() as usize;
            sorted_data[index]
        },
        QuantileMethod::Midpoint => {
            let lower_index = (q * (n - 1) as f64).floor() as usize;
            let upper_index = (q * (n - 1) as f64).ceil() as usize;
            (sorted_data[lower_index] + sorted_data[upper_index]) / 2.0
        },
        QuantileMethod::Nearest => {
            let index = (q * (n - 1) as f64).round() as usize;
            sorted_data[index]
        },
    }
}

/// Core quantile computation for f32 data with specified interpolation method.
/// 
/// This is the f32 specialization of the quantile computation algorithm, providing
/// identical mathematical behavior to the f64 version while maintaining appropriate
/// precision for single-precision floating-point data.
///
/// # Precision Considerations
///
/// - Uses f32 arithmetic throughout to maintain consistency
/// - Handles edge cases with same logic as f64 version  
/// - May have slightly different numerical behavior due to precision differences
///
/// # Arguments
/// * `sorted_data` - Slice of f32 data sorted in ascending order
/// * `q` - Quantile value in [0.0, 1.0]
/// * `method` - Interpolation method to use
///
/// # Returns
/// Computed quantile value as f32
fn compute_quantile_f32(sorted_data: &[f32], q: f32, method: QuantileMethod) -> f32 {
    let n = sorted_data.len();
    
    match method {
        QuantileMethod::Linear => {
            let index = q * (n - 1) as f32;
            let lower_index = index.floor() as usize;
            let upper_index = index.ceil() as usize;
            
            if lower_index == upper_index {
                sorted_data[lower_index]
            } else {
                let fraction = index - lower_index as f32;
                sorted_data[lower_index] * (1.0 - fraction) + sorted_data[upper_index] * fraction
            }
        },
        QuantileMethod::Lower => {
            let index = (q * (n - 1) as f32).floor() as usize;
            sorted_data[index]
        },
        QuantileMethod::Higher => {
            let index = (q * (n - 1) as f32).ceil() as usize;
            sorted_data[index]
        },
        QuantileMethod::Midpoint => {
            let lower_index = (q * (n - 1) as f32).floor() as usize;
            let upper_index = (q * (n - 1) as f32).ceil() as usize;
            (sorted_data[lower_index] + sorted_data[upper_index]) / 2.0
        },
        QuantileMethod::Nearest => {
            let index = (q * (n - 1) as f32).round() as usize;
            sorted_data[index]
        },
    }
}

impl Quantiles<f64> for VectorF64 {
    fn quantile(&self, q: f64, method: Option<QuantileMethod>) -> f64 {
        assert!(!self.is_empty(), "Cannot compute quantile of empty vector");
        assert!(q >= 0.0 && q <= 1.0, "Quantile must be between 0.0 and 1.0, got {}", q);
        
        let method = method.unwrap_or_default();
        let mut sorted_data = self.as_slice_unchecked().to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        compute_quantile_f64(&sorted_data, q, method)
    }
    
    fn median(&self) -> f64 {
        self.quantile(0.5, None)
    }
    
    fn percentile(&self, p: f64, method: Option<QuantileMethod>) -> f64 {
        assert!(p >= 0.0 && p <= 100.0, "Percentile must be between 0.0 and 100.0, got {}", p);
        self.quantile(p / 100.0, method)
    }
    
    fn quartiles(&self) -> (f64, f64, f64) {
        let q1 = self.quantile(0.25, None);
        let q2 = self.quantile(0.5, None);
        let q3 = self.quantile(0.75, None);
        (q1, q2, q3)
    }
    
    fn iqr(&self) -> f64 {
        let (q1, _, q3) = self.quartiles();
        q3 - q1
    }
}

impl Quantiles<f32> for VectorF32 {
    fn quantile(&self, q: f32, method: Option<QuantileMethod>) -> f32 {
        assert!(!self.is_empty(), "Cannot compute quantile of empty vector");
        assert!(q >= 0.0 && q <= 1.0, "Quantile must be between 0.0 and 1.0, got {}", q);
        
        let method = method.unwrap_or_default();
        let mut sorted_data = self.as_slice_unchecked().to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        compute_quantile_f32(&sorted_data, q, method)
    }
    
    fn median(&self) -> f32 {
        self.quantile(0.5, None)
    }
    
    fn percentile(&self, p: f32, method: Option<QuantileMethod>) -> f32 {
        assert!(p >= 0.0 && p <= 100.0, "Percentile must be between 0.0 and 100.0, got {}", p);
        self.quantile(p / 100.0, method)
    }
    
    fn quartiles(&self) -> (f32, f32, f32) {
        let q1 = self.quantile(0.25, None);
        let q2 = self.quantile(0.5, None);
        let q3 = self.quantile(0.75, None);
        (q1, q2, q3)
    }
    
    fn iqr(&self) -> f32 {
        let (q1, _, q3) = self.quartiles();
        q3 - q1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::vec64;
    
    #[test]
    fn test_median_odd_length() {
        let v = vec64![1, 2, 3, 4, 5];
        assert_eq!(v.median(), 3.0);
    }
    
    #[test]
    fn test_median_even_length() {
        let v = vec64![1, 2, 3, 4];
        assert_eq!(v.median(), 2.5);
    }
    
    #[test]
    fn test_quartiles() {
        let v = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (q1, q2, q3) = v.quartiles();
        assert_eq!(q2, 5.0); // median
        assert!(q1 < q2 && q2 < q3);
    }
    
    #[test]
    fn test_iqr() {
        let v = vec64![1, 2, 3, 4, 5];
        let iqr = v.iqr();
        assert!(iqr > 0.0);
    }
    
    #[test]
    fn test_percentile() {
        let v = vec64![1, 2, 3, 4, 5];
        assert_eq!(v.percentile(50.0, None), 3.0);
        assert_eq!(v.percentile(0.0, None), 1.0);
        assert_eq!(v.percentile(100.0, None), 5.0);
    }
    
    #[test]
    #[should_panic(expected = "Cannot compute quantile of empty vector")]
    fn test_median_empty() {
        let v = VectorF64::from_slice(&[]);
        v.median();
    }
    
    #[test]
    #[should_panic(expected = "Quantile must be between 0.0 and 1.0")]
    fn test_invalid_quantile() {
        let v = vec64![1, 2, 3];
        v.quantile(1.5, None);
    }
    
    #[test]
    #[should_panic(expected = "Percentile must be between 0.0 and 100.0")]
    fn test_invalid_percentile() {
        let v = vec64![1, 2, 3];
        v.percentile(150.0, None);
    }
}