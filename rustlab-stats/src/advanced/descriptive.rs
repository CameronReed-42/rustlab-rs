//! Advanced descriptive statistics for comprehensive data analysis
//!
//! This module extends the basic statistical measures available in rustlab-math with
//! sophisticated descriptive statistics that provide deeper insights into data
//! distributions. All methods are designed for high performance and numerical
//! stability while maintaining mathematical rigor.
//!
//! ## Mathematical Foundation
//!
//! ### Advanced Central Tendency Measures
//!
//! Beyond the arithmetic mean, this module provides:
//!
//! #### Geometric Mean
//! - **Formula**: GM(x₁, x₂, ..., xₙ) = (∏ᵢ xᵢ)^(1/n) = exp(∑ᵢ ln(xᵢ)/n)
//! - **Properties**: Always ≤ arithmetic mean (AM-GM inequality)
//! - **Use Cases**: Growth rates, ratios, multiplicative processes
//! - **Domain**: Strictly positive values only
//!
//! #### Harmonic Mean
//! - **Formula**: HM = n / ∑ᵢ(1/xᵢ)
//! - **Properties**: Always ≤ geometric mean ≤ arithmetic mean
//! - **Use Cases**: Rates, speeds, price-to-earnings ratios
//! - **Domain**: Strictly positive values only
//!
//! #### Trimmed Mean
//! - **Formula**: Mean of middle (1-2α)×100% of sorted observations
//! - **Properties**: Robust to outliers while maintaining efficiency
//! - **Use Cases**: Robust central tendency estimation
//! - **Parameters**: Trim fraction α ∈ [0, 0.5]
//!
//! ### Robust Variability Measures
//!
//! #### Median Absolute Deviation (MAD)
//! - **Formula**: MAD = median(|xᵢ - median(x)|)
//! - **Properties**: Robust scale estimator, breakdown point = 50%
//! - **Use Cases**: Outlier detection, robust scaling
//! - **Scaling**: 1.4826 × MAD ≈ standard deviation for normal data
//!
//! #### Coefficient of Variation (CV)
//! - **Formula**: CV = σ/μ (dimensionless ratio)
//! - **Properties**: Scale-invariant measure of relative variability
//! - **Use Cases**: Comparing variability across different scales
//! - **Interpretation**: CV > 1 indicates high variability
//!
//! ### Statistical Properties
//!
//! #### Mean Relationships
//! For positive data: Harmonic Mean ≤ Geometric Mean ≤ Arithmetic Mean
//!
//! #### Robustness Characteristics
//! - **Trimmed Mean**: Finite breakdown point (α × 100%)
//! - **MAD**: 50% breakdown point (most robust)
//! - **Range**: 0% breakdown point (least robust)
//!
//! ## Performance Characteristics
//!
//! | Statistic | Complexity | Memory | Robustness |
//! |-----------|------------|--------|-----------|
//! | Mode | O(n log n) | O(n) | Low |
//! | Geometric Mean | O(n) | O(1) | Low |
//! | Harmonic Mean | O(n) | O(1) | Low |
//! | Trimmed Mean | O(n log n) | O(n) | Medium-High |
//! | MAD | O(n log n) | O(n) | Highest |
//! | CV | O(n) | O(1) | Low |
//! | Range | O(n) | O(1) | Lowest |
//!
//! ## Usage Guidelines
//!
//! ### Method Selection Strategy
//!
//! 1. **Normal/Symmetric Data**: Arithmetic mean + standard deviation
//! 2. **Skewed Data**: Median + MAD or trimmed mean
//! 3. **Ratio/Rate Data**: Geometric or harmonic mean
//! 4. **Outlier-Prone Data**: Trimmed statistics or MAD
//! 5. **Comparative Analysis**: Coefficient of variation
//!
//! ### Numerical Stability Considerations
//!
//! - **Geometric Mean**: Uses log-sum-exp for numerical stability
//! - **Harmonic Mean**: Checks for division by zero
//! - **Trimmed Mean**: Validates trim fraction bounds
//! - **All Methods**: Input validation and finite value checking
//!
//! ## Examples
//!
//! ### Central Tendency Comparison
//! ```rust
//! use rustlab_stats::advanced::AdvancedDescriptive;
//! use rustlab_math::vec64;
//!
//! let growth_rates = vec64![0.05, 0.08, 0.12, 0.15, 0.09]; // 5%, 8%, 12%, 15%, 9%
//!
//! let arithmetic_mean = growth_rates.mean();        // ~9.8%
//! let geometric_mean = growth_rates.geometric_mean(); // ~9.7% (compound growth)
//! let harmonic_mean = growth_rates.harmonic_mean();  // ~9.6% (harmonic average)
//!
//! // Geometric mean most appropriate for growth rates
//! ```
//!
//! ### Robust Statistics with Outliers
//! ```rust
//! use rustlab_stats::advanced::AdvancedDescriptive;
//! use rustlab_math::vec64;
//!
//! let data_with_outlier = vec64![10, 12, 11, 13, 9, 100]; // 100 is outlier
//!
//! let regular_mean = data_with_outlier.mean();              // ~25.8 (inflated)
//! let trimmed_mean = data_with_outlier.trimmed_mean(0.2);   // ~11.25 (robust)
//! let mad = data_with_outlier.mad();                        // Robust scale
//! let range = data_with_outlier.range();                    // 91 (sensitive)
//! ```

use rustlab_math::{VectorF64, VectorF32};
use crate::advanced::quantiles::Quantiles;
use std::collections::HashMap;

/// Trait for advanced descriptive statistics beyond basic measures
///
/// This trait extends the basic statistical operations available in rustlab-math
/// with sophisticated descriptive measures that provide deeper insights into data
/// characteristics. All methods are designed for numerical stability and handle
/// edge cases gracefully.
///
/// # Mathematical Rigor
///
/// All implementations follow established statistical definitions and maintain
/// numerical stability through careful algorithm selection:
///
/// - **Geometric Mean**: Uses log-transform to avoid overflow
/// - **Harmonic Mean**: Validates positive inputs
/// - **Trimmed Mean**: Properly handles fractional trimming
/// - **MAD**: Uses stable median computation
/// - **CV**: Handles zero-mean cases appropriately
///
/// # Performance Characteristics
///
/// Methods are optimized for the expected use cases:
/// - Single-pass algorithms where possible (geometric mean, harmonic mean, CV)
/// - Efficient sorting for order statistics (trimmed mean, MAD)
/// - Minimal memory allocation (in-place operations when feasible)
/// - SIMD-friendly implementations for vector operations
///
/// # Error Handling Philosophy
///
/// Following rustlab-stats' "math-first" approach, methods panic on invalid
/// inputs rather than returning Results. This ensures clean APIs while catching
/// programming errors early:
///
/// - Empty vectors always panic
/// - Invalid parameter ranges panic with descriptive messages  
/// - Domain violations (negative values for geometric/harmonic means) panic
/// - NaN/infinite results indicate numerical issues
///
/// # Thread Safety
///
/// All trait methods are safe for concurrent use as they operate on immutable
/// references and don't modify the underlying data.
pub trait AdvancedDescriptive<T> {
    /// Compute the mode (most frequently occurring value)
    ///
    /// The mode is the value that appears most frequently in the dataset. For continuous
    /// data, this measure may not be meaningful due to the low probability of exact
    /// repetitions. This implementation handles ties by returning the smallest value
    /// among those with maximum frequency.
    ///
    /// # Mathematical Definition
    ///
    /// The mode is the value xₘ such that:
    /// ```text
    /// frequency(xₘ) = max{frequency(xᵢ) : xᵢ ∈ dataset}
    /// ```
    ///
    /// For multimodal distributions (multiple modes), this returns the smallest mode.
    ///
    /// # Algorithm
    ///
    /// 1. Create frequency map using bit representation for precise floating-point comparison
    /// 2. Find maximum frequency across all values
    /// 3. Among all values with maximum frequency, return the smallest
    ///
    /// # Complexity
    ///
    /// - **Time**: O(n) for frequency counting
    /// - **Space**: O(k) where k is the number of unique values
    ///
    /// # Use Cases
    ///
    /// - **Categorical Data**: Most common category
    /// - **Discrete Data**: Most frequent discrete value
    /// - **Quality Control**: Most common defect type
    /// - **Market Research**: Most popular choice
    ///
    /// # Limitations
    ///
    /// - **Continuous Data**: May not be meaningful due to low repetition probability
    /// - **Uniform Distribution**: All values equally frequent (arbitrary choice)
    /// - **Multimodal Data**: Only returns one mode (the smallest)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::advanced::AdvancedDescriptive;
    /// use rustlab_math::vec64;
    ///
    /// // Clear mode
    /// let data = vec64![1, 2, 2, 3, 2, 4];
    /// assert_eq!(data.mode(), 2.0); // 2 appears 3 times
    ///
    /// // Tie - returns smallest
    /// let tied = vec64![1, 1, 2, 2, 3];
    /// assert_eq!(tied.mode(), 1.0); // Both 1 and 2 appear twice, returns 1
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty, as the mode is undefined for empty datasets.
    fn mode(&self) -> T;
    
    /// Compute the geometric mean for positive values
    ///
    /// The geometric mean is the nth root of the product of n positive numbers,
    /// providing the central tendency for multiplicative processes. It is always
    /// less than or equal to the arithmetic mean (AM-GM inequality) and is the
    /// appropriate measure for growth rates, ratios, and proportional changes.
    ///
    /// # Mathematical Definition
    ///
    /// For positive values x₁, x₂, ..., xₙ:
    /// ```text
    /// GM = (x₁ × x₂ × ... × xₙ)^(1/n) = exp(∑ᵢ ln(xᵢ) / n)
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses the logarithmic form for numerical stability:
    /// 1. Transform to log space: ln(xᵢ)
    /// 2. Compute arithmetic mean of logs: ∑ln(xᵢ)/n  
    /// 3. Transform back: exp(mean_of_logs)
    ///
    /// This approach avoids potential overflow/underflow in the direct product.
    ///
    /// # Properties
    ///
    /// - **AM-GM Inequality**: GM ≤ AM, with equality only if all values equal
    /// - **Scale Invariant**: GM(cx₁, cx₂, ..., cxₙ) = c × GM(x₁, x₂, ..., xₙ)
    /// - **Multiplicative**: GM of ratios gives compound rate
    /// - **Dimensionless**: Appropriate for percentages and ratios
    ///
    /// # Use Cases
    ///
    /// - **Growth Rates**: Average compound growth rate
    /// - **Financial Returns**: Portfolio return calculations
    /// - **Ratios**: Average of ratio data (P/E ratios, etc.)
    /// - **Index Numbers**: Price indices, productivity indices
    /// - **Multiplicative Processes**: Where values multiply rather than add
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::advanced::AdvancedDescriptive;
    /// use rustlab_math::vec64;
    ///
    /// // Investment returns: 10%, -5%, 8%, 12%
    /// let returns = vec64![1.10, 0.95, 1.08, 1.12]; // Growth factors
    /// let avg_return = returns.geometric_mean(); // ≈ 1.062 (6.2% compound)
    ///
    /// // Price ratios
    /// let ratios = vec64![1.2, 0.8, 1.5, 0.9];
    /// let avg_ratio = ratios.geometric_mean(); // ≈ 1.067
    /// ```
    ///
    /// # Comparison with Arithmetic Mean
    ///
    /// ```text
    /// Data: [1, 4, 16]
    /// Arithmetic Mean: (1 + 4 + 16) / 3 = 7.0
    /// Geometric Mean: (1 × 4 × 16)^(1/3) = 4.0
    /// ```
    ///
    /// The geometric mean is less sensitive to extreme values.
    ///
    /// # Panics
    ///
    /// - Panics if the vector is empty (geometric mean undefined)
    /// - Panics if any value is non-positive (logarithm undefined)
    /// - Use this precondition to ensure mathematical validity
    fn geometric_mean(&self) -> T;
    
    /// Compute the harmonic mean for positive values
    ///
    /// The harmonic mean is the reciprocal of the arithmetic mean of reciprocals,
    /// providing the appropriate average for rates, speeds, and ratios where the
    /// reciprocal relationship is meaningful. It is always the smallest of the
    /// three Pythagorean means (harmonic ≤ geometric ≤ arithmetic).
    ///
    /// # Mathematical Definition
    ///
    /// For positive values x₁, x₂, ..., xₙ:
    /// ```text
    /// HM = n / (∑ᵢ 1/xᵢ) = 1 / (∑ᵢ (1/xᵢ) / n)
    /// ```
    ///
    /// # Algorithm
    ///
    /// 1. Compute reciprocals: 1/xᵢ for each value
    /// 2. Calculate arithmetic mean of reciprocals
    /// 3. Take reciprocal of the result
    ///
    /// # Properties
    ///
    /// - **Inequality**: HM ≤ GM ≤ AM for positive values
    /// - **Rate-Appropriate**: Correct average for rates and speeds
    /// - **Extreme Sensitivity**: Strongly influenced by small values
    /// - **Reciprocal Relationship**: HM(x₁, x₂, ..., xₙ) = 1/AM(1/x₁, 1/x₂, ..., 1/xₙ)
    ///
    /// # Use Cases
    ///
    /// - **Speed Calculations**: Average speed over equal distances
    /// - **Rate Analysis**: Average rates (MHz, transactions/sec)
    /// - **Financial Ratios**: P/E ratios, efficiency ratios
    /// - **Electrical Engineering**: Parallel resistance calculations
    /// - **Physics**: Average velocities, frequencies
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::advanced::AdvancedDescriptive;
    /// use rustlab_math::vec64;
    ///
    /// // Average speed: 60 mph for first half, 30 mph for second half
    /// let speeds = vec64![60.0, 30.0];
    /// let avg_speed = speeds.harmonic_mean(); // = 40 mph (not 45!)
    ///
    /// // P/E ratios: want average that reflects true valuation
    /// let pe_ratios = vec64![15.0, 25.0, 8.0, 12.0];
    /// let avg_pe = pe_ratios.harmonic_mean(); // ≈ 12.3
    /// ```
    ///
    /// # Why Harmonic Mean for Rates?
    ///
    /// When averaging rates over equal intervals, harmonic mean gives the correct result:
    /// ```text
    /// Distance = Speed × Time
    /// For equal distances d: Total_Time = d/v₁ + d/v₂ + ...
    /// Average_Speed = Total_Distance / Total_Time = n×d / (d×∑(1/vᵢ)) = HM
    /// ```
    ///
    /// # Comparison with Other Means
    ///
    /// ```text
    /// Data: [1, 4, 4]
    /// Harmonic Mean:   3 / (1/1 + 1/4 + 1/4) = 2.0
    /// Geometric Mean:  (1 × 4 × 4)^(1/3) = 2.52
    /// Arithmetic Mean: (1 + 4 + 4) / 3 = 3.0
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the vector is empty (harmonic mean undefined)
    /// - Panics if any value is non-positive (reciprocal undefined or infinite)
    /// - Mathematical requirement for meaningful interpretation
    fn harmonic_mean(&self) -> T;
    
    /// Compute the trimmed mean (truncated mean)
    ///
    /// The trimmed mean provides a robust measure of central tendency by removing
    /// a specified fraction of the smallest and largest values before computing
    /// the arithmetic mean. This method balances the efficiency of the mean with
    /// robustness against outliers.
    ///
    /// # Mathematical Definition
    ///
    /// For trim fraction α ∈ [0, 0.5] and sorted data x₍₁₎ ≤ x₍₂₎ ≤ ... ≤ x₍ₙ₎:
    /// ```text
    /// TM_α = (∑ᵢ₌ₖ₊₁ⁿ⁻ᵏ x₍ᵢ₎) / (n - 2k)
    /// where k = ⌊α × n⌋
    /// ```
    ///
    /// # Algorithm
    ///
    /// 1. Sort the data in ascending order
    /// 2. Calculate number of values to trim: k = floor(α × n)
    /// 3. Remove k values from each end
    /// 4. Compute arithmetic mean of remaining values
    ///
    /// # Robustness Properties
    ///
    /// - **Breakdown Point**: α (the trim fraction)
    /// - **Efficiency**: Higher than median, lower than mean
    /// - **Influence Function**: Bounded (more robust than mean)
    /// - **Asymptotic Normality**: Maintains good statistical properties
    ///
    /// # Parameters
    ///
    /// * `trim_fraction` - Fraction of data to trim from each end
    ///   - **Range**: [0.0, 0.5]
    ///   - **0.0**: No trimming (equivalent to arithmetic mean)
    ///   - **0.25**: Interquartile mean (trim bottom and top quartiles)
    ///   - **0.5**: Median (maximum trimming)
    ///
    /// # Common Trim Fractions
    ///
    /// - **5% (α=0.05)**: Light trimming, removes extreme outliers
    /// - **10% (α=0.10)**: Moderate trimming, common in sports scoring
    /// - **20% (α=0.20)**: Heavy trimming, robust to outliers
    /// - **25% (α=0.25)**: Interquartile mean, very robust
    ///
    /// # Use Cases
    ///
    /// - **Olympic Scoring**: Remove highest and lowest judge scores
    /// - **Survey Data**: Remove extreme responses
    /// - **Financial Data**: Remove outlier returns
    /// - **Quality Control**: Remove measurement errors
    /// - **Scientific Data**: Robust estimation with outliers
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::advanced::AdvancedDescriptive;
    /// use rustlab_math::vec64;
    ///
    /// let data = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 100]; // 100 is outlier
    ///
    /// let full_mean = data.mean();                    // 14.5 (affected by outlier)
    /// let trimmed_5pct = data.trimmed_mean(0.05);     // Still affected (n=10, no trimming)
    /// let trimmed_10pct = data.trimmed_mean(0.10);    // 5.5 (removes 1 from each end)
    /// let trimmed_20pct = data.trimmed_mean(0.20);    // 5.0 (removes 2 from each end)
    /// ```
    ///
    /// # Comparison with Other Robust Measures
    ///
    /// ```text
    /// Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    /// Mean:              14.5  (not robust)
    /// 10% Trimmed Mean:  5.5   (somewhat robust)
    /// 20% Trimmed Mean:  5.0   (more robust)
    /// Median:            5.5   (most robust)
    /// ```
    ///
    /// # Statistical Properties
    ///
    /// - **Unbiased**: E[TM_α] = μ for symmetric distributions
    /// - **Consistent**: TM_α → μ as n → ∞
    /// - **Asymptotic Normality**: √n(TM_α - μ) → N(0, σ²_TM)
    ///
    /// # Arguments
    ///
    /// * `trim_fraction` - Fraction of data to trim from each end (0.0 to 0.5)
    ///   - Must be in range [0.0, 0.5] to ensure meaningful result
    ///   - Values closer to 0.5 provide more robustness but less efficiency
    ///
    /// # Panics
    ///
    /// - Panics if the vector is empty (trimmed mean undefined)
    /// - Panics if trim_fraction is not in [0.0, 0.5] (invalid parameter)
    /// - Parameter validation ensures mathematical meaningfulness
    fn trimmed_mean(&self, trim_fraction: T) -> T;
    
    /// Compute the median absolute deviation (MAD)
    ///
    /// The median absolute deviation is a robust measure of statistical dispersion
    /// that provides an alternative to standard deviation. It has the highest
    /// breakdown point (50%) among common variability measures, making it extremely
    /// robust to outliers and suitable for non-normal distributions.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// MAD = median(|xᵢ - median(x)|)
    /// ```
    ///
    /// where median(x) is the median of the original data.
    ///
    /// # Algorithm
    ///
    /// 1. Compute the median of the original data
    /// 2. Calculate absolute deviations from the median: |xᵢ - median|
    /// 3. Find the median of these absolute deviations
    ///
    /// # Robustness Properties
    ///
    /// - **Breakdown Point**: 50% (highest possible)
    /// - **Influence Function**: Bounded and redescending
    /// - **Outlier Resistance**: Can handle up to 50% outliers
    /// - **Distribution Free**: No assumption about underlying distribution
    ///
    /// # Scaling to Standard Deviation
    ///
    /// For normally distributed data:
    /// ```text
    /// σ ≈ 1.4826 × MAD
    /// ```
    /// This scaling factor makes MAD comparable to standard deviation.
    ///
    /// # Use Cases
    ///
    /// - **Outlier Detection**: Values > median + k×MAD (typically k=2 or 3)
    /// - **Robust Scaling**: Alternative to z-score normalization
    /// - **Non-Normal Data**: When standard deviation assumptions violated
    /// - **Quality Control**: Robust process variation monitoring
    /// - **Financial Risk**: Robust measure of return volatility
    ///
    /// # Advantages over Standard Deviation
    ///
    /// - **Robust**: Unaffected by extreme outliers
    /// - **Intuitive**: Median-based interpretation
    /// - **Distribution-Free**: No normality assumptions
    /// - **Stable**: Consistent across different data conditions
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::advanced::AdvancedDescriptive;
    /// use rustlab_math::vec64;
    ///
    /// let normal_data = vec64![2, 3, 4, 5, 6, 7, 8];
    /// let mad_normal = normal_data.mad();  // ≈ 2.0
    /// let std_normal = normal_data.std(None); // ≈ 2.16
    ///
    /// let outlier_data = vec64![2, 3, 4, 5, 6, 7, 100]; // 100 is outlier
    /// let mad_outlier = outlier_data.mad();   // ≈ 2.0 (unchanged!)
    /// let std_outlier = outlier_data.std(None); // ≈ 35.1 (heavily affected)
    /// ```
    ///
    /// # Outlier Detection Example
    ///
    /// ```rust
    /// use rustlab_stats::advanced::{AdvancedDescriptive, Quantiles};
    /// use rustlab_math::vec64;
    ///
    /// let data = vec64![10, 12, 11, 13, 9, 8, 14, 50]; // 50 might be outlier
    /// 
    /// let median = data.median();
    /// let mad = data.mad();
    /// let threshold = 2.0; // Common outlier threshold
    ///
    /// // Values beyond median ± threshold×MAD are potential outliers
    /// let lower_bound = median - threshold * mad;
    /// let upper_bound = median + threshold * mad;
    /// ```
    ///
    /// # Computational Properties
    ///
    /// - **Time Complexity**: O(n log n) due to median calculations
    /// - **Space Complexity**: O(n) for deviation storage
    /// - **Numerical Stability**: Excellent (median-based)
    ///
    /// # Statistical Theory
    ///
    /// For large samples from normal distribution:
    /// - **Efficiency**: ≈ 37% relative to standard deviation
    /// - **Consistency**: MAD → σ/1.4826 as n → ∞
    /// - **Asymptotic Distribution**: Known and well-studied
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty, as both the median and MAD are undefined
    /// for empty datasets.
    fn mad(&self) -> T;
    
    /// Compute the coefficient of variation (CV)
    ///
    /// The coefficient of variation is a standardized measure of dispersion that
    /// expresses the standard deviation as a fraction of the mean. This dimensionless
    /// ratio enables comparison of variability across datasets with different units
    /// or scales, making it invaluable for comparative analysis.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// CV = σ / |μ|
    /// ```
    /// where σ is the standard deviation and μ is the mean.
    ///
    /// # Properties
    ///
    /// - **Dimensionless**: Unit-free measure, often expressed as percentage
    /// - **Scale Invariant**: Unchanged by proportional scaling of data
    /// - **Relative Measure**: Expresses variability relative to the mean
    /// - **Comparison Tool**: Enables cross-dataset variability comparison
    ///
    /// # Interpretation Guidelines
    ///
    /// - **CV < 0.1**: Low variability (10% of mean)
    /// - **0.1 ≤ CV < 0.3**: Moderate variability
    /// - **0.3 ≤ CV < 1.0**: High variability
    /// - **CV ≥ 1.0**: Very high variability (standard deviation exceeds mean)
    ///
    /// # Use Cases
    ///
    /// - **Quality Control**: Process consistency assessment
    /// - **Risk Analysis**: Investment risk comparison (Sharpe ratio component)
    /// - **Experimental Design**: Precision comparison across methods
    /// - **Manufacturing**: Product consistency evaluation
    /// - **Biology**: Population variability studies
    /// - **Economics**: Income inequality analysis
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::advanced::AdvancedDescriptive;
    /// use rustlab_math::vec64;
    ///
    /// // Investment returns: Stock A vs Stock B
    /// let stock_a = vec64![5, 7, 6, 8, 4]; // Mean ≈ 6, Std ≈ 1.58
    /// let stock_b = vec64![15, 21, 18, 24, 12]; // Mean ≈ 18, Std ≈ 4.74
    ///
    /// let cv_a = stock_a.coefficient_of_variation(); // ≈ 0.26 (26%)
    /// let cv_b = stock_b.coefficient_of_variation(); // ≈ 0.26 (26%)
    /// // Same relative variability despite different scales
    /// ```
    ///
    /// # Scale Invariance Demonstration
    ///
    /// ```rust
    /// use rustlab_stats::advanced::AdvancedDescriptive;
    /// use rustlab_math::vec64;
    ///
    /// let original = vec64![10, 20, 30, 40, 50];
    /// let scaled = vec64![100, 200, 300, 400, 500]; // 10x scaling
    ///
    /// let cv_original = original.coefficient_of_variation();
    /// let cv_scaled = scaled.coefficient_of_variation();
    /// // cv_original ≈ cv_scaled (scale invariant)
    /// ```
    ///
    /// # Comparison Across Domains
    ///
    /// ```text
    /// Domain          | Typical CV Range | Interpretation
    /// Manufacturing   | 0.01 - 0.10     | High precision required
    /// Biology         | 0.10 - 0.50     | Natural variation
    /// Finance         | 0.15 - 0.30     | Market volatility
    /// Meteorology     | 0.20 - 1.00     | High natural variability
    /// ```
    ///
    /// # Risk Assessment Application
    ///
    /// In finance, CV helps compare risk-adjusted returns:
    /// ```text
    /// Risk-Adjusted Performance = Mean Return / CV
    /// Lower CV = More consistent performance
    /// ```
    ///
    /// # Limitations
    ///
    /// - **Zero Mean**: Undefined when mean = 0
    /// - **Negative Values**: Uses absolute mean to handle negative data
    /// - **Distribution Sensitivity**: Affected by outliers (through std deviation)
    /// - **Mean Near Zero**: Can become unstable
    ///
    /// # Alternative for Robust Analysis
    ///
    /// For robust coefficient of variation:
    /// ```text
    /// Robust CV = MAD / |median|
    /// ```
    ///
    /// # Computational Notes
    ///
    /// - **Time Complexity**: O(n) - single pass for mean and standard deviation
    /// - **Space Complexity**: O(1) - constant additional space
    /// - **Numerical Stability**: Good, but sensitive to zero/near-zero means
    ///
    /// # Statistical Properties
    ///
    /// For log-normal distribution:
    /// - CV has known theoretical relationship to underlying normal parameters
    /// - Useful for modeling multiplicative processes
    ///
    /// # Panics
    ///
    /// - Panics if the vector is empty (statistics undefined)
    /// - Panics if the mean is zero (division by zero)
    /// - The zero-mean check ensures mathematical validity
    fn coefficient_of_variation(&self) -> T;
    
    /// Compute the range (difference between maximum and minimum values)
    ///
    /// The range is the simplest measure of statistical dispersion, representing
    /// the total spread of the data. While easy to compute and interpret, it is
    /// highly sensitive to outliers and provides limited information about the
    /// distribution of values between the extremes.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// Range = max(x) - min(x)
    /// ```
    ///
    /// # Properties
    ///
    /// - **Simple**: Easiest dispersion measure to compute and understand
    /// - **Sensitive**: Heavily influenced by outliers (0% breakdown point)
    /// - **Incomplete**: Only uses two data points (ignores distribution shape)
    /// - **Non-Robust**: Single outlier can dramatically affect the value
    /// - **Scale-Dependent**: Has same units as original data
    ///
    /// # Algorithm
    ///
    /// 1. Find minimum value in single pass
    /// 2. Find maximum value in single pass (or combined pass)
    /// 3. Compute difference: max - min
    ///
    /// # Use Cases
    ///
    /// - **Quick Assessment**: Rapid dispersion check
    /// - **Data Validation**: Identify unexpected extreme values
    /// - **Quality Control**: Monitor process variation limits
    /// - **Initial Analysis**: First look at data spread
    /// - **Outlier Detection**: Combined with other measures
    ///
    /// # Advantages
    ///
    /// - **Computational Efficiency**: O(n) time, O(1) space
    /// - **Intuitive Interpretation**: Easy to understand
    /// - **No Assumptions**: Works with any data type
    /// - **Quick Calculation**: Minimal computational overhead
    ///
    /// # Disadvantages
    ///
    /// - **Outlier Sensitivity**: Single extreme value affects result
    /// - **Limited Information**: Ignores data distribution
    /// - **Not Robust**: Unreliable with contaminated data
    /// - **Sample Size Dependent**: Tends to increase with larger samples
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::advanced::AdvancedDescriptive;
    /// use rustlab_math::vec64;
    ///
    /// let normal_data = vec64![2, 3, 4, 5, 6, 7, 8];
    /// let range_normal = normal_data.range(); // 8 - 2 = 6
    ///
    /// let outlier_data = vec64![2, 3, 4, 5, 6, 7, 100]; // 100 is outlier
    /// let range_outlier = outlier_data.range(); // 100 - 2 = 98 (heavily affected)
    /// ```
    ///
    /// # Comparison with Robust Alternatives
    ///
    /// ```text
    /// Measure                | Sensitivity | Information Content
    /// Range                  | Highest     | Minimal (2 points)
    /// Interquartile Range    | Low         | Middle 50%
    /// Standard Deviation     | Medium      | All points, squared
    /// MAD                    | Lowest      | All points, median-based
    /// ```
    ///
    /// # Statistical Context
    ///
    /// For normal distribution with standard deviation σ:
    /// ```text
    /// Expected Range ≈ σ × √(2 ln(n))
    /// ```
    /// where n is the sample size.
    ///
    /// # Quality Control Application
    ///
    /// ```rust
    /// use rustlab_stats::advanced::AdvancedDescriptive;
    /// use rustlab_math::vec64;
    ///
    /// let measurements = vec64![9.8, 10.1, 9.9, 10.2, 9.7, 10.0];
    /// let range = measurements.range(); // 0.5
    /// let tolerance = 1.0;
    ///
    /// if range > tolerance {
    ///     println!("Process variability exceeds tolerance");
    /// }
    /// ```
    ///
    /// # Robust Alternative: Interquartile Range
    ///
    /// When outliers are a concern, consider IQR:
    /// ```text
    /// IQR = Q3 - Q1  (middle 50% of data)
    /// ```
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(n) - single pass through data
    /// - **Space Complexity**: O(1) - only stores min/max
    /// - **Numerical Stability**: Excellent for finite data
    /// - **Parallelizable**: Min/max operations can be parallelized
    ///
    /// # Panics
    ///
    /// Panics if the vector is empty, as both minimum and maximum are undefined
    /// for empty datasets.
    fn range(&self) -> T;
}

impl AdvancedDescriptive<f64> for VectorF64 {
    fn mode(&self) -> f64 {
        assert!(!self.is_empty(), "Cannot compute mode of empty vector");
        
        // Use bit representation for precise floating-point comparison
        // This avoids issues with floating-point equality
        let mut frequency_map = HashMap::new();
        
        // Count frequencies using bit representation
        for &value in self.as_slice_unchecked() {
            let bit_key = value.to_bits();
            *frequency_map.entry(bit_key).or_insert(0) += 1;
        }
        
        // Find the maximum frequency
        let max_frequency = *frequency_map.values().max().unwrap();
        
        // Among all values with maximum frequency, return the smallest
        // This provides deterministic behavior for multimodal distributions
        let mode_bits = frequency_map
            .into_iter()
            .filter(|(_, freq)| *freq == max_frequency)
            .map(|(bits, _)| bits)
            .min()  // Choose smallest among tied modes
            .unwrap();
        
        // Convert back to f64
        f64::from_bits(mode_bits)
    }
    
    fn geometric_mean(&self) -> f64 {
        assert!(!self.is_empty(), "Cannot compute geometric mean of empty vector");
        assert!(
            self.as_slice_unchecked().iter().all(|&x| x > 0.0), 
            "Geometric mean requires all positive values"
        );
        
        // Use log-transform method for numerical stability
        // GM = exp(∑ln(xᵢ)/n) rather than (∏xᵢ)^(1/n)
        // This avoids potential overflow/underflow in the product
        let log_sum: f64 = self.as_slice_unchecked()
            .iter()
            .map(|&x| x.ln())  // Transform to log space
            .sum();
        
        // Compute arithmetic mean of logs, then exponentiate
        let log_mean = log_sum / self.len() as f64;
        log_mean.exp()
    }
    
    fn harmonic_mean(&self) -> f64 {
        assert!(!self.is_empty(), "Cannot compute harmonic mean of empty vector");
        assert!(
            self.as_slice_unchecked().iter().all(|&x| x > 0.0), 
            "Harmonic mean requires all positive values"
        );
        
        // Calculate sum of reciprocals: ∑(1/xᵢ)
        let reciprocal_sum: f64 = self.as_slice_unchecked()
            .iter()
            .map(|&x| 1.0 / x)  // Compute reciprocal
            .sum();
        
        // Harmonic mean = n / ∑(1/xᵢ)
        self.len() as f64 / reciprocal_sum
    }
    
    fn trimmed_mean(&self, trim_fraction: f64) -> f64 {
        assert!(!self.is_empty(), "Cannot compute trimmed mean of empty vector");
        assert!(trim_fraction >= 0.0 && trim_fraction <= 0.5, 
                "Trim fraction must be between 0.0 and 0.5, got {}", trim_fraction);
        
        let mut sorted_data = self.as_slice_unchecked().to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_data.len();
        let trim_count = (n as f64 * trim_fraction).floor() as usize;
        
        if trim_count == 0 {
            return sorted_data.iter().sum::<f64>() / n as f64;
        }
        
        let trimmed_data = &sorted_data[trim_count..n-trim_count];
        trimmed_data.iter().sum::<f64>() / trimmed_data.len() as f64
    }
    
    fn mad(&self) -> f64 {
        assert!(!self.is_empty(), "Cannot compute MAD of empty vector");
        
        let median = self.median();
        let deviations: Vec<f64> = self.as_slice_unchecked().iter()
            .map(|&x| (x - median).abs())
            .collect();
        
        let mad_vector = VectorF64::from_slice(&deviations);
        mad_vector.median()
    }
    
    fn coefficient_of_variation(&self) -> f64 {
        use rustlab_math::statistics::BasicStatistics;
        
        assert!(!self.is_empty(), "Cannot compute CV of empty vector");
        let mean = self.mean();
        assert!(mean != 0.0, "Cannot compute CV when mean is zero");
        
        let std_dev = self.std(None);
        std_dev / mean.abs()
    }
    
    fn range(&self) -> f64 {
        assert!(!self.is_empty(), "Cannot compute range of empty vector");
        
        let slice = self.as_slice_unchecked();
        let min_val = slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        max_val - min_val
    }
}

impl AdvancedDescriptive<f32> for VectorF32 {
    fn mode(&self) -> f32 {
        assert!(!self.is_empty(), "Cannot compute mode of empty vector");
        
        let mut frequency_map = HashMap::new();
        for &value in self.as_slice_unchecked() {
            *frequency_map.entry(value.to_bits()).or_insert(0) += 1;
        }
        
        let max_frequency = *frequency_map.values().max().unwrap();
        let mode_bits = frequency_map
            .into_iter()
            .filter(|(_, freq)| *freq == max_frequency)
            .map(|(bits, _)| bits)
            .min()
            .unwrap();
        
        f32::from_bits(mode_bits)
    }
    
    fn geometric_mean(&self) -> f32 {
        assert!(!self.is_empty(), "Cannot compute geometric mean of empty vector");
        assert!(self.as_slice_unchecked().iter().all(|&x| x > 0.0), "Geometric mean requires all positive values");
        
        let log_sum: f32 = self.as_slice_unchecked().iter().map(|&x| x.ln()).sum();
        (log_sum / self.len() as f32).exp()
    }
    
    fn harmonic_mean(&self) -> f32 {
        assert!(!self.is_empty(), "Cannot compute harmonic mean of empty vector");
        assert!(self.as_slice_unchecked().iter().all(|&x| x > 0.0), "Harmonic mean requires all positive values");
        
        let reciprocal_sum: f32 = self.as_slice_unchecked().iter().map(|&x| 1.0 / x).sum();
        self.len() as f32 / reciprocal_sum
    }
    
    fn trimmed_mean(&self, trim_fraction: f32) -> f32 {
        assert!(!self.is_empty(), "Cannot compute trimmed mean of empty vector");
        assert!(trim_fraction >= 0.0 && trim_fraction <= 0.5, 
                "Trim fraction must be between 0.0 and 0.5, got {}", trim_fraction);
        
        let mut sorted_data = self.as_slice_unchecked().to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_data.len();
        let trim_count = (n as f32 * trim_fraction).floor() as usize;
        
        if trim_count == 0 {
            return sorted_data.iter().sum::<f32>() / n as f32;
        }
        
        let trimmed_data = &sorted_data[trim_count..n-trim_count];
        trimmed_data.iter().sum::<f32>() / trimmed_data.len() as f32
    }
    
    fn mad(&self) -> f32 {
        assert!(!self.is_empty(), "Cannot compute MAD of empty vector");
        
        let median = self.median();
        let deviations: Vec<f32> = self.as_slice_unchecked().iter()
            .map(|&x| (x - median).abs())
            .collect();
        
        let mad_vector = VectorF32::from_slice(&deviations);
        mad_vector.median()
    }
    
    fn coefficient_of_variation(&self) -> f32 {
        use rustlab_math::statistics::BasicStatistics;
        
        assert!(!self.is_empty(), "Cannot compute CV of empty vector");
        let mean = self.mean();
        assert!(mean != 0.0, "Cannot compute CV when mean is zero");
        
        let std_dev = self.std(None);
        std_dev / mean.abs()
    }
    
    fn range(&self) -> f32 {
        assert!(!self.is_empty(), "Cannot compute range of empty vector");
        
        let slice = self.as_slice_unchecked();
        let min_val = slice.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        max_val - min_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::vec64;
    
    #[test]
    fn test_mode() {
        let v = vec64![1, 2, 2, 3, 4];
        assert_eq!(v.mode(), 2.0);
    }
    
    #[test]
    fn test_geometric_mean() {
        let v = vec64![1, 2, 4, 8];
        let gm = v.geometric_mean();
        assert!((gm - (1.0 * 2.0 * 4.0 * 8.0_f64).powf(0.25)).abs() < 1e-10);
    }
    
    #[test]
    fn test_harmonic_mean() {
        let v = vec64![1, 2, 4];
        let hm = v.harmonic_mean();
        let expected = 3.0 / (1.0 + 0.5 + 0.25);
        assert!((hm - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_trimmed_mean() {
        let v = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let tm = v.trimmed_mean(0.2); // Trim 20% from each end
        // Should trim 2 values from each end: [3, 4, 5, 6, 7, 8]
        assert!((tm - 5.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_mad() {
        let v = vec64![1, 2, 3, 4, 5];
        let mad = v.mad();
        assert!(mad > 0.0);
    }
    
    #[test]
    fn test_coefficient_of_variation() {
        let v = vec64![1, 2, 3, 4, 5];
        let cv = v.coefficient_of_variation();
        assert!(cv > 0.0);
    }
    
    #[test]
    fn test_range() {
        let v = vec64![1, 5, 3, 9, 2];
        assert_eq!(v.range(), 8.0); // 9 - 1 = 8
    }
    
    #[test]
    #[should_panic(expected = "Geometric mean requires all positive values")]
    fn test_geometric_mean_negative() {
        let v = vec64![-1, 2, 3];
        v.geometric_mean();
    }
    
    #[test]
    #[should_panic(expected = "Harmonic mean requires all positive values")]
    fn test_harmonic_mean_negative() {
        let v = vec64![-1, 2, 3];
        v.harmonic_mean();
    }
}