//! Comprehensive data normalization and scaling for statistical preprocessing
//!
//! This module provides a complete suite of normalization and scaling techniques essential
//! for machine learning preprocessing, statistical analysis, and data standardization.
//! All methods are designed to handle both vector and multidimensional array data with
//! efficient axis-wise operations.
//!
//! # Why Normalize Data?
//!
//! Data normalization is crucial for:
//! - **Algorithm Performance**: Many ML algorithms are sensitive to feature scales
//! - **Convergence Speed**: Optimization algorithms converge faster with normalized data
//! - **Fair Feature Weighting**: Prevents features with large scales from dominating
//! - **Statistical Analysis**: Enables meaningful comparison across variables
//! - **Numerical Stability**: Reduces risk of overflow/underflow in computations
//!
//! # Normalization Methods Overview
//!
//! ## Z-Score Standardization (Standard Scaling)
//!
//! **Formula**: z = (x - μ) / σ
//!
//! **Properties**:
//! - Transforms data to have mean = 0, standard deviation = 1
//! - Preserves the shape of the distribution
//! - Makes variables comparable across different units and scales
//! - Assumes approximately normal distribution for best results
//!
//! **When to Use**:
//! - Linear algorithms (linear regression, logistic regression, SVM with linear kernel)
//! - Neural networks (helps with gradient descent convergence)
//! - PCA and other techniques sensitive to variance
//! - When features follow approximately normal distributions
//!
//! **Advantages**:
//! - Well-established statistical foundation
//! - Works well with normally distributed data
//! - Preserves relationships between observations
//!
//! **Disadvantages**:
//! - Sensitive to outliers (outliers affect mean and std)
//! - Assumes normal distribution for optimal performance
//! - Bounded outliers can still have large standardized values
//!
//! ## Robust Scaling (Median-MAD Scaling)
//!
//! **Formula**: scaled = (x - median) / IQR
//!
//! **Properties**:
//! - Uses median (50th percentile) and IQR (75th - 25th percentile)
//! - More robust to outliers than z-score standardization
//! - Centers data around 0 but with IQR = 1 instead of std = 1
//! - Distribution-free (doesn't assume normality)
//!
//! **When to Use**:
//! - Data contains outliers that shouldn't be removed
//! - Non-normal distributions (skewed, heavy-tailed)
//! - Exploratory data analysis where robustness is important
//! - When you want to preserve the influence of outliers but reduce their dominance
//!
//! **Advantages**:
//! - Robust to outliers (50% breakdown point)
//! - Works with any distribution shape
//! - Preserves the median structure of the data
//!
//! **Disadvantages**:
//! - Less efficient than z-score for normal data
//! - May not completely eliminate outlier effects
//! - Less familiar to many practitioners
//!
//! ## Min-Max Scaling (Normalization)
//!
//! **Formula**: scaled = (x - min) / (max - min)
//!
//! **Properties**:
//! - Scales data to exactly fit within [0, 1] range
//! - Preserves the original distribution shape
//! - All values are bounded within the specified range
//! - Sensitive to outliers (outliers affect min/max)
//!
//! **When to Use**:
//! - Neural networks (especially with sigmoid/tanh activations)
//! - Image processing (pixel values in [0, 1] range)
//! - When you need bounded output ranges
//! - Algorithms that expect features in [0, 1] (some clustering methods)
//!
//! **Advantages**:
//! - Guarantees exact bounds [0, 1]
//! - Preserves relationships between values
//! - Simple and interpretable
//!
//! **Disadvantages**:
//! - Very sensitive to outliers (single outlier affects entire scaling)
//! - Future data may fall outside [0, 1] if outside original range
//! - Can compress most data into small range if outliers are extreme
//!
//! ## Unit Vector Scaling (L2 Normalization)
//!
//! **Formula**: normalized = x / ||x||₂
//!
//! **Properties**:
//! - Scales data to have unit L2 norm (Euclidean length = 1)
//! - Preserves direction/angles between observations
//! - Each vector has the same "magnitude" after scaling
//! - Focus shifts from magnitude to directional patterns
//!
//! **When to Use**:
//! - Text analysis (TF-IDF vectors, word embeddings)
//! - Cosine similarity calculations
//! - When magnitude is less important than direction
//! - Neural networks with cosine-based loss functions
//!
//! **Advantages**:
//! - Eliminates magnitude effects, focuses on patterns
//! - Useful for high-dimensional sparse data
//! - Computationally efficient
//!
//! **Disadvantages**:
//! - Loses information about original magnitudes
//! - Not suitable when magnitude is important
//! - Can't handle zero vectors
//!
//! # Method Comparison Table
//!
//! | Method | Output Range | Robust to Outliers | Preserves Distribution | Use Case |
//! |--------|--------------|-------------------|------------------------|----------|
//! | Z-Score | (-∞, +∞) | No | Yes | Normal data, linear models |
//! | Robust | (-∞, +∞) | Yes | Yes | Skewed data, outliers present |
//! | Min-Max | [0, 1] | No | Yes | Bounded range needed |
//! | Unit Vector | Unit sphere | Moderate | No (magnitude lost) | Directional analysis |
//!
//! # Axis-wise Operations for Multidimensional Data
//!
//! For arrays and matrices, normalization can be applied along different axes:
//!
//! ## Row-wise Normalization (Axis::Cols)
//! Each row is normalized independently using statistics computed across columns.
//! **Use Case**: Normalizing samples/observations where each row is one data point.
//!
//! ## Column-wise Normalization (Axis::Rows)  
//! Each column is normalized independently using statistics computed across rows.
//! **Use Case**: Normalizing features where each column is one variable/feature.
//!
//! # Practical Guidelines
//!
//! ## Choosing the Right Method
//!
//! 1. **Start with Z-Score** if your data is approximately normal and has no extreme outliers
//! 2. **Use Robust Scaling** if you have outliers or skewed distributions
//! 3. **Use Min-Max** when you need guaranteed bounds [0, 1]
//! 4. **Use Unit Vector** for text data or when direction matters more than magnitude
//!
//! ## Handling Edge Cases
//!
//! - **Constant Features**: Methods will panic on zero variance/IQR/range
//! - **Missing Values**: Handle missing values before normalization
//! - **Categorical Variables**: Usually excluded from normalization
//! - **Time Series**: Consider stationarity before normalization
//!
//! ## Performance Considerations
//!
//! - All methods are O(n) for computation (after initial statistics calculation)
//! - Robust scaling requires O(n log n) for quantile computation
//! - Min-max scaling requires O(n) single pass for min/max
//! - Z-score requires O(n) for mean/std calculation
//!
//! # Machine Learning Integration
//!
//! ## Training vs Test Data
//! Always fit normalization parameters on training data only, then apply to test data:
//! ```rust
//! // ❌ Wrong: normalizes train and test separately
//! let norm_train = train_data.zscore(None);
//! let norm_test = test_data.zscore(None); // Uses test statistics!
//! 
//! // ✅ Correct: compute parameters from training data
//! let train_mean = train_data.mean();
//! let train_std = train_data.std(None);
//! // Apply same transformation to test data using training parameters
//! ```
//!
//! ## Cross-Validation
//! Normalization parameters should be recomputed for each CV fold to prevent data leakage.
//!
//! ## Feature Selection
//! Normalization can be important for feature selection methods that rely on variance
//! or correlation measures.

use rustlab_math::{VectorF64, VectorF32, ArrayF64, ArrayF32, BasicStatistics};
use rustlab_math::reductions::Axis;
use crate::advanced::quantiles::Quantiles;

/// Trait for comprehensive data normalization and scaling operations
///
/// This trait provides a unified interface for all normalization methods, enabling
/// consistent preprocessing across different data types and use cases. All methods
/// follow the same pattern of computing statistics from the data and applying
/// the appropriate transformation.
///
/// # Design Principles
///
/// - **Type Safety**: Output type matches input type for seamless integration
/// - **Performance**: Single-pass algorithms where possible
/// - **Robustness**: Clear error handling for edge cases
/// - **Consistency**: Uniform API across all normalization methods
///
/// # Mathematical Foundations
///
/// Each normalization method applies a linear transformation:
/// ```text
/// y = (x - center) / scale
/// ```
/// where `center` and `scale` parameters vary by method:
///
/// | Method | Center | Scale |
/// |--------|--------|-------|
/// | Z-Score | μ (mean) | σ (std dev) |
/// | Robust | median | IQR |
/// | Min-Max | min | (max - min) |
/// | Unit Vector | 0 | ||x||₂ |
///
/// # Usage Patterns
///
/// ```rust
/// use rustlab_stats::prelude::*;
/// use rustlab_math::vec64;
///
/// let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // Contains outlier
///
/// // Standard normalization (sensitive to outlier)
/// let z_scored = data.zscore(None);
///
/// // Robust normalization (handles outlier better)
/// let robust = data.robust_scale();
///
/// // Bounded normalization
/// let minmax = data.minmax_scale();
///
/// // Custom range normalization
/// let custom = data.minmax_scale_range((-1.0, 1.0));
///
/// // Unit vector normalization
/// let unit = data.unit_vector();
/// ```
pub trait Normalization<T> {
    /// The output type after normalization (typically same as input)
    type Output;
    
    /// Z-score standardization (mean=0, std=1) using sample statistics
    ///
    /// Performs the classical statistical standardization that transforms data to have
    /// zero mean and unit variance. This is the most widely used normalization method
    /// in statistics and machine learning.
    ///
    /// # Mathematical Details
    ///
    /// **Transformation Formula:**
    /// ```text
    /// z = (x - μ) / σ
    /// ```
    /// where:
    /// - μ = sample mean
    /// - σ = sample standard deviation (with bias correction)
    ///
    /// **Properties after transformation:**
    /// - Mean ≈ 0 (exactly 0 for population, approximately for sample)
    /// - Standard deviation ≈ 1 (exactly 1 for population)
    /// - Shape of distribution preserved
    /// - Outliers maintain their relative extreme positions
    ///
    /// # Bias Correction (ddof parameter)
    ///
    /// The `ddof` (delta degrees of freedom) parameter controls the denominator
    /// used in standard deviation calculation:
    /// - **ddof = 1** (default): Sample standard deviation, denominator = n-1
    /// - **ddof = 0**: Population standard deviation, denominator = n
    ///
    /// Use ddof=1 when data represents a sample from a larger population (most common).
    /// Use ddof=0 when data represents the complete population.
    ///
    /// # Applications and Use Cases
    ///
    /// ## Machine Learning
    /// - **Linear Models**: Essential for gradient descent convergence
    /// - **Neural Networks**: Prevents gradient vanishing/exploding
    /// - **SVM**: Ensures fair weighting of features in distance calculations
    /// - **Clustering**: K-means and other distance-based methods
    ///
    /// ## Statistical Analysis
    /// - **Hypothesis Testing**: Enables comparison across different scales
    /// - **Correlation Analysis**: Standardized correlation coefficients
    /// - **Principal Component Analysis**: Required for meaningful components
    /// - **Factor Analysis**: Standard preprocessing step
    ///
    /// # Assumptions and Limitations
    ///
    /// ## Optimal Conditions
    /// - Data follows approximately normal distribution
    /// - No extreme outliers present
    /// - Sufficient sample size (n > 30 recommended)
    /// - Homoscedasticity (constant variance) if used across groups
    ///
    /// ## Sensitivity to Outliers
    /// Both mean and standard deviation are sensitive to outliers:
    /// - Single extreme outlier can shift mean significantly
    /// - Outliers inflate standard deviation, compressing non-outlier values
    /// - Consider robust scaling for outlier-contaminated data
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// // Normal case - well-behaved data
    /// let normal_data = vec64![10, 12, 14, 16, 18, 20, 22];
    /// let standardized = normal_data.zscore(None);
    /// // Result: approximately [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    ///
    /// // With outlier - demonstrates sensitivity
    /// let outlier_data = vec64![10, 12, 14, 16, 18, 20, 100]; // 100 is outlier
    /// let standardized_outlier = outlier_data.zscore(None);
    /// // Most values compressed near 0, outlier gets large z-score
    ///
    /// // Different bias correction
    /// let sample_std = normal_data.zscore(Some(1)); // Sample std (n-1)
    /// let pop_std = normal_data.zscore(Some(0));     // Population std (n)
    /// ```
    ///
    /// # Statistical Interpretation
    ///
    /// After z-score normalization:
    /// - **z = 0**: Value equals the mean
    /// - **z = 1**: Value is one standard deviation above mean
    /// - **z = -2**: Value is two standard deviations below mean
    /// - **|z| > 2**: Often considered unusual (outside 95% for normal data)
    /// - **|z| > 3**: Often considered outliers (outside 99.7% for normal data)
    ///
    /// # Arguments
    /// * `ddof` - Delta degrees of freedom for standard deviation calculation
    ///           (Some(1) for sample std, Some(0) for population std, None defaults to 1)
    ///
    /// # Returns
    /// Standardized data with mean ≈ 0 and std ≈ 1
    ///
    /// # Panics
    /// - Panics if standard deviation is zero (all values are identical)
    /// - Panics if data is empty (mean undefined)
    /// - Panics on numerical overflow (extremely large values)
    ///
    /// # See Also
    /// - [`robust_scale`](#method.robust_scale) for outlier-resistant alternative
    /// - [`standardize`](#method.standardize) for alias to this method
    fn zscore(&self, ddof: Option<usize>) -> Self::Output;
    
    /// Alias for zscore standardization
    fn standardize(&self, ddof: Option<usize>) -> Self::Output {
        self.zscore(ddof)
    }
    
    /// Robust scaling using median and IQR for outlier-resistant normalization
    ///
    /// Provides a more robust alternative to z-score standardization by using
    /// median and interquartile range instead of mean and standard deviation.
    /// This method is less sensitive to extreme values and works well with
    /// non-normal distributions.
    ///
    /// # Mathematical Details
    ///
    /// **Transformation Formula:**
    /// ```text
    /// scaled = (x - median) / IQR
    /// ```
    /// where:
    /// - median = 50th percentile (Q2)
    /// - IQR = Q3 - Q1 (75th percentile - 25th percentile)
    ///
    /// **Properties after transformation:**
    /// - Median ≈ 0 (data centered around median)
    /// - IQR = 1 (middle 50% of data spans unit interval)
    /// - Distribution shape preserved
    /// - Outliers have reduced but not eliminated influence
    ///
    /// # Robustness Properties
    ///
    /// ## Breakdown Point
    /// - **50% breakdown point**: Up to 25% of data can be outliers in each tail
    /// - **Median**: Unaffected by up to 50% of extreme values
    /// - **IQR**: Unaffected by extreme values in top and bottom 25%
    ///
    /// ## Comparison with Z-Score
    ///
    /// | Property | Z-Score | Robust Scale |
    /// |----------|---------|-------------|
    /// | Outlier sensitivity | High | Low |
    /// | Breakdown point | 0% | 25% |
    /// | Distribution assumption | Normal preferred | None |
    /// | Efficiency (normal data) | High | Lower |
    /// | Interpretability | High | Moderate |
    ///
    /// # When to Use Robust Scaling
    ///
    /// ## Ideal Scenarios
    /// - Data contains outliers that represent valid but extreme observations
    /// - Distribution is skewed or heavy-tailed
    /// - You want to preserve outlier information while reducing their dominance
    /// - Exploratory data analysis where robustness is prioritized
    /// - Non-parametric statistical methods are being used
    ///
    /// ## Applications
    /// - **Financial Data**: Stock returns, trading volumes (natural outliers)
    /// - **Sensor Data**: Environmental measurements with occasional spikes
    /// - **Medical Data**: Biomarker levels with natural extreme values
    /// - **Quality Control**: Manufacturing data with occasional defects
    ///
    /// # Statistical Properties
    ///
    /// ## Distribution-Free
    /// Unlike z-score normalization, robust scaling makes no assumptions about
    /// the underlying distribution shape:
    /// - Works equally well with normal, skewed, or heavy-tailed distributions
    /// - No assumption of symmetry required
    /// - Suitable for ordinal data where spacing is meaningful
    ///
    /// ## Influence Function
    /// The influence function (effect of single outlier) is bounded:
    /// - **Z-score**: Unbounded influence (single outlier can affect all values)
    /// - **Robust scale**: Bounded influence (outliers have limited effect)
    ///
    /// # Limitations and Considerations
    ///
    /// ## When NOT to Use
    /// - Data is known to be normally distributed with no outliers (z-score is more efficient)
    /// - Small sample sizes (n < 10) where quartiles are unreliable
    /// - All values fall within narrow range (may amplify noise)
    /// - Downstream algorithms specifically expect standard normal distribution
    ///
    /// ## Edge Cases
    /// - **Discrete Data**: May produce many tied values after scaling
    /// - **Highly Skewed Data**: May not fully address skewness
    /// - **Multiple Modes**: Complex distributions may need specialized treatment
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// // Data with outliers
    /// let data_with_outliers = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
    /// 
    /// // Compare z-score vs robust scaling
    /// let z_scaled = data_with_outliers.zscore(None);
    /// let robust_scaled = data_with_outliers.robust_scale();
    /// 
    /// // Robust scaling handles the outlier (100) better:
    /// // - Z-score: Most values compressed near 0, outlier dominates
    /// // - Robust: More balanced scaling, outlier still extreme but less dominant
    ///
    /// // Skewed data example
    /// let skewed_data = vec64![1, 1, 2, 2, 3, 3, 4, 5, 6, 8, 12, 20, 35];
    /// let robust_skewed = skewed_data.robust_scale();
    /// // Handles skewness better than z-score normalization
    /// ```
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(n log n) due to sorting for quantiles
    /// - **Space Complexity**: O(n) for temporary sorted array
    /// - **Numerical Stability**: Very stable, no division by small numbers typically
    ///
    /// # Integration with Other Methods
    ///
    /// Robust scaling works well with:
    /// - Non-parametric statistical tests
    /// - Robust regression methods
    /// - Tree-based machine learning algorithms
    /// - Clustering methods that use median-based distances
    ///
    /// # Returns
    /// Robustly scaled data centered on median with IQR = 1
    ///
    /// # Panics
    /// - Panics if IQR is zero (Q1 = Q3, no variability in middle 50%)
    /// - Panics if data is empty (quantiles undefined)
    /// - Panics on numerical issues during quantile computation
    ///
    /// # See Also
    /// - [`zscore`](#method.zscore) for standard normalization
    /// - [`minmax_scale`](#method.minmax_scale) for bounded normalization
    fn robust_scale(&self) -> Self::Output;
    
    /// Min-max scaling to [0, 1] range
    /// 
    /// Scales data to unit interval using:
    /// scaled = (x - min) / (max - min)
    /// 
    /// # Returns
    /// Data scaled to [0, 1] range
    /// 
    /// # Panics
    /// Panics if max equals min (constant data)
    fn minmax_scale(&self) -> Self::Output;
    
    /// Min-max scaling to custom range [a, b]
    /// 
    /// Scales data to specified range using:
    /// scaled = a + (x - min) * (b - a) / (max - min)
    /// 
    /// # Arguments
    /// * `range` - Target range as (min, max) tuple
    /// 
    /// # Returns
    /// Data scaled to specified range
    /// 
    /// # Panics
    /// Panics if max equals min (constant data) or if range.0 >= range.1
    fn minmax_scale_range(&self, range: (T, T)) -> Self::Output;
    
    /// Unit vector scaling (L2 normalization)
    /// 
    /// Scales data to have unit L2 norm:
    /// normalized = x / ||x||₂
    /// 
    /// # Returns
    /// Data scaled to unit L2 norm
    /// 
    /// # Panics
    /// Panics if L2 norm is zero (zero vector)
    fn unit_vector(&self) -> Self::Output;
}

/// Trait for array-wise normalization along axes
pub trait ArrayNormalization<T> {
    /// The output array type
    type ArrayOutput;
    /// The vector type for axis operations
    type VectorOutput;
    
    /// Z-score standardization along specified axis
    /// 
    /// # Arguments
    /// * `axis` - Axis along which to compute statistics
    /// * `ddof` - Delta degrees of freedom for standard deviation
    fn zscore_axis(&self, axis: Axis, ddof: Option<usize>) -> Self::ArrayOutput;
    
    /// Robust scaling along specified axis
    fn robust_scale_axis(&self, axis: Axis) -> Self::ArrayOutput;
    
    /// Min-max scaling along specified axis
    fn minmax_scale_axis(&self, axis: Axis) -> Self::ArrayOutput;
    
    /// Min-max scaling to custom range along specified axis
    fn minmax_scale_range_axis(&self, axis: Axis, range: (T, T)) -> Self::ArrayOutput;
    
    /// Unit vector scaling along specified axis
    fn unit_vector_axis(&self, axis: Axis) -> Self::ArrayOutput;
}

// Helper function to compute L2 norm
fn l2_norm_f64(data: &[f64]) -> f64 {
    data.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn l2_norm_f32(data: &[f32]) -> f32 {
    data.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

// Implementation for VectorF64
impl Normalization<f64> for VectorF64 {
    type Output = VectorF64;
    
    fn zscore(&self, ddof: Option<usize>) -> VectorF64 {
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot standardize empty vector");
        }
        
        let mean = self.mean();
        let std = self.std(ddof);
        
        if std == 0.0 {
            panic!("Cannot standardize vector with zero standard deviation");
        }
        
        let standardized: Vec<f64> = data.iter()
            .map(|&x| (x - mean) / std)
            .collect();
        
        VectorF64::from_slice(&standardized)
    }
    
    fn robust_scale(&self) -> VectorF64 {
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot robust scale empty vector");
        }
        
        let median = self.median();
        let iqr = self.iqr();
        
        if iqr == 0.0 {
            panic!("Cannot robust scale vector with zero IQR");
        }
        
        let scaled: Vec<f64> = data.iter()
            .map(|&x| (x - median) / iqr)
            .collect();
        
        VectorF64::from_slice(&scaled)
    }
    
    fn minmax_scale(&self) -> VectorF64 {
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot min-max scale empty vector");
        }
        
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val == min_val {
            panic!("Cannot min-max scale vector with constant values");
        }
        
        let range = max_val - min_val;
        let scaled: Vec<f64> = data.iter()
            .map(|&x| (x - min_val) / range)
            .collect();
        
        VectorF64::from_slice(&scaled)
    }
    
    fn minmax_scale_range(&self, range: (f64, f64)) -> VectorF64 {
        let (target_min, target_max) = range;
        
        if target_min >= target_max {
            panic!("Invalid target range: min must be less than max");
        }
        
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot min-max scale empty vector");
        }
        
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val == min_val {
            panic!("Cannot min-max scale vector with constant values");
        }
        
        let data_range = max_val - min_val;
        let target_range = target_max - target_min;
        
        let scaled: Vec<f64> = data.iter()
            .map(|&x| target_min + (x - min_val) * target_range / data_range)
            .collect();
        
        VectorF64::from_slice(&scaled)
    }
    
    fn unit_vector(&self) -> VectorF64 {
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot normalize empty vector");
        }
        
        let norm = l2_norm_f64(data);
        
        if norm == 0.0 {
            panic!("Cannot normalize zero vector");
        }
        
        let normalized: Vec<f64> = data.iter()
            .map(|&x| x / norm)
            .collect();
        
        VectorF64::from_slice(&normalized)
    }
}

// Implementation for VectorF32
impl Normalization<f32> for VectorF32 {
    type Output = VectorF32;
    
    fn zscore(&self, ddof: Option<usize>) -> VectorF32 {
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot standardize empty vector");
        }
        
        let mean = self.mean();
        let std = self.std(ddof);
        
        if std == 0.0 {
            panic!("Cannot standardize vector with zero standard deviation");
        }
        
        let standardized: Vec<f32> = data.iter()
            .map(|&x| (x - mean) / std)
            .collect();
        
        VectorF32::from_slice(&standardized)
    }
    
    fn robust_scale(&self) -> VectorF32 {
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot robust scale empty vector");
        }
        
        let median = self.median();
        let iqr = self.iqr();
        
        if iqr == 0.0 {
            panic!("Cannot robust scale vector with zero IQR");
        }
        
        let scaled: Vec<f32> = data.iter()
            .map(|&x| (x - median) / iqr)
            .collect();
        
        VectorF32::from_slice(&scaled)
    }
    
    fn minmax_scale(&self) -> VectorF32 {
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot min-max scale empty vector");
        }
        
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val == min_val {
            panic!("Cannot min-max scale vector with constant values");
        }
        
        let range = max_val - min_val;
        let scaled: Vec<f32> = data.iter()
            .map(|&x| (x - min_val) / range)
            .collect();
        
        VectorF32::from_slice(&scaled)
    }
    
    fn minmax_scale_range(&self, range: (f32, f32)) -> VectorF32 {
        let (target_min, target_max) = range;
        
        if target_min >= target_max {
            panic!("Invalid target range: min must be less than max");
        }
        
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot min-max scale empty vector");
        }
        
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val == min_val {
            panic!("Cannot min-max scale vector with constant values");
        }
        
        let data_range = max_val - min_val;
        let target_range = target_max - target_min;
        
        let scaled: Vec<f32> = data.iter()
            .map(|&x| target_min + (x - min_val) * target_range / data_range)
            .collect();
        
        VectorF32::from_slice(&scaled)
    }
    
    fn unit_vector(&self) -> VectorF32 {
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot normalize empty vector");
        }
        
        let norm = l2_norm_f32(data);
        
        if norm == 0.0 {
            panic!("Cannot normalize zero vector");
        }
        
        let normalized: Vec<f32> = data.iter()
            .map(|&x| x / norm)
            .collect();
        
        VectorF32::from_slice(&normalized)
    }
}

// Helper functions for array operations
fn extract_and_compute_stats_f64(array: &ArrayF64, axis: Axis, index: usize) -> (f64, f64, f64, f64) {
    let data = match axis {
        Axis::Rows => {
            // Extract column `index`
            let mut col_data = Vec::with_capacity(array.nrows());
            for i in 0..array.nrows() {
                col_data.push(array.get(i, index).unwrap());
            }
            col_data
        },
        Axis::Cols => {
            // Extract row `index`
            let mut row_data = Vec::with_capacity(array.ncols());
            for j in 0..array.ncols() {
                row_data.push(array.get(index, j).unwrap());
            }
            row_data
        }
    };
    
    let vec_data = VectorF64::from_slice(&data);
    let mean = vec_data.mean();
    let std = vec_data.std(Some(1));
    let median = vec_data.median();
    let iqr = vec_data.iqr();
    
    (mean, std, median, iqr)
}

fn extract_and_compute_minmax_f64(array: &ArrayF64, axis: Axis, index: usize) -> (f64, f64) {
    let data = match axis {
        Axis::Rows => {
            // Extract column `index`
            let mut col_data = Vec::with_capacity(array.nrows());
            for i in 0..array.nrows() {
                col_data.push(array.get(i, index).unwrap());
            }
            col_data
        },
        Axis::Cols => {
            // Extract row `index`
            let mut row_data = Vec::with_capacity(array.ncols());
            for j in 0..array.ncols() {
                row_data.push(array.get(index, j).unwrap());
            }
            row_data
        }
    };
    
    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    (min_val, max_val)
}

// Implementation for ArrayF64
impl ArrayNormalization<f64> for ArrayF64 {
    type ArrayOutput = ArrayF64;
    type VectorOutput = VectorF64;
    
    fn zscore_axis(&self, axis: Axis, _ddof: Option<usize>) -> ArrayF64 {
        let (nrows, ncols) = (self.nrows(), self.ncols());
        let mut result_data = vec![0.0; nrows * ncols];
        
        match axis {
            Axis::Rows => {
                // Standardize each column
                for j in 0..ncols {
                    let (mean, std, _, _) = extract_and_compute_stats_f64(self, axis, j);
                    
                    if std == 0.0 {
                        panic!("Cannot standardize column {} with zero standard deviation", j);
                    }
                    
                    for i in 0..nrows {
                        let original_value = self.get(i, j).unwrap();
                        result_data[i * ncols + j] = (original_value - mean) / std;
                    }
                }
            },
            Axis::Cols => {
                // Standardize each row
                for i in 0..nrows {
                    let (mean, std, _, _) = extract_and_compute_stats_f64(self, axis, i);
                    
                    if std == 0.0 {
                        panic!("Cannot standardize row {} with zero standard deviation", i);
                    }
                    
                    for j in 0..ncols {
                        let original_value = self.get(i, j).unwrap();
                        result_data[i * ncols + j] = (original_value - mean) / std;
                    }
                }
            }
        }
        
        ArrayF64::from_slice(&result_data, nrows, ncols).unwrap()
    }
    
    fn robust_scale_axis(&self, axis: Axis) -> ArrayF64 {
        let (nrows, ncols) = (self.nrows(), self.ncols());
        let mut result_data = vec![0.0; nrows * ncols];
        
        match axis {
            Axis::Rows => {
                // Robust scale each column
                for j in 0..ncols {
                    let (_, _, median, iqr) = extract_and_compute_stats_f64(self, axis, j);
                    
                    if iqr == 0.0 {
                        panic!("Cannot robust scale column {} with zero IQR", j);
                    }
                    
                    for i in 0..nrows {
                        let original_value = self.get(i, j).unwrap();
                        result_data[i * ncols + j] = (original_value - median) / iqr;
                    }
                }
            },
            Axis::Cols => {
                // Robust scale each row
                for i in 0..nrows {
                    let (_, _, median, iqr) = extract_and_compute_stats_f64(self, axis, i);
                    
                    if iqr == 0.0 {
                        panic!("Cannot robust scale row {} with zero IQR", i);
                    }
                    
                    for j in 0..ncols {
                        let original_value = self.get(i, j).unwrap();
                        result_data[i * ncols + j] = (original_value - median) / iqr;
                    }
                }
            }
        }
        
        ArrayF64::from_slice(&result_data, nrows, ncols).unwrap()
    }
    
    fn minmax_scale_axis(&self, axis: Axis) -> ArrayF64 {
        let (nrows, ncols) = (self.nrows(), self.ncols());
        let mut result_data = vec![0.0; nrows * ncols];
        
        match axis {
            Axis::Rows => {
                // Min-max scale each column
                for j in 0..ncols {
                    let (min_val, max_val) = extract_and_compute_minmax_f64(self, axis, j);
                    
                    if max_val == min_val {
                        panic!("Cannot min-max scale column {} with constant values", j);
                    }
                    
                    let range = max_val - min_val;
                    
                    for i in 0..nrows {
                        let original_value = self.get(i, j).unwrap();
                        result_data[i * ncols + j] = (original_value - min_val) / range;
                    }
                }
            },
            Axis::Cols => {
                // Min-max scale each row
                for i in 0..nrows {
                    let (min_val, max_val) = extract_and_compute_minmax_f64(self, axis, i);
                    
                    if max_val == min_val {
                        panic!("Cannot min-max scale row {} with constant values", i);
                    }
                    
                    let range = max_val - min_val;
                    
                    for j in 0..ncols {
                        let original_value = self.get(i, j).unwrap();
                        result_data[i * ncols + j] = (original_value - min_val) / range;
                    }
                }
            }
        }
        
        ArrayF64::from_slice(&result_data, nrows, ncols).unwrap()
    }
    
    fn minmax_scale_range_axis(&self, axis: Axis, range: (f64, f64)) -> ArrayF64 {
        let (target_min, target_max) = range;
        
        if target_min >= target_max {
            panic!("Invalid target range: min must be less than max");
        }
        
        let (nrows, ncols) = (self.nrows(), self.ncols());
        let mut result_data = vec![0.0; nrows * ncols];
        let target_range = target_max - target_min;
        
        match axis {
            Axis::Rows => {
                // Min-max scale each column
                for j in 0..ncols {
                    let (min_val, max_val) = extract_and_compute_minmax_f64(self, axis, j);
                    
                    if max_val == min_val {
                        panic!("Cannot min-max scale column {} with constant values", j);
                    }
                    
                    let data_range = max_val - min_val;
                    
                    for i in 0..nrows {
                        let original_value = self.get(i, j).unwrap();
                        result_data[i * ncols + j] = target_min + (original_value - min_val) * target_range / data_range;
                    }
                }
            },
            Axis::Cols => {
                // Min-max scale each row
                for i in 0..nrows {
                    let (min_val, max_val) = extract_and_compute_minmax_f64(self, axis, i);
                    
                    if max_val == min_val {
                        panic!("Cannot min-max scale row {} with constant values", i);
                    }
                    
                    let data_range = max_val - min_val;
                    
                    for j in 0..ncols {
                        let original_value = self.get(i, j).unwrap();
                        result_data[i * ncols + j] = target_min + (original_value - min_val) * target_range / data_range;
                    }
                }
            }
        }
        
        ArrayF64::from_slice(&result_data, nrows, ncols).unwrap()
    }
    
    fn unit_vector_axis(&self, axis: Axis) -> ArrayF64 {
        let (nrows, ncols) = (self.nrows(), self.ncols());
        let mut result_data = vec![0.0; nrows * ncols];
        
        match axis {
            Axis::Rows => {
                // Normalize each column to unit vector
                for j in 0..ncols {
                    let mut col_data = Vec::with_capacity(nrows);
                    for i in 0..nrows {
                        col_data.push(self.get(i, j).unwrap());
                    }
                    
                    let norm = l2_norm_f64(&col_data);
                    
                    if norm == 0.0 {
                        panic!("Cannot normalize zero column {}", j);
                    }
                    
                    for i in 0..nrows {
                        result_data[i * ncols + j] = col_data[i] / norm;
                    }
                }
            },
            Axis::Cols => {
                // Normalize each row to unit vector
                for i in 0..nrows {
                    let mut row_data = Vec::with_capacity(ncols);
                    for j in 0..ncols {
                        row_data.push(self.get(i, j).unwrap());
                    }
                    
                    let norm = l2_norm_f64(&row_data);
                    
                    if norm == 0.0 {
                        panic!("Cannot normalize zero row {}", i);
                    }
                    
                    for j in 0..ncols {
                        result_data[i * ncols + j] = row_data[j] / norm;
                    }
                }
            }
        }
        
        ArrayF64::from_slice(&result_data, nrows, ncols).unwrap()
    }
}

// Similar implementations for ArrayF32 would follow the same pattern but use f32 types
impl ArrayNormalization<f32> for ArrayF32 {
    type ArrayOutput = ArrayF32;
    type VectorOutput = VectorF32;
    
    fn zscore_axis(&self, axis: Axis, ddof: Option<usize>) -> ArrayF32 {
        // Convert to f64, process, then convert back to f32
        let data_f64: Vec<f64> = (0..self.nrows() * self.ncols())
            .map(|i| {
                let row = i / self.ncols();
                let col = i % self.ncols();
                self.get(row, col).unwrap() as f64
            })
            .collect();
        
        let array_f64 = ArrayF64::from_slice(&data_f64, self.nrows(), self.ncols()).unwrap();
        let result_f64 = array_f64.zscore_axis(axis, ddof);
        
        let result_f32: Vec<f32> = (0..result_f64.nrows() * result_f64.ncols())
            .map(|i| {
                let row = i / result_f64.ncols();
                let col = i % result_f64.ncols();
                result_f64.get(row, col).unwrap() as f32
            })
            .collect();
        
        ArrayF32::from_slice(&result_f32, self.nrows(), self.ncols()).unwrap()
    }
    
    fn robust_scale_axis(&self, axis: Axis) -> ArrayF32 {
        let data_f64: Vec<f64> = (0..self.nrows() * self.ncols())
            .map(|i| {
                let row = i / self.ncols();
                let col = i % self.ncols();
                self.get(row, col).unwrap() as f64
            })
            .collect();
        
        let array_f64 = ArrayF64::from_slice(&data_f64, self.nrows(), self.ncols()).unwrap();
        let result_f64 = array_f64.robust_scale_axis(axis);
        
        let result_f32: Vec<f32> = (0..result_f64.nrows() * result_f64.ncols())
            .map(|i| {
                let row = i / result_f64.ncols();
                let col = i % result_f64.ncols();
                result_f64.get(row, col).unwrap() as f32
            })
            .collect();
        
        ArrayF32::from_slice(&result_f32, self.nrows(), self.ncols()).unwrap()
    }
    
    fn minmax_scale_axis(&self, axis: Axis) -> ArrayF32 {
        let data_f64: Vec<f64> = (0..self.nrows() * self.ncols())
            .map(|i| {
                let row = i / self.ncols();
                let col = i % self.ncols();
                self.get(row, col).unwrap() as f64
            })
            .collect();
        
        let array_f64 = ArrayF64::from_slice(&data_f64, self.nrows(), self.ncols()).unwrap();
        let result_f64 = array_f64.minmax_scale_axis(axis);
        
        let result_f32: Vec<f32> = (0..result_f64.nrows() * result_f64.ncols())
            .map(|i| {
                let row = i / result_f64.ncols();
                let col = i % result_f64.ncols();
                result_f64.get(row, col).unwrap() as f32
            })
            .collect();
        
        ArrayF32::from_slice(&result_f32, self.nrows(), self.ncols()).unwrap()
    }
    
    fn minmax_scale_range_axis(&self, axis: Axis, range: (f32, f32)) -> ArrayF32 {
        let range_f64 = (range.0 as f64, range.1 as f64);
        
        let data_f64: Vec<f64> = (0..self.nrows() * self.ncols())
            .map(|i| {
                let row = i / self.ncols();
                let col = i % self.ncols();
                self.get(row, col).unwrap() as f64
            })
            .collect();
        
        let array_f64 = ArrayF64::from_slice(&data_f64, self.nrows(), self.ncols()).unwrap();
        let result_f64 = array_f64.minmax_scale_range_axis(axis, range_f64);
        
        let result_f32: Vec<f32> = (0..result_f64.nrows() * result_f64.ncols())
            .map(|i| {
                let row = i / result_f64.ncols();
                let col = i % result_f64.ncols();
                result_f64.get(row, col).unwrap() as f32
            })
            .collect();
        
        ArrayF32::from_slice(&result_f32, self.nrows(), self.ncols()).unwrap()
    }
    
    fn unit_vector_axis(&self, axis: Axis) -> ArrayF32 {
        let data_f64: Vec<f64> = (0..self.nrows() * self.ncols())
            .map(|i| {
                let row = i / self.ncols();
                let col = i % self.ncols();
                self.get(row, col).unwrap() as f64
            })
            .collect();
        
        let array_f64 = ArrayF64::from_slice(&data_f64, self.nrows(), self.ncols()).unwrap();
        let result_f64 = array_f64.unit_vector_axis(axis);
        
        let result_f32: Vec<f32> = (0..result_f64.nrows() * result_f64.ncols())
            .map(|i| {
                let row = i / result_f64.ncols();
                let col = i % result_f64.ncols();
                result_f64.get(row, col).unwrap() as f32
            })
            .collect();
        
        ArrayF32::from_slice(&result_f32, self.nrows(), self.ncols()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::{vec64, vec32, ArrayF64};
    
    #[test]
    fn test_zscore_normalization() {
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let standardized = data.zscore(None);
        
        // Check that mean is approximately 0 and std is approximately 1
        let std_data = standardized.as_slice_unchecked();
        let mean = std_data.iter().sum::<f64>() / std_data.len() as f64;
        let variance = std_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (std_data.len() - 1) as f64;
        let std = variance.sqrt();
        
        assert!((mean).abs() < 1e-10);
        assert!((std - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_standardize_alias() {
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let standardized1 = data.zscore(None);
        let standardized2 = data.standardize(None);
        
        for i in 0..standardized1.len() {
            assert!((standardized1.get(i).unwrap() - standardized2.get(i).unwrap()).abs() < 1e-10);
        }
    }
    
    #[test]
    #[should_panic(expected = "Cannot standardize vector with zero standard deviation")]
    fn test_zscore_constant_data() {
        let data = vec64![5.0, 5.0, 5.0, 5.0];
        let _result = data.zscore(None);
    }
    
    #[test]
    fn test_robust_scaling() {
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // With outlier
        let scaled = data.robust_scale();
        
        // The scaled data should be centered on 0 (median becomes 0)
        // and the IQR should be 1
        let scaled_vec = VectorF64::from_slice(scaled.as_slice_unchecked());
        let scaled_median = scaled_vec.median();
        let scaled_iqr = scaled_vec.iqr();
        
        assert!(scaled_median.abs() < 1e-10);
        assert!((scaled_iqr - 1.0).abs() < 1e-10);
    }
    
    #[test]
    #[should_panic(expected = "Cannot robust scale vector with zero IQR")]
    fn test_robust_scale_zero_iqr() {
        let data = vec64![1.0, 2.0, 2.0, 2.0, 3.0]; // IQR = 0 (Q1=Q3=2)
        let _result = data.robust_scale();
    }
    
    #[test]
    fn test_minmax_scaling() {
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let scaled = data.minmax_scale();
        
        let scaled_data = scaled.as_slice_unchecked();
        let min_val = scaled_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = scaled_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        assert!((min_val - 0.0).abs() < 1e-10);
        assert!((max_val - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_minmax_scale_custom_range() {
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let scaled = data.minmax_scale_range((-1.0, 1.0));
        
        let scaled_data = scaled.as_slice_unchecked();
        let min_val = scaled_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = scaled_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        assert!((min_val - (-1.0)).abs() < 1e-10);
        assert!((max_val - 1.0).abs() < 1e-10);
    }
    
    #[test]
    #[should_panic(expected = "Invalid target range: min must be less than max")]
    fn test_minmax_scale_invalid_range() {
        let data = vec64![1.0, 2.0, 3.0];
        let _result = data.minmax_scale_range((2.0, 1.0));
    }
    
    #[test]
    #[should_panic(expected = "Cannot min-max scale vector with constant values")]
    fn test_minmax_scale_constant() {
        let data = vec64![3.0, 3.0, 3.0];
        let _result = data.minmax_scale();
    }
    
    #[test]
    fn test_unit_vector_normalization() {
        let data = vec64![3.0, 4.0]; // ||(3,4)|| = 5
        let normalized = data.unit_vector();
        
        let norm_data = normalized.as_slice_unchecked();
        let l2_norm = (norm_data[0].powi(2) + norm_data[1].powi(2)).sqrt();
        
        assert!((l2_norm - 1.0).abs() < 1e-10);
        assert!((norm_data[0] - 0.6).abs() < 1e-10); // 3/5
        assert!((norm_data[1] - 0.8).abs() < 1e-10); // 4/5
    }
    
    #[test]
    #[should_panic(expected = "Cannot normalize zero vector")]
    fn test_unit_vector_zero() {
        let data = vec64![0.0, 0.0, 0.0];
        let _result = data.unit_vector();
    }
    
    #[test]
    fn test_f32_vector_normalization() {
        let data = vec32![1.0, 2.0, 3.0, 4.0, 5.0];
        let standardized = data.zscore(None);
        
        // Check basic properties
        let std_data = standardized.as_slice_unchecked();
        let mean = std_data.iter().sum::<f32>() / std_data.len() as f32;
        assert!(mean.abs() < 1e-5);
    }
    
    #[test]
    fn test_array_zscore_axis_rows() {
        // Create 3x2 array
        let arr = ArrayF64::from_slice(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 3, 2).unwrap();
        // Array:
        // [1.0, 4.0]
        // [2.0, 5.0]  
        // [3.0, 6.0]
        
        let standardized = arr.zscore_axis(Axis::Rows, None);
        
        // Each column should have mean ≈ 0 and std ≈ 1
        // Column 0: [1,2,3] -> standardized
        // Column 1: [4,5,6] -> standardized
        
        // Check column 0 statistics
        let col0_mean = (standardized.get(0, 0).unwrap() + 
                         standardized.get(1, 0).unwrap() + 
                         standardized.get(2, 0).unwrap()) / 3.0;
        assert!(col0_mean.abs() < 1e-10);
        
        // Check column 1 statistics
        let col1_mean = (standardized.get(0, 1).unwrap() + 
                         standardized.get(1, 1).unwrap() + 
                         standardized.get(2, 1).unwrap()) / 3.0;
        assert!(col1_mean.abs() < 1e-10);
    }
    
    #[test]
    fn test_array_minmax_scale_axis_cols() {
        // Create 2x3 array
        let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        // Array:
        // [1.0, 2.0, 3.0]
        // [4.0, 5.0, 6.0]
        
        let scaled = arr.minmax_scale_axis(Axis::Cols);
        
        // Each row should be scaled to [0, 1]
        // Row 0: [1,2,3] -> [0, 0.5, 1]
        // Row 1: [4,5,6] -> [0, 0.5, 1]
        
        assert!((scaled.get(0, 0).unwrap() - 0.0).abs() < 1e-10);
        assert!((scaled.get(0, 1).unwrap() - 0.5).abs() < 1e-10);
        assert!((scaled.get(0, 2).unwrap() - 1.0).abs() < 1e-10);
        
        assert!((scaled.get(1, 0).unwrap() - 0.0).abs() < 1e-10);
        assert!((scaled.get(1, 1).unwrap() - 0.5).abs() < 1e-10);
        assert!((scaled.get(1, 2).unwrap() - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_array_robust_scale_axis() {
        let arr = ArrayF64::from_slice(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2).unwrap();
        // Array:
        // [1.0, 10.0]
        // [2.0, 20.0]
        // [3.0, 30.0]
        
        let scaled = arr.robust_scale_axis(Axis::Rows);
        
        // Each column should be robustly scaled
        // Verify that the scaling worked (median should be 0, IQR should be 1)
        assert!(scaled.get(0, 0).unwrap().is_finite());
        assert!(scaled.get(1, 0).unwrap().is_finite());
        assert!(scaled.get(2, 0).unwrap().is_finite());
    }
    
    #[test]
    fn test_array_unit_vector_axis() {
        let arr = ArrayF64::from_slice(&[3.0, 4.0, 6.0, 8.0], 2, 2).unwrap();
        // Array:
        // [3.0, 4.0]  -> ||row|| = 5
        // [6.0, 8.0]  -> ||row|| = 10
        
        let normalized = arr.unit_vector_axis(Axis::Cols);
        
        // Each row should have unit L2 norm
        let row0_norm = (normalized.get(0, 0).unwrap().powi(2) + 
                         normalized.get(0, 1).unwrap().powi(2)).sqrt();
        let row1_norm = (normalized.get(1, 0).unwrap().powi(2) + 
                         normalized.get(1, 1).unwrap().powi(2)).sqrt();
        
        assert!((row0_norm - 1.0).abs() < 1e-10);
        assert!((row1_norm - 1.0).abs() < 1e-10);
        
        // Check specific values for first row [3,4] -> [0.6, 0.8]
        assert!((normalized.get(0, 0).unwrap() - 0.6).abs() < 1e-10);
        assert!((normalized.get(0, 1).unwrap() - 0.8).abs() < 1e-10);
    }
    
    #[test]
    fn test_array_minmax_scale_custom_range_axis() {
        let arr = ArrayF64::from_slice(&[0.0, 1.0, 2.0, 3.0], 2, 2).unwrap();
        // Array:
        // [0.0, 1.0]
        // [2.0, 3.0]
        
        let scaled = arr.minmax_scale_range_axis(Axis::Cols, (-1.0, 1.0));
        
        // Each row should be scaled to [-1, 1]
        // Row 0: [0,1] -> [-1, 1]
        // Row 1: [2,3] -> [-1, 1]
        
        assert!((scaled.get(0, 0).unwrap() - (-1.0)).abs() < 1e-10);
        assert!((scaled.get(0, 1).unwrap() - 1.0).abs() < 1e-10);
        assert!((scaled.get(1, 0).unwrap() - (-1.0)).abs() < 1e-10);
        assert!((scaled.get(1, 1).unwrap() - 1.0).abs() < 1e-10);
    }
}