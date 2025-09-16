//! Comprehensive correlation and covariance analysis for multivariate relationships
//!
//! This module provides a complete suite of correlation measures and covariance calculations
//! for analyzing linear and non-linear relationships between variables. It includes both
//! pairwise analysis tools and matrix operations for multivariate datasets.
//!
//! # Correlation Analysis Theory
//!
//! ## What is Correlation?
//!
//! Correlation quantifies the strength and direction of linear relationships between variables.
//! Unlike covariance, correlation is standardized to the range [-1, +1], making it scale-invariant
//! and directly interpretable across different units and contexts.
//!
//! ## Correlation vs Causation
//!
//! **Critical Warning**: Correlation does not imply causation. High correlation indicates
//! statistical association but does not establish that one variable causes changes in another.
//! Always consider:
//! - Confounding variables
//! - Temporal relationships  
//! - Underlying mechanisms
//! - Simpson's paradox
//!
//! # Correlation Methods Comparison
//!
//! ## Pearson Product-Moment Correlation
//!
//! **Best for**: Linear relationships between continuous variables
//!
//! **Mathematical Definition:**
//! ```text
//! r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ - x̄)² Σ(yᵢ - ȳ)²]
//! ```
//!
//! **Properties:**
//! - Measures linear association strength
//! - Sensitive to outliers
//! - Requires approximately normal distributions
//! - Most commonly used correlation measure
//!
//! **Interpretation:**
//! - |r| = 1.0: Perfect linear relationship
//! - |r| = 0.7-0.9: Strong relationship
//! - |r| = 0.3-0.7: Moderate relationship  
//! - |r| = 0.1-0.3: Weak relationship
//! - |r| < 0.1: No meaningful linear relationship
//!
//! ## Spearman Rank Correlation
//!
//! **Best for**: Monotonic relationships, ordinal data, non-normal distributions
//!
//! **Mathematical Definition:**
//! ```text
//! ρ = Pearson correlation of rank(X), rank(Y)
//! ```
//!
//! **Properties:**
//! - Based on ranks rather than raw values
//! - Robust to outliers and non-normality
//! - Detects monotonic (not necessarily linear) relationships
//! - Distribution-free (non-parametric)
//!
//! **When to Use:**
//! - Data contains outliers
//! - Relationship is monotonic but not linear
//! - Variables are ordinal (ranked data)
//! - Distributions are highly skewed
//!
//! ## Kendall's Tau Correlation
//!
//! **Best for**: Small samples, many ties, robust alternative to Spearman
//!
//! **Mathematical Definition:**
//! ```text
//! τ = (C - D) / (C + D)
//! ```
//! where C = concordant pairs, D = discordant pairs
//!
//! **Properties:**
//! - Based on pairwise concordance
//! - More robust than Spearman for small samples
//! - Better handling of tied values
//! - Closer to normal distribution under null hypothesis
//!
//! **When to Use:**
//! - Small sample sizes (n < 30)
//! - Many tied values in data
//! - Hypothesis testing required
//! - Need robust measure of association
//!
//! # Covariance Analysis
//!
//! ## Sample vs Population Covariance
//!
//! **Sample Covariance (Unbiased):**
//! ```text
//! sᵪᵧ = Σ(xᵢ - x̄)(yᵢ - ȳ) / (n - 1)
//! ```
//!
//! **Population Covariance (Biased):**
//! ```text
//! σᵪᵧ = Σ(xᵢ - x̄)(yᵢ - ȳ) / n
//! ```
//!
//! The sample covariance uses (n-1) denominator (Bessel's correction) to provide an
//! unbiased estimator of population covariance.
//!
//! ## Relationship to Correlation
//!
//! ```text
//! Correlation = Covariance / (std(X) × std(Y))
//! ```
//!
//! Correlation is standardized covariance, making it dimensionless and comparable
//! across different scales.
//!
//! # Matrix Operations
//!
//! ## Correlation Matrix
//!
//! For multivariate data with p variables, the correlation matrix R is p×p where:
//! - R[i,j] = correlation between variables i and j
//! - Diagonal elements R[i,i] = 1 (perfect self-correlation)
//! - Matrix is symmetric: R[i,j] = R[j,i]
//! - All elements in [-1, +1]
//!
//! ## Covariance Matrix
//!
//! The covariance matrix Σ has properties:
//! - Σ[i,j] = covariance between variables i and j  
//! - Diagonal elements Σ[i,i] = variance of variable i
//! - Matrix is positive semi-definite
//! - Units: product of original variable units
//!
//! # Cross-Covariance and Time Series
//!
//! ## Lagged Relationships
//!
//! Cross-covariance at lag k measures relationship between X(t) and Y(t+k):
//! ```text
//! γᵪᵧ(k) = E[(X(t) - μₙ)(Y(t+k) - μᵧ)]
//! ```
//!
//! **Applications:**
//! - Economic indicators: Leading/lagging relationships
//! - Signal processing: Time delay estimation
//! - Finance: Cross-asset correlations with delays
//! - Engineering: System identification
//!
//! ## Cross-Covariance Function
//!
//! The complete cross-covariance function shows relationship strength across all lags,
//! enabling identification of:
//! - Optimal lag for maximum correlation
//! - Periodic patterns in relationships
//! - Causal direction indicators
//!
//! # Practical Applications
//!
//! ## Finance and Economics
//! - **Portfolio Optimization**: Diversification through low correlations
//! - **Risk Management**: Correlation during market stress
//! - **Pairs Trading**: Find highly correlated asset pairs
//! - **Factor Analysis**: Identify common risk factors
//!
//! ## Quality Control and Manufacturing
//! - **Process Variables**: Identify key relationships
//! - **Defect Analysis**: Correlate defect types with process parameters
//! - **Predictive Maintenance**: Equipment parameter correlations
//!
//! ## Medical and Biological Research
//! - **Biomarker Studies**: Correlate biomarkers with outcomes
//! - **Genetic Analysis**: Gene expression correlations
//! - **Clinical Trials**: Treatment response relationships
//!
//! ## Machine Learning and Data Science
//! - **Feature Selection**: Remove highly correlated features
//! - **Dimensionality Reduction**: PCA relies on correlation matrix
//! - **Anomaly Detection**: Detect unusual correlation patterns
//! - **Model Validation**: Check residual correlations
//!
//! # Statistical Significance and Interpretation
//!
//! ## Hypothesis Testing
//!
//! **Null Hypothesis**: H₀: ρ = 0 (no correlation)
//! **Test Statistic**: t = r√(n-2) / √(1-r²)
//! **Distribution**: t-distribution with n-2 degrees of freedom
//!
//! ## Effect Size Guidelines (Cohen's Conventions)
//! - **Small Effect**: |r| = 0.10
//! - **Medium Effect**: |r| = 0.30  
//! - **Large Effect**: |r| = 0.50
//!
//! Note: These are general guidelines; context-specific interpretation is crucial.
//!
//! ## Multiple Comparisons
//!
//! When computing many correlations simultaneously, adjust for multiple testing:
//! - **Bonferroni Correction**: α' = α / m (conservative)
//! - **False Discovery Rate**: Control expected proportion of false discoveries
//! - **Random Matrix Theory**: Account for correlations due to chance
//!
//! # Computational Considerations
//!
//! ## Numerical Stability
//! - Use numerically stable algorithms for correlation computation
//! - Avoid subtraction of large similar numbers
//! - Consider precision requirements for downstream analysis
//!
//! ## Missing Data
//! - **Listwise Deletion**: Remove observations with any missing values
//! - **Pairwise Deletion**: Use all available pairs for each correlation
//! - **Imputation**: Fill missing values before correlation analysis
//!
//! ## Large-Scale Data
//! - **Streaming Algorithms**: Update correlations incrementally
//! - **Sparse Methods**: Exploit sparsity in correlation matrices
//! - **Parallel Computing**: Compute correlation matrices in parallel

use rustlab_math::{VectorF64, VectorF32, ArrayF64, ArrayF32};
use rustlab_math::statistics::BasicStatistics;

/// Enumeration of correlation methods with different assumptions and use cases
///
/// Each correlation method captures different aspects of variable relationships and has
/// specific strengths and limitations. The choice of method should be based on data
/// characteristics, distributional assumptions, and the type of relationship expected.
///
/// # Method Selection Guide
///
/// | Characteristic | Pearson | Spearman | Kendall |
/// |---------------|---------|----------|----------|
/// | Relationship Type | Linear | Monotonic | Monotonic |
/// | Data Level | Interval/Ratio | Ordinal+ | Ordinal+ |
/// | Distribution Assumption | Normal | None | None |
/// | Outlier Sensitivity | High | Low | Low |
/// | Sample Size | Any | Medium+ | Small+ |
/// | Computational Cost | O(n) | O(n log n) | O(n²) |
/// | Ties Handling | N/A | Average ranks | Natural |
/// | Hypothesis Testing | Well-established | Good | Excellent |
///
/// # Detailed Comparison
///
/// ## Pearson Correlation
/// **Strengths:**
/// - Most powerful for linear relationships
/// - Well-established statistical theory
/// - Computationally efficient
/// - Directly interpretable as linear association strength
///
/// **Limitations:**
/// - Assumes linear relationship
/// - Sensitive to outliers
/// - Requires approximately normal distributions for inference
/// - May miss non-linear relationships
///
/// **Use When:**
/// - Variables are continuous and approximately normal
/// - Relationship is expected to be linear
/// - No extreme outliers present
/// - Maximum statistical power needed
///
/// ## Spearman Correlation
/// **Strengths:**
/// - Detects any monotonic relationship
/// - Robust to outliers and non-normality
/// - Appropriate for ordinal data
/// - Distribution-free inference
///
/// **Limitations:**
/// - Less powerful than Pearson for linear relationships
/// - Requires ranking step (computational cost)
/// - May be affected by tied values
/// - Loses information about actual distances
///
/// **Use When:**
/// - Data contains outliers
/// - Distributions are non-normal
/// - Relationship is monotonic but not necessarily linear
/// - Variables are ordinal
///
/// ## Kendall's Tau
/// **Strengths:**
/// - Excellent for small samples
/// - Robust handling of tied values
/// - Better distributional properties for testing
/// - Most robust to outliers
///
/// **Limitations:**
/// - Computationally expensive O(n²)
/// - Lower statistical power for large samples
/// - Less familiar to many users
/// - Generally smaller magnitude than other measures
///
/// **Use When:**
/// - Sample size is small (n < 30)
/// - Many tied values present
/// - Robust measure needed
/// - Hypothesis testing is primary goal
///
/// # Examples of Method Differences
///
/// ```rust
/// use rustlab_stats::prelude::*;
/// use rustlab_math::vec64;
///
/// // Linear relationship - Pearson will be strongest
/// let x = vec64![1, 2, 3, 4, 5];
/// let y = vec64![2, 4, 6, 8, 10];
/// 
/// // Non-linear but monotonic - Spearman captures this better
/// let x2 = vec64![1, 2, 3, 4, 5];
/// let y2 = vec64![1, 4, 9, 16, 25]; // x² relationship
/// 
/// // With outliers - Spearman and Kendall more robust
/// let x3 = vec64![1, 2, 3, 4, 100]; // Extreme outlier
/// let y3 = vec64![2, 4, 6, 8, 200]; // Corresponding outlier
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrelationMethod {
    /// Pearson product-moment correlation (linear relationships)
    Pearson,
    /// Spearman rank correlation (monotonic relationships)
    Spearman,
    /// Kendall's tau correlation (concordance-based)
    Kendall,
}

impl Default for CorrelationMethod {
    fn default() -> Self {
        CorrelationMethod::Pearson
    }
}

/// Trait for correlation analysis between vectors
pub trait Correlation<T> {
    /// Compute Pearson correlation coefficient between two vectors
    /// 
    /// Measures linear relationships between variables.
    /// Returns values in [-1, 1] where:
    /// - 1: Perfect positive linear relationship
    /// - 0: No linear relationship  
    /// - -1: Perfect negative linear relationship
    /// 
    /// # Panics
    /// Panics if vectors have different lengths or if either has zero variance
    fn pearson_correlation(&self, other: &Self) -> T;
    
    /// Compute Spearman rank correlation coefficient
    /// 
    /// Measures monotonic relationships by correlating the ranks of values.
    /// More robust to outliers than Pearson correlation.
    /// 
    /// # Panics
    /// Panics if vectors have different lengths
    fn spearman_correlation(&self, other: &Self) -> T;
    
    /// Compute Kendall's tau correlation coefficient
    /// 
    /// Measures concordance between two variables based on relative ordering.
    /// Robust to outliers and suitable for small samples.
    /// 
    /// # Panics
    /// Panics if vectors have different lengths
    fn kendall_tau(&self, other: &Self) -> T;
    
    /// Compute correlation using specified method
    /// 
    /// # Arguments
    /// * `other` - The other vector to correlate with
    /// * `method` - Correlation method to use
    fn correlation(&self, other: &Self, method: CorrelationMethod) -> T;
}

/// Trait for covariance analysis between vectors
pub trait Covariance<T> {
    /// Compute sample covariance between two vectors
    /// 
    /// Covariance measures how much two variables change together.
    /// Uses sample covariance formula (n-1 denominator).
    /// 
    /// # Panics
    /// Panics if vectors have different lengths or are empty
    fn covariance(&self, other: &Self) -> T;
    
    /// Compute population covariance between two vectors
    /// 
    /// Uses population covariance formula (n denominator).
    /// 
    /// # Panics
    /// Panics if vectors have different lengths or are empty
    fn covariance_pop(&self, other: &Self) -> T;
}

// Helper function to compute ranks for Spearman correlation
fn compute_ranks_f64(data: &[f64]) -> Vec<f64> {
    let mut indexed_data: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let mut ranks = vec![0.0; data.len()];
    let mut i = 0;
    
    while i < indexed_data.len() {
        let current_value = indexed_data[i].1;
        let mut j = i;
        
        // Find all elements with the same value
        while j < indexed_data.len() && indexed_data[j].1 == current_value {
            j += 1;
        }
        
        // Assign average rank to tied elements
        let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed_data[k].0] = avg_rank;
        }
        
        i = j;
    }
    
    ranks
}

fn compute_ranks_f32(data: &[f32]) -> Vec<f32> {
    let mut indexed_data: Vec<(usize, f32)> = data.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let mut ranks = vec![0.0; data.len()];
    let mut i = 0;
    
    while i < indexed_data.len() {
        let current_value = indexed_data[i].1;
        let mut j = i;
        
        // Find all elements with the same value
        while j < indexed_data.len() && indexed_data[j].1 == current_value {
            j += 1;
        }
        
        // Assign average rank to tied elements
        let avg_rank = (i + j - 1) as f32 / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed_data[k].0] = avg_rank;
        }
        
        i = j;
    }
    
    ranks
}

impl Correlation<f64> for VectorF64 {
    fn pearson_correlation(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let x_slice = self.as_slice_unchecked();
        let y_slice = other.as_slice_unchecked();
        
        let x_mean = self.mean();
        let y_mean = other.mean();
        
        let mut numerator = 0.0;
        let mut x_sum_sq = 0.0;
        let mut y_sum_sq = 0.0;
        
        for (&x, &y) in x_slice.iter().zip(y_slice.iter()) {
            let x_dev = x - x_mean;
            let y_dev = y - y_mean;
            
            numerator += x_dev * y_dev;
            x_sum_sq += x_dev * x_dev;
            y_sum_sq += y_dev * y_dev;
        }
        
        let denominator = (x_sum_sq * y_sum_sq).sqrt();
        assert!(denominator != 0.0, "Cannot compute correlation when one or both variables have zero variance");
        
        numerator / denominator
    }
    
    fn spearman_correlation(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let x_ranks = compute_ranks_f64(self.as_slice_unchecked());
        let y_ranks = compute_ranks_f64(other.as_slice_unchecked());
        
        let x_rank_vec = VectorF64::from_slice(&x_ranks);
        let y_rank_vec = VectorF64::from_slice(&y_ranks);
        
        x_rank_vec.pearson_correlation(&y_rank_vec)
    }
    
    fn kendall_tau(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let x_slice = self.as_slice_unchecked();
        let y_slice = other.as_slice_unchecked();
        let n = self.len();
        
        let mut concordant = 0;
        let mut discordant = 0;
        
        for i in 0..n {
            for j in (i+1)..n {
                let x_diff = (x_slice[j] - x_slice[i]).signum();
                let y_diff = (y_slice[j] - y_slice[i]).signum();
                
                let product = x_diff * y_diff;
                if product > 0.0 {
                    concordant += 1;
                } else if product < 0.0 {
                    discordant += 1;
                }
                // If product == 0, it's a tie, so we ignore it
            }
        }
        
        let total_pairs = n * (n - 1) / 2;
        (concordant - discordant) as f64 / total_pairs as f64
    }
    
    fn correlation(&self, other: &Self, method: CorrelationMethod) -> f64 {
        match method {
            CorrelationMethod::Pearson => self.pearson_correlation(other),
            CorrelationMethod::Spearman => self.spearman_correlation(other),
            CorrelationMethod::Kendall => self.kendall_tau(other),
        }
    }
}

impl Covariance<f64> for VectorF64 {
    fn covariance(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let x_slice = self.as_slice_unchecked();
        let y_slice = other.as_slice_unchecked();
        
        let x_mean = self.mean();
        let y_mean = other.mean();
        
        let covariance_sum: f64 = x_slice.iter().zip(y_slice.iter())
            .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
            .sum();
        
        covariance_sum / (self.len() - 1) as f64
    }
    
    fn covariance_pop(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let x_slice = self.as_slice_unchecked();
        let y_slice = other.as_slice_unchecked();
        
        let x_mean = self.mean();
        let y_mean = other.mean();
        
        let covariance_sum: f64 = x_slice.iter().zip(y_slice.iter())
            .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
            .sum();
        
        covariance_sum / self.len() as f64
    }
}

impl Correlation<f32> for VectorF32 {
    fn pearson_correlation(&self, other: &Self) -> f32 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let x_slice = self.as_slice_unchecked();
        let y_slice = other.as_slice_unchecked();
        
        let x_mean = self.mean();
        let y_mean = other.mean();
        
        let mut numerator = 0.0;
        let mut x_sum_sq = 0.0;
        let mut y_sum_sq = 0.0;
        
        for (&x, &y) in x_slice.iter().zip(y_slice.iter()) {
            let x_dev = x - x_mean;
            let y_dev = y - y_mean;
            
            numerator += x_dev * y_dev;
            x_sum_sq += x_dev * x_dev;
            y_sum_sq += y_dev * y_dev;
        }
        
        let denominator = (x_sum_sq * y_sum_sq).sqrt();
        assert!(denominator != 0.0, "Cannot compute correlation when one or both variables have zero variance");
        
        numerator / denominator
    }
    
    fn spearman_correlation(&self, other: &Self) -> f32 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let x_ranks = compute_ranks_f32(self.as_slice_unchecked());
        let y_ranks = compute_ranks_f32(other.as_slice_unchecked());
        
        let x_rank_vec = VectorF32::from_slice(&x_ranks);
        let y_rank_vec = VectorF32::from_slice(&y_ranks);
        
        x_rank_vec.pearson_correlation(&y_rank_vec)
    }
    
    fn kendall_tau(&self, other: &Self) -> f32 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let x_slice = self.as_slice_unchecked();
        let y_slice = other.as_slice_unchecked();
        let n = self.len();
        
        let mut concordant = 0;
        let mut discordant = 0;
        
        for i in 0..n {
            for j in (i+1)..n {
                let x_diff = (x_slice[j] - x_slice[i]).signum();
                let y_diff = (y_slice[j] - y_slice[i]).signum();
                
                let product = x_diff * y_diff;
                if product > 0.0 {
                    concordant += 1;
                } else if product < 0.0 {
                    discordant += 1;
                }
            }
        }
        
        let total_pairs = n * (n - 1) / 2;
        (concordant - discordant) as f32 / total_pairs as f32
    }
    
    fn correlation(&self, other: &Self, method: CorrelationMethod) -> f32 {
        match method {
            CorrelationMethod::Pearson => self.pearson_correlation(other),
            CorrelationMethod::Spearman => self.spearman_correlation(other),
            CorrelationMethod::Kendall => self.kendall_tau(other),
        }
    }
}

impl Covariance<f32> for VectorF32 {
    fn covariance(&self, other: &Self) -> f32 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let x_slice = self.as_slice_unchecked();
        let y_slice = other.as_slice_unchecked();
        
        let x_mean = self.mean();
        let y_mean = other.mean();
        
        let covariance_sum: f32 = x_slice.iter().zip(y_slice.iter())
            .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
            .sum();
        
        covariance_sum / (self.len() - 1) as f32
    }
    
    fn covariance_pop(&self, other: &Self) -> f32 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let x_slice = self.as_slice_unchecked();
        let y_slice = other.as_slice_unchecked();
        
        let x_mean = self.mean();
        let y_mean = other.mean();
        
        let covariance_sum: f32 = x_slice.iter().zip(y_slice.iter())
            .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
            .sum();
        
        covariance_sum / self.len() as f32
    }
}

/// Trait for computing correlation and covariance matrices
pub trait CorrelationMatrix<T> {
    /// Vector output type
    type VectorOutput;
    /// Array output type  
    type ArrayOutput;
    
    /// Compute correlation matrix between columns
    /// 
    /// Returns a symmetric matrix where element (i,j) is the correlation
    /// between columns i and j.
    fn correlation_matrix(&self, method: Option<CorrelationMethod>) -> Self::ArrayOutput;
    
    /// Compute covariance matrix between columns
    /// 
    /// Returns a symmetric matrix where element (i,j) is the covariance
    /// between columns i and j.
    fn covariance_matrix(&self, population: bool) -> Self::ArrayOutput;
}

/// Trait for cross-covariance and lag analysis
pub trait CrossCovariance<T> {
    /// Compute cross-covariance between two vectors at specified lag
    /// 
    /// Cross-covariance measures the covariance between x(t) and y(t+lag).
    /// Positive lag means y is shifted forward, negative lag means y is shifted backward.
    /// 
    /// # Arguments
    /// * `other` - The other vector for cross-covariance
    /// * `lag` - The lag offset (can be positive or negative)
    /// * `population` - Whether to use population (true) or sample (false) covariance
    /// 
    /// # Returns
    /// Cross-covariance value at the specified lag
    fn cross_covariance(&self, other: &Self, lag: i32, population: bool) -> T;
    
    /// Compute cross-covariance function for a range of lags
    /// 
    /// # Arguments
    /// * `other` - The other vector for cross-covariance
    /// * `max_lag` - Maximum lag to compute (computes from -max_lag to +max_lag)
    /// * `population` - Whether to use population covariance
    /// 
    /// # Returns
    /// Vector of cross-covariances, indexed from lag -max_lag to +max_lag
    fn cross_covariance_function(&self, other: &Self, max_lag: usize, population: bool) -> Self;
}

// Helper function to extract column data from arrays
fn extract_column_f64(array: &ArrayF64, col_index: usize) -> VectorF64 {
    let mut data = Vec::with_capacity(array.nrows());
    for i in 0..array.nrows() {
        data.push(array.get(i, col_index).unwrap());
    }
    VectorF64::from_slice(&data)
}

fn extract_column_f32(array: &ArrayF32, col_index: usize) -> VectorF32 {
    let mut data = Vec::with_capacity(array.nrows());
    for i in 0..array.nrows() {
        data.push(array.get(i, col_index).unwrap());
    }
    VectorF32::from_slice(&data)
}

impl CorrelationMatrix<f64> for ArrayF64 {
    type VectorOutput = VectorF64;
    type ArrayOutput = ArrayF64;
    
    fn correlation_matrix(&self, method: Option<CorrelationMethod>) -> ArrayF64 {
        let method = method.unwrap_or_default();
        let n_cols = self.ncols();
        let mut corr_data = vec![0.0; n_cols * n_cols];
        
        for i in 0..n_cols {
            for j in 0..n_cols {
                let corr_value = if i == j {
                    1.0 // Diagonal elements are always 1
                } else if i < j {
                    // Compute correlation for upper triangle
                    let col_i = extract_column_f64(self, i);
                    let col_j = extract_column_f64(self, j);
                    col_i.correlation(&col_j, method)
                } else {
                    // Use symmetry for lower triangle
                    corr_data[j * n_cols + i]
                };
                
                corr_data[i * n_cols + j] = corr_value;
            }
        }
        
        ArrayF64::from_slice(&corr_data, n_cols, n_cols).unwrap()
    }
    
    fn covariance_matrix(&self, population: bool) -> ArrayF64 {
        let n_cols = self.ncols();
        let mut cov_data = vec![0.0; n_cols * n_cols];
        
        for i in 0..n_cols {
            for j in 0..n_cols {
                let cov_value = if i <= j {
                    // Compute covariance for upper triangle and diagonal
                    let col_i = extract_column_f64(self, i);
                    let col_j = extract_column_f64(self, j);
                    
                    if population {
                        col_i.covariance_pop(&col_j)
                    } else {
                        col_i.covariance(&col_j)
                    }
                } else {
                    // Use symmetry for lower triangle
                    cov_data[j * n_cols + i]
                };
                
                cov_data[i * n_cols + j] = cov_value;
            }
        }
        
        ArrayF64::from_slice(&cov_data, n_cols, n_cols).unwrap()
    }
}

impl CorrelationMatrix<f32> for ArrayF32 {
    type VectorOutput = VectorF32;
    type ArrayOutput = ArrayF32;
    
    fn correlation_matrix(&self, method: Option<CorrelationMethod>) -> ArrayF32 {
        let method = method.unwrap_or_default();
        let n_cols = self.ncols();
        let mut corr_data = vec![0.0; n_cols * n_cols];
        
        for i in 0..n_cols {
            for j in 0..n_cols {
                let corr_value = if i == j {
                    1.0 // Diagonal elements are always 1
                } else if i < j {
                    // Compute correlation for upper triangle
                    let col_i = extract_column_f32(self, i);
                    let col_j = extract_column_f32(self, j);
                    col_i.correlation(&col_j, method)
                } else {
                    // Use symmetry for lower triangle
                    corr_data[j * n_cols + i]
                };
                
                corr_data[i * n_cols + j] = corr_value;
            }
        }
        
        ArrayF32::from_slice(&corr_data, n_cols, n_cols).unwrap()
    }
    
    fn covariance_matrix(&self, population: bool) -> ArrayF32 {
        let n_cols = self.ncols();
        let mut cov_data = vec![0.0; n_cols * n_cols];
        
        for i in 0..n_cols {
            for j in 0..n_cols {
                let cov_value = if i <= j {
                    // Compute covariance for upper triangle and diagonal
                    let col_i = extract_column_f32(self, i);
                    let col_j = extract_column_f32(self, j);
                    
                    if population {
                        col_i.covariance_pop(&col_j)
                    } else {
                        col_i.covariance(&col_j)
                    }
                } else {
                    // Use symmetry for lower triangle
                    cov_data[j * n_cols + i]
                };
                
                cov_data[i * n_cols + j] = cov_value;
            }
        }
        
        ArrayF32::from_slice(&cov_data, n_cols, n_cols).unwrap()
    }
}

impl CrossCovariance<f64> for VectorF64 {
    fn cross_covariance(&self, other: &Self, lag: i32, population: bool) -> f64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let self_slice = self.as_slice_unchecked();
        let other_slice = other.as_slice_unchecked();
        
        // Determine the overlapping range based on lag
        let (start_self, start_other, overlap_len) = if lag >= 0 {
            // Positive lag: other is shifted forward
            let lag = lag as usize;
            if lag >= self.len() {
                return 0.0; // No overlap
            }
            (0, lag, self.len() - lag)
        } else {
            // Negative lag: self is shifted forward  
            let lag = (-lag) as usize;
            if lag >= self.len() {
                return 0.0; // No overlap
            }
            (lag, 0, self.len() - lag)
        };
        
        if overlap_len == 0 {
            return 0.0;
        }
        
        // Compute means of overlapping portions
        let self_mean: f64 = self_slice[start_self..start_self + overlap_len].iter().sum::<f64>() / overlap_len as f64;
        let other_mean: f64 = other_slice[start_other..start_other + overlap_len].iter().sum::<f64>() / overlap_len as f64;
        
        // Compute cross-covariance
        let cross_cov_sum: f64 = self_slice[start_self..start_self + overlap_len]
            .iter()
            .zip(other_slice[start_other..start_other + overlap_len].iter())
            .map(|(&x, &y)| (x - self_mean) * (y - other_mean))
            .sum();
        
        let denominator = if population {
            overlap_len as f64
        } else {
            (overlap_len - 1) as f64
        };
        
        cross_cov_sum / denominator
    }
    
    fn cross_covariance_function(&self, other: &Self, max_lag: usize, population: bool) -> VectorF64 {
        let total_lags = 2 * max_lag + 1;
        let mut ccf_data = Vec::with_capacity(total_lags);
        
        for lag_offset in 0..total_lags {
            let lag = lag_offset as i32 - max_lag as i32;
            let ccv = self.cross_covariance(other, lag, population);
            ccf_data.push(ccv);
        }
        
        VectorF64::from_slice(&ccf_data)
    }
}

impl CrossCovariance<f32> for VectorF32 {
    fn cross_covariance(&self, other: &Self, lag: i32, population: bool) -> f32 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length");
        assert!(!self.is_empty(), "Vectors cannot be empty");
        
        let self_slice = self.as_slice_unchecked();
        let other_slice = other.as_slice_unchecked();
        
        // Determine the overlapping range based on lag
        let (start_self, start_other, overlap_len) = if lag >= 0 {
            // Positive lag: other is shifted forward
            let lag = lag as usize;
            if lag >= self.len() {
                return 0.0; // No overlap
            }
            (0, lag, self.len() - lag)
        } else {
            // Negative lag: self is shifted forward  
            let lag = (-lag) as usize;
            if lag >= self.len() {
                return 0.0; // No overlap
            }
            (lag, 0, self.len() - lag)
        };
        
        if overlap_len == 0 {
            return 0.0;
        }
        
        // Compute means of overlapping portions
        let self_mean: f32 = self_slice[start_self..start_self + overlap_len].iter().sum::<f32>() / overlap_len as f32;
        let other_mean: f32 = other_slice[start_other..start_other + overlap_len].iter().sum::<f32>() / overlap_len as f32;
        
        // Compute cross-covariance
        let cross_cov_sum: f32 = self_slice[start_self..start_self + overlap_len]
            .iter()
            .zip(other_slice[start_other..start_other + overlap_len].iter())
            .map(|(&x, &y)| (x - self_mean) * (y - other_mean))
            .sum();
        
        let denominator = if population {
            overlap_len as f32
        } else {
            (overlap_len - 1) as f32
        };
        
        cross_cov_sum / denominator
    }
    
    fn cross_covariance_function(&self, other: &Self, max_lag: usize, population: bool) -> VectorF32 {
        let total_lags = 2 * max_lag + 1;
        let mut ccf_data = Vec::with_capacity(total_lags);
        
        for lag_offset in 0..total_lags {
            let lag = lag_offset as i32 - max_lag as i32;
            let ccv = self.cross_covariance(other, lag, population);
            ccf_data.push(ccv);
        }
        
        VectorF32::from_slice(&ccf_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::vec64;
    
    #[test]
    fn test_pearson_correlation_perfect_positive() {
        let x = vec64![1, 2, 3, 4, 5];
        let y = vec64![2, 4, 6, 8, 10]; // Perfect positive correlation
        
        let corr = x.pearson_correlation(&y);
        assert!((corr - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_pearson_correlation_perfect_negative() {
        let x = vec64![1, 2, 3, 4, 5];
        let y = vec64![10, 8, 6, 4, 2]; // Perfect negative correlation
        
        let corr = x.pearson_correlation(&y);
        assert!((corr - (-1.0)).abs() < 1e-10);
    }
    
    #[test]
    fn test_pearson_correlation_no_correlation() {
        let x = vec64![1, 2, 3, 4, 5];
        let y = vec64![1, 1, 1, 1, 1]; // No variation in y
        
        // This should panic due to zero variance
        let result = std::panic::catch_unwind(|| {
            x.pearson_correlation(&y)
        });
        assert!(result.is_err());
    }
    
    #[test]
    fn test_spearman_correlation() {
        let x = vec64![1, 2, 3, 4, 5];
        let y = vec64![1, 4, 9, 16, 25]; // Monotonic but not linear
        
        let corr = x.spearman_correlation(&y);
        assert!((corr - 1.0).abs() < 1e-10); // Should be perfect since both are monotonic
    }
    
    #[test]
    fn test_kendall_tau() {
        let x = vec64![1, 2, 3, 4];
        let y = vec64![1, 3, 2, 4]; // One inversion
        
        let tau = x.kendall_tau(&y);
        // With one discordant pair out of 6 total pairs: (6-2)/6 = 4/6 = 2/3
        assert!((tau - (2.0/3.0)).abs() < 1e-10);
    }
    
    #[test]
    fn test_covariance() {
        let x = vec64![1, 2, 3, 4, 5];
        let y = vec64![2, 4, 6, 8, 10];
        
        let cov = x.covariance(&y);
        // For these perfectly correlated data, covariance should be positive
        assert!(cov > 0.0);
        
        // Test relationship: cov = corr * std_x * std_y
        let corr = x.pearson_correlation(&y);
        let std_x = x.std(None);
        let std_y = y.std(None);
        let expected_cov = corr * std_x * std_y;
        
        assert!((cov - expected_cov).abs() < 1e-10);
    }
    
    #[test]
    fn test_covariance_vs_population() {
        let x = vec64![1, 2, 3, 4, 5];
        let y = vec64![2, 4, 6, 8, 10];
        
        let sample_cov = x.covariance(&y);
        let pop_cov = x.covariance_pop(&y);
        
        // Population covariance should be smaller (uses n instead of n-1)
        assert!(pop_cov < sample_cov);
        
        // They should be related by factor of (n-1)/n
        let n = x.len() as f64;
        let expected_pop_cov = sample_cov * (n - 1.0) / n;
        assert!((pop_cov - expected_pop_cov).abs() < 1e-10);
    }
    
    #[test]
    fn test_correlation_method_enum() {
        let x = vec64![1, 2, 3, 4, 5];
        let y = vec64![2, 4, 6, 8, 10];
        
        let pearson1 = x.correlation(&y, CorrelationMethod::Pearson);
        let pearson2 = x.pearson_correlation(&y);
        assert!((pearson1 - pearson2).abs() < 1e-10);
        
        let spearman1 = x.correlation(&y, CorrelationMethod::Spearman);
        let spearman2 = x.spearman_correlation(&y);
        assert!((spearman1 - spearman2).abs() < 1e-10);
        
        let kendall1 = x.correlation(&y, CorrelationMethod::Kendall);
        let kendall2 = x.kendall_tau(&y);
        assert!((kendall1 - kendall2).abs() < 1e-10);
    }
    
    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_different_length_vectors() {
        let x = vec64![1, 2, 3];
        let y = vec64![1, 2, 3, 4];
        x.pearson_correlation(&y);
    }
    
    #[test]
    fn test_covariance_matrix() {
        // Create a 4x3 matrix
        let data = vec![
            1.0, 2.0, 3.0,  // Column 1: [1, 4, 7, 10]
            4.0, 5.0, 6.0,  // Column 2: [2, 5, 8, 11] 
            7.0, 8.0, 9.0,  // Column 3: [3, 6, 9, 12]
            10.0, 11.0, 12.0
        ];
        let arr = ArrayF64::from_slice(&data, 4, 3).unwrap();
        
        let cov_matrix = arr.covariance_matrix(false); // Sample covariance
        
        // Should be 3x3 symmetric matrix
        assert_eq!(cov_matrix.shape(), (3, 3));
        
        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                let val_ij = cov_matrix.get(i, j).unwrap();
                let val_ji = cov_matrix.get(j, i).unwrap();
                assert!((val_ij - val_ji).abs() < 1e-10, "Matrix should be symmetric");
            }
        }
        
        // Diagonal elements should be positive (variances)
        for i in 0..3 {
            let diagonal_val = cov_matrix.get(i, i).unwrap();
            assert!(diagonal_val > 0.0, "Diagonal elements should be positive");
        }
    }
    
    #[test]
    fn test_correlation_matrix() {
        // Create a simple 3x2 matrix
        let data = vec![
            1.0, 10.0,  // Column 1: [1, 2, 3], Column 2: [10, 20, 30] (perfect correlation)
            2.0, 20.0,
            3.0, 30.0
        ];
        let arr = ArrayF64::from_slice(&data, 3, 2).unwrap();
        
        let corr_matrix = arr.correlation_matrix(None);
        
        // Should be 2x2 matrix
        assert_eq!(corr_matrix.shape(), (2, 2));
        
        // Diagonal should be 1.0
        assert!((corr_matrix.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((corr_matrix.get(1, 1).unwrap() - 1.0).abs() < 1e-10);
        
        // Off-diagonal should be perfect correlation (1.0)
        assert!((corr_matrix.get(0, 1).unwrap() - 1.0).abs() < 1e-10);
        assert!((corr_matrix.get(1, 0).unwrap() - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_cross_covariance_zero_lag() {
        let x = vec64![1, 2, 3, 4, 5];
        let y = vec64![2, 4, 6, 8, 10];
        
        // Cross-covariance at lag 0 should equal regular covariance
        let cross_cov = x.cross_covariance(&y, 0, false);
        let regular_cov = x.covariance(&y);
        
        assert!((cross_cov - regular_cov).abs() < 1e-10);
    }
    
    #[test]
    fn test_cross_covariance_positive_lag() {
        let x = vec64![1, 0, 0, 0];  // Impulse at start
        let y = vec64![0, 1, 0, 0];  // Impulse at second position
        
        // Should have max cross-covariance at lag +1
        let cross_cov_0 = x.cross_covariance(&y, 0, true);
        let cross_cov_1 = x.cross_covariance(&y, 1, true);
        let cross_cov_neg1 = x.cross_covariance(&y, -1, true);
        
        // Lag +1 should have stronger relationship than lag 0
        assert!(cross_cov_1.abs() > cross_cov_0.abs());
        // And stronger than lag -1
        assert!(cross_cov_1.abs() > cross_cov_neg1.abs());
    }
    
    #[test]
    fn test_cross_covariance_function() {
        let x = vec64![1, 2, 3, 4, 5];
        let y = vec64![5, 4, 3, 2, 1];
        
        let ccf = x.cross_covariance_function(&y, 2, false);
        
        // Should have 2*2+1 = 5 values (lags -2, -1, 0, +1, +2)
        assert_eq!(ccf.len(), 5);
        
        // Middle value (index 2) should be lag 0
        let lag_0_ccf = ccf.get(2).unwrap();
        let lag_0_direct = x.cross_covariance(&y, 0, false);
        assert!((lag_0_ccf - lag_0_direct).abs() < 1e-10);
    }
    
    #[test]
    fn test_cross_covariance_large_lag() {
        let x = vec64![1, 2, 3];
        let y = vec64![4, 5, 6];
        
        // Lag larger than vector size should return 0
        let large_lag_cov = x.cross_covariance(&y, 10, false);
        assert_eq!(large_lag_cov, 0.0);
        
        let negative_large_lag_cov = x.cross_covariance(&y, -10, false);
        assert_eq!(negative_large_lag_cov, 0.0);
    }
}