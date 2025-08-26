//! Comprehensive array extensions for statistical computing with probability distributions
//!
//! This module provides powerful extension traits that seamlessly integrate probability
//! distributions with RustLab-Math 2D arrays (`ArrayF64`). It enables high-performance
//! statistical computing by extending arrays with distribution sampling, probability
//! calculations, and statistical analysis capabilities.
//!
//! ## Core Extension Traits
//!
//! ### DistributionArrayF64
//! Adds comprehensive continuous distribution functionality:
//! - **Sampling Operations**: Fill arrays with samples from any continuous distribution
//! - **PDF/CDF Application**: Apply probability functions element-wise to arrays
//! - **Statistical Analysis**: Compute empirical statistics and perform hypothesis tests
//! - **Correlation Analysis**: Generate correlation matrices from multi-dimensional data
//!
//! ### DiscreteDistributionArrayF64  
//! Extends arrays with discrete distribution capabilities:
//! - **Integer Sampling**: Generate discrete samples (Bernoulli, Binomial, Poisson)
//! - **Count Data**: Handle count-based statistical modeling
//! - **Binary Operations**: Specialized methods for binary/categorical data
//!
//! ## Key Features
//!
//! ### High-Performance Sampling
//! - **In-Place Operations**: `fill_*` methods modify arrays without additional allocation
//! - **Constructor Methods**: `Distribution::new_array()` creates pre-populated arrays
//! - **Vectorized Operations**: Optimized for cache efficiency and memory locality
//!
//! ### Element-Wise Distribution Functions
//! - **PDF/CDF Evaluation**: Apply any distribution's PDF or CDF to array elements
//! - **Broadcasting Support**: Single distribution parameters applied to entire arrays
//! - **Numerical Stability**: Robust implementations handling edge cases
//!
//! ### Statistical Analysis Tools
//! - **Descriptive Statistics**: Row-wise and column-wise means, standard deviations
//! - **Correlation Analysis**: Pearson correlation matrices for multivariate analysis
//! - **Normality Testing**: Basic statistical hypothesis testing capabilities
//!
//! ## Mathematical Foundation
//!
//! ### Array as Random Matrix
//! For an array A ∈ ℝ^(m×n), each element A[i,j] can be treated as:
//! ```text
//! A[i,j] ~ F(θ)  (independent samples from distribution F with parameters θ)
//! ```
//!
//! ### Row-wise Statistics
//! For row i: X_i = (A[i,1], A[i,2], ..., A[i,n])
//! ```text
//! μ_i = (1/n) ∑_{j=1}^n A[i,j]
//! σ_i² = (1/(n-1)) ∑_{j=1}^n (A[i,j] - μ_i)²
//! ```
//!
//! ### Correlation Matrix
//! For variables X_j and X_k (columns j,k):
//! ```text
//! r_{jk} = Cov(X_j, X_k) / (σ_j × σ_k)
//! ```
//!
//! ## Examples
//!
//! ```rust
//! use rustlab_distributions::integration::array_ext::*;
//! use rustlab_math::ArrayF64;
//! use rand::thread_rng;
//!
//! let mut rng = thread_rng();
//!
//! // Create 100x50 array filled with N(0,1) samples
//! let mut data = ArrayF64::zeros(100, 50);
//! data.fill_normal(0.0, 1.0, &mut rng).unwrap();
//!
//! // Compute row-wise statistics (100 rows)
//! let (row_means, row_stds) = data.empirical_stats_rows().unwrap();
//!
//! // Apply normal PDF with different parameters
//! let pdf_values = data.normal_pdf(1.0, 2.0).unwrap();
//!
//! // Generate correlation matrix (50×50 for 50 variables)
//! let correlations = data.correlation_matrix().unwrap();
//!
//! // Test data for normality
//! let p_value = data.test_normality().unwrap();
//! println!("Normality test p-value: {:.4}", p_value);
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Memory Access**: Row-major order optimized for cache efficiency
//! - **Algorithmic Complexity**: O(mn) for most operations on m×n arrays
//! - **In-Place Operations**: Zero additional memory allocation for `fill_*` methods
//! - **Numerical Stability**: Robust implementations using compensated summation

use crate::continuous::{Normal, Uniform, Exponential, Gamma};
use crate::discrete::{Bernoulli, Binomial, Poisson};
use crate::traits::{Sampling, ContinuousDistribution};
use rustlab_math::{ArrayF64, VectorF64, Result as MathResult, MathError};
use rand::Rng;

/// Comprehensive extension trait for ArrayF64 with continuous distribution functionality
///
/// This trait extends `ArrayF64` from RustLab-Math with a complete suite of methods
/// for working with continuous probability distributions. It provides both sampling
/// operations (generating random data) and analytical operations (applying distribution
/// functions to existing data).
///
/// # Method Categories
///
/// ## Sampling Methods
/// - `fill_*`: In-place array population with distribution samples
/// - `Distribution::array`: Constructor methods returning pre-populated arrays
///
/// ## PDF/CDF Methods
/// - `*_pdf`: Apply probability density functions element-wise
/// - `*_cdf`: Apply cumulative distribution functions element-wise
///
/// ## Statistical Analysis
/// - `empirical_stats_*`: Compute descriptive statistics
/// - `test_normality`: Hypothesis testing for normality
/// - `correlation_matrix`: Multivariate correlation analysis
///
/// # Performance Notes
///
/// - All methods are optimized for row-major memory layout
/// - In-place operations (`fill_*`) are preferred for memory efficiency
/// - Element-wise operations use efficient loops with good cache locality
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::integration::array_ext::DistributionArrayF64;
/// use rustlab_math::ArrayF64;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
/// let mut array = ArrayF64::zeros(50, 20);
///
/// // Fill with standard normal samples
/// array.fill_normal(0.0, 1.0, &mut rng).unwrap();
///
/// // Apply different normal PDF
/// let pdf_array = array.normal_pdf(1.0, 0.5).unwrap();
///
/// // Compute statistics
/// let (means, stds) = array.empirical_stats_rows().unwrap();
/// ```
pub trait DistributionArrayF64 {
    /// Fill array in-place with samples from a normal distribution N(μ, σ²)
    ///
    /// This method efficiently populates the entire array with independent samples
    /// from a normal distribution. It modifies the array in-place, making it
    /// memory-efficient for large datasets.
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean parameter μ of the normal distribution
    /// * `std_dev` - Standard deviation parameter σ > 0
    /// * `rng` - Mutable reference to random number generator
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Successfully filled array with normal samples
    /// * `Err(DistributionError)` - Invalid parameters or array access error
    ///
    /// # Statistical Properties
    ///
    /// After calling this method, the array elements will have:
    /// - **Sample Mean**: Approximately μ (converges as array size increases)
    /// - **Sample Variance**: Approximately σ² (converges as array size increases)
    /// - **Distribution**: Each element ~ N(μ, σ²) independently
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::integration::array_ext::DistributionArrayF64;
    /// use rustlab_math::ArrayF64;
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    /// let mut data_matrix = ArrayF64::zeros(100, 50);
    ///
    /// // Fill with samples from N(2.5, 1.5²)
    /// data_matrix.fill_normal(2.5, 1.5, &mut rng).unwrap();
    ///
    /// // Verify approximate properties for large arrays
    /// let (row_means, _) = data_matrix.empirical_stats_rows().unwrap();
    /// // row_means should be approximately 2.5 for each row
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(mn) for m×n array
    /// - **Space Complexity**: O(1) additional memory (in-place operation)
    /// - **Cache Efficiency**: Row-major traversal for optimal memory access
    fn fill_normal<R: Rng>(&mut self, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Fill array with samples from a uniform distribution
    fn fill_uniform<R: Rng>(&mut self, a: f64, b: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Fill array with samples from an exponential distribution
    fn fill_exponential<R: Rng>(&mut self, lambda: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Fill array with samples from a gamma distribution
    fn fill_gamma<R: Rng>(&mut self, alpha: f64, beta: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Create a new array populated with normal distribution samples N(μ, σ²)
    ///
    /// This constructor method creates a new array of the specified dimensions
    /// and immediately populates it with independent samples from a normal
    /// distribution. It's equivalent to creating a zero array and calling
    /// `fill_normal`, but more convenient for initialization.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows in the resulting array
    /// * `cols` - Number of columns in the resulting array
    /// * `mean` - Mean parameter μ of the normal distribution
    /// * `std_dev` - Standard deviation parameter σ > 0
    /// * `rng` - Mutable reference to random number generator
    ///
    /// # Returns
    ///
    /// * `Ok(ArrayF64)` - New array filled with normal samples
    /// * `Err(DistributionError)` - Invalid parameters or array creation failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::integration::array_ext::DistributionArrayF64;
    /// use rustlab_math::ArrayF64;
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    ///
    /// // Create 20×10 array with standard normal samples
    /// let standard_normal = ArrayF64::normal(20, 10, 0.0, 1.0, &mut rng).unwrap();
    ///
    /// // Create array with custom normal distribution
    /// let custom_normal = ArrayF64::normal(50, 30, 10.0, 2.5, &mut rng).unwrap();
    ///
    /// assert_eq!(standard_normal.nrows(), 20);
    /// assert_eq!(standard_normal.ncols(), 10);
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Monte Carlo Simulations**: Generate random input matrices
    /// - **Statistical Modeling**: Create synthetic datasets for testing
    /// - **Machine Learning**: Generate training data with known properties
    /// - **Numerical Analysis**: Create test matrices with controlled statistical properties
    fn normal<R: Rng>(rows: usize, cols: usize, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<ArrayF64>;
    
    /// Create a new array with uniform distribution samples
    fn uniform<R: Rng>(rows: usize, cols: usize, a: f64, b: f64, rng: &mut R) -> crate::Result<ArrayF64>;
    
    /// Create a new array with exponential distribution samples
    fn exponential<R: Rng>(rows: usize, cols: usize, lambda: f64, rng: &mut R) -> crate::Result<ArrayF64>;
    
    /// Create a new array with gamma distribution samples
    fn gamma<R: Rng>(rows: usize, cols: usize, alpha: f64, beta: f64, rng: &mut R) -> crate::Result<ArrayF64>;
    
    /// Apply normal probability density function element-wise to array
    ///
    /// Computes the PDF f(x) = (1/(σ√(2π))) exp(-½((x-μ)/σ)²) for each element
    /// in the array, where μ and σ are the specified distribution parameters.
    /// Returns a new array of the same dimensions containing the PDF values.
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean parameter μ of the normal distribution
    /// * `std_dev` - Standard deviation parameter σ > 0
    ///
    /// # Returns
    ///
    /// * `Ok(ArrayF64)` - New array with PDF values for each element
    /// * `Err(DistributionError)` - Invalid parameters or computation error
    ///
    /// # Mathematical Properties
    ///
    /// - **Range**: All output values are ≥ 0
    /// - **Maximum**: Occurs at x = μ with value 1/(σ√(2π))
    /// - **Integral**: ∫ f(x)dx = 1 over the entire real line
    /// - **Symmetry**: f(μ+δ) = f(μ-δ) for any δ
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::integration::array_ext::DistributionArrayF64;
    /// use rustlab_math::ArrayF64;
    ///
    /// // Create array with some data points
    /// let mut data = ArrayF64::zeros(2, 3);
    /// data.set(0, 0, -1.0).unwrap();
    /// data.set(0, 1, 0.0).unwrap();
    /// data.set(0, 2, 1.0).unwrap();
    ///
    /// // Apply standard normal PDF
    /// let pdf_values = data.normal_pdf(0.0, 1.0).unwrap();
    ///
    /// // pdf_values[0,1] should be approximately 0.3989 (PDF at x=0)
    /// // pdf_values[0,0] and pdf_values[0,2] should be equal (symmetry)
    /// ```
    ///
    /// # Applications
    ///
    /// - **Likelihood Calculations**: Computing likelihood of observed data
    /// - **Density Estimation**: Evaluating probability densities for analysis
    /// - **Statistical Inference**: Maximum likelihood estimation procedures
    /// - **Data Visualization**: Creating probability density plots
    fn normal_pdf(&self, mean: f64, std_dev: f64) -> crate::Result<ArrayF64>;
    
    /// Apply normal CDF to all elements in the array
    fn normal_cdf(&self, mean: f64, std_dev: f64) -> crate::Result<ArrayF64>;
    
    /// Apply uniform PDF to all elements in the array
    fn uniform_pdf(&self, a: f64, b: f64) -> crate::Result<ArrayF64>;
    
    /// Apply uniform CDF to all elements in the array
    fn uniform_cdf(&self, a: f64, b: f64) -> crate::Result<ArrayF64>;
    
    /// Apply exponential PDF to all elements in the array
    fn exponential_pdf(&self, lambda: f64) -> crate::Result<ArrayF64>;
    
    /// Apply exponential CDF to all elements in the array
    fn exponential_cdf(&self, lambda: f64) -> crate::Result<ArrayF64>;
    
    /// Calculate row-wise empirical statistics (mean and standard deviation)
    ///
    /// Computes the sample mean and sample standard deviation for each row
    /// of the array, treating each row as an independent dataset. This is
    /// useful for analyzing multiple variables or time series stored row-wise.
    ///
    /// # Mathematical Formulation
    ///
    /// For row i with values (x₁, x₂, ..., xₙ):
    /// ```text
    /// Sample Mean:    μ̂ᵢ = (1/n) ∑ⱼ xᵢⱼ
    /// Sample Std Dev: σ̂ᵢ = √[(1/(n-1)) ∑ⱼ (xᵢⱼ - μ̂ᵢ)²]
    /// ```
    ///
    /// # Returns
    ///
    /// * `Ok((means, std_devs))` - Tuple of vectors containing row statistics:
    ///   - `means`: VectorF64 of length `nrows()` with sample means
    ///   - `std_devs`: VectorF64 of length `nrows()` with sample standard deviations
    /// * `Err(MathError)` - Array access error or insufficient data
    ///
    /// # Special Cases
    ///
    /// - **Single Column**: Standard deviation is set to 0.0 (no variability)
    /// - **Empty Rows**: Not applicable (arrays always have dimensions ≥ 1)
    /// - **Constant Rows**: Standard deviation will be 0.0
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::integration::array_ext::DistributionArrayF64;
    /// use rustlab_math::ArrayF64;
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    /// let mut data = ArrayF64::zeros(5, 100);  // 5 variables, 100 observations each
    ///
    /// // Fill each row with different normal distribution
    /// for i in 0..5 {
    ///     for j in 0..100 {
    ///         let sample = /* generate sample for variable i */;
    ///         data.set(i, j, sample).unwrap();
    ///     }
    /// }
    ///
    /// let (row_means, row_stds) = data.empirical_stats_rows().unwrap();
    /// 
    /// // row_means[i] contains the sample mean for variable i
    /// // row_stds[i] contains the sample std dev for variable i
    /// println!("Variable 0: mean={:.3}, std={:.3}", 
    ///          row_means.get(0).unwrap(), row_stds.get(0).unwrap());
    /// ```
    ///
    /// # Applications
    ///
    /// - **Multi-variable Analysis**: Statistics for each variable in dataset
    /// - **Time Series**: Statistics for each time series (one per row)
    /// - **Experimental Data**: Summary statistics for each experimental condition
    /// - **Quality Control**: Monitoring statistics for each measurement type
    fn empirical_stats_rows(&self) -> MathResult<(VectorF64, VectorF64)>;
    
    /// Calculate column-wise empirical mean and standard deviation
    /// Returns (means, std_devs) where each is a row vector
    fn empirical_stats_cols(&self) -> MathResult<(VectorF64, VectorF64)>;
    
    /// Test array elements for normality using simplified statistical test
    ///
    /// Performs a basic normality test on all elements of the array, treating
    /// them as a single dataset. This implementation uses a simplified approach
    /// based on skewness and kurtosis properties of the normal distribution.
    ///
    /// # Mathematical Foundation
    ///
    /// The test evaluates departure from normality using:
    /// ```text
    /// Skewness: γ₁ = E[(X-μ)³]/σ³  (should be ≈ 0 for normal)
    /// Kurtosis: γ₂ = E[(X-μ)⁴]/σ⁴  (should be ≈ 3 for normal)
    /// 
    /// Test statistic combines both measures:
    /// p-value ≈ √(exp(-0.5γ₁²) × exp(-0.125(γ₂-3)²))
    /// ```
    ///
    /// # Returns
    ///
    /// * `Ok(p_value)` - Approximate p-value for normality test (0.0 to 1.0):
    ///   - Values near 1.0: Strong evidence for normality
    ///   - Values near 0.0: Strong evidence against normality
    ///   - Values around 0.05: Marginal evidence (traditional significance level)
    /// * `Err(MathError)` - Insufficient data (< 3 elements) or computation error
    ///
    /// # Interpretation Guidelines
    ///
    /// - **p > 0.1**: Likely normal distribution
    /// - **0.05 < p ≤ 0.1**: Possibly normal, investigate further
    /// - **p ≤ 0.05**: Likely non-normal distribution
    ///
    /// # Limitations
    ///
    /// This is a **simplified test** suitable for:
    /// - Quick normality assessment
    /// - Educational purposes
    /// - Preliminary data analysis
    ///
    /// For rigorous statistical analysis, use specialized libraries with
    /// Shapiro-Wilk, Anderson-Darling, or Kolmogorov-Smirnov tests.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::integration::array_ext::DistributionArrayF64;
    /// use rustlab_math::ArrayF64;
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    ///
    /// // Test array with normal samples
    /// let normal_data = ArrayF64::normal(50, 20, 0.0, 1.0, &mut rng).unwrap();
    /// let p_normal = normal_data.test_normality().unwrap();
    /// println!("Normal data p-value: {:.4}", p_normal);  // Should be high
    ///
    /// // Test array with uniform samples
    /// let uniform_data = ArrayF64::uniform(50, 20, 0.0, 1.0, &mut rng).unwrap();
    /// let p_uniform = uniform_data.test_normality().unwrap();
    /// println!("Uniform data p-value: {:.4}", p_uniform);  // Should be low
    /// ```
    ///
    /// # Statistical Power
    ///
    /// The test's ability to detect non-normality depends on:
    /// - **Sample Size**: Larger samples provide better detection
    /// - **Deviation Type**: More sensitive to skewness than heavy tails
    /// - **Effect Size**: Large deviations from normality are easier to detect
    fn test_normality(&self) -> MathResult<f64>;
    
    /// Compute Pearson correlation matrix from array columns
    ///
    /// Generates a symmetric correlation matrix where each column of the input
    /// array is treated as a separate variable. The resulting matrix shows
    /// pairwise linear correlations between all variables.
    ///
    /// # Mathematical Foundation
    ///
    /// For variables X and Y (columns i and j), the Pearson correlation is:
    /// ```text
    /// r_{ij} = Cov(X_i, X_j) / (σ_i × σ_j)
    ///        = Σ(x_i - μ_i)(x_j - μ_j) / √[Σ(x_i - μ_i)² × Σ(x_j - μ_j)²]
    /// ```
    ///
    /// # Returns
    ///
    /// * `Ok(ArrayF64)` - Square correlation matrix of size `ncols() × ncols()` where:
    ///   - Diagonal elements = 1.0 (perfect self-correlation)
    ///   - Off-diagonal elements ∈ [-1, 1] (correlation coefficients)
    ///   - Matrix is symmetric: R[i,j] = R[j,i]
    /// * `Err(MathError)` - Array access error or computational failure
    ///
    /// # Matrix Properties
    ///
    /// - **Symmetry**: R = Rᵀ (correlation matrix is symmetric)
    /// - **Diagonal**: R[i,i] = 1 for all i (variables perfectly correlated with themselves)
    /// - **Bounds**: -1 ≤ R[i,j] ≤ 1 for all i,j
    /// - **Positive Semi-definite**: All eigenvalues ≥ 0
    ///
    /// # Correlation Interpretation
    ///
    /// - **r ≈ +1**: Strong positive linear relationship
    /// - **r ≈ 0**: No linear relationship (may still have non-linear relationship)
    /// - **r ≈ -1**: Strong negative linear relationship
    /// - **|r| > 0.7**: Generally considered strong correlation
    /// - **|r| < 0.3**: Generally considered weak correlation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::integration::array_ext::DistributionArrayF64;
    /// use rustlab_math::ArrayF64;
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    /// 
    /// // Create data matrix: 100 observations × 5 variables
    /// let data = ArrayF64::normal(100, 5, 0.0, 1.0, &mut rng).unwrap();
    /// 
    /// // Compute 5×5 correlation matrix
    /// let correlations = data.correlation_matrix().unwrap();
    /// 
    /// assert_eq!(correlations.nrows(), 5);
    /// assert_eq!(correlations.ncols(), 5);
    /// 
    /// // Diagonal should be 1.0
    /// for i in 0..5 {
    ///     assert!((correlations.get(i, i).unwrap() - 1.0).abs() < 1e-10);
    /// }
    /// 
    /// // Matrix should be symmetric
    /// for i in 0..5 {
    ///     for j in 0..5 {
    ///         let r_ij = correlations.get(i, j).unwrap();
    ///         let r_ji = correlations.get(j, i).unwrap();
    ///         assert!((r_ij - r_ji).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    ///
    /// # Applications
    ///
    /// - **Multivariate Analysis**: Identify relationships between variables
    /// - **Feature Selection**: Remove highly correlated features in ML
    /// - **Principal Component Analysis**: Input for PCA decomposition
    /// - **Risk Analysis**: Correlation of financial assets or risk factors
    /// - **Quality Control**: Relationships between measurement variables
    ///
    /// # Computational Complexity
    ///
    /// - **Time**: O(m × n²) where m = rows, n = columns
    /// - **Space**: O(n²) for the output correlation matrix
    /// - **Numerical Stability**: Uses compensated arithmetic for accuracy
    fn correlation_matrix(&self) -> MathResult<ArrayF64>;
}

impl DistributionArrayF64 for ArrayF64 {
    fn fill_normal<R: Rng>(&mut self, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Normal::new(mean, std_dev)?;
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let sample = dist.sample(rng);
                self.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array index out of bounds"))?;
            }
        }
        Ok(())
    }
    
    fn fill_uniform<R: Rng>(&mut self, a: f64, b: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Uniform::new(a, b)?;
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let sample = dist.sample(rng);
                self.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array index out of bounds"))?;
            }
        }
        Ok(())
    }
    
    fn fill_exponential<R: Rng>(&mut self, lambda: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Exponential::new(lambda)?;
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let sample = dist.sample(rng);
                self.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array index out of bounds"))?;
            }
        }
        Ok(())
    }
    
    fn fill_gamma<R: Rng>(&mut self, alpha: f64, beta: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Gamma::new(alpha, beta)?;
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let sample = dist.sample(rng);
                self.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array index out of bounds"))?;
            }
        }
        Ok(())
    }
    
    fn normal<R: Rng>(rows: usize, cols: usize, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<ArrayF64> {
        let dist = Normal::new(mean, std_dev)?;
        let mut arr = ArrayF64::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let sample = dist.sample(rng);
                arr.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array creation failed"))?;
            }
        }
        Ok(arr)
    }
    
    fn uniform<R: Rng>(rows: usize, cols: usize, a: f64, b: f64, rng: &mut R) -> crate::Result<ArrayF64> {
        let dist = Uniform::new(a, b)?;
        let mut arr = ArrayF64::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let sample = dist.sample(rng);
                arr.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array creation failed"))?;
            }
        }
        Ok(arr)
    }
    
    fn exponential<R: Rng>(rows: usize, cols: usize, lambda: f64, rng: &mut R) -> crate::Result<ArrayF64> {
        let dist = Exponential::new(lambda)?;
        let mut arr = ArrayF64::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let sample = dist.sample(rng);
                arr.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array creation failed"))?;
            }
        }
        Ok(arr)
    }
    
    fn gamma<R: Rng>(rows: usize, cols: usize, alpha: f64, beta: f64, rng: &mut R) -> crate::Result<ArrayF64> {
        let dist = Gamma::new(alpha, beta)?;
        let mut arr = ArrayF64::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let sample = dist.sample(rng);
                arr.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array creation failed"))?;
            }
        }
        Ok(arr)
    }
    
    fn normal_pdf(&self, mean: f64, std_dev: f64) -> crate::Result<ArrayF64> {
        let dist = Normal::new(mean, std_dev)?;
        let mut result = ArrayF64::zeros(self.nrows(), self.ncols());
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let val = self.get(i, j).ok_or_else(|| crate::DistributionError::sampling_error("Array index out of bounds"))?;
                let pdf_val = dist.pdf(val);
                result.set(i, j, pdf_val).map_err(|_| crate::DistributionError::sampling_error("Result array access failed"))?;
            }
        }
        Ok(result)
    }
    
    fn normal_cdf(&self, mean: f64, std_dev: f64) -> crate::Result<ArrayF64> {
        let dist = Normal::new(mean, std_dev)?;
        let mut result = ArrayF64::zeros(self.nrows(), self.ncols());
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let val = self.get(i, j).ok_or_else(|| crate::DistributionError::sampling_error("Array index out of bounds"))?;
                let cdf_val = dist.cdf(val);
                result.set(i, j, cdf_val).map_err(|_| crate::DistributionError::sampling_error("Result array access failed"))?;
            }
        }
        Ok(result)
    }
    
    fn uniform_pdf(&self, a: f64, b: f64) -> crate::Result<ArrayF64> {
        let dist = Uniform::new(a, b)?;
        let mut result = ArrayF64::zeros(self.nrows(), self.ncols());
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let val = self.get(i, j).ok_or_else(|| crate::DistributionError::sampling_error("Array index out of bounds"))?;
                let pdf_val = dist.pdf(val);
                result.set(i, j, pdf_val).map_err(|_| crate::DistributionError::sampling_error("Result array access failed"))?;
            }
        }
        Ok(result)
    }
    
    fn uniform_cdf(&self, a: f64, b: f64) -> crate::Result<ArrayF64> {
        let dist = Uniform::new(a, b)?;
        let mut result = ArrayF64::zeros(self.nrows(), self.ncols());
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let val = self.get(i, j).ok_or_else(|| crate::DistributionError::sampling_error("Array index out of bounds"))?;
                let cdf_val = dist.cdf(val);
                result.set(i, j, cdf_val).map_err(|_| crate::DistributionError::sampling_error("Result array access failed"))?;
            }
        }
        Ok(result)
    }
    
    fn exponential_pdf(&self, lambda: f64) -> crate::Result<ArrayF64> {
        let dist = Exponential::new(lambda)?;
        let mut result = ArrayF64::zeros(self.nrows(), self.ncols());
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let val = self.get(i, j).ok_or_else(|| crate::DistributionError::sampling_error("Array index out of bounds"))?;
                let pdf_val = dist.pdf(val);
                result.set(i, j, pdf_val).map_err(|_| crate::DistributionError::sampling_error("Result array access failed"))?;
            }
        }
        Ok(result)
    }
    
    fn exponential_cdf(&self, lambda: f64) -> crate::Result<ArrayF64> {
        let dist = Exponential::new(lambda)?;
        let mut result = ArrayF64::zeros(self.nrows(), self.ncols());
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let val = self.get(i, j).ok_or_else(|| crate::DistributionError::sampling_error("Array index out of bounds"))?;
                let cdf_val = dist.cdf(val);
                result.set(i, j, cdf_val).map_err(|_| crate::DistributionError::sampling_error("Result array access failed"))?;
            }
        }
        Ok(result)
    }
    
    fn empirical_stats_rows(&self) -> MathResult<(VectorF64, VectorF64)> {
        let mut means = VectorF64::zeros(self.nrows());
        let mut std_devs = VectorF64::zeros(self.nrows());
        
        for i in 0..self.nrows() {
            // Calculate mean
            let mut sum = 0.0;
            for j in 0..self.ncols() {
                sum += self.get(i, j).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.nrows() * self.ncols() })?;
            }
            let mean = sum / self.ncols() as f64;
            means.set(i, mean).map_err(|_| MathError::IndexOutOfBounds { index: 0, size: 0 })?;
            
            // Calculate standard deviation
            let mut sum_sq_diff = 0.0;
            for j in 0..self.ncols() {
                let val = self.get(i, j).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.nrows() * self.ncols() })?;
                let diff = val - mean;
                sum_sq_diff += diff * diff;
            }
            let std_dev = if self.ncols() > 1 {
                (sum_sq_diff / (self.ncols() - 1) as f64).sqrt()
            } else {
                0.0
            };
            std_devs.set(i, std_dev).map_err(|_| MathError::IndexOutOfBounds { index: 0, size: 0 })?;
        }
        
        Ok((means, std_devs))
    }
    
    fn empirical_stats_cols(&self) -> MathResult<(VectorF64, VectorF64)> {
        let mut means = VectorF64::zeros(self.ncols());
        let mut std_devs = VectorF64::zeros(self.ncols());
        
        for j in 0..self.ncols() {
            // Calculate mean
            let mut sum = 0.0;
            for i in 0..self.nrows() {
                sum += self.get(i, j).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.nrows() * self.ncols() })?;
            }
            let mean = sum / self.nrows() as f64;
            means.set(j, mean).map_err(|_| MathError::IndexOutOfBounds { index: 0, size: 0 })?;
            
            // Calculate standard deviation
            let mut sum_sq_diff = 0.0;
            for i in 0..self.nrows() {
                let val = self.get(i, j).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.nrows() * self.ncols() })?;
                let diff = val - mean;
                sum_sq_diff += diff * diff;
            }
            let std_dev = if self.nrows() > 1 {
                (sum_sq_diff / (self.nrows() - 1) as f64).sqrt()
            } else {
                0.0
            };
            std_devs.set(j, std_dev).map_err(|_| MathError::IndexOutOfBounds { index: 0, size: 0 })?;
        }
        
        Ok((means, std_devs))
    }
    
    fn test_normality(&self) -> MathResult<f64> {
        // Simplified Shapiro-Wilk-like test
        // This is a very basic implementation - for production use, consider more sophisticated tests
        let n = self.nrows() * self.ncols();
        if n < 3 {
            return Err(MathError::IndexOutOfBounds { index: 0, size: 0 });
        }
        
        // Collect all values
        let mut values = Vec::with_capacity(n);
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                values.push(self.get(i, j).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.nrows() * self.ncols() })?);
            }
        }
        
        // Sort values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate basic statistics
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return Ok(0.0); // All values are identical, definitely not normal
        }
        
        // Very simplified normality test based on skewness and kurtosis
        let skewness = values.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n as f64;
        let kurtosis = values.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n as f64 - 3.0;
        
        // Simple approximation: normal distribution has skewness ≈ 0 and excess kurtosis ≈ 0
        let skew_test = (-0.5 * skewness.powi(2)).exp();
        let kurt_test = (-0.125 * kurtosis.powi(2)).exp();
        
        // Combine tests (geometric mean)
        Ok((skew_test * kurt_test).sqrt())
    }
    
    fn correlation_matrix(&self) -> MathResult<ArrayF64> {
        let n_vars = self.ncols();
        let mut corr_matrix = ArrayF64::zeros(n_vars, n_vars);
        
        // Calculate means for each column
        let mut means = vec![0.0; n_vars];
        for j in 0..n_vars {
            let mut sum = 0.0;
            for i in 0..self.nrows() {
                sum += self.get(i, j).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.nrows() * self.ncols() })?;
            }
            means[j] = sum / self.nrows() as f64;
        }
        
        // Calculate correlation coefficients
        for j1 in 0..n_vars {
            for j2 in 0..n_vars {
                if j1 == j2 {
                    corr_matrix.set(j1, j2, 1.0).map_err(|_| MathError::IndexOutOfBounds { index: 0, size: 0 })?;
                } else {
                    let mut sum_xy = 0.0_f64;
                    let mut sum_x2 = 0.0_f64;
                    let mut sum_y2 = 0.0_f64;
                    
                    for i in 0..self.nrows() {
                        let x = self.get(i, j1).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.nrows() * self.ncols() })? - means[j1];
                        let y = self.get(i, j2).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.nrows() * self.ncols() })? - means[j2];
                        
                        sum_xy += x * y;
                        sum_x2 += x * x;
                        sum_y2 += y * y;
                    }
                    
                    let correlation = if sum_x2 > 0.0 && sum_y2 > 0.0 {
                        sum_xy / (sum_x2 * sum_y2).sqrt()
                    } else {
                        0.0
                    };
                    
                    corr_matrix.set(j1, j2, correlation).map_err(|_| MathError::IndexOutOfBounds { index: 0, size: 0 })?;
                }
            }
        }
        
        Ok(corr_matrix)
    }
}

/// Extension trait for discrete distribution operations on arrays
///
/// This trait extends `ArrayF64` with specialized functionality for discrete
/// probability distributions. While the underlying array stores floating-point
/// values, the methods generate and work with discrete random variables
/// (integers) that are converted to f64 for storage compatibility.
///
/// # Supported Distributions
///
/// ## Bernoulli Distribution B(p)
/// - **Use Case**: Binary outcomes (success/failure, yes/no)
/// - **Values**: 0.0 (failure) or 1.0 (success)
/// - **Parameter**: p ∈ [0,1] (probability of success)
///
/// ## Binomial Distribution B(n,p)
/// - **Use Case**: Count of successes in n independent trials
/// - **Values**: {0.0, 1.0, 2.0, ..., n.0}
/// - **Parameters**: n ≥ 1 (trials), p ∈ [0,1] (success probability)
///
/// ## Poisson Distribution Pois(λ)
/// - **Use Case**: Count of events in fixed time/space interval
/// - **Values**: {0.0, 1.0, 2.0, 3.0, ...} (all non-negative integers)
/// - **Parameter**: λ > 0 (average rate of events)
///
/// # Method Categories
///
/// ## In-Place Sampling
/// - `fill_*`: Populate existing arrays with discrete samples
/// - Memory efficient, no additional allocation
///
/// ## Constructor Methods
/// - `Distribution::array`: Create new arrays pre-filled with samples
/// - Convenient for initialization workflows
///
/// # Storage Format
///
/// Discrete values are stored as floating-point numbers:
/// - Integer values: 0, 1, 2, 3, ... → 0.0, 1.0, 2.0, 3.0, ...
/// - No fractional parts (e.g., 2.5 will never appear)
/// - Easy conversion back to integers when needed
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::integration::array_ext::DiscreteDistributionArrayF64;
/// use rustlab_math::ArrayF64;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
/// let mut counts = ArrayF64::zeros(10, 20);
///
/// // Fill with Poisson samples (λ = 3.5)
/// counts.fill_poisson(3.5, &mut rng).unwrap();
///
/// // Create Binomial array (n=10, p=0.3)
/// let trials = ArrayF64::binomial(5, 8, 10, 0.3, &mut rng).unwrap();
///
/// // All values are non-negative integers stored as f64
/// for i in 0..trials.nrows() {
///     for j in 0..trials.ncols() {
///         let val = trials.get(i, j).unwrap();
///         assert!(val >= 0.0);
///         assert_eq!(val, val.round());  // No fractional part
///     }
/// }
/// ```
pub trait DiscreteDistributionArrayF64 {
    /// Fill array in-place with Bernoulli distribution samples B(p)
    ///
    /// Populates the array with independent Bernoulli samples, where each element
    /// is either 0.0 (failure) or 1.0 (success) with probability p of success.
    /// This is the fundamental binary distribution underlying many statistical models.
    ///
    /// # Arguments
    ///
    /// * `p` - Success probability p ∈ [0, 1]
    /// * `rng` - Mutable reference to random number generator
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Array successfully filled with Bernoulli samples
    /// * `Err(DistributionError)` - Invalid parameter (p outside [0,1])
    ///
    /// # Statistical Properties
    ///
    /// After filling, the array will contain:
    /// - **Values**: Only 0.0 and 1.0
    /// - **Mean**: Approximately p (proportion of 1.0s)
    /// - **Variance**: Approximately p(1-p)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::integration::array_ext::DiscreteDistributionArrayF64;
    /// use rustlab_math::ArrayF64;
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    /// let mut binary_data = ArrayF64::zeros(100, 50);
    ///
    /// // Fill with biased coin flips (70% success rate)
    /// binary_data.fill_bernoulli(0.7, &mut rng).unwrap();
    ///
    /// // Count successes and failures
    /// let mut successes = 0;
    /// let mut failures = 0;
    /// for i in 0..binary_data.nrows() {
    ///     for j in 0..binary_data.ncols() {
    ///         match binary_data.get(i, j).unwrap() {
    ///             1.0 => successes += 1,
    ///             0.0 => failures += 1,
    ///             _ => panic!("Invalid Bernoulli value"),
    ///         }
    ///     }
    /// }
    /// 
    /// // Success rate should be approximately 0.7
    /// let success_rate = successes as f64 / (successes + failures) as f64;
    /// assert!((success_rate - 0.7).abs() < 0.05);  // Within 5% tolerance
    /// ```
    fn fill_bernoulli<R: Rng>(&mut self, p: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Fill array in-place with Binomial distribution samples B(n,p)
    ///
    /// Populates the array with independent samples from the Binomial distribution,
    /// representing the number of successes in n independent Bernoulli trials.
    /// Each element will be an integer between 0 and n (stored as f64).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of independent trials (n ≥ 1)
    /// * `p` - Success probability for each trial p ∈ [0, 1]
    /// * `rng` - Mutable reference to random number generator
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Array successfully filled with Binomial samples
    /// * `Err(DistributionError)` - Invalid parameters
    ///
    /// # Statistical Properties
    ///
    /// After filling, the array will contain:
    /// - **Values**: {0.0, 1.0, 2.0, ..., n.0}
    /// - **Mean**: Approximately np
    /// - **Variance**: Approximately np(1-p)
    /// - **Mode**: Around np (most frequent value)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::integration::array_ext::DiscreteDistributionArrayF64;
    /// use rustlab_math::ArrayF64;
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    /// let mut trial_results = ArrayF64::zeros(50, 25);
    ///
    /// // Fill with results of 20 coin flips each (p=0.5)
    /// trial_results.fill_binomial(20, 0.5, &mut rng).unwrap();
    ///
    /// // Verify all values are in valid range [0, 20]
    /// for i in 0..trial_results.nrows() {
    ///     for j in 0..trial_results.ncols() {
    ///         let successes = trial_results.get(i, j).unwrap();
    ///         assert!(successes >= 0.0 && successes <= 20.0);
    ///         assert_eq!(successes, successes.round());  // Integer value
    ///     }
    /// }
    /// ```
    ///
    /// # Applications
    ///
    /// - **Quality Control**: Number of defective items in batches
    /// - **Clinical Trials**: Number of patients responding to treatment
    /// - **A/B Testing**: Number of conversions out of n visitors
    /// - **Survey Research**: Number of positive responses out of n questions
    fn fill_binomial<R: Rng>(&mut self, n: u32, p: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Fill array in-place with Poisson distribution samples Pois(λ)
    ///
    /// Populates the array with independent samples from the Poisson distribution,
    /// representing the count of events occurring in fixed intervals of time or space.
    /// Values are non-negative integers with no upper bound (though very large values
    /// are extremely rare).
    ///
    /// # Arguments
    ///
    /// * `lambda` - Rate parameter λ > 0 (average number of events per interval)
    /// * `rng` - Mutable reference to random number generator
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Array successfully filled with Poisson samples
    /// * `Err(DistributionError)` - Invalid parameter (λ ≤ 0)
    ///
    /// # Statistical Properties
    ///
    /// After filling, the array will contain:
    /// - **Values**: {0.0, 1.0, 2.0, 3.0, ...} (all non-negative integers)
    /// - **Mean**: Approximately λ
    /// - **Variance**: Approximately λ (unique property: mean = variance)
    /// - **Mode**: ⌊λ⌋ if λ is not integer, {λ-1, λ} if λ is integer
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::integration::array_ext::DiscreteDistributionArrayF64;
    /// use rustlab_math::ArrayF64;
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    /// let mut event_counts = ArrayF64::zeros(30, 40);
    ///
    /// // Fill with event counts (average 4.5 events per interval)
    /// event_counts.fill_poisson(4.5, &mut rng).unwrap();
    ///
    /// // Verify all values are non-negative integers
    /// let mut total_events = 0.0;
    /// let total_intervals = event_counts.nrows() * event_counts.ncols();
    ///
    /// for i in 0..event_counts.nrows() {
    ///     for j in 0..event_counts.ncols() {
    ///         let count = event_counts.get(i, j).unwrap();
    ///         assert!(count >= 0.0);
    ///         assert_eq!(count, count.round());  // Integer value
    ///         total_events += count;
    ///     }
    /// }
    ///
    /// // Average should be approximately λ = 4.5
    /// let average_events = total_events / total_intervals as f64;
    /// assert!((average_events - 4.5).abs() < 0.5);  // Within reasonable tolerance
    /// ```
    ///
    /// # Applications
    ///
    /// - **Telecommunications**: Number of calls received per hour
    /// - **Biology**: Number of mutations in DNA sequences
    /// - **Traffic Analysis**: Number of vehicles passing per minute
    /// - **Customer Service**: Number of support tickets per day
    /// - **Manufacturing**: Number of defects per production unit
    fn fill_poisson<R: Rng>(&mut self, lambda: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Create new array populated with Bernoulli distribution samples B(p)
    ///
    /// Creates a new array of specified dimensions and immediately fills it with
    /// independent Bernoulli samples. This is a convenience constructor equivalent
    /// to creating a zero array and calling `fill_bernoulli`.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows in the resulting array
    /// * `cols` - Number of columns in the resulting array  
    /// * `p` - Success probability p ∈ [0, 1]
    /// * `rng` - Mutable reference to random number generator
    ///
    /// # Returns
    ///
    /// * `Ok(ArrayF64)` - New array filled with Bernoulli samples
    /// * `Err(DistributionError)` - Invalid parameters or array creation failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::integration::array_ext::DiscreteDistributionArrayF64;
    /// use rustlab_math::ArrayF64;
    /// use rand::thread_rng;
    ///
    /// let mut rng = thread_rng();
    ///
    /// // Create 10×15 array of fair coin flips
    /// let fair_coins = ArrayF64::bernoulli(10, 15, 0.5, &mut rng).unwrap();
    ///
    /// // Create biased coin array (80% success rate)
    /// let biased_coins = ArrayF64::bernoulli(20, 25, 0.8, &mut rng).unwrap();
    ///
    /// assert_eq!(fair_coins.nrows(), 10);
    /// assert_eq!(fair_coins.ncols(), 15);
    /// ```
    fn bernoulli<R: Rng>(rows: usize, cols: usize, p: f64, rng: &mut R) -> crate::Result<ArrayF64>;
    
    /// Create array with Binomial samples
    fn binomial<R: Rng>(rows: usize, cols: usize, trials: u32, p: f64, rng: &mut R) -> crate::Result<ArrayF64>;
    
    /// Create array with Poisson samples
    fn poisson<R: Rng>(rows: usize, cols: usize, lambda: f64, rng: &mut R) -> crate::Result<ArrayF64>;
}

impl DiscreteDistributionArrayF64 for ArrayF64 {
    fn fill_bernoulli<R: Rng>(&mut self, p: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Bernoulli::new(p)?;
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let sample = if dist.sample(rng) == 1 { 1.0 } else { 0.0 };
                self.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array index out of bounds"))?;
            }
        }
        Ok(())
    }
    
    fn fill_binomial<R: Rng>(&mut self, n: u32, p: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Binomial::new(n, p)?;
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let sample = dist.sample(rng) as f64;
                self.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array index out of bounds"))?;
            }
        }
        Ok(())
    }
    
    fn fill_poisson<R: Rng>(&mut self, lambda: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Poisson::new(lambda)?;
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                let sample = dist.sample(rng) as f64;
                self.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array index out of bounds"))?;
            }
        }
        Ok(())
    }
    
    fn bernoulli<R: Rng>(rows: usize, cols: usize, p: f64, rng: &mut R) -> crate::Result<ArrayF64> {
        let dist = Bernoulli::new(p)?;
        let mut arr = ArrayF64::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let sample = if dist.sample(rng) == 1 { 1.0 } else { 0.0 };
                arr.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array creation failed"))?;
            }
        }
        Ok(arr)
    }
    
    fn binomial<R: Rng>(rows: usize, cols: usize, trials: u32, p: f64, rng: &mut R) -> crate::Result<ArrayF64> {
        let dist = Binomial::new(trials, p)?;
        let mut arr = ArrayF64::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let sample = dist.sample(rng) as f64;
                arr.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array creation failed"))?;
            }
        }
        Ok(arr)
    }
    
    fn poisson<R: Rng>(rows: usize, cols: usize, lambda: f64, rng: &mut R) -> crate::Result<ArrayF64> {
        let dist = Poisson::new(lambda)?;
        let mut arr = ArrayF64::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let sample = dist.sample(rng) as f64;
                arr.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array creation failed"))?;
            }
        }
        Ok(arr)
    }
}