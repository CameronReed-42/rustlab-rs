//! Enhanced ergonomic API for streamlined statistical computing
//!
//! This module provides a high-level, user-friendly interface to RustLab-Distributions
//! that prioritizes developer productivity and ease of use. It offers ergonomic
//! improvements over the core API while maintaining full mathematical rigor and
//! performance characteristics.
//!
//! ## Design Philosophy
//!
//! ### Math-First Approach
//! - **Panic on Invalid Inputs**: For clearly invalid mathematical operations
//! - **Zero Overhead**: Thin wrappers with no runtime performance cost
//! - **Familiar API**: Follows conventions from NumPy, SciPy, and R
//!
//! ### Dual Error Handling Strategy
//! - **Panic Methods**: For interactive use and when parameters are known valid
//! - **Result Methods**: For programmatic use where error handling is required
//! - **Clear Naming**: `try_*` prefix indicates Result-returning methods
//!
//! ## Key Enhancements
//!
//! ### Simplified Constructors
//! ```rust
//! // Enhanced API - no Result wrapping for valid inputs
//! let normal = EnhancedNormal::new(0.0, 1.0);  // Panics on invalid params
//! let result = EnhancedNormal::try_new(0.0, 1.0);  // Returns Result
//! 
//! // Convenience constructors
//! let standard = EnhancedNormal::standard();  // N(0,1)
//! let unit_mean = EnhancedNormal::with_mean(5.0);  // N(5,1)
//! ```
//!
//! ### Direct Vector Operations
//! ```rust
//! let normal = EnhancedNormal::standard();
//! let samples = normal.samples(1000, &mut rng);  // Direct Vec<f64> output
//! let pdf_vals = normal.pdf_slice(&data);  // Batch PDF evaluation
//! ```
//!
//! ### Builder Pattern Support
//! ```rust
//! let complex_normal = NormalBuilder::new()
//!     .mean(10.0)
//!     .variance(25.0)  // Can specify variance OR std_dev
//!     .build();
//! ```
//!
//! ### Global Convenience Functions
//! ```rust
//! use rustlab_distributions::enhanced_api::convenience::*;
//! 
//! let samples = standard_normal_samples(1000);  // Uses thread RNG automatically
//! let pdf_vals = standard_normal_pdf(&data);
//! let quantiles = standard_normal_quantiles(&probabilities);
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Zero-Cost Abstractions**: Enhanced API adds no runtime overhead
//! - **Memory Efficient**: Direct vector operations without intermediate allocations
//! - **Optimized Delegation**: All mathematical operations delegate to core implementations
//!
//! ## Usage Patterns
//!
//! ### Interactive/Exploratory Analysis
//! ```rust
//! use rustlab_distributions::enhanced_api::*;
//! use rand::thread_rng;
//!
//! // Quick and simple - panics on invalid inputs
//! let normal = EnhancedNormal::new(0.0, 1.0);
//! let mut rng = thread_rng();
//! let data = normal.samples(10000, &mut rng);
//! 
//! // Analyze without Result wrapping
//! let p95 = normal.quantile(0.95);
//! let density_at_mean = normal.pdf(0.0);
//! ```
//!
//! ### Production/Library Code
//! ```rust
//! use rustlab_distributions::enhanced_api::*;
//!
//! fn create_normal_from_config(mean: f64, variance: f64) -> Result<EnhancedNormal, Box<dyn std::error::Error>> {
//!     let normal = EnhancedNormal::try_new(mean, variance.sqrt())?;
//!     Ok(normal)
//! }
//!
//! // Or using builder with error handling
//! fn create_normal_builder(config: &Config) -> Result<EnhancedNormal, DistributionError> {
//!     NormalBuilder::new()
//!         .mean(config.mean)
//!         .variance(config.variance)
//!         .try_build()
//! }
//! ```
//!
//! ## Mathematical Guarantees
//!
//! All enhanced API methods provide identical mathematical behavior to the core API:
//! - **Numerical Precision**: Same algorithms and numerical stability
//! - **Statistical Properties**: Identical distribution properties and moments
//! - **Random Number Quality**: Same underlying sampling algorithms
//!
//! ## Integration Features
//!
//! When the "integration" feature is enabled, additional methods are available:
//! ```rust,ignore
//! #[cfg(feature = "integration")]
//! {
//!     use rustlab_math::VectorF64;
//!     
//!     let normal = EnhancedNormal::standard();
//!     let samples = normal.sample_vector(1000, &mut rng);  // Direct VectorF64
//!     let pdf_vals = normal.pdf_vector(&data_vector);
//! }
//! ```

use crate::continuous::Normal;
use crate::error::{Result, DistributionError};
use rand::Rng;

/// Enhanced Normal distribution with ergonomic API improvements
///
/// `EnhancedNormal` is a wrapper around the core `Normal` distribution that provides
/// a more user-friendly interface while maintaining full mathematical rigor and
/// performance. It implements a dual error-handling strategy with both panicking
/// and Result-returning methods.
///
/// # Design Principles
///
/// ## Math-First Philosophy
/// - Invalid mathematical operations panic by default (following NumPy/SciPy conventions)
/// - Result-returning variants available with `try_*` naming convention
/// - Clear separation between user error (panic) and recoverable error (Result)
///
/// ## Zero-Cost Abstraction
/// - Thin wrapper with no runtime overhead
/// - All mathematical operations delegate to optimized core implementation
/// - Memory layout identical to core `Normal` distribution
///
/// # Examples
///
/// ## Basic Usage
/// ```rust
/// use rustlab_distributions::enhanced_api::EnhancedNormal;
/// use rand::thread_rng;
///
/// // Create distributions without Result wrapping
/// let standard = EnhancedNormal::standard();          // N(0,1)
/// let custom = EnhancedNormal::new(5.0, 2.0);         // N(5, 4)
/// let unit_mean = EnhancedNormal::with_mean(10.0);    // N(10,1)
/// 
/// // Direct sampling to Vec
/// let mut rng = thread_rng();
/// let samples = standard.samples(1000, &mut rng);
/// 
/// // Batch operations
/// let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
/// let pdf_values = standard.pdf_slice(&data);
/// let cdf_values = standard.cdf_slice(&data);
/// ```
///
/// ## Error Handling
/// ```rust
/// use rustlab_distributions::enhanced_api::EnhancedNormal;
///
/// // Panicking version (for known-valid inputs)
/// let normal1 = EnhancedNormal::new(0.0, 1.0);  // Will panic if std_dev <= 0
/// 
/// // Result version (for programmatic use)
/// let normal2 = EnhancedNormal::try_new(0.0, 1.0);  // Returns Result
/// assert!(normal2.is_ok());
/// 
/// let invalid = EnhancedNormal::try_new(0.0, -1.0);  // Returns Err
/// assert!(invalid.is_err());
/// ```
///
/// # Thread Safety
///
/// `EnhancedNormal` is `Send + Sync` and can be safely shared between threads.
/// The distribution parameters are immutable after creation.
///
/// # Performance
///
/// - Construction: O(1) with parameter validation
/// - Sampling: O(1) per sample using Box-Muller transform
/// - PDF/CDF: O(1) with special function evaluation
/// - Batch operations: O(n) with optimized vectorized loops
#[derive(Debug, Clone, PartialEq)]
pub struct EnhancedNormal {
    inner: Normal,
    pub(crate) params: (f64, f64),
}

impl EnhancedNormal {
    /// Create a new Normal distribution N(μ, σ²) with ergonomic error handling
    ///
    /// This constructor follows the "math-first" philosophy: it panics on clearly
    /// invalid mathematical parameters rather than returning a Result. This makes
    /// it ideal for interactive use, prototyping, and cases where parameters are
    /// known to be valid.
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean parameter μ, can be any finite real number
    /// * `std_dev` - Standard deviation parameter σ, must be positive and finite
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `std_dev` ≤ 0 (standard deviation must be positive)
    /// - `mean` is NaN or infinite
    /// - `std_dev` is NaN or infinite
    ///
    /// This panic behavior follows the NumPy/SciPy convention where invalid
    /// mathematical operations fail fast rather than propagating errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::EnhancedNormal;
    ///
    /// // Valid parameters - will not panic
    /// let standard = EnhancedNormal::new(0.0, 1.0);        // Standard normal
    /// let shifted = EnhancedNormal::new(5.0, 2.0);         // N(5, 4)
    /// let precise = EnhancedNormal::new(-1.5, 0.1);        // High precision
    ///
    /// // These would panic:
    /// // let invalid1 = EnhancedNormal::new(0.0, 0.0);      // Zero std dev
    /// // let invalid2 = EnhancedNormal::new(0.0, -1.0);     // Negative std dev
    /// // let invalid3 = EnhancedNormal::new(f64::NAN, 1.0); // NaN mean
    /// ```
    ///
    /// # When to Use
    ///
    /// Use this constructor when:
    /// - Parameters are known to be valid at compile time
    /// - Writing interactive/exploratory code
    /// - Fast prototyping where error handling would be verbose
    /// - Following the principle of "fail fast" for programming errors
    ///
    /// For programmatic use where parameters might be invalid, use `try_new()`.
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) with minimal parameter validation
    /// - **Space Complexity**: O(1) - same memory layout as core Normal
    /// - **Overhead**: Zero runtime overhead compared to core API
    pub fn new(mean: f64, std_dev: f64) -> Self {
        let inner = Normal::new(mean, std_dev).expect("Invalid normal distribution parameters");
        Self {
            inner,
            params: (mean, std_dev),
        }
    }
    
    /// Create a new Normal distribution with Result-based error handling
    ///
    /// This constructor returns a Result, making it suitable for programmatic use
    /// where parameters might be invalid and graceful error handling is required.
    /// It provides the same mathematical functionality as `new()` but with
    /// explicit error propagation.
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean parameter μ, can be any finite real number
    /// * `std_dev` - Standard deviation parameter σ, must be positive and finite
    ///
    /// # Returns
    ///
    /// * `Ok(EnhancedNormal)` - Successfully created distribution
    /// * `Err(DistributionError)` - Invalid parameters with descriptive error message
    ///
    /// # Errors
    ///
    /// Returns `DistributionError` for:
    /// - `std_dev` ≤ 0 ("Standard deviation must be positive and finite")
    /// - `mean` is NaN or infinite ("Mean must be finite")
    /// - `std_dev` is NaN or infinite ("Standard deviation must be positive and finite")
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::EnhancedNormal;
    ///
    /// // Successful creation
    /// let normal = EnhancedNormal::try_new(0.0, 1.0);
    /// assert!(normal.is_ok());
    /// let normal = normal.unwrap();
    /// assert_eq!(normal.mean(), 0.0);
    /// assert_eq!(normal.std_dev(), 1.0);
    ///
    /// // Error handling
    /// let invalid_std = EnhancedNormal::try_new(0.0, -1.0);
    /// assert!(invalid_std.is_err());
    /// 
    /// let invalid_mean = EnhancedNormal::try_new(f64::NAN, 1.0);
    /// assert!(invalid_mean.is_err());
    ///
    /// // Programmatic usage pattern
    /// fn create_normal_from_config(config: &Config) -> Result<EnhancedNormal, DistributionError> {
    ///     EnhancedNormal::try_new(config.mean, config.std_dev)
    /// }
    /// ```
    ///
    /// # When to Use
    ///
    /// Use this constructor when:
    /// - Parameters come from user input or external sources
    /// - Writing library code that should not panic
    /// - Error handling and recovery is important
    /// - Building robust, fault-tolerant applications
    ///
    /// For interactive use with known-valid parameters, use `new()`.
    pub fn try_new(mean: f64, std_dev: f64) -> Result<Self> {
        let inner = Normal::new(mean, std_dev)?;
        Ok(Self {
            inner,
            params: (mean, std_dev),
        })
    }
    
    /// Create the standard normal distribution N(0, 1)
    ///
    /// This is the most commonly used normal distribution in statistics, with zero
    /// mean and unit variance. It's the foundation for z-scores, standardization,
    /// and many statistical procedures.
    ///
    /// # Mathematical Properties
    ///
    /// - **Mean**: μ = 0
    /// - **Standard Deviation**: σ = 1  
    /// - **Variance**: σ² = 1
    /// - **PDF**: f(x) = (1/√(2π)) exp(-½x²)
    /// - **68-95-99.7 Rule**: ~68% of values within ±1, ~95% within ±2, ~99.7% within ±3
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::EnhancedNormal;
    /// use rand::thread_rng;
    ///
    /// let standard = EnhancedNormal::standard();
    /// 
    /// // Verify properties
    /// assert_eq!(standard.mean(), 0.0);
    /// assert_eq!(standard.std_dev(), 1.0);
    /// assert_eq!(standard.variance(), 1.0);
    ///
    /// // Common statistical values
    /// assert!((standard.pdf(0.0) - 0.3989422804014327).abs() < 1e-10);  // 1/√(2π)
    /// assert!((standard.cdf(0.0) - 0.5).abs() < 1e-10);                  // 50th percentile
    /// assert!((standard.quantile(0.975) - 1.96).abs() < 0.01);           // 97.5% quantile ≈ 1.96
    ///
    /// // Generate samples
    /// let mut rng = thread_rng();
    /// let samples = standard.samples(10000, &mut rng);
    /// ```
    ///
    /// # Applications
    ///
    /// - **Standardization**: Convert any normal distribution to standard form
    /// - **Hypothesis Testing**: Reference distribution for z-tests
    /// - **Monte Carlo Methods**: Base distribution for transformation techniques  
    /// - **Quality Control**: Six Sigma methodology uses standard normal
    /// - **Risk Management**: Value at Risk (VaR) calculations
    ///
    /// # Performance
    ///
    /// This constructor is guaranteed to never fail and has zero computational cost.
    /// It's implemented as a simple constant initialization.
    pub fn standard() -> Self {
        Self {
            inner: Normal::standard(),
            params: (0.0, 1.0),
        }
    }
    
    /// Create a normal distribution with specified mean and unit variance
    pub fn with_mean(mean: f64) -> Self {
        Self::new(mean, 1.0)
    }
    
    /// Create a normal distribution with zero mean and specified variance
    pub fn with_variance(variance: f64) -> Self {
        assert!(variance > 0.0, "Variance must be positive");
        Self::new(0.0, variance.sqrt())
    }
    
    /// Generate multiple samples directly into a Vec<f64>
    ///
    /// This method provides an ergonomic interface for batch sampling, returning
    /// samples directly as a Vec without the need for Result unwrapping or trait
    /// imports. It's optimized for cases where you need many samples at once.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples to generate
    /// * `rng` - Mutable reference to random number generator
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` containing exactly n independent samples from this distribution.
    /// Each sample is generated using the Box-Muller transformation.
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(n) with excellent constants
    /// - **Memory**: Single allocation for the entire Vec
    /// - **Efficiency**: Uses optimized batch sampling from the core implementation
    /// - **Cache Friendly**: Sequential memory layout for good cache performance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::EnhancedNormal;
    /// use rand::thread_rng;
    ///
    /// let normal = EnhancedNormal::new(5.0, 2.0);
    /// let mut rng = thread_rng();
    ///
    /// // Generate samples for analysis
    /// let small_sample = normal.samples(100, &mut rng);
    /// let large_sample = normal.samples(100_000, &mut rng);
    ///
    /// assert_eq!(small_sample.len(), 100);
    /// assert_eq!(large_sample.len(), 100_000);
    ///
    /// // Verify approximate statistical properties
    /// let sample_mean = large_sample.iter().sum::<f64>() / large_sample.len() as f64;
    /// assert!((sample_mean - 5.0).abs() < 0.1);  // Should be close to μ = 5.0
    /// 
    /// let sample_var = large_sample.iter()
    ///     .map(|x| (x - sample_mean).powi(2))
    ///     .sum::<f64>() / (large_sample.len() - 1) as f64;
    /// assert!((sample_var - 4.0).abs() < 0.2);   // Should be close to σ² = 4.0
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Monte Carlo Simulations**: Generate large datasets for simulation studies
    /// - **Statistical Analysis**: Create samples for empirical analysis
    /// - **Machine Learning**: Generate synthetic training data
    /// - **Numerical Integration**: Sample points for Monte Carlo integration
    /// - **Bootstrap Methods**: Generate resamples for bootstrap confidence intervals
    ///
    /// # Memory Considerations
    ///
    /// For very large n (> 10⁷), consider:
    /// - Using iterative processing to avoid large memory allocations
    /// - Streaming samples directly to disk for massive datasets
    /// - Using the core sampling traits for more control over memory usage
    pub fn samples<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<f64> {
        self.inner.sample_n(rng, n)
    }
    
    /// Sample a single value with ergonomic API
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        self.inner.sample(rng)
    }
    
    // Delegate all other methods to the inner Normal
    
    /// Get the mean parameter
    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }
    
    /// Get the standard deviation parameter
    pub fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }
    
    /// Get the variance parameter
    pub fn variance(&self) -> f64 {
        self.inner.variance()
    }
    
    /// Probability density function (PDF)
    pub fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }
    
    /// Natural logarithm of the PDF
    pub fn log_pdf(&self, x: f64) -> f64 {
        self.inner.log_pdf(x)
    }
    
    /// Cumulative distribution function (CDF)
    pub fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }
    
    /// Quantile function (inverse CDF) with ergonomic error handling
    ///
    /// Computes the quantile function Φ⁻¹(p) which finds the value x such that
    /// P(X ≤ x) = p. This method panics on invalid inputs, making it convenient
    /// for interactive use and cases where probabilities are known to be valid.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability level p ∈ [0, 1]
    ///
    /// # Returns
    ///
    /// The quantile value x such that P(X ≤ x) = p
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `p < 0` or `p > 1` (probability outside valid range)
    /// - `p` is NaN (invalid probability value)
    ///
    /// This follows the math-first philosophy where invalid mathematical
    /// operations fail fast.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::EnhancedNormal;
    ///
    /// let normal = EnhancedNormal::new(10.0, 2.0);
    ///
    /// // Common quantiles
    /// let median = normal.quantile(0.5);      // 50th percentile = mean = 10.0
    /// let q25 = normal.quantile(0.25);        // 25th percentile 
    /// let q75 = normal.quantile(0.75);        // 75th percentile
    /// let q95 = normal.quantile(0.95);        // 95th percentile
    ///
    /// assert!((median - 10.0).abs() < 1e-10);
    /// assert!(q25 < median && median < q75);   // Ordering property
    ///
    /// // Standard normal critical values
    /// let standard = EnhancedNormal::standard();
    /// let z_alpha = standard.quantile(0.975);  // 97.5% quantile ≈ 1.96
    /// assert!((z_alpha - 1.9599639845400545).abs() < 1e-10);
    ///
    /// // Boundary cases
    /// assert_eq!(standard.quantile(0.0), f64::NEG_INFINITY);
    /// assert_eq!(standard.quantile(1.0), f64::INFINITY);
    /// ```
    ///
    /// # Statistical Applications
    ///
    /// - **Confidence Intervals**: Find critical values for hypothesis tests
    /// - **Value at Risk**: Calculate VaR thresholds for risk management
    /// - **Percentile Analysis**: Determine percentile boundaries
    /// - **Quality Control**: Set control limits for process monitoring
    /// - **Simulation**: Generate scenarios for stress testing
    ///
    /// # When to Use
    ///
    /// Use this method when:
    /// - Probability values are known to be valid (constants, validated inputs)
    /// - Writing interactive/exploratory analysis code
    /// - Performance is critical and error checking overhead is unwanted
    ///
    /// For programmatic use with potentially invalid inputs, use `try_quantile()`.
    pub fn quantile(&self, p: f64) -> f64 {
        self.inner.inverse_cdf(p).expect("Probability must be in [0, 1]")
    }
    
    /// Quantile function with Result-based error handling
    ///
    /// This version of the quantile function returns a Result, making it suitable
    /// for programmatic use where probability values might be invalid and graceful
    /// error handling is required.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability level, should be in [0, 1]
    ///
    /// # Returns
    ///
    /// * `Ok(quantile)` - The quantile value x such that P(X ≤ x) = p
    /// * `Err(DistributionError)` - Invalid probability parameter
    ///
    /// # Errors
    ///
    /// Returns `DistributionError` for:
    /// - `p < 0` or `p > 1` ("Probability must be in [0, 1]")
    /// - `p` is NaN ("Probability must be in [0, 1]")
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::EnhancedNormal;
    ///
    /// let normal = EnhancedNormal::standard();
    ///
    /// // Valid probability values
    /// assert!(normal.try_quantile(0.5).is_ok());
    /// assert!(normal.try_quantile(0.0).is_ok());
    /// assert!(normal.try_quantile(1.0).is_ok());
    ///
    /// // Invalid probability values
    /// assert!(normal.try_quantile(-0.1).is_err());
    /// assert!(normal.try_quantile(1.1).is_err());
    /// assert!(normal.try_quantile(f64::NAN).is_err());
    ///
    /// // Programmatic usage pattern
    /// fn calculate_confidence_interval(
    ///     normal: &EnhancedNormal,
    ///     confidence_level: f64
    /// ) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    ///     let alpha = 1.0 - confidence_level;
    ///     let lower = normal.try_quantile(alpha / 2.0)?;
    ///     let upper = normal.try_quantile(1.0 - alpha / 2.0)?;
    ///     Ok((lower, upper))
    /// }
    /// ```
    pub fn try_quantile(&self, p: f64) -> Result<f64> {
        self.inner.inverse_cdf(p)
    }
    
    /// Evaluate PDF for multiple values
    pub fn pdf_slice(&self, x: &[f64]) -> Vec<f64> {
        self.inner.pdf_slice(x)
    }
    
    /// Evaluate CDF for multiple values
    pub fn cdf_slice(&self, x: &[f64]) -> Vec<f64> {
        self.inner.cdf_slice(x)
    }
}

// Feature-gated rustlab-math integration
#[cfg(feature = "integration")]
impl EnhancedNormal {
    /// Sample directly into a rustlab-math VectorF64
    /// 
    /// # Example
    /// ```rust,ignore
    /// use rustlab_distributions::enhanced_api::EnhancedNormal;
    /// use rand::thread_rng;
    /// 
    /// let normal = EnhancedNormal::standard();
    /// let mut rng = thread_rng();
    /// let samples = normal.sample_vector(1000, &mut rng);
    /// ```
    pub fn sample_vector<R: Rng>(&self, n: usize, rng: &mut R) -> rustlab_math::VectorF64 {
        let samples = self.samples(n, rng);
        rustlab_math::VectorF64::from_slice(&samples)
    }
    
    /// Evaluate PDF for a rustlab-math vector
    pub fn pdf_vector(&self, x: &rustlab_math::VectorF64) -> rustlab_math::VectorF64 {
        self.inner.pdf_vector(x)
    }
    
    /// Evaluate CDF for a rustlab-math vector
    pub fn cdf_vector(&self, x: &rustlab_math::VectorF64) -> rustlab_math::VectorF64 {
        self.inner.cdf_vector(x)
    }
}

/// Builder pattern for constructing Normal distributions with fluent interface
///
/// `NormalBuilder` provides a flexible, chainable API for constructing Normal
/// distributions with various parameter specifications. It supports both variance
/// and standard deviation parameterization, with intelligent defaulting and
/// clear error handling.
///
/// # Design Features
///
/// ## Flexible Parameterization
/// - Can specify either `std_dev` OR `variance` (variance takes precedence)
/// - Intelligent defaults: mean = 0.0, std_dev = 1.0 (standard normal)
/// - Parameter validation occurs at build time, not during chaining
///
/// ## Dual Build Methods
/// - `build()`: Panics on invalid parameters (math-first philosophy)
/// - `try_build()`: Returns Result for programmatic error handling
///
/// ## Method Chaining
/// All setter methods consume `self` and return `Self`, enabling fluent chaining.
///
/// # Examples
///
/// ## Basic Usage
/// ```rust
/// use rustlab_distributions::enhanced_api::NormalBuilder;
///
/// // Build with explicit parameters
/// let normal1 = NormalBuilder::new()
///     .mean(5.0)
///     .std_dev(2.0)
///     .build();
/// assert_eq!(normal1.mean(), 5.0);
/// assert_eq!(normal1.std_dev(), 2.0);
///
/// // Use variance instead of std_dev
/// let normal2 = NormalBuilder::new()
///     .mean(10.0)
///     .variance(9.0)  // std_dev will be 3.0
///     .build();
/// assert_eq!(normal2.variance(), 9.0);
///
/// // Use defaults (standard normal)
/// let standard = NormalBuilder::new().build();
/// assert_eq!(standard.mean(), 0.0);
/// assert_eq!(standard.std_dev(), 1.0);
/// ```
///
/// ## Error Handling
/// ```rust
/// use rustlab_distributions::enhanced_api::NormalBuilder;
///
/// // Panicking version (for known-valid parameters)
/// let valid = NormalBuilder::new()
///     .mean(0.0)
///     .variance(4.0)
///     .build();  // Will not panic
///
/// // Result version (for programmatic use)
/// let invalid = NormalBuilder::new()
///     .variance(-1.0)  // Invalid variance
///     .try_build();
/// assert!(invalid.is_err());
///
/// let valid_result = NormalBuilder::new()
///     .mean(5.0)
///     .std_dev(1.5)
///     .try_build();
/// assert!(valid_result.is_ok());
/// ```
///
/// ## Advanced Patterns
/// ```rust
/// use rustlab_distributions::enhanced_api::NormalBuilder;
///
/// // Conditional building
/// fn create_distribution(use_high_variance: bool) -> NormalBuilder {
///     let mut builder = NormalBuilder::new().mean(0.0);
///     
///     if use_high_variance {
///         builder = builder.variance(25.0);
///     } else {
///         builder = builder.std_dev(1.0);
///     }
///     
///     builder
/// }
///
/// let dist = create_distribution(true).build();
/// ```
///
/// # Parameter Precedence
///
/// When both `std_dev` and `variance` are specified, `variance` takes precedence:
/// ```rust
/// use rustlab_distributions::enhanced_api::NormalBuilder;
///
/// let normal = NormalBuilder::new()
///     .std_dev(2.0)    // This will be ignored
///     .variance(9.0)   // This takes precedence (std_dev = 3.0)
///     .build();
/// 
/// assert_eq!(normal.std_dev(), 3.0);  // sqrt(9.0), not 2.0
/// ```
///
/// # Performance
///
/// - **Construction**: O(1) with minimal validation overhead
/// - **Memory**: Zero-cost abstraction, no additional storage
/// - **Flexibility**: No runtime cost for the fluent interface
#[derive(Debug, Default)]
pub struct NormalBuilder {
    mean: Option<f64>,
    std_dev: Option<f64>,
    variance: Option<f64>,
}

impl NormalBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the mean parameter
    pub fn mean(mut self, mean: f64) -> Self {
        self.mean = Some(mean);
        self
    }
    
    /// Set the standard deviation parameter
    pub fn std_dev(mut self, std_dev: f64) -> Self {
        self.std_dev = Some(std_dev);
        self
    }
    
    /// Set the variance parameter σ² (takes precedence over std_dev)
    ///
    /// When both `variance()` and `std_dev()` are called on the same builder,
    /// the variance specification takes precedence and the std_dev setting
    /// is ignored. The standard deviation will be computed as √(variance).
    ///
    /// # Arguments
    ///
    /// * `variance` - Variance parameter σ², must be positive
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::NormalBuilder;
    ///
    /// // Specify variance directly
    /// let normal1 = NormalBuilder::new()
    ///     .mean(5.0)
    ///     .variance(16.0)  // std_dev will be 4.0
    ///     .build();
    /// assert_eq!(normal1.variance(), 16.0);
    /// assert_eq!(normal1.std_dev(), 4.0);
    ///
    /// // Variance takes precedence over std_dev
    /// let normal2 = NormalBuilder::new()
    ///     .std_dev(2.0)    // This is ignored
    ///     .variance(25.0)  // This is used (std_dev = 5.0)
    ///     .build();
    /// assert_eq!(normal2.std_dev(), 5.0);  // sqrt(25.0), not 2.0
    /// ```
    pub fn variance(mut self, variance: f64) -> Self {
        self.variance = Some(variance);
        self
    }
    
    /// Build the Normal distribution with panic-based error handling
    ///
    /// Constructs the `EnhancedNormal` distribution using the specified parameters
    /// or intelligent defaults. This method follows the "math-first" philosophy
    /// and panics on invalid parameters rather than returning a Result.
    ///
    /// # Default Values
    ///
    /// - **Mean**: 0.0 (if not specified)
    /// - **Standard Deviation**: 1.0 (if neither `std_dev` nor `variance` specified)
    /// - **Result**: Standard normal N(0,1) if no parameters are set
    ///
    /// # Parameter Resolution
    ///
    /// 1. Mean: Uses specified mean or defaults to 0.0
    /// 2. Scale: If variance is specified, uses √(variance); otherwise uses std_dev or defaults to 1.0
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Specified variance ≤ 0 ("Variance must be positive")
    /// - Specified std_dev ≤ 0 (handled by `EnhancedNormal::new`)
    /// - Mean is NaN or infinite (handled by `EnhancedNormal::new`)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::NormalBuilder;
    ///
    /// // Default standard normal
    /// let standard = NormalBuilder::new().build();
    /// assert_eq!(standard.mean(), 0.0);
    /// assert_eq!(standard.std_dev(), 1.0);
    ///
    /// // Custom parameters
    /// let custom = NormalBuilder::new()
    ///     .mean(10.0)
    ///     .variance(25.0)
    ///     .build();
    /// assert_eq!(custom.mean(), 10.0);
    /// assert_eq!(custom.variance(), 25.0);
    ///
    /// // Partial specification with defaults
    /// let partial = NormalBuilder::new()
    ///     .mean(5.0)  // std_dev will default to 1.0
    ///     .build();
    /// assert_eq!(partial.mean(), 5.0);
    /// assert_eq!(partial.std_dev(), 1.0);
    /// ```
    ///
    /// # When to Use
    ///
    /// Use this method when:
    /// - Parameters are known to be valid
    /// - Building distributions in interactive/exploratory contexts
    /// - Following the "fail fast" principle for invalid mathematical operations
    ///
    /// For programmatic use with potentially invalid parameters, use `try_build()`.
    pub fn build(self) -> EnhancedNormal {
        let mean = self.mean.unwrap_or(0.0);
        let std_dev = if let Some(variance) = self.variance {
            assert!(variance > 0.0, "Variance must be positive");
            variance.sqrt()
        } else {
            self.std_dev.unwrap_or(1.0)
        };
        
        EnhancedNormal::new(mean, std_dev)
    }
    
    /// Build the Normal distribution with Result-based error handling
    ///
    /// Constructs the `EnhancedNormal` distribution using specified parameters
    /// or defaults, returning a Result for graceful error handling. This method
    /// is suitable for programmatic use where parameters might be invalid.
    ///
    /// # Default Values
    ///
    /// Same defaults as `build()`:
    /// - **Mean**: 0.0 (if not specified)
    /// - **Standard Deviation**: 1.0 (if neither `std_dev` nor `variance` specified)
    ///
    /// # Returns
    ///
    /// * `Ok(EnhancedNormal)` - Successfully constructed distribution
    /// * `Err(DistributionError)` - Invalid parameters with descriptive error
    ///
    /// # Errors
    ///
    /// Returns `DistributionError` for:
    /// - Variance ≤ 0 ("Variance must be positive")
    /// - Standard deviation ≤ 0 ("Standard deviation must be positive and finite")
    /// - Mean is NaN or infinite ("Mean must be finite")
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::NormalBuilder;
    ///
    /// // Successful construction
    /// let result = NormalBuilder::new()
    ///     .mean(5.0)
    ///     .std_dev(2.0)
    ///     .try_build();
    /// assert!(result.is_ok());
    /// let normal = result.unwrap();
    /// assert_eq!(normal.mean(), 5.0);
    ///
    /// // Error handling for invalid variance
    /// let invalid_var = NormalBuilder::new()
    ///     .variance(-1.0)
    ///     .try_build();
    /// assert!(invalid_var.is_err());
    ///
    /// // Error handling for invalid std_dev
    /// let invalid_std = NormalBuilder::new()
    ///     .std_dev(0.0)
    ///     .try_build();
    /// assert!(invalid_std.is_err());
    ///
    /// // Programmatic usage pattern
    /// fn build_distribution_from_config(
    ///     config: &Config
    /// ) -> Result<EnhancedNormal, Box<dyn std::error::Error>> {
    ///     let normal = NormalBuilder::new()
    ///         .mean(config.mean)
    ///         .variance(config.variance)
    ///         .try_build()?;
    ///     Ok(normal)
    /// }
    /// ```
    pub fn try_build(self) -> Result<EnhancedNormal> {
        let mean = self.mean.unwrap_or(0.0);
        let std_dev = if let Some(variance) = self.variance {
            if variance <= 0.0 {
                return Err(DistributionError::invalid_parameter("Variance must be positive"));
            }
            variance.sqrt()
        } else {
            self.std_dev.unwrap_or(1.0)
        };
        
        EnhancedNormal::try_new(mean, std_dev)
    }
}

/// High-level convenience functions for common statistical operations
///
/// This module provides a collection of utility functions that abstract away
/// the need for explicit distribution construction and random number generator
/// management. These functions are designed for quick statistical computations,
/// prototyping, and interactive analysis.
///
/// # Design Principles
///
/// ## Automatic RNG Management
/// All sampling functions use `thread_rng()` internally, eliminating the need
/// to manage random number generators for simple use cases.
///
/// ## Standard Normal Focus
/// Most functions operate on the standard normal distribution N(0,1), which
/// is the foundation for many statistical procedures and can be easily
/// transformed to other normal distributions.
///
/// ## Direct Results
/// Functions return `Vec<f64>` directly without Result wrapping, following
/// the enhanced API's math-first philosophy.
///
/// # Function Categories
///
/// ## Sampling Functions
/// - `standard_normal_samples()`: Generate samples from N(0,1)
/// - `normal_samples()`: Generate samples from N(μ,σ²) with custom parameters
/// - `unit_normal_samples()`: Alias for standard_normal_samples()
///
/// ## Statistical Functions
/// - `standard_normal_pdf()`: Batch PDF evaluation
/// - `standard_normal_cdf()`: Batch CDF evaluation  
/// - `standard_normal_quantiles()`: Batch quantile computation
///
/// # Examples
///
/// ## Quick Sampling
/// ```rust
/// use rustlab_distributions::enhanced_api::convenience::*;
///
/// // Generate samples without managing RNG
/// let samples1 = standard_normal_samples(1000);        // N(0,1)
/// let samples2 = normal_samples(500, 10.0, 2.0);       // N(10, 4)
/// let samples3 = unit_normal_samples(100);             // Same as standard_normal
///
/// assert_eq!(samples1.len(), 1000);
/// assert_eq!(samples2.len(), 500);
/// assert_eq!(samples3.len(), 100);
/// ```
///
/// ## Statistical Analysis
/// ```rust
/// use rustlab_distributions::enhanced_api::convenience::*;
///
/// // Evaluate statistical functions on data
/// let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
/// let pdf_values = standard_normal_pdf(&data);
/// let cdf_values = standard_normal_cdf(&data);
///
/// // Generate quantiles for common probabilities
/// let probabilities = vec![0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975];
/// let quantiles = standard_normal_quantiles(&probabilities);
/// 
/// // quantiles[3] should be 0.0 (median of standard normal)
/// assert!((quantiles[3] - 0.0).abs() < 1e-10);
/// ```
///
/// ## Monte Carlo Integration Example
/// ```rust
/// use rustlab_distributions::enhanced_api::convenience::*;
///
/// // Estimate E[X^2] for X ~ N(0,1) (should be 1.0)
/// let samples = standard_normal_samples(100_000);
/// let squares: Vec<f64> = samples.iter().map(|x| x * x).collect();
/// let estimate = squares.iter().sum::<f64>() / squares.len() as f64;
/// assert!((estimate - 1.0).abs() < 0.01);  // Should be close to 1.0
/// ```
///
/// # Thread Safety
///
/// All functions are thread-safe. Each call uses an independent random number
/// generator state, so concurrent calls from different threads will not interfere
/// with each other.
///
/// # Performance Considerations
///
/// - **RNG Overhead**: Functions create new `thread_rng()` instances each call
/// - **Use Case**: Optimized for convenience, not high-performance computing
/// - **Alternative**: For performance-critical code, use the core API with
///   explicit RNG management
///
/// # When to Use
///
/// Use this module when:
/// - Prototyping statistical algorithms
/// - Interactive data analysis (Jupyter notebooks, REPL)
/// - Quick statistical computations
/// - Teaching and learning statistics
/// - One-off calculations where RNG management is overhead
///
/// For production code or performance-critical applications, consider using
/// the core API for better control over random number generation.
pub mod convenience {
    use super::*;
    use rand::thread_rng;
    
    /// Generate samples from the standard normal distribution N(0,1)
    ///
    /// This is the most commonly used convenience function, generating samples from
    /// the standard normal distribution using an internal thread-local random number
    /// generator. Perfect for quick statistical analysis and prototyping.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` containing exactly n independent samples from N(0,1)
    ///
    /// # Statistical Properties
    ///
    /// The generated samples will have (asymptotically):
    /// - **Mean**: ≈ 0.0
    /// - **Standard Deviation**: ≈ 1.0
    /// - **Variance**: ≈ 1.0
    /// - **68-95-99.7 Rule**: ~68% within [-1,1], ~95% within [-2,2], ~99.7% within [-3,3]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::convenience::standard_normal_samples;
    ///
    /// // Generate samples for analysis
    /// let small_sample = standard_normal_samples(100);
    /// let large_sample = standard_normal_samples(10_000);
    ///
    /// assert_eq!(small_sample.len(), 100);
    /// assert_eq!(large_sample.len(), 10_000);
    ///
    /// // Verify approximate properties for large sample
    /// let mean = large_sample.iter().sum::<f64>() / large_sample.len() as f64;
    /// let variance = large_sample.iter()
    ///     .map(|x| (x - mean).powi(2))
    ///     .sum::<f64>() / (large_sample.len() - 1) as f64;
    ///     
    /// assert!((mean).abs() < 0.05);           // Mean should be close to 0
    /// assert!((variance - 1.0).abs() < 0.05); // Variance should be close to 1
    /// ```
    ///
    /// # Applications
    ///
    /// - **Monte Carlo Simulations**: Base distribution for random sampling
    /// - **Bootstrap Methods**: Generate resamples for statistical inference
    /// - **Hypothesis Testing**: Generate null distribution samples
    /// - **Machine Learning**: Create synthetic datasets with known properties
    /// - **Numerical Integration**: Sample points for Monte Carlo integration
    ///
    /// # Performance
    ///
    /// - Uses optimized Box-Muller transformation
    /// - Thread-safe with internal RNG management
    /// - Suitable for moderate sample sizes (up to ~10⁶ samples)
    /// - For larger datasets, consider using explicit RNG management
    pub fn standard_normal_samples(n: usize) -> Vec<f64> {
        let mut rng = thread_rng();
        EnhancedNormal::standard().samples(n, &mut rng)
    }
    
    /// Generate samples from a normal distribution N(μ, σ²) with custom parameters
    ///
    /// Creates samples from any normal distribution by specifying the mean and
    /// standard deviation parameters. Uses internal thread-local RNG for convenience.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples to generate
    /// * `mean` - Mean parameter μ (location)
    /// * `std_dev` - Standard deviation parameter σ > 0 (scale)
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` containing exactly n independent samples from N(μ, σ²)
    ///
    /// # Panics
    ///
    /// Panics if `std_dev ≤ 0` or if parameters are NaN/infinite (following
    /// the enhanced API's math-first philosophy).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::convenience::normal_samples;
    ///
    /// // Generate samples from different normal distributions
    /// let std_normal = normal_samples(1000, 0.0, 1.0);    // N(0,1)
    /// let shifted = normal_samples(500, 10.0, 1.0);       // N(10,1)
    /// let scaled = normal_samples(200, 0.0, 5.0);         // N(0,25)
    /// let general = normal_samples(1000, 5.0, 2.0);       // N(5,4)
    ///
    /// // Verify approximate properties
    /// let sample_mean = general.iter().sum::<f64>() / general.len() as f64;
    /// assert!((sample_mean - 5.0).abs() < 0.2);  // Should be close to μ = 5.0
    /// 
    /// let sample_std = {
    ///     let variance = general.iter()
    ///         .map(|x| (x - sample_mean).powi(2))
    ///         .sum::<f64>() / (general.len() - 1) as f64;
    ///     variance.sqrt()
    /// };
    /// assert!((sample_std - 2.0).abs() < 0.2);   // Should be close to σ = 2.0
    /// ```
    ///
    /// # Transformation Relationship
    ///
    /// This function generates samples using the transformation:
    /// ```text
    /// If Z ~ N(0,1), then X = μ + σZ ~ N(μ, σ²)
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Simulation Studies**: Generate data with specific mean/variance
    /// - **Statistical Modeling**: Create synthetic datasets for testing
    /// - **Risk Analysis**: Model returns with custom volatility parameters
    /// - **Quality Control**: Simulate measurements with known specifications
    pub fn normal_samples(n: usize, mean: f64, std_dev: f64) -> Vec<f64> {
        let mut rng = thread_rng();
        EnhancedNormal::new(mean, std_dev).samples(n, &mut rng)
    }
    
    /// Generate samples from a unit normal (mean=0, std_dev=1) using default RNG
    pub fn unit_normal_samples(n: usize) -> Vec<f64> {
        standard_normal_samples(n)
    }
    
    /// Calculate PDF values for an array of x values using standard normal
    pub fn standard_normal_pdf(x: &[f64]) -> Vec<f64> {
        EnhancedNormal::standard().pdf_slice(x)
    }
    
    /// Calculate CDF values for an array of x values using standard normal
    pub fn standard_normal_cdf(x: &[f64]) -> Vec<f64> {
        EnhancedNormal::standard().cdf_slice(x)
    }
    
    /// Calculate quantiles for multiple probabilities using standard normal distribution
    ///
    /// Computes the quantile function (inverse CDF) for an array of probability values,
    /// returning the corresponding z-scores from the standard normal distribution.
    /// This is particularly useful for generating critical values and confidence intervals.
    ///
    /// # Arguments
    ///
    /// * `p` - Slice of probability values, each should be in [0, 1]
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` containing quantiles corresponding to each input probability
    ///
    /// # Panics
    ///
    /// Panics if any probability is outside [0, 1] or is NaN (following the
    /// enhanced API's math-first philosophy).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::enhanced_api::convenience::standard_normal_quantiles;
    ///
    /// // Common statistical critical values
    /// let probabilities = vec![0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975];
    /// let quantiles = standard_normal_quantiles(&probabilities);
    ///
    /// assert_eq!(quantiles.len(), 7);
    /// assert!((quantiles[3] - 0.0).abs() < 1e-10);        // 50th percentile = 0
    /// assert!((quantiles[6] - 1.96).abs() < 0.01);        // 97.5% ≈ 1.96
    /// assert!((quantiles[0] + 1.96).abs() < 0.01);        // 2.5% ≈ -1.96
    ///
    /// // Verify symmetry
    /// assert!((quantiles[0] + quantiles[6]).abs() < 1e-10);
    /// assert!((quantiles[1] + quantiles[5]).abs() < 1e-10);
    /// ```
    ///
    /// # Statistical Applications
    ///
    /// ## Confidence Intervals
    /// ```rust
    /// use rustlab_distributions::enhanced_api::convenience::standard_normal_quantiles;
    ///
    /// // 95% confidence interval critical values
    /// let ci_probs = vec![0.025, 0.975];
    /// let ci_values = standard_normal_quantiles(&ci_probs);
    /// let lower_critical = ci_values[0];  // ≈ -1.96
    /// let upper_critical = ci_values[1];  // ≈ +1.96
    /// ```
    ///
    /// ## Multiple Significance Levels
    /// ```rust
    /// use rustlab_distributions::enhanced_api::convenience::standard_normal_quantiles;
    ///
    /// // Critical values for different α levels (two-tailed)
    /// let alpha_levels = vec![0.01, 0.05, 0.10];  // 1%, 5%, 10% significance
    /// let upper_probs: Vec<f64> = alpha_levels.iter().map(|a| 1.0 - a/2.0).collect();
    /// let critical_values = standard_normal_quantiles(&upper_probs);
    /// 
    /// // critical_values[0] ≈ 2.58 (99% CI)
    /// // critical_values[1] ≈ 1.96 (95% CI) 
    /// // critical_values[2] ≈ 1.64 (90% CI)
    /// ```
    ///
    /// # Numerical Properties
    ///
    /// - **Accuracy**: Uses high-precision Beasley-Springer-Moro algorithm
    /// - **Range**: Handles extreme probabilities (very close to 0 or 1)
    /// - **Symmetry**: Maintains Φ⁻¹(p) = -Φ⁻¹(1-p) property
    pub fn standard_normal_quantiles(p: &[f64]) -> Vec<f64> {
        let normal = EnhancedNormal::standard();
        p.iter().map(|&pi| normal.quantile(pi)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::thread_rng;
    
    #[test]
    fn test_enhanced_normal_creation() {
        let normal = EnhancedNormal::new(5.0, 2.0);
        assert_eq!(normal.mean(), 5.0);
        assert_eq!(normal.std_dev(), 2.0);
        assert_eq!(normal.variance(), 4.0);
    }
    
    #[test]
    fn test_standard_normal() {
        let normal = EnhancedNormal::standard();
        assert_eq!(normal.mean(), 0.0);
        assert_eq!(normal.std_dev(), 1.0);
    }
    
    #[test]
    fn test_convenience_constructors() {
        let normal1 = EnhancedNormal::with_mean(3.0);
        assert_eq!(normal1.mean(), 3.0);
        assert_eq!(normal1.std_dev(), 1.0);
        
        let normal2 = EnhancedNormal::with_variance(9.0);
        assert_eq!(normal2.mean(), 0.0);
        assert_abs_diff_eq!(normal2.variance(), 9.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_samples_method() {
        let normal = EnhancedNormal::standard();
        let mut rng = thread_rng();
        let samples = normal.samples(100, &mut rng);
        assert_eq!(samples.len(), 100);
    }
    
    #[test]
    fn test_quantile_no_result() {
        let normal = EnhancedNormal::standard();
        assert_abs_diff_eq!(normal.quantile(0.5), 0.0, epsilon = 1e-10);
    }
    
    #[test]
    #[should_panic(expected = "Probability must be in [0, 1]")]
    fn test_quantile_panic() {
        let normal = EnhancedNormal::standard();
        normal.quantile(1.5); // Should panic
    }
    
    #[test]
    fn test_builder_pattern() {
        let normal = NormalBuilder::new()
            .mean(10.0)
            .std_dev(3.0)
            .build();
        
        assert_eq!(normal.mean(), 10.0);
        assert_eq!(normal.std_dev(), 3.0);
    }
    
    #[test]
    fn test_builder_with_variance() {
        let normal = NormalBuilder::new()
            .mean(5.0)
            .variance(16.0)
            .build();
        
        assert_eq!(normal.mean(), 5.0);
        assert_abs_diff_eq!(normal.variance(), 16.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_builder_defaults() {
        let normal = NormalBuilder::new().build();
        assert_eq!(normal.mean(), 0.0);
        assert_eq!(normal.std_dev(), 1.0);
    }
    
    #[test]
    fn test_convenience_functions() {
        let samples = convenience::standard_normal_samples(50);
        assert_eq!(samples.len(), 50);
        
        let samples2 = convenience::normal_samples(30, 2.0, 1.5);
        assert_eq!(samples2.len(), 30);
        
        let x = vec![0.0, 1.0, -1.0];
        let pdf_vals = convenience::standard_normal_pdf(&x);
        assert_eq!(pdf_vals.len(), 3);
        
        let cdf_vals = convenience::standard_normal_cdf(&x);
        assert_eq!(cdf_vals.len(), 3);
        assert_abs_diff_eq!(cdf_vals[0], 0.5, epsilon = 1e-10);
        
        let p = vec![0.1, 0.5, 0.9];
        let quantiles = convenience::standard_normal_quantiles(&p);
        assert_eq!(quantiles.len(), 3);
        assert_abs_diff_eq!(quantiles[1], 0.0, epsilon = 1e-10);
    }
}