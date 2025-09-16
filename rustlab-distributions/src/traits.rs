//! Core traits for probability distributions
//!
//! This module defines the fundamental trait system that provides a consistent
//! interface across all probability distributions in the RustLab ecosystem.
//! The trait hierarchy allows for both type safety and performance optimization
//! while maintaining mathematical rigor.
//!
//! ## Trait Hierarchy
//!
//! ```text
//! Distribution (base trait)
//! ├── ContinuousDistribution (for continuous random variables)
//! ├── DiscreteDistribution (for discrete random variables)
//! └── MultivariateDistribution (for vector-valued random variables)
//!
//! Sampling (orthogonal sampling trait)
//!
//! Fitting traits (parameter estimation):
//! ├── MaximumLikelihood
//! └── MethodOfMoments
//! ```
//!
//! ## Design Philosophy
//!
//! - **Type Safety**: Associated types ensure compile-time correctness
//! - **Performance**: Minimal overhead through zero-cost abstractions
//! - **Extensibility**: Easy to implement custom distributions
//! - **Mathematical Accuracy**: Direct mapping from statistical theory
//! - **Ecosystem Integration**: Seamless integration with RustLab-Math when enabled

use crate::error::Result;
use rand::Rng;

#[cfg(feature = "integration")]
use rustlab_math::VectorF64;

/// Base trait for all probability distributions
///
/// This trait defines the fundamental interface that all probability distributions
/// must implement. It provides the core mathematical properties that characterize
/// any probability distribution.
///
/// # Associated Types
///
/// - `Params`: The parameter type(s) for the distribution (e.g., `(f64, f64)` for Normal(μ,σ²))
/// - `Support`: The domain type of the distribution (`f64` for continuous, `i64` for discrete)
///
/// # Mathematical Foundation
///
/// Every probability distribution is characterized by:
/// - **Parameters**: The values that define the specific distribution instance
/// - **Support**: The set of values where the distribution has non-zero density/mass
/// - **Moments**: Mean, variance, and higher-order moments
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::{Distribution, Normal};
///
/// // Create a normal distribution N(0, 1)
/// let normal = Normal::new((0.0, 1.0)).unwrap();
///
/// // Access basic properties
/// assert_eq!(normal.mean(), 0.0);
/// assert_eq!(normal.variance(), 1.0);
/// assert_eq!(normal.std(), 1.0);
///
/// // Access parameters
/// let (mu, sigma) = normal.params();
/// assert_eq!(*mu, 0.0);
/// assert_eq!(*sigma, 1.0);
/// ```
///
/// # Implementation Guidelines
///
/// When implementing this trait:
/// 1. Validate parameters in `new()` and return appropriate errors
/// 2. Store parameters in canonical form (e.g., σ not σ²)
/// 3. Implement moments analytically when possible
/// 4. Handle edge cases gracefully (infinite variance, etc.)
pub trait Distribution {
    /// The type of parameters for this distribution
    ///
    /// This should be the minimal set of parameters needed to fully specify
    /// the distribution. For example:
    /// - Normal: `(f64, f64)` for (μ, σ)
    /// - Gamma: `(f64, f64)` for (shape, rate)
    /// - Binomial: `(u32, f64)` for (n, p)
    type Params;
    
    /// The support (domain) of the distribution
    ///
    /// This specifies the type of values that the distribution can take:
    /// - Continuous distributions: `f64`
    /// - Discrete distributions: `i64` or `u32`
    /// - Multivariate distributions: `VectorF64` (when integration feature is enabled)
    type Support;
    
    /// Create a new distribution with the given parameters
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters defining this distribution instance
    ///
    /// # Returns
    ///
    /// Returns `Ok(distribution)` if parameters are valid, otherwise returns
    /// an appropriate `DistributionError`.
    ///
    /// # Errors
    ///
    /// - `InvalidParameter`: When parameters are outside valid ranges
    /// - `InvalidParameter`: When parameters are NaN or infinite (when not allowed)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Distribution, Normal, DistributionError};
    ///
    /// // Valid parameters
    /// let normal = Normal::new((0.0, 1.0)).unwrap();
    ///
    /// // Invalid parameters (negative standard deviation)
    /// let result = Normal::new((0.0, -1.0));
    /// assert!(result.is_err());
    /// ```
    fn new(params: Self::Params) -> Result<Self> where Self: Sized;
    
    /// Get the parameters of the distribution
    ///
    /// Returns a reference to the parameters used to construct this distribution.
    /// The parameters are guaranteed to be in their canonical form and valid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Distribution, Normal};
    ///
    /// let normal = Normal::new((2.5, 1.5)).unwrap();
    /// let (mu, sigma) = normal.params();
    /// assert_eq!(*mu, 2.5);
    /// assert_eq!(*sigma, 1.5);
    /// ```
    fn params(&self) -> &Self::Params;
    
    /// Mean (expected value) of the distribution
    ///
    /// Computes E[X] where X is a random variable with this distribution.
    /// For continuous distributions: E[X] = ∫ x f(x) dx
    /// For discrete distributions: E[X] = Σ x P(X = x)
    ///
    /// # Returns
    ///
    /// The mean value, or `f64::NAN` if the mean doesn't exist,
    /// or `f64::INFINITY`/`f64::NEG_INFINITY` for unbounded means.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Distribution, Normal, Exponential};
    ///
    /// // Normal distribution: mean = μ
    /// let normal = Normal::new((5.0, 2.0)).unwrap();
    /// assert_eq!(normal.mean(), 5.0);
    ///
    /// // Exponential distribution: mean = 1/λ
    /// let exponential = Exponential::new(0.5).unwrap();
    /// assert_eq!(exponential.mean(), 2.0);
    /// ```
    fn mean(&self) -> f64;
    
    /// Variance of the distribution
    ///
    /// Computes Var[X] = E[(X - E[X])²] where X is a random variable with this distribution.
    /// Equivalently: Var[X] = E[X²] - (E[X])²
    ///
    /// # Returns
    ///
    /// The variance value, or `f64::NAN` if the variance doesn't exist,
    /// or `f64::INFINITY` for infinite variance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Distribution, Normal, Exponential};
    ///
    /// // Normal distribution: variance = σ²
    /// let normal = Normal::new((0.0, 3.0)).unwrap();
    /// assert_eq!(normal.variance(), 9.0);
    ///
    /// // Exponential distribution: variance = 1/λ²
    /// let exponential = Exponential::new(2.0).unwrap();
    /// assert_eq!(exponential.variance(), 0.25);
    /// ```
    fn variance(&self) -> f64;
    
    /// Standard deviation of the distribution
    ///
    /// Computes the standard deviation σ = √Var[X]. This is provided as a convenience
    /// method with a default implementation.
    ///
    /// # Returns
    ///
    /// The standard deviation, or `f64::NAN` if variance doesn't exist.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Distribution, Normal};
    ///
    /// let normal = Normal::new((0.0, 2.5)).unwrap();
    /// assert_eq!(normal.std(), 2.5);
    /// assert_eq!(normal.std(), normal.variance().sqrt());
    /// ```
    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Trait for continuous probability distributions
///
/// This trait extends the base `Distribution` trait to provide functionality
/// specific to continuous random variables, including probability density functions,
/// cumulative distribution functions, and quantile functions.
///
/// # Mathematical Foundation
///
/// A continuous distribution is characterized by:
/// - **PDF**: f(x) such that P(a ≤ X ≤ b) = ∫ᵃᵇ f(x) dx
/// - **CDF**: F(x) = P(X ≤ x) = ∫₋∞ˣ f(t) dt
/// - **Quantile**: F⁻¹(p) such that F(F⁻¹(p)) = p
/// - **Properties**: ∫₋∞^∞ f(x) dx = 1, f(x) ≥ 0
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::{ContinuousDistribution, Normal};
/// use std::f64::consts::PI;
///
/// let normal = Normal::new((0.0, 1.0)).unwrap();
///
/// // PDF at x = 0 for standard normal
/// let pdf_0 = normal.pdf(0.0);
/// assert!((pdf_0 - 1.0 / (2.0 * PI).sqrt()).abs() < 1e-10);
///
/// // CDF at x = 0 for standard normal (should be 0.5)
/// assert!((normal.cdf(0.0) - 0.5).abs() < 1e-10);
///
/// // 95th percentile (approximately 1.645 for standard normal)
/// let q95 = normal.quantile(0.95).unwrap();
/// assert!((q95 - 1.6448536269514729).abs() < 1e-10);
/// ```
///
/// # Implementation Notes
///
/// - PDF implementations should handle edge cases (return 0.0 outside support)
/// - CDF implementations should be monotonically increasing
/// - Quantile implementations should handle boundary cases (p=0, p=1)
/// - Log-PDF should be implemented directly for numerical stability when possible
pub trait ContinuousDistribution: Distribution<Support = f64> {
    /// Probability density function (PDF)
    ///
    /// Computes the probability density f(x) at point x. For continuous distributions,
    /// this represents the "density" of probability at a specific point, not the
    /// probability itself.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the PDF
    ///
    /// # Returns
    ///
    /// The density value f(x) ≥ 0. Returns 0.0 for points outside the support.
    ///
    /// # Mathematical Properties
    ///
    /// - f(x) ≥ 0 for all x
    /// - ∫₋∞^∞ f(x) dx = 1
    /// - P(a ≤ X ≤ b) = ∫ᵃᵇ f(x) dx
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{ContinuousDistribution, Normal, Exponential};
    ///
    /// // Standard normal PDF at x = 0
    /// let normal = Normal::new((0.0, 1.0)).unwrap();
    /// let pdf_0 = normal.pdf(0.0);
    /// assert!((pdf_0 - 0.3989422804014327).abs() < 1e-10);
    ///
    /// // Exponential PDF
    /// let exp_dist = Exponential::new(2.0).unwrap();
    /// assert_eq!(exp_dist.pdf(0.0), 2.0);  // f(0) = λ for exponential
    /// ```
    fn pdf(&self, x: f64) -> f64;
    
    /// Natural logarithm of the probability density function
    ///
    /// Computes ln(f(x)) where f(x) is the PDF. This is often more numerically
    /// stable than computing ln(pdf(x)), especially for distributions with very
    /// small or large PDF values.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the log-PDF
    ///
    /// # Returns
    ///
    /// The natural logarithm of the density, or `-∞` if the PDF is 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{ContinuousDistribution, Normal};
    ///
    /// let normal = Normal::new((0.0, 1.0)).unwrap();
    /// 
    /// // For numerical stability, log_pdf should be implemented directly
    /// let log_pdf = normal.log_pdf(0.0);
    /// let pdf_log = normal.pdf(0.0).ln();
    /// assert!((log_pdf - pdf_log).abs() < 1e-14);
    /// ```
    ///
    /// # Implementation Note
    ///
    /// The default implementation calls `self.pdf(x).ln()`, but distributions
    /// should override this for better numerical stability when possible.
    fn log_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
    
    /// Cumulative distribution function (CDF)
    ///
    /// Computes F(x) = P(X ≤ x), the probability that a random variable X
    /// with this distribution is less than or equal to x.
    ///
    /// # Arguments
    ///
    /// * `x` - The upper bound for the probability calculation
    ///
    /// # Returns
    ///
    /// The cumulative probability F(x) ∈ [0, 1].
    ///
    /// # Mathematical Properties
    ///
    /// - F(x) is non-decreasing: F(a) ≤ F(b) for a ≤ b
    /// - lim_{x→-∞} F(x) = 0
    /// - lim_{x→+∞} F(x) = 1
    /// - F'(x) = f(x) (PDF is derivative of CDF)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{ContinuousDistribution, Normal, Uniform};
    ///
    /// // Standard normal CDF
    /// let normal = Normal::new((0.0, 1.0)).unwrap();
    /// assert!((normal.cdf(0.0) - 0.5).abs() < 1e-10);
    /// assert!((normal.cdf(1.96) - 0.975).abs() < 1e-3);
    ///
    /// // Uniform distribution CDF
    /// let uniform = Uniform::new((0.0, 1.0)).unwrap();
    /// assert_eq!(uniform.cdf(0.5), 0.5);
    /// assert_eq!(uniform.cdf(-1.0), 0.0);
    /// assert_eq!(uniform.cdf(2.0), 1.0);
    /// ```
    fn cdf(&self, x: f64) -> f64;
    
    /// Quantile function (inverse CDF)
    ///
    /// Computes the quantile function F⁻¹(p), which finds the value x such that
    /// F(x) = p. This is the inverse of the CDF.
    ///
    /// # Arguments
    ///
    /// * `p` - The probability level, must be in [0, 1]
    ///
    /// # Returns
    ///
    /// Returns `Ok(x)` such that `cdf(x) = p`, or an error if p is not in [0, 1].
    ///
    /// # Errors
    ///
    /// - `InvalidParameter`: When p is not in the range [0, 1]
    /// - `InvalidParameter`: When p is NaN
    ///
    /// # Mathematical Properties
    ///
    /// - F⁻¹(F(x)) = x (within numerical precision)
    /// - F(F⁻¹(p)) = p for p ∈ [0, 1]
    /// - F⁻¹ is non-decreasing
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{ContinuousDistribution, Normal};
    ///
    /// let normal = Normal::new((0.0, 1.0)).unwrap();
    ///
    /// // Median (50th percentile) should be 0 for standard normal
    /// let median = normal.quantile(0.5).unwrap();
    /// assert!(median.abs() < 1e-10);
    ///
    /// // 95th percentile
    /// let q95 = normal.quantile(0.95).unwrap();
    /// assert!((q95 - 1.6448536269514729).abs() < 1e-8);
    ///
    /// // Error handling
    /// assert!(normal.quantile(-0.1).is_err());
    /// assert!(normal.quantile(1.1).is_err());
    /// ```
    fn quantile(&self, p: f64) -> Result<f64>;
    
    /// Evaluate PDF for an array of values (requires "integration" feature)
    ///
    /// Applies the PDF function element-wise to a vector of input values,
    /// returning a vector of the same length with the corresponding PDF values.
    ///
    /// # Arguments
    ///
    /// * `x` - Vector of points at which to evaluate the PDF
    ///
    /// # Returns
    ///
    /// A new vector containing PDF values for each input point.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use rustlab_distributions::{ContinuousDistribution, Normal};
    /// use rustlab_math::vec64;
    ///
    /// let normal = Normal::new((0.0, 1.0)).unwrap();
    /// let x_values = vec64![-1.0, 0.0, 1.0];
    /// let pdf_values = normal.pdf_array(&x_values);
    ///
    /// assert_eq!(pdf_values.len(), 3);
    /// assert!((pdf_values[1] - normal.pdf(0.0)).abs() < 1e-14);
    /// ```
    #[cfg(feature = "integration")]
    fn pdf_array(&self, x: &VectorF64) -> VectorF64 {
        let values: Vec<f64> = (0..x.len()).map(|i| self.pdf(x.get(i).unwrap())).collect();
        VectorF64::from_vec(values)
    }
    
    /// Evaluate CDF for an array of values (requires "integration" feature)
    ///
    /// Applies the CDF function element-wise to a vector of input values,
    /// returning a vector of the same length with the corresponding CDF values.
    ///
    /// # Arguments
    ///
    /// * `x` - Vector of points at which to evaluate the CDF
    ///
    /// # Returns
    ///
    /// A new vector containing CDF values for each input point.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use rustlab_distributions::{ContinuousDistribution, Normal};
    /// use rustlab_math::vec64;
    ///
    /// let normal = Normal::new((0.0, 1.0)).unwrap();
    /// let x_values = vec64![-2.0, -1.0, 0.0, 1.0, 2.0];
    /// let cdf_values = normal.cdf_array(&x_values);
    ///
    /// assert_eq!(cdf_values.len(), 5);
    /// assert!((cdf_values[2] - 0.5).abs() < 1e-10); // CDF(0) = 0.5 for standard normal
    /// ```
    #[cfg(feature = "integration")]
    fn cdf_array(&self, x: &VectorF64) -> VectorF64 {
        let values: Vec<f64> = (0..x.len()).map(|i| self.cdf(x.get(i).unwrap())).collect();
        VectorF64::from_vec(values)
    }
}

/// Trait for discrete probability distributions
pub trait DiscreteDistribution: Distribution<Support = i64> {
    /// Probability mass function (PMF)
    fn pmf(&self, k: i64) -> f64;
    
    /// Natural logarithm of the PMF
    fn log_pmf(&self, k: i64) -> f64 {
        self.pmf(k).ln()
    }
    
    /// Cumulative distribution function (CDF)
    fn cdf(&self, k: i64) -> f64;
    
    /// Quantile function (inverse CDF)
    fn quantile(&self, p: f64) -> Result<i64>;
}

/// Trait for sampling from distributions
pub trait Sampling: Distribution {
    /// Sample a single value from the distribution
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Support;
    
    /// Sample n values from the distribution (feature-gated)
    #[cfg(feature = "integration")]
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> VectorF64;
    
    /// Sample values into a pre-allocated array (feature-gated)
    #[cfg(feature = "integration")]
    fn sample_into<R: Rng>(&self, rng: &mut R, output: &mut VectorF64);
}

/// Trait for multivariate distributions
pub trait MultivariateDistribution {
    /// Dimensionality of the distribution
    fn dim(&self) -> usize;
    
    /// Mean vector (feature-gated)
    #[cfg(feature = "integration")]
    fn mean_vector(&self) -> VectorF64;
    
    /// Covariance matrix (feature-gated)
    #[cfg(feature = "integration")]
    fn covariance_matrix(&self) -> VectorF64;
}

/// Trait for distributions that support maximum likelihood estimation (feature-gated)
#[cfg(feature = "integration")]
pub trait MaximumLikelihood: Distribution {
    /// Estimate parameters from data using maximum likelihood
    fn fit(data: &VectorF64) -> Result<Self::Params>;
}

/// Trait for distributions that support method of moments estimation (feature-gated)
#[cfg(feature = "integration")]
pub trait MethodOfMoments: Distribution {
    /// Estimate parameters from data using method of moments
    fn fit_moments(data: &VectorF64) -> Result<Self::Params>;
}