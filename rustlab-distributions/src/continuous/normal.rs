//! Normal (Gaussian) distribution implementation
//!
//! This module provides a comprehensive, high-precision implementation of the normal
//! (Gaussian) distribution, which is arguably the most important probability distribution
//! in statistics and natural sciences.
//!
//! ## Mathematical Background
//!
//! The normal distribution is a continuous probability distribution characterized by
//! its bell-shaped curve. It is parameterized by two parameters:
//! - **μ (mean)**: The location parameter, determining the center of the distribution
//! - **σ (standard deviation)**: The scale parameter, determining the spread
//!
//! ## Key Properties
//!
//! - **PDF**: f(x) = (1/(σ√(2π))) × exp(-½((x-μ)/σ)²)
//! - **CDF**: F(x) = ½(1 + erf((x-μ)/(σ√2)))
//! - **Mean**: E[X] = μ
//! - **Variance**: Var[X] = σ²
//! - **Skewness**: 0 (perfectly symmetric)
//! - **Kurtosis**: 3 (mesokurtic)
//!
//! ## Implementation Details
//!
//! - **High Precision**: Uses `rustlab-special::erf` for CDF calculations
//! - **Numerical Stability**: Direct log-PDF computation avoids overflow
//! - **Efficient Sampling**: Box-Muller transformation with spare value caching
//! - **Quantile Function**: Beasley-Springer-Moro algorithm for high accuracy
//!
//! ## Applications
//!
//! - **Statistics**: Central limit theorem, hypothesis testing, confidence intervals
//! - **Finance**: Risk modeling, option pricing, portfolio theory
//! - **Engineering**: Quality control, signal processing, measurement errors
//! - **Science**: Natural phenomena modeling, experimental data analysis

use crate::error::{Result, DistributionError};
use rustlab_special::error_functions::erf;
use rand::Rng;
use std::f64::consts::{SQRT_2, TAU};

/// Normal (Gaussian) distribution N(μ, σ²)
///
/// The normal distribution is a continuous probability distribution that is widely
/// used in statistics, natural sciences, and engineering. It is characterized by
/// its symmetric bell-shaped curve.
///
/// # Parameters
///
/// - **μ (mu)**: Mean parameter, the location of the distribution center
/// - **σ (sigma)**: Standard deviation parameter (σ > 0), controlling the spread
///
/// # Mathematical Properties
///
/// - **Support**: x ∈ (-∞, +∞)
/// - **Mean**: E[X] = μ
/// - **Variance**: Var[X] = σ²
/// - **Standard deviation**: σ
/// - **Skewness**: 0 (perfectly symmetric)
/// - **Kurtosis**: 3 (mesokurtic)
/// - **Entropy**: ½ ln(2πeσ²)
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::{Normal, Distribution, ContinuousDistribution, Sampling};
/// use rand::thread_rng;
///
/// // Create standard normal distribution N(0, 1)
/// let standard = Normal::new(0.0, 1.0).unwrap();
/// assert_eq!(standard.mean(), 0.0);
/// assert_eq!(standard.variance(), 1.0);
///
/// // Create general normal distribution N(5, 2²)
/// let normal = Normal::new(5.0, 2.0).unwrap();
///
/// // Compute PDF and CDF
/// let pdf = normal.pdf(5.0);  // Should be maximum at mean
/// let cdf = normal.cdf(5.0);  // Should be 0.5 at mean
/// assert!((cdf - 0.5).abs() < 1e-10);
///
/// // Sample from distribution
/// let mut rng = thread_rng();
/// let sample = normal.sample(&mut rng);
///
/// // Quantile function (inverse CDF)
/// let median = normal.quantile(0.5).unwrap();
/// assert!((median - 5.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Normal {
    /// Mean parameter (μ)
    pub mean: f64,
    /// Standard deviation parameter (σ > 0)
    pub std_dev: f64,
    /// Cached parameters tuple for trait implementation
    params: (f64, f64),
}

impl Normal {
    /// Create a new Normal distribution N(μ, σ²)
    ///
    /// Constructs a normal distribution with specified mean and standard deviation.
    /// This constructor validates the parameters to ensure mathematical validity.
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean parameter μ (location), can be any finite real number
    /// * `std_dev` - Standard deviation parameter σ (scale), must be positive and finite
    ///
    /// # Returns
    ///
    /// Returns `Ok(Normal)` if parameters are valid, otherwise returns an error.
    ///
    /// # Errors
    ///
    /// - `InvalidParameter`: When mean is not finite (NaN or ±∞)
    /// - `InvalidParameter`: When std_dev ≤ 0, NaN, or ±∞
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::Normal;
    ///
    /// // Standard normal distribution N(0, 1)
    /// let standard = Normal::new(0.0, 1.0).unwrap();
    /// assert_eq!(standard.mean(), 0.0);
    /// assert_eq!(standard.std_dev(), 1.0);
    ///
    /// // General normal distribution N(10, 2.5²)
    /// let general = Normal::new(10.0, 2.5).unwrap();
    /// assert_eq!(general.mean(), 10.0);
    /// assert_eq!(general.variance(), 6.25);
    ///
    /// // Error cases
    /// assert!(Normal::new(f64::NAN, 1.0).is_err());      // NaN mean
    /// assert!(Normal::new(0.0, 0.0).is_err());           // Zero std_dev
    /// assert!(Normal::new(0.0, -1.0).is_err());          // Negative std_dev
    /// assert!(Normal::new(0.0, f64::INFINITY).is_err()); // Infinite std_dev
    /// ```
    ///
    /// # Mathematical Note
    ///
    /// The normal distribution is often denoted as N(μ, σ²) where σ² is the variance.
    /// This constructor takes the standard deviation σ, not the variance σ².
    pub fn new(mean: f64, std_dev: f64) -> Result<Self> {
        if !mean.is_finite() {
            return Err(DistributionError::invalid_parameter("Mean must be finite"));
        }
        
        if !std_dev.is_finite() || std_dev <= 0.0 {
            return Err(DistributionError::invalid_parameter("Standard deviation must be positive and finite"));
        }
        
        Ok(Normal { 
            mean, 
            std_dev,
            params: (mean, std_dev),
        })
    }
    
    /// Create a standard normal distribution (μ=0, σ=1)
    pub fn standard() -> Self {
        Normal { 
            mean: 0.0, 
            std_dev: 1.0,
            params: (0.0, 1.0),
        }
    }
    
    /// Get the mean parameter
    pub fn mean(&self) -> f64 {
        self.mean
    }
    
    /// Get the standard deviation parameter
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }
    
    /// Get the variance parameter
    pub fn variance(&self) -> f64 {
        self.std_dev * self.std_dev
    }
    
    /// Probability density function (PDF)
    pub fn pdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        
        let z = (x - self.mean) / self.std_dev;
        let coefficient = 1.0 / (self.std_dev * (TAU).sqrt());
        coefficient * (-0.5 * z * z).exp()
    }
    
    /// Natural logarithm of the PDF
    pub fn log_pdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        
        let z = (x - self.mean) / self.std_dev;
        -0.5 * (TAU.ln() + 2.0 * self.std_dev.ln() + z * z)
    }
    
    /// Cumulative distribution function (CDF)
    pub fn cdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return if x.is_sign_positive() { 1.0 } else { 0.0 };
        }
        
        let z = (x - self.mean) / (self.std_dev * SQRT_2);
        0.5 * (1.0 + erf(z))
    }
    
    /// Quantile function (inverse CDF) using Beasley-Springer-Moro algorithm
    pub fn inverse_cdf(&self, p: f64) -> Result<f64> {
        if !p.is_finite() || p < 0.0 || p > 1.0 {
            return Err(DistributionError::invalid_parameter("Probability must be in [0, 1]"));
        }
        
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        if p == 0.5 {
            return Ok(self.mean);
        }
        
        // Use Beasley-Springer-Moro algorithm
        let z = inverse_standard_normal_cdf(p);
        Ok(self.mean + self.std_dev * z)
    }
    
    /// Sample a value from the distribution using Box-Muller transform
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        // Box-Muller transform
        static mut SPARE: Option<f64> = None;
        static mut HAS_SPARE: bool = false;
        
        unsafe {
            if HAS_SPARE {
                HAS_SPARE = false;
                return SPARE.unwrap(); // Already transformed
            }
            
            HAS_SPARE = true;
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            
            let radius = (-2.0 * u1.ln()).sqrt();
            let z0 = radius * (TAU * u2).cos();
            let z1 = radius * (TAU * u2).sin();
            
            // Transform to desired mean and std_dev
            let sample0 = self.mean + self.std_dev * z0;
            let sample1 = self.mean + self.std_dev * z1;
            
            SPARE = Some(sample1);
            sample0
        }
    }
    
    /// Sample multiple values from the distribution
    pub fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
    
    /// Evaluate PDF for multiple values
    pub fn pdf_slice(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| self.pdf(xi)).collect()
    }
    
    /// Evaluate CDF for multiple values
    pub fn cdf_slice(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| self.cdf(xi)).collect()
    }
    
    /// Calculate the moment generating function at t
    pub fn mgf(&self, t: f64) -> f64 {
        if !t.is_finite() {
            return f64::NAN;
        }
        (self.mean * t + 0.5 * self.variance() * t * t).exp()
    }
    
    /// Calculate the characteristic function at t
    pub fn cf(&self, t: f64) -> num_complex::Complex<f64> {
        use num_complex::Complex;
        if !t.is_finite() {
            return Complex::new(f64::NAN, f64::NAN);
        }
        
        let real_part = (self.mean * t - 0.5 * self.variance() * t * t).exp();
        Complex::new(real_part, 0.0)
    }
    
    /// Skewness (always 0 for normal distribution)
    pub fn skewness(&self) -> f64 {
        0.0
    }
    
    /// Kurtosis (always 3 for normal distribution)
    pub fn kurtosis(&self) -> f64 {
        3.0
    }
    
    /// Excess kurtosis (always 0 for normal distribution)
    pub fn excess_kurtosis(&self) -> f64 {
        0.0
    }
    
    /// Mode (equals mean for normal distribution)
    pub fn mode(&self) -> f64 {
        self.mean
    }
    
    /// Median (equals mean for normal distribution)
    pub fn median(&self) -> f64 {
        self.mean
    }
    
    /// Entropy of the distribution
    pub fn entropy(&self) -> f64 {
        0.5 * (TAU * std::f64::consts::E * self.variance()).ln()
    }
}

/// Implementation for rustlab-math integration (feature-gated)
#[cfg(feature = "integration")]
impl Normal {
    /// Evaluate PDF for a rustlab-math vector
    pub fn pdf_vector(&self, x: &rustlab_math::VectorF64) -> rustlab_math::VectorF64 {
        let values: Vec<f64> = (0..x.len()).map(|i| self.pdf(x.get(i).unwrap())).collect();
        rustlab_math::VectorF64::from_slice(&values)
    }
    
    /// Evaluate CDF for a rustlab-math vector
    pub fn cdf_vector(&self, x: &rustlab_math::VectorF64) -> rustlab_math::VectorF64 {
        let values: Vec<f64> = (0..x.len()).map(|i| self.cdf(x.get(i).unwrap())).collect();
        rustlab_math::VectorF64::from_slice(&values)
    }
    
    /// Sample into a rustlab-math vector
    pub fn sample_vector<R: Rng>(&self, rng: &mut R, n: usize) -> rustlab_math::VectorF64 {
        let samples = self.sample_n(rng, n);
        rustlab_math::VectorF64::from_slice(&samples)
    }
}

/// Inverse standard normal CDF using Beasley-Springer-Moro algorithm
fn inverse_standard_normal_cdf(p: f64) -> f64 {
    // Constants for the Beasley-Springer-Moro algorithm
    const A0: f64 = 2.50662823884;
    const A1: f64 = -18.61500062529;
    const A2: f64 = 41.39119773534;
    const A3: f64 = -25.44106049637;
    const B1: f64 = -8.47351093090;
    const B2: f64 = 23.08336743743;
    const B3: f64 = -21.06224101826;
    const B4: f64 = 3.13082909833;
    const C0: f64 = 0.3374754822726147;
    const C1: f64 = 0.9761690190917186;
    const C2: f64 = 0.1607979714918209;
    const C3: f64 = 0.0276438810333863;
    const C4: f64 = 0.0038405729373609;
    const C5: f64 = 0.0003951896511919;
    const C6: f64 = 0.0000321767881768;
    const C7: f64 = 0.0000002888167364;
    const C8: f64 = 0.0000003960315187;
    
    let y = p - 0.5;
    
    if y.abs() < 0.42 {
        // Central region
        let r = y * y;
        let x = y * (((A3 * r + A2) * r + A1) * r + A0) / 
                ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0);
        x
    } else {
        // Tail regions
        let r = if y > 0.0 { 1.0 - p } else { p };
        let x = if r > 0.0 { (-r.ln()).ln() } else { 0.0 };
        
        let result = C0 + x * (C1 + x * (C2 + x * (C3 + x * (C4 + 
                     x * (C5 + x * (C6 + x * (C7 + x * C8)))))));
        
        if y < 0.0 { -result } else { result }
    }
}

// Implement common distribution traits
impl crate::Distribution for Normal {
    type Params = (f64, f64);
    type Support = f64;
    
    fn new(params: Self::Params) -> Result<Self> {
        Self::new(params.0, params.1)
    }
    
    fn params(&self) -> &Self::Params {
        &self.params
    }
    
    fn mean(&self) -> f64 {
        self.mean
    }
    
    fn variance(&self) -> f64 {
        self.variance()
    }
}

impl crate::ContinuousDistribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        self.pdf(x)
    }
    
    fn cdf(&self, x: f64) -> f64 {
        self.cdf(x)
    }
    
    fn quantile(&self, p: f64) -> Result<f64> {
        self.inverse_cdf(p)
    }
}

impl crate::Sampling for Normal {
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Support {
        self.sample(rng)
    }
    
    #[cfg(feature = "integration")]
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> rustlab_math::VectorF64 {
        let samples = self.sample_n(rng, n);
        rustlab_math::VectorF64::from_slice(&samples)
    }
    
    #[cfg(feature = "integration")]
    fn sample_into<R: Rng>(&self, rng: &mut R, output: &mut rustlab_math::VectorF64) {
        let len = output.len();
        if let Some(slice) = output.as_mut_slice() {
            for i in 0..len {
                slice[i] = self.sample(rng);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::{thread_rng, SeedableRng};
    
    #[test]
    fn test_normal_creation() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        assert_eq!(normal.mean(), 0.0);
        assert_eq!(normal.std_dev(), 1.0);
        assert_eq!(normal.variance(), 1.0);
    }
    
    #[test]
    fn test_standard_normal() {
        let normal = Normal::standard();
        assert_eq!(normal.mean(), 0.0);
        assert_eq!(normal.std_dev(), 1.0);
    }
    
    #[test]
    fn test_invalid_parameters() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
        assert!(Normal::new(f64::NAN, 1.0).is_err());
        assert!(Normal::new(0.0, f64::INFINITY).is_err());
    }
    
    #[test]
    fn test_pdf() {
        let normal = Normal::standard();
        
        // PDF at mean should be highest
        let pdf_at_mean = normal.pdf(0.0);
        assert_abs_diff_eq!(pdf_at_mean, 1.0 / (TAU).sqrt(), epsilon = 1e-10);
        
        // PDF should be symmetric
        assert_abs_diff_eq!(normal.pdf(1.0), normal.pdf(-1.0), epsilon = 1e-10);
        
        // PDF should decrease as we move away from mean
        assert!(normal.pdf(0.0) > normal.pdf(1.0));
        assert!(normal.pdf(1.0) > normal.pdf(2.0));
    }
    
    #[test]
    fn test_cdf() {
        let normal = Normal::standard();
        
        // CDF at mean should be 0.5
        assert_abs_diff_eq!(normal.cdf(0.0), 0.5, epsilon = 1e-10);
        
        // CDF should be monotonically increasing
        assert!(normal.cdf(-1.0) < normal.cdf(0.0));
        assert!(normal.cdf(0.0) < normal.cdf(1.0));
        
        // Extreme values
        assert_abs_diff_eq!(normal.cdf(-5.0), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(normal.cdf(5.0), 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_quantile() {
        let normal = Normal::standard();
        
        // Quantile should be inverse of CDF
        assert_abs_diff_eq!(normal.inverse_cdf(0.5).unwrap(), 0.0, epsilon = 1e-10);
        
        // Test some known values
        assert_abs_diff_eq!(normal.inverse_cdf(0.8413).unwrap(), 1.0, epsilon = 0.01);
        assert_abs_diff_eq!(normal.inverse_cdf(0.1587).unwrap(), -1.0, epsilon = 0.01);
    }
    
    #[test]
    fn test_sampling() {
        let normal = Normal::new(5.0, 2.0).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42); // Use deterministic seed
        
        // Sample many values and check if sample statistics are reasonable
        let samples = normal.sample_n(&mut rng, 10000);
        let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let sample_var = samples.iter()
            .map(|&x| (x - sample_mean).powi(2))
            .sum::<f64>() / (samples.len() - 1) as f64;
        
        assert_abs_diff_eq!(sample_mean, 5.0, epsilon = 0.1);
        assert_abs_diff_eq!(sample_var, 4.0, epsilon = 0.2);
    }
    
    #[test]
    fn test_moments() {
        let normal = Normal::new(3.0, 2.0).unwrap();
        
        assert_eq!(normal.skewness(), 0.0);
        assert_eq!(normal.kurtosis(), 3.0);
        assert_eq!(normal.excess_kurtosis(), 0.0);
        assert_eq!(normal.mode(), 3.0);
        assert_eq!(normal.median(), 3.0);
    }
    
    #[test]
    fn test_mgf() {
        let normal = Normal::standard();
        
        // MGF of standard normal at t=0 should be 1
        assert_abs_diff_eq!(normal.mgf(0.0), 1.0, epsilon = 1e-10);
        
        // Known value: MGF of N(0,1) at t=1 should be exp(0.5)
        assert_abs_diff_eq!(normal.mgf(1.0), (0.5_f64).exp(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_entropy() {
        let normal = Normal::standard();
        let expected_entropy = 0.5 * (TAU * std::f64::consts::E).ln();
        assert_abs_diff_eq!(normal.entropy(), expected_entropy, epsilon = 1e-10);
    }
}