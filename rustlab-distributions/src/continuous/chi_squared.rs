//! Chi-squared distribution implementation
//!
//! This module provides a comprehensive implementation of the chi-squared distribution
//! using rustlab-special for mathematical functions.

use crate::error::{Result, DistributionError};
use rustlab_special::gamma_functions::{gamma, lgamma};
use rustlab_special::incomplete_gamma::gamma_p;
use rand::Rng;
use std::f64::consts::PI;

/// Chi-squared distribution
#[derive(Debug, Clone, PartialEq)]
pub struct ChiSquared {
    /// Degrees of freedom parameter (k > 0)
    pub df: f64,
}

impl ChiSquared {
    /// Create a new Chi-squared distribution
    /// 
    /// # Arguments
    /// * `df` - Degrees of freedom parameter (k > 0)
    /// 
    /// # Example
    /// ```
    /// use rustlab_distributions::ChiSquared;
    /// let chi2 = ChiSquared::new(5.0).unwrap();
    /// ```
    pub fn new(df: f64) -> Result<Self> {
        if !df.is_finite() || df <= 0.0 {
            return Err(DistributionError::invalid_parameter("Degrees of freedom must be positive and finite"));
        }
        
        Ok(ChiSquared { df })
    }
    
    /// Get the degrees of freedom parameter
    pub fn df(&self) -> f64 {
        self.df
    }
    
    /// Get the mean (equals df)
    pub fn mean(&self) -> f64 {
        self.df
    }
    
    /// Get the variance (equals 2*df)
    pub fn variance(&self) -> f64 {
        2.0 * self.df
    }
    
    /// Standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    
    /// Probability density function (PDF)
    pub fn pdf(&self, x: f64) -> f64 {
        if !x.is_finite() || x < 0.0 {
            return 0.0;
        }
        
        if x == 0.0 {
            return if self.df < 2.0 {
                f64::INFINITY
            } else if self.df == 2.0 {
                0.5
            } else {
                0.0
            };
        }
        
        let k = self.df / 2.0;
        let coefficient = 1.0 / (2.0_f64.powf(k) * gamma(k));
        coefficient * x.powf(k - 1.0) * (-x / 2.0).exp()
    }
    
    /// Natural logarithm of the PDF
    pub fn log_pdf(&self, x: f64) -> f64 {
        if !x.is_finite() || x < 0.0 {
            return f64::NEG_INFINITY;
        }
        
        if x == 0.0 {
            return if self.df < 2.0 {
                f64::INFINITY
            } else if self.df == 2.0 {
                0.5_f64.ln()
            } else {
                f64::NEG_INFINITY
            };
        }
        
        let k = self.df / 2.0;
        let log_coefficient = -(k * 2.0_f64.ln() + lgamma(k));
        log_coefficient + (k - 1.0) * x.ln() - x / 2.0
    }
    
    /// Cumulative distribution function (CDF)
    pub fn cdf(&self, x: f64) -> f64 {
        if !x.is_finite() || x < 0.0 {
            return 0.0;
        }
        
        if x == 0.0 {
            return 0.0;
        }
        
        let k = self.df / 2.0;
        gamma_p(k, x / 2.0)
    }
    
    /// Quantile function (inverse CDF) using Newton-Raphson method
    pub fn inverse_cdf(&self, p: f64) -> Result<f64> {
        if !p.is_finite() || p < 0.0 || p > 1.0 {
            return Err(DistributionError::invalid_parameter("Probability must be in [0, 1]"));
        }
        
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        
        // Initial guess using Wilson-Hilferty transformation
        let h = 2.0 / (9.0 * self.df);
        let z = inverse_standard_normal_cdf(p);
        let mut x = self.df * (1.0 - h + z * h.sqrt()).powi(3);
        x = x.max(0.01); // Ensure positive starting value
        
        // Newton-Raphson iterations
        for _ in 0..100 {
            let fx = self.cdf(x) - p;
            let fpx = self.pdf(x);
            
            if fx.abs() < 1e-12 || fpx.abs() < f64::EPSILON {
                break;
            }
            
            let new_x = x - fx / fpx;
            
            if (new_x - x).abs() < 1e-12 || new_x <= 0.0 {
                break;
            }
            
            x = new_x;
        }
        
        Ok(x.max(0.0))
    }
    
    /// Sample from the distribution using gamma sampling
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        // Chi-squared with df degrees of freedom is Gamma(df/2, 2)
        sample_gamma(rng, self.df / 2.0, 2.0)
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
    
    /// Skewness
    pub fn skewness(&self) -> f64 {
        2.0 * (2.0 / self.df).sqrt()
    }
    
    /// Kurtosis
    pub fn kurtosis(&self) -> f64 {
        3.0 + 12.0 / self.df
    }
    
    /// Excess kurtosis
    pub fn excess_kurtosis(&self) -> f64 {
        12.0 / self.df
    }
    
    /// Mode
    pub fn mode(&self) -> f64 {
        if self.df >= 2.0 {
            self.df - 2.0
        } else {
            0.0
        }
    }
    
    /// Median (approximate using Wilson-Hilferty transformation)
    pub fn median(&self) -> f64 {
        let h = 2.0 / (9.0 * self.df);
        self.df * (1.0 - h).powi(3)
    }
    
    /// Entropy of the distribution
    pub fn entropy(&self) -> f64 {
        let k = self.df / 2.0;
        k + (2.0 * gamma(k)).ln() + (1.0 - k) * digamma(k)
    }
}

/// Implementation for rustlab-math integration (feature-gated)
#[cfg(feature = "integration")]
impl ChiSquared {
    /// Evaluate PDF for a rustlab-math vector
    pub fn pdf_vector(&self, x: &rustlab_math::VectorF64) -> rustlab_math::VectorF64 {
        let values: Vec<f64> = (0..x.len()).map(|i| self.pdf(x.get(i).unwrap())).collect();
        rustlab_math::VectorF64::from_vec(values)
    }
    
    /// Evaluate CDF for a rustlab-math vector
    pub fn cdf_vector(&self, x: &rustlab_math::VectorF64) -> rustlab_math::VectorF64 {
        let values: Vec<f64> = (0..x.len()).map(|i| self.cdf(x.get(i).unwrap())).collect();
        rustlab_math::VectorF64::from_vec(values)
    }
    
    /// Sample into a rustlab-math vector
    pub fn sample_vector<R: Rng>(&self, rng: &mut R, n: usize) -> rustlab_math::VectorF64 {
        let samples = self.sample_n(rng, n);
        rustlab_math::VectorF64::from_vec(samples)
    }
}

// Helper functions

/// Simple inverse standard normal CDF approximation
fn inverse_standard_normal_cdf(p: f64) -> f64 {
    // Beasley-Springer-Moro approximation (simplified)
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    
    let y = if p > 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * y.ln()).sqrt();
    
    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    
    let result = t - numerator / denominator;
    
    if p > 0.5 { result } else { -result }
}

/// Sample from gamma distribution using Marsaglia-Tsang method
fn sample_gamma<R: Rng>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    if shape < 1.0 {
        // Use the method for shape < 1
        let u: f64 = rng.gen();
        sample_gamma(rng, shape + 1.0, scale) * u.powf(1.0 / shape)
    } else {
        // Marsaglia-Tsang method for shape >= 1
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        
        loop {
            let z = {
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                let magnitude = (-2.0 * u1.ln()).sqrt();
                let angle = 2.0 * PI * u2;
                magnitude * angle.cos()
            };
            
            let v = (1.0 + c * z).powi(3);
            
            if v > 0.0 {
                let u: f64 = rng.gen();
                if u < 1.0 - 0.0331 * z.powi(4) {
                    return d * v * scale;
                }
                if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
                    return d * v * scale;
                }
            }
        }
    }
}

/// Simple digamma function approximation
fn digamma(x: f64) -> f64 {
    if x < 6.0 {
        digamma(x + 1.0) - 1.0 / x
    } else {
        let inv_x = 1.0 / x;
        let inv_x_2 = inv_x * inv_x;
        x.ln() - 0.5 * inv_x - inv_x_2 / 12.0 + inv_x_2 * inv_x_2 / 120.0
    }
}

// Implement common distribution traits
impl crate::Distribution for ChiSquared {
    type Params = f64;
    type Support = f64;
    
    fn new(params: Self::Params) -> Result<Self> {
        Self::new(params)
    }
    
    fn params(&self) -> &Self::Params {
        &self.df
    }
    
    fn mean(&self) -> f64 {
        self.mean()
    }
    
    fn variance(&self) -> f64 {
        self.variance()
    }
}

impl crate::ContinuousDistribution for ChiSquared {
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

impl crate::Sampling for ChiSquared {
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Support {
        self.sample(rng)
    }
    
    #[cfg(feature = "integration")]
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> rustlab_math::VectorF64 {
        let samples = self.sample_n(rng, n);
        rustlab_math::VectorF64::from_vec(samples)
    }
    
    #[cfg(feature = "integration")]
    fn sample_into<R: Rng>(&self, rng: &mut R, output: &mut rustlab_math::VectorF64) {
        for i in 0..output.len() {
            let sample = self.sample(rng);
            output.set(i, sample).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::thread_rng;
    
    #[test]
    fn test_chi_squared_creation() {
        let chi2 = ChiSquared::new(5.0).unwrap();
        assert_eq!(chi2.df(), 5.0);
    }
    
    #[test]
    fn test_invalid_parameters() {
        assert!(ChiSquared::new(0.0).is_err());
        assert!(ChiSquared::new(-1.0).is_err());
        assert!(ChiSquared::new(f64::NAN).is_err());
        assert!(ChiSquared::new(f64::INFINITY).is_err());
    }
    
    #[test]
    fn test_moments() {
        let chi2 = ChiSquared::new(5.0).unwrap();
        
        assert_eq!(chi2.mean(), 5.0);
        assert_eq!(chi2.variance(), 10.0);
        assert_abs_diff_eq!(chi2.std_dev(), 10.0_f64.sqrt(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_pdf() {
        let chi2 = ChiSquared::new(2.0).unwrap();
        
        // For df=2, PDF should be exponential: 0.5 * exp(-x/2)
        assert_abs_diff_eq!(chi2.pdf(0.0), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(chi2.pdf(2.0), 0.5 * (-1.0_f64).exp(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_cdf() {
        let chi2 = ChiSquared::new(1.0).unwrap();
        
        // CDF should be monotonically increasing
        assert!(chi2.cdf(0.0) < chi2.cdf(1.0));
        assert!(chi2.cdf(1.0) < chi2.cdf(2.0));
        
        // CDF at 0 should be 0
        assert_eq!(chi2.cdf(0.0), 0.0);
    }
    
    #[test]
    fn test_quantile() {
        let chi2 = ChiSquared::new(2.0).unwrap();
        
        // Test that quantile is approximately the inverse of CDF
        let x = 3.0;
        let p = chi2.cdf(x);
        let x_recovered = chi2.inverse_cdf(p).unwrap();
        assert_abs_diff_eq!(x, x_recovered, epsilon = 1e-6);
    }
    
    #[test]
    fn test_sampling() {
        let chi2 = ChiSquared::new(10.0).unwrap();
        let mut rng = thread_rng();
        
        let samples = chi2.sample_n(&mut rng, 1000);
        let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        
        // Mean should be close to df
        assert_abs_diff_eq!(sample_mean, 10.0, epsilon = 1.0);
    }
    
    #[test]
    fn test_mode() {
        let chi2_1 = ChiSquared::new(1.0).unwrap();
        let chi2_3 = ChiSquared::new(3.0).unwrap();
        
        assert_eq!(chi2_1.mode(), 0.0);  // df < 2
        assert_eq!(chi2_3.mode(), 1.0);  // df - 2
    }
}