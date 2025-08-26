//! Student's t-distribution implementation
//!
//! This module provides a comprehensive implementation of the Student's t-distribution
//! using rustlab-special for mathematical functions.

use crate::error::{Result, DistributionError};
use rustlab_special::gamma_functions::{gamma, lgamma};
// Note: Incomplete beta function needs to be implemented
use rand::Rng;
use std::f64::consts::PI;

/// Student's t-distribution
#[derive(Debug, Clone, PartialEq)]
pub struct StudentT {
    /// Degrees of freedom parameter (ν > 0)
    pub df: f64,
}

impl StudentT {
    /// Create a new Student's t-distribution
    /// 
    /// # Arguments
    /// * `df` - Degrees of freedom parameter (ν > 0)
    /// 
    /// # Example
    /// ```
    /// use rustlab_distributions::StudentT;
    /// let t_dist = StudentT::new(5.0).unwrap();
    /// ```
    pub fn new(df: f64) -> Result<Self> {
        if !df.is_finite() || df <= 0.0 {
            return Err(DistributionError::invalid_parameter("Degrees of freedom must be positive and finite"));
        }
        
        Ok(StudentT { df })
    }
    
    /// Get the degrees of freedom parameter
    pub fn df(&self) -> f64 {
        self.df
    }
    
    /// Get the mean (0 for df > 1, undefined otherwise)
    pub fn mean(&self) -> f64 {
        if self.df > 1.0 {
            0.0
        } else {
            f64::NAN
        }
    }
    
    /// Get the variance (df/(df-2) for df > 2, undefined otherwise)
    pub fn variance(&self) -> f64 {
        if self.df > 2.0 {
            self.df / (self.df - 2.0)
        } else if self.df > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
    
    /// Standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    
    /// Probability density function (PDF)
    pub fn pdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        
        let gamma_term = gamma((self.df + 1.0) / 2.0) / (gamma(self.df / 2.0) * (self.df * PI).sqrt());
        let power_term = (1.0 + x * x / self.df).powf(-(self.df + 1.0) / 2.0);
        
        gamma_term * power_term
    }
    
    /// Natural logarithm of the PDF
    pub fn log_pdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        
        let log_gamma_term = lgamma((self.df + 1.0) / 2.0) - lgamma(self.df / 2.0) - 0.5 * (self.df * PI).ln();
        let log_power_term = -(self.df + 1.0) / 2.0 * (1.0 + x * x / self.df).ln();
        
        log_gamma_term + log_power_term
    }
    
    /// Cumulative distribution function (CDF)
    pub fn cdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return if x.is_sign_positive() { 1.0 } else { 0.0 };
        }
        
        // Simplified CDF approximation using gamma functions
        // This is a placeholder - proper implementation would use incomplete beta
        if x == 0.0 { return 0.5; }
        
        // For now, use a rough approximation
        let standardized = x / (self.df / (self.df - 2.0)).sqrt();
        0.5 + 0.5 * (standardized / (1.0 + standardized * standardized).sqrt()).tanh()
    }
    
    /// Quantile function (inverse CDF) using Newton-Raphson method
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
            return Ok(0.0);
        }
        
        // Use normal approximation as initial guess for large df
        let mut x = if self.df > 30.0 {
            // Use normal approximation
            inverse_standard_normal_cdf(p)
        } else {
            // Simple initial guess based on symmetry
            if p < 0.5 {
                -1.0
            } else {
                1.0
            }
        };
        
        // Newton-Raphson iterations
        for _ in 0..100 {
            let fx = self.cdf(x) - p;
            let fpx = self.pdf(x);
            
            if fx.abs() < 1e-12 || fpx.abs() < f64::EPSILON {
                break;
            }
            
            let new_x = x - fx / fpx;
            
            if (new_x - x).abs() < 1e-12 {
                break;
            }
            
            x = new_x;
        }
        
        Ok(x)
    }
    
    /// Sample from the distribution using acceptance-rejection method
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        // Use ratio of uniforms method for t-distribution
        if self.df >= 1000.0 {
            // For very large df, use normal distribution
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let magnitude = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * PI * u2;
            magnitude * angle.cos()
        } else {
            // Use the fact that T = Z / sqrt(V/df) where Z ~ N(0,1) and V ~ Chi2(df)
            let z = {
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                let magnitude = (-2.0 * u1.ln()).sqrt();
                let angle = 2.0 * PI * u2;
                magnitude * angle.cos()
            };
            
            // Generate chi-squared random variable
            let chi2 = sample_chi_squared(rng, self.df);
            
            z / (chi2 / self.df).sqrt()
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
    
    /// Skewness (0 for df > 3, undefined otherwise)
    pub fn skewness(&self) -> f64 {
        if self.df > 3.0 {
            0.0
        } else {
            f64::NAN
        }
    }
    
    /// Kurtosis (6/(df-4) + 3 for df > 4, undefined otherwise)
    pub fn kurtosis(&self) -> f64 {
        if self.df > 4.0 {
            6.0 / (self.df - 4.0) + 3.0
        } else {
            f64::NAN
        }
    }
    
    /// Excess kurtosis
    pub fn excess_kurtosis(&self) -> f64 {
        if self.df > 4.0 {
            6.0 / (self.df - 4.0)
        } else {
            f64::NAN
        }
    }
    
    /// Mode (always 0)
    pub fn mode(&self) -> f64 {
        0.0
    }
    
    /// Median (always 0)
    pub fn median(&self) -> f64 {
        0.0
    }
}

/// Implementation for rustlab-math integration (feature-gated)
#[cfg(feature = "integration")]
impl StudentT {
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

/// Sample from chi-squared distribution using gamma sampling
fn sample_chi_squared<R: Rng>(rng: &mut R, df: f64) -> f64 {
    // Chi-squared with df degrees of freedom is Gamma(df/2, 2)
    sample_gamma(rng, df / 2.0, 2.0)
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

// Implement common distribution traits
impl crate::Distribution for StudentT {
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

impl crate::ContinuousDistribution for StudentT {
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

impl crate::Sampling for StudentT {
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
    fn test_student_t_creation() {
        let t_dist = StudentT::new(5.0).unwrap();
        assert_eq!(t_dist.df(), 5.0);
    }
    
    #[test]
    fn test_invalid_parameters() {
        assert!(StudentT::new(0.0).is_err());
        assert!(StudentT::new(-1.0).is_err());
        assert!(StudentT::new(f64::NAN).is_err());
        assert!(StudentT::new(f64::INFINITY).is_err());
    }
    
    #[test]
    fn test_moments() {
        let t_dist = StudentT::new(5.0).unwrap();
        
        assert_eq!(t_dist.mean(), 0.0);
        assert_abs_diff_eq!(t_dist.variance(), 5.0 / 3.0, epsilon = 1e-10);
        assert_eq!(t_dist.skewness(), 0.0);
        assert_eq!(t_dist.mode(), 0.0);
        assert_eq!(t_dist.median(), 0.0);
    }
    
    #[test]
    fn test_pdf() {
        let t_dist = StudentT::new(1.0).unwrap(); // Cauchy distribution
        
        // PDF should be symmetric around 0
        assert_abs_diff_eq!(t_dist.pdf(1.0), t_dist.pdf(-1.0), epsilon = 1e-10);
        
        // PDF at 0 should be maximum
        assert!(t_dist.pdf(0.0) > t_dist.pdf(1.0));
    }
    
    #[test]
    fn test_cdf() {
        let t_dist = StudentT::new(5.0).unwrap();
        
        // CDF at 0 should be 0.5
        assert_abs_diff_eq!(t_dist.cdf(0.0), 0.5, epsilon = 1e-10);
        
        // CDF should be monotonically increasing
        assert!(t_dist.cdf(-1.0) < t_dist.cdf(0.0));
        assert!(t_dist.cdf(0.0) < t_dist.cdf(1.0));
    }
    
    #[test]
    fn test_quantile() {
        let t_dist = StudentT::new(5.0).unwrap();
        
        // Quantile at 0.5 should be 0
        assert_abs_diff_eq!(t_dist.inverse_cdf(0.5).unwrap(), 0.0, epsilon = 1e-10);
        
        // Test symmetry
        let q25 = t_dist.inverse_cdf(0.25).unwrap();
        let q75 = t_dist.inverse_cdf(0.75).unwrap();
        assert_abs_diff_eq!(q25, -q75, epsilon = 1e-6);
    }
    
    #[test]
    fn test_sampling() {
        let t_dist = StudentT::new(10.0).unwrap();
        let mut rng = thread_rng();
        
        let samples = t_dist.sample_n(&mut rng, 1000);
        let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        
        // Mean should be close to 0
        assert_abs_diff_eq!(sample_mean, 0.0, epsilon = 0.1);
    }
}