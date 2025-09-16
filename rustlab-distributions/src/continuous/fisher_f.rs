//! Fisher F-distribution implementation
//!
//! This module provides a comprehensive implementation of the F-distribution
//! using rustlab-special for mathematical functions.

use crate::error::{Result, DistributionError};
use rustlab_special::gamma_functions::lgamma;
// Note: Incomplete beta function needs to be implemented
use rand::Rng;

/// Fisher F-distribution
#[derive(Debug, Clone, PartialEq)]
pub struct FisherF {
    /// Numerator degrees of freedom (d1 > 0)
    pub df1: f64,
    /// Denominator degrees of freedom (d2 > 0)
    pub df2: f64,
    /// Cached parameters tuple for trait implementation
    params: (f64, f64),
}

impl FisherF {
    /// Create a new F-distribution
    /// 
    /// # Arguments
    /// * `df1` - Numerator degrees of freedom (d1 > 0)
    /// * `df2` - Denominator degrees of freedom (d2 > 0)
    /// 
    /// # Example
    /// ```
    /// use rustlab_distributions::FisherF;
    /// let f_dist = FisherF::new(5.0, 10.0).unwrap();
    /// ```
    pub fn new(df1: f64, df2: f64) -> Result<Self> {
        if !df1.is_finite() || df1 <= 0.0 {
            return Err(DistributionError::invalid_parameter("Numerator degrees of freedom must be positive and finite"));
        }
        
        if !df2.is_finite() || df2 <= 0.0 {
            return Err(DistributionError::invalid_parameter("Denominator degrees of freedom must be positive and finite"));
        }
        
        Ok(FisherF { 
            df1, 
            df2,
            params: (df1, df2),
        })
    }
    
    /// Get the numerator degrees of freedom parameter
    pub fn df1(&self) -> f64 {
        self.df1
    }
    
    /// Get the denominator degrees of freedom parameter
    pub fn df2(&self) -> f64 {
        self.df2
    }
    
    /// Get the mean (df2/(df2-2) for df2 > 2, undefined otherwise)
    pub fn mean(&self) -> f64 {
        if self.df2 > 2.0 {
            self.df2 / (self.df2 - 2.0)
        } else {
            f64::NAN
        }
    }
    
    /// Get the variance
    pub fn variance(&self) -> f64 {
        if self.df2 > 4.0 {
            let d1 = self.df1;
            let d2 = self.df2;
            2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) * (d2 - 2.0) * (d2 - 4.0))
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
        if !x.is_finite() || x <= 0.0 {
            return 0.0;
        }
        
        let d1 = self.df1;
        let d2 = self.df2;
        
        let log_numerator = lgamma((d1 + d2) / 2.0) + (d1 / 2.0) * d1.ln() + (d2 / 2.0) * d2.ln();
        let log_denominator = lgamma(d1 / 2.0) + lgamma(d2 / 2.0);
        let log_coefficient = log_numerator - log_denominator;
        
        let log_power_term = (d1 / 2.0 - 1.0) * x.ln() - ((d1 + d2) / 2.0) * (d2 + d1 * x).ln();
        
        (log_coefficient + log_power_term).exp()
    }
    
    /// Natural logarithm of the PDF
    pub fn log_pdf(&self, x: f64) -> f64 {
        if !x.is_finite() || x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        
        let d1 = self.df1;
        let d2 = self.df2;
        
        let log_numerator = lgamma((d1 + d2) / 2.0) + (d1 / 2.0) * d1.ln() + (d2 / 2.0) * d2.ln();
        let log_denominator = lgamma(d1 / 2.0) + lgamma(d2 / 2.0);
        let log_coefficient = log_numerator - log_denominator;
        
        let log_power_term = (d1 / 2.0 - 1.0) * x.ln() - ((d1 + d2) / 2.0) * (d2 + d1 * x).ln();
        
        log_coefficient + log_power_term
    }
    
    /// Cumulative distribution function (CDF)
    pub fn cdf(&self, x: f64) -> f64 {
        if !x.is_finite() || x <= 0.0 {
            return 0.0;
        }
        
        let d1 = self.df1;
        let d2 = self.df2;
        
        // Fisher F CDF using the relationship with incomplete beta function
        // CDF_F(x; df1, df2) = I_{df1*x/(df1*x + df2)}(df1/2, df2/2)
        let t = (d1 * x) / (d1 * x + d2);
        incomplete_beta_f(t, d1 / 2.0, d2 / 2.0)
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
        
        // Initial guess using mean if available, otherwise use 1.0
        let mut x = if self.df2 > 2.0 {
            self.mean()
        } else {
            1.0
        };
        
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
    
    /// Sample from the distribution using the ratio of chi-squared variables
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        // F = (X1/df1) / (X2/df2) where X1 ~ Chi2(df1) and X2 ~ Chi2(df2)
        let chi2_1 = sample_chi_squared(rng, self.df1);
        let chi2_2 = sample_chi_squared(rng, self.df2);
        
        (chi2_1 / self.df1) / (chi2_2 / self.df2)
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
    
    /// Skewness (for df2 > 6)
    pub fn skewness(&self) -> f64 {
        if self.df2 > 6.0 {
            let d1 = self.df1;
            let d2 = self.df2;
            (2.0 * d1 + d2 - 2.0) / (d2 - 6.0) * (2.0 / d1).sqrt() * ((d2 - 4.0) / (d1 + d2 - 2.0)).sqrt()
        } else {
            f64::NAN
        }
    }
    
    /// Kurtosis (for df2 > 8)
    pub fn kurtosis(&self) -> f64 {
        if self.df2 > 8.0 {
            let d1 = self.df1;
            let d2 = self.df2;
            let numerator = 12.0 * (d1 * (d1 + d2 - 2.0) * (d2 - 4.0) + (d2 - 2.0) * (d2 - 2.0));
            let denominator = d1 * (d2 - 6.0) * (d2 - 8.0) * (d1 + d2 - 2.0);
            3.0 + numerator / denominator
        } else {
            f64::NAN
        }
    }
    
    /// Excess kurtosis
    pub fn excess_kurtosis(&self) -> f64 {
        if self.df2 > 8.0 {
            self.kurtosis() - 3.0
        } else {
            f64::NAN
        }
    }
    
    /// Mode (for df1 > 2)
    pub fn mode(&self) -> f64 {
        if self.df1 > 2.0 {
            (self.df1 - 2.0) / self.df1 * self.df2 / (self.df2 + 2.0)
        } else {
            0.0
        }
    }
}

/// Implementation for rustlab-math integration (feature-gated)
#[cfg(feature = "integration")]
impl FisherF {
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

/// Sample from chi-squared distribution
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
                let angle = 2.0 * std::f64::consts::PI * u2;
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
impl crate::Distribution for FisherF {
    type Params = (f64, f64);
    type Support = f64;
    
    fn new(params: Self::Params) -> Result<Self> {
        Self::new(params.0, params.1)
    }
    
    fn params(&self) -> &Self::Params {
        &self.params
    }
    
    fn mean(&self) -> f64 {
        self.mean()
    }
    
    fn variance(&self) -> f64 {
        self.variance()
    }
}

impl crate::ContinuousDistribution for FisherF {
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

impl crate::Sampling for FisherF {
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
    fn test_fisher_f_creation() {
        let f_dist = FisherF::new(5.0, 10.0).unwrap();
        assert_eq!(f_dist.df1(), 5.0);
        assert_eq!(f_dist.df2(), 10.0);
    }
    
    #[test]
    fn test_invalid_parameters() {
        assert!(FisherF::new(0.0, 10.0).is_err());
        assert!(FisherF::new(5.0, 0.0).is_err());
        assert!(FisherF::new(-1.0, 10.0).is_err());
        assert!(FisherF::new(5.0, -1.0).is_err());
        assert!(FisherF::new(f64::NAN, 10.0).is_err());
        assert!(FisherF::new(5.0, f64::INFINITY).is_err());
    }
    
    #[test]
    fn test_moments() {
        let f_dist = FisherF::new(5.0, 10.0).unwrap();
        
        // Mean should be df2/(df2-2) = 10/8 = 1.25
        assert_abs_diff_eq!(f_dist.mean(), 1.25, epsilon = 1e-10);
    }
    
    #[test]
    fn test_pdf() {
        let f_dist = FisherF::new(2.0, 3.0).unwrap();
        
        // PDF should be positive for positive x
        assert!(f_dist.pdf(0.5) > 0.0);
        assert!(f_dist.pdf(1.0) > 0.0);
        assert!(f_dist.pdf(2.0) > 0.0);
        
        // PDF should be 0 for non-positive x
        assert_eq!(f_dist.pdf(0.0), 0.0);
        assert_eq!(f_dist.pdf(-1.0), 0.0);
    }
    
    #[test]
    fn test_cdf() {
        let f_dist = FisherF::new(3.0, 5.0).unwrap();
        
        // CDF should be monotonically increasing
        assert!(f_dist.cdf(0.5) < f_dist.cdf(1.0));
        assert!(f_dist.cdf(1.0) < f_dist.cdf(2.0));
        
        // CDF at 0 should be 0
        assert_eq!(f_dist.cdf(0.0), 0.0);
        
        // CDF should approach 1 as x increases
        assert!(f_dist.cdf(100.0) > 0.9);
    }
    
    #[test]
    fn test_quantile() {
        let f_dist = FisherF::new(5.0, 10.0).unwrap();
        
        // Test that quantile is approximately the inverse of CDF
        let x = 1.5;
        let p = f_dist.cdf(x);
        let x_recovered = f_dist.inverse_cdf(p).unwrap();
        assert_abs_diff_eq!(x, x_recovered, epsilon = 1e-4);
    }
    
    #[test]
    fn test_sampling() {
        let f_dist = FisherF::new(10.0, 20.0).unwrap();
        let mut rng = thread_rng();
        
        let samples = f_dist.sample_n(&mut rng, 1000);
        
        // All samples should be positive
        for &sample in &samples {
            assert!(sample > 0.0);
        }
        
        let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        
        // Mean should be approximately df2/(df2-2) = 20/18 ≈ 1.11
        assert_abs_diff_eq!(sample_mean, 20.0/18.0, epsilon = 0.2);
    }
    
    #[test]
    fn test_mode() {
        let f_dist = FisherF::new(5.0, 10.0).unwrap();
        
        // Mode should be (df1-2)/df1 * df2/(df2+2) = 3/5 * 10/12 = 0.5
        assert_abs_diff_eq!(f_dist.mode(), 0.5, epsilon = 1e-10);
    }
}

/// Compute the regularized incomplete beta function I_x(a,b) for Fisher F distribution
fn incomplete_beta_f(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    
    // Use continued fraction expansion for better numerical stability
    // The continued fraction for I_x(a,b) is more efficient when x < (a+1)/(a+b+2)
    let use_complement = x > (a + 1.0) / (a + b + 2.0);
    
    if use_complement {
        // Compute 1 - I_{1-x}(b,a) for better convergence
        1.0 - incomplete_beta_cf_f(1.0 - x, b, a)
    } else {
        incomplete_beta_cf_f(x, a, b)
    }
}

/// Compute incomplete beta using continued fraction expansion for Fisher F
fn incomplete_beta_cf_f(x: f64, a: f64, b: f64) -> f64 {
    use rustlab_special::gamma_functions::lgamma;
    
    // Compute the coefficient
    let ln_beta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let coeff = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;
    
    // Continued fraction: I_x(a,b) = coeff * cf
    let cf = continued_fraction_beta_f(x, a, b);
    
    coeff * cf
}

/// Continued fraction for incomplete beta function for Fisher F
fn continued_fraction_beta_f(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPSILON: f64 = 1e-15;
    
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    
    // First convergent
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < EPSILON {
        d = EPSILON;
    }
    d = 1.0 / d;
    let mut h = d;
    
    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let m2 = 2 * m;
        
        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < EPSILON {
            d = EPSILON;
        }
        c = 1.0 + aa / c;
        if c.abs() < EPSILON {
            c = EPSILON;
        }
        d = 1.0 / d;
        h *= d * c;
        
        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < EPSILON {
            d = EPSILON;
        }
        c = 1.0 + aa / c;
        if c.abs() < EPSILON {
            c = EPSILON;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        
        if (del - 1.0).abs() < EPSILON {
            break;
        }
    }
    
    h
}