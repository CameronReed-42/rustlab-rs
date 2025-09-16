//! Beta distribution implementation
//!
//! This module provides a comprehensive implementation of the beta distribution
//! using rustlab-special for mathematical functions.

use crate::error::{Result, DistributionError};
use rustlab_special::gamma_functions::{lgamma, digamma};
// Note: Incomplete beta function needs to be implemented
use rand::Rng;
use std::f64::consts::PI;

/// Beta distribution
#[derive(Debug, Clone, PartialEq)]
pub struct Beta {
    /// Shape parameter alpha (α > 0)
    pub alpha: f64,
    /// Shape parameter beta (β > 0)
    pub beta: f64,
    /// Cached parameters tuple for trait implementation
    params: (f64, f64),
}

impl Beta {
    /// Create a new Beta distribution
    /// 
    /// # Arguments
    /// * `alpha` - Shape parameter alpha (α > 0)
    /// * `beta` - Shape parameter beta (β > 0)
    /// 
    /// # Example
    /// ```
    /// use rustlab_distributions::Beta;
    /// let beta_dist = Beta::new(2.0, 3.0).unwrap();
    /// ```
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(DistributionError::invalid_parameter("Alpha parameter must be positive and finite"));
        }
        
        if !beta.is_finite() || beta <= 0.0 {
            return Err(DistributionError::invalid_parameter("Beta parameter must be positive and finite"));
        }
        
        Ok(Beta { 
            alpha, 
            beta,
            params: (alpha, beta),
        })
    }
    
    /// Get the alpha parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    
    /// Get the beta parameter
    pub fn beta(&self) -> f64 {
        self.beta
    }
    
    /// Get the mean (alpha/(alpha+beta))
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
    
    /// Get the variance (alpha*beta/((alpha+beta)²*(alpha+beta+1)))
    pub fn variance(&self) -> f64 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }
    
    /// Standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    
    /// Probability density function (PDF)
    pub fn pdf(&self, x: f64) -> f64 {
        if !x.is_finite() || x < 0.0 || x > 1.0 {
            return 0.0;
        }
        
        // Handle boundary cases
        if x == 0.0 {
            return if self.alpha < 1.0 {
                f64::INFINITY
            } else if self.alpha == 1.0 {
                self.beta
            } else {
                0.0
            };
        }
        
        if x == 1.0 {
            return if self.beta < 1.0 {
                f64::INFINITY
            } else if self.beta == 1.0 {
                self.alpha
            } else {
                0.0
            };
        }
        
        let log_pdf = self.log_pdf(x);
        log_pdf.exp()
    }
    
    /// Natural logarithm of the PDF
    pub fn log_pdf(&self, x: f64) -> f64 {
        if !x.is_finite() || x < 0.0 || x > 1.0 {
            return f64::NEG_INFINITY;
        }
        
        // Handle boundary cases
        if x == 0.0 {
            return if self.alpha < 1.0 {
                f64::INFINITY
            } else if self.alpha == 1.0 {
                self.beta.ln()
            } else {
                f64::NEG_INFINITY
            };
        }
        
        if x == 1.0 {
            return if self.beta < 1.0 {
                f64::INFINITY
            } else if self.beta == 1.0 {
                self.alpha.ln()
            } else {
                f64::NEG_INFINITY
            };
        }
        
        // log(Beta(α,β)) = log(Γ(α+β)) - log(Γ(α)) - log(Γ(β))
        let log_normalizing = lgamma(self.alpha + self.beta) - lgamma(self.alpha) - lgamma(self.beta);
        log_normalizing + (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln()
    }
    
    /// Cumulative distribution function (CDF)
    pub fn cdf(&self, x: f64) -> f64 {
        if !x.is_finite() || x < 0.0 {
            return 0.0;
        }
        
        if x >= 1.0 {
            return 1.0;
        }
        
        if x == 0.0 {
            return 0.0;
        }
        
        // Compute the regularized incomplete beta function I_x(α, β)
        // Using the continued fraction expansion
        incomplete_beta(x, self.alpha, self.beta)
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
            return Ok(1.0);
        }
        
        // Initial guess using mean
        let mut x = self.mean();
        
        // Newton-Raphson iterations
        for _ in 0..100 {
            let fx = self.cdf(x) - p;
            let fpx = self.pdf(x);
            
            if fx.abs() < 1e-12 || fpx.abs() < f64::EPSILON {
                break;
            }
            
            let new_x = x - fx / fpx;
            
            if (new_x - x).abs() < 1e-12 || new_x <= 0.0 || new_x >= 1.0 {
                break;
            }
            
            x = new_x.max(1e-15).min(1.0 - 1e-15);
        }
        
        Ok(x.max(0.0).min(1.0))
    }
    
    /// Sample from the distribution using acceptance-rejection or transformation
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        // Use gamma transformation: if X ~ Gamma(α, 1), Y ~ Gamma(β, 1)
        // then X/(X+Y) ~ Beta(α, β)
        let x = sample_gamma(rng, self.alpha, 1.0);
        let y = sample_gamma(rng, self.beta, 1.0);
        x / (x + y)
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
        let sum = self.alpha + self.beta;
        2.0 * (self.beta - self.alpha) * (sum + 1.0).sqrt() / ((sum + 2.0) * (self.alpha * self.beta).sqrt())
    }
    
    /// Kurtosis
    pub fn kurtosis(&self) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        let sum = a + b;
        
        let numerator = 6.0 * ((a - b) * (a - b) * (sum + 1.0) - a * b * (sum + 2.0));
        let denominator = a * b * (sum + 2.0) * (sum + 3.0);
        
        3.0 + numerator / denominator
    }
    
    /// Excess kurtosis
    pub fn excess_kurtosis(&self) -> f64 {
        self.kurtosis() - 3.0
    }
    
    /// Mode
    pub fn mode(&self) -> f64 {
        if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        } else if self.alpha < 1.0 && self.beta < 1.0 {
            // Bimodal at endpoints
            f64::NAN
        } else if self.alpha < 1.0 && self.beta >= 1.0 {
            0.0
        } else if self.alpha >= 1.0 && self.beta < 1.0 {
            1.0
        } else {
            // alpha == 1 or beta == 1 case - uniform-like
            0.5
        }
    }
    
    /// Median (approximate using inverse CDF)
    pub fn median(&self) -> f64 {
        self.inverse_cdf(0.5).unwrap_or(self.mean())
    }
    
    /// Entropy of the distribution
    pub fn entropy(&self) -> f64 {
        let sum = self.alpha + self.beta;
        lgamma(self.alpha) + lgamma(self.beta) - lgamma(sum)
            - (self.alpha - 1.0) * digamma(self.alpha)
            - (self.beta - 1.0) * digamma(self.beta)
            + (sum - 2.0) * digamma(sum)
    }
    
    /// Check if distribution is symmetric
    pub fn is_symmetric(&self) -> bool {
        (self.alpha - self.beta).abs() < f64::EPSILON
    }
}

/// Implementation for rustlab-math integration (feature-gated)
#[cfg(feature = "integration")]
impl Beta {
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
impl crate::Distribution for Beta {
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

impl crate::ContinuousDistribution for Beta {
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

impl crate::Sampling for Beta {
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
    fn test_beta_creation() {
        let beta_dist = Beta::new(2.0, 3.0).unwrap();
        assert_eq!(beta_dist.alpha(), 2.0);
        assert_eq!(beta_dist.beta(), 3.0);
    }
    
    #[test]
    fn test_invalid_parameters() {
        assert!(Beta::new(0.0, 3.0).is_err());
        assert!(Beta::new(2.0, 0.0).is_err());
        assert!(Beta::new(-1.0, 3.0).is_err());
        assert!(Beta::new(2.0, -1.0).is_err());
        assert!(Beta::new(f64::NAN, 3.0).is_err());
        assert!(Beta::new(2.0, f64::INFINITY).is_err());
    }
    
    #[test]
    fn test_moments() {
        let beta_dist = Beta::new(2.0, 3.0).unwrap();
        
        // Mean = α/(α+β) = 2/5 = 0.4
        assert_abs_diff_eq!(beta_dist.mean(), 0.4, epsilon = 1e-10);
        
        // Variance = αβ/((α+β)²(α+β+1)) = 6/(25*6) = 0.04
        assert_abs_diff_eq!(beta_dist.variance(), 0.04, epsilon = 1e-10);
    }
    
    #[test]
    fn test_uniform_distribution() {
        let uniform = Beta::new(1.0, 1.0).unwrap(); // Beta(1,1) = Uniform(0,1)
        
        // Mean should be 0.5
        assert_abs_diff_eq!(uniform.mean(), 0.5, epsilon = 1e-10);
        
        // PDF should be 1 everywhere in [0,1]
        assert_abs_diff_eq!(uniform.pdf(0.3), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(uniform.pdf(0.7), 1.0, epsilon = 1e-10);
        
        // CDF should be linear
        assert_abs_diff_eq!(uniform.cdf(0.3), 0.3, epsilon = 1e-10);
        assert_abs_diff_eq!(uniform.cdf(0.7), 0.7, epsilon = 1e-10);
    }
    
    #[test]
    fn test_pdf() {
        let beta_dist = Beta::new(2.0, 2.0).unwrap();
        
        // PDF should be 0 outside [0,1]
        assert_eq!(beta_dist.pdf(-0.1), 0.0);
        assert_eq!(beta_dist.pdf(1.1), 0.0);
        
        // PDF should be symmetric for Beta(2,2)
        assert_abs_diff_eq!(beta_dist.pdf(0.3), beta_dist.pdf(0.7), epsilon = 1e-10);
        
        // Maximum should be at 0.5
        assert!(beta_dist.pdf(0.5) > beta_dist.pdf(0.3));
        assert!(beta_dist.pdf(0.5) > beta_dist.pdf(0.7));
    }
    
    #[test]
    fn test_cdf() {
        let beta_dist = Beta::new(2.0, 2.0).unwrap();
        
        // CDF should be 0 at 0 and 1 at 1
        assert_eq!(beta_dist.cdf(0.0), 0.0);
        assert_eq!(beta_dist.cdf(1.0), 1.0);
        
        // CDF should be monotonically increasing
        assert!(beta_dist.cdf(0.3) < beta_dist.cdf(0.5));
        assert!(beta_dist.cdf(0.5) < beta_dist.cdf(0.7));
        
        // CDF should be 0.5 at median for symmetric distribution
        assert_abs_diff_eq!(beta_dist.cdf(0.5), 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_quantile() {
        let beta_dist = Beta::new(3.0, 2.0).unwrap();
        
        // Test that quantile is approximately the inverse of CDF
        let x = 0.6;
        let p = beta_dist.cdf(x);
        let x_recovered = beta_dist.inverse_cdf(p).unwrap();
        assert_abs_diff_eq!(x, x_recovered, epsilon = 1e-6);
    }
    
    #[test]
    fn test_sampling() {
        let beta_dist = Beta::new(5.0, 2.0).unwrap();
        let mut rng = thread_rng();
        
        let samples = beta_dist.sample_n(&mut rng, 1000);
        
        // All samples should be in [0, 1]
        for &sample in &samples {
            assert!(sample >= 0.0 && sample <= 1.0);
        }
        
        let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        
        // Mean should be close to α/(α+β) = 5/7 ≈ 0.714
        assert_abs_diff_eq!(sample_mean, 5.0/7.0, epsilon = 0.1);
    }
    
    #[test]
    fn test_mode() {
        let beta_dist1 = Beta::new(3.0, 2.0).unwrap();
        let beta_dist2 = Beta::new(1.0, 1.0).unwrap();
        
        // Mode = (α-1)/(α+β-2) = 2/3 for Beta(3,2)
        assert_abs_diff_eq!(beta_dist1.mode(), 2.0/3.0, epsilon = 1e-10);
        
        // Beta(1,1) is uniform, mode could be anywhere
        assert_abs_diff_eq!(beta_dist2.mode(), 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_symmetry() {
        let symmetric = Beta::new(2.0, 2.0).unwrap();
        let asymmetric = Beta::new(2.0, 3.0).unwrap();
        
        assert!(symmetric.is_symmetric());
        assert!(!asymmetric.is_symmetric());
    }
}

/// Compute the regularized incomplete beta function I_x(a,b) = B(x;a,b) / B(a,b)
/// where B(x;a,b) is the incomplete beta function and B(a,b) is the complete beta function
fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
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
        1.0 - incomplete_beta_cf(1.0 - x, b, a)
    } else {
        incomplete_beta_cf(x, a, b)
    }
}

/// Compute incomplete beta using continued fraction expansion
fn incomplete_beta_cf(x: f64, a: f64, b: f64) -> f64 {
    use rustlab_special::gamma_functions::lgamma;
    
    // Compute the coefficient
    let ln_beta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let coeff = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;
    
    // Continued fraction: I_x(a,b) = coeff * cf
    let cf = continued_fraction_beta(x, a, b);
    
    coeff * cf
}

/// Continued fraction for incomplete beta function
fn continued_fraction_beta(x: f64, a: f64, b: f64) -> f64 {
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