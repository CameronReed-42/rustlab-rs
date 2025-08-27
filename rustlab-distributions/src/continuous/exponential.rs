//! Exponential distribution
//!
//! The exponential distribution is a continuous distribution that models the time
//! between events in a Poisson process. It is defined by a single parameter:
//! - λ (lambda): rate parameter (events per unit time)
//!
//! The PDF is: f(x) = λ * exp(-λx) for x ≥ 0

use crate::error::{DistributionError, Result};
use crate::traits::{Distribution, ContinuousDistribution, Sampling};
#[cfg(feature = "integration")]
use rustlab_math::VectorF64;
use rand::Rng;
use rand_distr::{Exp as RandExp, Distribution as RandDistribution};

/// Exponential distribution parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialParams {
    /// Rate parameter (λ)
    pub lambda: f64,
}

/// Exponential distribution
#[derive(Debug, Clone)]
pub struct Exponential {
    params: ExponentialParams,
}

impl Exponential {
    /// Create a new exponential distribution
    ///
    /// # Arguments
    /// * `lambda` - Rate parameter (must be positive)
    ///
    /// # Example
    /// ```
    /// use rustlab_distributions::Exponential;
    /// let exp = Exponential::new(2.0).unwrap();
    /// ```
    pub fn new(lambda: f64) -> Result<Self> {
        if lambda <= 0.0 {
            return Err(DistributionError::invalid_parameter(
                "Rate parameter must be positive"
            ));
        }
        if !lambda.is_finite() {
            return Err(DistributionError::invalid_parameter(
                "Rate parameter must be finite"
            ));
        }
        
        Ok(Exponential {
            params: ExponentialParams { lambda },
        })
    }
    
    /// Create a standard exponential distribution (λ = 1)
    ///
    /// # Example
    /// ```
    /// use rustlab_distributions::Exponential;
    /// let exp = Exponential::standard();
    /// ```
    pub fn standard() -> Self {
        Exponential {
            params: ExponentialParams { lambda: 1.0 },
        }
    }
    
    /// Get the rate parameter
    pub fn lambda(&self) -> f64 {
        self.params.lambda
    }
    
    /// Get the scale parameter (1/λ)
    pub fn scale(&self) -> f64 {
        1.0 / self.params.lambda
    }
    
    /// Check if a value is within the support of the distribution
    pub fn is_in_support(&self, x: f64) -> bool {
        x >= 0.0 && x.is_finite()
    }
}

impl Distribution for Exponential {
    type Params = ExponentialParams;
    type Support = f64;
    
    fn new(params: Self::Params) -> Result<Self> {
        Exponential::new(params.lambda)
    }
    
    fn params(&self) -> &Self::Params {
        &self.params
    }
    
    fn mean(&self) -> f64 {
        1.0 / self.params.lambda
    }
    
    fn variance(&self) -> f64 {
        1.0 / (self.params.lambda * self.params.lambda)
    }
}

impl ContinuousDistribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.params.lambda * (-self.params.lambda * x).exp()
        }
    }
    
    fn log_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            self.params.lambda.ln() - self.params.lambda * x
        }
    }
    
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.params.lambda * x).exp()
        }
    }
    
    fn quantile(&self, p: f64) -> Result<f64> {
        if p < 0.0 || p > 1.0 {
            return Err(DistributionError::invalid_parameter(
                "Probability must be between 0 and 1"
            ));
        }
        
        if p == 0.0 {
            Ok(0.0)
        } else if p == 1.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(-(1.0 - p).ln() / self.params.lambda)
        }
    }
}

impl Sampling for Exponential {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let exp_dist = RandExp::new(self.params.lambda).unwrap();
        exp_dist.sample(rng)
    }
    
    #[cfg(feature = "integration")]
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> VectorF64 {
        let exp_dist = RandExp::new(self.params.lambda).unwrap();
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            samples.push(exp_dist.sample(rng));
        }
        VectorF64::from_vec(samples)
    }
    
    #[cfg(feature = "integration")]
    fn sample_into<R: Rng>(&self, rng: &mut R, output: &mut VectorF64) {
        let exp_dist = RandExp::new(self.params.lambda).unwrap();
        for i in 0..output.len() {
            output.set(i, exp_dist.sample(rng)).unwrap();
        }
    }
}

// Additional methods specific to Exponential distribution
impl Exponential {
    /// Calculate the skewness of the distribution
    /// For exponential distributions, skewness is always 2
    pub fn skewness(&self) -> f64 {
        2.0
    }
    
    /// Calculate the kurtosis (excess kurtosis) of the distribution
    /// For exponential distributions, excess kurtosis is always 6
    pub fn kurtosis(&self) -> f64 {
        6.0
    }
    
    /// Calculate the entropy of the distribution
    /// H(X) = 1 - ln(λ)
    pub fn entropy(&self) -> f64 {
        1.0 - self.params.lambda.ln()
    }
    
    /// Calculate the moment generating function at t
    /// MGF(t) = λ/(λ-t) for t < λ
    pub fn mgf(&self, t: f64) -> f64 {
        if t >= self.params.lambda {
            f64::INFINITY
        } else {
            self.params.lambda / (self.params.lambda - t)
        }
    }
    
    /// Calculate the characteristic function at t
    /// CF(t) = λ/(λ-it)
    pub fn cf(&self, t: f64) -> (f64, f64) {
        let denom_real = self.params.lambda;
        let denom_imag = -t;
        let denom_mag_sq = denom_real * denom_real + denom_imag * denom_imag;
        
        let real_part = self.params.lambda * denom_real / denom_mag_sq;
        let imag_part = self.params.lambda * denom_imag / denom_mag_sq;
        
        (real_part, imag_part)
    }
    
    /// Calculate the survival function (1 - CDF)
    pub fn sf(&self, x: f64) -> f64 {
        if x < 0.0 {
            1.0
        } else {
            (-self.params.lambda * x).exp()
        }
    }
    
    /// Calculate the hazard function (PDF / SF)
    /// For exponential, the hazard rate is constant = λ
    pub fn hazard(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.params.lambda
        }
    }
    
    /// Calculate the cumulative hazard function
    pub fn cumulative_hazard(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.params.lambda * x
        }
    }
    
    /// Calculate the n-th moment about the origin
    /// E[X^n] = n! / λ^n
    pub fn moment(&self, n: u32) -> f64 {
        let mut factorial = 1.0;
        for i in 1..=n {
            factorial *= i as f64;
        }
        factorial / self.params.lambda.powi(n as i32)
    }
    
    /// Calculate the n-th central moment
    pub fn central_moment(&self, n: u32) -> f64 {
        match n {
            0 => 1.0,
            1 => 0.0,
            2 => self.variance(),
            3 => 2.0 * self.variance().powf(1.5), // Skewness * std^3
            4 => {
                let var = self.variance();
                9.0 * var * var // (Kurtosis + 3) * var^2
            }
            _ => {
                // For higher moments, use the general formula
                // This is an approximation
                let var = self.variance();
                var.powf(n as f64 / 2.0) * 2.0_f64.powf(n as f64 - 2.0)
            }
        }
    }
    
    /// Calculate the median
    pub fn median(&self) -> f64 {
        self.quantile(0.5).unwrap()
    }
    
    /// Calculate the mode
    /// For exponential distributions, the mode is always 0
    pub fn mode(&self) -> f64 {
        0.0
    }
    
    /// Calculate the interquartile range
    pub fn iqr(&self) -> f64 {
        self.quantile(0.75).unwrap() - self.quantile(0.25).unwrap()
    }
    
    /// Memory-less property: P(X > s + t | X > s) = P(X > t)
    pub fn memoryless_prob(&self, s: f64, t: f64) -> f64 {
        if s < 0.0 || t < 0.0 {
            return 0.0;
        }
        self.sf(t)
    }
}

/// Convenience function to create a standard exponential distribution
pub fn exponential_standard() -> Exponential {
    Exponential::standard()
}

/// Convenience function to create an exponential distribution with specified rate
pub fn exponential(lambda: f64) -> Result<Exponential> {
    Exponential::new(lambda)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_exponential_creation() {
        let exp = Exponential::new(2.0).unwrap();
        assert_eq!(exp.lambda(), 2.0);
        assert_eq!(exp.scale(), 0.5);
        
        let standard = Exponential::standard();
        assert_eq!(standard.lambda(), 1.0);
        
        // Test invalid parameters
        assert!(Exponential::new(0.0).is_err());
        assert!(Exponential::new(-1.0).is_err());
        assert!(Exponential::new(f64::NAN).is_err());
    }
    
    #[test]
    fn test_exponential_properties() {
        let exp = Exponential::new(2.0).unwrap();
        
        // Test mean and variance
        assert_eq!(exp.mean(), 0.5);
        assert_eq!(exp.variance(), 0.25);
        assert_eq!(exp.std(), 0.5);
        
        // Test skewness and kurtosis
        assert_eq!(exp.skewness(), 2.0);
        assert_eq!(exp.kurtosis(), 6.0);
        
        // Test entropy
        let entropy = exp.entropy();
        assert_abs_diff_eq!(entropy, 1.0 - 2.0_f64.ln(), epsilon = 1e-14);
    }
    
    #[test]
    fn test_exponential_pdf() {
        let exp = Exponential::new(1.0).unwrap();
        
        // Test PDF at specific points
        assert_eq!(exp.pdf(-1.0), 0.0);
        assert_eq!(exp.pdf(0.0), 1.0);
        assert_abs_diff_eq!(exp.pdf(1.0), 1.0 / std::f64::consts::E, epsilon = 1e-14);
        
        // Test log PDF
        assert_eq!(exp.log_pdf(-1.0), f64::NEG_INFINITY);
        assert_eq!(exp.log_pdf(0.0), 0.0);
        assert_abs_diff_eq!(exp.log_pdf(1.0), -1.0, epsilon = 1e-14);
    }
    
    #[test]
    fn test_exponential_cdf() {
        let exp = Exponential::new(1.0).unwrap();
        
        // Test CDF bounds
        assert_eq!(exp.cdf(-1.0), 0.0);
        assert_eq!(exp.cdf(0.0), 0.0);
        assert_eq!(exp.cdf(f64::INFINITY), 1.0);
        
        // Test CDF at specific points
        assert_abs_diff_eq!(exp.cdf(1.0), 1.0 - 1.0 / std::f64::consts::E, epsilon = 1e-14);
        
        // Test CDF is monotonic
        let values = [0.0, 0.5, 1.0, 2.0, 5.0];
        for i in 1..values.len() {
            assert!(exp.cdf(values[i]) >= exp.cdf(values[i-1]));
        }
    }
    
    #[test]
    fn test_exponential_quantile() {
        let exp = Exponential::new(2.0).unwrap();
        
        // Test quantile bounds
        assert_eq!(exp.quantile(0.0).unwrap(), 0.0);
        assert_eq!(exp.quantile(1.0).unwrap(), f64::INFINITY);
        
        // Test quantile-CDF consistency
        let test_values = [0.1, 0.25, 0.5, 0.75, 0.9];
        for &p in &test_values {
            let q = exp.quantile(p).unwrap();
            let cdf_q = exp.cdf(q);
            assert_abs_diff_eq!(cdf_q, p, epsilon = 1e-14);
        }
        
        // Test invalid probabilities
        assert!(exp.quantile(-0.1).is_err());
        assert!(exp.quantile(1.1).is_err());
    }
    
    #[test]
    fn test_exponential_survival() {
        let exp = Exponential::new(1.0).unwrap();
        
        // Test survival function
        assert_eq!(exp.sf(-1.0), 1.0);
        assert_eq!(exp.sf(0.0), 1.0);
        assert_abs_diff_eq!(exp.sf(1.0), 1.0 / std::f64::consts::E, epsilon = 1e-14);
        
        // Test SF + CDF = 1
        for x in [0.0, 0.5, 1.0, 2.0, 5.0] {
            assert_abs_diff_eq!(exp.cdf(x) + exp.sf(x), 1.0, epsilon = 1e-14);
        }
    }
    
    #[test]
    fn test_exponential_hazard() {
        let exp = Exponential::new(2.0).unwrap();
        
        // Hazard rate should be constant = λ
        assert_eq!(exp.hazard(-1.0), 0.0);
        assert_eq!(exp.hazard(0.0), 2.0);
        assert_eq!(exp.hazard(1.0), 2.0);
        assert_eq!(exp.hazard(10.0), 2.0);
        
        // Cumulative hazard
        assert_eq!(exp.cumulative_hazard(-1.0), 0.0);
        assert_eq!(exp.cumulative_hazard(0.0), 0.0);
        assert_eq!(exp.cumulative_hazard(5.0), 10.0);
    }
    
    #[test]
    fn test_exponential_moments() {
        let exp = Exponential::new(2.0).unwrap();
        
        // Test raw moments
        assert_eq!(exp.moment(0), 1.0);
        assert_eq!(exp.moment(1), 0.5); // Mean
        assert_eq!(exp.moment(2), 0.5); // E[X^2] = 2/λ^2
        assert_eq!(exp.moment(3), 0.75); // E[X^3] = 6/λ^3 = 6/8 = 0.75
        
        // Test central moments
        assert_eq!(exp.central_moment(0), 1.0);
        assert_eq!(exp.central_moment(1), 0.0);
        assert_eq!(exp.central_moment(2), 0.25); // Variance
    }
    
    #[test]
    fn test_exponential_mgf() {
        let exp = Exponential::new(2.0).unwrap();
        
        // Test MGF at valid points
        assert_eq!(exp.mgf(0.0), 1.0);
        assert_eq!(exp.mgf(1.0), 2.0);
        
        // Test MGF at boundary
        assert_eq!(exp.mgf(2.0), f64::INFINITY);
        assert_eq!(exp.mgf(3.0), f64::INFINITY);
    }
    
    #[test]
    fn test_exponential_descriptive() {
        let exp = Exponential::new(1.0).unwrap();
        
        assert_abs_diff_eq!(exp.median(), 2.0_f64.ln(), epsilon = 1e-14);
        assert_eq!(exp.mode(), 0.0);
        
        let iqr = exp.iqr();
        let expected_iqr = exp.quantile(0.75).unwrap() - exp.quantile(0.25).unwrap();
        assert_abs_diff_eq!(iqr, expected_iqr, epsilon = 1e-14);
    }
    
    #[test]
    fn test_memoryless_property() {
        let exp = Exponential::new(1.0).unwrap();
        
        // P(X > s + t | X > s) = P(X > t)
        let s = 2.0;
        let t = 3.0;
        assert_eq!(exp.memoryless_prob(s, t), exp.sf(t));
    }
}