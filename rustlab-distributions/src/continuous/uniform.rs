//! Uniform distribution
//!
//! The uniform distribution is a continuous distribution with constant probability
//! density over a specified interval [a, b]. It is defined by two parameters:
//! - a: lower bound
//! - b: upper bound
//!
//! The PDF is: f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise

use crate::error::{DistributionError, Result};
use crate::traits::{Distribution, ContinuousDistribution, Sampling};
#[cfg(feature = "integration")]
use rustlab_math::VectorF64;
use rand::Rng;
use rand_distr::{Uniform as RandUniform, Distribution as RandDistribution};

/// Uniform distribution parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformParams {
    /// Lower bound (a)
    pub a: f64,
    /// Upper bound (b)
    pub b: f64,
}

/// Uniform distribution
#[derive(Debug, Clone)]
pub struct Uniform {
    params: UniformParams,
    range: f64,
}

impl Uniform {
    /// Create a new uniform distribution
    ///
    /// # Arguments
    /// * `a` - Lower bound of the distribution
    /// * `b` - Upper bound of the distribution (must be greater than a)
    ///
    /// # Example
    /// ```
    /// use rustlab_distributions::Uniform;
    /// let uniform = Uniform::new(0.0, 1.0).unwrap();
    /// ```
    pub fn new(a: f64, b: f64) -> Result<Self> {
        if b <= a {
            return Err(DistributionError::invalid_parameter(
                "Upper bound must be greater than lower bound"
            ));
        }
        if !a.is_finite() || !b.is_finite() {
            return Err(DistributionError::invalid_parameter(
                "Both bounds must be finite"
            ));
        }
        
        Ok(Uniform {
            params: UniformParams { a, b },
            range: b - a,
        })
    }
    
    /// Create a standard uniform distribution on [0, 1]
    ///
    /// # Example
    /// ```
    /// use rustlab_distributions::Uniform;
    /// let std_uniform = Uniform::standard();
    /// ```
    pub fn standard() -> Self {
        Uniform {
            params: UniformParams { a: 0.0, b: 1.0 },
            range: 1.0,
        }
    }
    
    /// Get the lower bound parameter
    pub fn a(&self) -> f64 {
        self.params.a
    }
    
    /// Get the upper bound parameter
    pub fn b(&self) -> f64 {
        self.params.b
    }
    
    /// Get the range (b - a)
    pub fn range(&self) -> f64 {
        self.range
    }
    
    /// Check if a value is within the support of the distribution
    pub fn is_in_support(&self, x: f64) -> bool {
        x >= self.params.a && x <= self.params.b
    }
}

impl Distribution for Uniform {
    type Params = UniformParams;
    type Support = f64;
    
    fn new(params: Self::Params) -> Result<Self> {
        Uniform::new(params.a, params.b)
    }
    
    fn params(&self) -> &Self::Params {
        &self.params
    }
    
    fn mean(&self) -> f64 {
        (self.params.a + self.params.b) / 2.0
    }
    
    fn variance(&self) -> f64 {
        self.range * self.range / 12.0
    }
}

impl ContinuousDistribution for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if self.is_in_support(x) {
            1.0 / self.range
        } else {
            0.0
        }
    }
    
    fn log_pdf(&self, x: f64) -> f64 {
        if self.is_in_support(x) {
            -self.range.ln()
        } else {
            f64::NEG_INFINITY
        }
    }
    
    fn cdf(&self, x: f64) -> f64 {
        if x < self.params.a {
            0.0
        } else if x > self.params.b {
            1.0
        } else {
            (x - self.params.a) / self.range
        }
    }
    
    fn quantile(&self, p: f64) -> Result<f64> {
        if p < 0.0 || p > 1.0 {
            return Err(DistributionError::invalid_parameter(
                "Probability must be between 0 and 1"
            ));
        }
        
        Ok(self.params.a + p * self.range)
    }
}

impl Sampling for Uniform {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let uniform_dist = RandUniform::new(self.params.a, self.params.b);
        uniform_dist.sample(rng)
    }
    
    #[cfg(feature = "integration")]
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> VectorF64 {
        let uniform_dist = RandUniform::new(self.params.a, self.params.b);
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            samples.push(uniform_dist.sample(rng));
        }
        VectorF64::from_vec(samples)
    }
    
    #[cfg(feature = "integration")]
    fn sample_into<R: Rng>(&self, rng: &mut R, output: &mut VectorF64) {
        let uniform_dist = RandUniform::new(self.params.a, self.params.b);
        for i in 0..output.len() {
            output.set(i, uniform_dist.sample(rng)).unwrap();
        }
    }
}

// Additional methods specific to Uniform distribution
impl Uniform {
    /// Calculate the skewness of the distribution
    /// For uniform distributions, skewness is always 0
    pub fn skewness(&self) -> f64 {
        0.0
    }
    
    /// Calculate the kurtosis (excess kurtosis) of the distribution
    /// For uniform distributions, excess kurtosis is always -1.2
    pub fn kurtosis(&self) -> f64 {
        -1.2
    }
    
    /// Calculate the entropy of the distribution
    /// H(X) = log(b - a)
    pub fn entropy(&self) -> f64 {
        self.range.ln()
    }
    
    /// Calculate the moment generating function at t
    /// MGF(t) = (e^(bt) - e^(at))/(t(b-a)) for t ≠ 0, 1 for t = 0
    pub fn mgf(&self, t: f64) -> f64 {
        if t.abs() < 1e-10 {
            1.0
        } else {
            ((self.params.b * t).exp() - (self.params.a * t).exp()) / (t * self.range)
        }
    }
    
    /// Calculate the characteristic function at t
    /// CF(t) = (e^(ibt) - e^(iat))/(it(b-a)) for t ≠ 0, 1 for t = 0
    pub fn cf(&self, t: f64) -> (f64, f64) {
        if t.abs() < 1e-10 {
            (1.0, 0.0)
        } else {
            let _exp_ibt = (self.params.b * t).cos() + (self.params.b * t).sin() * std::f64::consts::E;
            let _exp_iat = (self.params.a * t).cos() + (self.params.a * t).sin() * std::f64::consts::E;
            
            let real_part = ((self.params.b * t).sin() - (self.params.a * t).sin()) / (t * self.range);
            let imag_part = ((self.params.a * t).cos() - (self.params.b * t).cos()) / (t * self.range);
            
            (real_part, imag_part)
        }
    }
    
    /// Calculate the survival function (1 - CDF)
    pub fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
    
    /// Calculate the hazard function (PDF / SF)
    pub fn hazard(&self, x: f64) -> f64 {
        if self.is_in_support(x) {
            let sf = self.sf(x);
            if sf > 0.0 {
                self.pdf(x) / sf
            } else {
                f64::INFINITY
            }
        } else {
            0.0
        }
    }
    
    /// Calculate the cumulative hazard function
    pub fn cumulative_hazard(&self, x: f64) -> f64 {
        let sf = self.sf(x);
        if sf > 0.0 {
            -sf.ln()
        } else {
            f64::INFINITY
        }
    }
    
    /// Calculate the n-th moment about the origin
    pub fn moment(&self, n: u32) -> f64 {
        if n == 0 {
            1.0
        } else {
            let n_f = n as f64;
            (self.params.b.powf(n_f + 1.0) - self.params.a.powf(n_f + 1.0)) / ((n_f + 1.0) * self.range)
        }
    }
    
    /// Calculate the n-th central moment
    pub fn central_moment(&self, n: u32) -> f64 {
        match n {
            0 => 1.0,
            1 => 0.0,
            2 => self.variance(),
            3 => 0.0, // Skewness * std^3
            4 => {
                let var = self.variance();
                var * var * (self.kurtosis() + 3.0)
            }
            _ => {
                // For higher moments, use numerical integration approximation
                let mean = self.mean();
                let mut sum = 0.0;
                let num_samples = 10000;
                let step = self.range / num_samples as f64;
                
                for i in 0..num_samples {
                    let x = self.params.a + i as f64 * step;
                    sum += (x - mean).powf(n as f64) * step / self.range;
                }
                sum
            }
        }
    }
    
    /// Calculate the median
    pub fn median(&self) -> f64 {
        self.quantile(0.5).unwrap()
    }
    
    /// Calculate the mode
    /// For uniform distributions, any value in [a, b] is a mode
    pub fn mode(&self) -> f64 {
        self.mean() // Return the mean as a representative mode
    }
    
    /// Calculate the interquartile range
    pub fn iqr(&self) -> f64 {
        self.quantile(0.75).unwrap() - self.quantile(0.25).unwrap()
    }
}

/// Convenience function to create a standard uniform distribution on [0, 1]
pub fn uniform_01() -> Uniform {
    Uniform::standard()
}

/// Convenience function to create a uniform distribution with specified bounds
pub fn uniform(a: f64, b: f64) -> Result<Uniform> {
    Uniform::new(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_uniform_creation() {
        let uniform = Uniform::new(0.0, 1.0).unwrap();
        assert_eq!(uniform.a(), 0.0);
        assert_eq!(uniform.b(), 1.0);
        assert_eq!(uniform.range(), 1.0);
        
        let standard = Uniform::standard();
        assert_eq!(standard.a(), 0.0);
        assert_eq!(standard.b(), 1.0);
        
        // Test invalid parameters
        assert!(Uniform::new(1.0, 1.0).is_err());
        assert!(Uniform::new(2.0, 1.0).is_err());
        assert!(Uniform::new(f64::NAN, 1.0).is_err());
        assert!(Uniform::new(0.0, f64::NAN).is_err());
    }
    
    #[test]
    fn test_uniform_properties() {
        let uniform = Uniform::new(2.0, 8.0).unwrap();
        
        // Test mean and variance
        assert_eq!(uniform.mean(), 5.0);
        assert_abs_diff_eq!(uniform.variance(), 3.0, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.std(), 3.0_f64.sqrt(), epsilon = 1e-14);
        
        // Test skewness and kurtosis
        assert_eq!(uniform.skewness(), 0.0);
        assert_eq!(uniform.kurtosis(), -1.2);
        
        // Test entropy
        let entropy = uniform.entropy();
        assert_abs_diff_eq!(entropy, 6.0_f64.ln(), epsilon = 1e-14);
    }
    
    #[test]
    fn test_uniform_pdf() {
        let uniform = Uniform::new(0.0, 4.0).unwrap();
        
        // Test PDF within support
        assert_abs_diff_eq!(uniform.pdf(0.0), 0.25, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.pdf(2.0), 0.25, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.pdf(4.0), 0.25, epsilon = 1e-14);
        
        // Test PDF outside support
        assert_eq!(uniform.pdf(-1.0), 0.0);
        assert_eq!(uniform.pdf(5.0), 0.0);
        
        // Test log PDF
        assert_abs_diff_eq!(uniform.log_pdf(2.0), (0.25_f64).ln(), epsilon = 1e-14);
        assert_eq!(uniform.log_pdf(-1.0), f64::NEG_INFINITY);
    }
    
    #[test]
    fn test_uniform_cdf() {
        let uniform = Uniform::new(1.0, 5.0).unwrap();
        
        // Test CDF bounds
        assert_eq!(uniform.cdf(0.0), 0.0);
        assert_eq!(uniform.cdf(1.0), 0.0);
        assert_eq!(uniform.cdf(5.0), 1.0);
        assert_eq!(uniform.cdf(6.0), 1.0);
        
        // Test CDF within support
        assert_abs_diff_eq!(uniform.cdf(2.0), 0.25, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.cdf(3.0), 0.5, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.cdf(4.0), 0.75, epsilon = 1e-14);
        
        // Test CDF is monotonic
        let values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5];
        for i in 1..values.len() {
            assert!(uniform.cdf(values[i]) >= uniform.cdf(values[i-1]));
        }
    }
    
    #[test]
    fn test_uniform_quantile() {
        let uniform = Uniform::new(0.0, 10.0).unwrap();
        
        // Test quantile bounds
        assert_eq!(uniform.quantile(0.0).unwrap(), 0.0);
        assert_eq!(uniform.quantile(1.0).unwrap(), 10.0);
        
        // Test quantile within range
        assert_abs_diff_eq!(uniform.quantile(0.25).unwrap(), 2.5, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.quantile(0.5).unwrap(), 5.0, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.quantile(0.75).unwrap(), 7.5, epsilon = 1e-14);
        
        // Test quantile-CDF consistency
        let test_values = [0.1, 0.25, 0.5, 0.75, 0.9];
        for &p in &test_values {
            let q = uniform.quantile(p).unwrap();
            let cdf_q = uniform.cdf(q);
            assert_abs_diff_eq!(cdf_q, p, epsilon = 1e-14);
        }
        
        // Test invalid probabilities
        assert!(uniform.quantile(-0.1).is_err());
        assert!(uniform.quantile(1.1).is_err());
    }
    
    #[test]
    fn test_uniform_support() {
        let uniform = Uniform::new(-2.0, 3.0).unwrap();
        
        assert!(uniform.is_in_support(-2.0));
        assert!(uniform.is_in_support(0.0));
        assert!(uniform.is_in_support(3.0));
        assert!(!uniform.is_in_support(-3.0));
        assert!(!uniform.is_in_support(4.0));
    }
    
    #[test]
    fn test_uniform_moments() {
        let uniform = Uniform::new(0.0, 6.0).unwrap();
        
        // Test raw moments
        assert_abs_diff_eq!(uniform.moment(0), 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.moment(1), 3.0, epsilon = 1e-14); // Mean
        assert_abs_diff_eq!(uniform.moment(2), 12.0, epsilon = 1e-14);
        
        // Test central moments
        assert_abs_diff_eq!(uniform.central_moment(0), 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.central_moment(1), 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.central_moment(2), 3.0, epsilon = 1e-14); // Variance
        assert_abs_diff_eq!(uniform.central_moment(3), 0.0, epsilon = 1e-14);
    }
    
    #[test]
    fn test_uniform_mgf() {
        let uniform = Uniform::new(0.0, 2.0).unwrap();
        
        // Test MGF at t = 0
        assert_abs_diff_eq!(uniform.mgf(0.0), 1.0, epsilon = 1e-14);
        
        // Test MGF at other values
        let t = 1.0;
        let expected = (2.0_f64.exp() - 1.0) / (t * 2.0);
        assert_abs_diff_eq!(uniform.mgf(t), expected, epsilon = 1e-14);
    }
    
    #[test]
    fn test_uniform_descriptive_statistics() {
        let uniform = Uniform::new(10.0, 20.0).unwrap();
        
        assert_abs_diff_eq!(uniform.median(), 15.0, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.mode(), 15.0, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.iqr(), 5.0, epsilon = 1e-14);
    }
    
    #[test]
    fn test_uniform_hazard_functions() {
        let uniform = Uniform::new(0.0, 4.0).unwrap();
        
        // Test survival function
        assert_abs_diff_eq!(uniform.sf(1.0), 0.75, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.sf(2.0), 0.5, epsilon = 1e-14);
        assert_abs_diff_eq!(uniform.sf(3.0), 0.25, epsilon = 1e-14);
        
        // Test hazard function
        let hazard = uniform.hazard(1.0);
        assert_abs_diff_eq!(hazard, 0.25 / 0.75, epsilon = 1e-14);
    }
}