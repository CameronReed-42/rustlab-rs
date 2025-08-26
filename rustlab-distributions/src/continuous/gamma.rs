//! Gamma distribution
//!
//! The gamma distribution is a continuous distribution that generalizes the exponential
//! distribution. It is defined by two parameters:
//! - α (alpha): shape parameter (must be positive)
//! - β (beta): rate parameter (must be positive)
//!
//! The PDF is: f(x) = (β^α / Γ(α)) * x^(α-1) * exp(-βx) for x ≥ 0
//!
//! Note: This uses the rate parameterization. Some sources use scale parameterization
//! where scale = 1/rate.

use crate::error::{DistributionError, Result};
use crate::traits::{Distribution, ContinuousDistribution, Sampling};
#[cfg(feature = "integration")]
use rustlab_math::VectorF64;
use rustlab_special::{gamma as gamma_fn, lgamma, digamma, gamma_p, gamma_q};
use rand::Rng;
use rand_distr::{Gamma as RandGamma, Distribution as RandDistribution};

/// Gamma distribution parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaParams {
    /// Shape parameter (α)
    pub alpha: f64,
    /// Rate parameter (β)
    pub beta: f64,
}

/// Gamma distribution
#[derive(Debug, Clone)]
pub struct Gamma {
    params: GammaParams,
}

impl Gamma {
    /// Create a new gamma distribution
    ///
    /// # Arguments
    /// * `alpha` - Shape parameter (must be positive)
    /// * `beta` - Rate parameter (must be positive)
    ///
    /// # Example
    /// ```
    /// use rustlab_distributions::Gamma;
    /// let gamma_dist = Gamma::new(2.0, 1.0).unwrap();
    /// ```
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(DistributionError::invalid_parameter(
                "Shape parameter must be positive"
            ));
        }
        if beta <= 0.0 {
            return Err(DistributionError::invalid_parameter(
                "Rate parameter must be positive"
            ));
        }
        if !alpha.is_finite() || !beta.is_finite() {
            return Err(DistributionError::invalid_parameter(
                "Parameters must be finite"
            ));
        }
        
        Ok(Gamma {
            params: GammaParams { alpha, beta },
        })
    }
    
    /// Create a gamma distribution from shape and scale parameters
    ///
    /// # Arguments
    /// * `alpha` - Shape parameter (must be positive)
    /// * `scale` - Scale parameter (must be positive)
    ///
    /// Note: scale = 1/rate
    pub fn from_shape_scale(alpha: f64, scale: f64) -> Result<Self> {
        if scale <= 0.0 {
            return Err(DistributionError::invalid_parameter(
                "Scale parameter must be positive"
            ));
        }
        let beta = 1.0 / scale;
        Self::new(alpha, beta)
    }
    
    /// Create a standard gamma distribution (alpha=1, beta=1)
    /// This is equivalent to the exponential distribution with rate=1
    pub fn standard() -> Self {
        Gamma {
            params: GammaParams { alpha: 1.0, beta: 1.0 },
        }
    }
    
    /// Get the shape parameter
    pub fn alpha(&self) -> f64 {
        self.params.alpha
    }
    
    /// Get the rate parameter
    pub fn beta(&self) -> f64 {
        self.params.beta
    }
    
    /// Get the scale parameter (1/rate)
    pub fn scale(&self) -> f64 {
        1.0 / self.params.beta
    }
    
    /// Check if a value is within the support of the distribution
    pub fn is_in_support(&self, x: f64) -> bool {
        x >= 0.0 && x.is_finite()
    }
}

impl Distribution for Gamma {
    type Params = GammaParams;
    type Support = f64;
    
    fn new(params: Self::Params) -> Result<Self> {
        Gamma::new(params.alpha, params.beta)
    }
    
    fn params(&self) -> &Self::Params {
        &self.params
    }
    
    fn mean(&self) -> f64 {
        self.params.alpha / self.params.beta
    }
    
    fn variance(&self) -> f64 {
        self.params.alpha / (self.params.beta * self.params.beta)
    }
}

impl ContinuousDistribution for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || !x.is_finite() {
            return 0.0;
        }
        
        if x == 0.0 {
            return if self.params.alpha < 1.0 {
                f64::INFINITY
            } else if self.params.alpha == 1.0 {
                self.params.beta
            } else {
                0.0
            };
        }
        
        // Use log-space computation for numerical stability
        let log_pdf = self.log_pdf(x);
        if log_pdf.is_finite() {
            log_pdf.exp()
        } else {
            0.0
        }
    }
    
    fn log_pdf(&self, x: f64) -> f64 {
        if x < 0.0 || !x.is_finite() {
            return f64::NEG_INFINITY;
        }
        
        if x == 0.0 {
            return if self.params.alpha < 1.0 {
                f64::INFINITY
            } else if self.params.alpha == 1.0 {
                self.params.beta.ln()
            } else {
                f64::NEG_INFINITY
            };
        }
        
        // log(pdf) = α*log(β) - log(Γ(α)) + (α-1)*log(x) - β*x
        self.params.alpha * self.params.beta.ln() - lgamma(self.params.alpha) 
            + (self.params.alpha - 1.0) * x.ln() - self.params.beta * x
    }
    
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if !x.is_finite() {
            return 1.0;
        }
        
        // Use regularized incomplete gamma function from rustlab-special
        gamma_p(self.params.alpha, self.params.beta * x)
    }
    
    fn quantile(&self, p: f64) -> Result<f64> {
        if p < 0.0 || p > 1.0 {
            return Err(DistributionError::invalid_parameter(
                "Probability must be between 0 and 1"
            ));
        }
        
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        
        // Use Newton-Raphson method for quantile computation
        self.quantile_newton_raphson(p)
    }
}

impl Sampling for Gamma {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let gamma_dist = RandGamma::new(self.params.alpha, 1.0 / self.params.beta).unwrap();
        gamma_dist.sample(rng)
    }
    
    #[cfg(feature = "integration")]
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> VectorF64 {
        let gamma_dist = RandGamma::new(self.params.alpha, 1.0 / self.params.beta).unwrap();
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            samples.push(gamma_dist.sample(rng));
        }
        VectorF64::from_vec(samples)
    }
    
    #[cfg(feature = "integration")]
    fn sample_into<R: Rng>(&self, rng: &mut R, output: &mut VectorF64) {
        let gamma_dist = RandGamma::new(self.params.alpha, 1.0 / self.params.beta).unwrap();
        for i in 0..output.len() {
            output.set(i, gamma_dist.sample(rng)).unwrap();
        }
    }
}

// Additional methods specific to Gamma distribution
impl Gamma {
    /// Calculate the skewness of the distribution
    /// Skewness = 2 / sqrt(α)
    pub fn skewness(&self) -> f64 {
        2.0 / self.params.alpha.sqrt()
    }
    
    /// Calculate the kurtosis (excess kurtosis) of the distribution
    /// Excess kurtosis = 6 / α
    pub fn kurtosis(&self) -> f64 {
        6.0 / self.params.alpha
    }
    
    /// Calculate the entropy of the distribution
    /// H(X) = α - log(β) + log(Γ(α)) + (1-α)*ψ(α)
    pub fn entropy(&self) -> f64 {
        self.params.alpha - self.params.beta.ln() + lgamma(self.params.alpha) 
            + (1.0 - self.params.alpha) * digamma(self.params.alpha)
    }
    
    /// Calculate the moment generating function at t
    /// MGF(t) = (β/(β-t))^α for t < β
    pub fn mgf(&self, t: f64) -> f64 {
        if t >= self.params.beta {
            f64::INFINITY
        } else {
            (self.params.beta / (self.params.beta - t)).powf(self.params.alpha)
        }
    }
    
    /// Calculate the characteristic function at t
    /// CF(t) = (β/(β-it))^α
    pub fn cf(&self, t: f64) -> (f64, f64) {
        let denom_real = self.params.beta;
        let denom_imag = -t;
        let denom_mag_sq = denom_real * denom_real + denom_imag * denom_imag;
        
        let ratio_real = self.params.beta * denom_real / denom_mag_sq;
        let ratio_imag = self.params.beta * denom_imag / denom_mag_sq;
        
        // (ratio_real + i*ratio_imag)^alpha
        let ratio_mag = (ratio_real * ratio_real + ratio_imag * ratio_imag).sqrt();
        let ratio_arg = ratio_imag.atan2(ratio_real);
        
        let result_mag = ratio_mag.powf(self.params.alpha);
        let result_arg = self.params.alpha * ratio_arg;
        
        (result_mag * result_arg.cos(), result_mag * result_arg.sin())
    }
    
    /// Calculate the survival function (1 - CDF)
    pub fn sf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 1.0;
        }
        if !x.is_finite() {
            return 0.0;
        }
        
        // Use complementary regularized incomplete gamma function
        gamma_q(self.params.alpha, self.params.beta * x)
    }
    
    /// Calculate the hazard function (PDF / SF)
    pub fn hazard(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        
        let sf = self.sf(x);
        if sf > 0.0 {
            self.pdf(x) / sf
        } else {
            f64::INFINITY
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
    /// E[X^n] = β^(-n) * Γ(α+n) / Γ(α)
    pub fn moment(&self, n: u32) -> f64 {
        if n == 0 {
            return 1.0;
        }
        
        let n_f = n as f64;
        let numerator = gamma_fn(self.params.alpha + n_f);
        let denominator = gamma_fn(self.params.alpha);
        
        numerator / (denominator * self.params.beta.powf(n_f))
    }
    
    /// Calculate the mode of the distribution
    /// Mode = (α-1)/β for α ≥ 1, 0 for α < 1
    pub fn mode(&self) -> f64 {
        if self.params.alpha >= 1.0 {
            (self.params.alpha - 1.0) / self.params.beta
        } else {
            0.0
        }
    }
    
    /// Calculate the median (approximately)
    /// Uses the Wilson-Hilferty approximation
    pub fn median(&self) -> f64 {
        self.quantile(0.5).unwrap_or_else(|_| {
            // Fallback to Wilson-Hilferty approximation
            let h = 2.0 / (9.0 * self.params.alpha);
            let approx = self.params.alpha * (1.0 - h).powf(3.0);
            approx / self.params.beta
        })
    }
    
    /// Newton-Raphson method for quantile computation
    fn quantile_newton_raphson(&self, p: f64) -> Result<f64> {
        // Initial guess using Wilson-Hilferty approximation
        let _h = 2.0 / (9.0 * self.params.alpha);
        let chi_quantile = self.wilson_hilferty_quantile(p);
        let mut x = (chi_quantile / self.params.beta).max(1e-10);
        
        // Newton-Raphson iterations
        for _ in 0..50 {
            let cdf_x = self.cdf(x);
            let pdf_x = self.pdf(x);
            
            if pdf_x.abs() < 1e-12 {
                break;
            }
            
            let error = cdf_x - p;
            if error.abs() < 1e-12 {
                break;
            }
            
            let new_x = x - error / pdf_x;
            x = new_x.max(1e-10);
        }
        
        Ok(x)
    }
    
    /// Wilson-Hilferty approximation for initial quantile guess
    fn wilson_hilferty_quantile(&self, p: f64) -> f64 {
        // Use standard normal quantile approximation
        let z = self.standard_normal_quantile(p);
        let h = 2.0 / (9.0 * self.params.alpha);
        self.params.alpha * (1.0 - h + z * h.sqrt()).powf(3.0)
    }
    
    /// Approximate standard normal quantile using Beasley-Springer-Moro algorithm
    fn standard_normal_quantile(&self, p: f64) -> f64 {
        let a = [0.0, -3.969683028665376e+01, 2.209460984245205e+02,
                 -2.759285104469687e+02, 1.383577518672690e+02,
                 -3.066479806614716e+01, 2.506628277459239e+00];
        
        let b = [0.0, -5.447609879822406e+01, 1.615858368580409e+02,
                 -1.556989798598866e+02, 6.680131188771972e+01,
                 -1.328068155288572e+01];
        
        let c = [0.0, -7.784894002430293e-03, -3.223964580411365e-01,
                 -2.400758277161838e+00, -2.549732539343734e+00,
                 4.374664141464968e+00, 2.938163982698783e+00];
        
        let d = [0.0, 7.784695709041462e-03, 3.224671290700398e-01,
                 2.445134137142996e+00, 3.754408661907416e+00];
        
        let p_low = 0.02425;
        let p_high = 1.0 - p_low;
        
        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
            ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
            (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
            ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
        }
    }
}

/// Convenience function to create a standard gamma distribution
pub fn gamma_standard() -> Gamma {
    Gamma::standard()
}

/// Convenience function to create a gamma distribution with specified parameters
pub fn gamma(alpha: f64, beta: f64) -> Result<Gamma> {
    Gamma::new(alpha, beta)
}

/// Convenience function to create a gamma distribution from shape and scale
pub fn gamma_from_shape_scale(alpha: f64, scale: f64) -> Result<Gamma> {
    Gamma::from_shape_scale(alpha, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_gamma_creation() {
        let gamma_dist = Gamma::new(2.0, 1.0).unwrap();
        assert_eq!(gamma_dist.alpha(), 2.0);
        assert_eq!(gamma_dist.beta(), 1.0);
        assert_eq!(gamma_dist.scale(), 1.0);
        
        let gamma_from_scale = Gamma::from_shape_scale(2.0, 0.5).unwrap();
        assert_eq!(gamma_from_scale.alpha(), 2.0);
        assert_eq!(gamma_from_scale.beta(), 2.0);
        assert_eq!(gamma_from_scale.scale(), 0.5);
        
        let standard = Gamma::standard();
        assert_eq!(standard.alpha(), 1.0);
        assert_eq!(standard.beta(), 1.0);
        
        // Test invalid parameters
        assert!(Gamma::new(0.0, 1.0).is_err());
        assert!(Gamma::new(-1.0, 1.0).is_err());
        assert!(Gamma::new(1.0, 0.0).is_err());
        assert!(Gamma::new(1.0, -1.0).is_err());
        assert!(Gamma::new(f64::NAN, 1.0).is_err());
        assert!(Gamma::new(1.0, f64::NAN).is_err());
    }
    
    #[test]
    fn test_gamma_properties() {
        let gamma_dist = Gamma::new(2.0, 0.5).unwrap();
        
        // Test mean and variance
        assert_eq!(gamma_dist.mean(), 4.0);
        assert_eq!(gamma_dist.variance(), 8.0);
        assert_abs_diff_eq!(gamma_dist.std(), 8.0_f64.sqrt(), epsilon = 1e-14);
        
        // Test skewness and kurtosis
        assert_abs_diff_eq!(gamma_dist.skewness(), 2.0 / 2.0_f64.sqrt(), epsilon = 1e-14);
        assert_abs_diff_eq!(gamma_dist.kurtosis(), 6.0 / 2.0, epsilon = 1e-14);
        
        // Test mode
        assert_eq!(gamma_dist.mode(), 2.0);
        
        // Test that mode is at 0 for alpha < 1
        let gamma_sub1 = Gamma::new(0.5, 1.0).unwrap();
        assert_eq!(gamma_sub1.mode(), 0.0);
    }
    
    #[test]
    fn test_gamma_pdf() {
        let gamma_dist = Gamma::new(2.0, 1.0).unwrap();
        
        // Test PDF at specific points
        assert_eq!(gamma_dist.pdf(-1.0), 0.0);
        assert_eq!(gamma_dist.pdf(0.0), 0.0);
        
        let pdf_1 = gamma_dist.pdf(1.0);
        let expected_pdf_1 = 1.0 * (-1.0_f64).exp();
        assert_abs_diff_eq!(pdf_1, expected_pdf_1, epsilon = 1e-12);
        
        let pdf_2 = gamma_dist.pdf(2.0);
        let expected_pdf_2 = 2.0 * (-2.0_f64).exp();
        assert_abs_diff_eq!(pdf_2, expected_pdf_2, epsilon = 1e-12);
        
        // Test log PDF
        let log_pdf_1 = gamma_dist.log_pdf(1.0);
        assert_abs_diff_eq!(log_pdf_1, expected_pdf_1.ln(), epsilon = 1e-12);
        
        // Test PDF outside support
        assert_eq!(gamma_dist.pdf(-1.0), 0.0);
        assert_eq!(gamma_dist.log_pdf(-1.0), f64::NEG_INFINITY);
    }
    
    #[test]
    fn test_gamma_cdf() {
        let gamma_dist = Gamma::new(1.0, 1.0).unwrap(); // Exponential distribution
        
        // Test CDF bounds
        assert_eq!(gamma_dist.cdf(-1.0), 0.0);
        assert_eq!(gamma_dist.cdf(f64::INFINITY), 1.0);
        
        // For exponential distribution, CDF(x) = 1 - exp(-x)
        let cdf_1 = gamma_dist.cdf(1.0);
        let expected_cdf_1 = 1.0 - (-1.0_f64).exp();
        assert_abs_diff_eq!(cdf_1, expected_cdf_1, epsilon = 1e-12);
        
        // Test that CDF is monotonic
        let values = [0.0, 0.5, 1.0, 2.0, 5.0];
        for i in 1..values.len() {
            assert!(gamma_dist.cdf(values[i]) >= gamma_dist.cdf(values[i-1]));
        }
    }
    
    #[test]
    fn test_gamma_quantile() {
        let gamma_dist = Gamma::new(1.0, 1.0).unwrap(); // Exponential distribution
        
        // Test quantile bounds
        assert_eq!(gamma_dist.quantile(0.0).unwrap(), 0.0);
        assert_eq!(gamma_dist.quantile(1.0).unwrap(), f64::INFINITY);
        
        // For exponential distribution, quantile(p) = -ln(1-p)
        let q_05 = gamma_dist.quantile(0.5).unwrap();
        let expected_q_05 = -(1.0_f64 - 0.5).ln();
        assert_abs_diff_eq!(q_05, expected_q_05, epsilon = 1e-10);
        
        // Test quantile-CDF consistency
        let test_values = [0.1, 0.25, 0.5, 0.75, 0.9];
        for &p in &test_values {
            let q = gamma_dist.quantile(p).unwrap();
            let cdf_q = gamma_dist.cdf(q);
            assert_abs_diff_eq!(cdf_q, p, epsilon = 1e-10);
        }
        
        // Test invalid probabilities
        assert!(gamma_dist.quantile(-0.1).is_err());
        assert!(gamma_dist.quantile(1.1).is_err());
    }
    
    #[test]
    fn test_gamma_support() {
        let gamma_dist = Gamma::new(2.0, 1.0).unwrap();
        
        assert!(gamma_dist.is_in_support(0.0));
        assert!(gamma_dist.is_in_support(1.0));
        assert!(gamma_dist.is_in_support(100.0));
        assert!(!gamma_dist.is_in_support(-1.0));
        assert!(!gamma_dist.is_in_support(f64::NAN));
        assert!(!gamma_dist.is_in_support(f64::INFINITY));
    }
    
    #[test]
    fn test_gamma_moments() {
        let gamma_dist = Gamma::new(3.0, 2.0).unwrap();
        
        // Test raw moments
        assert_abs_diff_eq!(gamma_dist.moment(0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(gamma_dist.moment(1), 1.5, epsilon = 1e-12); // Mean
        
        // Second moment should be related to variance
        let second_moment = gamma_dist.moment(2);
        let expected_second_moment = gamma_dist.variance() + gamma_dist.mean().powi(2);
        assert_abs_diff_eq!(second_moment, expected_second_moment, epsilon = 1e-12);
    }
    
    #[test]
    fn test_gamma_mgf() {
        let gamma_dist = Gamma::new(2.0, 3.0).unwrap();
        
        // Test MGF at valid points
        let mgf_1 = gamma_dist.mgf(1.0);
        let expected_mgf_1 = (3.0_f64 / 2.0).powf(2.0);
        assert_abs_diff_eq!(mgf_1, expected_mgf_1, epsilon = 1e-12);
        
        // Test MGF at boundary
        assert_eq!(gamma_dist.mgf(3.0), f64::INFINITY);
        assert_eq!(gamma_dist.mgf(4.0), f64::INFINITY);
    }
    
    #[test]
    fn test_gamma_survival_functions() {
        let gamma_dist = Gamma::new(2.0, 1.0).unwrap();
        
        // Test survival function
        let x = 1.0;
        let cdf_x = gamma_dist.cdf(x);
        let sf_x = gamma_dist.sf(x);
        assert_abs_diff_eq!(cdf_x + sf_x, 1.0, epsilon = 1e-12);
        
        // Test hazard function
        let hazard_x = gamma_dist.hazard(x);
        let expected_hazard = gamma_dist.pdf(x) / gamma_dist.sf(x);
        assert_abs_diff_eq!(hazard_x, expected_hazard, epsilon = 1e-12);
    }
    
    #[test]
    fn test_gamma_entropy() {
        let gamma_dist = Gamma::new(2.0, 1.0).unwrap();
        
        // Test that entropy is finite
        let entropy = gamma_dist.entropy();
        assert!(entropy.is_finite());
        assert!(entropy > 0.0); // Gamma distribution should have positive entropy
    }
    
    #[test]
    fn test_gamma_special_cases() {
        // Test alpha = 1 (exponential distribution)
        let exp_dist = Gamma::new(1.0, 2.0).unwrap();
        
        // At x = 0, PDF should equal beta
        let pdf_0 = exp_dist.pdf(0.0);
        assert_abs_diff_eq!(pdf_0, 2.0, epsilon = 1e-12);
        
        // Test alpha < 1 (PDF goes to infinity at x = 0)
        let sub_exp = Gamma::new(0.5, 1.0).unwrap();
        assert_eq!(sub_exp.pdf(0.0), f64::INFINITY);
        assert_eq!(sub_exp.log_pdf(0.0), f64::INFINITY);
    }
}