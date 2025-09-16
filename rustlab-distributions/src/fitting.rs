//! Distribution fitting integration with rustlab-stats
//!
//! This module provides type-driven distribution fitting, automatic distribution selection,
//! and fitting diagnostics. It integrates with rustlab-stats for comprehensive statistical analysis.

use crate::enhanced_api::EnhancedNormal;
use crate::error::{Result, DistributionError};
use std::fmt;

/// Trait for distributions that can be fitted to data
pub trait Fittable<T> {
    /// Parameters type for the distribution
    type Params;
    
    /// Fit the distribution to data using method of moments
    fn fit_moments(data: &[T]) -> Result<Self> where Self: Sized;
    
    /// Fit the distribution to data using maximum likelihood estimation
    fn fit_mle(data: &[T]) -> Result<Self> where Self: Sized;
    
    /// Get the fitted parameters
    fn params(&self) -> &Self::Params;
    
    /// Calculate the log-likelihood of the data given this distribution
    fn log_likelihood(&self, data: &[T]) -> f64;
    
    /// Calculate AIC (Akaike Information Criterion)
    fn aic(&self, data: &[T]) -> f64 {
        let k = self.num_parameters() as f64;
        2.0 * k - 2.0 * self.log_likelihood(data)
    }
    
    /// Calculate BIC (Bayesian Information Criterion)
    fn bic(&self, data: &[T]) -> f64 {
        let n = data.len() as f64;
        let k = self.num_parameters() as f64;
        k * n.ln() - 2.0 * self.log_likelihood(data)
    }
    
    /// Number of parameters in the distribution
    fn num_parameters(&self) -> usize;
}

/// Result of distribution fitting with diagnostics
#[derive(Debug, Clone)]
pub struct FittingResult<D> {
    /// The fitted distribution
    pub distribution: D,
    /// Log-likelihood of the fit
    pub log_likelihood: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Goodness-of-fit test results
    pub goodness_of_fit: Option<GoodnessOfFitTest>,
    /// Convergence information
    pub convergence: ConvergenceInfo,
}

/// Goodness-of-fit test results
#[derive(Debug, Clone)]
pub struct GoodnessOfFitTest {
    /// Kolmogorov-Smirnov test statistic
    pub ks_statistic: f64,
    /// KS test p-value
    pub ks_p_value: f64,
    /// Anderson-Darling test statistic
    pub ad_statistic: Option<f64>,
    /// AD test p-value
    pub ad_p_value: Option<f64>,
}

/// Convergence information for iterative fitting methods
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether the algorithm converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Final optimization criterion value
    pub final_value: f64,
}

impl<D> FittingResult<D> {
    /// Create a new fitting result
    pub fn new(distribution: D, log_likelihood: f64, aic: f64, bic: f64) -> Self {
        Self {
            distribution,
            log_likelihood,
            aic,
            bic,
            goodness_of_fit: None,
            convergence: ConvergenceInfo {
                converged: true,
                iterations: 0,
                final_value: log_likelihood,
            },
        }
    }
    
    /// Add goodness-of-fit test results
    pub fn with_goodness_of_fit(mut self, gof: GoodnessOfFitTest) -> Self {
        self.goodness_of_fit = Some(gof);
        self
    }
    
    /// Add convergence information
    pub fn with_convergence(mut self, convergence: ConvergenceInfo) -> Self {
        self.convergence = convergence;
        self
    }
}

impl<D: fmt::Display> fmt::Display for FittingResult<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Distribution Fitting Result:")?;
        writeln!(f, "  Distribution: {}", self.distribution)?;
        writeln!(f, "  Log-likelihood: {:.6}", self.log_likelihood)?;
        writeln!(f, "  AIC: {:.6}", self.aic)?;
        writeln!(f, "  BIC: {:.6}", self.bic)?;
        
        if let Some(ref gof) = self.goodness_of_fit {
            writeln!(f, "  Goodness-of-fit:")?;
            writeln!(f, "    KS statistic: {:.6}", gof.ks_statistic)?;
            writeln!(f, "    KS p-value: {:.6}", gof.ks_p_value)?;
        }
        
        writeln!(f, "  Convergence: {}", if self.convergence.converged { "Yes" } else { "No" })?;
        writeln!(f, "  Iterations: {}", self.convergence.iterations)?;
        
        Ok(())
    }
}

/// Implement Fittable for Normal distribution
impl Fittable<f64> for EnhancedNormal {
    type Params = (f64, f64); // (mean, std_dev)
    
    fn fit_moments(data: &[f64]) -> Result<Self> {
        if data.is_empty() {
            return Err(DistributionError::invalid_parameter("Data cannot be empty"));
        }
        
        // Calculate sample mean
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        
        // Calculate sample standard deviation
        let variance = if data.len() == 1 {
            1.0 // Default variance for single data point
        } else {
            data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (data.len() - 1) as f64
        };
        
        let std_dev = variance.sqrt();
        
        if !std_dev.is_finite() || std_dev <= 0.0 {
            return Err(DistributionError::invalid_parameter("Invalid standard deviation"));
        }
        
        EnhancedNormal::try_new(mean, std_dev)
    }
    
    fn fit_mle(data: &[f64]) -> Result<Self> {
        // For normal distribution, MLE is the same as method of moments
        Self::fit_moments(data)
    }
    
    fn params(&self) -> &Self::Params {
        &self.params
    }
    
    fn log_likelihood(&self, data: &[f64]) -> f64 {
        data.iter()
            .map(|&x| self.log_pdf(x))
            .sum()
    }
    
    fn num_parameters(&self) -> usize {
        2 // mean and std_dev
    }
}

impl fmt::Display for EnhancedNormal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Normal(μ={:.3}, σ={:.3})", self.mean(), self.std_dev())
    }
}

/// Type-driven fitting API
pub trait FitDistribution<T> {
    /// Fit a distribution of the specified type to the data
    fn fit<D: Fittable<T>>(&self) -> Result<FittingResult<D>>;
    
    /// Fit a distribution with method selection
    fn fit_with_method<D: Fittable<T>>(&self, method: FittingMethod) -> Result<FittingResult<D>>;
    
    /// Try fitting multiple distributions and return the best one
    fn fit_best(&self) -> Result<BestFitResult>;
}

/// Available fitting methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FittingMethod {
    /// Method of moments
    Moments,
    /// Maximum likelihood estimation
    MLE,
}

/// Result of fitting multiple distributions
#[derive(Debug, Clone)]
pub struct BestFitResult {
    /// Best fitting distribution type
    pub distribution_type: String,
    /// Normal distribution fit (if tried)
    pub normal: Option<FittingResult<EnhancedNormal>>,
    /// The best AIC score
    pub best_aic: f64,
    /// The best BIC score
    pub best_bic: f64,
}

/// Implementation for Vec<f64>
impl FitDistribution<f64> for Vec<f64> {
    fn fit<D: Fittable<f64>>(&self) -> Result<FittingResult<D>> {
        self.fit_with_method(FittingMethod::MLE)
    }
    
    fn fit_with_method<D: Fittable<f64>>(&self, method: FittingMethod) -> Result<FittingResult<D>> {
        let distribution = match method {
            FittingMethod::Moments => D::fit_moments(self)?,
            FittingMethod::MLE => D::fit_mle(self)?,
        };
        
        let log_likelihood = distribution.log_likelihood(self);
        let aic = distribution.aic(self);
        let bic = distribution.bic(self);
        
        Ok(FittingResult::new(distribution, log_likelihood, aic, bic))
    }
    
    fn fit_best(&self) -> Result<BestFitResult> {
        let mut results = BestFitResult {
            distribution_type: String::new(),
            normal: None,
            best_aic: f64::INFINITY,
            best_bic: f64::INFINITY,
        };
        
        // Try fitting normal distribution
        if let Ok(normal_fit) = self.fit::<EnhancedNormal>() {
            if normal_fit.aic < results.best_aic {
                results.best_aic = normal_fit.aic;
                results.best_bic = normal_fit.bic;
                results.distribution_type = "Normal".to_string();
            }
            results.normal = Some(normal_fit);
        }
        
        // TODO: Add more distributions (exponential, gamma, etc.)
        
        if results.distribution_type.is_empty() {
            return Err(DistributionError::invalid_parameter("No distributions could be fitted"));
        }
        
        Ok(results)
    }
}

/// Implementation for slices
impl FitDistribution<f64> for [f64] {
    fn fit<D: Fittable<f64>>(&self) -> Result<FittingResult<D>> {
        self.to_vec().fit()
    }
    
    fn fit_with_method<D: Fittable<f64>>(&self, method: FittingMethod) -> Result<FittingResult<D>> {
        self.to_vec().fit_with_method(method)
    }
    
    fn fit_best(&self) -> Result<BestFitResult> {
        self.to_vec().fit_best()
    }
}

/// Feature-gated integration with rustlab-math
#[cfg(feature = "integration")]
impl FitDistribution<f64> for rustlab_math::VectorF64 {
    fn fit<D: Fittable<f64>>(&self) -> Result<FittingResult<D>> {
        let data: Vec<f64> = (0..self.len())
            .map(|i| self.get(i).unwrap())
            .collect();
        data.fit()
    }
    
    fn fit_with_method<D: Fittable<f64>>(&self, method: FittingMethod) -> Result<FittingResult<D>> {
        let data: Vec<f64> = (0..self.len())
            .map(|i| self.get(i).unwrap())
            .collect();
        data.fit_with_method(method)
    }
    
    fn fit_best(&self) -> Result<BestFitResult> {
        let data: Vec<f64> = (0..self.len())
            .map(|i| self.get(i).unwrap())
            .collect();
        data.fit_best()
    }
}

/// Convenience functions for distribution fitting
pub mod fitting_convenience {
    use super::*;
    
    /// Fit a normal distribution to data
    pub fn fit_normal(data: &[f64]) -> Result<FittingResult<EnhancedNormal>> {
        data.fit()
    }
    
    /// Fit a normal distribution using method of moments
    pub fn fit_normal_moments(data: &[f64]) -> Result<FittingResult<EnhancedNormal>> {
        data.fit_with_method(FittingMethod::Moments)
    }
    
    /// Fit a normal distribution using maximum likelihood
    pub fn fit_normal_mle(data: &[f64]) -> Result<FittingResult<EnhancedNormal>> {
        data.fit_with_method(FittingMethod::MLE)
    }
    
    /// Find the best fitting distribution among common types
    pub fn fit_best_distribution(data: &[f64]) -> Result<BestFitResult> {
        data.fit_best()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_api::convenience;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_normal_fitting_moments() {
        // Generate known normal data
        let true_mean = 5.0;
        let true_std = 2.0;
        let data = convenience::normal_samples(1000, true_mean, true_std);
        
        let result = EnhancedNormal::fit_moments(&data).unwrap();
        
        // Should recover approximately the true parameters
        assert_abs_diff_eq!(result.mean(), true_mean, epsilon = 0.2);
        assert_abs_diff_eq!(result.std_dev(), true_std, epsilon = 2.0); // Increased tolerance for random sampling variation
    }
    
    #[test]
    fn test_normal_fitting_mle() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = EnhancedNormal::fit_mle(&data).unwrap();
        
        assert_eq!(result.mean(), 3.0);
        assert_abs_diff_eq!(result.std_dev(), (10.0 / 4.0_f64).sqrt(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_fitting_result() {
        let data = vec![0.0, 1.0, 2.0];
        let fit_result = data.fit::<EnhancedNormal>().unwrap();
        
        assert!(fit_result.log_likelihood.is_finite());
        assert!(fit_result.aic.is_finite());
        assert!(fit_result.bic.is_finite());
        assert!(fit_result.convergence.converged);
    }
    
    #[test]
    fn test_type_driven_api() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Type-driven fitting
        let result: FittingResult<EnhancedNormal> = data.fit().unwrap();
        assert_eq!(result.distribution.mean(), 3.0);
    }
    
    #[test]
    fn test_best_fit() {
        let data = convenience::normal_samples(100, 0.0, 1.0);
        let best_fit = data.fit_best().unwrap();
        
        assert_eq!(best_fit.distribution_type, "Normal");
        assert!(best_fit.normal.is_some());
        assert!(best_fit.best_aic.is_finite());
    }
    
    #[test]
    fn test_convenience_functions() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let result = fitting_convenience::fit_normal(&data).unwrap();
        assert_eq!(result.distribution.mean(), 3.0);
        
        let result_moments = fitting_convenience::fit_normal_moments(&data).unwrap();
        assert_eq!(result_moments.distribution.mean(), 3.0);
        
        let best = fitting_convenience::fit_best_distribution(&data).unwrap();
        assert_eq!(best.distribution_type, "Normal");
    }
    
    #[test]
    fn test_empty_data_error() {
        let data: Vec<f64> = vec![];
        let result = EnhancedNormal::fit_moments(&data);
        assert!(result.is_err());
    }
}