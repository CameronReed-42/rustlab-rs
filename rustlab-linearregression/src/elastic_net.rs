//! Elastic Net (combined L1/L2 regularization)
//! 
//! Mathematical formulation:
//! β = argmin ||y - Xβ||² + α[ρ||β||₁ + (1-ρ)||β||²₂]

use rustlab_math::{ArrayF64, VectorF64};
use crate::error::{LinearRegressionError, Result};
use crate::traits::{LinearModel, FittedModel, RegularizedModel, IterativeModel};

/// Elastic Net with combined L1/L2 penalties
#[derive(Clone, Debug)]
pub struct ElasticNet {
    alpha: f64,
    l1_ratio: f64,  // ρ in [0, 1]
    max_iter: usize,
    tolerance: f64,
    fit_intercept: bool,
}

impl ElasticNet {
    /// Create a new Elastic Net regression instance
    pub fn new(alpha: f64, l1_ratio: f64) -> Result<Self> {
        if alpha < 0.0 {
            return Err(LinearRegressionError::InvalidParameter {
                name: "alpha".to_string(),
                value: alpha,
                constraint: ">= 0".to_string(),
            });
        }
        
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(LinearRegressionError::InvalidParameter {
                name: "l1_ratio".to_string(),
                value: l1_ratio,
                constraint: "in [0, 1]".to_string(),
            });
        }
        
        Ok(Self {
            alpha,
            l1_ratio,
            max_iter: 1000,
            tolerance: 1e-4,
            fit_intercept: true,
        })
    }
}

/// Fitted Elastic Net model
#[derive(Clone, Debug)]
pub struct FittedElasticNet {
    coefficients: VectorF64,
    intercept: Option<f64>,
    n_iter: usize,
}

impl LinearModel for ElasticNet {
    type Fitted = FittedElasticNet;
    
    fn fit(&self, X: &ArrayF64, _y: &VectorF64) -> Result<Self::Fitted> {
        // TODO: Implement coordinate descent for Elastic Net
        let n_features = X.ncols();
        Ok(FittedElasticNet {
            coefficients: VectorF64::zeros(n_features),
            intercept: Some(0.0),
            n_iter: 0,
        })
    }
}

impl FittedModel for FittedElasticNet {
    fn predict(&self, X: &ArrayF64) -> VectorF64 {
        let n_samples = X.nrows();
        VectorF64::fill(n_samples, self.intercept.unwrap_or(0.0))
    }
    
    fn score(&self, _X: &ArrayF64, _y: &VectorF64) -> f64 {
        0.0
    }
    
    fn coefficients(&self) -> &VectorF64 {
        &self.coefficients
    }
    
    fn intercept(&self) -> Option<f64> {
        self.intercept
    }
    
    fn predict_interval(&self, _X: &ArrayF64, _alpha: f64) -> Result<(VectorF64, VectorF64, VectorF64)> {
        Err(LinearRegressionError::NotFitted { operation: "predict_interval".to_string() })
    }
    
    fn residuals(&self) -> Option<&VectorF64> { None }
    fn standard_errors(&self) -> Option<&VectorF64> { None }
    fn t_statistics(&self) -> Option<VectorF64> { None }
    fn p_values(&self) -> Option<VectorF64> { None }
}

impl RegularizedModel for ElasticNet {
    fn set_alpha(&mut self, alpha: f64) -> Result<()> {
        if alpha < 0.0 {
            return Err(LinearRegressionError::InvalidParameter {
                name: "alpha".to_string(),
                value: alpha,
                constraint: ">= 0".to_string(),
            });
        }
        self.alpha = alpha;
        Ok(())
    }
    
    fn alpha(&self) -> f64 {
        self.alpha
    }
    
    fn set_l1_ratio(&mut self, ratio: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&ratio) {
            return Err(LinearRegressionError::InvalidParameter {
                name: "l1_ratio".to_string(),
                value: ratio,
                constraint: "in [0, 1]".to_string(),
            });
        }
        self.l1_ratio = ratio;
        Ok(())
    }
}

impl IterativeModel for ElasticNet {
    fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }
    
    fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }
    
    fn n_iter(&self) -> Option<usize> {
        None
    }
}