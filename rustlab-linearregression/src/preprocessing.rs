//! Feature preprocessing utilities with math-first design

use rustlab_math::{ArrayF64, VectorF64, BasicStatistics};
use crate::error::{LinearRegressionError, Result};

/// Polynomial features transformation
/// 
/// Generates polynomial features up to specified degree
#[derive(Clone, Debug)]
pub struct PolynomialFeatures {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
}

impl PolynomialFeatures {
    /// Create a new polynomial feature generator
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            include_bias: true,
            interaction_only: false,
        }
    }
    
    /// Transform input features into polynomial features
    pub fn transform(&self, X: &ArrayF64) -> Result<ArrayF64> {
        // TODO: Implement polynomial feature generation
        Ok(X.clone())
    }
}

/// Standard scaler for feature normalization
/// 
/// Transforms features to zero mean and unit variance
#[derive(Clone, Debug)]
pub struct StandardScaler {
    means: Option<VectorF64>,
    stds: Option<VectorF64>,
}

impl StandardScaler {
    /// Create a new standard scaler
    pub fn new() -> Self {
        Self {
            means: None,
            stds: None,
        }
    }
    
    /// Fit scaler to data
    pub fn fit(&mut self, X: &ArrayF64) -> Result<()> {
        let n_features = X.ncols();
        let mut means = VectorF64::zeros(n_features);
        let mut stds = VectorF64::zeros(n_features);
        
        for j in 0..n_features {
            if let Some(col) = X.col(j) {
                means[j] = col.mean();
                stds[j] = col.std(None);
                if stds[j] < 1e-10 {
                    stds[j] = 1.0;
                }
            }
        }
        
        self.means = Some(means);
        self.stds = Some(stds);
        Ok(())
    }
    
    /// Transform data using fitted parameters
    pub fn transform(&self, X: &ArrayF64) -> Result<ArrayF64> {
        let (means, stds) = match (&self.means, &self.stds) {
            (Some(m), Some(s)) => (m, s),
            _ => return Err(LinearRegressionError::NotFitted {
                operation: "transform".to_string(),
            }),
        };
        
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let mut X_scaled = ArrayF64::zeros(n_samples, n_features);
        
        for i in 0..n_samples {
            for j in 0..n_features {
                X_scaled[(i, j)] = (X[(i, j)] - means[j]) / stds[j];
            }
        }
        
        Ok(X_scaled)
    }
    
    /// Fit and transform in one step
    pub fn fit_transform(&mut self, X: &ArrayF64) -> Result<ArrayF64> {
        self.fit(X)?;
        self.transform(X)
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}