//! Ordinary Least Squares (OLS) regression with AI-optimized math-first API
//! 
//! # Mathematical Specification
//! 
//! OLS finds the best-fitting linear relationship by minimizing squared errors:
//! β* = argmin_β ||y - Xβ||²
//! 
//! Closed-form solution via normal equations:
//! β = (X'X)⁻¹X'y
//! 
//! For numerical stability with ill-conditioned matrices:
//! - QR decomposition: X = QR, then β = R⁻¹Q'y
//! - SVD: X = UΣV', then β = VΣ⁺U'y (pseudoinverse)
//! 
//! # For AI Code Generation
//! 
//! - Use `LinearRegression` alias for intuitive naming
//! - Matrix operations use `^` operator: `X.transpose() ^ X`
//! - Always pass references: `model.fit(&X, &y)`
//! - Type inference works throughout - no explicit types needed
//! - Common pattern: `LinearRegression::new().fit(&X, &y)?`

use rustlab_math::{ArrayF64, VectorF64, BasicStatistics};
use rustlab_linearalgebra::BasicLinearAlgebra;
use crate::error::{LinearRegressionError, Result};
use crate::traits::{LinearModel, FittedModel};

/// Ordinary Least Squares linear regression with sklearn-like API
/// 
/// # Mathematical Specification
/// 
/// Solves the optimization problem:
/// minimize ||y - Xβ||² with respect to β ∈ ℝᵖ
/// 
/// This has the closed-form solution:
/// β = (X'X)⁻¹X'y (normal equations)
/// 
/// # Dimensions
/// 
/// - Input: X (n × p), y (n)
/// - Output: β (p), optionally β₀ (intercept)
/// - Constraint: n > p for unique solution
/// 
/// # For AI Code Generation
/// 
/// - Use `LinearRegression` alias for clarity
/// - Builder pattern for configuration: `.with_intercept(true)`
/// - Default settings work for most cases
/// - Returns `Result` - use `?` operator
/// - Common usage: `LinearRegression::new().fit(&X, &y)?`
/// 
/// # Example
/// 
/// ```rust
/// use rustlab_linearregression::prelude::*;
/// use rustlab_math::{array64, vec64};
/// 
/// let X = array64![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let y = vec64![3.0, 7.0, 11.0];
/// 
/// // Type inference handles everything
/// let model = LinearRegression::new()
///     .with_intercept(true)      // Optional - true by default
///     .with_normalization(false); // Optional - false by default
/// 
/// let fitted = model.fit(&X, &y)?;
/// println!("R² = {:.3}", fitted.r_squared());
/// ```
#[derive(Clone, Debug)]
pub struct OrdinaryLeastSquares {
    fit_intercept: bool,
    normalize: bool,
}

impl OrdinaryLeastSquares {
    /// Create new OLS model with sensible defaults for most use cases
    /// 
    /// # Default Configuration
    /// 
    /// - `fit_intercept = true`: Includes β₀ term (y-intercept)
    /// - `normalize = false`: Features used as-is
    /// 
    /// # For AI Code Generation
    /// 
    /// - This is the starting point for all OLS models
    /// - Defaults work for 90% of use cases
    /// - Chain methods for customization
    /// - Type inference: `let model = LinearRegression::new()`
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let model = LinearRegression::new();  // Ready to use
    /// let fitted = model.fit(&X, &y)?;
    /// ```
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            normalize: false,
        }
    }
    
    /// Configure whether to fit an intercept term (β₀)
    /// 
    /// # Mathematical Effect
    /// 
    /// - `true`: Model is ŷ = β₀ + β₁x₁ + ... + βₚxₚ
    /// - `false`: Model is ŷ = β₁x₁ + ... + βₚxₚ (forced through origin)
    /// 
    /// # When to Use
    /// 
    /// - `true` (default): Most regression problems
    /// - `false`: When relationship must pass through origin (physics laws, proportional relationships)
    /// 
    /// # For AI Code Generation
    /// 
    /// - Builder pattern: returns `Self` for chaining
    /// - Usually keep default (true)
    /// - Only set false for specific domain requirements
    /// 
    /// # Example
    /// 
    /// ```rust
    /// // Force through origin for proportional relationship
    /// let model = LinearRegression::new()
    ///     .with_intercept(false);  // No β₀ term
    /// ```
    pub fn with_intercept(mut self, fit: bool) -> Self {
        self.fit_intercept = fit;
        self
    }
    
    /// Configure feature normalization (standardization)
    /// 
    /// # Mathematical Effect
    /// 
    /// When `true`, transforms each feature:
    /// x'ⱼ = (xⱼ - μⱼ) / σⱼ
    /// 
    /// Where μⱼ = mean(xⱼ), σⱼ = std(xⱼ)
    /// 
    /// # When to Use
    /// 
    /// - `true`: Features have very different scales
    /// - `true`: Comparing coefficient magnitudes
    /// - `false` (default): Features already scaled similarly
    /// 
    /// # For AI Code Generation
    /// 
    /// - Normalization helps with numerical stability
    /// - Coefficients become scale-independent when normalized
    /// - Predictions automatically denormalized
    /// 
    /// # Example
    /// 
    /// ```rust
    /// // Normalize when features have different scales
    /// let model = LinearRegression::new()
    ///     .with_normalization(true);  // Standardize features
    /// ```
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Default for OrdinaryLeastSquares {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for more intuitive naming matching sklearn convention
/// 
/// # For AI Code Generation
/// 
/// - Use this alias instead of `OrdinaryLeastSquares`
/// - More recognizable for Python/sklearn users
/// - Exact same functionality
/// 
/// # Example
/// 
/// ```rust
/// let model = LinearRegression::new();  // Preferred
/// // vs
/// let model = OrdinaryLeastSquares::new();  // Works but less intuitive
/// ```
pub type LinearRegression = OrdinaryLeastSquares;

/// Fitted OLS model containing coefficients and comprehensive diagnostics
/// 
/// # Mathematical Contents
/// 
/// After fitting, contains:
/// - β̂ = (X'X)⁻¹X'y: Estimated coefficients
/// - β̂₀: Intercept (if fitted)
/// - e = y - ŷ: Residuals
/// - SE(β̂): Standard errors of coefficients
/// - R²: Coefficient of determination
/// - R²_adj: Adjusted R² for model complexity
/// 
/// # Statistical Inference
/// 
/// Provides methods for:
/// - Hypothesis testing: t-statistics, p-values
/// - Confidence intervals for predictions
/// - Model diagnostics: residual analysis
/// 
/// # For AI Code Generation
/// 
/// - Access via `model.fit(&X, &y)?` return value
/// - All diagnostics computed during fit (no lazy evaluation)
/// - Use methods like `r_squared()`, `adjusted_r_squared()`
/// - Implements `FittedModel` trait for predictions
/// 
/// # Example
/// 
/// ```rust
/// let fitted = LinearRegression::new().fit(&X, &y)?;
/// 
/// // Access diagnostics
/// println!("R² = {:.3}", fitted.r_squared());
/// println!("Adjusted R² = {:.3}", fitted.adjusted_r_squared());
/// println!("Coefficients: {:?}", fitted.coefficients());
/// 
/// // Make predictions
/// let predictions = fitted.predict(&X_test);
/// 
/// // Statistical inference
/// if let Some(p_values) = fitted.p_values() {
///     for (i, &p) in p_values.iter().enumerate() {
///         if p < 0.05 {
///             println!("Feature {} is significant (p = {:.4})", i, p);
///         }
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct FittedOLS {
    coefficients: VectorF64,
    intercept: Option<f64>,
    residuals: VectorF64,
    standard_errors: VectorF64,
    r_squared: f64,
    adjusted_r_squared: f64,
    n_samples: usize,
    n_features: usize,
    feature_means: Option<VectorF64>,
    feature_stds: Option<VectorF64>,
}

impl LinearModel for OrdinaryLeastSquares {
    type Fitted = FittedOLS;
    
    fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<Self::Fitted> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        
        // Check dimensions
        if n_samples != y.len() {
            return Err(LinearRegressionError::DimensionMismatch {
                x_rows: n_samples,
                y_len: y.len(),
            });
        }
        
        if n_samples <= n_features {
            return Err(LinearRegressionError::InvalidInput(
                format!("Need n_samples > n_features, got {} <= {}", n_samples, n_features)
            ));
        }
        
        // Prepare design matrix
        let (X_design, feature_means, feature_stds) = if self.fit_intercept {
            // Add intercept column using RustLab math operations
            let mut X_with_intercept = ArrayF64::zeros(n_samples, n_features + 1);
            
            // Set first column to 1s (intercept)
            for i in 0..n_samples {
                X_with_intercept[(i, 0)] = 1.0;
            }
            
            // Copy and optionally normalize features
            if self.normalize {
                let mut means = VectorF64::zeros(n_features);
                let mut stds = VectorF64::zeros(n_features);
                
                for j in 0..n_features {
                    // Calculate mean using RustLab operations
                    if let Some(col) = X.col(j) {
                        let mean = col.mean();
                        let std = col.std(None);
                        
                        means[j] = mean;
                        stds[j] = if std > 1e-10 { std } else { 1.0 };
                        
                        // Normalize and copy
                        for i in 0..n_samples {
                            X_with_intercept[(i, j + 1)] = (X[(i, j)] - mean) / stds[j];
                        }
                    }
                }
                
                (X_with_intercept, Some(means), Some(stds))
            } else {
                // Just copy features
                for i in 0..n_samples {
                    for j in 0..n_features {
                        X_with_intercept[(i, j + 1)] = X[(i, j)];
                    }
                }
                (X_with_intercept, None, None)
            }
        } else {
            // No intercept
            if self.normalize {
                let mut X_normalized = ArrayF64::zeros(n_samples, n_features);
                let mut means = VectorF64::zeros(n_features);
                let mut stds = VectorF64::zeros(n_features);
                
                for j in 0..n_features {
                    if let Some(col) = X.col(j) {
                        let mean = col.mean();
                        let std = col.std(None);
                        
                        means[j] = mean;
                        stds[j] = if std > 1e-10 { std } else { 1.0 };
                        
                        for i in 0..n_samples {
                            X_normalized[(i, j)] = (X[(i, j)] - mean) / stds[j];
                        }
                    }
                }
                
                (X_normalized, Some(means), Some(stds))
            } else {
                (X.clone(), None, None)
            }
        };
        
        // Use normal equations instead of QR to avoid the solve assertion error
        // β = (X'X)⁻¹X'y - mathematically equivalent but avoids the problematic QR solve
        let xt = X_design.transpose();
        let xtx = &xt ^ &X_design;
        let y_matrix = ArrayF64::from_vector_column(y);
        let xty = &xt ^ &y_matrix;
        
        // Solve using matrix inversion
        let xtx_inv = xtx.inv()
            .map_err(|e| LinearRegressionError::LinearAlgebra(e.to_string()))?;
        
        let beta_matrix = &xtx_inv ^ &xty;
        
        // Extract coefficients
        let (intercept, coefficients) = if self.fit_intercept {
            let intercept = beta_matrix[(0, 0)];
            let mut coef = VectorF64::zeros(n_features);
            for i in 0..n_features {
                coef[i] = beta_matrix[(i + 1, 0)];
            }
            (Some(intercept), coef)
        } else {
            let mut coef = VectorF64::zeros(n_features);
            for i in 0..n_features {
                coef[i] = beta_matrix[(i, 0)];
            }
            (None, coef)
        };
        
        // Calculate predictions and residuals using math-first operations
        let y_pred = &X_design ^ &beta_matrix;
        
        let mut predictions = VectorF64::zeros(n_samples);
        for i in 0..n_samples {
            predictions[i] = y_pred[(i, 0)];
        }
        
        // Use RustLab's vector operations for residuals
        let residuals = y - &predictions;
        
        // Calculate R² using RustLab operations
        let y_mean = y.mean();
        let squared_residuals = &residuals * &residuals;
        let ss_res = squared_residuals.sum_elements();
        let y_centered = y - y_mean;
        let squared_centered = &y_centered * &y_centered;
        let ss_tot = squared_centered.sum_elements();
        let r_squared = 1.0 - ss_res / ss_tot;
        
        // Adjusted R²
        let adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n_samples as f64 - 1.0) 
            / (n_samples as f64 - n_features as f64 - 1.0);
        
        // Calculate standard errors using residual variance
        let dof = n_samples - n_features - if self.fit_intercept { 1 } else { 0 };
        let mse = ss_res / dof as f64;
        
        // Standard errors from diagonal of (X'X)⁻¹
        let xtx = &X_design.transpose() ^ &X_design;
        
        let xtx_inv = xtx.inv()
            .map_err(|e| LinearRegressionError::LinearAlgebra(e.to_string()))?;
        
        let se_size = if self.fit_intercept { n_features + 1 } else { n_features };
        let mut standard_errors = VectorF64::zeros(se_size);
        for i in 0..standard_errors.len() {
            standard_errors[i] = (mse * xtx_inv[(i, i)]).sqrt();
        }
        
        // Extract standard errors for coefficients only (not intercept)
        let coef_std_errors = if self.fit_intercept {
            let mut coef_se = VectorF64::zeros(n_features);
            for i in 0..n_features {
                coef_se[i] = standard_errors[i + 1];
            }
            coef_se
        } else {
            standard_errors.clone()
        };
        
        Ok(FittedOLS {
            coefficients,
            intercept,
            residuals,
            standard_errors: coef_std_errors,
            r_squared,
            adjusted_r_squared,
            n_samples,
            n_features,
            feature_means,
            feature_stds,
        })
    }
}

impl FittedModel for FittedOLS {
    fn predict(&self, X: &ArrayF64) -> VectorF64 {
        let n_samples = X.nrows();
        let mut predictions = VectorF64::zeros(n_samples);
        
        for i in 0..n_samples {
            let mut pred = self.intercept.unwrap_or(0.0);
            
            for j in 0..self.n_features {
                let x_val = if let (Some(means), Some(stds)) = (&self.feature_means, &self.feature_stds) {
                    // Apply same normalization as during training
                    (X[(i, j)] - means[j]) / stds[j]
                } else {
                    X[(i, j)]
                };
                
                pred += self.coefficients[j] * x_val;
            }
            
            predictions[i] = pred;
        }
        
        predictions
    }
    
    fn score(&self, X: &ArrayF64, y: &VectorF64) -> f64 {
        let predictions = self.predict(X);
        
        // Calculate R² using RustLab math operations
        let residuals = y - &predictions;
        let squared_residuals = &residuals * &residuals;
        let ss_res = squared_residuals.sum_elements();
        
        let y_mean = y.mean();
        let y_centered = y - y_mean;
        let squared_centered = &y_centered * &y_centered;
        let ss_tot = squared_centered.sum_elements();
        
        1.0 - ss_res / ss_tot
    }
    
    fn coefficients(&self) -> &VectorF64 {
        &self.coefficients
    }
    
    fn intercept(&self) -> Option<f64> {
        self.intercept
    }
    
    fn predict_interval(&self, X: &ArrayF64, alpha: f64) -> Result<(VectorF64, VectorF64, VectorF64)> {
        use rustlab_distributions::continuous::StudentT;
        
        let predictions = self.predict(X);
        let n_samples = X.nrows();
        
        // Calculate t-critical value
        let dof = self.n_samples - self.n_features - 1;
        let t_dist = StudentT::new(dof as f64)
            .map_err(|e| LinearRegressionError::Statistics(e.to_string()))?;
        let t_critical = t_dist.inverse_cdf(1.0 - alpha / 2.0)
            .map_err(|e| LinearRegressionError::Statistics(e.to_string()))?;
        
        // Standard error of predictions (simplified - assumes homoscedasticity)
        let squared_residuals = &self.residuals * &self.residuals;
        let mse = squared_residuals.sum_elements() / dof as f64;
        let se = mse.sqrt();
        
        let mut lower = VectorF64::zeros(n_samples);
        let mut upper = VectorF64::zeros(n_samples);
        
        for i in 0..n_samples {
            let margin = t_critical * se;
            lower[i] = predictions[i] - margin;
            upper[i] = predictions[i] + margin;
        }
        
        Ok((predictions, lower, upper))
    }
    
    fn residuals(&self) -> Option<&VectorF64> {
        Some(&self.residuals)
    }
    
    fn standard_errors(&self) -> Option<&VectorF64> {
        Some(&self.standard_errors)
    }
    
    fn t_statistics(&self) -> Option<VectorF64> {
        // t = β / SE(β)
        Some(&self.coefficients / &self.standard_errors)
    }
    
    fn p_values(&self) -> Option<VectorF64> {
        use rustlab_distributions::continuous::StudentT;
        
        let t_stats = self.t_statistics()?;
        let dof = self.n_samples - self.n_features - 1;
        
        let t_dist = StudentT::new(dof as f64).ok()?;
        
        let mut p_values = VectorF64::zeros(self.n_features);
        for i in 0..self.n_features {
            // Two-tailed test
            p_values[i] = 2.0 * (1.0 - t_dist.cdf(t_stats[i].abs()));
        }
        
        Some(p_values)
    }
}

impl FittedOLS {
    /// Get coefficient of determination (R²) measuring model fit quality
    /// 
    /// # Mathematical Specification
    /// 
    /// R² = 1 - RSS/TSS = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
    /// 
    /// # Interpretation
    /// 
    /// - R² ∈ [0, 1] for training data
    /// - R² = 1: Perfect fit (explains all variance)
    /// - R² = 0.8: Model explains 80% of variance
    /// - R² = 0: No better than mean prediction
    /// 
    /// # For AI Code Generation
    /// 
    /// - Higher is better (closer to 1)
    /// - Compare models: higher R² = better fit
    /// - Training R² often optimistic - check test R²
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let fitted = model.fit(&X_train, &y_train)?;
    /// if fitted.r_squared() > 0.7 {
    ///     println!("Good model fit!");
    /// }
    /// ```
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }
    
    /// Get adjusted R² that penalizes model complexity
    /// 
    /// # Mathematical Specification
    /// 
    /// R²_adj = 1 - (1 - R²) × (n - 1) / (n - p - 1)
    /// 
    /// Where n = samples, p = features
    /// 
    /// # When to Use
    /// 
    /// - Comparing models with different numbers of features
    /// - R²_adj < R² always (penalty for complexity)
    /// - Can be negative if model is very poor
    /// 
    /// # For AI Code Generation
    /// 
    /// - Better metric than R² for model selection
    /// - Accounts for overfitting risk
    /// - Use when comparing models with different p
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let simple_model = LinearRegression::new().fit(&X_simple, &y)?;
    /// let complex_model = LinearRegression::new().fit(&X_complex, &y)?;
    /// 
    /// // Compare adjusted R² for fair comparison
    /// if simple_model.adjusted_r_squared() > complex_model.adjusted_r_squared() {
    ///     println!("Simpler model is better!");
    /// }
    /// ```
    pub fn adjusted_r_squared(&self) -> f64 {
        self.adjusted_r_squared
    }
    
    /// Get number of samples (n) used in model fitting
    /// 
    /// # For AI Code Generation
    /// 
    /// - Use for degrees of freedom calculations
    /// - Check sample size adequacy: n > 10p rule of thumb
    /// - Useful for diagnostic calculations
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let fitted = model.fit(&X, &y)?;
    /// let n = fitted.n_samples();
    /// let p = fitted.n_features();
    /// 
    /// if n < 10 * p {
    ///     println!("Warning: May need more samples for reliable estimates");
    /// }
    /// ```
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
    
    /// Get number of features (p) in the model
    /// 
    /// # For AI Code Generation
    /// 
    /// - Does NOT include intercept in count
    /// - Use for degrees of freedom: df = n - p - 1
    /// - Check for overfitting risk when p approaches n
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let fitted = model.fit(&X, &y)?;
    /// println!("Model uses {} features", fitted.n_features());
    /// 
    /// // Calculate degrees of freedom
    /// let df = fitted.n_samples() - fitted.n_features() - 1;
    /// println!("Degrees of freedom: {}", df);
    /// ```
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}