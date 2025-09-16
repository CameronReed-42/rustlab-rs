//! Core traits for linear models with AI-optimized math-first API design
//!
//! This module defines the fundamental interfaces for all linear regression models in RustLab.
//! The traits are designed to support type inference and prevent common AI code generation errors.

use rustlab_math::{ArrayF64, VectorF64};
use crate::error::Result;

/// Core trait for linear models following sklearn-like API with math-first design
/// 
/// # Mathematical Specification
/// 
/// All linear models solve optimization problems of the form:
/// minimize f(β) = L(y, X, β) + R(β)
/// 
/// Where:
/// - L(y, X, β) is the loss function (usually ||y - Xβ||²)
/// - R(β) is the regularization term (0 for OLS, λ||β||² for Ridge, λ||β||₁ for Lasso)
/// - β ∈ ℝᵖ are the coefficients to estimate
/// 
/// # Dimensions
/// 
/// - Input: X (n × p), y (n) where n = samples, p = features
/// - Output: β (p), optionally β₀ ∈ ℝ (intercept)
/// 
/// # For AI Code Generation
/// 
/// - Always use references: `model.fit(&X, &y)` prevents ownership issues
/// - Return type enables inference: `let fitted = model.fit(&X, &y)?`
/// - No explicit typing needed: compiler infers `Self::Fitted` type
/// - Common usage: `LinearRegression::new()` → `.fit(&X, &y)?` → `.predict(&X_test)`
/// 
/// # Example
/// 
/// ```rust
/// use rustlab_linearregression::prelude::*;
/// use rustlab_math::{array64, vec64};
/// 
/// let X = array64![[1.0, 2.0], [3.0, 4.0]];
/// let y = vec64![2.0, 6.0];
/// 
/// let model = LinearRegression::new();
/// let fitted = model.fit(&X, &y)?;  // Type inferred
/// let predictions = fitted.predict(&X);
/// ```
/// 
/// # See Also
/// 
/// - [`FittedModel`]: Interface for making predictions and diagnostics
/// - [`LinearRegression`]: Ordinary Least Squares implementation
/// - [`RidgeRegression`]: L2 regularized regression
pub trait LinearModel {
    /// Type of the fitted model (enables type inference)
    type Fitted: FittedModel;
    
    /// Fit the model to training data using math-first operators
    /// 
    /// # Mathematical Specification
    /// 
    /// Solves the optimization problem:
    /// β* = argmin_{β} ||y - Xβ||² + λ·R(β)
    /// 
    /// Where:
    /// - X ∈ ℝⁿˣᵖ is the design matrix
    /// - y ∈ ℝⁿ is the response vector  
    /// - β ∈ ℝᵖ are the coefficients
    /// - R(β) is model-specific regularization
    /// 
    /// # Dimensions
    /// 
    /// - Input: X (n × p), y (n)
    /// - Constraint: X.nrows() == y.len()
    /// - Output: Fitted model with β (p) coefficients
    /// 
    /// # Complexity
    /// 
    /// - Time: O(p³ + np²) for most methods (normal equations)
    /// - Space: O(p²) for coefficient computation
    /// 
    /// # For AI Code Generation
    /// 
    /// - Always use references: `&X, &y` prevents ownership moves
    /// - Return type inferred: `let fitted = model.fit(&X, &y)?`
    /// - Error handling: Use `?` operator for `Result` types
    /// - Matrix ops use `^`: Inside implementation use `X.transpose() ^ X`
    /// - Common pattern: fit → predict → evaluate metrics
    /// 
    /// # Example
    /// 
    /// ```rust
    /// use rustlab_linearregression::prelude::*;
    /// use rustlab_math::{array64, vec64};
    /// 
    /// let X = array64![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let y = vec64![2.1, 5.9, 10.1];
    /// 
    /// let model = LinearRegression::new();
    /// let fitted = model.fit(&X, &y)?;
    /// 
    /// println!("R² score: {:.3}", fitted.score(&X, &y));
    /// ```
    /// 
    /// # Errors
    /// 
    /// - `DimensionMismatch`: X.nrows() != y.len() - ensure same number of samples
    /// - `RankDeficient`: X'X is singular - remove correlated features or use regularization
    /// - `InsufficientData`: n < p - need more samples than features for unique solution
    /// 
    /// # See Also
    /// 
    /// - [`predict`]: Make predictions with fitted model
    /// - [`score`]: Evaluate model performance with R²
    /// - [`fit_weighted`]: Weighted regression for heteroscedastic errors
    fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<Self::Fitted>;
    
    /// Fit model with sample weights for weighted least squares
    /// 
    /// # Mathematical Specification
    /// 
    /// Solves: β* = argmin_{β} Σᵢ wᵢ(yᵢ - xᵢ'β)² + λ·R(β)
    /// 
    /// Equivalent to solving: β* = (X'WX + λR)⁻¹X'Wy
    /// where W = diag(w₁, w₂, ..., wₙ)
    /// 
    /// # Dimensions
    /// 
    /// - Input: X (n × p), y (n), weights (n)
    /// - Constraint: All inputs must have same n
    /// - Output: Same as standard fit
    /// 
    /// # For AI Code Generation
    /// 
    /// - All weights must be positive: `weights > 0`
    /// - Higher weights = more influence on fit
    /// - Use when observations have different uncertainties
    /// - Default implementation: may ignore weights (check model docs)
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let X = array64![[1.0], [2.0], [3.0]];
    /// let y = vec64![1.0, 2.0, 10.0];  // Last point is outlier
    /// let weights = vec64![1.0, 1.0, 0.1];  // Downweight outlier
    /// 
    /// let fitted = LinearRegression::new().fit_weighted(&X, &y, &weights)?;
    /// ```
    fn fit_weighted(&self, X: &ArrayF64, y: &VectorF64, _weights: &VectorF64) -> Result<Self::Fitted> {
        // Default implementation ignores weights
        self.fit(X, y)
    }
}

/// Trait for fitted models with prediction and diagnostic capabilities
/// 
/// # Mathematical Specification
/// 
/// A fitted model represents the solution β* to the optimization problem
/// and provides methods for prediction and statistical inference.
/// 
/// # For AI Code Generation
/// 
/// - Predictions use matrix ops: `X ^ β` (not `X * β`)
/// - Always pass references: `fitted.predict(&X_test)`
/// - R² score in [0, 1]: higher is better (1 = perfect fit)
/// - Coefficients returned by reference: no copying needed
/// - Statistical methods return `Option<T>`: check availability first
/// 
/// # Example
/// 
/// ```rust
/// let fitted = model.fit(&X, &y)?;
/// let predictions = fitted.predict(&X_test);
/// let r2 = fitted.score(&X_test, &y_test);
/// let beta = fitted.coefficients();  // Reference to coefficients
/// ```
pub trait FittedModel {
    /// Make predictions using the fitted model with math-first operations
    /// 
    /// # Mathematical Specification
    /// 
    /// Computes predictions using:
    /// ŷ = Xβ (no intercept) or ŷ = Xβ + β₀ (with intercept)
    /// 
    /// Where:
    /// - X ∈ ℝᵐˣᵖ is the test feature matrix
    /// - β ∈ ℝᵖ are the fitted coefficients  
    /// - β₀ ∈ ℝ is the intercept (if fitted)
    /// - ŷ ∈ ℝᵐ are the predictions
    /// 
    /// # Dimensions
    /// 
    /// - Input: X (m × p) where p matches training features
    /// - Output: predictions (m)
    /// - Constraint: X.ncols() == training_features
    /// 
    /// # Complexity
    /// 
    /// - Time: O(mp) for matrix-vector multiplication
    /// - Space: O(m) for output vector
    /// 
    /// # For AI Code Generation
    /// 
    /// - Always use reference: `fitted.predict(&X_test)`
    /// - Matrix ops use `^`: internally `X ^ coefficients`
    /// - Return type inferred: `let predictions = fitted.predict(&X)`
    /// - Broadcasting handles intercept automatically
    /// - Common pattern: fit → predict → evaluate
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let X_train = array64![[1.0, 2.0], [3.0, 4.0]];
    /// let y_train = vec64![3.0, 7.0];
    /// let fitted = LinearRegression::new().fit(&X_train, &y_train)?;
    /// 
    /// let X_test = array64![[2.0, 3.0], [4.0, 5.0]];
    /// let predictions = fitted.predict(&X_test);  // Type inferred
    /// println!("Predictions: {:?}", predictions);
    /// ```
    /// 
    /// # Errors
    /// 
    /// - Panics if X.ncols() != fitted_features (dimension mismatch)
    /// 
    /// # See Also
    /// 
    /// - [`predict_interval`]: Predictions with uncertainty bounds
    /// - [`score`]: Evaluate prediction quality with R²
    fn predict(&self, X: &ArrayF64) -> VectorF64;
    
    /// Compute coefficient of determination (R²) for model evaluation
    /// 
    /// # Mathematical Specification
    /// 
    /// R² = 1 - RSS/TSS = 1 - Σᵢ(yᵢ - ŷᵢ)² / Σᵢ(yᵢ - ȳ)²
    /// 
    /// Where:
    /// - RSS = Residual Sum of Squares (unexplained variance)
    /// - TSS = Total Sum of Squares (total variance)  
    /// - ȳ = mean(y) is the sample mean
    /// - ŷᵢ = predictions from model
    /// 
    /// # Interpretation
    /// 
    /// - R² = 1: Perfect fit (model explains all variance)
    /// - R² = 0: Model no better than mean prediction
    /// - R² < 0: Model worse than mean (overfitted or poor model)
    /// - Typical values: 0.7-0.9 good, >0.9 excellent
    /// 
    /// # For AI Code Generation
    /// 
    /// - Higher R² = better model fit
    /// - Use for model comparison: `model_a.score()` vs `model_b.score()`  
    /// - Test set R² typically lower than training R² (generalization gap)
    /// - Quick evaluation: `if fitted.score(&X, &y) > 0.8 { /* good model */ }`
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let train_r2 = fitted.score(&X_train, &y_train);
    /// let test_r2 = fitted.score(&X_test, &y_test);
    /// 
    /// println!("Training R²: {:.3}", train_r2);
    /// println!("Test R²: {:.3}", test_r2);
    /// 
    /// if test_r2 > 0.8 {
    ///     println!("Good model performance!");
    /// }
    /// ```
    fn score(&self, X: &ArrayF64, y: &VectorF64) -> f64;
    
    /// Get fitted coefficients (β) as reference for zero-copy access
    /// 
    /// # Mathematical Specification
    /// 
    /// Returns the solution β* ∈ ℝᵖ to the optimization problem:
    /// β* = argmin_β ||y - Xβ||² + λ·R(β)
    /// 
    /// # Dimensions
    /// 
    /// - Output: β (p) where p = number of features
    /// 
    /// # For AI Code Generation
    /// 
    /// - Returns reference: no copying, efficient access
    /// - Use for inspection: `println!("Coefficients: {:?}", fitted.coefficients())`
    /// - Individual access: `let beta0 = fitted.coefficients()[0]`
    /// - Magnitude indicates feature importance (for standardized features)
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let fitted = model.fit(&X, &y)?;
    /// let beta = fitted.coefficients();  // &VectorF64 - no copy
    /// 
    /// println!("Feature 1 coefficient: {:.3}", beta[0]);
    /// println!("All coefficients: {:?}", beta);
    /// 
    /// // Find most important feature (largest absolute coefficient)
    /// let max_coef = beta.iter().enumerate()
    ///     .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
    ///     .unwrap();
    /// println!("Most important feature: {}", max_coef.0);
    /// ```
    fn coefficients(&self) -> &VectorF64;
    
    /// Get intercept term (β₀) if model was fitted with intercept
    /// 
    /// # Mathematical Specification
    /// 
    /// Returns β₀ ∈ ℝ such that predictions are:
    /// ŷ = Xβ + β₀
    /// 
    /// Returns None if fit_intercept=false was used.
    /// 
    /// # For AI Code Generation
    /// 
    /// - Check availability: `if let Some(b0) = fitted.intercept() { ... }`
    /// - Default handling: `fitted.intercept().unwrap_or(0.0)`
    /// - Interpretation: value when all features = 0
    /// 
    /// # Example
    /// 
    /// ```rust
    /// match fitted.intercept() {
    ///     Some(intercept) => println!("y-intercept: {:.3}", intercept),
    ///     None => println!("Model fitted without intercept"),
    /// }
    /// ```
    fn intercept(&self) -> Option<f64>;
    
    /// Generate predictions with confidence intervals for uncertainty quantification
    /// 
    /// # Mathematical Specification
    /// 
    /// Returns (ŷ, ŷ_lower, ŷ_upper) where:
    /// - ŷ = Xβ̂ are point predictions  
    /// - [ŷ_lower, ŷ_upper] are (1-α)100% confidence intervals
    /// - Uses t-distribution with (n-p) degrees of freedom
    /// 
    /// # Parameters
    /// 
    /// - alpha: Significance level (e.g., 0.05 for 95% confidence)
    /// 
    /// # For AI Code Generation
    /// 
    /// - Returns tuple: `let (pred, lower, upper) = fitted.predict_interval(&X, 0.05)?`
    /// - Use for uncertainty: wider intervals = less certain predictions
    /// - Plot intervals: `plot(pred, fill_between=[lower, upper])`
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let (predictions, lower, upper) = fitted.predict_interval(&X_test, 0.05)?;
    /// 
    /// for i in 0..predictions.len() {
    ///     println!("Prediction {}: {:.2} [{:.2}, {:.2}]", 
    ///              i, predictions[i], lower[i], upper[i]);
    /// }
    /// ```
    fn predict_interval(&self, X: &ArrayF64, alpha: f64) -> Result<(VectorF64, VectorF64, VectorF64)>;
    
    /// Get residuals (y - ŷ) from training data if available
    /// 
    /// # Mathematical Specification
    /// 
    /// residuals = y - ŷ = y - Xβ̂
    /// 
    /// Used for diagnostic plots and model validation.
    /// 
    /// # For AI Code Generation
    /// 
    /// - Returns Option: `if let Some(residuals) = fitted.residuals() { ... }`
    /// - Use for diagnostics: check for patterns, outliers
    /// - Should be randomly scattered around 0 for good fit
    /// 
    /// # Example
    /// 
    /// ```rust
    /// if let Some(residuals) = fitted.residuals() {
    ///     let mean_residual = residuals.mean();
    ///     println!("Mean residual: {:.6}", mean_residual);  // Should be ~0
    /// }
    /// ```
    fn residuals(&self) -> Option<&VectorF64>;
    
    /// Get standard errors of coefficients for statistical inference
    /// 
    /// # Mathematical Specification
    /// 
    /// SE(β̂ⱼ) = sqrt(σ̂² · (X'X)⁻¹ⱼⱼ)
    /// 
    /// Where σ̂² is the residual variance estimate.
    /// 
    /// # For AI Code Generation
    /// 
    /// - Returns Option: check availability first
    /// - Used with coefficients: `coef[i] ± 1.96 * se[i]` for 95% CI
    /// - Smaller SE = more precise coefficient estimate
    fn standard_errors(&self) -> Option<&VectorF64>;
    
    /// Get t-statistics for hypothesis testing H₀: βⱼ = 0
    /// 
    /// # Mathematical Specification
    /// 
    /// t-stat = β̂ⱼ / SE(β̂ⱼ)
    /// 
    /// Follows t-distribution with (n-p-1) degrees of freedom under H₀.
    /// 
    /// # For AI Code Generation
    /// 
    /// - Large |t-stat| indicates significant coefficient
    /// - Rule of thumb: |t| > 2 suggests significance (p < 0.05)
    /// 
    /// # Example
    /// 
    /// ```rust
    /// if let Some(t_stats) = fitted.t_statistics() {
    ///     for (i, &t) in t_stats.iter().enumerate() {
    ///         if t.abs() > 2.0 {
    ///             println!("Feature {} is significant (t = {:.2})", i, t);
    ///         }
    ///     }
    /// }
    /// ```
    fn t_statistics(&self) -> Option<VectorF64>;
    
    /// Get p-values for coefficient significance testing
    /// 
    /// # Mathematical Specification
    /// 
    /// p-value = P(|T| > |t-observed|) where T ~ t(n-p-1)
    /// 
    /// Tests H₀: βⱼ = 0 vs H₁: βⱼ ≠ 0
    /// 
    /// # For AI Code Generation
    /// 
    /// - Small p-value (< 0.05) = reject H₀, coefficient is significant
    /// - Use for feature selection: keep features with p < 0.05
    /// - Multiple testing: consider Bonferroni correction
    /// 
    /// # Example
    /// 
    /// ```rust
    /// if let Some(p_vals) = fitted.p_values() {
    ///     let significant_features: Vec<_> = p_vals.iter().enumerate()
    ///         .filter(|(_, &p)| p < 0.05)
    ///         .collect();
    ///     
    ///     println!("Significant features: {:?}", significant_features);
    /// }
    /// ```
    fn p_values(&self) -> Option<VectorF64>;
}

/// Trait for models supporting regularization (Ridge, Lasso, Elastic Net)
/// 
/// # Mathematical Specification
/// 
/// Regularized models solve:
/// β* = argmin_{β} ||y - Xβ||² + α·R(β)
/// 
/// Where R(β) depends on the regularization type:
/// - Ridge: R(β) = ||β||₂² = Σβⱼ²
/// - Lasso: R(β) = ||β||₁ = Σ|βⱼ|  
/// - Elastic Net: R(β) = ρ||β||₁ + (1-ρ)||β||₂²
/// 
/// # For AI Code Generation
/// 
/// - Higher α = more regularization = simpler model
/// - α = 0 reduces to OLS (no regularization)
/// - Use cross-validation to select optimal α
/// - Ridge: α ∈ [0.01, 100], Lasso: α ∈ [0.001, 10]
/// 
/// # Example
/// 
/// ```rust
/// let mut model = RidgeRegression::new();
/// model.set_alpha(1.0)?;
/// let fitted = model.fit(&X, &y)?;
/// ```
pub trait RegularizedModel: LinearModel {
    /// Set regularization strength (α/λ parameter)
    /// 
    /// # Mathematical Effect
    /// 
    /// Controls the trade-off between fit and complexity:
    /// - α = 0: No regularization (OLS solution)
    /// - α → ∞: Maximum regularization (β → 0)
    /// - Typical values: [0.001, 100] depending on data scale
    /// 
    /// # For AI Code Generation
    /// 
    /// - Use cross-validation to find optimal α
    /// - Start with α = 1.0 as reasonable default
    /// - GridSearchCV can automate α selection
    /// - Return Result: check for invalid α values
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let mut ridge = RidgeRegression::new();
    /// ridge.set_alpha(0.1)?;  // Light regularization
    /// 
    /// let mut lasso = LassoRegression::new();  
    /// lasso.set_alpha(0.01)?; // Moderate regularization
    /// ```
    /// 
    /// # Errors
    /// 
    /// - Returns error if α < 0 (regularization strength must be non-negative)
    fn set_alpha(&mut self, alpha: f64) -> Result<()>;
    
    /// Get current regularization strength
    /// 
    /// # For AI Code Generation
    /// 
    /// - Use for inspection: `println!("Current α: {}", model.alpha())`
    /// - Compare models: `if model_a.alpha() > model_b.alpha() { ... }`
    /// - Track parameter changes during tuning
    fn alpha(&self) -> f64;
    
    /// Set L1 ratio for Elastic Net models (ρ parameter)
    /// 
    /// # Mathematical Specification
    /// 
    /// Elastic Net penalty: α[ρ||β||₁ + (1-ρ)||β||₂²/2]
    /// 
    /// Where:
    /// - ρ = 0: Pure Ridge regression (L2 only)
    /// - ρ = 1: Pure Lasso regression (L1 only)
    /// - 0 < ρ < 1: Combined L1/L2 regularization
    /// 
    /// # For AI Code Generation
    /// 
    /// - Only relevant for ElasticNet models
    /// - Default implementation ignores (OK for Ridge/Lasso)
    /// - Typical values: ρ ∈ [0.1, 0.9]
    /// - Use ρ = 0.5 as balanced default
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let mut elastic_net = ElasticNet::new();
    /// elastic_net.set_alpha(0.1)?;
    /// elastic_net.set_l1_ratio(0.5)?;  // Equal L1/L2 mix
    /// ```
    fn set_l1_ratio(&mut self, _ratio: f64) -> Result<()> {
        Ok(())  // Default: ignore for non-elastic-net models
    }
}

/// Trait for iterative optimization-based models (Lasso, Elastic Net, Logistic)
/// 
/// # Mathematical Context
/// 
/// Some models require iterative algorithms:
/// - Lasso: Coordinate descent (no closed-form solution)
/// - Elastic Net: Coordinate descent 
/// - Logistic: IRLS or gradient methods
/// 
/// # For AI Code Generation
/// 
/// - Higher max_iter = longer training but better convergence
/// - Lower tolerance = more precise but slower convergence
/// - Check n_iter() to detect convergence issues
/// - Default values usually sufficient for most problems
/// 
/// # Example
/// 
/// ```rust
/// let mut lasso = LassoRegression::new();
/// lasso.set_max_iter(1000);    // Increase if not converging
/// lasso.set_tolerance(1e-6);   // Tighten for more precision
/// let fitted = lasso.fit(&X, &y)?;
/// 
/// if let Some(iters) = fitted.n_iter() {
///     println!("Converged in {} iterations", iters);
/// }
/// ```
pub trait IterativeModel: LinearModel {
    /// Set maximum number of iterations for convergence
    /// 
    /// # For AI Code Generation
    /// 
    /// - Default: Usually 1000 iterations sufficient
    /// - Increase if convergence warnings occur
    /// - Very large datasets may need more iterations
    /// - No return value: always succeeds
    /// 
    /// # Common Values
    /// 
    /// - Quick testing: 100-500
    /// - Production: 1000-5000  
    /// - Difficult problems: 10000+
    fn set_max_iter(&mut self, max_iter: usize);
    
    /// Set convergence tolerance for optimization
    /// 
    /// # Mathematical Specification
    /// 
    /// Algorithm stops when ||β_new - β_old||₂ < tolerance
    /// 
    /// # For AI Code Generation
    /// 
    /// - Smaller tolerance = more precise solution
    /// - Default: Usually 1e-4 or 1e-5
    /// - Very tight tolerance may cause slow convergence
    /// 
    /// # Common Values
    /// 
    /// - Quick testing: 1e-3
    /// - Standard: 1e-4 to 1e-5
    /// - High precision: 1e-6 or smaller
    fn set_tolerance(&mut self, tol: f64);
    
    /// Get number of iterations used in the last fit
    /// 
    /// # For AI Code Generation
    /// 
    /// - Returns None if model not yet fitted
    /// - Use to check convergence: compare with max_iter
    /// - If n_iter == max_iter, may need more iterations
    /// 
    /// # Example
    /// 
    /// ```rust
    /// let fitted = model.fit(&X, &y)?;
    /// 
    /// match fitted.n_iter() {
    ///     Some(iters) if iters < model.max_iter() => {
    ///         println!("Converged in {} iterations", iters);
    ///     }
    ///     Some(iters) => {
    ///         println!("Warning: Max iterations reached ({})", iters);
    ///     }
    ///     None => {
    ///         println!("Model not fitted or iterations not tracked");
    ///     }
    /// }
    /// ```
    fn n_iter(&self) -> Option<usize>;
}