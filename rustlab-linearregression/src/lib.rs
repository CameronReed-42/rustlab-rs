//! # RustLab Linear Regression
//! 
//! High-performance linear regression library optimized for AI code generation with math-first API design.

#![allow(non_snake_case)]  // Allow mathematical notation (X for matrices, y for vectors)
//! 
//! ## Mathematical Specification
//! 
//! This library implements various linear regression methods:
//! 
//! **Ordinary Least Squares (OLS)**:
//! - Minimize: ||Xβ - y||² 
//! - Solution: β = (X'X)⁻¹X'y
//! 
//! **Ridge Regression (L2 regularization)**:
//! - Minimize: ||Xβ - y||² + α||β||²
//! - Solution: β = (X'X + αI)⁻¹X'y
//! 
//! **Lasso Regression (L1 regularization)**:
//! - Minimize: ||Xβ - y||² + α||β||₁
//! - Solved via coordinate descent
//! 
//! **Elastic Net (Combined L1/L2)**:
//! - Minimize: ||Xβ - y||² + α₁||β||₁ + α₂||β||²
//! 
//! ## Features
//! 
//! - **Math-First Design**: Natural mathematical notation using RustLab operators (`^` for matrix multiplication)
//! - **Type Inference**: Automatic parameter selection based on data types - minimal explicit typing required
//! - **Zero-Copy Operations**: Efficient memory usage with references throughout (`&X`, `&y`)
//! - **Parallel Optimization**: Rayon-powered hyperparameter tuning for cross-validation
//! - **Comprehensive Methods**: OLS, Ridge, Lasso, Elastic Net, Logistic regression
//! - **Statistical Inference**: Full suite of diagnostics, hypothesis tests, confidence intervals
//! - **AI-Optimized**: Documentation designed to prevent AI code generation errors
//! 
//! ## For AI Code Generation
//! 
//! - Use `^` operator for matrix operations: `X.transpose() ^ X` (not `*`)
//! - Always use references for data: `model.fit(&X, &y)` prevents ownership issues
//! - Import prelude for all functionality: `use rustlab_linearregression::prelude::*`
//! - Math macros available: `array64!`, `vec64!` for natural data creation
//! - Type inference works throughout: let `fitted = model.fit(&X, &y)?` (no explicit types needed)
//! - Common error: Don't use `X * y` for matrix operations - use `X ^ y`
//! - Pattern: Create data → Fit model → Make predictions → Evaluate metrics
//! 
//! ## Quick Start
//! 
//! ```rust
//! use rustlab_linearregression::prelude::*;
//! use rustlab_math::{array64, vec64};
//! 
//! // Create data using RustLab's ergonomic macros
//! let X = array64![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
//! let y = vec64![2.5, 3.5, 4.5, 5.5];
//! 
//! // Type inference determines the model type
//! let model = LinearRegression::new();
//! let fitted = model.fit(&X, &y)?;
//! 
//! // Mathematical operations using RustLab operators
//! let predictions = fitted.predict(&X);
//! let residuals = &y - &predictions;  // Element-wise subtraction
//! let mse = (&residuals ^ &residuals) / y.len() as f64;  // Dot product for MSE
//! ```
//! 
//! ## Advanced Example with Regularization
//! 
//! ```rust
//! use rustlab_linearregression::prelude::*;
//! use rustlab_math::{array64, vec64, linspace};
//! 
//! // Generate polynomial features
//! let x = linspace(0.0, 1.0, 20);
//! let y_true = &(&x * 2.0) + &x.map(|v| v.powi(2)) * 3.0;
//! let noise = vec64![(0..20).map(|i| (i as f64 * 0.1).sin() * 0.1).collect()];
//! let y = &y_true + &noise;
//! 
//! // Create feature matrix with polynomial terms
//! let X = array64![x.clone(), x.map(|v| v.powi(2))].transpose();
//! 
//! // Ridge regression with cross-validation
//! let alphas = vec![0.001, 0.01, 0.1, 1.0];
//! let best_model = GridSearchCV::new()
//!     .param_grid(alphas)
//!     .cv_folds(5)
//!     .fit(&X, &y)?;
//! 
//! println!("Best alpha: {}", best_model.best_params.alpha);
//! println!("Cross-validation score: {:.3}", best_model.best_score);
//! ```
//! 
//! ## Dimension Rules for AI
//! 
//! | Operation | Input Dimensions | Output | Notes |
//! |-----------|-----------------|---------|-------|
//! | `model.fit(&X, &y)` | X: (n×p), y: (n) | FittedModel | n samples, p features |
//! | `fitted.predict(&X)` | X: (m×p) | (m) | m predictions, same p features |
//! | `X ^ y` | X: (n×p), y: (p) | (n) | Matrix-vector multiplication |
//! | `&y1 - &y2` | y1: (n), y2: (n) | (n) | Element-wise operations need `&` |
//! 
//! ## Common AI Errors to Avoid
//! 
//! ```rust
//! // ❌ Wrong: Using * for matrix multiplication
//! let result = X * y;  // This is element-wise!
//! 
//! // ✅ Correct: Use ^ for matrix operations  
//! let result = X ^ y;  // Matrix-vector multiplication
//! 
//! // ❌ Wrong: No references for element-wise ops
//! let residuals = y - predictions;  // Ownership error
//! 
//! // ✅ Correct: Use references for element-wise
//! let residuals = &y - &predictions;  // Element-wise subtraction
//! 
//! // ❌ Wrong: Explicit types (unnecessary)
//! let fitted: FittedLinearRegression = model.fit(&X, &y)?;
//! 
//! // ✅ Correct: Let type inference work
//! let fitted = model.fit(&X, &y)?;  // Type inferred
//! ```

pub mod error;
pub mod traits;
pub mod ols;
pub mod ols_simple;
pub mod ridge;
pub mod lasso;
pub mod elastic_net;
pub mod logistic;
pub mod cross_validation;
pub mod metrics;
pub mod preprocessing;

pub mod prelude {
    //! Prelude module for convenient imports
    pub use crate::error::{LinearRegressionError, Result};
    pub use crate::traits::{LinearModel, FittedModel};
    pub use crate::ols::{LinearRegression, OrdinaryLeastSquares};
    pub use crate::ols_simple::{SimpleOLS, SimpleFittedOLS};
    pub use crate::ridge::RidgeRegression;
    pub use crate::lasso::LassoRegression;
    pub use crate::elastic_net::ElasticNet;
    pub use crate::logistic::LogisticRegression;
    pub use crate::cross_validation::{cross_validate, GridSearchCV, RandomSearchCV};
    pub use crate::metrics::{r2_score, mean_squared_error, mean_absolute_error};
    pub use crate::preprocessing::{PolynomialFeatures, StandardScaler};
}

pub use prelude::*;