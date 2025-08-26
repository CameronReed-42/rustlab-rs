# RustLab Linear Regression Development Plan

## Overview

Create a high-performance, math-first linear regression library that leverages:
- **faer's SIMD and multi-core capabilities** (automatic parallelism)
- **rustlab-math's ergonomic operators** (`^` for matrix ops, natural syntax)
- **rustlab-linearalgebra's decompositions** (QR, Cholesky, SVD)
- **Zero-copy views** for memory efficiency
- **Arrow-rs integration** for data interoperability

## Design Principles

### 1. Math-First API Design
```rust
// Natural mathematical syntax with zero explicit typing
let model = LinearRegression::new();
let fitted = model.fit(&X, &y)?;
let predictions = fitted.predict(&X_test);
let residuals = &y - &predictions;
```

### 2. Leverage Existing Infrastructure
- Use `rustlab-math`'s `ArrayF64` and `VectorF64` types
- Use `^` operator for matrix multiplication
- Use existing decompositions from `rustlab-linearalgebra`
- Exploit faer's automatic SIMD and rayon parallelism

### 3. Zero-Copy Architecture
- Use `ArrayView` and `VectorView` for slicing
- Leverage faer's `MatRef` and `ColRef` internally
- Minimize allocations in hot paths
- Enable efficient Arrow data integration

### 4. Zero-Explicit-Typing Philosophy
- Rely on Rust's type inference wherever possible
- Design APIs that enable natural type flow
- Use wildcard imports for maximum ergonomics
- Minimize cognitive load for users

## Implementation Strategies for Maximum Ergonomics

### Strategy 1: Design APIs for Type Inference
```rust
// ✅ Good: Return types that can be inferred
impl LinearRegression {
    pub fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<FittedLinearRegression> {
        // Implementation enables: let fitted = model.fit(&X, &y)?;
    }
}

// ✅ Good: Builder pattern with inference-friendly methods
impl LinearRegression {
    pub fn solver(mut self, solver: Solver) -> Self {
        // Enables: LinearRegression::new().solver(Solver::QR).fit(&X, &y)?
    }
}

// ❌ Avoid: Requiring explicit type annotations
// Don't design APIs that force: let fitted: FittedLinearRegression = ...
```

### Strategy 2: Leverage Method Chaining
```rust
// ✅ Fluent APIs that don't require intermediate variables
let result = GridSearchCV::new()
    .param_grid(vec![0.001, 0.01, 0.1, 1.0])
    .cv(5)
    .scoring(ScoringMetric::R2)
    .fit(&X, &y)?;

// ✅ Mathematical operations chain naturally
let score = (&y - &predictions).square().mean();
```

### Strategy 3: Ergonomic Import Patterns
```rust
// ✅ Provide convenient wildcard imports
use rustlab_math::*;           // All math utilities
use rustlab_linearregression::*; // All regression types

// ✅ Re-export commonly used items
pub use rustlab_math::{ArrayF64, VectorF64, array64, vec64, linspace};

// ✅ Group related functionality
pub mod prelude {
    pub use super::{LinearRegression, RidgeRegression, GridSearchCV};
    pub use rustlab_math::{array64, vec64, linspace, PI, E};
}
```

### Strategy 4: Math-First Operator Usage
```rust
// ✅ Use ^ for matrix operations (natural math notation)
let XtX = X.transpose() ^ X;     // X'X
let Xty = X.transpose() ^ y;     // X'y
let predictions = X ^ coef;       // Xβ

// ✅ Natural arithmetic with broadcasting
let residuals = y - predictions;
let mse = residuals.square().mean();
let r2 = 1.0 - residuals.var() / y.var();
```

### Strategy 5: Convenience Macros and Functions
```rust
// ✅ Use ergonomic creation functions
let X = array64![
    [1.0, 2.0],
    [3.0, 4.0]
];
let y = vec64![5.0, 11.0];

// ✅ Mathematical sequences
let x = linspace(0.0, 2.0 * PI, 100);
let features = array64![x.sin(), x.cos(), x].transpose();

// ✅ Common matrices
let I = eye(n_features);
let ones = ones_vec(n_samples);
let zeros = zeros_vec(n_features);
```

### Strategy 6: Minimize Generic Type Parameters
```rust
// ✅ Good: Concrete types with clear semantics
pub struct LinearRegression {
    solver: Solver,
    fit_intercept: bool,
}

// ❌ Avoid: Excessive generics that require annotation
// pub struct LinearRegression<T, E, S> where ...
// Forces: LinearRegression::<f64, StandardError, QRSolver>::new()
```

### Strategy 7: Context-Aware Error Messages
```rust
// ✅ Provide helpful inference hints in errors
impl LinearRegression {
    pub fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<FittedLinearRegression> {
        if X.nrows() != y.len() {
            return Err(RegressionError::DimensionMismatch {
                X_rows: X.nrows(),
                y_len: y.len(),
                hint: "Try: let y = vec64![...] with same length as X rows"
            });
        }
    }
}
```

### Strategy 8: Mathematical Constants Integration
```rust
// ✅ Re-export math constants for natural usage
pub use rustlab_math::{PI, E, TAU, SQRT_2, LN_2};

// ✅ Enable natural mathematical expressions
let x = linspace(-PI, PI, 100);
let y = &x.sin() * E + &x.cos() * PI / 2.0;
let noise = vec64![(0..100).map(|i| (i as f64 * TAU / 100.0).sin() * 0.1).collect()];
```

### Strategy 9: Consistent Return Types
```rust
// ✅ Consistent patterns enable inference
impl FittedLinearRegression {
    pub fn predict(&self, X: &ArrayF64) -> VectorF64 { /* ... */ }
    pub fn score(&self, X: &ArrayF64, y: &VectorF64) -> f64 { /* ... */ }
    pub fn residuals(&self, X: &ArrayF64, y: &VectorF64) -> VectorF64 { /* ... */ }
}

// User can chain without explicit types:
// let score = model.fit(&X, &y)?.score(&X_test, &y_test);
```

### Strategy 10: Documentation with Inference Examples
```rust
/// Fit linear regression model
/// 
/// # Examples
/// ```rust
/// use rustlab_linearregression::*;
/// use rustlab_math::*;
/// 
/// // No explicit types needed - pure inference
/// let X = array64![[1.0, 2.0], [3.0, 4.0]];
/// let y = vec64![5.0, 11.0];
/// let fitted = LinearRegression::new().fit(&X, &y)?;
/// let predictions = fitted.predict(&X);
/// ```
pub fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<FittedLinearRegression> {
```

## Implementation Plan

### Phase 1: Core Structure

#### 1.1 Create Crate Structure
```toml
[package]
name = "rustlab-linearregression"
version = "0.1.0"
edition = "2021"

[dependencies]
rustlab-math = { path = "../rustlab-math" }
rustlab-linearalgebra = { path = "../rustlab-linearalgebra" }
arrow = { version = "54", features = ["ffi"] }
arrow-array = "54"
arrow-buffer = "54"
thiserror = "1.0"
approx = "0.5"  # For testing

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

#### 1.2 Module Structure
```
src/
├── lib.rs              # Main exports and traits
├── linear_regression.rs # Core LinearRegression implementation
├── ridge.rs            # Ridge regression with regularization
├── solvers.rs          # QR, Cholesky, SVD solver implementations
├── diagnostics.rs      # R², MSE, residual analysis
├── arrow_integration.rs # Arrow-rs data loading
├── error.rs            # Error types
└── utils.rs            # Helper functions
```

### Phase 2: Core LinearRegression Implementation

#### 2.1 Main Structure
```rust
use rustlab_math::{ArrayF64, VectorF64, ArrayView, VectorView};

/// Linear regression model with sklearn-like API
pub struct LinearRegression {
    /// Solver to use for fitting
    solver: Solver,
    /// Whether to fit intercept
    fit_intercept: bool,
    /// Copy X in fit (for safety)
    copy_x: bool,
}

/// Fitted linear regression model
pub struct FittedLinearRegression {
    /// Coefficients (weights)
    pub coef_: VectorF64,
    /// Intercept (if fit_intercept=true)
    pub intercept_: Option<f64>,
    /// Training R² score
    pub score_: f64,
    /// Number of features
    n_features_: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Solver {
    /// QR decomposition (default, stable)
    QR,
    /// Cholesky decomposition (fast for X'X)
    Cholesky,
    /// SVD (handles rank-deficient)
    SVD,
}
```

#### 2.2 Math-First Fit Implementation
```rust
use rustlab_math::*;  // Import everything for maximum ergonomics

impl LinearRegression {
    pub fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<FittedLinearRegression> {
        // Validate dimensions
        if X.nrows() != y.len() {
            return Err(RegressionError::DimensionMismatch);
        }
        
        // Add intercept column if needed (math-first approach)
        let X_fit = if self.fit_intercept {
            let ones = ones_vec(X.nrows());  // Type inferred
            hstack![ones.as_column(), X]  // No explicit typing needed
        } else {
            X.view()
        };
        
        // Solve using selected method - type inference handles everything
        let coef = match self.solver {
            Solver::QR => self.solve_qr(&X_fit, y)?,
            Solver::Cholesky => self.solve_cholesky(&X_fit, y)?,
            Solver::SVD => self.solve_svd(&X_fit, y)?,
        };
        
        // Extract intercept and coefficients - no explicit types
        let (intercept, coef_) = if self.fit_intercept {
            (Some(coef[0]), coef.slice(1..))
        } else {
            (None, coef)
        };
        
        // Calculate R² using math-first operators - pure inference
        let y_pred = self.predict_raw(&X_fit, &coef_);
        let residuals = y - &y_pred;
        let ss_res = residuals ^ residuals;  // Dot product with ^ operator
        let y_mean = y.mean();
        let y_centered = y - y_mean;
        let ss_tot = y_centered ^ y_centered;
        let score_ = 1.0 - ss_res / ss_tot;
        
        Ok(FittedLinearRegression {
            coef_,
            intercept_,
            score_,
            n_features_: X.ncols(),
        })
    }
}
```

### Phase 3: Solver Implementations

#### 3.1 QR Solver (using rustlab-linearalgebra)
```rust
fn solve_qr(&self, X: &ArrayView<f64>, y: &VectorF64) -> Result<VectorF64> {
    use rustlab_linearalgebra::DecompositionMethods;
    
    // QR decomposition with faer's SIMD optimization
    let qr = X.to_owned().qr()?;
    
    // Solve Q'y = R*beta
    let y_mat = y.as_column();
    let solution = qr.solve(&y_mat)?;
    
    Ok(solution.as_vector())
}
```

#### 3.2 Cholesky Solver (Normal Equations)
```rust
fn solve_cholesky(&self, X: &ArrayView<f64>, y: &VectorF64) -> Result<VectorF64> {
    // Form normal equations: X'X beta = X'y
    // Using ^ operator for matrix multiplication
    let XtX = X.transpose() ^ X;  // Automatic SIMD
    let Xty = X.transpose() ^ y;   // Automatic SIMD
    
    // Cholesky decomposition
    let chol = XtX.cholesky()?;
    let solution = chol.solve(&Xty.as_column())?;
    
    Ok(solution.as_vector())
}
```

#### 3.3 SVD Solver (Rank-Deficient Cases)
```rust
fn solve_svd(&self, X: &ArrayView<f64>, y: &VectorF64) -> Result<VectorF64> {
    let svd = X.to_owned().svd()?;
    
    // Pseudoinverse solution: beta = V * S^-1 * U' * y
    let solution = svd.solve(&y.as_column())?;
    
    Ok(solution.as_vector())
}
```

### Phase 4: Ridge Regression & Hyperparameter Optimization

#### 4.1 Ridge Regression Implementation
```rust
use rustlab_math::*;  // Maximum ergonomics

pub struct RidgeRegression {
    alpha: f64,  // No need for verbose docs in example
    base: LinearRegression,
}

impl RidgeRegression {
    pub fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<FittedRidgeRegression> {
        let X_fit = if self.base.fit_intercept {
            let ones = ones_vec(X.nrows());  // Inferred type
            hstack![ones.as_column(), X]
        } else {
            X.view()
        };
        
        // Form normal equations - all types inferred
        let XtX = X_fit.transpose() ^ &X_fit;
        let reg_matrix = eye(XtX.nrows()) * self.alpha;
        
        // Handle intercept regularization elegantly
        let XtX_reg = if self.base.fit_intercept {
            let mut reg = reg_matrix;
            reg[(0, 0)] = 0.0;  // Skip intercept
            XtX + reg
        } else {
            XtX + reg_matrix
        };
        
        let Xty = X_fit.transpose() ^ y;
        
        // Solve - inference handles everything
        let chol = XtX_reg.cholesky()?;
        let solution = chol.solve(&Xty.as_column())?;
        
        // Return fitted model...
    }
}
```

#### 4.2 Parallelized Hyperparameter Search
```rust
use rayon::prelude::*;
use std::sync::Arc;

/// Grid search for optimal hyperparameters with automatic parallelization
pub struct GridSearchCV {
    /// Estimator to optimize
    estimator: RidgeRegression,
    /// Parameter grid to search
    param_grid: ParamGrid,
    /// Number of CV folds
    cv: usize,
    /// Scoring metric
    scoring: ScoringMetric,
    /// Number of parallel jobs (-1 for all cores)
    n_jobs: i32,
}

#[derive(Clone)]
pub struct ParamGrid {
    alphas: Vec<f64>,
    solvers: Vec<Solver>,
}

#[derive(Clone, Copy)]
pub enum ScoringMetric {
    R2,
    MSE,
    MAE,
}

impl GridSearchCV {
    pub fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<GridSearchResult> {
        // Pre-compute reusable matrices for efficiency
        let X_arc = Arc::new(X.clone());
        let y_arc = Arc::new(y.clone());
        
        // Generate all parameter combinations
        let param_combinations: Vec<_> = self.param_grid.alphas.iter()
            .flat_map(|&alpha| {
                self.param_grid.solvers.iter()
                    .map(move |&solver| (alpha, solver))
            })
            .collect();
        
        // Determine parallelism level
        let n_jobs = if self.n_jobs == -1 {
            rayon::current_num_threads()
        } else {
            self.n_jobs as usize
        };
        
        // Create thread pool for controlled parallelism
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_jobs)
            .build()?;
        
        // Parallel hyperparameter search with cross-validation
        let results = pool.install(|| {
            param_combinations.par_iter()
                .map(|&(alpha, solver)| {
                    let cv_scores = self.cross_validate(
                        &X_arc, 
                        &y_arc, 
                        alpha, 
                        solver
                    )?;
                    
                    let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
                    
                    Ok(HyperparamResult {
                        alpha,
                        solver,
                        cv_scores,
                        mean_score,
                        std_score: calculate_std(&cv_scores),
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;
        
        // Find best parameters
        let best_result = results.iter()
            .max_by(|a, b| a.mean_score.partial_cmp(&b.mean_score).unwrap())
            .unwrap();
        
        // Refit on entire dataset with best parameters
        let best_estimator = RidgeRegression {
            alpha: best_result.alpha,
            base: LinearRegression::new().solver(best_result.solver),
        };
        
        let best_model = best_estimator.fit(X, y)?;
        
        Ok(GridSearchResult {
            best_params: (best_result.alpha, best_result.solver),
            best_score: best_result.mean_score,
            best_model,
            cv_results: results,
        })
    }
    
    fn cross_validate(
        &self, 
        X: &Arc<ArrayF64>, 
        y: &Arc<VectorF64>,
        alpha: f64,
        solver: Solver,
    ) -> Result<Vec<f64>> {
        let n_samples = X.nrows();
        let fold_size = n_samples / self.cv;
        
        // Parallel cross-validation folds
        (0..self.cv).into_par_iter()
            .map(|fold| {
                // Create train/test split
                let test_start = fold * fold_size;
                let test_end = if fold == self.cv - 1 { 
                    n_samples 
                } else { 
                    (fold + 1) * fold_size 
                };
                
                // Use views for zero-copy splitting
                let (X_train, X_test, y_train, y_test) = 
                    self.split_data(X, y, test_start, test_end)?;
                
                // Train model
                let model = RidgeRegression {
                    alpha,
                    base: LinearRegression::new().solver(solver),
                };
                let fitted = model.fit(&X_train, &y_train)?;
                
                // Score on test set
                let score = match self.scoring {
                    ScoringMetric::R2 => fitted.r2_score(&X_test, &y_test),
                    ScoringMetric::MSE => -fitted.mse(&X_test, &y_test),
                    ScoringMetric::MAE => -fitted.mae(&X_test, &y_test),
                };
                
                Ok(score)
            })
            .collect()
    }
}

/// Randomized search for continuous hyperparameters
pub struct RandomizedSearchCV {
    estimator: RidgeRegression,
    param_distributions: ParamDistributions,
    n_iter: usize,
    cv: usize,
    n_jobs: i32,
}

pub struct ParamDistributions {
    alpha_range: (f64, f64),  // Will sample log-uniformly
    solvers: Vec<Solver>,
}

impl RandomizedSearchCV {
    pub fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<GridSearchResult> {
        use rand::prelude::*;
        use rand_distr::{LogNormal, Distribution};
        
        // Generate random parameter combinations
        let mut rng = thread_rng();
        let log_normal = LogNormal::new(
            (self.param_distributions.alpha_range.0.ln() + 
             self.param_distributions.alpha_range.1.ln()) / 2.0,
            1.0
        )?;
        
        let param_combinations: Vec<_> = (0..self.n_iter)
            .map(|_| {
                let alpha = log_normal.sample(&mut rng)
                    .max(self.param_distributions.alpha_range.0)
                    .min(self.param_distributions.alpha_range.1);
                let solver = self.param_distributions.solvers
                    .choose(&mut rng)
                    .copied()
                    .unwrap();
                (alpha, solver)
            })
            .collect();
        
        // Rest is similar to GridSearchCV but with random params...
    }
}

/// Efficient batch solving for multiple alpha values
pub struct RidgeCV {
    alphas: Vec<f64>,
    cv: usize,
    /// Whether to use efficient Leave-One-Out CV
    efficient_loo: bool,
}

impl RidgeCV {
    /// Fit using efficient batch solving
    pub fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<FittedRidgeCV> {
        // Pre-compute matrices that are reused across alphas
        let X_fit = if self.fit_intercept {
            add_intercept_column(X)
        } else {
            X.clone()
        };
        
        let XtX = &X_fit.transpose() ^ &X_fit;  // Computed once!
        let Xty = &X_fit.transpose() ^ y;       // Computed once!
        
        // For efficient LOO-CV, compute eigendecomposition once
        let eigen = if self.efficient_loo {
            Some(XtX.eigenvalues()?)
        } else {
            None
        };
        
        // Parallel evaluation of all alphas
        let cv_scores: Vec<_> = self.alphas.par_iter()
            .map(|&alpha| {
                if self.efficient_loo {
                    // Use generalized cross-validation (GCV) approximation
                    self.compute_gcv_score(&XtX, &Xty, &eigen.as_ref().unwrap(), alpha)
                } else {
                    // Standard k-fold CV
                    self.compute_cv_score(&X_fit, y, alpha)
                }
            })
            .collect::<Result<_>>()?;
        
        // Find best alpha
        let (best_idx, best_score) = cv_scores.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let best_alpha = self.alphas[best_idx];
        
        // Refit with best alpha
        let final_model = self.solve_ridge(&XtX, &Xty, best_alpha)?;
        
        Ok(FittedRidgeCV {
            alpha_: best_alpha,
            coef_: final_model.coef_,
            intercept_: final_model.intercept_,
            cv_scores_: cv_scores,
            alphas_: self.alphas.clone(),
        })
    }
    
    /// Efficient GCV score computation using eigenvalues
    fn compute_gcv_score(
        &self, 
        XtX: &ArrayF64, 
        Xty: &VectorF64,
        eigenvalues: &VectorF64,
        alpha: f64
    ) -> Result<f64> {
        // GCV = RSS / (n * (1 - tr(H)/n)^2)
        // where H is the hat matrix
        // For ridge: tr(H) = sum(λᵢ/(λᵢ + α))
        
        let n = XtX.nrows() as f64;
        
        // Compute trace of hat matrix efficiently using eigenvalues
        let trace_H = eigenvalues.iter()
            .map(|&lambda| lambda / (lambda + alpha))
            .sum::<f64>();
        
        // Solve for this alpha
        let beta = self.solve_ridge(XtX, Xty, alpha)?;
        
        // Compute residual sum of squares
        let rss = compute_rss(&self.X, &self.y, &beta);
        
        // GCV score
        let gcv = rss / (n * (1.0 - trace_H / n).powi(2));
        
        Ok(-gcv)  // Negative because we maximize
    }
}
```

### Phase 5: Arrow-rs Integration

#### 5.1 Add to rustlab-math
```rust
// In rustlab-math/src/arrow_integration.rs
use arrow_array::{Float64Array, ArrayRef};
use arrow_buffer::Buffer;

impl ArrayF64 {
    /// Create from Arrow Float64Array with zero-copy when possible
    pub fn from_arrow(arrow_array: &Float64Array, rows: usize, cols: usize) -> Result<Self> {
        let values = arrow_array.values();
        
        // Check if we can use zero-copy
        if arrow_array.null_count() == 0 {
            // No nulls, can potentially zero-copy
            let slice = values.as_slice();
            Self::from_slice(slice, rows, cols)
        } else {
            // Has nulls, need to handle
            Err(MathError::ArrowHasNulls)
        }
    }
    
    /// Convert to Arrow array
    pub fn to_arrow(&self) -> Float64Array {
        let data = self.to_vec();
        Float64Array::from(data)
    }
}

impl VectorF64 {
    /// Create from Arrow Float64Array with zero-copy when possible
    pub fn from_arrow(arrow_array: &Float64Array) -> Result<Self> {
        let values = arrow_array.values();
        
        if arrow_array.null_count() == 0 {
            let slice = values.as_slice();
            Ok(Self::from_slice(slice))
        } else {
            Err(MathError::ArrowHasNulls)
        }
    }
}
```

#### 5.2 DataFrame-like API
```rust
use arrow::record_batch::RecordBatch;

/// Fit regression from Arrow RecordBatch
pub fn fit_from_arrow(
    &self,
    batch: &RecordBatch,
    feature_columns: &[&str],
    target_column: &str,
) -> Result<FittedLinearRegression> {
    // Extract feature matrix
    let mut features = Vec::new();
    for col_name in feature_columns {
        let array = batch.column_by_name(col_name)
            .ok_or(RegressionError::ColumnNotFound)?;
        let float_array = array.as_any()
            .downcast_ref::<Float64Array>()
            .ok_or(RegressionError::InvalidType)?;
        features.push(VectorF64::from_arrow(float_array)?);
    }
    
    // Stack features into matrix
    let X = ArrayF64::from_columns(&features)?;
    
    // Extract target
    let target_array = batch.column_by_name(target_column)
        .ok_or(RegressionError::ColumnNotFound)?;
    let y = VectorF64::from_arrow(target_array.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or(RegressionError::InvalidType)?)?;
    
    self.fit(&X, &y)
}
```

### Phase 6: Diagnostics

```rust
pub trait RegressionDiagnostics {
    /// Calculate R² score
    fn r2_score(&self, X: &ArrayF64, y: &VectorF64) -> f64;
    
    /// Calculate Mean Squared Error
    fn mse(&self, X: &ArrayF64, y: &VectorF64) -> f64;
    
    /// Get residuals
    fn residuals(&self, X: &ArrayF64, y: &VectorF64) -> VectorF64;
    
    /// Get standardized residuals
    fn standardized_residuals(&self, X: &ArrayF64, y: &VectorF64) -> VectorF64;
}

impl RegressionDiagnostics for FittedLinearRegression {
    fn r2_score(&self, X: &ArrayF64, y: &VectorF64) -> f64 {
        let y_pred = self.predict(X);
        let residuals = y - &y_pred;
        let ss_res = residuals.dot(&residuals);
        
        let y_mean = y.mean();
        let y_centered = y - y_mean;
        let ss_tot = y_centered.dot(&y_centered);
        
        1.0 - ss_res / ss_tot
    }
    
    fn mse(&self, X: &ArrayF64, y: &VectorF64) -> f64 {
        let y_pred = self.predict(X);
        let residuals = y - &y_pred;
        residuals.dot(&residuals) / (y.len() as f64)
    }
}
```

### Phase 7: Testing & Benchmarking

#### 7.1 Test Suite
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rustlab_math::*;  // Clean imports
    
    #[test]
    fn test_simple_regression() {
        // Pure math notation - y = 2x + 1
        let X = array64![[1.0], [2.0], [3.0], [4.0]];
        let y = vec64![3.0, 5.0, 7.0, 9.0];
        
        let model = LinearRegression::new();
        let fitted = model.fit(&X, &y).unwrap();
        
        assert_relative_eq!(fitted.coef_[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept_.unwrap(), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_multivariate_regression() {
        // Synthetic data with mathematical relationships
        let n_samples = 50;
        let x1 = linspace(-PI, PI, n_samples);
        let x2 = arange(0.0, n_samples as f64, 1.0);
        
        // True relationship: y = 2sin(x₁) + 0.5x₂ + e
        let y_true = &x1.sin() * 2.0 + &x2 * 0.5 + E;
        let noise = vec64![(0..n_samples).map(|i| (i as f64).sin() * 0.01).collect()];
        let y = &y_true + &noise;
        
        // Stack features - type inference
        let X = ArrayF64::from_columns(&[x1, x2]).unwrap();
        
        let fitted = LinearRegression::new()
            .solver(Solver::QR)
            .fit(&X, &y)
            .unwrap();
        
        assert!(fitted.score_ > 0.95);
    }
    
    #[test]
    fn test_ridge_regularization() {
        // Ill-conditioned problem
        let X = array64![
            [1.0, 1.0001],  
            [2.0, 2.0002],
            [3.0, 3.0003],
            [4.0, 4.0004]
        ];
        let y = vec64![1.0, 2.0, 3.0, 4.0];
        
        // Ridge handles collinearity
        let fitted = RidgeRegression::new(alpha: 1.0)
            .fit(&X, &y)
            .unwrap();
        
        assert!(fitted.coef_.norm() < 10.0);
    }
    
    #[test]
    fn test_sklearn_parity() {
        // Compare with sklearn - y = x₁ + 2x₂ + 3x₃ + 1
        let X = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let y = vec64![10.0, 26.0, 42.0, 58.0];
        
        let fitted = LinearRegression::new().fit(&X, &y).unwrap();
        
        let expected_coef = vec64![1.0, 2.0, 3.0];
        for (actual, expected) in fitted.coef_.iter().zip(expected_coef.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-10);
        }
        assert_relative_eq!(fitted.intercept_.unwrap(), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_cross_validation() {
        // Complex polynomial + trigonometric features
        let x = linspace(-2.0 * PI, 2.0 * PI, 200);
        let X = ArrayF64::from_columns(&[
            x.clone(),
            x.square(),
            x.sin(),
        ]).unwrap();
        
        let y = &(&x * 2.0 + &x.square() * 0.5) + &x.sin() * 1.5;
        
        // Grid search - inference handles types
        let result = GridSearchCV::new()
            .param_grid(vec![0.001, 0.01, 0.1, 1.0, 10.0])
            .cv(5)
            .fit(&X, &y)
            .unwrap();
        
        assert!(result.best_alpha > 0.0001);
        assert!(result.best_alpha < 100.0);
        assert!(result.best_score > 0.8);
    }
}
```

#### 7.2 Benchmarks
```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("solvers");
    
    // Test different sizes
    for size in [100, 1000, 10000].iter() {
        let X = ArrayF64::random(*size, 20);
        let y = VectorF64::random(*size);
        
        group.bench_function(&format!("QR_{}", size), |b| {
            b.iter(|| {
                LinearRegression::new()
                    .solver(Solver::QR)
                    .fit(&X, &y)
            })
        });
        
        group.bench_function(&format!("Cholesky_{}", size), |b| {
            b.iter(|| {
                LinearRegression::new()
                    .solver(Solver::Cholesky)
                    .fit(&X, &y)
            })
        });
    }
}
```

### Phase 8: Examples

#### 8.1 Basic Usage
```rust
use rustlab_linearregression::*;  // Import everything
use rustlab_math::*;  // Maximum ergonomics

fn main() -> Result<()> {
    // Simple example - no explicit types needed
    let X = array64![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ];
    let y = vec64![3.0, 5.0, 7.0, 9.0];
    
    // Fit model - inference handles types
    let model = LinearRegression::new().solver(Solver::QR);
    let fitted = model.fit(&X, &y)?;
    
    // Make predictions - clean and simple
    let X_test = array64![[5.0, 6.0]];
    let predictions = fitted.predict(&X_test);
    
    println!("Coefficients: {:?}", fitted.coef_);
    println!("Intercept: {:?}", fitted.intercept_);
    println!("R² score: {}", fitted.score_);
    
    // Generate synthetic data - pure math expressions
    let n_samples = 100;
    let x1 = linspace(0.0, 2.0 * PI, n_samples);
    let noise = vec64![(0..n_samples).map(|i| (i as f64 * 0.1).sin() * 0.1).collect()];
    let y_synthetic = &x1.sin() * 2.0 + &x1.cos() * E + noise;
    
    // Advanced example - still no explicit types
    let features = array64![x1.sin(), x1.cos(), x1].transpose();
    let advanced_model = LinearRegression::new();
    let advanced_fitted = advanced_model.fit(&features, &y_synthetic)?;
    
    Ok(())
}
```

#### 8.2 Arrow Integration
```rust
use arrow::record_batch::RecordBatch;
use rustlab_linearregression::LinearRegression;

fn main() -> Result<()> {
    // Load data from Parquet
    let batch = load_parquet("data.parquet")?;
    
    // Fit directly from Arrow data
    let model = LinearRegression::new();
    let fitted = model.fit_from_arrow(
        &batch,
        &["feature1", "feature2", "feature3"],
        "target"
    )?;
    
    // Predictions also return Arrow arrays
    let predictions = fitted.predict_arrow(&test_batch)?;
    
    Ok(())
}
```

## Performance Optimizations

### 1. SIMD Exploitation
- All matrix operations use faer's SIMD implementations
- The `^` operator automatically uses SIMD for matrix multiplication
- Element-wise operations leverage SIMD through faer

### 2. Multi-core Parallelism
- Large matrix operations automatically use rayon
- QR and SVD decompositions are parallelized by faer
- No manual parallel code needed

### 3. Zero-Copy Operations
- Use views wherever possible
- Arrow integration allows zero-copy data loading
- Minimize allocations in prediction phase

### 4. Cache-Friendly Access
- Data structures are cache-aligned (`#[repr(C, align(64))]`)
- Column-major storage matches faer's layout
- Efficient memory access patterns

## API Design Goals

1. **sklearn Compatibility**: Familiar API for Python users
2. **Math-First Syntax**: Natural mathematical operations
3. **Type Safety**: Leverage Rust's type system
4. **Performance**: 3-25x faster than alternatives
5. **Extensibility**: Easy to add new regression methods

## Success Metrics

1. **Performance**: Beat scikit-learn by 3x+ on large datasets
2. **Accuracy**: Match sklearn results to machine precision
3. **Usability**: Clean, intuitive API with good docs
4. **Integration**: Seamless Arrow data pipeline
5. **Testing**: 95%+ code coverage

## Timeline

- **Week 1**: Core structure and basic LinearRegression
- **Week 2**: All three solvers implemented and tested
- **Week 3**: Ridge regression and diagnostics
- **Week 4**: Arrow integration in rustlab-math
- **Week 5**: DataFrame API and examples
- **Week 6**: Benchmarking and optimization
- **Week 7**: Documentation and polish
- **Week 8**: Release preparation

## Detailed Implementation Checklist

### Phase 1: Core Setup ⬜
- [ ] Create rustlab-linearregression crate structure
  - [ ] Initialize Cargo.toml with dependencies
  - [ ] Set up module structure (lib.rs, error.rs, etc.)
  - [ ] Configure for workspace integration
- [ ] Define core error types
  - [ ] LinearRegressionError enum with helpful hints
  - [ ] Implement std::error::Error
  - [ ] Add thiserror derives
  - [ ] Include type inference hints in error messages
- [ ] Set up CI/testing infrastructure
  - [ ] Basic test harness
  - [ ] GitHub Actions workflow
  - [ ] Code coverage setup
- [ ] Implement ergonomic design strategies
  - [ ] Create prelude module with common imports
  - [ ] Re-export rustlab-math types and macros
  - [ ] Design builder patterns for method chaining
  - [ ] Ensure all APIs support type inference

### Phase 2: Core LinearRegression ⬜
- [ ] Implement LinearRegression struct
  - [ ] Basic struct with solver enum
  - [ ] Builder pattern for configuration
  - [ ] Default implementations
- [ ] Implement fit method
  - [ ] Input validation
  - [ ] Intercept handling (add ones column)
  - [ ] Solver dispatch logic
- [ ] Implement predict method
  - [ ] Basic prediction (X @ coef + intercept)
  - [ ] Batch prediction support
  - [ ] Input shape validation
- [ ] Add math-first operators
  - [ ] Use ^ for matrix multiplication
  - [ ] Natural arithmetic operators
  - [ ] Zero-copy operations where possible
  - [ ] Use convenience macros (vec64!, array64!, linspace, arange, etc.)
  - [ ] Leverage mathematical constants (PI, E, TAU, etc.)
  - [ ] Minimize explicit type annotations (use `rustlab_math::*`)
  - [ ] Rely on Rust's type inference for ergonomics

### Phase 3: Solver Implementations ⬜
- [ ] QR Solver
  - [ ] Integration with rustlab-linearalgebra QR
  - [ ] Solve normal equations via QR
  - [ ] Handle rank-deficient cases
  - [ ] Add comprehensive tests
- [ ] Cholesky Solver
  - [ ] Form normal equations (X'X, X'y)
  - [ ] Use Cholesky decomposition
  - [ ] Add positive-definite checks
  - [ ] Performance optimizations
- [ ] SVD Solver
  - [ ] Integration with rustlab-linearalgebra SVD
  - [ ] Pseudoinverse computation
  - [ ] Rank detection
  - [ ] Numerical stability tests

### Phase 4: Ridge Regression ⬜
- [ ] Basic Ridge implementation
  - [ ] RidgeRegression struct
  - [ ] Regularization parameter handling
  - [ ] Modified normal equations
- [ ] Efficient batch solvers
  - [ ] RidgeCV with multiple alphas
  - [ ] Pre-compute reusable matrices
  - [ ] Parallel alpha evaluation
- [ ] Hyperparameter optimization
  - [ ] GridSearchCV implementation
  - [ ] RandomizedSearchCV
  - [ ] Cross-validation infrastructure
  - [ ] Parallel CV fold evaluation
- [ ] Advanced optimizations
  - [ ] GCV (Generalized CV) implementation
  - [ ] Eigendecomposition caching
  - [ ] Warm-start capabilities

### Phase 5: Model Diagnostics ⬜
- [ ] Basic metrics
  - [ ] R² score calculation
  - [ ] MSE/RMSE computation
  - [ ] MAE computation
- [ ] Residual analysis
  - [ ] Raw residuals
  - [ ] Standardized residuals
  - [ ] Residual plots data
- [ ] Statistical tests
  - [ ] Coefficient significance
  - [ ] F-statistic
  - [ ] Confidence intervals
- [ ] Model selection criteria
  - [ ] AIC/BIC
  - [ ] Adjusted R²
  - [ ] Cross-validation scores

### Phase 6: Arrow Integration ⬜
- [ ] Core Arrow support in rustlab-math
  - [ ] Add arrow dependencies
  - [ ] ArrayF64::from_arrow implementation
  - [ ] VectorF64::from_arrow implementation
  - [ ] Zero-copy conversions where possible
- [ ] Arrow to faer conversions
  - [ ] Handle null values appropriately
  - [ ] Support different Arrow types
  - [ ] Efficient batch conversions
- [ ] DataFrame-like API
  - [ ] fit_from_arrow method
  - [ ] predict_to_arrow method
  - [ ] RecordBatch integration
- [ ] File I/O support
  - [ ] Parquet file loading
  - [ ] CSV via Arrow
  - [ ] Streaming support for large files

### Phase 7: Performance Optimization ⬜
- [ ] Benchmark suite
  - [ ] Comparison with scikit-learn
  - [ ] Comparison with Linfa
  - [ ] Large-scale dataset tests
  - [ ] SIMD verification benchmarks
- [ ] Memory optimizations
  - [ ] Minimize allocations
  - [ ] Use views everywhere possible
  - [ ] Lazy evaluation where beneficial
- [ ] Parallelism tuning
  - [ ] Optimal thread pool sizes
  - [ ] Work-stealing configuration
  - [ ] NUMA awareness (if applicable)
- [ ] Cache optimizations
  - [ ] Data layout optimization
  - [ ] Prefetching strategies
  - [ ] Loop tiling where beneficial

### Phase 8: Testing & Validation ⬜
- [ ] Unit tests
  - [ ] Each solver method
  - [ ] Edge cases (empty data, single point)
  - [ ] Numerical accuracy tests
- [ ] Integration tests
  - [ ] Full pipeline tests
  - [ ] Cross-validation accuracy
  - [ ] Hyperparameter search convergence
- [ ] Comparison tests
  - [ ] Results match scikit-learn (within tolerance)
  - [ ] Performance meets targets
  - [ ] Memory usage validation
- [ ] Property-based tests
  - [ ] Invariant checking
  - [ ] Numerical stability
  - [ ] Solver equivalence

### Phase 9: Documentation ⬜
- [ ] API documentation
  - [ ] All public types documented
  - [ ] Examples showing type inference (no explicit types)
  - [ ] Math formulas in docs using unicode notation
  - [ ] Include wildcard import examples (use rustlab_math::*)
- [ ] User guide
  - [ ] Getting started tutorial emphasizing ergonomics
  - [ ] Common use cases with minimal typing
  - [ ] Performance tips leveraging faer optimizations
  - [ ] Type inference best practices
- [ ] Examples
  - [ ] Basic linear regression (zero explicit types)
  - [ ] Ridge with CV (fluent API)
  - [ ] Arrow data pipeline (seamless integration)
  - [ ] Large-scale example (parallel hyperparameter search)
- [ ] Architecture docs
  - [ ] Design decisions (math-first philosophy)
  - [ ] Performance characteristics (SIMD + multicore)
  - [ ] Extension guide (maintaining ergonomics)
  - [ ] Ergonomic API design patterns

### Phase 10: Polish & Release ⬜
- [ ] API review
  - [ ] Consistent naming
  - [ ] Ergonomic method chains
  - [ ] Type inference optimization
- [ ] Performance validation
  - [ ] Meet 3x sklearn target
  - [ ] Memory usage acceptable
  - [ ] Compilation time reasonable
- [ ] Release preparation
  - [ ] Version numbering
  - [ ] CHANGELOG.md
  - [ ] Migration guide (if needed)
- [ ] Community engagement
  - [ ] Reddit/HN announcement
  - [ ] Blog post with benchmarks
  - [ ] Comparison with alternatives

### Bonus Features (Post-1.0) ⬜
- [ ] Additional solvers
  - [ ] Coordinate descent for Lasso
  - [ ] ADMM for distributed solving
  - [ ] Stochastic gradient descent
- [ ] Advanced regularization
  - [ ] Elastic Net
  - [ ] Group Lasso
  - [ ] Adaptive Lasso
- [ ] Robust regression
  - [ ] Huber regression
  - [ ] RANSAC
  - [ ] Theil-Sen estimator
- [ ] Time series support
  - [ ] Autoregressive models
  - [ ] Online updating
  - [ ] Rolling window regression

### Integration Milestones ⬜
- [ ] rustlab-math enhancements complete
- [ ] rustlab-linearalgebra integration tested
- [ ] Arrow ecosystem connected
- [ ] Plotting integration (rustlab-plotting)
- [ ] Full rustlab ecosystem demo

This plan creates a high-performance linear regression library that fully exploits faer's capabilities while maintaining the math-first philosophy of the rustlab ecosystem.