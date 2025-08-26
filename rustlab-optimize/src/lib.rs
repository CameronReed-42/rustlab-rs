//! # RustLab Optimize - Math-First Optimization and Curve Fitting
//!
//! Scientific optimization library with mathematical notation, automatic algorithm selection,
//! and one-line solutions for common optimization and curve fitting problems.
//!
//! ## Design Philosophy
//!
//! - **Math-First**: APIs that match textbook notation and mathematical intuition
//! - **Zero-Friction**: Most problems solved with a single function call
//! - **Smart Defaults**: Automatic algorithm selection based on problem characteristics  
//! - **Type-Driven**: Leverage Rust's type system for ergonomics and correctness
//! - **Performance-Aware**: Fast implementations with clear computational complexity
//!
//! ## For AI Code Generation
//!
//! This library is optimized for AI code generation tools. Key patterns:
//!
//! **Function Minimization:**
//! - `minimize_1d(f)` → scalar result for f(x)
//! - `minimize_2d(f)` → tuple (x,y) for f(x,y)
//! - `minimize(f)` → OptimizationResult for f(x⃗)
//!
//! **Curve Fitting:**
//! - `fit_linear(x, y)` → LinearFit for y = a + bx
//! - `fit_exponential(x, y)` → ExponentialFit for y = A·exp(-kx)
//! - `curve_fit(x, y, model)` → OptimizationResult for custom models
//!
//! **Error Handling:** All functions return `Result<T, Error>` - use `?` for propagation
//!
//! ## Quick Start Examples
//!
//! ```rust
//! use rustlab_optimize::*;
//! use rustlab_math::{vec64, linspace};
//!
//! // 1D minimization: find minimum of (x - 2)²
//! let x_min = minimize_1d(|x| (x - 2.0).powi(2)).solve()?;
//! assert!((x_min - 2.0).abs() < 1e-6);
//!
//! // 2D minimization: minimize Rosenbrock function  
//! let (x, y) = minimize_2d(|x, y| (1.0 - x).powi(2) + 100.0 * (y - x*x).powi(2))
//!     .from(-1.2, 1.0)
//!     .solve()?;
//! 
//! // Curve fitting: exponential decay
//! let t = linspace(0.0, 5.0, 20);
//! let y = vec64![10.0, 8.2, 6.7, 5.5, 4.5, 3.7, 3.0, 2.5, 2.0, 1.6, 
//!                1.3, 1.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2];
//!
//! let fit = fit_exponential(&t, &y)?;
//! println!("y = {:.2} * exp(-{:.2}*t), half-life = {:.2}", 
//!          fit.amplitude, fit.decay_rate, fit.half_life);
//!
//! // N-dimensional with constraints
//! let result = minimize(|x| x.iter().map(|&xi| xi.powi(2)).sum())
//!     .from(&[1.0, 2.0, 3.0])
//!     .bounds(&[-5.0, -5.0, -5.0], &[5.0, 5.0, 5.0])
//!     .solve()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Algorithm Selection Guide
//!
//! The library automatically chooses optimal algorithms based on problem characteristics:
//!
//! | Problem Type | Auto-Selected Algorithm | When to Override |
//! |--------------|-------------------------|------------------|
//! | **Curve Fitting** | Levenberg-Marquardt | Never - it's optimal |
//! | **Small Scale** (n ≤ 10) | BFGS | Try Nelder-Mead for noisy functions |
//! | **Large Scale** (n > 1000) | Gradient Descent | Use BFGS if memory allows |
//! | **Noisy/Non-smooth** | Nelder-Mead | BFGS for constrained problems |
//!
//! **Explicit Algorithm Selection:**
//!
//! ```rust
//! use rustlab_optimize::*;
//! 
//! // Force specific algorithm
//! let result = minimize(objective_function)
//!     .from(&[1.0, 2.0])
//!     .using_levenberg_marquardt()  // Override auto-selection
//!     .tolerance(1e-10)
//!     .solve()?;
//!
//! // Algorithm options: .using_bfgs(), .using_gradient_descent(), 
//! //                   .using_nelder_mead(), .using_levenberg_marquardt()
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Core modules
pub mod minimize;
pub mod fit;
pub mod algorithms;
pub mod models;
pub mod core;
pub mod bounds;

// Re-exports for convenience
pub use minimize::{minimize, minimize_1d, minimize_2d, least_squares, curve_fit};
pub use fit::{fit, fit_linear, fit_exponential, fit_exponential_advanced, fit_polynomial, fit_sinusoidal};
pub use core::{OptimizationResult, Algorithm, Error, Result};

// Common fitting result types
pub use models::{LinearFit, ExponentialFit, PolynomialFit, SinusoidalFit};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{
        minimize, minimize_1d, minimize_2d, least_squares, curve_fit,
        fit, fit_linear, fit_exponential, fit_exponential_advanced, fit_polynomial, fit_sinusoidal,
        OptimizationResult, Algorithm, Error, Result,
    };
    pub use crate::models::{LinearFit, ExponentialFit, PolynomialFit, SinusoidalFit};
    pub use crate::bounds::{Bounds, BoundsTransformer};
    
    // Re-export common rustlab-math types
    pub use rustlab_math::{VectorF64, ArrayF64};
}