//! RustLab Numerical Methods - High-Performance Scientific Computing
//!
//! This crate provides a comprehensive suite of numerical algorithms essential for
//! scientific computing, engineering analysis, and mathematical modeling. All methods
//! are designed for high performance, numerical stability, and seamless integration
//! with the RustLab ecosystem.
//!
//! ## Core Numerical Methods
//!
//! ### 🔄 Interpolation
//! - **1D Methods**: Linear, polynomial (Lagrange/Newton), cubic splines
//! - **2D Methods**: Bilinear, bicubic interpolation for surface data
//! - **Advanced Features**: Derivative evaluation, extrapolation control, vectorized operations
//!
//! ### ∫ Numerical Integration
//! - **Basic Quadrature**: Trapezoidal, Simpson's rule, Simpson's 3/8
//! - **Adaptive Methods**: Romberg integration with Richardson extrapolation
//! - **Performance**: Parallel evaluation, error estimation, convergence analysis
//!
//! ### 📈 Numerical Differentiation
//! - **Finite Differences**: Forward, backward, central differences (multiple orders)
//! - **Advanced Techniques**: Richardson extrapolation, complex-step differentiation
//! - **Applications**: Gradient computation, sensitivity analysis, optimization
//!
//! ### 🎯 Root Finding
//! - **Robust Methods**: Bisection, Brent's method, Illinois method
//! - **Fast Convergence**: Newton-Raphson, secant method, Ridders' method
//! - **Hybrid Algorithms**: Combining reliability with speed
//!
//! ## Design Principles
//!
//! ### Mathematical Rigor
//! - Well-established algorithms with proven convergence properties
//! - Comprehensive error analysis and numerical stability considerations
//! - Multiple precision options and adaptive parameter selection
//!
//! ### High Performance
//! - Zero-cost abstractions with minimal overhead
//! - Optional parallel processing with Rayon
//! - SIMD-friendly data structures and algorithms
//! - Cache-efficient memory access patterns
//!
//! ### Ecosystem Integration
//! - Native support for `VectorF64` and `ArrayF64` from RustLab-Math
//! - Composable algorithms for complex numerical workflows
//! - Consistent error handling and type safety
//!
//! ## Quick Start Examples
//!
//! ```rust
//! use rustlab_numerical::*;
//! use rustlab_math::VectorF64;
//!
//! // Interpolate data points
//! let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
//! let y = VectorF64::from_slice(&[0.0, 1.0, 4.0, 9.0]);
//! let spline = CubicSpline::natural(&x, &y)?;
//! let value = spline.eval(1.5)?;
//!
//! // Integrate a function numerically
//! let integral = simpson(|x| x.sin(), 0.0, std::f64::consts::PI, 1000)?;
//!
//! // Find root of equation
//! let root = brent(|x| x*x - 2.0, 1.0, 2.0, 1e-12, 100)?;
//!
//! // Compute numerical derivative
//! let derivative = central_diff(|x| x.powi(3), 2.0, 1e-8)?;
//! # Ok::<(), rustlab_numerical::NumericalError>(())
//! ```
//!
//! ## Performance Characteristics
//!
//! | Operation | Method | Complexity | Typical Performance |
//! |-----------|--------|------------|--------------------|
//! | Linear Interpolation | Binary search | O(log n) | ~10ns per eval |
//! | Cubic Spline | Tridiagonal solve | O(n) setup, O(log n) eval | ~50ns per eval |
//! | Simpson Integration | Fixed step | O(n) | ~1μs for 1000 steps |
//! | Newton Root Finding | Quadratic convergence | O(log log ε) | ~5 iterations typical |
//!
//! ## Feature Flags
//!
//! - `std` (default): Standard library support
//! - `rayon` (default): Parallel processing capabilities
//!
//! ## Mathematical Foundations
//!
//! All algorithms implement well-established numerical methods:
//!
//! - **Interpolation**: Based on polynomial approximation theory and spline theory
//! - **Integration**: Newton-Cotes formulas and adaptive quadrature
//! - **Differentiation**: Finite difference calculus with error analysis
//! - **Root Finding**: Convergence theory for iterative methods
//!
//! See the [full documentation](https://docs.rs/rustlab-numerical) for detailed
//! mathematical background, algorithm descriptions, and performance benchmarks.

pub mod error;
pub mod interpolation;
pub mod integration;
pub mod differentiation;
pub mod roots;
pub mod utils;

/// Core error types for numerical computations
pub use error::{NumericalError, Result};

/// Re-export interpolation methods and traits
/// 
/// Provides 1D and 2D interpolation capabilities including linear, polynomial,
/// and spline methods with support for derivatives and extrapolation control.
pub use interpolation::*;

/// Re-export numerical integration methods
/// 
/// Includes quadrature rules (trapezoidal, Simpson's), adaptive methods
/// (Romberg), and parallel integration capabilities.
pub use integration::*;

/// Re-export numerical differentiation methods
/// 
/// Provides finite difference schemes with multiple accuracy orders,
/// Richardson extrapolation, and complex-step differentiation.
pub use differentiation::*;

/// Re-export root finding algorithms
/// 
/// Includes bracketing methods (bisection, Brent's) and iterative methods
/// (Newton-Raphson, secant) with hybrid approaches for robustness.
pub use roots::*;