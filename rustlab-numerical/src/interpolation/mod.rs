//! Advanced interpolation methods for 1D and 2D data analysis
//!
//! This module provides a comprehensive suite of interpolation algorithms designed for
//! scientific computing, data analysis, and numerical modeling. All methods implement
//! the unified trait system for consistent APIs while maintaining optimal performance
//! and numerical accuracy.
//!
//! ## Core Interpolation Methods
//!
//! ### 1D Interpolation
//!
//! #### Linear Interpolation
//! - **Use Case**: Fast, simple interpolation for piecewise linear data
//! - **Complexity**: O(log n) evaluation after O(n log n) sorting
//! - **Properties**: C⁰ continuous, exact for linear functions
//! - **Best For**: Quick approximations, monotonic data, real-time applications
//!
//! #### Polynomial Interpolation
//! - **Lagrange Form**: Direct polynomial construction, O(n²) evaluation
//! - **Newton Form**: Divided differences, O(n) evaluation after setup
//! - **Properties**: Exact interpolation, can exhibit Runge phenomenon
//! - **Best For**: Smooth functions, small datasets (n < 10), analytical work
//!
//! #### Cubic Spline Interpolation
//! - **Natural Splines**: Zero second derivatives at boundaries
//! - **Clamped Splines**: Specified first derivatives at boundaries
//! - **Properties**: C² continuous, minimal curvature, no oscillations
//! - **Best For**: Smooth data, derivative information needed, large datasets
//!
//! ### 2D Interpolation
//!
//! #### Bilinear Interpolation
//! - **Use Case**: Fast rectangular grid interpolation
//! - **Properties**: Linear in each dimension, C⁰ continuous
//! - **Applications**: Image processing, regular grids, real-time graphics
//!
//! #### Bicubic Interpolation
//! - **Use Case**: High-quality surface interpolation
//! - **Properties**: Cubic polynomials, C¹ continuous, smooth derivatives
//! - **Applications**: Image scaling, surface fitting, smooth animations
//!
//! ## Examples
//!
//! ### Quick Start
//! ```rust
//! use rustlab_numerical::interpolation::*;
//! use rustlab_math::VectorF64;
//!
//! // Sample data: y = x²
//! let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0]);
//! let y = VectorF64::from_slice(&[0.0, 1.0, 4.0, 9.0, 16.0]);
//!
//! // Linear interpolation (fast)
//! let linear = LinearInterpolator::new(&x, &y)?;
//! let val1 = linear.eval(2.5)?; // ≈ 6.5
//!
//! // Cubic spline (smooth)
//! let spline = CubicSpline::natural(&x, &y)?;
//! let val2 = spline.eval(2.5)?; // ≈ 6.25 (closer to true x²)
//!
//! // With derivatives
//! let deriv = spline.eval_derivative(2.5)?; // ≈ 5.0 (should be 2*2.5)
//! # Ok::<(), rustlab_numerical::NumericalError>(())
//! ```

mod linear;
mod polynomial;
mod spline;
mod bivariate;
mod traits;
mod utils;

pub use linear::*;
pub use polynomial::*;
pub use spline::*;
pub use bivariate::*;
pub use traits::*;

/// Re-export commonly used 1D interpolators
/// 
/// These are the most frequently used interpolation methods for 1D data.
/// All implement the `Interpolator1D` trait for consistent API.
pub use linear::LinearInterpolator;
pub use polynomial::{LagrangeInterpolator, NewtonInterpolator};
pub use spline::{CubicSpline, BoundaryCondition};

/// Re-export 2D interpolators for surface data
/// 
/// These methods work with regular grids to interpolate 2D surfaces.
/// All implement the `Interpolator2D` trait.
pub use bivariate::{BilinearInterpolator, BicubicInterpolator};