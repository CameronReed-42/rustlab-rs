//! Advanced numerical integration methods for definite integrals
//!
//! This module provides a comprehensive suite of numerical quadrature algorithms for
//! computing definite integrals of continuous functions. All methods are designed for
//! high accuracy, numerical stability, and performance optimization with careful error
//! handling and convergence analysis.
//!
//! ## Core Integration Methods
//!
//! ### Newton-Cotes Quadrature Family
//!
//! #### Trapezoidal Rule
//! - **Formula**: ∫[a,b] f(x)dx ≈ (h/2)[f(a) + 2∑f(xᵢ) + f(b)]
//! - **Order**: O(h²) - second-order accuracy
//! - **Error**: -h³(b-a)f''(ξ)/12 for some ξ ∈ [a,b]
//! - **Use Case**: General-purpose integration, exact for linear functions
//! - **Performance**: O(n) function evaluations, minimal memory usage
//!
//! #### Simpson's Rule (1/3 Rule)
//! - **Formula**: ∫[a,b] f(x)dx ≈ (h/3)[f(a) + 4∑f(odd) + 2∑f(even) + f(b)]
//! - **Order**: O(h⁴) - fourth-order accuracy
//! - **Error**: -h⁵(b-a)f⁽⁴⁾(ξ)/90 for some ξ ∈ [a,b]
//! - **Use Case**: High-accuracy integration, exact for cubic polynomials
//! - **Constraint**: Requires even number of intervals
//!
//! #### Simpson's 3/8 Rule
//! - **Formula**: ∫[a,b] f(x)dx ≈ (3h/8)[f(a) + 3∑f(mod3≠0) + 2∑f(mod3=0) + f(b)]
//! - **Order**: O(h⁴) - fourth-order accuracy
//! - **Error**: -3h⁵(b-a)f⁽⁴⁾(ξ)/80 for some ξ ∈ [a,b]
//! - **Use Case**: Alternative to Simpson's 1/3, better stability
//! - **Constraint**: Requires intervals divisible by 3
//!
//! ### Adaptive Methods
//!
//! #### Romberg Integration
//! - **Technique**: Richardson extrapolation on trapezoidal estimates
//! - **Convergence**: Exponential for smooth functions
//! - **Accuracy**: Arbitrarily high precision achievable
//! - **Error Control**: Automatic refinement until tolerance met
//! - **Use Case**: High-precision integration of smooth functions
//!
//! ## Mathematical Foundations
//!
//! ### Error Analysis
//!
//! For smooth functions f ∈ C^k[a,b], the error bounds are:
//!
//! ```text
//! Trapezoidal: |E| ≤ (b-a)h²/12 × max|f''(x)|
//! Simpson's:   |E| ≤ (b-a)h⁴/180 × max|f⁽⁴⁾(x)|
//! Romberg:     |E| ≤ C × 4^(-m) for m Richardson levels
//! ```
//!
//! ### Convergence Rates
//!
//! As the number of intervals n increases:
//! - Trapezoidal: Error ∝ O(1/n²)
//! - Simpson's: Error ∝ O(1/n⁴)
//! - Romberg: Error ∝ O(4^(-m)) exponentially
//!
//! ## Performance Characteristics
//!
//! | Method | Accuracy | Function Calls | Memory | Best For |
//! |--------|----------|----------------|---------|----------|
//! | Trapezoidal | O(h²) | n+1 | O(1) | Linear functions, real-time |
//! | Simpson's | O(h⁴) | n+1 | O(1) | Smooth functions, moderate precision |
//! | Simpson's 3/8 | O(h⁴) | n+1 | O(1) | Alternative to Simpson's 1/3 |
//! | Romberg | O(4^-m) | ~2^m | O(m²) | High precision, smooth functions |
//!
//! ## Usage Guidelines
//!
//! ### Choosing the Right Method
//!
//! 1. **Trapezoidal Rule**: 
//!    - Functions with limited smoothness
//!    - Real-time applications requiring speed
//!    - Linear or nearly linear functions
//!
//! 2. **Simpson's Rule**: 
//!    - Smooth functions requiring moderate precision
//!    - Functions well-approximated by cubic polynomials
//!    - Standard scientific computing applications
//!
//! 3. **Romberg Integration**: 
//!    - High-precision requirements (≥10 digits)
//!    - Very smooth functions (C^∞)
//!    - Research applications requiring maximum accuracy
//!
//! ### Numerical Stability Considerations
//!
//! - **Function Evaluation**: All methods check for NaN/infinite values
//! - **Overflow Protection**: Results validated for finite values
//! - **Domain Validation**: Integration bounds must be finite
//! - **Parameter Validation**: Intervals, tolerances checked for validity
//!
//! ## Examples
//!
//! ### Quick Integration
//! ```rust
//! use rustlab_numerical::integration::*;
//!
//! // Simple polynomial integration
//! let result = simpson(|x| x.powi(3), 0.0, 2.0, 1000)?;
//! assert!((result - 4.0).abs() < 1e-12); // Exact: ∫₀² x³ dx = 4
//!
//! // Transcendental function with high precision
//! let result = romberg(|x| x.exp(), 0.0, 1.0, 1e-12, 20)?;
//! let exact = std::f64::consts::E - 1.0;
//! assert!((result - exact).abs() < 1e-12);
//! # Ok::<(), rustlab_numerical::NumericalError>(())
//! ```
//!
//! ### Adaptive Integration Strategy
//! ```rust
//! use rustlab_numerical::integration::*;
//!
//! fn adaptive_integrate<F>(f: F, a: f64, b: f64, target_error: f64) -> Result<f64>
//! where F: Fn(f64) -> f64 + Copy
//! {
//!     // Try Romberg first for high precision
//!     if let Ok(result) = romberg(f, a, b, target_error, 15) {
//!         return Ok(result);
//!     }
//!     
//!     // Fall back to Simpson's rule with refined mesh
//!     let n = ((b - a) / target_error.sqrt()).ceil() as usize;
//!     let n_even = if n % 2 == 0 { n } else { n + 1 };
//!     simpson(f, a, b, n_even)
//! }
//! # fn main() -> rustlab_numerical::Result<()> { Ok(()) }
//! ```
//!
//! ## Integration with Scientific Computing Ecosystem
//!
//! ### Function Types Supported
//! - Closure functions: `|x| x.sin() + x.cos()`
//! - Function pointers: Standard mathematical functions
//! - Smooth continuous functions over finite intervals
//! - Piecewise continuous functions (with care at discontinuities)
//!
//! ### Error Handling Philosophy
//! - **Fail Fast**: Invalid parameters cause immediate errors
//! - **Numerical Safety**: Function evaluations checked for stability
//! - **Graceful Degradation**: Adaptive methods provide best estimate
//! - **Clear Diagnostics**: Detailed error messages for debugging
//!
//! ## Advanced Usage Patterns
//!
//! ### Custom Convergence Criteria
//! ```rust,ignore
//! // Custom Romberg with absolute tolerance
//! fn romberg_absolute<F>(f: F, a: f64, b: f64, abs_tol: f64) -> Result<f64>
//! where F: Fn(f64) -> f64
//! {
//!     // Implementation would check absolute rather than relative error
//!     romberg(f, a, b, abs_tol / (b - a).abs(), 20)
//! }
//! ```
//!
//! ### Vector-Valued Integration
//! ```rust,ignore
//! // Integrate multiple functions simultaneously
//! fn integrate_system<F>(functions: Vec<F>, a: f64, b: f64, n: usize) -> Vec<Result<f64>>
//! where F: Fn(f64) -> f64
//! {
//!     functions.into_iter()
//!         .map(|f| simpson(f, a, b, n))
//!         .collect()
//! }
//! ```
//!
//! ## Future Extensions
//! - **Gaussian Quadrature**: Legendre, Chebyshev, Laguerre, Hermite rules
//! - **Multi-dimensional Integration**: Monte Carlo, cubature rules
//! - **Singularity Handling**: Adaptive mesh refinement near singularities
//! - **Infinite Intervals**: Coordinate transformations for [0,∞), (-∞,∞)
//! - **Parallel Integration**: Multi-threaded evaluation for large intervals

mod quadrature;

pub use quadrature::*;