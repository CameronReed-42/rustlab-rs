//! Advanced root finding algorithms for solving nonlinear equations
//!
//! This module provides a comprehensive collection of robust, high-performance algorithms
//! for finding roots of nonlinear equations f(x) = 0. All methods are designed with
//! rigorous convergence analysis, optimal error control, and practical applicability
//! to scientific computing and engineering problems.
//!
//! ## Core Root Finding Methods
//!
//! ### Bracketing Methods (Guaranteed Convergence)
//!
//! #### Bisection Method
//! - **Requirement**: Initial bracket [a,b] with f(a)⋅f(b) < 0
//! - **Convergence**: Linear, guaranteed convergence
//! - **Rate**: |xₙ₊₁ - r| ≤ |xₙ - r|/2
//! - **Robustness**: Most reliable, never fails for continuous functions
//! - **Use Case**: When reliability is more important than speed
//!
//! #### Brent's Method
//! - **Technique**: Combines inverse quadratic interpolation, secant, and bisection
//! - **Convergence**: Superlinear, guaranteed within initial bracket
//! - **Rate**: Usually faster than bisection, as reliable as bisection
//! - **Robustness**: Optimal balance of speed and reliability
//! - **Use Case**: General-purpose root finding, recommended default method
//!
//! #### Ridders' Method
//! - **Technique**: Exponential interpolation with automatic bracketing
//! - **Convergence**: Superlinear O(√2) ≈ 1.41 convergence order
//! - **Rate**: |xₙ₊₁ - r| ≤ C|xₙ - r|^1.41
//! - **Advantage**: Faster than bisection, simpler than Brent's
//! - **Use Case**: When smooth function behavior is expected
//!
//! #### Illinois Method (Modified Regula Falsi)
//! - **Technique**: False position with stagnation prevention
//! - **Convergence**: Superlinear, maintains bracketing property
//! - **Rate**: Faster than bisection, avoids slow convergence of basic false position
//! - **Advantage**: Good for functions with unequal curvature
//! - **Use Case**: Alternative to Brent's when simpler logic is preferred
//!
//! ### Open Methods (Fast Convergence, No Bracketing Required)
//!
//! #### Newton-Raphson Method
//! - **Requirement**: Function f(x) and derivative f'(x)
//! - **Convergence**: Quadratic near simple roots
//! - **Rate**: |xₙ₊₁ - r| ≤ C|xₙ - r|²
//! - **Formula**: xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
//! - **Use Case**: When derivatives are available and good initial guess exists
//!
//! #### Secant Method
//! - **Requirement**: Only function f(x), no derivative needed
//! - **Convergence**: Superlinear with golden ratio φ ≈ 1.618
//! - **Rate**: |xₙ₊₁ - r| ≤ C|xₙ - r|^φ
//! - **Formula**: xₙ₊₁ = xₙ - f(xₙ)⋅(xₙ - xₙ₋₁)/(f(xₙ) - f(xₙ₋₁))
//! - **Use Case**: When derivatives are unavailable or expensive to compute
//!
//! ## Mathematical Foundations
//!
//! ### Convergence Theory
//!
//! For a simple root r (where f(r) = 0 and f'(r) ≠ 0), convergence orders are:
//!
//! ```text
//! Method          Order    Rate Constant
//! Bisection       1.0      0.5
//! Illinois        ~1.7     Varies
//! Ridders'        ~1.4     Function-dependent
//! Secant          φ≈1.618  |f''(r)|/(2|f'(r)|)
//! Newton-Raphson  2.0      |f''(r)|/(2|f'(r)|)
//! Brent           1.0-φ    Adaptive
//! ```
//!
//! ### Error Analysis
//!
//! #### Bracketing Methods Error Bounds
//! For methods that maintain a bracket [aₙ, bₙ]:
//! ```text
//! |xₙ - r| ≤ |bₙ - aₙ| = (b₀ - a₀)/2ⁿ  (Bisection)
//! |xₙ - r| ≤ |bₙ - aₙ| ≤ C⋅qⁿ, q < 1   (Superlinear methods)
//! ```
//!
//! #### Open Methods Error Analysis
//! For Newton-Raphson near simple roots:
//! ```text
//! eₙ₊₁ = eₙ² ⋅ f''(r)/(2f'(r)) + O(eₙ³)
//! ```
//!
//! For Secant method:
//! ```text
//! eₙ₊₁ ≈ eₙ ⋅ eₙ₋₁ ⋅ f''(r)/(2f'(r))
//! ```
//!
//! ### Function Evaluation Requirements
//!
//! | Method | f(x) Calls/Iteration | Derivative Required | Bracketing Required |
//! |--------|---------------------|-------------------|-------------------|
//! | Bisection | 1 | No | Yes |
//! | Brent | 1-2 | No | Yes |
//! | Ridders | 2 | No | Yes |
//! | Illinois | 1 | No | Yes |
//! | Newton | 2 (f + f') | Yes | No |
//! | Secant | 1 | No | No |
//!
//! ## Usage Guidelines
//!
//! ### Method Selection Strategy
//!
//! 1. **Default Choice**: Brent's method
//!    - Best general-purpose algorithm
//!    - Combines speed and reliability
//!    - Handles most practical cases optimally
//!
//! 2. **High-Precision Applications**: Newton-Raphson
//!    - When derivatives are easily computable
//!    - Requires good initial approximation
//!    - Fastest convergence near roots
//!
//! 3. **Robust Applications**: Bisection
//!    - When function is poorly behaved
//!    - Guaranteed convergence essential
//!    - Can tolerate slower convergence
//!
//! 4. **Derivative-Free Speed**: Secant method
//!    - When derivatives unavailable/expensive
//!    - Good initial approximations available
//!    - Can accept non-guaranteed convergence
//!
//! 5. **Smooth Functions**: Ridders' method
//!    - When function is well-behaved
//!    - Alternative to Brent's method
//!    - Simpler implementation than Brent's
//!
//! ### Convergence Conditions
//!
//! #### Sufficient Conditions for Convergence
//!
//! **Newton-Raphson**:
//! - f'(x) ≠ 0 near the root
//! - Initial guess x₀ in basin of attraction
//! - f, f' continuous in neighborhood of root
//!
//! **Secant Method**:
//! - f continuous, f'(r) ≠ 0
//! - Initial points x₀, x₁ sufficiently close to root
//! - Function not too nonlinear in the region
//!
//! **Bracketing Methods**:
//! - f continuous on [a,b]
//! - f(a)⋅f(b) < 0 (Intermediate Value Theorem)
//!
//! ## Examples
//!
//! ### Basic Root Finding
//! ```rust
//! use rustlab_numerical::roots::*;
//!
//! // Find √2 using different methods
//! let f = |x: f64| x*x - 2.0;
//! let df = |x: f64| 2.0*x;
//!
//! // Reliable bracketing method
//! let result1 = brent(f, 1.0, 2.0, 1e-12, 100)?;
//! 
//! // Fast convergence with derivative
//! let result2 = newton_raphson(f, df, 1.5, 1e-12, 100)?;
//! 
//! // Derivative-free alternative
//! let result3 = secant(f, 1.0, 2.0, 1e-12, 100)?;
//!
//! println!("Brent:   {:.15} ({} iterations)", result1.root, result1.iterations);
//! println!("Newton:  {:.15} ({} iterations)", result2.root, result2.iterations);
//! println!("Secant:  {:.15} ({} iterations)", result3.root, result3.iterations);
//! # Ok::<(), rustlab_numerical::NumericalError>(())
//! ```
//!
//! ### Transcendental Equation
//! ```rust
//! use rustlab_numerical::roots::*;
//! use std::f64::consts::PI;
//!
//! // Solve cos(x) - x = 0 (fixed point of cosine)
//! let f = |x: f64| x.cos() - x;
//! let df = |x: f64| -x.sin() - 1.0;
//!
//! // Multiple approaches for comparison
//! let brent_result = brent(f, 0.0, PI/2.0, 1e-15, 100)?;
//! let newton_result = newton_raphson(f, df, 0.5, 1e-15, 100)?;
//! let ridders_result = ridders(f, 0.0, PI/2.0, 1e-15, 100)?;
//!
//! // All should find the same root: ~0.7390851332151607
//! assert!((brent_result.root - newton_result.root).abs() < 1e-14);
//! assert!((brent_result.root - ridders_result.root).abs() < 1e-14);
//! # Ok::<(), rustlab_numerical::NumericalError>(())

mod scalar;

pub use scalar::*;