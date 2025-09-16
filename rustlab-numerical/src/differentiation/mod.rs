//! Advanced numerical differentiation methods for derivative approximation
//!
//! This module provides a comprehensive suite of algorithms for computing numerical
//! derivatives of continuous functions. All methods are designed for high accuracy,
//! numerical stability, and optimal error control with rigorous mathematical foundations
//! and practical applicability to scientific computing problems.
//!
//! ## Core Differentiation Methods
//!
//! ### Finite Difference Schemes
//!
//! #### Forward Differences
//! - **Formula**: f'(x) ≈ [aₙf(x+nh) + ... + a₁f(x+h) + a₀f(x)] / h
//! - **Accuracy Orders**: 1st, 2nd, 3rd, 4th order available
//! - **Error**: O(h^p) where p is the accuracy order
//! - **Use Case**: When only forward points are accessible (boundaries)
//! - **Stability**: Good for well-conditioned problems
//!
//! #### Backward Differences
//! - **Formula**: f'(x) ≈ [a₀f(x) + a₁f(x-h) + ... + aₙf(x-nh)] / h
//! - **Accuracy Orders**: 1st, 2nd, 3rd, 4th order available
//! - **Error**: O(h^p) where p is the accuracy order
//! - **Use Case**: When only backward points are accessible (boundaries)
//! - **Symmetry**: Mirror image of forward differences
//!
//! #### Central Differences
//! - **Formula**: f'(x) ≈ [aₙf(x+nh) + ... + a₁f(x+h) - a₁f(x-h) - ... - aₙf(x-nh)] / h
//! - **Accuracy Orders**: 2nd, 4th, 6th order available
//! - **Error**: O(h^p) with even orders only
//! - **Use Case**: Interior points, highest accuracy per function evaluation
//! - **Advantage**: Symmetric stencil reduces error coefficients
//!
//! ### Advanced Methods
//!
//! #### Richardson Extrapolation
//! - **Technique**: Combine estimates at multiple step sizes to eliminate error terms
//! - **Formula**: D = (2^p × D(h/2) - D(h)) / (2^p - 1)
//! - **Convergence**: Higher-order accuracy from lower-order methods
//! - **Use Case**: When maximum precision is required
//! - **Computational Cost**: ~2x base method for significantly improved accuracy
//!
//! #### Complex-Step Differentiation
//! - **Formula**: f'(x) = Im[f(x + ih)] / h
//! - **Accuracy**: Machine precision (no subtractive cancellation)
//! - **Error**: O(h²) + round-off, can use h ≈ 10⁻²⁰⁰
//! - **Use Case**: When exact derivatives are critical
//! - **Limitation**: Function must be analytic (complex differentiable)
//!
//! ## Mathematical Foundations
//!
//! ### Taylor Series Analysis
//!
//! All finite difference methods derive from Taylor series expansions:
//!
//! ```text
//! f(x+h) = f(x) + hf'(x) + h²f''(x)/2! + h³f'''(x)/3! + O(h⁴)
//! f(x-h) = f(x) - hf'(x) + h²f''(x)/2! - h³f'''(x)/3! + O(h⁴)
//! ```
//!
//! Linear combinations of these expansions eliminate lower-order terms.
//!
//! ### Error Analysis
//!
//! For functions f ∈ C^(p+1)[a,b], the truncation error is:
//!
//! ```text
//! Forward/Backward (order p):  |E| ≤ C₁ × h^p × max|f^(p+1)(x)|
//! Central (order p):          |E| ≤ C₂ × h^p × max|f^(p+1)(x)|
//! Richardson (level k):       |E| ≤ C₃ × h^(p+2k)
//! Complex-step:               |E| ≤ C₄ × h² (no cancellation error)
//! ```
//!
//! where C₁, C₂, C₃, C₄ are method-dependent constants.
//!
//! ### Step Size Selection
//!
//! Optimal step size balances truncation vs. round-off error:
//!
//! ```text
//! h_optimal ≈ (ε_machine / |f^(p+1)(x)|)^(1/(p+1))
//! ```
//!
//! where ε_machine ≈ 2.22 × 10⁻¹⁶ for double precision.
//!
//! ## Performance Characteristics
//!
//! | Method | Order | Function Calls | Memory | Error Type | Best h |
//! |--------|-------|----------------|---------|------------|--------|
//! | Forward 1st | O(h) | 2 | O(1) | Truncation | ~10⁻⁸ |
//! | Forward 4th | O(h⁴) | 5 | O(1) | Truncation | ~10⁻⁴ |
//! | Central 2nd | O(h²) | 2 | O(1) | Truncation | ~10⁻⁶ |
//! | Central 6th | O(h⁶) | 6 | O(1) | Truncation | ~10⁻³ |
//! | Richardson | O(h^(p+2)) | 2×base | O(1) | Extrapolation | Variable |
//! | Complex-step | O(h²) | 1 | O(1) | Round-off only | ~10⁻¹⁰⁰ |
//!
//! ## Usage Guidelines
//!
//! ### Method Selection Strategy
//!
//! 1. **General Purpose**: Central differences (2nd or 4th order)
//!    - Good balance of accuracy and computational cost
//!    - Suitable for most scientific applications
//!
//! 2. **High Precision**: Richardson extrapolation or complex-step
//!    - When derivative accuracy is critical
//!    - Research applications requiring maximum precision
//!
//! 3. **Boundary Points**: Forward/backward differences
//!    - When central differences cannot be applied
//!    - Near domain boundaries or discontinuities
//!
//! 4. **Real-time Applications**: Forward differences (1st order)
//!    - Minimal function evaluations
//!    - Acceptable accuracy for control applications
//!
//! ### Numerical Stability Considerations
//!
//! - **Function Regularity**: Ensure f ∈ C^(p+1) for order p accuracy
//! - **Step Size**: Too small → round-off errors, too large → truncation errors
//! - **Condition Number**: Well-conditioned functions give better results
//! - **Domain Validity**: Ensure all evaluation points are in function domain
//!
//! ## Examples
//!
//! ### Basic Derivative Computation
//! ```rust
//! use rustlab_numerical::differentiation::*;
//!
//! // Polynomial function: f(x) = x³ - 2x² + x
//! let f = |x: f64| x.powi(3) - 2.0*x.powi(2) + x;
//! let x = 2.0;
//!
//! // Central difference (4th order)
//! let df_central = central_diff(f, x, 1e-5, 4)?;
//! 
//! // Richardson extrapolation for higher precision
//! let df_richardson = richardson_extrapolation(
//!     f, x, 1e-3, 
//!     |f, x, h, ord| central_diff(f, x, h, ord), 2
//! )?;
//!
//! println!("Central 4th:  {:.12}", df_central);
//! println!("Richardson:   {:.12}", df_richardson);
//! # Ok::<(), rustlab_numerical::NumericalError>(())
//! ```
//!
//! ### Complex-Step for Maximum Accuracy
//! ```rust
//! use rustlab_numerical::differentiation::complex_step_diff;
//! use num_complex::Complex64;
//!
//! // Transcendental function requiring high precision
//! let f = |z: Complex64| (z.sin() * z.exp()) / (z.cos() + 1.0);
//! let x = 1.5;
//!
//! // Machine precision derivative
//! let df_exact = complex_step_diff(f, x, 1e-200)?;
//! println!("Exact derivative: {:.15}", df_exact);
//! # Ok::<(), rustlab_numerical::NumericalError>(())
//! ```
//!
//! ### Error Analysis and Convergence Study
//! ```rust,ignore
//! use rustlab_numerical::differentiation::{central_diff, forward_diff};
//!
//! fn convergence_study() -> Result<()> {
//!     let f = |x: f64| x.powi(5);          // f(x) = x^5
//!     let df_exact = |x: f64| 5.0 * x.powi(4);  // f'(x) = 5x^4
//!     let x = 1.5;
//!     let exact = df_exact(x);
//!
//!     println!("h        Forward 1st   Central 2nd   Central 4th");
//!     for i in 1..=8 {
//!         let h = 10.0_f64.powi(-i);
//!         
//!         let err_f1 = (forward_diff(f, x, h, 1)? - exact).abs();
//!         let err_c2 = (central_diff(f, x, h, 2)? - exact).abs();
//!         let err_c4 = (central_diff(f, x, h, 4)? - exact).abs();
//!         
//!         println!("{:.0e}  {:.2e}    {:.2e}    {:.2e}", h, err_f1, err_c2, err_c4);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Integration with Scientific Computing
//!
//! ### Automatic Differentiation Interface
//! ```rust,ignore
//! // Future: Automatic differentiation support
//! trait AutoDiff {
//!     fn gradient(&self, x: &[f64]) -> Vec<f64>;
//!     fn jacobian(&self, x: &[f64]) -> Array2<f64>;
//! }
//! ```
//!
//! ### Vector Field Differentiation
//! ```rust,ignore
//! // Compute divergence of vector field
//! fn divergence_2d<F1, F2>(fx: F1, fy: F2, x: f64, y: f64, h: f64) -> Result<f64>
//! where 
//!     F1: Fn(f64, f64) -> f64,
//!     F2: Fn(f64, f64) -> f64,
//! {
//!     let dfdx = central_diff(|x| fx(x, y), x, h, 4)?;
//!     let dfdy = central_diff(|y| fy(x, y), y, h, 4)?;
//!     Ok(dfdx + dfdy)
//! }
//! ```
//!
//! ## Future Extensions
//! - **Higher-Order Derivatives**: Second, third derivatives with optimal stencils
//! - **Multidimensional Gradients**: Partial derivatives, Jacobians, Hessians
//! - **Automatic Differentiation**: Forward and reverse mode AD
//! - **Sparse Derivatives**: Efficient computation for large-scale problems
//! - **Adaptive Step Selection**: Automatic optimal step size determination
//! - **Vector Functions**: Simultaneous differentiation of multiple components

mod finite_diff;

pub use finite_diff::*;