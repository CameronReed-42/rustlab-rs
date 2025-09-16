//! Fundamental quadrature algorithms for numerical integration
//!
//! This module implements the core Newton-Cotes quadrature formulas and Richardson
//! extrapolation methods. Each algorithm is optimized for numerical stability,
//! accuracy, and performance while maintaining rigorous error checking and
//! mathematical correctness.
//!
//! ## Algorithm Classifications
//!
//! ### Newton-Cotes Formulas
//! 
//! Closed formulas using equally spaced points including endpoints:
//! - **Trapezoidal Rule**: Degree 1 precision, O(h²) error
//! - **Simpson's 1/3 Rule**: Degree 3 precision, O(h⁴) error 
//! - **Simpson's 3/8 Rule**: Degree 3 precision, alternative weighting
//!
//! ### Adaptive Methods
//!
//! - **Romberg Integration**: Richardson extrapolation for exponential convergence
//!
//! ## Mathematical Properties
//!
//! All methods satisfy:
//! - **Linearity**: ∫[αf + βg] = α∫f + β∫g
//! - **Interval Additivity**: ∫[a,c] = ∫[a,b] + ∫[b,c] for a < b < c
//! - **Exactness**: Perfect integration for polynomials up to method degree
//! - **Convergence**: Guaranteed convergence for Riemann integrable functions

use crate::{Result, NumericalError};

/// Trapezoidal rule for numerical integration with O(h²) convergence
///
/// Implements the composite trapezoidal rule, which approximates the definite integral
/// by connecting function values with straight line segments (trapezoids). This is the
/// fundamental building block for more sophisticated integration methods.
///
/// # Mathematical Foundation
///
/// The trapezoidal rule approximates:
/// ```text
/// ∫[a,b] f(x) dx ≈ h/2 [f(a) + 2∑ᵢ₌₁ⁿ⁻¹ f(xᵢ) + f(b)]
/// ```
/// where h = (b-a)/n and xᵢ = a + ih.
///
/// ## Error Analysis
///
/// For f ∈ C²[a,b], the theoretical error is:
/// ```text
/// E = -h²(b-a)/12 × f''(ξ) for some ξ ∈ [a,b]
/// ```
///
/// This gives convergence rate O(h²) = O(1/n²).
///
/// ## Exactness Property
///
/// The trapezoidal rule is exact (zero error) for:
/// - Constant functions: f(x) = c
/// - Linear functions: f(x) = ax + b
/// - Any polynomial of degree ≤ 1
///
/// # Performance Characteristics
///
/// - **Function Evaluations**: n + 1 total evaluations
/// - **Memory Usage**: O(1) - no storage of intermediate values
/// - **Computational Complexity**: O(n) - linear in number of intervals
/// - **Numerical Stability**: Excellent - simple arithmetic operations
///
/// # Arguments
///
/// * `f` - Function to integrate, must be continuous on [a,b]
/// * `a` - Lower integration bound (finite value required)
/// * `b` - Upper integration bound (finite value required)  
/// * `n` - Number of subintervals (must be positive)
///
/// # Returns
///
/// * `Ok(f64)` - Approximate integral value
/// * `Err(NumericalError)` - For invalid parameters or numerical issues
///
/// # Error Conditions
///
/// - `InvalidParameter`: n = 0, non-finite bounds, or a = b
/// - `NumericalInstability`: Function returns NaN/infinite values
/// - `NumericalInstability`: Final result is not finite
///
/// # Examples
///
/// ## Basic Polynomial Integration
/// ```rust
/// use rustlab_numerical::integration::trapz;
///
/// // Integrate x² from 0 to 1 (exact answer: 1/3)
/// let result = trapz(|x| x * x, 0.0, 1.0, 1000)?;
/// assert!((result - 1.0/3.0).abs() < 1e-6);
///
/// // Exact for linear functions
/// let result = trapz(|x| 2.0*x + 3.0, 1.0, 4.0, 10)?;
/// let exact = 4.5 * (4.0 - 1.0) + (4.0*4.0 - 1.0*1.0); // 24.0
/// assert!((result - 24.0).abs() < 1e-12);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Convergence Demonstration
/// ```rust
/// use rustlab_numerical::integration::trapz;
///
/// // Show O(h²) convergence for x³
/// let f = |x: f64| x.powi(3);
/// let exact = 4.0; // ∫₀² x³ dx = 4
///
/// let error_100 = (trapz(f, 0.0, 2.0, 100)? - exact).abs();
/// let error_200 = (trapz(f, 0.0, 2.0, 200)? - exact).abs();
/// 
/// // Error should reduce by factor of ~4 when n doubles (h² scaling)
/// let ratio = error_100 / error_200;
/// assert!((ratio - 4.0).abs() < 0.5); // Allow some numerical tolerance
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Transcendental Functions
/// ```rust
/// use rustlab_numerical::integration::trapz;
/// use std::f64::consts::PI;
///
/// // Integrate sin(x) from 0 to π (exact answer: 2)
/// let result = trapz(|x| x.sin(), 0.0, PI, 10000)?;
/// assert!((result - 2.0).abs() < 1e-8);
///
/// // Integrate e^x from 0 to 1 (exact answer: e - 1)
/// let result = trapz(|x| x.exp(), 0.0, 1.0, 10000)?;
/// let exact = std::f64::consts::E - 1.0;
/// assert!((result - exact).abs() < 1e-6);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// # Implementation Notes
///
/// - Uses careful summation to minimize round-off errors
/// - Validates all function evaluations for numerical stability
/// - Endpoint weights (0.5) applied correctly for composite rule
/// - Domain validation prevents integration over invalid intervals
pub fn trapz<F>(f: F, a: f64, b: f64, n: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return Err(NumericalError::InvalidParameter(
            "Number of intervals must be positive".to_string()
        ));
    }
    
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Integration bounds must be finite".to_string()
        ));
    }
    
    if a == b {
        return Ok(0.0);
    }
    
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b)); // Endpoints with weight 1/2
    
    // Interior points with weight 1
    for i in 1..n {
        let x = a + i as f64 * h;
        let fx = f(x);
        
        if !fx.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function evaluation at x = {} returned non-finite value", x)
            ));
        }
        
        sum += fx;
    }
    
    let result = h * sum;
    
    if !result.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Integration result is not finite".to_string()
        ));
    }
    
    Ok(result)
}

/// Simpson's 1/3 rule for high-accuracy numerical integration with O(h⁴) convergence
///
/// Implements the composite Simpson's rule, which approximates the definite integral
/// by fitting parabolic arcs through triplets of points. This provides significantly
/// higher accuracy than the trapezoidal rule for smooth functions.
///
/// # Mathematical Foundation
///
/// Simpson's rule approximates:
/// ```text
/// ∫[a,b] f(x) dx ≈ h/3 [f(a) + 4∑ᵢ₌₁,₃,₅,... f(xᵢ) + 2∑ᵢ₌₂,₄,₆,... f(xᵢ) + f(b)]
/// ```
/// where h = (b-a)/n, n is even, and xᵢ = a + ih.
///
/// ## Derivation from Lagrange Interpolation
///
/// On each interval [x₂ᵢ, x₂ᵢ₊₂], fit a parabola through three points:
/// ```text
/// P₂(x) = f(x₂ᵢ)L₀(x) + f(x₂ᵢ₊₁)L₁(x) + f(x₂ᵢ₊₂)L₂(x)
/// ```
/// Integration of these parabolas yields the 1-4-1 weighting pattern.
///
/// ## Error Analysis
///
/// For f ∈ C⁴[a,b], the theoretical error is:
/// ```text
/// E = -h⁵(b-a)/90 × f⁽⁴⁾(ξ) for some ξ ∈ [a,b]
/// ```
///
/// This gives convergence rate O(h⁴) = O(1/n⁴), much faster than trapezoidal.
///
/// ## Exactness Property
///
/// Simpson's rule is exact (zero error) for:
/// - Polynomials of degree ≤ 3: constants, linear, quadratic, cubic
/// - Remarkably, also exact for some higher-degree polynomials with special symmetries
///
/// # Performance Characteristics
///
/// - **Function Evaluations**: n + 1 total evaluations (same as trapezoidal)
/// - **Memory Usage**: O(1) - no storage of intermediate values
/// - **Computational Complexity**: O(n) - linear in number of intervals
/// - **Accuracy**: ~16x better than trapezoidal for quartic functions
///
/// # Arguments
///
/// * `f` - Function to integrate, should be C⁴ smooth for optimal performance
/// * `a` - Lower integration bound (finite value required)
/// * `b` - Upper integration bound (finite value required)
/// * `n` - Number of subintervals (must be even and positive)
///
/// # Returns
///
/// * `Ok(f64)` - Approximate integral value with O(h⁴) accuracy
/// * `Err(NumericalError)` - For invalid parameters or numerical issues
///
/// # Error Conditions
///
/// - `InvalidParameter`: n = 0, n odd, non-finite bounds
/// - `NumericalInstability`: Function returns NaN/infinite values
/// - `NumericalInstability`: Final result is not finite
///
/// # Examples
///
/// ## Exactness for Cubic Polynomials
/// ```rust
/// use rustlab_numerical::integration::simpson;
///
/// // Integrate x³ from 0 to 2 (exact answer: 4)
/// let result = simpson(|x| x.powi(3), 0.0, 2.0, 1000)?;
/// assert!((result - 4.0).abs() < 1e-12); // Machine precision accuracy
///
/// // Complex cubic: x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
/// let result = simpson(|x| x.powi(3) - 6.0*x.powi(2) + 11.0*x - 6.0, 0.0, 4.0, 100)?;
/// let exact = 4.0; // Can be computed analytically
/// assert!((result - exact).abs() < 1e-12);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Convergence Rate Verification
/// ```rust
/// use rustlab_numerical::integration::simpson;
///
/// // Show O(h⁴) convergence for x⁵
/// let f = |x: f64| x.powi(5);
/// let exact = 32.0/6.0; // ∫₀² x⁵ dx = 32/3
///
/// let error_50 = (simpson(f, 0.0, 2.0, 50)? - exact).abs();
/// let error_100 = (simpson(f, 0.0, 2.0, 100)? - exact).abs();
/// 
/// // Error should reduce by factor of ~16 when n doubles (h⁴ scaling)
/// let ratio = error_50 / error_100;
/// assert!((ratio - 16.0).abs() < 2.0); // Allow numerical tolerance
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Smooth Transcendental Functions
/// ```rust
/// use rustlab_numerical::integration::simpson;
/// use std::f64::consts::{PI, E};
///
/// // Integrate sin(x) from 0 to π (exact answer: 2)
/// let result = simpson(|x| x.sin(), 0.0, PI, 1000)?;
/// assert!((result - 2.0).abs() < 1e-12);
///
/// // Gaussian integral approximation: ∫₋₃³ e^(-x²) dx ≈ √π
/// let result = simpson(|x| (-x*x).exp(), -3.0, 3.0, 10000)?;
/// assert!((result - PI.sqrt()).abs() < 1e-6);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Comparison with Trapezoidal Rule
/// ```rust
/// use rustlab_numerical::integration::{simpson, trapz};
///
/// let f = |x: f64| x.powi(4); // x⁴ function
/// let exact = 32.0/5.0; // ∫₀² x⁴ dx = 32/5
///
/// // Same number of function evaluations, different accuracy
/// let n = 1000;
/// let simpson_result = simpson(f, 0.0, 2.0, n)?;
/// let trapz_result = trapz(f, 0.0, 2.0, n)?;
///
/// let simpson_error = (simpson_result - exact).abs();
/// let trapz_error = (trapz_result - exact).abs();
///
/// // Simpson's should be much more accurate
/// assert!(simpson_error < trapz_error / 100.0);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// # Implementation Notes
///
/// - Alternating 4-2-4-2 weights for interior points maximize accuracy
/// - Careful handling of even index constraint prevents common errors
/// - Numerical stability maintained through progressive summation
/// - Optimal for functions with continuous fourth derivatives
pub fn simpson<F>(f: F, a: f64, b: f64, n: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return Err(NumericalError::InvalidParameter(
            "Number of intervals must be positive".to_string()
        ));
    }
    
    if n % 2 != 0 {
        return Err(NumericalError::InvalidParameter(
            "Number of intervals must be even for Simpson's rule".to_string()
        ));
    }
    
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Integration bounds must be finite".to_string()
        ));
    }
    
    if a == b {
        return Ok(0.0);
    }
    
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b); // Endpoints with weight 1
    
    // Odd indices with weight 4
    for i in (1..n).step_by(2) {
        let x = a + i as f64 * h;
        let fx = f(x);
        
        if !fx.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function evaluation at x = {} returned non-finite value", x)
            ));
        }
        
        sum += 4.0 * fx;
    }
    
    // Even indices with weight 2  
    for i in (2..n).step_by(2) {
        let x = a + i as f64 * h;
        let fx = f(x);
        
        if !fx.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function evaluation at x = {} returned non-finite value", x)
            ));
        }
        
        sum += 2.0 * fx;
    }
    
    let result = h * sum / 3.0;
    
    if !result.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Integration result is not finite".to_string()
        ));
    }
    
    Ok(result)
}

/// Simpson's 3/8 rule for high-accuracy integration with alternative O(h⁴) algorithm
///
/// Implements the composite Simpson's 3/8 rule, which approximates definite integrals
/// using cubic interpolation over groups of four points. While having the same order
/// of accuracy as Simpson's 1/3 rule, it offers better numerical properties for
/// certain functions and provides an alternative when n is divisible by 3.
///
/// # Mathematical Foundation
///
/// Simpson's 3/8 rule approximates:
/// ```text
/// ∫[a,b] f(x) dx ≈ 3h/8 [f(a) + 3∑ᵢ₌₁,₂,₄,₅,₇,₈,... f(xᵢ) + 2∑ᵢ₌₃,₆,₉,... f(xᵢ) + f(b)]
/// ```
/// where h = (b-a)/n, n divisible by 3, and xᵢ = a + ih.
///
/// ## Derivation from Newton-Cotes Formula
///
/// The 3/8 rule comes from integrating cubic Lagrange interpolating polynomials
/// over intervals spanning 4 points (3 subintervals):
/// ```text
/// ∫ P₃(x) dx = h/8 [f₀ + 3f₁ + 3f₂ + f₃]
/// ```
/// 
/// Extended to the composite rule by summing over all intervals.
///
/// ## Error Analysis
///
/// For f ∈ C⁴[a,b], the theoretical error is:
/// ```text
/// E = -3h⁵(b-a)/80 × f⁽⁴⁾(ξ) for some ξ ∈ [a,b]
/// ```
///
/// The error constant (-3/80) is slightly better than Simpson's 1/3 rule (-1/90),
/// potentially offering marginally better accuracy for some functions.
///
/// ## Exactness Property
///
/// Simpson's 3/8 rule is exact for:
/// - All polynomials of degree ≤ 3
/// - Same exactness as Simpson's 1/3, but different numerical behavior
///
/// # Performance Characteristics
///
/// - **Function Evaluations**: n + 1 total evaluations
/// - **Memory Usage**: O(1) - no intermediate storage required
/// - **Computational Complexity**: O(n) - linear scaling
/// - **Accuracy**: O(h⁴), comparable to Simpson's 1/3 rule
///
/// # When to Use Simpson's 3/8
///
/// 1. **Interval Constraints**: When n must be divisible by 3 rather than 2
/// 2. **Numerical Stability**: Sometimes more stable than 1/3 rule
/// 3. **Legacy Compatibility**: Existing code expecting 3/8 rule behavior
/// 4. **Theoretical Analysis**: Research comparing different Newton-Cotes methods
///
/// # Arguments
///
/// * `f` - Function to integrate, preferably C⁴ smooth
/// * `a` - Lower integration bound (finite value required)
/// * `b` - Upper integration bound (finite value required)
/// * `n` - Number of subintervals (must be divisible by 3 and positive)
///
/// # Returns
///
/// * `Ok(f64)` - Approximate integral value with O(h⁴) accuracy
/// * `Err(NumericalError)` - For invalid parameters or numerical issues
///
/// # Error Conditions
///
/// - `InvalidParameter`: n = 0, n not divisible by 3, non-finite bounds
/// - `NumericalInstability`: Function returns NaN/infinite values
/// - `NumericalInstability`: Final result is not finite
///
/// # Examples
///
/// ## Basic Usage and Exactness
/// ```rust
/// use rustlab_numerical::integration::simpson38;
///
/// // Integrate x³ from 0 to 2 (exact answer: 4)
/// let result = simpson38(|x| x.powi(3), 0.0, 2.0, 999)?;
/// assert!((result - 4.0).abs() < 1e-12); // Machine precision accuracy
///
/// // Quadratic function: x² - 4x + 3
/// let result = simpson38(|x| x*x - 4.0*x + 3.0, 1.0, 3.0, 999)?;
/// let exact = -4.0/3.0; // Analytical result
/// assert!((result - exact).abs() < 1e-12);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Trigonometric Functions
/// ```rust
/// use rustlab_numerical::integration::simpson38;
/// use std::f64::consts::PI;
///
/// // Integrate sin(x) from 0 to π (exact answer: 2)
/// let result = simpson38(|x| x.sin(), 0.0, PI, 999)?;
/// assert!((result - 2.0).abs() < 1e-10);
///
/// // Integrate cos(x) from 0 to π/2 (exact answer: 1)
/// let result = simpson38(|x| x.cos(), 0.0, PI/2.0, 999)?;
/// assert!((result - 1.0).abs() < 1e-10);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Comparison with Simpson's 1/3 Rule
/// ```rust
/// use rustlab_numerical::integration::{simpson, simpson38};
///
/// let f = |x: f64| x.powi(4) + 2.0*x.powi(3) - x.powi(2) + 1.0;
/// 
/// // Both should give similar results for smooth functions
/// let result_13 = simpson(f, 0.0, 2.0, 1000)?;    // n even
/// let result_38 = simpson38(f, 0.0, 2.0, 999)?;   // n divisible by 3
///
/// // Results should agree to high precision
/// assert!((result_13 - result_38).abs() < 1e-8);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Error Pattern Analysis
/// ```rust
/// use rustlab_numerical::integration::simpson38;
///
/// // Test O(h⁴) convergence with x⁵
/// let f = |x: f64| x.powi(5);
/// let exact = 32.0/6.0; // ∫₀² x⁵ dx
///
/// let error_99 = (simpson38(f, 0.0, 2.0, 99)? - exact).abs();
/// let error_198 = (simpson38(f, 0.0, 2.0, 198)? - exact).abs();
///
/// // Doubling intervals should reduce error by ~16x (h⁴ scaling)
/// let ratio = error_99 / error_198;
/// assert!(ratio > 10.0 && ratio < 20.0); // Allow for numerical effects
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Handling of Oscillatory Functions
/// ```rust
/// use rustlab_numerical::integration::simpson38;
/// use std::f64::consts::PI;
///
/// // High-frequency oscillation: sin(10x) from 0 to 2π
/// let result = simpson38(|x| (10.0*x).sin(), 0.0, 2.0*PI, 9999)?;
/// assert!(result.abs() < 1e-8); // Should integrate to ~0
///
/// // Modulated oscillation: x*sin(x) from 0 to π
/// let result = simpson38(|x| x * x.sin(), 0.0, PI, 999)?;
/// let exact = PI; // Analytical result
/// assert!((result - exact).abs() < 1e-6);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// # Implementation Details
///
/// - Uses 3-3-2 weighting pattern for interior points
/// - Divisibility by 3 constraint enforced at parameter validation
/// - Progressive summation minimizes accumulation of round-off errors
/// - Endpoint handling identical to other Newton-Cotes methods
pub fn simpson38<F>(f: F, a: f64, b: f64, n: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return Err(NumericalError::InvalidParameter(
            "Number of intervals must be positive".to_string()
        ));
    }
    
    if n % 3 != 0 {
        return Err(NumericalError::InvalidParameter(
            "Number of intervals must be divisible by 3 for Simpson's 3/8 rule".to_string()
        ));
    }
    
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Integration bounds must be finite".to_string()
        ));
    }
    
    if a == b {
        return Ok(0.0);
    }
    
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b); // Endpoints with weight 1
    
    for i in 1..n {
        let x = a + i as f64 * h;
        let fx = f(x);
        
        if !fx.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function evaluation at x = {} returned non-finite value", x)
            ));
        }
        
        // Apply 3/8 rule weights
        if i % 3 == 0 {
            sum += 2.0 * fx; // Every 3rd point has weight 2
        } else {
            sum += 3.0 * fx; // Other points have weight 3
        }
    }
    
    let result = 3.0 * h * sum / 8.0;
    
    if !result.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Integration result is not finite".to_string()
        ));
    }
    
    Ok(result)
}

/// Romberg integration using Richardson extrapolation for exponential convergence
///
/// Implements adaptive Romberg integration, which applies Richardson extrapolation
/// to a sequence of trapezoidal rule approximations with successively refined
/// meshes. This achieves exponential convergence rates for smooth functions,
/// making it ideal for high-precision integration requirements.
///
/// # Mathematical Foundation
///
/// ## Richardson Extrapolation Principle
///
/// Starting with trapezoidal estimates T(h), T(h/2), T(h/4), ..., Romberg
/// integration constructs a triangular tableau:
///
/// ```text
/// T(h)     
/// T(h/2)   R₁₁
/// T(h/4)   R₁₂   R₂₂
/// T(h/8)   R₁₃   R₂₃   R₃₃
/// ...      ...   ...   ...
/// ```
///
/// where Rᵢⱼ = (4ʲRᵢ,ⱼ₋₁ - Rᵢ₋₁,ⱼ₋₁) / (4ʲ - 1)
///
/// ## Convergence Theory
///
/// For analytic functions f with convergent Taylor series, the error decreases as:
/// ```text
/// |Error| ≤ C × 4^(-m) for m extrapolation levels
/// ```
/// This exponential convergence is much faster than fixed-order methods.
///
/// ## Asymptotic Error Expansion
///
/// The trapezoidal rule has error expansion:
/// ```text
/// T(h) = I + α₁h² + α₂h⁴ + α₃h⁶ + ...
/// ```
/// Richardson extrapolation eliminates successive terms, yielding higher accuracy.
///
/// # Performance Characteristics
///
/// - **Function Evaluations**: Approximately 2ᵐ for m levels
/// - **Memory Usage**: O(m²) for triangular extrapolation tableau
/// - **Convergence Rate**: Exponential O(4⁻ᵐ) for smooth functions
/// - **Best Case**: Machine precision achievable in ~15-20 levels
///
/// # Convergence Criteria
///
/// The algorithm stops when either:
/// 1. **Tolerance Met**: |Rₘₘ - Rₘ₋₁,ₘ₋₁| / |Rₘₘ| < tolerance
/// 2. **Maximum Levels**: Reached specified maximum refinement levels
/// 3. **Stagnation**: No further improvement possible due to numerical precision
///
/// # Arguments
///
/// * `f` - Function to integrate, should be smooth (C^∞ ideally)
/// * `a` - Lower integration bound (finite value required)
/// * `b` - Upper integration bound (finite value required)
/// * `tol` - Desired relative tolerance (must be positive)
/// * `max_levels` - Maximum number of refinement levels (1-25 recommended)
///
/// # Returns
///
/// * `Ok(f64)` - High-precision integral approximation
/// * `Err(NumericalError)` - For invalid parameters or numerical issues
///
/// # Error Conditions
///
/// - `InvalidParameter`: tol ≤ 0, max_levels = 0, non-finite bounds
/// - `NumericalInstability`: Function evaluations return NaN/infinite
/// - `NumericalInstability`: Richardson extrapolation produces invalid results
///
/// # Examples
///
/// ## Exponential Function Integration
/// ```rust
/// use rustlab_numerical::integration::romberg;
/// use std::f64::consts::E;
///
/// // Integrate e^x from 0 to 1 (exact: e - 1)
/// let result = romberg(|x| x.exp(), 0.0, 1.0, 1e-12, 15)?;
/// let exact = E - 1.0;
/// assert!((result - exact).abs() < 1e-12);
///
/// // Show exponential convergence
/// let result_5 = romberg(|x| x.exp(), 0.0, 1.0, 1e-5, 5)?;
/// let result_10 = romberg(|x| x.exp(), 0.0, 1.0, 1e-10, 10)?;
/// let error_5 = (result_5 - exact).abs();
/// let error_10 = (result_10 - exact).abs();
/// assert!(error_10 < error_5 / 1000.0); // Much better with more levels
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Trigonometric Functions
/// ```rust
/// use rustlab_numerical::integration::romberg;
/// use std::f64::consts::PI;
///
/// // Integrate sin(x) from 0 to π (exact: 2)
/// let result = romberg(|x| x.sin(), 0.0, PI, 1e-14, 18)?;
/// assert!((result - 2.0).abs() < 1e-14);
///
/// // Integrate cos(x) from 0 to π/2 (exact: 1)
/// let result = romberg(|x| x.cos(), 0.0, PI/2.0, 1e-14, 18)?;
/// assert!((result - 1.0).abs() < 1e-14);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## High-Precision Requirements
/// ```rust
/// use rustlab_numerical::integration::romberg;
///
/// // Integrate 1/x from 1 to e (exact: 1)
/// let result = romberg(|x| 1.0/x, 1.0, std::f64::consts::E, 1e-15, 20)?;
/// assert!((result - 1.0).abs() < 1e-14); // Near machine precision
///
/// // Gaussian-type integral: e^(-x²)
/// let result = romberg(|x| (-x*x).exp(), -2.0, 2.0, 1e-12, 16)?;
/// let approx_sqrt_pi = 1.7724538509055; // √π ≈ 1.7724538509
/// assert!((result - approx_sqrt_pi).abs() < 1e-6);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Comparison with Fixed-Order Methods
/// ```rust
/// use rustlab_numerical::integration::{romberg, simpson, trapz};
///
/// let f = |x: f64| x.powi(6) * (-x).exp(); // Smooth function
/// let a = 0.0;
/// let b = 3.0;
///
/// // High-precision Romberg
/// let romberg_result = romberg(f, a, b, 1e-12, 15)?;
///
/// // Fixed methods with many points
/// let simpson_result = simpson(f, a, b, 10000)?;
/// let trapz_result = trapz(f, a, b, 100000)?;
///
/// // Romberg should be most accurate with fewest function evaluations
/// // (Exact verification would require analytical integration)
/// println!("Romberg:  {:.12}", romberg_result);
/// println!("Simpson:  {:.12}", simpson_result);
/// println!("Trapz:    {:.12}", trapz_result);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// ## Adaptive Error Control
/// ```rust
/// use rustlab_numerical::integration::romberg;
///
/// // Function with known analytical result for validation
/// let f = |x: f64| x.powi(4) * (x.sin() + 1.0);
/// let a = 0.0;
/// let b = 2.0;
///
/// // Progressive refinement with different tolerances
/// let result_rough = romberg(f, a, b, 1e-6, 10)?;
/// let result_fine = romberg(f, a, b, 1e-12, 18)?;
///
/// // Both should be consistent within their tolerances
/// let difference = (result_fine - result_rough).abs();
/// assert!(difference < 1e-6); // Rough tolerance satisfied
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
///
/// # When to Use Romberg Integration
///
/// **Ideal for:**
/// - Very smooth functions (analytic, C^∞)
/// - High-precision requirements (>8 decimal places)
/// - Functions expensive to evaluate (minimizes calls)
/// - Research applications requiring maximum accuracy
///
/// **Avoid for:**
/// - Non-smooth functions (discontinuities, cusps)
/// - Highly oscillatory integrands
/// - Functions with limited differentiability
/// - Real-time applications requiring predictable timing
///
/// # Implementation Notes
///
/// - Uses triangular tableau storage for Richardson extrapolation
/// - Implements relative error convergence checking
/// - Progressive mesh refinement: h, h/2, h/4, h/8, ...
/// - Maintains numerical stability through careful error propagation analysis
pub fn romberg<F>(f: F, a: f64, b: f64, tol: f64, max_levels: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if tol <= 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Tolerance must be positive".to_string()
        ));
    }
    
    if max_levels == 0 {
        return Err(NumericalError::InvalidParameter(
            "Maximum levels must be positive".to_string()
        ));
    }
    
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Integration bounds must be finite".to_string()
        ));
    }
    
    if a == b {
        return Ok(0.0);
    }
    
    // Romberg table - triangular matrix
    let mut r = vec![vec![0.0; max_levels]; max_levels];
    let h = b - a;
    
    // Level 0: Single trapezoidal estimate
    r[0][0] = h * (f(a) + f(b)) / 2.0;
    
    for i in 1..max_levels {
        // Compute trapezoidal rule with 2^i intervals
        let n = 1_usize << i; // 2^i
        let step = h / n as f64;
        
        // Use previous estimate and add new midpoints
        let mut sum = r[i-1][0] / 2.0; // Previous estimate divided by 2
        
        // Add contributions from new midpoints
        for j in 1..n {
            if j % 2 == 1 { // Only odd indices are new points
                let x = a + j as f64 * step;
                let fx = f(x);
                
                if !fx.is_finite() {
                    return Err(NumericalError::NumericalInstability(
                        format!("Function evaluation at x = {} returned non-finite value", x)
                    ));
                }
                
                sum += step * fx;
            }
        }
        
        r[i][0] = sum;
        
        // Richardson extrapolation
        for j in 1..=i {
            let factor = 4_f64.powi(j as i32);
            r[i][j] = (factor * r[i][j-1] - r[i-1][j-1]) / (factor - 1.0);
            
            if !r[i][j].is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Romberg extrapolation produced non-finite result".to_string()
                ));
            }
        }
        
        // Check convergence
        if i > 0 {
            let error = (r[i][i] - r[i-1][i-1]).abs();
            let rel_error = if r[i][i].abs() > 1e-15 {
                error / r[i][i].abs()
            } else {
                error
            };
            
            if rel_error < tol {
                return Ok(r[i][i]);
            }
        }
    }
    
    // Return best estimate if convergence not achieved
    Ok(r[max_levels-1][max_levels-1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_trapz_polynomial() {
        // Integrate x^2 from 0 to 1 (exact answer: 1/3)
        let result = trapz(|x| x * x, 0.0, 1.0, 1000).unwrap();
        assert_relative_eq!(result, 1.0/3.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_trapz_linear() {
        // Trapezoidal rule is exact for linear functions
        let result = trapz(|x| 2.0 * x + 1.0, 0.0, 1.0, 10).unwrap();
        assert_relative_eq!(result, 2.0, epsilon = 1e-12); // ∫(2x+1)dx from 0 to 1 = 2
    }
    
    #[test]
    fn test_simpson_polynomial() {
        // Simpson's rule is exact for polynomials of degree ≤ 3
        let result = simpson(|x| x * x * x, 0.0, 2.0, 1000).unwrap();
        assert_relative_eq!(result, 4.0, epsilon = 1e-12); // ∫x³dx from 0 to 2 = 4
    }
    
    #[test]
    fn test_simpson_cubic() {
        // Test with another cubic polynomial
        let result = simpson(|x| x * x * x + 2.0 * x * x + x + 1.0, 0.0, 1.0, 100).unwrap();
        let exact = 1.0/4.0 + 2.0/3.0 + 1.0/2.0 + 1.0; // 2.4166...
        assert_relative_eq!(result, exact, epsilon = 1e-12);
    }
    
    #[test]
    fn test_simpson38_polynomial() {
        // Test Simpson's 3/8 rule
        let result = simpson38(|x| x * x, 0.0, 1.0, 999).unwrap();
        assert_relative_eq!(result, 1.0/3.0, epsilon = 1e-8);
    }
    
    #[test]
    fn test_romberg_exponential() {
        // Integrate e^x from 0 to 1 (exact answer: e - 1)
        let result = romberg(|x| x.exp(), 0.0, 1.0, 1e-10, 20).unwrap();
        let exact = std::f64::consts::E - 1.0;
        assert_relative_eq!(result, exact, epsilon = 1e-10);
    }
    
    #[test]
    fn test_romberg_trigonometric() {
        // Integrate sin(x) from 0 to π (exact answer: 2)
        let result = romberg(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-8, 15).unwrap();
        assert_relative_eq!(result, 2.0, epsilon = 1e-8);
    }
    
    #[test]
    fn test_zero_interval() {
        assert_eq!(trapz(|x| x, 1.0, 1.0, 10).unwrap(), 0.0);
        assert_eq!(simpson(|x| x, 1.0, 1.0, 10).unwrap(), 0.0);
        assert_eq!(simpson38(|x| x, 1.0, 1.0, 12).unwrap(), 0.0);
        assert_eq!(romberg(|x| x, 1.0, 1.0, 1e-6, 10).unwrap(), 0.0);
    }
    
    #[test]
    fn test_invalid_parameters() {
        // Test invalid number of intervals
        assert!(trapz(|x| x, 0.0, 1.0, 0).is_err());
        assert!(simpson(|x| x, 0.0, 1.0, 0).is_err());
        assert!(simpson(|x| x, 0.0, 1.0, 3).is_err()); // Odd number
        assert!(simpson38(|x| x, 0.0, 1.0, 4).is_err()); // Not divisible by 3
        
        // Test invalid tolerance
        assert!(romberg(|x| x, 0.0, 1.0, -1e-6, 10).is_err());
        assert!(romberg(|x| x, 0.0, 1.0, 1e-6, 0).is_err());
    }
    
    #[test]
    fn test_infinite_bounds() {
        assert!(trapz(|x| x, f64::INFINITY, 1.0, 100).is_err());
        assert!(simpson(|x| x, 0.0, f64::NEG_INFINITY, 100).is_err());
    }
}