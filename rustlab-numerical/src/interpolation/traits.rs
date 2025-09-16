//! Core traits and types for interpolation methods
//!
//! This module defines the fundamental interface for all interpolation algorithms
//! in RustLab-Numerical. It provides a unified API that allows different interpolation
//! methods to be used interchangeably while maintaining type safety and performance.
//!
//! ## Trait Hierarchy
//!
//! ### 1D Interpolation
//! ```text
//! Interpolator1D (base trait)
//! └── DifferentiableInterpolator1D (adds derivative support)
//! ```
//!
//! ### 2D Interpolation
//! ```text
//! Interpolator2D (independent trait for surface interpolation)
//! ```
//!
//! ## Key Features
//!
//! ### Domain Management
//! - Automatic domain tracking and validation
//! - Configurable extrapolation behavior
//! - Boundary condition handling
//!
//! ### Vectorized Operations
//! - Batch evaluation for multiple points
//! - SIMD-friendly default implementations
//! - Memory-efficient result collection
//!
//! ### Derivative Support
//! - First and second derivative evaluation
//! - Consistent numerical differentiation interface
//! - Optimal accuracy for smooth interpolants
//!
//! ## Examples
//!
//! ### Basic Interpolation Interface
//! ```rust
//! use rustlab_numerical::interpolation::traits::*;
//! use rustlab_numerical::interpolation::LinearInterpolator;
//! use rustlab_math::VectorF64;
//!
//! // Any interpolator implements Interpolator1D
//! fn evaluate_at_points<T: Interpolator1D>(interp: &T, points: &[f64]) -> Vec<f64> {
//!     points.iter()
//!         .filter_map(|&x| interp.eval(x).ok())
//!         .collect()
//! }
//! 
//! let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
//! let y = VectorF64::from_slice(&[0.0, 1.0, 4.0]);
//! let linear = LinearInterpolator::new(&x, &y)?;
//! 
//! let results = evaluate_at_points(&linear, &[0.5, 1.5]);
//! # Ok::<(), rustlab_numerical::NumericalError>(())
//! ```
//!
//! ### Derivative Evaluation
//! ```rust
//! use rustlab_numerical::interpolation::traits::*;
//! use rustlab_numerical::interpolation::CubicSpline;
//! use rustlab_math::VectorF64;
//!
//! let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
//! let y = VectorF64::from_slice(&[0.0, 1.0, 4.0, 9.0]); // x^2
//! let spline = CubicSpline::natural(&x, &y)?;
//!
//! // Evaluate function and derivatives
//! let x_eval = 1.5;
//! let value = spline.eval(x_eval)?;                    // f(1.5)
//! let first_deriv = spline.eval_derivative(x_eval)?;   // f'(1.5) ≈ 3.0
//! let second_deriv = spline.eval_second_derivative(x_eval)?; // f''(1.5) ≈ 2.0
//! # Ok::<(), rustlab_numerical::NumericalError>(())
//! ```
//!
//! ## Extrapolation Strategies
//!
//! Different approaches for handling values outside the interpolation domain:
//!
//! - **Error**: Strict domain enforcement (default, safest)
//! - **Constant**: Use boundary values (simple, discontinuous derivatives)
//! - **Linear**: Linear extrapolation (smooth, can be unstable)
//! - **NaN**: Return NaN (explicit handling required)
//!
//! ## Performance Considerations
//!
//! ### Default Implementations
//! The `eval_vec` method provides an efficient default implementation that:
//! - Pre-allocates result vectors
//! - Uses vectorized operations where possible
//! - Handles errors gracefully without panicking
//!
//! ### Custom Implementations
//! For specialized interpolators, override `eval_vec` to:
//! - Leverage SIMD operations
//! - Implement custom memory layouts
//! - Optimize for specific use cases
//!
//! ## Thread Safety
//!
//! All interpolation traits require `Send + Sync`, making them safe for:
//! - Parallel evaluation across multiple threads
//! - Shared read-only access from concurrent contexts
//! - Integration with async/await patterns

use rustlab_math::VectorF64;
use crate::Result;

/// Core trait for 1D interpolation methods
///
/// This trait defines the essential interface that all 1D interpolation algorithms
/// must implement. It provides a consistent API for evaluating interpolants at
/// single points or arrays of points, along with domain management.
///
/// # Mathematical Foundation
///
/// An interpolator constructs a function f̂(x) that satisfies:
/// ```text
/// f̂(x_i) = y_i  for all data points (x_i, y_i)
/// ```
///
/// The specific form of f̂ depends on the interpolation method:
/// - **Linear**: Piecewise linear function
/// - **Polynomial**: Single polynomial of degree n-1
/// - **Spline**: Piecewise polynomial with continuity constraints
///
/// # Implementation Requirements
///
/// Implementors must ensure:
/// 1. **Consistency**: `eval(x)` returns the same value for the same input
/// 2. **Domain Accuracy**: `domain()` returns the exact interpolation bounds
/// 3. **Thread Safety**: All methods are safe for concurrent access
/// 4. **Error Handling**: Clear errors for invalid inputs
///
/// # Performance Guidelines
///
/// - `eval()` should be optimized for repeated calls
/// - Consider caching expensive computations
/// - Override `eval_vec()` for better vectorized performance
/// - Use binary search or similar for O(log n) lookup in sorted data
///
/// # Examples
///
/// ## Basic Implementation Pattern
/// ```rust,ignore
/// struct MyInterpolator {
///     x_data: Vec<f64>,
///     y_data: Vec<f64>,
///     // ... other fields
/// }
///
/// impl Interpolator1D for MyInterpolator {
///     fn eval(&self, x: f64) -> Result<f64> {
///         if !self.in_domain(x) {
///             return Err(NumericalError::OutOfBounds(
///                 format!("x = {} outside domain {:?}", x, self.domain())
///             ));
///         }
///         // Interpolation logic here
///         Ok(interpolated_value)
///     }
///     
///     fn domain(&self) -> (f64, f64) {
///         (*self.x_data.first().unwrap(), *self.x_data.last().unwrap())
///     }
/// }
/// ```
///
/// ## Usage as Trait Object
/// ```rust
/// use rustlab_numerical::interpolation::traits::*;
/// 
/// fn integrate_interpolant(interp: &dyn Interpolator1D, a: f64, b: f64, n: usize) -> Result<f64> {
///     let h = (b - a) / n as f64;
///     let mut sum = 0.0;
///     
///     for i in 0..=n {
///         let x = a + i as f64 * h;
///         let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
///         sum += weight * interp.eval(x)?;
///     }
///     
///     Ok(sum * h)
/// }
/// ```
pub trait Interpolator1D: Send + Sync {
    /// Evaluate the interpolant at a single point
    ///
    /// This is the core method that computes the interpolated value f̂(x) at
    /// the given point x. The implementation depends on the specific interpolation
    /// method but should always return consistent results for the same input.
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the interpolant
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Interpolated value at x
    /// * `Err(NumericalError)` - If x is outside domain or evaluation fails
    ///
    /// # Error Conditions
    ///
    /// - `OutOfBounds`: x is outside the interpolation domain and extrapolation is disabled
    /// - `NumericalInstability`: Evaluation failed due to numerical issues
    /// - `InvalidParameter`: x is NaN or infinite
    ///
    /// # Performance Notes
    ///
    /// This method is typically called many times, so implementations should:
    /// - Use efficient search algorithms (binary search for sorted data)
    /// - Cache intermediate computations when possible
    /// - Avoid unnecessary allocations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_numerical::interpolation::*;
    /// use rustlab_math::VectorF64;
    ///
    /// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
    /// let y = VectorF64::from_slice(&[1.0, 3.0, 5.0]); // f(x) = 2x + 1
    /// let interp = LinearInterpolator::new(&x, &y)?;
    ///
    /// assert!((interp.eval(0.5)? - 2.0).abs() < 1e-10); // Should be 2*0.5 + 1 = 2
    /// assert!((interp.eval(1.5)? - 4.0).abs() < 1e-10); // Should be 2*1.5 + 1 = 4
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    fn eval(&self, x: f64) -> Result<f64>;
    
    /// Evaluate the interpolant at multiple points efficiently
    ///
    /// This method provides vectorized evaluation of the interpolant at an array
    /// of points. The default implementation calls `eval()` for each point, but
    /// specialized interpolators can override this for better performance.
    ///
    /// # Arguments
    ///
    /// * `x` - Vector of points at which to evaluate the interpolant
    ///
    /// # Returns
    ///
    /// * `Ok(VectorF64)` - Vector of interpolated values with same length as input
    /// * `Err(NumericalError)` - If any evaluation fails
    ///
    /// # Error Handling
    ///
    /// If any single evaluation fails, the entire operation fails and returns
    /// an error. For partial results with error handling, use individual `eval()` calls.
    ///
    /// # Performance Optimizations
    ///
    /// Implementations can override this method to:
    /// - Use SIMD instructions for vectorized computation
    /// - Exploit sorted input for more efficient algorithms
    /// - Batch computations to reduce overhead
    /// - Parallel processing for large inputs
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_numerical::interpolation::*;
    /// use rustlab_math::VectorF64;
    ///
    /// let x_data = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
    /// let y_data = VectorF64::from_slice(&[0.0, 1.0, 4.0, 9.0]); // x^2
    /// let spline = CubicSpline::natural(&x_data, &y_data)?;
    ///
    /// // Evaluate at multiple points
    /// let x_eval = VectorF64::from_slice(&[0.5, 1.5, 2.5]);
    /// let results = spline.eval_vec(&x_eval)?;
    ///
    /// assert_eq!(results.len(), 3);
    /// // Results should be close to [0.25, 2.25, 6.25] for perfect x^2
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    ///
    /// # Custom Implementation Example
    ///
    /// ```rust,ignore
    /// impl Interpolator1D for MyFastInterpolator {
    ///     fn eval_vec(&self, x: &VectorF64) -> Result<VectorF64> {
    ///         // Custom vectorized implementation
    ///         let mut results = Vec::with_capacity(x.len());
    ///         
    ///         // Could use SIMD here for better performance
    ///         for i in 0..x.len() {
    ///             let xi = x.get(i).unwrap();
    ///             results.push(self.fast_eval(xi)?);
    ///         }
    ///         
    ///         Ok(VectorF64::from_slice(&results))
    ///     }
    /// }
    /// ```
    fn eval_vec(&self, x: &VectorF64) -> Result<VectorF64> {
        let mut results = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let xi = x.get(i).ok_or(crate::NumericalError::InvalidParameter("Index out of bounds".to_string()))?;
            results.push(self.eval(xi)?);
        }
        Ok(VectorF64::from_slice(&results))
    }
    
    /// Get the domain of interpolation as [x_min, x_max]
    ///
    /// Returns the bounds of the interpolation domain, i.e., the range of x values
    /// for which the interpolant is defined based on the original data points.
    /// Values outside this domain may trigger extrapolation behavior or errors
    /// depending on the implementation.
    ///
    /// # Returns
    ///
    /// A tuple `(x_min, x_max)` where:
    /// - `x_min`: Minimum x value in the interpolation domain
    /// - `x_max`: Maximum x value in the interpolation domain
    ///
    /// # Mathematical Interpretation
    ///
    /// For interpolation data points {(x₀, y₀), (x₁, y₁), ..., (xₙ, yₙ)}:
    /// ```text
    /// x_min = min(x₀, x₁, ..., xₙ)
    /// x_max = max(x₀, x₁, ..., xₙ)
    /// ```
    ///
    /// # Usage
    ///
    /// ```rust
    /// use rustlab_numerical::interpolation::*;
    /// use rustlab_math::VectorF64;
    ///
    /// let x = VectorF64::from_slice(&[1.0, 3.0, 2.0, 4.0]); // Unsorted data
    /// let y = VectorF64::from_slice(&[1.0, 9.0, 4.0, 16.0]);
    /// let interp = LinearInterpolator::new(&x, &y)?;
    ///
    /// let (min, max) = interp.domain();
    /// assert_eq!(min, 1.0); // Minimum of input x values
    /// assert_eq!(max, 4.0); // Maximum of input x values
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    fn domain(&self) -> (f64, f64);
    
    /// Check if a point is within the interpolation domain
    ///
    /// This is a convenience method that determines whether a given x value
    /// falls within the interpolation domain. It's useful for validating
    /// inputs before evaluation or implementing custom extrapolation logic.
    ///
    /// # Arguments
    ///
    /// * `x` - Point to check
    ///
    /// # Returns
    ///
    /// `true` if x is within [x_min, x_max], `false` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_numerical::interpolation::*;
    /// use rustlab_math::VectorF64;
    ///
    /// let x_data = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
    /// let y_data = VectorF64::from_slice(&[0.0, 1.0, 4.0]);
    /// let interp = LinearInterpolator::new(&x_data, &y_data)?;
    ///
    /// assert!(interp.in_domain(0.5));  // Within [0, 2]
    /// assert!(interp.in_domain(0.0));  // On boundary
    /// assert!(interp.in_domain(2.0));  // On boundary
    /// assert!(!interp.in_domain(-0.1)); // Outside domain
    /// assert!(!interp.in_domain(2.1));  // Outside domain
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    ///
    /// # Usage in Custom Logic
    ///
    /// ```rust,ignore
    /// fn safe_eval<T: Interpolator1D>(interp: &T, x: f64) -> Result<f64> {
    ///     if !interp.in_domain(x) {
    ///         // Custom extrapolation logic
    ///         let (min, max) = interp.domain();
    ///         if x < min {
    ///             return interp.eval(min); // Use boundary value
    ///         } else {
    ///             return interp.eval(max);
    ///         }
    ///     }
    ///     interp.eval(x)
    /// }
    /// ```
    fn in_domain(&self, x: f64) -> bool {
        let (min, max) = self.domain();
        x >= min && x <= max
    }
}

/// Strategies for handling values outside the interpolation domain
///
/// When evaluating an interpolant at points outside its defined domain,
/// different extrapolation strategies can be applied. Each has different
/// trade-offs between safety, smoothness, and numerical stability.
///
/// # Mathematical Background
///
/// Given interpolation domain [a, b], extrapolation handles evaluation at:
/// - x < a (left extrapolation)
/// - x > b (right extrapolation)
///
/// # Strategy Details
///
/// ## Error Mode (Default)
/// - **Behavior**: Returns `NumericalError::OutOfBounds`
/// - **Use Case**: Strict domain enforcement, safest option
/// - **Advantages**: Prevents accidental extrapolation errors
/// - **Disadvantages**: Requires explicit domain checking
///
/// ## Constant Mode
/// - **Behavior**: f̂(x) = f̂(a) for x < a, f̂(x) = f̂(b) for x > b
/// - **Use Case**: Simple boundary value extension
/// - **Advantages**: Bounded output, no instability
/// - **Disadvantages**: Discontinuous derivatives at boundaries
///
/// ## Linear Mode
/// - **Behavior**: Linear extension using boundary slopes
/// - **Use Case**: Smooth extrapolation for short distances
/// - **Advantages**: Continuous first derivative
/// - **Disadvantages**: Can become unstable for large extrapolations
///
/// ## NaN Mode
/// - **Behavior**: Returns `f64::NaN` for out-of-bounds values
/// - **Use Case**: Explicit handling of invalid regions
/// - **Advantages**: Clear indication of extrapolation
/// - **Disadvantages**: Requires NaN checking in downstream code
///
/// # Examples
///
/// ```rust
/// use rustlab_numerical::interpolation::traits::ExtrapolationMode;
/// use rustlab_numerical::interpolation::*;
/// use rustlab_math::VectorF64;
///
/// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
/// let y = VectorF64::from_slice(&[0.0, 1.0, 4.0]);
///
/// // Different extrapolation behaviors
/// match ExtrapolationMode::Error {
///     ExtrapolationMode::Error => {
///         // Will return error for x < 0 or x > 2
///         let interp = LinearInterpolator::new(&x, &y)?;
///         // assert!(interp.eval(-0.5).is_err());
///     }
///     ExtrapolationMode::Constant => {
///         // Would return y[0] = 0.0 for x < 0, y[2] = 4.0 for x > 2
///     }
///     ExtrapolationMode::Linear => {
///         // Would extrapolate using boundary slopes
///     }
///     ExtrapolationMode::NaN => {
///         // Would return NaN for out-of-bounds
///     }
/// }
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtrapolationMode {
    /// Return an error for out-of-bounds values (safest, default)
    Error,
    /// Use constant extrapolation (nearest boundary value)
    Constant,
    /// Use linear extrapolation (smooth but potentially unstable)
    Linear,
    /// Return NaN for out-of-bounds values (explicit handling required)
    NaN,
}

impl Default for ExtrapolationMode {
    fn default() -> Self {
        ExtrapolationMode::Error
    }
}

/// Extended trait for interpolators that support analytical derivatives
///
/// This trait extends `Interpolator1D` to provide analytical computation of
/// first and second derivatives. Not all interpolation methods support this
/// (e.g., linear interpolation has undefined second derivatives), but those
/// that do can provide more accurate derivatives than finite differences.
///
/// # Mathematical Foundation
///
/// For an interpolant f̂(x), this trait provides:
/// - First derivative: f̂'(x) = df̂/dx
/// - Second derivative: f̂''(x) = d²f̂/dx²
///
/// The accuracy and smoothness depend on the interpolation method:
/// - **Cubic Splines**: Continuous first and second derivatives
/// - **Polynomial**: Analytical derivatives of any order
/// - **Piecewise Linear**: First derivative exists but discontinuous
///
/// # Implementation Requirements
///
/// Implementors should ensure:
/// 1. Derivatives are consistent with the interpolant
/// 2. Same domain restrictions apply as for function evaluation
/// 3. Numerical stability is maintained, especially near boundaries
/// 4. Error handling follows the same patterns as `eval()`
///
/// # Examples
///
/// ## Cubic Spline Derivatives
/// ```rust
/// use rustlab_numerical::interpolation::traits::*;
/// use rustlab_numerical::interpolation::CubicSpline;
/// use rustlab_math::VectorF64;
///
/// // Interpolate x^3 (derivative should be 3x^2)
/// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
/// let y = VectorF64::from_slice(&[0.0, 1.0, 8.0, 27.0]); // x^3
/// let spline = CubicSpline::natural(&x, &y)?;
///
/// let x_eval = 2.0;
/// let value = spline.eval(x_eval)?;                    // Should be ≈ 8.0
/// let first_deriv = spline.eval_derivative(x_eval)?;   // Should be ≈ 12.0 (3*2^2)
/// let second_deriv = spline.eval_second_derivative(x_eval)?; // Should be ≈ 12.0 (6*2)
///
/// println!("f({}) = {:.3}", x_eval, value);
/// println!("f'({}) = {:.3}", x_eval, first_deriv);
/// println!("f''({}) = {:.3}", x_eval, second_deriv);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub trait DifferentiableInterpolator1D: Interpolator1D {
    /// Evaluate the first derivative f̂'(x) at a point
    ///
    /// Computes the analytical first derivative of the interpolant at the
    /// given point. This is typically more accurate than finite difference
    /// approximations and maintains consistency with the interpolant.
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the derivative
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - First derivative value f̂'(x)
    /// * `Err(NumericalError)` - If x is outside domain or evaluation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_numerical::interpolation::traits::*;
    /// use rustlab_numerical::interpolation::CubicSpline;
    /// use rustlab_math::VectorF64;
    ///
    /// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
    /// let y = VectorF64::from_slice(&[0.0, 1.0, 4.0]); // x^2
    /// let spline = CubicSpline::natural(&x, &y)?;
    ///
    /// // Derivative of x^2 at x=1.5 should be 2*1.5 = 3.0
    /// let deriv = spline.eval_derivative(1.5)?;
    /// assert!((deriv - 3.0).abs() < 0.1); // Allow for interpolation error
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    fn eval_derivative(&self, x: f64) -> Result<f64>;
    
    /// Evaluate the second derivative f̂''(x) at a point
    ///
    /// Computes the analytical second derivative of the interpolant at the
    /// given point. This provides information about the curvature of the
    /// interpolant and is useful for optimization and stability analysis.
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the second derivative
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Second derivative value f̂''(x)
    /// * `Err(NumericalError)` - If x is outside domain or evaluation fails
    ///
    /// # Mathematical Interpretation
    ///
    /// The second derivative represents:
    /// - **Curvature**: Positive values indicate upward curvature (convex)
    /// - **Inflection Points**: Zero values may indicate points of inflection
    /// - **Stability**: Large values may indicate numerical instability
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_numerical::interpolation::traits::*;
    /// use rustlab_numerical::interpolation::CubicSpline;
    /// use rustlab_math::VectorF64;
    ///
    /// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
    /// let y = VectorF64::from_slice(&[0.0, 1.0, 8.0, 27.0]); // x^3
    /// let spline = CubicSpline::natural(&x, &y)?;
    ///
    /// // Second derivative of x^3 at x=2 should be 6*2 = 12
    /// let second_deriv = spline.eval_second_derivative(2.0)?;
    /// println!("f''(2) = {:.3}", second_deriv);
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    fn eval_second_derivative(&self, x: f64) -> Result<f64>;
}

/// Core trait for 2D interpolation methods
///
/// This trait defines the interface for interpolating functions of two variables,
/// f̂(x, y), based on scattered data points or regular grids. 2D interpolation
/// is essential for surface fitting, image processing, and multidimensional data
/// analysis.
///
/// # Mathematical Foundation
///
/// A 2D interpolator constructs a function f̂(x, y) that satisfies:
/// ```text
/// f̂(x_i, y_i) = z_i  for all data points (x_i, y_i, z_i)
/// ```
///
/// Common approaches include:
/// - **Bilinear**: Linear interpolation in both dimensions
/// - **Bicubic**: Cubic interpolation with smooth derivatives
/// - **Scattered Data**: Radial basis functions, triangulation
///
/// # Implementation Requirements
///
/// Implementors must ensure:
/// 1. **Consistency**: Same input coordinates always return same result
/// 2. **Domain Accuracy**: `domain()` returns exact interpolation bounds
/// 3. **Thread Safety**: All methods safe for concurrent access
/// 4. **Error Handling**: Clear errors for invalid coordinates
///
/// # Examples
///
/// ## Basic 2D Evaluation
/// ```rust
/// use rustlab_numerical::interpolation::traits::*;
/// use rustlab_numerical::interpolation::BilinearInterpolator;
/// use rustlab_math::{VectorF64, ArrayF64};
///
/// // Create a 2D surface z = x*y
/// let x_grid = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
/// let y_grid = VectorF64::from_slice(&[0.0, 1.0]);
/// let mut z_data = ArrayF64::zeros(3, 2);
/// 
/// for i in 0..3 {
///     for j in 0..2 {
///         let x = x_grid.get(i).unwrap();
///         let y = y_grid.get(j).unwrap();
///         z_data.set(i, j, x * y).unwrap();
///     }
/// }
///
/// let interp = BilinearInterpolator::new(&x_grid, &y_grid, &z_data)?;
///
/// // Evaluate at interior point
/// let result = interp.eval(1.5, 0.5)?;
/// // Should be 1.5 * 0.5 = 0.75 for perfect bilinear interpolation
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub trait Interpolator2D: Send + Sync {
    /// Evaluate the interpolant at a 2D point (x, y)
    ///
    /// Computes the interpolated value f̂(x, y) at the given coordinates.
    /// The implementation depends on the specific 2D interpolation method
    /// but should always return consistent results for the same input.
    ///
    /// # Arguments
    ///
    /// * `x` - X-coordinate for evaluation
    /// * `y` - Y-coordinate for evaluation
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Interpolated value at (x, y)
    /// * `Err(NumericalError)` - If coordinates are outside domain or evaluation fails
    ///
    /// # Error Conditions
    ///
    /// - `OutOfBounds`: (x, y) is outside the interpolation domain
    /// - `NumericalInstability`: Evaluation failed due to numerical issues
    /// - `InvalidParameter`: x or y is NaN or infinite
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_numerical::interpolation::traits::*;
    /// use rustlab_numerical::interpolation::BilinearInterpolator;
    /// use rustlab_math::{VectorF64, ArrayF64};
    ///
    /// let x_grid = VectorF64::from_slice(&[0.0, 1.0]);
    /// let y_grid = VectorF64::from_slice(&[0.0, 1.0]);
    /// let mut z_data = ArrayF64::zeros(2, 2);
    /// 
    /// // Set corner values: z = x + y
    /// z_data.set(0, 0, 0.0).unwrap(); // (0,0) -> 0
    /// z_data.set(1, 0, 1.0).unwrap(); // (1,0) -> 1  
    /// z_data.set(0, 1, 1.0).unwrap(); // (0,1) -> 1
    /// z_data.set(1, 1, 2.0).unwrap(); // (1,1) -> 2
    ///
    /// let interp = BilinearInterpolator::new(&x_grid, &y_grid, &z_data)?;
    ///
    /// // Evaluate at center: should be (0+1+1+2)/4 = 1.0
    /// let result = interp.eval(0.5, 0.5)?;
    /// assert!((result - 1.0).abs() < 1e-10);
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    fn eval(&self, x: f64, y: f64) -> Result<f64>;
    
    /// Get the 2D domain of interpolation
    ///
    /// Returns the rectangular bounds of the interpolation domain as
    /// ((x_min, x_max), (y_min, y_max)). Values outside this domain
    /// may trigger extrapolation behavior or errors.
    ///
    /// # Returns
    ///
    /// A tuple of tuples `((x_min, x_max), (y_min, y_max))` where:
    /// - `x_min, x_max`: X-coordinate bounds
    /// - `y_min, y_max`: Y-coordinate bounds
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_numerical::interpolation::traits::*;
    /// use rustlab_numerical::interpolation::BilinearInterpolator;
    /// use rustlab_math::{VectorF64, ArrayF64};
    ///
    /// let x_grid = VectorF64::from_slice(&[1.0, 3.0, 2.0]); // Unsorted
    /// let y_grid = VectorF64::from_slice(&[0.0, 4.0, 2.0]); // Unsorted
    /// let z_data = ArrayF64::zeros(3, 3);
    ///
    /// let interp = BilinearInterpolator::new(&x_grid, &y_grid, &z_data)?;
    /// let ((x_min, x_max), (y_min, y_max)) = interp.domain();
    ///
    /// assert_eq!(x_min, 1.0); // Min of x_grid
    /// assert_eq!(x_max, 3.0); // Max of x_grid  
    /// assert_eq!(y_min, 0.0); // Min of y_grid
    /// assert_eq!(y_max, 4.0); // Max of y_grid
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    fn domain(&self) -> ((f64, f64), (f64, f64));
}