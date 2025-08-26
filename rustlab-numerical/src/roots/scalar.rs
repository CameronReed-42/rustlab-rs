//! Scalar root finding methods
//! 
//! This module provides various algorithms for finding roots of scalar functions:
//! - Bisection method (bracketing)
//! - Newton-Raphson method (requires derivative)
//! - Secant method (derivative-free)
//! - Brent's method (combines multiple approaches)
//! - Ridders' method (exponential interpolation)
//! - Illinois/Regula Falsi method (modified false position)

use crate::{Result, NumericalError};

/// Result of a root finding operation
#[derive(Debug, Clone)]
pub struct RootResult {
    /// The computed root
    pub root: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Final function value at the root
    pub function_value: f64,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Bisection method for root finding
/// 
/// Finds a root of f(x) = 0 in the interval [a, b] where f(a) and f(b) have opposite signs.
/// This method is guaranteed to converge but is relatively slow (linear convergence).
/// 
/// # Arguments
/// * `f` - Function whose root to find
/// * `a` - Left endpoint of bracketing interval
/// * `b` - Right endpoint of bracketing interval
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
/// 
/// # Example
/// ```
/// use rustlab_numerical::roots::bisection;
/// 
/// // Find root of x^2 - 2 = 0 (should be √2 ≈ 1.414)
/// let result = bisection(|x| x * x - 2.0, 1.0, 2.0, 1e-10, 100)?;
/// assert!((result.root - 2.0_f64.sqrt()).abs() < 1e-10);
/// assert!(result.converged);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn bisection<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Result<RootResult>
where
    F: Fn(f64) -> f64,
{
    if tol <= 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Tolerance must be positive".to_string()
        ));
    }
    
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Interval endpoints must be finite".to_string()
        ));
    }
    
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    
    let fa = f(a);
    let fb = f(b);
    
    if !fa.is_finite() || !fb.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Function evaluation returned non-finite value".to_string()
        ));
    }
    
    // Check if we already have a root at the endpoints
    if fa.abs() < tol {
        return Ok(RootResult {
            root: a,
            iterations: 0,
            function_value: fa,
            converged: true,
        });
    }
    
    if fb.abs() < tol {
        return Ok(RootResult {
            root: b,
            iterations: 0,
            function_value: fb,
            converged: true,
        });
    }
    
    // Check that the function has opposite signs at the endpoints
    if fa * fb > 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Function must have opposite signs at interval endpoints".to_string()
        ));
    }
    
    let mut iterations = 0;
    let mut c = (a + b) / 2.0;
    let mut fc = f(c);
    
    while iterations < max_iter {
        iterations += 1;
        
        if !fc.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function evaluation at x = {} returned non-finite value", c)
            ));
        }
        
        // Check convergence
        if fc.abs() < tol || (b - a) / 2.0 < tol {
            return Ok(RootResult {
                root: c,
                iterations,
                function_value: fc,
                converged: true,
            });
        }
        
        // Update interval
        if fa * fc < 0.0 {
            b = c;
        } else {
            a = c;
            // Note: we don't update fa since we'll compute fc again
        }
        
        c = (a + b) / 2.0;
        fc = f(c);
    }
    
    Ok(RootResult {
        root: c,
        iterations,
        function_value: fc,
        converged: false,
    })
}

/// Newton-Raphson method for root finding
/// 
/// Finds a root of f(x) = 0 using the derivative f'(x). Has quadratic convergence
/// when close to a simple root, but may not converge for poor initial guesses.
/// 
/// # Arguments
/// * `f` - Function whose root to find
/// * `df` - Derivative of the function
/// * `x0` - Initial guess
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
/// 
/// # Example
/// ```
/// use rustlab_numerical::roots::newton_raphson;
/// 
/// // Find root of x^3 - 2x - 5 = 0 (derivative: 3x^2 - 2)
/// let f = |x: f64| x * x * x - 2.0 * x - 5.0;
/// let df = |x: f64| 3.0 * x * x - 2.0;
/// let result = newton_raphson(f, df, 2.0, 1e-10, 100)?;
/// 
/// // Verify the root
/// assert!(f(result.root).abs() < 1e-10);
/// assert!(result.converged);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn newton_raphson<F, DF>(
    f: F, 
    df: DF, 
    mut x: f64, 
    tol: f64, 
    max_iter: usize
) -> Result<RootResult>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    if tol <= 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Tolerance must be positive".to_string()
        ));
    }
    
    if !x.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Initial guess must be finite".to_string()
        ));
    }
    
    let mut iterations = 0;
    
    while iterations < max_iter {
        let fx = f(x);
        let dfx = df(x);
        
        if !fx.is_finite() || !dfx.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function or derivative evaluation at x = {} returned non-finite value", x)
            ));
        }
        
        // Check convergence
        if fx.abs() < tol {
            return Ok(RootResult {
                root: x,
                iterations,
                function_value: fx,
                converged: true,
            });
        }
        
        // Check for zero derivative
        if dfx.abs() < 1e-15 {
            return Err(NumericalError::NumericalInstability(
                "Derivative is too close to zero - Newton-Raphson may not converge".to_string()
            ));
        }
        
        // Newton-Raphson update: x_{n+1} = x_n - f(x_n) / f'(x_n)
        let x_new = x - fx / dfx;
        
        if !x_new.is_finite() {
            return Err(NumericalError::NumericalInstability(
                "Newton-Raphson update produced non-finite value".to_string()
            ));
        }
        
        // Check for convergence in x
        if (x_new - x).abs() < tol {
            return Ok(RootResult {
                root: x_new,
                iterations: iterations + 1,
                function_value: f(x_new),
                converged: true,
            });
        }
        
        x = x_new;
        iterations += 1;
    }
    
    Ok(RootResult {
        root: x,
        iterations,
        function_value: f(x),
        converged: false,
    })
}

/// Secant method for root finding
/// 
/// Finds a root of f(x) = 0 using two initial points and approximating the derivative
/// with a finite difference. Has superlinear convergence (≈1.618) and doesn't require
/// the derivative.
/// 
/// # Arguments
/// * `f` - Function whose root to find
/// * `x0` - First initial point
/// * `x1` - Second initial point
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
/// 
/// # Example
/// ```
/// use rustlab_numerical::roots::secant;
/// 
/// // Find root of cos(x) - x = 0
/// let result = secant(|x| x.cos() - x, 0.0, 1.0, 1e-10, 100)?;
/// 
/// // Verify the root
/// let f = |x: f64| x.cos() - x;
/// assert!(f(result.root).abs() < 1e-10);
/// assert!(result.converged);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn secant<F>(f: F, mut x0: f64, mut x1: f64, tol: f64, max_iter: usize) -> Result<RootResult>
where
    F: Fn(f64) -> f64,
{
    if tol <= 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Tolerance must be positive".to_string()
        ));
    }
    
    if !x0.is_finite() || !x1.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Initial points must be finite".to_string()
        ));
    }
    
    let mut f0 = f(x0);
    let mut f1 = f(x1);
    
    if !f0.is_finite() || !f1.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Function evaluation returned non-finite value".to_string()
        ));
    }
    
    let mut iterations = 0;
    
    while iterations < max_iter {
        // Check convergence
        if f1.abs() < tol {
            return Ok(RootResult {
                root: x1,
                iterations,
                function_value: f1,
                converged: true,
            });
        }
        
        // Check for identical function values (would cause division by zero)
        if (f1 - f0).abs() < 1e-15 {
            return Err(NumericalError::NumericalInstability(
                "Function values are too close - secant method may not converge".to_string()
            ));
        }
        
        // Secant update: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        
        if !x2.is_finite() {
            return Err(NumericalError::NumericalInstability(
                "Secant method update produced non-finite value".to_string()
            ));
        }
        
        // Check for convergence in x
        if (x2 - x1).abs() < tol {
            let f2 = f(x2);
            return Ok(RootResult {
                root: x2,
                iterations: iterations + 1,
                function_value: f2,
                converged: true,
            });
        }
        
        // Update for next iteration
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x2);
        
        if !f1.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function evaluation at x = {} returned non-finite value", x1)
            ));
        }
        
        iterations += 1;
    }
    
    Ok(RootResult {
        root: x1,
        iterations,
        function_value: f1,
        converged: false,
    })
}

/// Brent's method for root finding
/// 
/// Combines the robustness of bisection with the speed of the secant method.
/// Uses inverse quadratic interpolation when possible, falls back to secant
/// method or bisection as needed. Guaranteed to converge.
/// 
/// # Arguments
/// * `f` - Function whose root to find
/// * `a` - Left endpoint of bracketing interval
/// * `b` - Right endpoint of bracketing interval
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
/// 
/// # Example
/// ```
/// use rustlab_numerical::roots::brent;
/// 
/// // Find root of x^3 + 4x^2 - 10 = 0 in [1, 2]
/// let result = brent(|x| x * x * x + 4.0 * x * x - 10.0, 1.0, 2.0, 1e-12, 100)?;
/// 
/// // Verify the root
/// let f = |x: f64| x * x * x + 4.0 * x * x - 10.0;
/// assert!(f(result.root).abs() < 1e-12);
/// assert!(result.converged);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn brent<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Result<RootResult>
where
    F: Fn(f64) -> f64,
{
    if tol <= 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Tolerance must be positive".to_string()
        ));
    }
    
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Interval endpoints must be finite".to_string()
        ));
    }
    
    let mut fa = f(a);
    let mut fb = f(b);
    
    if !fa.is_finite() || !fb.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Function evaluation returned non-finite value".to_string()
        ));
    }
    
    // Check if we already have a root at the endpoints
    if fa.abs() < tol {
        return Ok(RootResult {
            root: a,
            iterations: 0,
            function_value: fa,
            converged: true,
        });
    }
    
    if fb.abs() < tol {
        return Ok(RootResult {
            root: b,
            iterations: 0,
            function_value: fb,
            converged: true,
        });
    }
    
    // Ensure f(a) and f(b) have opposite signs
    if fa * fb > 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Function must have opposite signs at interval endpoints".to_string()
        ));
    }
    
    // Ensure |f(a)| >= |f(b)|
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    
    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = 0.0;
    
    let mut iterations = 0;
    
    while iterations < max_iter && fb.abs() > tol && (b - a).abs() > tol {
        iterations += 1;
        
        let mut s;
        
        if fa != fc && fb != fc {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }
        
        // Check if we should use bisection instead
        let delta = tol.abs();
        let condition1 = s < (3.0 * a + b) / 4.0 || s > b;
        let condition2 = mflag && (s - b).abs() >= (b - c).abs() / 2.0;
        let condition3 = !mflag && (s - b).abs() >= (c - d).abs() / 2.0;
        let condition4 = mflag && (b - c).abs() < delta;
        let condition5 = !mflag && (c - d).abs() < delta;
        
        if condition1 || condition2 || condition3 || condition4 || condition5 {
            // Use bisection
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }
        
        let fs = f(s);
        if !fs.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function evaluation at x = {} returned non-finite value", s)
            ));
        }
        
        // Update for next iteration
        d = c;
        c = b;
        fc = fb;
        
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        
        // Ensure |f(a)| >= |f(b)|
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
    
    Ok(RootResult {
        root: b,
        iterations,
        function_value: fb,
        converged: fb.abs() <= tol,
    })
}

/// Ridders' method for root finding
/// 
/// Uses exponential interpolation to find roots. Has superlinear convergence
/// and is more robust than Newton's method while being faster than bisection.
/// 
/// # Arguments
/// * `f` - Function whose root to find
/// * `a` - Left endpoint of bracketing interval
/// * `b` - Right endpoint of bracketing interval
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
/// 
/// # Example
/// ```
/// use rustlab_numerical::roots::ridders;
/// 
/// // Find root of e^x - 2 = 0 (should be ln(2) ≈ 0.693)
/// let result = ridders(|x| x.exp() - 2.0, 0.0, 1.0, 1e-12, 100)?;
/// assert!((result.root - 2.0_f64.ln()).abs() < 1e-12);
/// assert!(result.converged);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn ridders<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Result<RootResult>
where
    F: Fn(f64) -> f64,
{
    if tol <= 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Tolerance must be positive".to_string()
        ));
    }
    
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Interval endpoints must be finite".to_string()
        ));
    }
    
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    
    let mut fa = f(a);
    let mut fb = f(b);
    
    if !fa.is_finite() || !fb.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Function evaluation returned non-finite value".to_string()
        ));
    }
    
    // Check if we already have a root at the endpoints
    if fa.abs() < tol {
        return Ok(RootResult {
            root: a,
            iterations: 0,
            function_value: fa,
            converged: true,
        });
    }
    
    if fb.abs() < tol {
        return Ok(RootResult {
            root: b,
            iterations: 0,
            function_value: fb,
            converged: true,
        });
    }
    
    // Check that the function has opposite signs at the endpoints
    if fa * fb > 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Function must have opposite signs at interval endpoints".to_string()
        ));
    }
    
    let mut iterations = 0;
    
    while iterations < max_iter && (b - a).abs() > tol {
        iterations += 1;
        
        // Midpoint
        let c = (a + b) / 2.0;
        let fc = f(c);
        
        if !fc.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function evaluation at x = {} returned non-finite value", c)
            ));
        }
        
        if fc.abs() < tol {
            return Ok(RootResult {
                root: c,
                iterations,
                function_value: fc,
                converged: true,
            });
        }
        
        // Ridders' formula: exponential interpolation
        let discriminant = fc * fc - fa * fb;
        if discriminant < 0.0 {
            return Err(NumericalError::NumericalInstability(
                "Discriminant is negative in Ridders' method".to_string()
            ));
        }
        
        let sign = if fa < fb { 1.0 } else { -1.0 };
        let denominator = discriminant.sqrt();
        let x = c + sign * (c - a) * fc / denominator;
        
        let fx = f(x);
        if !fx.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function evaluation at x = {} returned non-finite value", x)
            ));
        }
        
        if fx.abs() < tol {
            return Ok(RootResult {
                root: x,
                iterations,
                function_value: fx,
                converged: true,
            });
        }
        
        // Update interval
        if fc * fx < 0.0 {
            a = c;
            fa = fc;
            b = x;
            fb = fx;
        } else if fa * fx < 0.0 {
            b = x;
            fb = fx;
        } else {
            a = x;
            fa = fx;
        }
    }
    
    let root = (a + b) / 2.0;
    Ok(RootResult {
        root,
        iterations,
        function_value: f(root),
        converged: (b - a).abs() <= tol,
    })
}

/// Illinois method (modified Regula Falsi) for root finding
/// 
/// A modification of the false position method that prevents slow convergence
/// by reducing the weight of the "stagnant" endpoint.
/// 
/// # Arguments
/// * `f` - Function whose root to find
/// * `a` - Left endpoint of bracketing interval
/// * `b` - Right endpoint of bracketing interval
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
/// 
/// # Example
/// ```
/// use rustlab_numerical::roots::illinois;
/// 
/// // Find root of x^3 - x - 1 = 0
/// let result = illinois(|x| x * x * x - x - 1.0, 1.0, 2.0, 1e-10, 100)?;
/// 
/// // Verify the root
/// let f = |x: f64| x * x * x - x - 1.0;
/// assert!(f(result.root).abs() < 1e-10);
/// assert!(result.converged);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn illinois<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Result<RootResult>
where
    F: Fn(f64) -> f64,
{
    if tol <= 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Tolerance must be positive".to_string()
        ));
    }
    
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Interval endpoints must be finite".to_string()
        ));
    }
    
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    
    let mut fa = f(a);
    let mut fb = f(b);
    
    if !fa.is_finite() || !fb.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Function evaluation returned non-finite value".to_string()
        ));
    }
    
    // Check if we already have a root at the endpoints
    if fa.abs() < tol {
        return Ok(RootResult {
            root: a,
            iterations: 0,
            function_value: fa,
            converged: true,
        });
    }
    
    if fb.abs() < tol {
        return Ok(RootResult {
            root: b,
            iterations: 0,
            function_value: fb,
            converged: true,
        });
    }
    
    // Check that the function has opposite signs at the endpoints
    if fa * fb > 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Function must have opposite signs at interval endpoints".to_string()
        ));
    }
    
    let mut side = 0; // Track which side was chosen in previous iteration
    let mut iterations = 0;
    
    while iterations < max_iter && (b - a).abs() > tol {
        iterations += 1;
        
        // Regula falsi (false position) formula
        let c = (fa * b - fb * a) / (fa - fb);
        let fc = f(c);
        
        if !fc.is_finite() {
            return Err(NumericalError::NumericalInstability(
                format!("Function evaluation at x = {} returned non-finite value", c)
            ));
        }
        
        if fc.abs() < tol {
            return Ok(RootResult {
                root: c,
                iterations,
                function_value: fc,
                converged: true,
            });
        }
        
        // Update interval with Illinois modification
        if fa * fc < 0.0 {
            b = c;
            fb = fc;
            if side == -1 {
                fa /= 2.0; // Reduce weight of stagnant side
            }
            side = -1;
        } else {
            a = c;
            fa = fc;
            if side == 1 {
                fb /= 2.0; // Reduce weight of stagnant side
            }
            side = 1;
        }
    }
    
    let root = (a + b) / 2.0;
    Ok(RootResult {
        root,
        iterations,
        function_value: f(root),
        converged: (b - a).abs() <= tol,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_bisection_simple() {
        // Find root of x^2 - 2 = 0 (√2 ≈ 1.414)
        let result = bisection(|x| x * x - 2.0, 1.0, 2.0, 1e-10, 100).unwrap();
        assert_relative_eq!(result.root, 2.0_f64.sqrt(), epsilon = 1e-10);
        assert!(result.converged);
        assert!(result.function_value.abs() < 1e-10);
    }
    
    #[test]
    fn test_newton_raphson_cubic() {
        // Find root of x^3 - 2x - 5 = 0
        let f = |x: f64| x * x * x - 2.0 * x - 5.0;
        let df = |x: f64| 3.0 * x * x - 2.0;
        let result = newton_raphson(f, df, 2.0, 1e-12, 100).unwrap();
        
        assert!(f(result.root).abs() < 1e-12);
        assert!(result.converged);
        assert!(result.iterations < 10); // Should converge quickly
    }
    
    #[test]
    fn test_secant_transcendental() {
        // Find root of cos(x) - x = 0
        let f = |x: f64| x.cos() - x;
        let result = secant(f, 0.0, 1.0, 1e-12, 100).unwrap();
        
        assert!(f(result.root).abs() < 1e-12);
        assert!(result.converged);
        assert_relative_eq!(result.root, 0.7390851332151607, epsilon = 1e-10);
    }
    
    #[test]
    fn test_brent_polynomial() {
        // Find root of x^3 + 4x^2 - 10 = 0
        let f = |x: f64| x * x * x + 4.0 * x * x - 10.0;
        let result = brent(f, 1.0, 2.0, 1e-12, 100).unwrap();
        
        assert!(f(result.root).abs() < 1e-12);
        assert!(result.converged);
    }
    
    #[test]
    fn test_ridders_polynomial() {
        // Find root of x^2 - 4 = 0 (should be 2.0)
        let f = |x: f64| x * x - 4.0;
        let result = ridders(f, 1.0, 3.0, 1e-10, 50).unwrap();
        
        // Check that we found a root (function value near zero)
        assert!(f(result.root).abs() < 1e-8);
        assert_relative_eq!(result.root, 2.0, epsilon = 1e-8);
    }
    
    #[test]
    fn test_illinois_cubic() {
        // Find root of x^3 - x - 1 = 0
        let f = |x: f64| x * x * x - x - 1.0;
        let result = illinois(f, 1.0, 2.0, 1e-10, 100).unwrap();
        
        assert!(f(result.root).abs() < 1e-10);
        assert!(result.converged);
    }
    
    #[test]
    fn test_no_bracketing() {
        // Function with same sign at both endpoints
        let f = |x: f64| x * x + 1.0; // Always positive
        
        assert!(bisection(f, -1.0, 1.0, 1e-6, 100).is_err());
        assert!(brent(f, -1.0, 1.0, 1e-6, 100).is_err());
        assert!(ridders(f, -1.0, 1.0, 1e-6, 100).is_err());
        assert!(illinois(f, -1.0, 1.0, 1e-6, 100).is_err());
    }
    
    #[test]
    fn test_zero_derivative() {
        // Function with zero derivative at the initial guess
        let f = |x: f64| x * x * x;
        let df = |_x: f64| 0.0; // Wrong derivative
        
        assert!(newton_raphson(f, df, 1.0, 1e-6, 100).is_err());
    }
    
    #[test]
    fn test_invalid_tolerance() {
        let f = |x: f64| x * x - 2.0;
        
        assert!(bisection(f, 1.0, 2.0, -1e-6, 100).is_err());
        assert!(newton_raphson(f, |x| 2.0 * x, 1.5, 0.0, 100).is_err());
        assert!(secant(f, 1.0, 2.0, -1e-6, 100).is_err());
    }
    
    #[test]
    fn test_root_at_endpoint() {
        let f = |x: f64| x - 1.0;
        
        let result = bisection(f, 1.0, 2.0, 1e-6, 100).unwrap();
        assert_relative_eq!(result.root, 1.0, epsilon = 1e-12);
        assert_eq!(result.iterations, 0);
        assert!(result.converged);
    }
    
    #[test]
    fn test_convergence_comparison() {
        // Compare convergence rates for f(x) = x^2 - 2
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        
        let bisect_result = bisection(f, 1.0, 2.0, 1e-10, 100).unwrap();
        let newton_result = newton_raphson(f, df, 1.5, 1e-10, 100).unwrap();
        let secant_result = secant(f, 1.0, 2.0, 1e-10, 100).unwrap();
        let brent_result = brent(f, 1.0, 2.0, 1e-10, 100).unwrap();
        
        // All should find the same root
        let expected = 2.0_f64.sqrt();
        assert_relative_eq!(bisect_result.root, expected, epsilon = 1e-10);
        assert_relative_eq!(newton_result.root, expected, epsilon = 1e-10);
        assert_relative_eq!(secant_result.root, expected, epsilon = 1e-10);
        assert_relative_eq!(brent_result.root, expected, epsilon = 1e-10);
        
        // Newton should converge fastest, bisection slowest
        assert!(newton_result.iterations < secant_result.iterations);
        assert!(secant_result.iterations < bisect_result.iterations);
        assert!(brent_result.iterations <= bisect_result.iterations);
    }
}