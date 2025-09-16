//! Finite difference methods for numerical differentiation
//! 
//! This module provides various finite difference approximations:
//! - Forward differences (1st to 4th order accuracy)
//! - Backward differences (1st to 4th order accuracy)
//! - Central differences (2nd to 6th order accuracy)
//! - Richardson extrapolation
//! - Complex-step differentiation

use crate::{Result, NumericalError};

/// Forward difference approximation of the first derivative
/// 
/// Computes f'(x) using forward differences with various orders of accuracy.
/// Uses the formula: f'(x) ≈ [a₁f(x+h) + a₂f(x+2h) + ... + aₙf(x+nh) + a₀f(x)] / h
/// 
/// # Arguments
/// * `f` - Function to differentiate
/// * `x` - Point at which to evaluate the derivative
/// * `h` - Step size
/// * `order` - Order of accuracy (1, 2, 3, or 4)
/// 
/// # Example
/// ```
/// use rustlab_numerical::differentiation::forward_diff;
/// 
/// // Differentiate x^2 at x = 2 (exact derivative is 4)
/// let result = forward_diff(|x| x * x, 2.0, 0.001, 2)?;
/// assert!((result - 4.0).abs() < 1e-6);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn forward_diff<F>(f: F, x: f64, h: f64, order: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if h == 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Step size cannot be zero".to_string()
        ));
    }
    
    if !h.is_finite() || !x.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Step size and evaluation point must be finite".to_string()
        ));
    }
    
    let result = match order {
        1 => {
            // First order: f'(x) ≈ (f(x+h) - f(x)) / h
            let f_x = f(x);
            let f_x_h = f(x + h);
            
            if !f_x.is_finite() || !f_x_h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (f_x_h - f_x) / h
        },
        2 => {
            // Second order: f'(x) ≈ (-3f(x) + 4f(x+h) - f(x+2h)) / (2h)
            let f_x = f(x);
            let f_x_h = f(x + h);
            let f_x_2h = f(x + 2.0 * h);
            
            if !f_x.is_finite() || !f_x_h.is_finite() || !f_x_2h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (-3.0 * f_x + 4.0 * f_x_h - f_x_2h) / (2.0 * h)
        },
        3 => {
            // Third order: f'(x) ≈ (-11f(x) + 18f(x+h) - 9f(x+2h) + 2f(x+3h)) / (6h)
            let f_x = f(x);
            let f_x_h = f(x + h);
            let f_x_2h = f(x + 2.0 * h);
            let f_x_3h = f(x + 3.0 * h);
            
            if !f_x.is_finite() || !f_x_h.is_finite() || !f_x_2h.is_finite() || !f_x_3h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (-11.0 * f_x + 18.0 * f_x_h - 9.0 * f_x_2h + 2.0 * f_x_3h) / (6.0 * h)
        },
        4 => {
            // Fourth order: f'(x) ≈ (-25f(x) + 48f(x+h) - 36f(x+2h) + 16f(x+3h) - 3f(x+4h)) / (12h)
            let f_x = f(x);
            let f_x_h = f(x + h);
            let f_x_2h = f(x + 2.0 * h);
            let f_x_3h = f(x + 3.0 * h);
            let f_x_4h = f(x + 4.0 * h);
            
            if !f_x.is_finite() || !f_x_h.is_finite() || !f_x_2h.is_finite() || 
               !f_x_3h.is_finite() || !f_x_4h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (-25.0 * f_x + 48.0 * f_x_h - 36.0 * f_x_2h + 16.0 * f_x_3h - 3.0 * f_x_4h) / (12.0 * h)
        },
        _ => {
            return Err(NumericalError::InvalidParameter(
                "Order must be 1, 2, 3, or 4".to_string()
            ));
        }
    };
    
    if !result.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Derivative computation resulted in non-finite value".to_string()
        ));
    }
    
    Ok(result)
}

/// Backward difference approximation of the first derivative
/// 
/// Computes f'(x) using backward differences with various orders of accuracy.
/// Uses the formula: f'(x) ≈ [a₁f(x-h) + a₂f(x-2h) + ... + aₙf(x-nh) + a₀f(x)] / h
/// 
/// # Arguments
/// * `f` - Function to differentiate
/// * `x` - Point at which to evaluate the derivative
/// * `h` - Step size
/// * `order` - Order of accuracy (1, 2, 3, or 4)
/// 
/// # Example
/// ```
/// use rustlab_numerical::differentiation::backward_diff;
/// 
/// // Differentiate sin(x) at x = π/2 (exact derivative is 0)
/// let result = backward_diff(|x| x.sin(), std::f64::consts::PI / 2.0, 0.001, 3)?;
/// assert!(result.abs() < 1e-6);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn backward_diff<F>(f: F, x: f64, h: f64, order: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if h == 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Step size cannot be zero".to_string()
        ));
    }
    
    if !h.is_finite() || !x.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Step size and evaluation point must be finite".to_string()
        ));
    }
    
    let result = match order {
        1 => {
            // First order: f'(x) ≈ (f(x) - f(x-h)) / h
            let f_x = f(x);
            let f_x_h = f(x - h);
            
            if !f_x.is_finite() || !f_x_h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (f_x - f_x_h) / h
        },
        2 => {
            // Second order: f'(x) ≈ (3f(x) - 4f(x-h) + f(x-2h)) / (2h)
            let f_x = f(x);
            let f_x_h = f(x - h);
            let f_x_2h = f(x - 2.0 * h);
            
            if !f_x.is_finite() || !f_x_h.is_finite() || !f_x_2h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (3.0 * f_x - 4.0 * f_x_h + f_x_2h) / (2.0 * h)
        },
        3 => {
            // Third order: f'(x) ≈ (11f(x) - 18f(x-h) + 9f(x-2h) - 2f(x-3h)) / (6h)
            let f_x = f(x);
            let f_x_h = f(x - h);
            let f_x_2h = f(x - 2.0 * h);
            let f_x_3h = f(x - 3.0 * h);
            
            if !f_x.is_finite() || !f_x_h.is_finite() || !f_x_2h.is_finite() || !f_x_3h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (11.0 * f_x - 18.0 * f_x_h + 9.0 * f_x_2h - 2.0 * f_x_3h) / (6.0 * h)
        },
        4 => {
            // Fourth order: f'(x) ≈ (25f(x) - 48f(x-h) + 36f(x-2h) - 16f(x-3h) + 3f(x-4h)) / (12h)
            let f_x = f(x);
            let f_x_h = f(x - h);
            let f_x_2h = f(x - 2.0 * h);
            let f_x_3h = f(x - 3.0 * h);
            let f_x_4h = f(x - 4.0 * h);
            
            if !f_x.is_finite() || !f_x_h.is_finite() || !f_x_2h.is_finite() || 
               !f_x_3h.is_finite() || !f_x_4h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (25.0 * f_x - 48.0 * f_x_h + 36.0 * f_x_2h - 16.0 * f_x_3h + 3.0 * f_x_4h) / (12.0 * h)
        },
        _ => {
            return Err(NumericalError::InvalidParameter(
                "Order must be 1, 2, 3, or 4".to_string()
            ));
        }
    };
    
    if !result.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Derivative computation resulted in non-finite value".to_string()
        ));
    }
    
    Ok(result)
}

/// Central difference approximation of the first derivative
/// 
/// Computes f'(x) using central differences with various orders of accuracy.
/// Central differences are generally more accurate than forward/backward differences.
/// 
/// # Arguments
/// * `f` - Function to differentiate
/// * `x` - Point at which to evaluate the derivative
/// * `h` - Step size
/// * `order` - Order of accuracy (2, 4, or 6)
/// 
/// # Example
/// ```
/// use rustlab_numerical::differentiation::central_diff;
/// 
/// // Differentiate x^3 at x = 1 (exact derivative is 3)
/// let result = central_diff(|x| x * x * x, 1.0, 0.001, 4)?;
/// assert!((result - 3.0).abs() < 1e-8);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn central_diff<F>(f: F, x: f64, h: f64, order: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    if h == 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Step size cannot be zero".to_string()
        ));
    }
    
    if !h.is_finite() || !x.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Step size and evaluation point must be finite".to_string()
        ));
    }
    
    let result = match order {
        2 => {
            // Second order: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
            let f_x_h = f(x + h);
            let f_x_neg_h = f(x - h);
            
            if !f_x_h.is_finite() || !f_x_neg_h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (f_x_h - f_x_neg_h) / (2.0 * h)
        },
        4 => {
            // Fourth order: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
            let f_x_2h = f(x + 2.0 * h);
            let f_x_h = f(x + h);
            let f_x_neg_h = f(x - h);
            let f_x_neg_2h = f(x - 2.0 * h);
            
            if !f_x_2h.is_finite() || !f_x_h.is_finite() || 
               !f_x_neg_h.is_finite() || !f_x_neg_2h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (-f_x_2h + 8.0 * f_x_h - 8.0 * f_x_neg_h + f_x_neg_2h) / (12.0 * h)
        },
        6 => {
            // Sixth order: f'(x) ≈ (-f(x-3h) + 9f(x-2h) - 45f(x-h) + 45f(x+h) - 9f(x+2h) + f(x+3h)) / (60h)
            let f_x_3h = f(x + 3.0 * h);
            let f_x_2h = f(x + 2.0 * h);
            let f_x_h = f(x + h);
            let f_x_neg_h = f(x - h);
            let f_x_neg_2h = f(x - 2.0 * h);
            let f_x_neg_3h = f(x - 3.0 * h);
            
            if !f_x_3h.is_finite() || !f_x_2h.is_finite() || !f_x_h.is_finite() ||
               !f_x_neg_h.is_finite() || !f_x_neg_2h.is_finite() || !f_x_neg_3h.is_finite() {
                return Err(NumericalError::NumericalInstability(
                    "Function evaluation returned non-finite value".to_string()
                ));
            }
            
            (-f_x_neg_3h + 9.0 * f_x_neg_2h - 45.0 * f_x_neg_h + 45.0 * f_x_h - 9.0 * f_x_2h + f_x_3h) / (60.0 * h)
        },
        _ => {
            return Err(NumericalError::InvalidParameter(
                "Order must be 2, 4, or 6 for central differences".to_string()
            ));
        }
    };
    
    if !result.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Derivative computation resulted in non-finite value".to_string()
        ));
    }
    
    Ok(result)
}

/// Richardson extrapolation for improved accuracy
/// 
/// Uses Richardson extrapolation to improve the accuracy of finite difference approximations.
/// Computes derivatives at step sizes h and h/2, then extrapolates to get a higher-order estimate.
/// 
/// # Arguments
/// * `f` - Function to differentiate
/// * `x` - Point at which to evaluate the derivative
/// * `h` - Initial step size
/// * `method` - Base differentiation method to use
/// * `order` - Order of the base method
/// 
/// # Example
/// ```
/// use rustlab_numerical::differentiation::{richardson_extrapolation, central_diff};
/// 
/// // Differentiate e^x at x = 0 (exact derivative is 1)
/// let result = richardson_extrapolation(
///     |x| x.exp(), 0.0, 0.1, 
///     |f, x, h, ord| central_diff(f, x, h, ord), 2
/// )?;
/// assert!((result - 1.0).abs() < 1e-12);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn richardson_extrapolation<F, M>(
    f: F, 
    x: f64, 
    h: f64, 
    method: M, 
    order: usize
) -> Result<f64>
where
    F: Fn(f64) -> f64 + Copy,
    M: Fn(F, f64, f64, usize) -> Result<f64>,
{
    if h == 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Step size cannot be zero".to_string()
        ));
    }
    
    // Compute derivative estimates with step sizes h and h/2
    let d_h = method(f, x, h, order)?;
    let d_h2 = method(f, x, h / 2.0, order)?;
    
    // Richardson extrapolation formula: D = (2^p * D(h/2) - D(h)) / (2^p - 1)
    // where p is the order of the method
    let p = order as f64;
    let factor = 2_f64.powf(p);
    let result = (factor * d_h2 - d_h) / (factor - 1.0);
    
    if !result.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Richardson extrapolation resulted in non-finite value".to_string()
        ));
    }
    
    Ok(result)
}

/// Complex-step differentiation for very high accuracy
/// 
/// Uses complex arithmetic to compute derivatives with machine precision accuracy.
/// This method is not affected by subtractive cancellation errors.
/// 
/// # Arguments
/// * `f` - Function to differentiate (must accept complex input)
/// * `x` - Point at which to evaluate the derivative
/// * `h` - Step size (can be very small, e.g., 1e-200)
/// 
/// # Example
/// ```
/// use rustlab_numerical::differentiation::complex_step_diff;
/// use num_complex::Complex64;
/// 
/// // Differentiate x^4 at x = 2 (exact derivative is 32)
/// let f = |z: Complex64| z * z * z * z;
/// let result = complex_step_diff(f, 2.0, 1e-200)?;
/// assert!((result - 32.0).abs() < 1e-14);
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn complex_step_diff<F>(f: F, x: f64, h: f64) -> Result<f64>
where
    F: Fn(num_complex::Complex64) -> num_complex::Complex64,
{
    use num_complex::Complex64;
    
    if h == 0.0 {
        return Err(NumericalError::InvalidParameter(
            "Step size cannot be zero".to_string()
        ));
    }
    
    if !h.is_finite() || !x.is_finite() {
        return Err(NumericalError::InvalidParameter(
            "Step size and evaluation point must be finite".to_string()
        ));
    }
    
    // Evaluate f(x + ih) where i is the imaginary unit
    let z = Complex64::new(x, h);
    let f_z = f(z);
    
    if !f_z.im.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Function evaluation returned non-finite imaginary part".to_string()
        ));
    }
    
    // The derivative is Im(f(x + ih)) / h
    let result = f_z.im / h;
    
    if !result.is_finite() {
        return Err(NumericalError::NumericalInstability(
            "Complex-step differentiation resulted in non-finite value".to_string()
        ));
    }
    
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;
    
    #[test]
    fn test_forward_diff_polynomial() {
        // f(x) = x^2, f'(x) = 2x
        let f = |x: f64| x * x;
        
        // Test at x = 3 (exact derivative is 6)
        let result1 = forward_diff(f, 3.0, 0.001, 1).unwrap();
        assert!((result1 - 6.0).abs() < 0.01); // First order is less accurate
        
        let result2 = forward_diff(f, 3.0, 0.001, 2).unwrap();
        assert!((result2 - 6.0).abs() < 1e-6); // Second order much better
        
        let result4 = forward_diff(f, 3.0, 0.001, 4).unwrap();
        assert!((result4 - 6.0).abs() < 1e-9); // Fourth order very accurate
    }
    
    #[test]
    fn test_backward_diff_trigonometric() {
        // f(x) = sin(x), f'(x) = cos(x)
        let f = |x: f64| x.sin();
        let x = std::f64::consts::PI / 4.0; // π/4
        let exact = x.cos(); // cos(π/4) = √2/2
        
        let result = backward_diff(f, x, 0.0001, 3).unwrap();
        assert_relative_eq!(result, exact, epsilon = 1e-8);
    }
    
    #[test]
    fn test_central_diff_exponential() {
        // f(x) = e^x, f'(x) = e^x
        let f = |x: f64| x.exp();
        let x = 1.0_f64;
        let exact = x.exp(); // e^1 = e
        
        let result2 = central_diff(f, x, 0.001, 2).unwrap();
        assert_relative_eq!(result2, exact, epsilon = 1e-6);
        
        let result4 = central_diff(f, x, 0.001, 4).unwrap();
        assert_relative_eq!(result4, exact, epsilon = 1e-10);
        
        let result6 = central_diff(f, x, 0.001, 6).unwrap();
        assert_relative_eq!(result6, exact, epsilon = 1e-12);
    }
    
    #[test]
    fn test_richardson_extrapolation() {
        // f(x) = x^3, f'(x) = 3x^2
        let f = |x: f64| x * x * x;
        let x = 2.0;
        let exact = 3.0 * x * x; // 3 * 4 = 12
        
        let result = richardson_extrapolation(
            f, x, 0.1, 
            |f, x, h, ord| central_diff(f, x, h, ord), 2
        ).unwrap();
        
        assert_relative_eq!(result, exact, epsilon = 1e-12);
    }
    
    #[test]
    fn test_complex_step_diff() {
        // f(x) = x^4, f'(x) = 4x^3
        let f = |z: Complex64| z * z * z * z;
        let x = 2.0;
        let exact = 4.0 * x * x * x; // 4 * 8 = 32
        
        let result = complex_step_diff(f, x, 1e-200).unwrap();
        assert_relative_eq!(result, exact, epsilon = 1e-14);
    }
    
    #[test]
    fn test_complex_step_trigonometric() {
        // f(x) = sin(x), f'(x) = cos(x)
        let f = |z: Complex64| z.sin();
        let x = std::f64::consts::PI / 3.0; // π/3
        let exact = x.cos(); // cos(π/3) = 1/2
        
        let result = complex_step_diff(f, x, 1e-100).unwrap();
        assert_relative_eq!(result, exact, epsilon = 1e-14);
    }
    
    #[test]
    fn test_zero_step_size() {
        let f = |x: f64| x * x;
        
        assert!(forward_diff(f, 1.0, 0.0, 1).is_err());
        assert!(backward_diff(f, 1.0, 0.0, 1).is_err());
        assert!(central_diff(f, 1.0, 0.0, 2).is_err());
        assert!(complex_step_diff(|z: Complex64| z * z, 1.0, 0.0).is_err());
    }
    
    #[test]
    fn test_invalid_orders() {
        let f = |x: f64| x * x;
        
        assert!(forward_diff(f, 1.0, 0.1, 5).is_err());
        assert!(backward_diff(f, 1.0, 0.1, 0).is_err());
        assert!(central_diff(f, 1.0, 0.1, 3).is_err()); // Only even orders
        assert!(central_diff(f, 1.0, 0.1, 8).is_err()); // Too high
    }
    
    #[test]
    fn test_infinite_inputs() {
        let f = |x: f64| x * x;
        
        assert!(forward_diff(f, f64::INFINITY, 0.1, 1).is_err());
        assert!(backward_diff(f, 1.0, f64::NEG_INFINITY, 1).is_err());
        assert!(central_diff(f, f64::NAN, 0.1, 2).is_err());
    }
}