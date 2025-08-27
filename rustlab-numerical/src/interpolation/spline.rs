//! Spline interpolation methods
//! 
//! This module provides spline interpolation techniques:
//! - Cubic spline interpolation with various boundary conditions
//! - Monotonic cubic spline interpolation
//! - Natural and clamped boundary conditions

use rustlab_math::VectorF64;
use crate::{Result, NumericalError};
use super::traits::{Interpolator1D, ExtrapolationMode, DifferentiableInterpolator1D};
use super::utils::{check_dimensions, check_monotonic, find_interval};

/// Boundary conditions for cubic splines
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Natural spline: second derivative is zero at endpoints
    Natural,
    /// Clamped spline: specify first derivative at endpoints
    Clamped { start_derivative: f64, end_derivative: f64 },
    /// Not-a-knot: third derivative is continuous at second and second-to-last points
    NotAKnot,
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        BoundaryCondition::Natural
    }
}

/// Cubic spline interpolator
/// 
/// Constructs a piecewise cubic polynomial that is C² continuous.
/// Each piece is defined by four coefficients: a, b, c, d such that
/// S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)² + d_i*(x - x_i)³
#[derive(Debug, Clone)]
pub struct CubicSpline {
    x: VectorF64,
    y: VectorF64,
    // Coefficients for each spline segment
    a: Vec<f64>, // S_i(x_i) = y_i
    b: Vec<f64>, // First derivative coefficients
    c: Vec<f64>, // Second derivative coefficients / 2
    d: Vec<f64>, // Third derivative coefficients / 6
    #[allow(dead_code)] // Stored for potential future introspection/debugging
    boundary: BoundaryCondition,
    extrapolation: ExtrapolationMode,
}

impl CubicSpline {
    /// Create a new cubic spline interpolator
    /// 
    /// # Arguments
    /// * `x` - Strictly increasing x values
    /// * `y` - Corresponding y values
    /// * `boundary` - Boundary condition to use
    /// 
    /// # Example
    /// ```
    /// use rustlab_core::vec;
    /// use rustlab_numerical::interpolation::{CubicSpline, BoundaryCondition, Interpolator1D};
    /// 
    /// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
    /// let y = VectorF64::from_slice(&[0.0, 1.0, 4.0, 9.0]); // Quadratic-like data
    /// let spline = CubicSpline::new(x, y, BoundaryCondition::Natural)?;
    /// 
    /// let result = spline.eval(1.5)?;
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    pub fn new(x: VectorF64, y: VectorF64, boundary: BoundaryCondition) -> Result<Self> {
        check_dimensions(&x, &y)?;
        check_monotonic(&x, "cubic spline interpolation")?;
        
        let n = x.len();
        if n < 3 {
            return Err(NumericalError::InsufficientData { 
                got: n, 
                need: 3 
            });
        }
        
        // Compute spline coefficients
        let coefficients = Self::compute_coefficients(&x, &y, boundary)?;
        
        Ok(Self {
            x,
            y,
            a: coefficients.0,
            b: coefficients.1,
            c: coefficients.2,
            d: coefficients.3,
            boundary,
            extrapolation: ExtrapolationMode::default(),
        })
    }
    
    /// Set the extrapolation mode
    pub fn with_extrapolation(mut self, mode: ExtrapolationMode) -> Self {
        self.extrapolation = mode;
        self
    }
    
    /// Compute spline coefficients by solving the tridiagonal system
    fn compute_coefficients(
        x: &VectorF64, 
        y: &VectorF64, 
        boundary: BoundaryCondition
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let n = x.len();
        let m = n - 1; // number of intervals
        
        // Step 1: Compute h_i = x_{i+1} - x_i
        let mut h = Vec::with_capacity(m);
        for i in 0..m {
            h.push(x.get(i + 1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i + 1)))? - x.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i)))?);
        }
        
        // Step 2: Set up tridiagonal system for second derivatives
        // A * M = B, where M is the vector of second derivatives at knots
        let mut alpha = vec![0.0; n];
        let mut beta = vec![0.0; n];
        let mut gamma = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        
        // Interior points (i = 1, ..., n-2)
        for i in 1..n-1 {
            alpha[i] = h[i-1];
            beta[i] = 2.0 * (h[i-1] + h[i]);
            gamma[i] = h[i];
            rhs[i] = 6.0 * ((y.get(i+1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", i+1)))? - y.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", i)))?) / h[i] - (y.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", i)))? - y.get(i-1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", i-1)))?) / h[i-1]);
        }
        
        // Apply boundary conditions
        match boundary {
            BoundaryCondition::Natural => {
                // Natural: M_0 = M_{n-1} = 0
                beta[0] = 1.0;
                gamma[0] = 0.0;
                rhs[0] = 0.0;
                
                alpha[n-1] = 0.0;
                beta[n-1] = 1.0;
                rhs[n-1] = 0.0;
            },
            BoundaryCondition::Clamped { start_derivative, end_derivative } => {
                // Clamped: specify first derivatives at endpoints
                beta[0] = 2.0 * h[0];
                gamma[0] = h[0];
                rhs[0] = 6.0 * ((y.get(1).ok_or_else(|| NumericalError::InvalidParameter("Index 1 out of bounds in y array".to_string()))? - y.get(0).ok_or_else(|| NumericalError::InvalidParameter("Index 0 out of bounds in y array".to_string()))?) / h[0] - start_derivative);
                
                alpha[n-1] = h[m-1];
                beta[n-1] = 2.0 * h[m-1];
                rhs[n-1] = 6.0 * (end_derivative - (y.get(n-1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", n-1)))? - y.get(n-2).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", n-2)))?) / h[m-1]);
            },
            BoundaryCondition::NotAKnot => {
                // Not-a-knot: third derivative continuous at x_1 and x_{n-2}
                if n < 4 {
                    return Err(NumericalError::InvalidParameter(
                        "Not-a-knot boundary condition requires at least 4 points".to_string()
                    ));
                }
                
                // At x_1: h_0 * M_0 - (h_0 + h_1) * M_1 + h_1 * M_2 = 0
                beta[0] = -(h[0] + h[1]);
                gamma[0] = h[1];
                alpha[0] = h[0];
                rhs[0] = 0.0;
                
                // At x_{n-2}: h_{n-3} * M_{n-3} - (h_{n-3} + h_{n-2}) * M_{n-2} + h_{n-2} * M_{n-1} = 0
                alpha[n-1] = h[m-2];
                beta[n-1] = -(h[m-2] + h[m-1]);
                gamma[n-1] = h[m-1];
                rhs[n-1] = 0.0;
            },
        }
        
        // Step 3: Solve tridiagonal system using Thomas algorithm
        let second_derivatives = Self::solve_tridiagonal(&alpha, &beta, &gamma, &rhs)?;
        
        // Step 4: Compute spline coefficients
        let mut a = Vec::with_capacity(m);
        let mut b = Vec::with_capacity(m);
        let mut c = Vec::with_capacity(m);
        let mut d = Vec::with_capacity(m);
        
        for i in 0..m {
            let yi = y.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", i)))?;
            let yi1 = y.get(i + 1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", i + 1)))?;
            let mi = second_derivatives[i];
            let mi1 = second_derivatives[i + 1];
            let hi = h[i];
            
            a.push(yi);
            b.push((yi1 - yi) / hi - hi * (2.0 * mi + mi1) / 6.0);
            c.push(mi / 2.0);
            d.push((mi1 - mi) / (6.0 * hi));
        }
        
        Ok((a, b, c, d))
    }
    
    /// Solve tridiagonal system using Thomas algorithm
    fn solve_tridiagonal(
        alpha: &[f64], 
        beta: &[f64], 
        gamma: &[f64], 
        rhs: &[f64]
    ) -> Result<Vec<f64>> {
        let n = beta.len();
        let mut c_prime = vec![0.0; n];
        let mut d_prime = vec![0.0; n];
        let mut x = vec![0.0; n];
        
        // Forward sweep
        c_prime[0] = gamma[0] / beta[0];
        d_prime[0] = rhs[0] / beta[0];
        
        for i in 1..n {
            let denominator = beta[i] - alpha[i] * c_prime[i-1];
            if denominator.abs() < 1e-15 {
                return Err(NumericalError::NumericalInstability(
                    "Tridiagonal system is singular".to_string()
                ));
            }
            
            if i < n - 1 {
                c_prime[i] = gamma[i] / denominator;
            }
            d_prime[i] = (rhs[i] - alpha[i] * d_prime[i-1]) / denominator;
        }
        
        // Back substitution
        x[n-1] = d_prime[n-1];
        for i in (0..n-1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i+1];
        }
        
        Ok(x)
    }
    
    /// Evaluate spline at a point within the domain
    fn eval_spline(&self, x: f64) -> Result<f64> {
        let i = find_interval(&self.x, x)
            .ok_or_else(|| NumericalError::OutOfBounds {
                value: x,
                min: self.x.get(0).unwrap_or(0.0),
                max: self.x.get(self.x.len()-1).unwrap_or(0.0),
            })?;
        
        let xi = self.x.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i)))?;
        let dx = x - xi;
        
        // S_i(x) = a_i + b_i*dx + c_i*dx² + d_i*dx³
        Ok(self.a[i] + self.b[i] * dx + self.c[i] * dx * dx + self.d[i] * dx * dx * dx)
    }
    
    /// Evaluate first derivative of spline
    fn eval_derivative_spline(&self, x: f64) -> Result<f64> {
        let i = find_interval(&self.x, x)
            .ok_or_else(|| NumericalError::OutOfBounds {
                value: x,
                min: self.x.get(0).unwrap_or(0.0),
                max: self.x.get(self.x.len()-1).unwrap_or(0.0),
            })?;
        
        let xi = self.x.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i)))?;
        let dx = x - xi;
        
        // S'_i(x) = b_i + 2*c_i*dx + 3*d_i*dx²
        Ok(self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx * dx)
    }
    
    /// Evaluate second derivative of spline
    fn eval_second_derivative_spline(&self, x: f64) -> Result<f64> {
        let i = find_interval(&self.x, x)
            .ok_or_else(|| NumericalError::OutOfBounds {
                value: x,
                min: self.x.get(0).unwrap_or(0.0),
                max: self.x.get(self.x.len()-1).unwrap_or(0.0),
            })?;
        
        let xi = self.x.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i)))?;
        let dx = x - xi;
        
        // S''_i(x) = 2*c_i + 6*d_i*dx
        Ok(2.0 * self.c[i] + 6.0 * self.d[i] * dx)
    }
    
    /// Handle extrapolation based on the current mode
    fn extrapolate(&self, x: f64) -> Result<f64> {
        let n = self.x.len();
        let x_min = self.x.get(0).ok_or_else(|| NumericalError::InvalidParameter("Index 0 out of bounds in x array".to_string()))?;
        let x_max = self.x.get(n-1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", n-1)))?;
        
        match self.extrapolation {
            ExtrapolationMode::Error => {
                Err(NumericalError::OutOfBounds {
                    value: x,
                    min: x_min,
                    max: x_max,
                })
            },
            ExtrapolationMode::Constant => {
                if x < x_min {
                    Ok(self.y.get(0).ok_or_else(|| NumericalError::InvalidParameter("Index 0 out of bounds in y array".to_string()))?)
                } else {
                    Ok(self.y.get(n-1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", n-1)))?)
                }
            },
            ExtrapolationMode::Linear => {
                if x < x_min {
                    // Linear extrapolation using first segment's derivative at x_0
                    let slope = self.b[0];
                    Ok(self.y.get(0).ok_or_else(|| NumericalError::InvalidParameter("Index 0 out of bounds in y array".to_string()))? + slope * (x - x_min))
                } else {
                    // Linear extrapolation using last segment's derivative at x_{n-1}
                    let i = n - 2; // last segment index
                    let xi = self.x.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i)))?;
                    let dx = x_max - xi;
                    let slope = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx * dx;
                    Ok(self.y.get(n-1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", n-1)))? + slope * (x - x_max))
                }
            },
            ExtrapolationMode::NaN => Ok(f64::NAN),
        }
    }
}

impl Interpolator1D for CubicSpline {
    fn eval(&self, x: f64) -> Result<f64> {
        if !self.in_domain(x) {
            return self.extrapolate(x);
        }
        
        self.eval_spline(x)
    }
    
    fn domain(&self) -> (f64, f64) {
        let n = self.x.len();
        (
            self.x.get(0).unwrap_or(0.0),
            self.x.get(n-1).unwrap_or(0.0)
        )
    }
}

impl DifferentiableInterpolator1D for CubicSpline {
    fn eval_derivative(&self, x: f64) -> Result<f64> {
        if !self.in_domain(x) {
            return match self.extrapolation {
                ExtrapolationMode::Error => {
                    let (min, max) = self.domain();
                    Err(NumericalError::OutOfBounds { value: x, min, max })
                },
                ExtrapolationMode::Linear => {
                    let n = self.x.len();
                    if x < self.x.get(0).ok_or_else(|| NumericalError::InvalidParameter("Index 0 out of bounds in x array".to_string()))? {
                        Ok(self.b[0])
                    } else {
                        let i = n - 2;
                        let xi = self.x.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i)))?;
                        let x_max = self.x.get(n-1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", n-1)))?;
                        let dx = x_max - xi;
                        Ok(self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx * dx)
                    }
                },
                _ => Ok(0.0), // Constant extrapolation has zero derivative
            };
        }
        
        self.eval_derivative_spline(x)
    }
    
    fn eval_second_derivative(&self, x: f64) -> Result<f64> {
        if !self.in_domain(x) {
            return match self.extrapolation {
                ExtrapolationMode::Error => {
                    let (min, max) = self.domain();
                    Err(NumericalError::OutOfBounds { value: x, min, max })
                },
                _ => Ok(0.0), // Linear and constant extrapolation have zero second derivative
            };
        }
        
        self.eval_second_derivative_spline(x)
    }
}

/// Convenience function for cubic spline interpolation with natural boundary conditions
/// 
/// # Example
/// ```
/// use rustlab_core::vec;
/// use rustlab_numerical::interpolation::interp1d_cubic_spline;
/// 
/// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
/// let y = VectorF64::from_slice(&[0.0, 1.0, 4.0, 9.0]);
/// let xi = VectorF64::from_slice(&[0.5, 1.5, 2.5]);
/// 
/// let yi = interp1d_cubic_spline(&x, &y, &xi)?;
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn interp1d_cubic_spline(x: &VectorF64, y: &VectorF64, xi: &VectorF64) -> Result<VectorF64> {
    let spline = CubicSpline::new(x.clone(), y.clone(), BoundaryCondition::Natural)?;
    spline.eval_vec(xi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::{vec64, VectorF64};
    use approx::assert_relative_eq;
    
    #[test]
    fn test_natural_spline_linear() {
        // Linear function should be reproduced exactly
        let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
        let y = VectorF64::from_slice(&[1.0, 3.0, 5.0, 7.0]); // y = 2x + 1
        let spline = CubicSpline::new(x, y, BoundaryCondition::Natural).unwrap();
        
        assert_relative_eq!(spline.eval(0.5).unwrap(), 2.0, epsilon = 1e-12);
        assert_relative_eq!(spline.eval(1.5).unwrap(), 4.0, epsilon = 1e-12);
        assert_relative_eq!(spline.eval(2.5).unwrap(), 6.0, epsilon = 1e-12);
    }
    
    #[test]
    fn test_natural_spline_quadratic() {
        // Quadratic function y = x²
        let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0, 4.0, 9.0]);
        let spline = CubicSpline::new(x, y, BoundaryCondition::Natural).unwrap();
        
        // Should be reasonably close to x² at intermediate points
        // (natural boundary conditions prevent exact reproduction)
        assert!((spline.eval(0.5).unwrap() - 0.25).abs() < 0.5);
        assert!((spline.eval(1.5).unwrap() - 2.25).abs() < 0.5);
        assert!((spline.eval(2.5).unwrap() - 6.25).abs() < 0.5);
    }
    
    #[test]
    fn test_clamped_spline() {
        // Test clamped boundary conditions
        let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0, 4.0]);
        let spline = CubicSpline::new(
            x, y, 
            BoundaryCondition::Clamped { start_derivative: 0.0, end_derivative: 4.0 }
        ).unwrap();
        
        // Check that derivatives at endpoints match specified values
        assert_relative_eq!(spline.eval_derivative(0.0).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(spline.eval_derivative(2.0).unwrap(), 4.0, epsilon = 1e-12);
    }
    
    #[test]
    fn test_spline_continuity() {
        // Test that spline is continuous at knots
        let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0, 3.0, 2.0]);
        let spline = CubicSpline::new(x.clone(), y.clone(), BoundaryCondition::Natural).unwrap();
        
        // Values at knots should match exactly
        for i in 0..x.len() {
            let xi = x.get(i).unwrap();
            let yi = y.get(i).unwrap();
            assert_relative_eq!(spline.eval(xi).unwrap(), yi, epsilon = 1e-12);
        }
    }
    
    #[test]
    fn test_spline_derivatives() {
        let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0, 4.0, 9.0]);
        let spline = CubicSpline::new(x, y, BoundaryCondition::Natural).unwrap();
        
        // Test that derivative evaluation works
        let deriv = spline.eval_derivative(1.5).unwrap();
        let second_deriv = spline.eval_second_derivative(1.5).unwrap();
        
        // Just check that they're finite numbers
        assert!(deriv.is_finite());
        assert!(second_deriv.is_finite());
    }
    
    #[test]
    fn test_insufficient_points() {
        let x = VectorF64::from_slice(&[0.0, 1.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0]);
        assert!(CubicSpline::new(x, y, BoundaryCondition::Natural).is_err());
    }
    
    #[test]
    fn test_not_a_knot_insufficient_points() {
        let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0, 4.0]);
        assert!(CubicSpline::new(x, y, BoundaryCondition::NotAKnot).is_err());
    }
    
    #[test]
    fn test_extrapolation_modes() {
        let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0, 4.0]);
        
        // Test with Error mode (default)
        let spline = CubicSpline::new(x.clone(), y.clone(), BoundaryCondition::Natural).unwrap();
        assert!(spline.eval(-1.0).is_err());
        assert!(spline.eval(3.0).is_err());
        
        // Test with Constant mode
        let spline = CubicSpline::new(x.clone(), y.clone(), BoundaryCondition::Natural)
            .unwrap()
            .with_extrapolation(ExtrapolationMode::Constant);
        assert_relative_eq!(spline.eval(-1.0).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(spline.eval(3.0).unwrap(), 4.0, epsilon = 1e-12);
        
        // Test with Linear mode
        let spline = CubicSpline::new(x.clone(), y.clone(), BoundaryCondition::Natural)
            .unwrap()
            .with_extrapolation(ExtrapolationMode::Linear);
        // Linear extrapolation should give finite results
        assert!(spline.eval(-1.0).unwrap().is_finite());
        assert!(spline.eval(3.0).unwrap().is_finite());
        
        // Test with NaN mode
        let spline = CubicSpline::new(x.clone(), y.clone(), BoundaryCondition::Natural)
            .unwrap()
            .with_extrapolation(ExtrapolationMode::NaN);
        assert!(spline.eval(-1.0).unwrap().is_nan());
        assert!(spline.eval(3.0).unwrap().is_nan());
    }
}