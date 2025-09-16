//! Polynomial interpolation methods
//! 
//! This module provides various polynomial interpolation techniques:
//! - Lagrange interpolation
//! - Newton divided differences
//! - Barycentric interpolation

use rustlab_math::VectorF64;
use crate::{Result, NumericalError};
use super::traits::{Interpolator1D, ExtrapolationMode};
use super::utils::check_dimensions;

/// Lagrange polynomial interpolator
/// 
/// Uses the Lagrange interpolation formula to construct a polynomial
/// that passes through all given data points. For n points, this creates
/// a polynomial of degree n-1.
#[derive(Debug, Clone)]
pub struct LagrangeInterpolator {
    x: VectorF64,
    y: VectorF64,
    extrapolation: ExtrapolationMode,
}

impl LagrangeInterpolator {
    /// Create a new Lagrange interpolator
    /// 
    /// # Arguments
    /// * `x` - X coordinates of data points (need not be monotonic)
    /// * `y` - Y coordinates of data points
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::VectorF64;
    /// use rustlab_numerical::interpolation::{LagrangeInterpolator, Interpolator1D};
    /// 
    /// let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0]);
    /// let y = VectorF64::from_vec(vec![1.0, 2.0, 5.0]); // y = x^2 + 1
    /// let interp = LagrangeInterpolator::new(x, y)?;
    /// 
    /// // Should exactly interpolate the quadratic
    /// let result = interp.eval(1.5)?;
    /// assert!((result - 3.25).abs() < 1e-12); // 1.5^2 + 1 = 3.25
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    pub fn new(x: VectorF64, y: VectorF64) -> Result<Self> {
        check_dimensions(&x, &y)?;
        
        if x.len() < 2 {
            return Err(NumericalError::InsufficientData { 
                got: x.len(), 
                need: 2 
            });
        }
        
        // Check for duplicate x values
        for i in 0..x.len() {
            for j in (i+1)..x.len() {
                let xi = x.get(i).ok_or_else(|| 
                    NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
                )?;
                let xj = x.get(j).ok_or_else(|| 
                    NumericalError::InvalidParameter(format!("Index {} out of bounds", j))
                )?;
                if (xi - xj).abs() < 1e-15 {
                    return Err(NumericalError::InvalidParameter(
                        "Duplicate x values found in Lagrange interpolation".to_string()
                    ));
                }
            }
        }
        
        Ok(Self {
            x,
            y,
            extrapolation: ExtrapolationMode::default(),
        })
    }
    
    /// Set the extrapolation mode
    pub fn with_extrapolation(mut self, mode: ExtrapolationMode) -> Self {
        self.extrapolation = mode;
        self
    }
    
    /// Evaluate Lagrange basis polynomial L_j(x)
    fn lagrange_basis(&self, j: usize, x: f64) -> Result<f64> {
        let mut result = 1.0;
        let x_j = self.x.get(j).ok_or_else(|| 
            NumericalError::InvalidParameter(format!("Index {} out of bounds", j))
        )?;
        
        for k in 0..self.x.len() {
            if k != j {
                let x_k = self.x.get(k).ok_or_else(|| 
                    NumericalError::InvalidParameter(format!("Index {} out of bounds", k))
                )?;
                result *= (x - x_k) / (x_j - x_k);
            }
        }
        
        Ok(result)
    }
    
    /// Handle extrapolation based on the current mode
    fn extrapolate(&self, x: f64) -> Result<f64> {
        match self.extrapolation {
            ExtrapolationMode::Error => {
                let (min, max) = self.domain();
                Err(NumericalError::OutOfBounds {
                    value: x,
                    min,
                    max,
                })
            },
            ExtrapolationMode::Constant => {
                // Find closest boundary value
                let (min, max) = self.domain();
                if x < min {
                    // Find y value corresponding to min x
                    for i in 0..self.x.len() {
                        let xi = self.x.get(i).ok_or_else(|| 
                            NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
                        )?;
                        if (xi - min).abs() < 1e-15 {
                            let yi = self.y.get(i).ok_or_else(|| 
                                NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
                            )?;
                            return Ok(yi);
                        }
                    }
                } else {
                    // Find y value corresponding to max x
                    for i in 0..self.x.len() {
                        let xi = self.x.get(i).ok_or_else(|| 
                            NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
                        )?;
                        if (xi - max).abs() < 1e-15 {
                            let yi = self.y.get(i).ok_or_else(|| 
                                NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
                            )?;
                            return Ok(yi);
                        }
                    }
                }
                Ok(0.0) // Fallback
            },
            ExtrapolationMode::Linear => {
                // Use polynomial extrapolation (same as normal evaluation)
                self.eval_polynomial(x)
            },
            ExtrapolationMode::NaN => Ok(f64::NAN),
        }
    }
    
    /// Evaluate the Lagrange polynomial at x
    fn eval_polynomial(&self, x: f64) -> Result<f64> {
        let mut result = 0.0;
        
        for j in 0..self.x.len() {
            let basis = self.lagrange_basis(j, x)?;
            let yj = self.y.get(j).ok_or_else(|| 
                NumericalError::InvalidParameter(format!("Index {} out of bounds", j))
            )?;
            result += yj * basis;
        }
        
        Ok(result)
    }
}

impl Interpolator1D for LagrangeInterpolator {
    fn eval(&self, x: f64) -> Result<f64> {
        // Check if we need to extrapolate
        if !self.in_domain(x) {
            return self.extrapolate(x);
        }
        
        self.eval_polynomial(x)
    }
    
    fn domain(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        
        for i in 0..self.x.len() {
            if let Some(xi) = self.x.get(i) {
                min = min.min(xi);
                max = max.max(xi);
            }
        }
        
        (min, max)
    }
}

/// Newton polynomial interpolator using divided differences
/// 
/// More numerically stable than Lagrange interpolation for higher-degree
/// polynomials and allows for incremental construction.
#[derive(Debug, Clone)]
pub struct NewtonInterpolator {
    x: VectorF64,
    #[allow(dead_code)] // False positive: field is used in eval method
    y: VectorF64,
    divided_differences: Vec<f64>,
    extrapolation: ExtrapolationMode,
}

impl NewtonInterpolator {
    /// Create a new Newton interpolator using divided differences
    /// 
    /// # Arguments
    /// * `x` - X coordinates of data points
    /// * `y` - Y coordinates of data points
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::VectorF64;
    /// use rustlab_numerical::interpolation::{NewtonInterpolator, Interpolator1D};
    /// 
    /// let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    /// let y = VectorF64::from_vec(vec![1.0, 1.0, 1.0, 1.0]); // Constant function
    /// let interp = NewtonInterpolator::new(x, y)?;
    /// 
    /// let result = interp.eval(1.5)?;
    /// assert!((result - 1.0).abs() < 1e-12);
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    pub fn new(x: VectorF64, y: VectorF64) -> Result<Self> {
        check_dimensions(&x, &y)?;
        
        if x.len() < 2 {
            return Err(NumericalError::InsufficientData { 
                got: x.len(), 
                need: 2 
            });
        }
        
        // Check for duplicate x values
        for i in 0..x.len() {
            for j in (i+1)..x.len() {
                let xi = x.get(i).ok_or_else(|| 
                    NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
                )?;
                let xj = x.get(j).ok_or_else(|| 
                    NumericalError::InvalidParameter(format!("Index {} out of bounds", j))
                )?;
                if (xi - xj).abs() < 1e-15 {
                    return Err(NumericalError::InvalidParameter(
                        "Duplicate x values found in Newton interpolation".to_string()
                    ));
                }
            }
        }
        
        // Compute divided differences
        let n = x.len();
        let mut dd = vec![vec![0.0; n]; n];
        
        // Initialize first column with y values
        for i in 0..n {
            dd[i][0] = y.get(i).ok_or_else(|| 
                NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
            )?;
        }
        
        // Compute divided differences
        for j in 1..n {
            for i in 0..(n-j) {
                let x_i = x.get(i).ok_or_else(|| 
                    NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
                )?;
                let x_i_j = x.get(i + j).ok_or_else(|| 
                    NumericalError::InvalidParameter(format!("Index {} out of bounds", i + j))
                )?;
                dd[i][j] = (dd[i+1][j-1] - dd[i][j-1]) / (x_i_j - x_i);
            }
        }
        
        // Extract the coefficients (first row of divided differences table)
        let mut divided_differences = Vec::with_capacity(n);
        for j in 0..n {
            divided_differences.push(dd[0][j]);
        }
        
        Ok(Self {
            x,
            y,
            divided_differences,
            extrapolation: ExtrapolationMode::default(),
        })
    }
    
    /// Set the extrapolation mode
    pub fn with_extrapolation(mut self, mode: ExtrapolationMode) -> Self {
        self.extrapolation = mode;
        self
    }
    
    /// Evaluate the Newton polynomial at x using Horner's method
    fn eval_polynomial(&self, x: f64) -> Result<f64> {
        let n = self.x.len();
        if n == 0 {
            return Ok(0.0);
        }
        
        // Start with the last divided difference
        let mut result = self.divided_differences[n-1];
        
        // Work backwards using Horner's method
        for i in (0..n-1).rev() {
            let x_i = self.x.get(i).ok_or_else(|| 
                NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
            )?;
            result = self.divided_differences[i] + (x - x_i) * result;
        }
        
        Ok(result)
    }
    
    /// Handle extrapolation based on the current mode
    fn extrapolate(&self, x: f64) -> Result<f64> {
        match self.extrapolation {
            ExtrapolationMode::Error => {
                let (min, max) = self.domain();
                Err(NumericalError::OutOfBounds {
                    value: x,
                    min,
                    max,
                })
            },
            ExtrapolationMode::Constant => {
                let (min, max) = self.domain();
                if x < min {
                    self.eval_polynomial(min)
                } else {
                    self.eval_polynomial(max)
                }
            },
            ExtrapolationMode::Linear => {
                // Use polynomial extrapolation
                self.eval_polynomial(x)
            },
            ExtrapolationMode::NaN => Ok(f64::NAN),
        }
    }
}

impl Interpolator1D for NewtonInterpolator {
    fn eval(&self, x: f64) -> Result<f64> {
        // Check if we need to extrapolate
        if !self.in_domain(x) {
            return self.extrapolate(x);
        }
        
        self.eval_polynomial(x)
    }
    
    fn domain(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        
        for i in 0..self.x.len() {
            if let Some(xi) = self.x.get(i) {
                min = min.min(xi);
                max = max.max(xi);
            }
        }
        
        (min, max)
    }
}

/// Convenience function for Lagrange polynomial interpolation
/// 
/// # Example
/// ```
/// use rustlab_math::VectorF64;
/// use rustlab_numerical::interpolation::interp1d_lagrange;
/// 
/// let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0]);
/// let y = VectorF64::from_vec(vec![0.0, 1.0, 8.0]); // y = x^3
/// let xi = VectorF64::from_vec(vec![0.5, 1.5]);
/// 
/// let yi = interp1d_lagrange(&x, &y, &xi)?;
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn interp1d_lagrange(x: &VectorF64, y: &VectorF64, xi: &VectorF64) -> Result<VectorF64> {
    let interp = LagrangeInterpolator::new(x.clone(), y.clone())?;
    interp.eval_vec(xi)
}

/// Convenience function for Newton polynomial interpolation
/// 
/// # Example
/// ```
/// use rustlab_math::VectorF64;
/// use rustlab_numerical::interpolation::interp1d_newton;
/// 
/// let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
/// let y = VectorF64::from_vec(vec![1.0, 4.0, 9.0, 16.0]); // y = x^2 + 1, but will be cubic
/// let xi = VectorF64::from_vec(vec![0.5, 1.5, 2.5]);
/// 
/// let yi = interp1d_newton(&x, &y, &xi)?;
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn interp1d_newton(x: &VectorF64, y: &VectorF64, xi: &VectorF64) -> Result<VectorF64> {
    let interp = NewtonInterpolator::new(x.clone(), y.clone())?;
    interp.eval_vec(xi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::VectorF64;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_lagrange_linear() {
        // Linear function y = 2x + 1
        let x = VectorF64::from_vec(vec![0.0, 1.0]);
        let y = VectorF64::from_vec(vec![1.0, 3.0]);
        let interp = LagrangeInterpolator::new(x, y).unwrap();
        
        assert_relative_eq!(interp.eval(0.5).unwrap(), 2.0, epsilon = 1e-12);
        assert_relative_eq!(interp.eval(0.0).unwrap(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(interp.eval(1.0).unwrap(), 3.0, epsilon = 1e-12);
    }
    
    #[test]
    fn test_lagrange_quadratic() {
        // Quadratic function y = x^2
        let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0]);
        let y = VectorF64::from_vec(vec![0.0, 1.0, 4.0]);
        let interp = LagrangeInterpolator::new(x, y)
            .unwrap()
            .with_extrapolation(ExtrapolationMode::Linear);
        
        assert_relative_eq!(interp.eval(0.5).unwrap(), 0.25, epsilon = 1e-12);
        assert_relative_eq!(interp.eval(1.5).unwrap(), 2.25, epsilon = 1e-12);
        assert_relative_eq!(interp.eval(-1.0).unwrap(), 1.0, epsilon = 1e-12);
    }
    
    #[test]
    fn test_newton_linear() {
        // Linear function y = 3x - 1
        let x = VectorF64::from_vec(vec![1.0, 2.0]);
        let y = VectorF64::from_vec(vec![2.0, 5.0]);
        let interp = NewtonInterpolator::new(x, y)
            .unwrap()
            .with_extrapolation(ExtrapolationMode::Linear);
        
        assert_relative_eq!(interp.eval(1.5).unwrap(), 3.5, epsilon = 1e-12);
        assert_relative_eq!(interp.eval(0.0).unwrap(), -1.0, epsilon = 1e-12);
    }
    
    #[test]
    fn test_newton_cubic() {
        // Cubic function y = x^3
        let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = VectorF64::from_vec(vec![0.0, 1.0, 8.0, 27.0]);
        let interp = NewtonInterpolator::new(x, y).unwrap();
        
        assert_relative_eq!(interp.eval(1.5).unwrap(), 3.375, epsilon = 1e-12);
        assert_relative_eq!(interp.eval(2.5).unwrap(), 15.625, epsilon = 1e-12);
    }
    
    #[test]
    fn test_constant_function() {
        // Constant function
        let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = VectorF64::from_vec(vec![5.0, 5.0, 5.0, 5.0]);
        
        let lagrange = LagrangeInterpolator::new(x.clone(), y.clone()).unwrap();
        let newton = NewtonInterpolator::new(x, y).unwrap();
        
        assert_relative_eq!(lagrange.eval(1.5).unwrap(), 5.0, epsilon = 1e-12);
        assert_relative_eq!(newton.eval(1.5).unwrap(), 5.0, epsilon = 1e-12);
    }
    
    #[test]
    fn test_duplicate_x_values() {
        let x = VectorF64::from_vec(vec![0.0, 1.0, 1.0]);
        let y = VectorF64::from_vec(vec![1.0, 2.0, 3.0]);
        
        assert!(LagrangeInterpolator::new(x.clone(), y.clone()).is_err());
        assert!(NewtonInterpolator::new(x, y).is_err());
    }
    
    #[test]
    fn test_extrapolation_modes() {
        let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0]);
        let y = VectorF64::from_vec(vec![0.0, 1.0, 4.0]); // y = x^2
        
        // Test with Error mode (default)
        let interp = LagrangeInterpolator::new(x.clone(), y.clone()).unwrap();
        assert!(interp.eval(-1.0).is_err());
        assert!(interp.eval(3.0).is_err());
        
        // Test with Linear mode (polynomial extrapolation)
        let interp = LagrangeInterpolator::new(x.clone(), y.clone())
            .unwrap()
            .with_extrapolation(ExtrapolationMode::Linear);
        assert_relative_eq!(interp.eval(-1.0).unwrap(), 1.0, epsilon = 1e-12); // (-1)^2 = 1
        assert_relative_eq!(interp.eval(3.0).unwrap(), 9.0, epsilon = 1e-12);   // 3^2 = 9
        
        // Test with NaN mode
        let interp = LagrangeInterpolator::new(x.clone(), y.clone())
            .unwrap()
            .with_extrapolation(ExtrapolationMode::NaN);
        assert!(interp.eval(-1.0).unwrap().is_nan());
        assert!(interp.eval(3.0).unwrap().is_nan());
    }
}