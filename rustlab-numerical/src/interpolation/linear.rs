//! Linear interpolation methods

use rustlab_math::VectorF64;
use crate::{Result, NumericalError};
use super::traits::{Interpolator1D, ExtrapolationMode};
use super::utils::{check_monotonic, check_dimensions, find_interval};

/// Linear interpolator for 1D data
#[derive(Debug, Clone)]
pub struct LinearInterpolator {
    x: VectorF64,
    y: VectorF64,
    extrapolation: ExtrapolationMode,
}

impl LinearInterpolator {
    /// Create a new linear interpolator
    /// 
    /// # Arguments
    /// * `x` - Strictly increasing x values
    /// * `y` - Corresponding y values
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::VectorF64;
    /// use rustlab_numerical::interpolation::{LinearInterpolator, Interpolator1D};
    /// 
    /// let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    /// let y = VectorF64::from_vec(vec![1.0, 3.0, 2.0, 5.0]);
    /// let interp = LinearInterpolator::new(x, y)?;
    /// 
    /// // Interpolate at x = 1.5
    /// let result = interp.eval(1.5)?;
    /// assert_eq!(result, 2.5);
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    pub fn new(x: VectorF64, y: VectorF64) -> Result<Self> {
        check_dimensions(&x, &y)?;
        check_monotonic(&x, "linear interpolation")?;
        
        if x.len() < 2 {
            return Err(NumericalError::InsufficientData { 
                got: x.len(), 
                need: 2 
            });
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
    
    /// Perform linear interpolation between two points
    #[inline]
    fn lerp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
    
    /// Handle extrapolation based on the current mode
    fn extrapolate(&self, x: f64) -> Result<f64> {
        let n = self.x.len();
        let x_min = self.x.get(0).ok_or_else(|| 
            NumericalError::InvalidParameter("Empty vector".to_string())
        )?;
        let x_max = self.x.get(n-1).ok_or_else(|| 
            NumericalError::InvalidParameter("Empty vector".to_string())
        )?;
        
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
                    self.y.get(0).ok_or_else(|| 
                        NumericalError::InvalidParameter("Empty vector".to_string())
                    )
                } else {
                    self.y.get(n-1).ok_or_else(|| 
                        NumericalError::InvalidParameter("Empty vector".to_string())
                    )
                }
            },
            ExtrapolationMode::Linear => {
                if x < x_min {
                    // Extrapolate using first two points
                    let x0 = self.x.get(0).ok_or_else(|| 
                        NumericalError::InvalidParameter("Empty vector".to_string())
                    )?;
                    let y0 = self.y.get(0).ok_or_else(|| 
                        NumericalError::InvalidParameter("Empty vector".to_string())
                    )?;
                    let x1 = self.x.get(1).ok_or_else(|| 
                        NumericalError::InvalidParameter("Insufficient data for linear extrapolation".to_string())
                    )?;
                    let y1 = self.y.get(1).ok_or_else(|| 
                        NumericalError::InvalidParameter("Insufficient data for linear extrapolation".to_string())
                    )?;
                    Ok(Self::lerp(x0, y0, x1, y1, x))
                } else {
                    // Extrapolate using last two points
                    let x0 = self.x.get(n-2).ok_or_else(|| 
                        NumericalError::InvalidParameter("Insufficient data for linear extrapolation".to_string())
                    )?;
                    let y0 = self.y.get(n-2).ok_or_else(|| 
                        NumericalError::InvalidParameter("Insufficient data for linear extrapolation".to_string())
                    )?;
                    let x1 = self.x.get(n-1).ok_or_else(|| 
                        NumericalError::InvalidParameter("Insufficient data for linear extrapolation".to_string())
                    )?;
                    let y1 = self.y.get(n-1).ok_or_else(|| 
                        NumericalError::InvalidParameter("Insufficient data for linear extrapolation".to_string())
                    )?;
                    Ok(Self::lerp(x0, y0, x1, y1, x))
                }
            },
            ExtrapolationMode::NaN => Ok(f64::NAN),
        }
    }
}

impl Interpolator1D for LinearInterpolator {
    fn eval(&self, x: f64) -> Result<f64> {
        // Check if we need to extrapolate
        if !self.in_domain(x) {
            return self.extrapolate(x);
        }
        
        // Find the interval containing x
        let i = find_interval(&self.x, x)
            .ok_or_else(|| NumericalError::InvalidParameter("Failed to find interval".to_string()))?;
        
        // Get values using rustlab-math's API
        let x0 = self.x.get(i).ok_or_else(|| 
            NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
        )?;
        let y0 = self.y.get(i).ok_or_else(|| 
            NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
        )?;
        let x1 = self.x.get(i+1).ok_or_else(|| 
            NumericalError::InvalidParameter(format!("Index {} out of bounds", i+1))
        )?;
        let y1 = self.y.get(i+1).ok_or_else(|| 
            NumericalError::InvalidParameter(format!("Index {} out of bounds", i+1))
        )?;
        
        // Linear interpolation
        Ok(Self::lerp(x0, y0, x1, y1, x))
    }
    
    fn domain(&self) -> (f64, f64) {
        let n = self.x.len();
        (
            self.x.get(0).unwrap_or(0.0), 
            self.x.get(n-1).unwrap_or(0.0)
        )
    }
}

/// Piecewise linear interpolation for scattered 1D data
/// 
/// This is a convenience function that creates a LinearInterpolator
/// and evaluates it at the given points.
/// 
/// # Example
/// ```
/// use rustlab_math::VectorF64;
/// use rustlab_numerical::interpolation::interp1d_linear;
/// 
/// let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
/// let y = VectorF64::from_vec(vec![1.0, 3.0, 2.0, 5.0]);
/// let xi = VectorF64::from_vec(vec![0.5, 1.5, 2.5]);
/// 
/// let yi = interp1d_linear(&x, &y, &xi)?;
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn interp1d_linear(x: &VectorF64, y: &VectorF64, xi: &VectorF64) -> Result<VectorF64> {
    let interp = LinearInterpolator::new(x.clone(), y.clone())?;
    interp.eval_vec(xi)
}

/// Simple linear interpolation between two points
/// 
/// # Arguments
/// * `x0`, `y0` - First point
/// * `x1`, `y1` - Second point  
/// * `x` - Point at which to interpolate
/// 
/// # Example
/// ```
/// use rustlab_numerical::interpolation::lerp;
/// 
/// let result = lerp(0.0, 1.0, 2.0, 5.0, 1.0);
/// assert_eq!(result, 3.0);
/// ```
#[inline]
pub fn lerp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::VectorF64;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_linear_interpolation() {
        let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = VectorF64::from_vec(vec![1.0, 3.0, 2.0, 5.0]);
        let interp = LinearInterpolator::new(x, y).unwrap();
        
        // Test exact points
        assert_eq!(interp.eval(0.0).unwrap(), 1.0);
        assert_eq!(interp.eval(1.0).unwrap(), 3.0);
        assert_eq!(interp.eval(2.0).unwrap(), 2.0);
        assert_eq!(interp.eval(3.0).unwrap(), 5.0);
        
        // Test interpolation
        assert_eq!(interp.eval(0.5).unwrap(), 2.0);
        assert_eq!(interp.eval(1.5).unwrap(), 2.5);
        assert_eq!(interp.eval(2.5).unwrap(), 3.5);
    }
    
    #[test]
    fn test_extrapolation_modes() {
        let x = VectorF64::from_vec(vec![0.0, 1.0]);
        let y = VectorF64::from_vec(vec![0.0, 2.0]);
        
        // Error mode (default)
        let interp = LinearInterpolator::new(x.clone(), y.clone()).unwrap();
        assert!(interp.eval(-1.0).is_err());
        assert!(interp.eval(2.0).is_err());
        
        // Constant mode
        let interp = LinearInterpolator::new(x.clone(), y.clone())
            .unwrap()
            .with_extrapolation(ExtrapolationMode::Constant);
        assert_eq!(interp.eval(-1.0).unwrap(), 0.0);
        assert_eq!(interp.eval(2.0).unwrap(), 2.0);
        
        // Linear mode
        let interp = LinearInterpolator::new(x.clone(), y.clone())
            .unwrap()
            .with_extrapolation(ExtrapolationMode::Linear);
        assert_eq!(interp.eval(-1.0).unwrap(), -2.0);
        assert_eq!(interp.eval(2.0).unwrap(), 4.0);
        
        // NaN mode
        let interp = LinearInterpolator::new(x.clone(), y.clone())
            .unwrap()
            .with_extrapolation(ExtrapolationMode::NaN);
        assert!(interp.eval(-1.0).unwrap().is_nan());
        assert!(interp.eval(2.0).unwrap().is_nan());
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let x = VectorF64::from_vec(vec![0.0, 1.0, 2.0]);
        let y = VectorF64::from_vec(vec![1.0, 2.0]);
        assert!(LinearInterpolator::new(x, y).is_err());
    }
    
    #[test]
    fn test_non_monotonic() {
        let x = VectorF64::from_vec(vec![0.0, 2.0, 1.0]);
        let y = VectorF64::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(LinearInterpolator::new(x, y).is_err());
    }
}