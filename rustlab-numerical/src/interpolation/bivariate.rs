//! 2D (bivariate) interpolation methods
//! 
//! This module provides interpolation techniques for 2D data:
//! - Bilinear interpolation
//! - Bicubic interpolation

use rustlab_math::{VectorF64, ArrayF64};
use crate::{Result, NumericalError};
use super::traits::{Interpolator2D, ExtrapolationMode};

/// Bilinear interpolator for 2D gridded data
/// 
/// Performs bilinear interpolation on a regular 2D grid. The interpolation
/// is linear in both x and y directions.
#[derive(Debug, Clone)]
pub struct BilinearInterpolator {
    x: VectorF64,
    y: VectorF64,
    z: ArrayF64,
    extrapolation: ExtrapolationMode,
}

impl BilinearInterpolator {
    /// Create a new bilinear interpolator
    /// 
    /// # Arguments
    /// * `x` - X coordinates (strictly increasing)
    /// * `y` - Y coordinates (strictly increasing) 
    /// * `z` - Z values on the grid, where z[i][j] corresponds to (x[i], y[j])
    /// 
    /// # Example
    /// ```
    /// use rustlab_core::{vec, rmat};
    /// use rustlab_numerical::interpolation::{BilinearInterpolator, Interpolator2D};
    /// 
    /// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
    /// let y = VectorF64::from_slice(&[0.0, 1.0]);
    /// let z = ArrayF64::from_vec2d(vec![
    ///     vec![1.0, 2.0],
    ///     vec![3.0, 4.0],
    ///     vec![5.0, 6.0]
    /// ]).unwrap();
    /// 
    /// let interp = BilinearInterpolator::new(x, y, z)?;
    /// let result = interp.eval(0.5, 0.5)?;
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    pub fn new(x: VectorF64, y: VectorF64, z: ArrayF64) -> Result<Self> {
        // Check dimensions
        if x.len() != z.nrows() {
            return Err(NumericalError::DimensionMismatch(x.len(), z.nrows()));
        }
        if y.len() != z.ncols() {
            return Err(NumericalError::DimensionMismatch(y.len(), z.ncols()));
        }
        
        if x.len() < 2 {
            return Err(NumericalError::InsufficientData { got: x.len(), need: 2 });
        }
        if y.len() < 2 {
            return Err(NumericalError::InsufficientData { got: y.len(), need: 2 });
        }
        
        // Check that x and y are strictly increasing
        for i in 1..x.len() {
            if x.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i)))? <= x.get(i-1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i-1)))? {
                return Err(NumericalError::NotMonotonic { method: "bilinear interpolation" });
            }
        }
        for i in 1..y.len() {
            if y.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", i)))? <= y.get(i-1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", i-1)))? {
                return Err(NumericalError::NotMonotonic { method: "bilinear interpolation" });
            }
        }
        
        Ok(Self {
            x,
            y,
            z,
            extrapolation: ExtrapolationMode::default(),
        })
    }
    
    /// Set the extrapolation mode
    pub fn with_extrapolation(mut self, mode: ExtrapolationMode) -> Self {
        self.extrapolation = mode;
        self
    }
    
    /// Find the grid cell containing the point (x, y)
    fn find_cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let i = self.find_x_interval(x)?;
        let j = self.find_y_interval(y)?;
        Some((i, j))
    }
    
    /// Find x interval containing the target value
    fn find_x_interval(&self, target: f64) -> Option<usize> {
        let n = self.x.len();
        if n < 2 {
            return None;
        }
        
        let x_min = self.x.get(0)?;
        let x_max = self.x.get(n-1)?;
        if target < x_min || target > x_max {
            return None;
        }
        
        // Binary search
        let mut left = 0;
        let mut right = n - 1;
        
        while right - left > 1 {
            let mid = (left + right) / 2;
            let x_mid = self.x.get(mid)?;
            if target < x_mid {
                right = mid;
            } else {
                left = mid;
            }
        }
        
        Some(left)
    }
    
    /// Find y interval containing the target value
    fn find_y_interval(&self, target: f64) -> Option<usize> {
        let n = self.y.len();
        if n < 2 {
            return None;
        }
        
        let y_min = self.y.get(0)?;
        let y_max = self.y.get(n-1)?;
        if target < y_min || target > y_max {
            return None;
        }
        
        // Binary search
        let mut left = 0;
        let mut right = n - 1;
        
        while right - left > 1 {
            let mid = (left + right) / 2;
            let y_mid = self.y.get(mid)?;
            if target < y_mid {
                right = mid;
            } else {
                left = mid;
            }
        }
        
        Some(left)
    }
    
    /// Perform bilinear interpolation within a grid cell
    fn bilinear_interp(&self, x: f64, y: f64, i: usize, j: usize) -> Result<f64> {
        // Get grid points
        let x1 = self.x.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i)))?;
        let x2 = self.x.get(i + 1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i + 1)))?;
        let y1 = self.y.get(j).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", j)))?;
        let y2 = self.y.get(j + 1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", j + 1)))?;
        
        // Get function values at corners
        let z11 = self.z.get(i, j).ok_or_else(|| NumericalError::InvalidParameter(format!("Index ({}, {}) out of bounds in z array", i, j)))?;     // (x1, y1)
        let z21 = self.z.get(i + 1, j).ok_or_else(|| NumericalError::InvalidParameter(format!("Index ({}, {}) out of bounds in z array", i + 1, j)))?; // (x2, y1)
        let z12 = self.z.get(i, j + 1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index ({}, {}) out of bounds in z array", i, j + 1)))?; // (x1, y2)
        let z22 = self.z.get(i + 1, j + 1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index ({}, {}) out of bounds in z array", i + 1, j + 1)))?; // (x2, y2)
        
        // Bilinear interpolation formula
        let dx = x2 - x1;
        let dy = y2 - y1;
        let tx = (x - x1) / dx;
        let ty = (y - y1) / dy;
        
        let z1 = z11 * (1.0 - tx) + z21 * tx; // Interpolate along x at y1
        let z2 = z12 * (1.0 - tx) + z22 * tx; // Interpolate along x at y2
        let z = z1 * (1.0 - ty) + z2 * ty;    // Interpolate along y
        
        Ok(z)
    }
    
    /// Handle extrapolation based on the current mode
    fn extrapolate(&self, x: f64, y: f64) -> Result<f64> {
        let ((x_min, x_max), (y_min, y_max)) = self.domain();
        
        match self.extrapolation {
            ExtrapolationMode::Error => {
                Err(NumericalError::OutOfBounds {
                    value: if x < x_min || x > x_max { x } else { y },
                    min: if x < x_min || x > x_max { x_min } else { y_min },
                    max: if x < x_min || x > x_max { x_max } else { y_max },
                })
            },
            ExtrapolationMode::Constant => {
                // Find nearest boundary point
                let x_clamp = x.clamp(x_min, x_max);
                let y_clamp = y.clamp(y_min, y_max);
                
                // Find the grid cell (this should now be valid)
                if let Some((i, j)) = self.find_cell(x_clamp, y_clamp) {
                    self.bilinear_interp(x_clamp, y_clamp, i, j)
                } else {
                    // Fallback to corner value
                    let i = if x_clamp <= x_min { 0 } else { self.x.len() - 2 };
                    let j = if y_clamp <= y_min { 0 } else { self.y.len() - 2 };
                    Ok(self.z.get(i, j).ok_or_else(|| NumericalError::InvalidParameter(format!("Index ({}, {}) out of bounds in z array", i, j)))?)
                }
            },
            ExtrapolationMode::Linear => {
                // For 2D, linear extrapolation is complex
                // For now, use bilinear extrapolation by extending the grid
                let x_clamp = x.clamp(x_min, x_max);
                let y_clamp = y.clamp(y_min, y_max);
                
                if let Some((i, j)) = self.find_cell(x_clamp, y_clamp) {
                    self.bilinear_interp(x_clamp, y_clamp, i, j)
                } else {
                    self.extrapolate_linear_2d(x, y)
                }
            },
            ExtrapolationMode::NaN => Ok(f64::NAN),
        }
    }
    
    /// Simple linear extrapolation for 2D (uses nearest edge)
    fn extrapolate_linear_2d(&self, x: f64, y: f64) -> Result<f64> {
        let ((x_min, x_max), (y_min, y_max)) = self.domain();
        
        // Clamp to boundary and use nearest edge
        let x_clamp = x.clamp(x_min, x_max);
        let y_clamp = y.clamp(y_min, y_max);
        
        if let Some((i, j)) = self.find_cell(x_clamp, y_clamp) {
            self.bilinear_interp(x_clamp, y_clamp, i, j)
        } else {
            // Use corner value as fallback
            Ok(self.z.get(0, 0).ok_or_else(|| NumericalError::InvalidParameter("Index (0, 0) out of bounds in z array".to_string()))?)
        }
    }
}

impl Interpolator2D for BilinearInterpolator {
    fn eval(&self, x: f64, y: f64) -> Result<f64> {
        if let Some((i, j)) = self.find_cell(x, y) {
            self.bilinear_interp(x, y, i, j)
        } else {
            self.extrapolate(x, y)
        }
    }
    
    fn domain(&self) -> ((f64, f64), (f64, f64)) {
        let x_min = self.x.get(0).unwrap_or(0.0);
        let x_max = self.x.get(self.x.len() - 1).unwrap_or(0.0);
        let y_min = self.y.get(0).unwrap_or(0.0);
        let y_max = self.y.get(self.y.len() - 1).unwrap_or(0.0);
        
        ((x_min, x_max), (y_min, y_max))
    }
}

/// Bicubic interpolator for 2D gridded data
/// 
/// Performs bicubic interpolation on a regular 2D grid. Uses cubic polynomials
/// in both x and y directions, providing C¹ continuity.
#[derive(Debug, Clone)]
pub struct BicubicInterpolator {
    x: VectorF64,
    y: VectorF64,
    z: ArrayF64,
    extrapolation: ExtrapolationMode,
}

impl BicubicInterpolator {
    /// Create a new bicubic interpolator
    /// 
    /// # Arguments
    /// * `x` - X coordinates (strictly increasing)
    /// * `y` - Y coordinates (strictly increasing)
    /// * `z` - Z values on the grid, where z[i][j] corresponds to (x[i], y[j])
    /// 
    /// # Example
    /// ```
    /// use rustlab_core::{vec, rmat};
    /// use rustlab_numerical::interpolation::{BicubicInterpolator, Interpolator2D};
    /// 
    /// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
    /// let y = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
    /// let z = ArrayF64::from_vec2d(vec![
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     vec![2.0, 3.0, 4.0, 5.0],
    ///     vec![3.0, 4.0, 5.0, 6.0],
    ///     vec![4.0, 5.0, 6.0, 7.0]
    /// ]).unwrap();
    /// 
    /// let interp = BicubicInterpolator::new(x, y, z)?;
    /// let result = interp.eval(1.5, 1.5)?;
    /// # Ok::<(), rustlab_numerical::NumericalError>(())
    /// ```
    pub fn new(x: VectorF64, y: VectorF64, z: ArrayF64) -> Result<Self> {
        // Check dimensions
        if x.len() != z.nrows() {
            return Err(NumericalError::DimensionMismatch(x.len(), z.nrows()));
        }
        if y.len() != z.ncols() {
            return Err(NumericalError::DimensionMismatch(y.len(), z.ncols()));
        }
        
        if x.len() < 4 {
            return Err(NumericalError::InsufficientData { got: x.len(), need: 4 });
        }
        if y.len() < 4 {
            return Err(NumericalError::InsufficientData { got: y.len(), need: 4 });
        }
        
        // Check that x and y are strictly increasing
        for i in 1..x.len() {
            if x.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i)))? <= x.get(i-1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i-1)))? {
                return Err(NumericalError::NotMonotonic { method: "bicubic interpolation" });
            }
        }
        for i in 1..y.len() {
            if y.get(i).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", i)))? <= y.get(i-1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", i-1)))? {
                return Err(NumericalError::NotMonotonic { method: "bicubic interpolation" });
            }
        }
        
        Ok(Self {
            x,
            y,
            z,
            extrapolation: ExtrapolationMode::default(),
        })
    }
    
    /// Set the extrapolation mode
    pub fn with_extrapolation(mut self, mode: ExtrapolationMode) -> Self {
        self.extrapolation = mode;
        self
    }
    
    /// Find the grid cell containing the point (x, y) for bicubic interpolation
    /// Returns indices such that we can use a 4x4 stencil
    fn find_bicubic_cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let i = self.find_x_interval_bicubic(x)?;
        let j = self.find_y_interval_bicubic(y)?;
        Some((i, j))
    }
    
    /// Find x interval for bicubic (needs 4 points)
    fn find_x_interval_bicubic(&self, target: f64) -> Option<usize> {
        let n = self.x.len();
        if n < 4 {
            return None;
        }
        
        let x_min = self.x.get(1)?; // Start from second point
        let x_max = self.x.get(n-2)?; // End at second-to-last point
        if target < x_min || target > x_max {
            return None;
        }
        
        // Binary search, but return index - 1 so we have room for 4-point stencil
        for i in 1..n-2 {
            let xi = self.x.get(i)?;
            let xi1 = self.x.get(i+1)?;
            if target >= xi && target <= xi1 {
                return Some(i-1); // Return i-1 so we can use points [i-1, i, i+1, i+2]
            }
        }
        
        None
    }
    
    /// Find y interval for bicubic (needs 4 points)
    fn find_y_interval_bicubic(&self, target: f64) -> Option<usize> {
        let n = self.y.len();
        if n < 4 {
            return None;
        }
        
        let y_min = self.y.get(1)?;
        let y_max = self.y.get(n-2)?;
        if target < y_min || target > y_max {
            return None;
        }
        
        for j in 1..n-2 {
            let yj = self.y.get(j)?;
            let yj1 = self.y.get(j+1)?;
            if target >= yj && target <= yj1 {
                return Some(j-1);
            }
        }
        
        None
    }
    
    /// Cubic interpolation in one dimension
    fn cubic_interp(&self, p: [f64; 4], t: f64) -> f64 {
        // Catmull-Rom spline coefficients
        let a = -0.5 * p[0] + 1.5 * p[1] - 1.5 * p[2] + 0.5 * p[3];
        let b = p[0] - 2.5 * p[1] + 2.0 * p[2] - 0.5 * p[3];
        let c = -0.5 * p[0] + 0.5 * p[2];
        let d = p[1];
        
        a * t * t * t + b * t * t + c * t + d
    }
    
    /// Perform bicubic interpolation within a grid cell
    fn bicubic_interp(&self, x: f64, y: f64, i: usize, j: usize) -> Result<f64> {
        // Get the 4x4 grid of values around the point
        let mut p = [[0.0; 4]; 4];
        for di in 0..4 {
            for dj in 0..4 {
                p[di][dj] = self.z.get(i + di, j + dj).ok_or_else(|| NumericalError::InvalidParameter(format!("Index ({}, {}) out of bounds in z array", i + di, j + dj)))?;
            }
        }
        
        // Get grid coordinates
        let x1 = self.x.get(i + 1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i + 1)))?;
        let x2 = self.x.get(i + 2).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in x array", i + 2)))?;
        let y1 = self.y.get(j + 1).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", j + 1)))?;
        let y2 = self.y.get(j + 2).ok_or_else(|| NumericalError::InvalidParameter(format!("Index {} out of bounds in y array", j + 2)))?;
        
        // Normalize coordinates to [0, 1]
        let tx = (x - x1) / (x2 - x1);
        let ty = (y - y1) / (y2 - y1);
        
        // Interpolate along x direction for each y
        let mut arr = [0.0; 4];
        for k in 0..4 {
            arr[k] = self.cubic_interp([p[0][k], p[1][k], p[2][k], p[3][k]], tx);
        }
        
        // Interpolate along y direction
        Ok(self.cubic_interp(arr, ty))
    }
    
    /// Handle extrapolation for bicubic interpolation
    fn extrapolate(&self, x: f64, y: f64) -> Result<f64> {
        match self.extrapolation {
            ExtrapolationMode::Error => {
                let ((x_min, x_max), (y_min, y_max)) = self.domain();
                Err(NumericalError::OutOfBounds {
                    value: if x < x_min || x > x_max { x } else { y },
                    min: if x < x_min || x > x_max { x_min } else { y_min },
                    max: if x < x_min || x > x_max { x_max } else { y_max },
                })
            },
            ExtrapolationMode::Constant => {
                // Use boundary values
                let ((x_min, x_max), (y_min, y_max)) = self.domain();
                let x_clamp = x.clamp(x_min, x_max);
                let y_clamp = y.clamp(y_min, y_max);
                
                // Find nearest valid cell and interpolate
                if let Some((i, j)) = self.find_bicubic_cell(x_clamp, y_clamp) {
                    self.bicubic_interp(x_clamp, y_clamp, i, j)
                } else {
                    // Fallback to boundary value
                    Ok(self.z.get(1, 1).ok_or_else(|| NumericalError::InvalidParameter("Index (1, 1) out of bounds in z array".to_string()))?)
                }
            },
            ExtrapolationMode::Linear | ExtrapolationMode::NaN => {
                if self.extrapolation == ExtrapolationMode::NaN {
                    Ok(f64::NAN)
                } else {
                    // Use cubic extrapolation (same as normal evaluation)
                    let ((x_min, x_max), (y_min, y_max)) = self.domain();
                    let x_clamp = x.clamp(x_min, x_max);
                    let y_clamp = y.clamp(y_min, y_max);
                    
                    if let Some((i, j)) = self.find_bicubic_cell(x_clamp, y_clamp) {
                        self.bicubic_interp(x_clamp, y_clamp, i, j)
                    } else {
                        Ok(self.z.get(1, 1).ok_or_else(|| NumericalError::InvalidParameter("Index (1, 1) out of bounds in z array".to_string()))?)
                    }
                }
            },
        }
    }
}

impl Interpolator2D for BicubicInterpolator {
    fn eval(&self, x: f64, y: f64) -> Result<f64> {
        if let Some((i, j)) = self.find_bicubic_cell(x, y) {
            self.bicubic_interp(x, y, i, j)
        } else {
            self.extrapolate(x, y)
        }
    }
    
    fn domain(&self) -> ((f64, f64), (f64, f64)) {
        // For bicubic, domain is smaller since we need 4x4 stencil
        let x_min = self.x.get(1).unwrap_or(0.0);
        let x_max = self.x.get(self.x.len() - 2).unwrap_or(0.0);
        let y_min = self.y.get(1).unwrap_or(0.0);
        let y_max = self.y.get(self.y.len() - 2).unwrap_or(0.0);
        
        ((x_min, x_max), (y_min, y_max))
    }
}

/// Convenience function for bilinear interpolation
/// 
/// # Example
/// ```
/// use rustlab_core::{vec, rmat};
/// use rustlab_numerical::interpolation::interp2d_bilinear;
/// 
/// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
/// let y = VectorF64::from_slice(&[0.0, 1.0]);
/// let z = ArrayF64::from_vec2d(vec![
///     vec![1.0, 2.0],
///     vec![3.0, 4.0],
///     vec![5.0, 6.0]
/// ]).unwrap();
/// 
/// let result = interp2d_bilinear(&x, &y, &z, 0.5, 0.5)?;
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn interp2d_bilinear(x: &VectorF64, y: &VectorF64, z: &ArrayF64, xi: f64, yi: f64) -> Result<f64> {
    let interp = BilinearInterpolator::new(x.clone(), y.clone(), z.clone())?;
    interp.eval(xi, yi)
}

/// Convenience function for bicubic interpolation
/// 
/// # Example
/// ```
/// use rustlab_core::{vec, rmat};
/// use rustlab_numerical::interpolation::interp2d_bicubic;
/// 
/// let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
/// let y = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
/// let z = ArrayF64::from_vec2d(vec![
///     vec![1.0, 2.0, 3.0, 4.0],
///     vec![2.0, 3.0, 4.0, 5.0],
///     vec![3.0, 4.0, 5.0, 6.0],
///     vec![4.0, 5.0, 6.0, 7.0]
/// ]).unwrap();
/// 
/// let result = interp2d_bicubic(&x, &y, &z, 1.5, 1.5)?;
/// # Ok::<(), rustlab_numerical::NumericalError>(())
/// ```
pub fn interp2d_bicubic(x: &VectorF64, y: &VectorF64, z: &ArrayF64, xi: f64, yi: f64) -> Result<f64> {
    let interp = BicubicInterpolator::new(x.clone(), y.clone(), z.clone())?;
    interp.eval(xi, yi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::{vec64, VectorF64, ArrayF64, array64};
    use approx::assert_relative_eq;
    
    #[test]
    fn test_bilinear_basic() {
        // Simple 2x2 grid
        let x = VectorF64::from_slice(&[0.0, 1.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0]);
        let z = array64![[1.0, 2.0], [3.0, 4.0]];
        
        let interp = BilinearInterpolator::new(x, y, z).unwrap();
        
        // Test corner points
        assert_relative_eq!(interp.eval(0.0, 0.0).unwrap(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(interp.eval(1.0, 0.0).unwrap(), 3.0, epsilon = 1e-12);
        assert_relative_eq!(interp.eval(0.0, 1.0).unwrap(), 2.0, epsilon = 1e-12);
        assert_relative_eq!(interp.eval(1.0, 1.0).unwrap(), 4.0, epsilon = 1e-12);
        
        // Test center point
        assert_relative_eq!(interp.eval(0.5, 0.5).unwrap(), 2.5, epsilon = 1e-12);
    }
    
    #[test]
    fn test_bilinear_larger_grid() {
        let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0]);
        let z = array64![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        
        let interp = BilinearInterpolator::new(x, y, z).unwrap();
        
        // Test interpolation in first cell
        assert_relative_eq!(interp.eval(0.5, 0.5).unwrap(), 2.5, epsilon = 1e-12);
        
        // Test interpolation in second cell
        assert_relative_eq!(interp.eval(1.5, 0.5).unwrap(), 4.5, epsilon = 1e-12);
    }
    
    #[test]
    fn test_bicubic_basic() {
        // 4x4 grid for bicubic
        let x = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0, 2.0, 3.0]);
        let z = array64![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]];
        
        let interp = BicubicInterpolator::new(x, y, z).unwrap();
        
        // Test that it passes through grid points (within valid domain)
        assert_relative_eq!(interp.eval(1.0, 1.0).unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(interp.eval(2.0, 2.0).unwrap(), 5.0, epsilon = 1e-10);
        
        // Test smooth interpolation
        let result = interp.eval(1.5, 1.5).unwrap();
        assert!(result.is_finite());
        assert!(result > 3.0 && result < 5.0); // Should be between corner values
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let x = VectorF64::from_slice(&[0.0, 1.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0]);
        let z = array64![[1.0, 2.0]]; // Wrong dimensions
        
        assert!(BilinearInterpolator::new(x, y, z).is_err());
    }
    
    #[test]
    fn test_insufficient_data() {
        let x = VectorF64::from_slice(&[0.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0]);
        let z = array64![[1.0, 2.0]];
        
        assert!(BilinearInterpolator::new(x, y, z).is_err());
    }
    
    #[test]
    fn test_extrapolation_modes() {
        let x = VectorF64::from_slice(&[0.0, 1.0]);
        let y = VectorF64::from_slice(&[0.0, 1.0]);
        let z = array64![[1.0, 2.0], [3.0, 4.0]];
        
        // Test Error mode (default)
        let interp = BilinearInterpolator::new(x.clone(), y.clone(), z.clone()).unwrap();
        assert!(interp.eval(-1.0, 0.5).is_err());
        assert!(interp.eval(0.5, -1.0).is_err());
        
        // Test NaN mode
        let interp = BilinearInterpolator::new(x.clone(), y.clone(), z.clone())
            .unwrap()
            .with_extrapolation(ExtrapolationMode::NaN);
        assert!(interp.eval(-1.0, 0.5).unwrap().is_nan());
        assert!(interp.eval(0.5, -1.0).unwrap().is_nan());
        
        // Test Constant mode
        let interp = BilinearInterpolator::new(x.clone(), y.clone(), z.clone())
            .unwrap()
            .with_extrapolation(ExtrapolationMode::Constant);
        let result = interp.eval(-1.0, 0.5).unwrap();
        assert!(result.is_finite());
    }
    
    #[test]
    fn test_bicubic_insufficient_points() {
        let x = VectorF64::from_slice(&[0.0, 1.0, 2.0]); // Only 3 points, need 4
        let y = VectorF64::from_slice(&[0.0, 1.0, 2.0]);
        let z = array64![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];
        
        assert!(BicubicInterpolator::new(x, y, z).is_err());
    }
}