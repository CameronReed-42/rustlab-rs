//! Utility functions for interpolation

use rustlab_math::VectorF64;
use crate::{Result, NumericalError};

/// Check that x values are strictly increasing
pub fn check_monotonic(x: &VectorF64, method: &'static str) -> Result<()> {
    if x.len() < 2 {
        return Ok(());
    }
    
    for i in 1..x.len() {
        // Using rustlab-math's get() method which returns Option<T>
        let xi = x.get(i).ok_or_else(|| 
            NumericalError::InvalidParameter(format!("Index {} out of bounds", i))
        )?;
        let xi_prev = x.get(i-1).ok_or_else(|| 
            NumericalError::InvalidParameter(format!("Index {} out of bounds", i-1))
        )?;
        
        if xi <= xi_prev {
            return Err(NumericalError::NotMonotonic { method });
        }
    }
    Ok(())
}

/// Check that x and y arrays have the same length
pub fn check_dimensions(x: &VectorF64, y: &VectorF64) -> Result<()> {
    if x.len() != y.len() {
        return Err(NumericalError::DimensionMismatch(x.len(), y.len()));
    }
    Ok(())
}

/// Find the interval [x[i], x[i+1]] that contains the target value
/// Returns the index i such that x[i] <= target < x[i+1]
/// For target >= x[n-1], returns n-2
pub fn find_interval(x: &VectorF64, target: f64) -> Option<usize> {
    let n = x.len();
    if n < 2 {
        return None;
    }
    
    // Using rustlab-math's get() which returns Option<T>
    let x_min = x.get(0)?;
    let x_max = x.get(n-1)?;
    if target < x_min || target > x_max {
        return None;
    }
    
    // Binary search for the interval
    let mut left = 0;
    let mut right = n - 1;
    
    while right - left > 1 {
        let mid = (left + right) / 2;
        let x_mid = x.get(mid)?;
        if target < x_mid {
            right = mid;
        } else {
            left = mid;
        }
    }
    
    Some(left)
}