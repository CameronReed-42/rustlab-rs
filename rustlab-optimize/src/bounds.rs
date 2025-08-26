//! Parameter bounds transformation system
//!
//! This module provides transformation functions to convert bounded optimization
//! problems into unbounded ones, which can then be solved by standard algorithms.
//!
//! # Transformation Methods
//!
//! ## Box Bounds: x ∈ [a, b]
//! Uses logit transformation to map bounded parameters to unbounded space:
//! - Forward: y = ln((x-a)/(b-x))  
//! - Inverse: x = a + (b-a)/(1 + exp(-y))
//!
//! ## One-sided Bounds
//! - Lower bound x ≥ a: y = ln(x - a)
//! - Upper bound x ≤ b: y = ln(b - x)
//!
//! # Integration with Algorithms
//!
//! The transformation is transparent to optimization algorithms:
//! 1. Transform bounded initial guess to unbounded space
//! 2. Algorithm optimizes in unbounded space
//! 3. Transform result back to bounded space for evaluation
//! 4. Transform gradients/Jacobians back to unbounded space

use rustlab_math::VectorF64;
use crate::core::{Result, Error};

/// Parameter bounds specification
#[derive(Debug, Clone)]
pub struct Bounds {
    /// Lower bounds for each parameter (can be -∞)
    pub lower: VectorF64,
    
    /// Upper bounds for each parameter (can be +∞) 
    pub upper: VectorF64,
}

impl Bounds {
    /// Create new bounds from lower and upper vectors
    pub fn new(lower: VectorF64, upper: VectorF64) -> Result<Self> {
        if lower.len() != upper.len() {
            return Err(Error::dimension_mismatch(lower.len(), upper.len()));
        }
        
        // Validate bounds
        for i in 0..lower.len() {
            if lower[i] >= upper[i] && upper[i].is_finite() && lower[i].is_finite() {
                return Err(Error::bounds_violation(i, lower[i], lower[i], upper[i]));
            }
        }
        
        Ok(Self { lower, upper })
    }
    
    /// Check if parameter is bounded on both sides
    pub fn is_box_bounded(&self, i: usize) -> bool {
        self.lower[i].is_finite() && self.upper[i].is_finite()
    }
    
    /// Check if parameter has only lower bound
    pub fn is_lower_bounded(&self, i: usize) -> bool {
        self.lower[i].is_finite() && !self.upper[i].is_finite()
    }
    
    /// Check if parameter has only upper bound  
    pub fn is_upper_bounded(&self, i: usize) -> bool {
        !self.lower[i].is_finite() && self.upper[i].is_finite()
    }
    
    /// Check if parameter is unbounded
    pub fn is_unbounded(&self, i: usize) -> bool {
        !self.lower[i].is_finite() && !self.upper[i].is_finite()
    }
    
    /// Get number of parameters
    pub fn len(&self) -> usize {
        self.lower.len()
    }
    
    /// Check if bounds are empty
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }
}

/// Bounds transformer handles parameter space transformations
pub struct BoundsTransformer {
    bounds: Bounds,
}

impl BoundsTransformer {
    /// Create new bounds transformer
    pub fn new(bounds: Bounds) -> Self {
        Self { bounds }
    }
    
    /// Transform bounded parameters to unbounded space
    /// 
    /// # Mathematical Transformations
    /// - Box bounds [a,b]: y = ln((x-a)/(b-x))
    /// - Lower bound x ≥ a: y = ln(x - a)  
    /// - Upper bound x ≤ b: y = ln(b - x)
    /// - Unbounded: y = x (identity)
    /// 
    /// # Arguments
    /// * `bounded_params` - Parameters in bounded space
    /// 
    /// # Returns
    /// Parameters transformed to unbounded space
    pub fn to_unbounded(&self, bounded_params: &VectorF64) -> Result<VectorF64> {
        if bounded_params.len() != self.bounds.len() {
            return Err(Error::dimension_mismatch(bounded_params.len(), self.bounds.len()));
        }
        
        let mut unbounded = VectorF64::zeros(bounded_params.len());
        
        for i in 0..bounded_params.len() {
            let x = bounded_params[i];
            let a = self.bounds.lower[i];
            let b = self.bounds.upper[i];
            
            let y = if self.bounds.is_box_bounded(i) {
                // Box bounds: y = ln((x-a)/(b-x))
                if x <= a || x >= b {
                    return Err(Error::bounds_violation(i, x, a, b));
                }
                ((x - a) / (b - x)).ln()
                
            } else if self.bounds.is_lower_bounded(i) {
                // Lower bound: y = ln(x - a)
                if x <= a {
                    return Err(Error::bounds_violation(i, x, a, f64::INFINITY));
                }
                (x - a).ln()
                
            } else if self.bounds.is_upper_bounded(i) {
                // Upper bound: y = ln(b - x)
                if x >= b {
                    return Err(Error::bounds_violation(i, x, f64::NEG_INFINITY, b));
                }
                (b - x).ln()
                
            } else {
                // Unbounded: y = x
                x
            };
            
            unbounded[i] = y;
        }
        
        Ok(unbounded)
    }
    
    /// Transform unbounded parameters back to bounded space
    /// 
    /// # Mathematical Transformations  
    /// - Box bounds: x = a + (b-a)/(1 + exp(-y))
    /// - Lower bound: x = a + exp(y)
    /// - Upper bound: x = b - exp(y)  
    /// - Unbounded: x = y (identity)
    /// 
    /// # Arguments
    /// * `unbounded_params` - Parameters in unbounded space
    /// 
    /// # Returns
    /// Parameters transformed back to bounded space
    pub fn to_bounded(&self, unbounded_params: &VectorF64) -> Result<VectorF64> {
        if unbounded_params.len() != self.bounds.len() {
            return Err(Error::dimension_mismatch(unbounded_params.len(), self.bounds.len()));
        }
        
        let mut bounded = VectorF64::zeros(unbounded_params.len());
        
        for i in 0..unbounded_params.len() {
            let y = unbounded_params[i];
            let a = self.bounds.lower[i];
            let b = self.bounds.upper[i];
            
            let x = if self.bounds.is_box_bounded(i) {
                // Box bounds: x = a + (b-a)/(1 + exp(-y))
                let exp_neg_y = (-y).exp();
                a + (b - a) / (1.0 + exp_neg_y)
                
            } else if self.bounds.is_lower_bounded(i) {
                // Lower bound: x = a + exp(y)
                a + y.exp()
                
            } else if self.bounds.is_upper_bounded(i) {
                // Upper bound: x = b - exp(y)  
                b - y.exp()
                
            } else {
                // Unbounded: x = y
                y
            };
            
            bounded[i] = x;
        }
        
        Ok(bounded)
    }
    
    /// Transform gradient from bounded to unbounded space
    /// 
    /// Applies chain rule to transform gradients:
    /// ∂f/∂y = (∂f/∂x) * (∂x/∂y)
    /// 
    /// # Arguments  
    /// * `bounded_gradient` - Gradient in bounded space
    /// * `unbounded_params` - Current parameters in unbounded space
    /// 
    /// # Returns
    /// Gradient transformed to unbounded space
    pub fn transform_gradient(&self, bounded_gradient: &VectorF64, unbounded_params: &VectorF64) -> Result<VectorF64> {
        if bounded_gradient.len() != self.bounds.len() || unbounded_params.len() != self.bounds.len() {
            return Err(Error::dimension_mismatch(bounded_gradient.len(), self.bounds.len()));
        }
        
        let mut unbounded_gradient = VectorF64::zeros(bounded_gradient.len());
        
        for i in 0..bounded_gradient.len() {
            let grad_x = bounded_gradient[i];
            let y = unbounded_params[i];
            let a = self.bounds.lower[i];
            let b = self.bounds.upper[i];
            
            // Compute dx/dy (Jacobian of inverse transformation)
            let dx_dy = if self.bounds.is_box_bounded(i) {
                // dx/dy = (b-a) * exp(y) / (1 + exp(y))^2
                let exp_y = y.exp();
                let denom = (1.0 + exp_y).powi(2);
                (b - a) * exp_y / denom
                
            } else if self.bounds.is_lower_bounded(i) {
                // dx/dy = exp(y)
                y.exp()
                
            } else if self.bounds.is_upper_bounded(i) {
                // dx/dy = -exp(y)
                -y.exp()
                
            } else {
                // dx/dy = 1 (unbounded)
                1.0
            };
            
            // Chain rule: ∂f/∂y = (∂f/∂x) * (∂x/∂y)
            unbounded_gradient[i] = grad_x * dx_dy;
        }
        
        Ok(unbounded_gradient)
    }
    
    /// Check if any parameters are actually bounded
    pub fn has_bounds(&self) -> bool {
        for i in 0..self.bounds.len() {
            if !self.bounds.is_unbounded(i) {
                return true;
            }
        }
        false
    }
    
    /// Get bounds reference
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::vec64;
    
    #[test]
    fn test_box_bounds_transformation() {
        // Test x ∈ [1, 3] 
        let lower = vec64![1.0];
        let upper = vec64![3.0];
        let bounds = Bounds::new(lower, upper).unwrap();
        let transformer = BoundsTransformer::new(bounds);
        
        // Test midpoint x = 2 should map to y = 0
        let bounded = vec64![2.0];
        let unbounded = transformer.to_unbounded(&bounded).unwrap();
        assert!((unbounded[0] - 0.0).abs() < 1e-10);
        
        // Test inverse transformation
        let recovered = transformer.to_bounded(&unbounded).unwrap();
        assert!((recovered[0] - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_lower_bound_transformation() {
        // Test x ≥ 0
        let lower = vec64![0.0];
        let upper = vec64![f64::INFINITY];
        let bounds = Bounds::new(lower, upper).unwrap();
        let transformer = BoundsTransformer::new(bounds);
        
        // Test x = 1 should map to y = ln(1) = 0
        let bounded = vec64![1.0];
        let unbounded = transformer.to_unbounded(&bounded).unwrap();
        assert!((unbounded[0] - 0.0).abs() < 1e-10);
        
        // Test inverse transformation
        let recovered = transformer.to_bounded(&unbounded).unwrap();
        assert!((recovered[0] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_upper_bound_transformation() {
        // Test x ≤ 5
        let lower = vec64![f64::NEG_INFINITY];
        let upper = vec64![5.0];
        let bounds = Bounds::new(lower, upper).unwrap();
        let transformer = BoundsTransformer::new(bounds);
        
        // Test x = 4 should map to y = ln(5-4) = ln(1) = 0
        let bounded = vec64![4.0];
        let unbounded = transformer.to_unbounded(&bounded).unwrap();
        assert!((unbounded[0] - 0.0).abs() < 1e-10);
        
        // Test inverse transformation
        let recovered = transformer.to_bounded(&unbounded).unwrap();
        assert!((recovered[0] - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_gradient_transformation() {
        // Test box bounds x ∈ [0, 2]
        let lower = vec64![0.0];
        let upper = vec64![2.0];
        let bounds = Bounds::new(lower, upper).unwrap();
        let transformer = BoundsTransformer::new(bounds);
        
        // At x = 1 (midpoint), y = 0
        let unbounded_params = vec64![0.0];
        let bounded_gradient = vec64![1.0];  // df/dx = 1
        
        let unbounded_gradient = transformer.transform_gradient(&bounded_gradient, &unbounded_params).unwrap();
        
        // At y = 0: dx/dy = (2-0) * exp(0) / (1 + exp(0))^2 = 2 * 1 / 4 = 0.5
        // So df/dy = df/dx * dx/dy = 1 * 0.5 = 0.5
        assert!((unbounded_gradient[0] - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_bounds_validation() {
        // Test invalid bounds (lower > upper)
        let lower = vec64![5.0];
        let upper = vec64![1.0];
        assert!(Bounds::new(lower, upper).is_err());
    }
    
    #[test]
    fn test_bounds_violation() {
        let lower = vec64![1.0];
        let upper = vec64![3.0];
        let bounds = Bounds::new(lower, upper).unwrap();
        let transformer = BoundsTransformer::new(bounds);
        
        // Test parameter outside bounds
        let invalid = vec64![0.5];  // Below lower bound
        assert!(transformer.to_unbounded(&invalid).is_err());
        
        let invalid = vec64![3.5];  // Above upper bound  
        assert!(transformer.to_unbounded(&invalid).is_err());
    }
}