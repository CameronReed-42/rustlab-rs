//! Basic statistical operations for vectors with AI-optimized documentation
//! 
//! This module provides fundamental statistical functions directly on Vector types,
//! following the math-first philosophy of rustlab-math. All functions use numerically
//! stable algorithms and integrate with RustLab's zero-cost abstractions.
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::{VectorF64, statistics::BasicStatistics};
//! 
//! let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
//! let mean = data.mean();           // Arithmetic mean
//! let std_dev = data.std(None);     // Sample standard deviation  
//! let pop_var = data.var_pop();     // Population variance
//! 
//! // Chain with other operations
//! let normalized = (&data - mean) / std_dev;  // Z-score normalization
//! ```

use crate::{VectorF64, VectorF32};

/// Basic statistical operations for vectors with AI-optimized documentation
/// 
/// This trait provides essential statistical functions that are commonly used
/// in data analysis, machine learning, and scientific computing. All methods
/// use numerically stable algorithms where possible.
pub trait BasicStatistics<T> {
    /// Compute the arithmetic mean (average) of the vector
    /// 
    /// # Mathematical Specification
    /// For vector x ∈ ℝⁿ:
    /// μ = (1/n) × Σᵢ(xᵢ) for i = 1..n
    /// The sum of all elements divided by the count
    /// 
    /// # Complexity
    /// - Time: O(n) single pass through data
    /// - Space: O(1) constant memory
    /// 
    /// # For AI Code Generation
    /// - Returns sample mean (not population mean distinction)
    /// - Always returns same type as vector elements (f64 or f32)
    /// - Common uses: central tendency, normalization, feature scaling
    /// - Often combined with std() for z-score normalization
    /// - Numerically stable for typical datasets
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, statistics::BasicStatistics};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let avg = data.mean();  // Returns 3.0
    /// 
    /// // Feature normalization
    /// let features = VectorF64::from_slice(&[100.0, 200.0, 150.0]);
    /// let mean_val = features.mean();
    /// let centered = &features - mean_val;  // Center around zero
    /// ```
    /// 
    /// # Panics
    /// Panics if the vector is empty (cannot compute mean of zero elements)
    /// 
    /// # See Also
    /// - [`var`]: Compute variance
    /// - [`std`]: Compute standard deviation
    /// - [`median`]: Compute median (robust central tendency)
    fn mean(&self) -> T;
    
    /// Compute the variance with degrees of freedom (sample variance by default)
    /// 
    /// # Mathematical Specification
    /// For vector x ∈ ℝⁿ with mean μ:
    /// σ² = (1/(n-ddof)) × Σᵢ(xᵢ - μ)² for i = 1..n
    /// where ddof = delta degrees of freedom
    /// 
    /// # Complexity
    /// - Time: O(n) two passes (mean calculation + variance)
    /// - Space: O(1) constant memory
    /// 
    /// # For AI Code Generation
    /// - ddof=None defaults to 1 (sample variance, unbiased estimator)
    /// - ddof=Some(0) gives population variance (biased estimator)
    /// - ddof=Some(1) gives sample variance (same as default)
    /// - Common uses: measure of spread, feature scaling, outlier detection
    /// - Always non-negative result
    /// - Use sqrt() to get standard deviation
    /// 
    /// # Arguments
    /// * `ddof` - Delta degrees of freedom. None = 1 (sample variance)
    ///           Some(0) = population variance, Some(1) = sample variance
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, statistics::BasicStatistics};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let sample_var = data.var(None);      // Sample variance (ddof=1)
    /// let pop_var = data.var(Some(0));      // Population variance 
    /// let custom_var = data.var(Some(2));   // Custom degrees of freedom
    /// 
    /// // Feature scaling: normalize to unit variance
    /// let scaled = &data / sample_var.sqrt();
    /// ```
    /// 
    /// # Panics
    /// Panics if vector length ≤ ddof (need at least ddof+1 elements)
    /// 
    /// # See Also
    /// - [`var_pop`]: Population variance shortcut (ddof=0)
    /// - [`std`]: Standard deviation (square root of variance)
    /// - [`mean`]: Mean calculation (used internally)
    fn var(&self, ddof: Option<usize>) -> T;
    
    /// Compute the population variance (ddof = 0)
    /// 
    /// # Panics
    /// Panics if the vector is empty
    fn var_pop(&self) -> T;
    
    /// Compute the standard deviation with degrees of freedom
    /// 
    /// # Mathematical Specification
    /// For vector x ∈ ℝⁿ:
    /// σ = √(variance) = √((1/(n-ddof)) × Σᵢ(xᵢ - μ)²)
    /// Square root of the variance
    /// 
    /// # Complexity
    /// - Time: O(n) for variance calculation + O(1) for sqrt
    /// - Space: O(1) constant memory
    /// 
    /// # For AI Code Generation
    /// - ddof=None defaults to 1 (sample standard deviation)
    /// - ddof=Some(0) gives population standard deviation
    /// - Same units as original data (unlike variance)
    /// - Common uses: feature scaling, outlier detection, confidence intervals
    /// - Always non-negative result
    /// - More interpretable than variance (same units as data)
    /// 
    /// # Arguments
    /// * `ddof` - Delta degrees of freedom. None = 1 (sample std dev)
    ///           Some(0) = population std dev
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, statistics::BasicStatistics};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let sample_std = data.std(None);      // Sample standard deviation
    /// let pop_std = data.std(Some(0));      // Population standard deviation
    /// 
    /// // Z-score normalization (standardization)
    /// let mean_val = data.mean();
    /// let std_val = data.std(None);
    /// let z_scores = (&data - mean_val) / std_val;
    /// 
    /// // Outlier detection (values > 2 std devs from mean)
    /// let outlier_threshold = 2.0 * std_val;
    /// ```
    /// 
    /// # Panics
    /// Panics if vector length ≤ ddof (need at least ddof+1 elements)
    /// 
    /// # See Also
    /// - [`std_pop`]: Population standard deviation shortcut
    /// - [`var`]: Variance (std²)
    /// - [`mean`]: Mean for normalization
    fn std(&self, ddof: Option<usize>) -> T;
    
    /// Compute the population standard deviation (ddof = 0)
    /// 
    /// # Panics
    /// Panics if the vector is empty
    fn std_pop(&self) -> T;
}

// Implementation for VectorF64
impl BasicStatistics<f64> for VectorF64 {
    fn mean(&self) -> f64 {
        assert!(!self.is_empty(), "Cannot compute mean of empty vector");
        self.sum_elements() / self.len() as f64
    }
    
    fn var(&self, ddof: Option<usize>) -> f64 {
        let ddof = ddof.unwrap_or(1);
        let n = self.len();
        assert!(n > ddof, "Vector length ({}) must be greater than ddof ({})", n, ddof);
        
        let mean_val = self.mean();
        let sum_sq_diff: f64 = (0..n)
            .map(|i| {
                let diff = self.inner[i] - mean_val;
                diff * diff
            })
            .sum();
        
        sum_sq_diff / (n - ddof) as f64
    }
    
    fn var_pop(&self) -> f64 {
        self.var(Some(0))
    }
    
    fn std(&self, ddof: Option<usize>) -> f64 {
        self.var(ddof).sqrt()
    }
    
    fn std_pop(&self) -> f64 {
        self.var_pop().sqrt()
    }
}

// Implementation for VectorF32
impl BasicStatistics<f32> for VectorF32 {
    fn mean(&self) -> f32 {
        assert!(!self.is_empty(), "Cannot compute mean of empty vector");
        self.sum_elements() / self.len() as f32
    }
    
    fn var(&self, ddof: Option<usize>) -> f32 {
        let ddof = ddof.unwrap_or(1);
        let n = self.len();
        assert!(n > ddof, "Vector length ({}) must be greater than ddof ({})", n, ddof);
        
        let mean_val = self.mean();
        let sum_sq_diff: f32 = (0..n)
            .map(|i| {
                let diff = self.inner[i] - mean_val;
                diff * diff
            })
            .sum();
        
        sum_sq_diff / (n - ddof) as f32
    }
    
    fn var_pop(&self) -> f32 {
        self.var(Some(0))
    }
    
    fn std(&self, ddof: Option<usize>) -> f32 {
        self.var(ddof).sqrt()
    }
    
    fn std_pop(&self) -> f32 {
        self.var_pop().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mean_f64() {
        let v = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(v.mean(), 3.0);
    }
    
    #[test]
    fn test_var_f64() {
        let v = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // Sample variance with ddof=1
        assert!((v.var(None) - 2.5).abs() < 1e-10);
        // Population variance with ddof=0
        assert!((v.var_pop() - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_std_f64() {
        let v = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // Sample std with ddof=1
        assert!((v.std(None) - 2.5_f64.sqrt()).abs() < 1e-10);
        // Population std with ddof=0
        assert!((v.std_pop() - 2.0_f64.sqrt()).abs() < 1e-10);
    }
    
    #[test]
    #[should_panic(expected = "Cannot compute mean of empty vector")]
    fn test_mean_empty() {
        let v = VectorF64::from_slice(&[]);
        v.mean();
    }
    
    #[test]
    #[should_panic(expected = "Vector length (1) must be greater than ddof (1)")]
    fn test_var_insufficient_length() {
        let v = VectorF64::from_slice(&[1.0]);
        v.var(None);
    }
    
    #[test]
    fn test_mean_f32() {
        let v = VectorF32::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(v.mean(), 3.0);
    }
    
    #[test]
    fn test_var_f32() {
        let v = VectorF32::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // Sample variance with ddof=1
        assert!((v.var(None) - 2.5).abs() < 1e-6);
        // Population variance with ddof=0
        assert!((v.var_pop() - 2.0).abs() < 1e-6);
    }
}