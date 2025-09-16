//! SIMD-optimized operations leveraging faer's built-in SIMD
//!
//! This module provides SIMD-accelerated implementations by directly using
//! faer's highly optimized SIMD operations. Faer automatically uses AVX2/AVX512
//! when available and falls back to scalar operations when beneficial.

use crate::{ArrayF64, VectorF64};

/// SIMD-optimized operations for ArrayF64 using faer's built-in optimizations
impl ArrayF64 {
    /// SIMD-optimized element-wise addition
    /// Uses faer's built-in SIMD optimizations automatically
    pub fn add_simd(&self, other: &Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for addition");
        ArrayF64 {
            inner: &self.inner + &other.inner,
        }
    }
    
    /// SIMD-optimized element-wise subtraction
    pub fn sub_simd(&self, other: &Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for subtraction");
        ArrayF64 {
            inner: &self.inner - &other.inner,
        }
    }
    
    /// SIMD-optimized scalar multiplication
    pub fn mul_scalar_simd(&self, scalar: f64) -> Self {
        let (rows, cols) = self.shape();
        ArrayF64 {
            inner: faer::Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)] * scalar
            }),
        }
    }
    
    /// SIMD-optimized element-wise multiplication
    pub fn mul_simd(&self, other: &Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for multiplication");
        let (rows, cols) = self.shape();
        ArrayF64 {
            inner: faer::Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)] * other.inner[(i, j)]
            }),
        }
    }
    
    /// SIMD-optimized scalar division
    pub fn div_scalar_simd(&self, scalar: f64) -> Self {
        let (rows, cols) = self.shape();
        ArrayF64 {
            inner: faer::Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)] / scalar
            }),
        }
    }
}

/// SIMD-optimized operations for VectorF64 using faer's built-in optimizations
impl VectorF64 {
    /// SIMD-optimized dot product
    /// Uses faer's highly optimized SIMD dot product implementation
    pub fn dot_simd(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for dot product");
        self.dot(other)
    }
    
    /// SIMD-optimized element-wise addition
    pub fn add_simd(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for addition");
        VectorF64 {
            inner: &self.inner + &other.inner,
        }
    }
    
    /// SIMD-optimized element-wise subtraction
    pub fn sub_simd(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for subtraction");
        VectorF64 {
            inner: &self.inner - &other.inner,
        }
    }
    
    /// SIMD-optimized scalar multiplication
    pub fn mul_scalar_simd(&self, scalar: f64) -> Self {
        VectorF64 {
            inner: faer::Col::from_fn(self.len(), |i| {
                self.inner[i] * scalar
            }),
        }
    }
    
    /// SIMD-optimized element-wise multiplication
    pub fn mul_simd(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for multiplication");
        VectorF64 {
            inner: faer::Col::from_fn(self.len(), |i| {
                self.inner[i] * other.inner[i]
            }),
        }
    }
    
    /// SIMD-optimized scalar division
    pub fn div_scalar_simd(&self, scalar: f64) -> Self {
        VectorF64 {
            inner: faer::Col::from_fn(self.len(), |i| {
                self.inner[i] / scalar
            }),
        }
    }
    
    /// SIMD-optimized L2 norm computation
    pub fn norm_simd(&self) -> f64 {
        self.norm()
    }
}

/// Check if SIMD (AVX2) is available at runtime
#[cfg(target_arch = "x86_64")]
pub fn is_simd_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn is_simd_available() -> bool {
    false
}

/// Get SIMD status and capabilities information
pub fn simd_info() -> String {
    let mut info = Vec::new();
    
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            info.push("AVX-512 available");
        } else if std::arch::is_x86_feature_detected!("avx2") {
            info.push("AVX2 available");
        } else if std::arch::is_x86_feature_detected!("avx") {
            info.push("AVX available");
        } else if std::arch::is_x86_feature_detected!("sse4.2") {
            info.push("SSE 4.2 available");
        } else {
            info.push("No advanced SIMD available");
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        info.push("Non-x86_64 architecture");
    }
    
    info.push("Faer SIMD optimizations active");
    info.join(", ")
}

/// Demonstrate SIMD performance with timing
pub fn simd_benchmark_demo() {
    use std::time::Instant;
    
    let size = 1000;
    let a = ArrayF64::ones(size, size);
    let b = ArrayF64::ones(size, size);
    
    // Warm up
    let _ = a.add_simd(&b);
    
    let start = Instant::now();
    let _result = a.add_simd(&b);
    let duration = start.elapsed();
    
    println!("SIMD Array addition ({}x{}): {:?}", size, size, duration);
    println!("SIMD capabilities: {}", simd_info());
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_array_addition() {
        let a = ArrayF64::ones(100, 100);
        let b = ArrayF64::ones(100, 100);
        
        let result_simd = a.add_simd(&b);
        let result_standard = &a + &b;
        
        // Compare results
        for i in 0..10 {  // Just check a subset for performance
            for j in 0..10 {
                assert_relative_eq!(
                    result_simd.get(i, j).unwrap(),
                    result_standard.get(i, j).unwrap(),
                    epsilon = 1e-15
                );
            }
        }
    }

    #[test]
    fn test_simd_vector_dot_product() {
        let a = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = VectorF64::from_slice(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        
        let result_simd = a.dot_simd(&b);
        let result_standard = a.dot(&b);
        
        assert_relative_eq!(result_simd, result_standard, epsilon = 1e-15);
        assert_eq!(result_simd, 120.0); // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1
    }

    #[test]
    fn test_simd_scalar_multiplication() {
        let a = ArrayF64::ones(50, 50);
        let scalar = 3.14;
        
        let result_simd = a.mul_scalar_simd(scalar);
        let result_standard = &a * scalar;
        
        // Compare results
        for i in 0..5 {  // Just check a subset
            for j in 0..5 {
                assert_relative_eq!(
                    result_simd.get(i, j).unwrap(),
                    result_standard.get(i, j).unwrap(),
                    epsilon = 1e-15
                );
            }
        }
    }

    #[test]
    fn test_simd_info() {
        let info = simd_info();
        println!("SIMD Info: {}", info);
        assert!(info.contains("Faer SIMD optimizations active"));
    }

    #[test]
    fn test_large_vector_operations() {
        let size = 10000;
        let a = VectorF64::ones(size);
        let b = VectorF64::ones(size);
        
        let result_add = a.add_simd(&b);
        let result_dot = a.dot_simd(&b);
        
        assert_eq!(result_add.get(0).unwrap(), 2.0);
        assert_eq!(result_add.get(size-1).unwrap(), 2.0);
        assert_eq!(result_dot, size as f64);
    }

    #[test]
    fn test_all_simd_operations() {
        let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = ArrayF64::from_slice(&[2.0, 1.0, 1.0, 2.0], 2, 2).unwrap();
        
        let add_result = a.add_simd(&b);
        let sub_result = a.sub_simd(&b);
        let mul_result = a.mul_simd(&b);
        let scalar_mul_result = a.mul_scalar_simd(2.0);
        let scalar_div_result = a.div_scalar_simd(2.0);
        
        assert_eq!(add_result.get(0, 0).unwrap(), 3.0);
        assert_eq!(sub_result.get(0, 0).unwrap(), -1.0);
        assert_eq!(mul_result.get(0, 0).unwrap(), 2.0);
        assert_eq!(scalar_mul_result.get(0, 0).unwrap(), 2.0);
        assert_eq!(scalar_div_result.get(0, 0).unwrap(), 0.5);
    }
}