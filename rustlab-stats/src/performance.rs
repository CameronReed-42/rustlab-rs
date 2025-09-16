//! Performance optimization utilities and benchmarking
//!
//! This module provides performance-optimized versions of statistical operations,
//! zero-copy algorithms, parallel implementations, and benchmarking utilities.

use rustlab_math::{VectorF64, BasicStatistics};

#[cfg(feature = "parallel")]
use rustlab_math::{ArrayF64, reductions::Axis};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use crate::normalization::Normalization;

/// Performance thresholds for determining when to use parallel processing
pub mod thresholds {
    /// Minimum size for parallel vector operations (empirically determined)
    pub const PARALLEL_VECTOR_THRESHOLD: usize = 10_000;
    
    /// Minimum size for parallel array operations
    pub const PARALLEL_ARRAY_THRESHOLD: usize = 50_000;
    
    /// Minimum size for parallel correlation computation
    pub const PARALLEL_CORRELATION_THRESHOLD: usize = 5_000;
    
    /// Chunk size for parallel processing (balance between overhead and parallelism)
    pub const PARALLEL_CHUNK_SIZE: usize = 1_000;
}

/// SIMD-optimized operations for supported platforms
pub mod simd {
    /// SIMD-optimized sum operation
    #[inline]
    pub fn sum_f64_simd(data: &[f64]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { sum_f64_avx2(data) };
            } else if is_x86_feature_detected!("sse2") {
                return unsafe { sum_f64_sse2(data) };
            }
        }
        
        // Fallback to standard implementation
        data.iter().sum()
    }
    
    /// SIMD-optimized dot product
    #[inline]
    pub fn dot_product_f64_simd(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { dot_product_f64_avx2(a, b) };
            } else if is_x86_feature_detected!("sse2") {
                return unsafe { dot_product_f64_sse2(a, b) };
            }
        }
        
        // Fallback to standard implementation
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
    
    /// SIMD-optimized variance computation
    #[inline]
    pub fn variance_f64_simd(data: &[f64], mean: f64) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { variance_f64_avx2(data, mean) };
            } else if is_x86_feature_detected!("sse2") {
                return unsafe { variance_f64_sse2(data, mean) };
            }
        }
        
        // Fallback to standard implementation
        data.iter().map(|&x| (x - mean).powi(2)).sum()
    }
    
    // Platform-specific SIMD implementations
    #[cfg(target_arch = "x86_64")]
    mod x86_64_simd {
        use std::arch::x86_64::*;
        
        #[target_feature(enable = "avx2")]
        /// AVX2-optimized sum for f64 slices
        pub unsafe fn sum_f64_avx2(data: &[f64]) -> f64 {
            let mut sum = _mm256_setzero_pd();
            let chunks = data.chunks_exact(4);
            let remainder = chunks.remainder();
            
            for chunk in chunks {
                let vals = _mm256_loadu_pd(chunk.as_ptr());
                sum = _mm256_add_pd(sum, vals);
            }
            
            // Horizontal sum of the AVX register
            let hi = _mm256_extractf128_pd(sum, 1);
            let lo = _mm256_castpd256_pd128(sum);
            let sum128 = _mm_add_pd(hi, lo);
            let sum_final = _mm_hadd_pd(sum128, sum128);
            
            let mut result = _mm_cvtsd_f64(sum_final);
            
            // Add remainder
            result += remainder.iter().sum::<f64>();
            result
        }
        
        #[target_feature(enable = "sse2")]
        /// SSE2-optimized sum for f64 slices
        pub unsafe fn sum_f64_sse2(data: &[f64]) -> f64 {
            let mut sum = _mm_setzero_pd();
            let chunks = data.chunks_exact(2);
            let remainder = chunks.remainder();
            
            for chunk in chunks {
                let vals = _mm_loadu_pd(chunk.as_ptr());
                sum = _mm_add_pd(sum, vals);
            }
            
            let sum_final = _mm_hadd_pd(sum, sum);
            let mut result = _mm_cvtsd_f64(sum_final);
            
            // Add remainder
            result += remainder.iter().sum::<f64>();
            result
        }
        
        #[target_feature(enable = "avx2")]
        /// AVX2-optimized dot product for f64 slices
        pub unsafe fn dot_product_f64_avx2(a: &[f64], b: &[f64]) -> f64 {
            let mut sum = _mm256_setzero_pd();
            let len = a.len().min(b.len());
            let chunks_a = a[..len].chunks_exact(4);
            let chunks_b = b[..len].chunks_exact(4);
            let remainder_a = chunks_a.remainder();
            let remainder_b = chunks_b.remainder();
            
            for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
                let vals_a = _mm256_loadu_pd(chunk_a.as_ptr());
                let vals_b = _mm256_loadu_pd(chunk_b.as_ptr());
                let prod = _mm256_mul_pd(vals_a, vals_b);
                sum = _mm256_add_pd(sum, prod);
            }
            
            // Horizontal sum
            let hi = _mm256_extractf128_pd(sum, 1);
            let lo = _mm256_castpd256_pd128(sum);
            let sum128 = _mm_add_pd(hi, lo);
            let sum_final = _mm_hadd_pd(sum128, sum128);
            
            let mut result = _mm_cvtsd_f64(sum_final);
            
            // Add remainder
            result += remainder_a.iter().zip(remainder_b.iter()).map(|(&x, &y)| x * y).sum::<f64>();
            result
        }
        
        #[target_feature(enable = "sse2")]
        /// SSE2-optimized dot product for f64 slices
        pub unsafe fn dot_product_f64_sse2(a: &[f64], b: &[f64]) -> f64 {
            let mut sum = _mm_setzero_pd();
            let len = a.len().min(b.len());
            let chunks_a = a[..len].chunks_exact(2);
            let chunks_b = b[..len].chunks_exact(2);
            let remainder_a = chunks_a.remainder();
            let remainder_b = chunks_b.remainder();
            
            for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
                let vals_a = _mm_loadu_pd(chunk_a.as_ptr());
                let vals_b = _mm_loadu_pd(chunk_b.as_ptr());
                let prod = _mm_mul_pd(vals_a, vals_b);
                sum = _mm_add_pd(sum, prod);
            }
            
            let sum_final = _mm_hadd_pd(sum, sum);
            let mut result = _mm_cvtsd_f64(sum_final);
            
            // Add remainder
            result += remainder_a.iter().zip(remainder_b.iter()).map(|(&x, &y)| x * y).sum::<f64>();
            result
        }
        
        #[target_feature(enable = "avx2")]
        /// AVX2-optimized variance calculation for f64 slices
        pub unsafe fn variance_f64_avx2(data: &[f64], mean: f64) -> f64 {
            let mean_vec = _mm256_set1_pd(mean);
            let mut sum = _mm256_setzero_pd();
            let chunks = data.chunks_exact(4);
            let remainder = chunks.remainder();
            
            for chunk in chunks {
                let vals = _mm256_loadu_pd(chunk.as_ptr());
                let diff = _mm256_sub_pd(vals, mean_vec);
                let squared = _mm256_mul_pd(diff, diff);
                sum = _mm256_add_pd(sum, squared);
            }
            
            // Horizontal sum
            let hi = _mm256_extractf128_pd(sum, 1);
            let lo = _mm256_castpd256_pd128(sum);
            let sum128 = _mm_add_pd(hi, lo);
            let sum_final = _mm_hadd_pd(sum128, sum128);
            
            let mut result = _mm_cvtsd_f64(sum_final);
            
            // Add remainder
            result += remainder.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
            result
        }
        
        #[target_feature(enable = "sse2")]
        /// SSE2-optimized variance calculation for f64 slices
        pub unsafe fn variance_f64_sse2(data: &[f64], mean: f64) -> f64 {
            let mean_vec = _mm_set1_pd(mean);
            let mut sum = _mm_setzero_pd();
            let chunks = data.chunks_exact(2);
            let remainder = chunks.remainder();
            
            for chunk in chunks {
                let vals = _mm_loadu_pd(chunk.as_ptr());
                let diff = _mm_sub_pd(vals, mean_vec);
                let squared = _mm_mul_pd(diff, diff);
                sum = _mm_add_pd(sum, squared);
            }
            
            let sum_final = _mm_hadd_pd(sum, sum);
            let mut result = _mm_cvtsd_f64(sum_final);
            
            // Add remainder
            result += remainder.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
            result
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    pub use x86_64_simd::*;
}

/// Adaptive performance trait that chooses the best implementation based on data size and available features
pub trait AdaptiveStats<T> {
    /// Compute mean using the most efficient method available
    fn mean_adaptive(&self) -> T;
    
    /// Compute variance using the most efficient method available
    fn var_adaptive(&self, ddof: Option<usize>) -> T;
    
    /// Compute correlation using the most efficient method available
    fn correlation_adaptive(&self, other: &Self) -> T;
}

/// Trait for zero-copy statistical operations that work with slices
pub trait ZeroCopyStats<T> {
    /// Compute mean without data copying using slice
    fn mean_slice(data: &[T]) -> T;
    
    /// Compute variance without data copying using slice
    fn var_slice(data: &[T], ddof: usize) -> T;
    
    /// Compute standard deviation without data copying using slice
    fn std_slice(data: &[T], ddof: usize) -> T;
    
    /// Find min and max in single pass without copying
    fn minmax_slice(data: &[T]) -> (T, T);
    
    /// Compute quantile without full sorting (using quickselect)
    fn quantile_slice_fast(data: &mut [T], q: f64) -> T;
}

/// Trait for parallel statistical operations
#[cfg(feature = "parallel")]
pub trait ParallelStats<T> {
    /// Parallel mean computation
    fn mean_parallel(&self) -> T;
    
    /// Parallel variance computation
    fn var_parallel(&self, ddof: Option<usize>) -> T;
    
    /// Parallel correlation computation
    fn correlation_parallel(&self, other: &Self) -> T;
    
    /// Parallel array operations along axis
    fn mean_axis_parallel(&self, axis: Axis) -> Self;
    
    /// Parallel normalization
    fn zscore_parallel(&self, ddof: Option<usize>) -> Self;
}

/// Streaming statistics for large datasets that don't fit in memory
#[derive(Debug, Clone, Copy)]
pub struct StreamingStats<T> {
    count: usize,
    mean: T,
    m2: T,  // For variance calculation using Welford's algorithm
    min: T,
    max: T,
}

impl<T> StreamingStats<T> 
where 
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + 
       std::ops::Mul<Output = T> + std::ops::Div<Output = T> + From<f64> + Into<f64>
{
    /// Create a new streaming statistics accumulator
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: T::from(0.0),
            m2: T::from(0.0),
            min: T::from(f64::INFINITY),
            max: T::from(f64::NEG_INFINITY),
        }
    }
    
    /// Add a new value to the streaming statistics
    pub fn update(&mut self, value: T) {
        self.count += 1;
        
        // Update min and max
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        
        // Welford's online algorithm for mean and variance
        let delta = value - self.mean;
        self.mean = self.mean + delta / T::from(self.count as f64);
        let delta2 = value - self.mean;
        self.m2 = self.m2 + delta * delta2;
    }
    
    /// Get the current count
    pub fn count(&self) -> usize {
        self.count
    }
    
    /// Get the current mean
    pub fn mean(&self) -> T {
        self.mean
    }
    
    /// Get the current sample variance
    pub fn variance(&self) -> T {
        if self.count < 2 {
            T::from(0.0)
        } else {
            self.m2 / T::from((self.count - 1) as f64)
        }
    }
    
    /// Get the current population variance
    pub fn variance_pop(&self) -> T {
        if self.count == 0 {
            T::from(0.0)
        } else {
            self.m2 / T::from(self.count as f64)
        }
    }
    
    /// Get the current standard deviation
    pub fn std_dev(&self) -> T {
        T::from(self.variance().into().sqrt())
    }
    
    /// Get min and max values
    pub fn minmax(&self) -> (T, T) {
        (self.min, self.max)
    }
    
    /// Merge with another streaming statistics accumulator
    pub fn merge(&mut self, other: &StreamingStats<T>) {
        if other.count == 0 {
            return;
        }
        
        if self.count == 0 {
            *self = *other;
            return;
        }
        
        let total_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean = self.mean + delta * T::from(other.count as f64) / T::from(total_count as f64);
        
        // Update variance using parallel algorithm
        let new_m2 = self.m2 + other.m2 + 
                     delta * delta * T::from(self.count as f64) * T::from(other.count as f64) / T::from(total_count as f64);
        
        self.count = total_count;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.min = if self.min < other.min { self.min } else { other.min };
        self.max = if self.max > other.max { self.max } else { other.max };
    }
}

impl<T> Default for StreamingStats<T>
where 
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + 
       std::ops::Mul<Output = T> + std::ops::Div<Output = T> + From<f64> + Into<f64>
{
    fn default() -> Self {
        Self::new()
    }
}

// Implementation for f64 zero-copy operations
impl ZeroCopyStats<f64> for f64 {
    fn mean_slice(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }
    
    fn var_slice(data: &[f64], ddof: usize) -> f64 {
        if data.len() <= ddof {
            return 0.0;
        }
        
        let mean = Self::mean_slice(data);
        let sum_sq_diff: f64 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        
        sum_sq_diff / (data.len() - ddof) as f64
    }
    
    fn std_slice(data: &[f64], ddof: usize) -> f64 {
        Self::var_slice(data, ddof).sqrt()
    }
    
    fn minmax_slice(data: &[f64]) -> (f64, f64) {
        if data.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        
        let mut min = data[0];
        let mut max = data[0];
        
        for &value in &data[1..] {
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }
        
        (min, max)
    }
    
    fn quantile_slice_fast(data: &mut [f64], q: f64) -> f64 {
        if data.is_empty() {
            return f64::NAN;
        }
        
        if q < 0.0 || q > 1.0 {
            panic!("Quantile must be between 0 and 1");
        }
        
        if q == 0.0 {
            return *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        }
        
        if q == 1.0 {
            return *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        }
        
        // Use quickselect algorithm for O(n) average case
        let index = ((data.len() - 1) as f64 * q) as usize;
        quickselect(data, index)
    }
}

// Implementation for f32 zero-copy operations
impl ZeroCopyStats<f32> for f32 {
    fn mean_slice(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f32>() / data.len() as f32
    }
    
    fn var_slice(data: &[f32], ddof: usize) -> f32 {
        if data.len() <= ddof {
            return 0.0;
        }
        
        let mean = Self::mean_slice(data);
        let sum_sq_diff: f32 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        
        sum_sq_diff / (data.len() - ddof) as f32
    }
    
    fn std_slice(data: &[f32], ddof: usize) -> f32 {
        Self::var_slice(data, ddof).sqrt()
    }
    
    fn minmax_slice(data: &[f32]) -> (f32, f32) {
        if data.is_empty() {
            return (f32::NAN, f32::NAN);
        }
        
        let mut min = data[0];
        let mut max = data[0];
        
        for &value in &data[1..] {
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }
        
        (min, max)
    }
    
    fn quantile_slice_fast(data: &mut [f32], q: f64) -> f32 {
        if data.is_empty() {
            return f32::NAN;
        }
        
        if q < 0.0 || q > 1.0 {
            panic!("Quantile must be between 0 and 1");
        }
        
        if q == 0.0 {
            return *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        }
        
        if q == 1.0 {
            return *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        }
        
        let index = ((data.len() - 1) as f64 * q) as usize;
        quickselect_f32(data, index)
    }
}

// Quickselect algorithm for fast quantile computation
fn quickselect(data: &mut [f64], k: usize) -> f64 {
    if data.len() == 1 {
        return data[0];
    }
    
    let pivot_index = partition(data);
    
    if k == pivot_index {
        data[k]
    } else if k < pivot_index {
        quickselect(&mut data[..pivot_index], k)
    } else {
        quickselect(&mut data[pivot_index + 1..], k - pivot_index - 1)
    }
}

fn quickselect_f32(data: &mut [f32], k: usize) -> f32 {
    if data.len() == 1 {
        return data[0];
    }
    
    let pivot_index = partition_f32(data);
    
    if k == pivot_index {
        data[k]
    } else if k < pivot_index {
        quickselect_f32(&mut data[..pivot_index], k)
    } else {
        quickselect_f32(&mut data[pivot_index + 1..], k - pivot_index - 1)
    }
}

fn partition(data: &mut [f64]) -> usize {
    let pivot = data[data.len() - 1];
    let mut i = 0;
    
    for j in 0..data.len() - 1 {
        if data[j] <= pivot {
            data.swap(i, j);
            i += 1;
        }
    }
    
    data.swap(i, data.len() - 1);
    i
}

fn partition_f32(data: &mut [f32]) -> usize {
    let pivot = data[data.len() - 1];
    let mut i = 0;
    
    for j in 0..data.len() - 1 {
        if data[j] <= pivot {
            data.swap(i, j);
            i += 1;
        }
    }
    
    data.swap(i, data.len() - 1);
    i
}

// Parallel implementations when rayon feature is enabled
#[cfg(feature = "parallel")]
// Adaptive implementations for VectorF64
impl AdaptiveStats<f64> for VectorF64 {
    fn mean_adaptive(&self) -> f64 {
        let data = self.as_slice_unchecked();
        if data.is_empty() {
            return 0.0;
        }
        
        // Choose the best implementation based on size and available features
        if data.len() >= thresholds::PARALLEL_VECTOR_THRESHOLD {
            #[cfg(feature = "parallel")]
            {
                return data.par_iter().sum::<f64>() / data.len() as f64;
            }
        }
        
        // Use SIMD for medium-sized datasets
        if data.len() >= 64 {
            return simd::sum_f64_simd(data) / data.len() as f64;
        }
        
        // Fallback to standard implementation for small datasets
        self.mean()
    }
    
    fn var_adaptive(&self, ddof: Option<usize>) -> f64 {
        let data = self.as_slice_unchecked();
        let ddof = ddof.unwrap_or(1);
        
        if data.len() <= ddof {
            return 0.0;
        }
        
        let mean = self.mean_adaptive();
        
        // Choose implementation based on size
        if data.len() >= thresholds::PARALLEL_VECTOR_THRESHOLD {
            #[cfg(feature = "parallel")]
            {
                let sum_sq_diff: f64 = data.par_iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum();
                return sum_sq_diff / (data.len() - ddof) as f64;
            }
        }
        
        // Use SIMD for medium-sized datasets
        if data.len() >= 64 {
            let sum_sq_diff = simd::variance_f64_simd(data, mean);
            return sum_sq_diff / (data.len() - ddof) as f64;
        }
        
        // Fallback to standard implementation
        self.var(Some(ddof))
    }
    
    fn correlation_adaptive(&self, other: &Self) -> f64 {
        let data1 = self.as_slice_unchecked();
        let data2 = other.as_slice_unchecked();
        
        if data1.len() != data2.len() {
            panic!("Vector lengths must match for correlation");
        }
        
        if data1.is_empty() {
            return f64::NAN;
        }
        
        let mean1 = self.mean_adaptive();
        let mean2 = other.mean_adaptive();
        
        // Choose implementation based on size
        if data1.len() >= thresholds::PARALLEL_CORRELATION_THRESHOLD {
            #[cfg(feature = "parallel")]
            {
                let (sum_prod, sum_sq1, sum_sq2): (f64, f64, f64) = data1.par_iter()
                    .zip(data2.par_iter())
                    .map(|(&x1, &x2)| {
                        let dx1 = x1 - mean1;
                        let dx2 = x2 - mean2;
                        (dx1 * dx2, dx1 * dx1, dx2 * dx2)
                    })
                    .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
                
                if sum_sq1 == 0.0 || sum_sq2 == 0.0 {
                    return f64::NAN;
                }
                
                return sum_prod / (sum_sq1.sqrt() * sum_sq2.sqrt());
            }
        }
        
        // Use SIMD for medium-sized datasets
        if data1.len() >= 64 {
            // Compute centered values
            let centered1: Vec<f64> = data1.iter().map(|&x| x - mean1).collect();
            let centered2: Vec<f64> = data2.iter().map(|&x| x - mean2).collect();
            
            let sum_prod = simd::dot_product_f64_simd(&centered1, &centered2);
            let sum_sq1 = simd::dot_product_f64_simd(&centered1, &centered1);
            let sum_sq2 = simd::dot_product_f64_simd(&centered2, &centered2);
            
            if sum_sq1 == 0.0 || sum_sq2 == 0.0 {
                return f64::NAN;
            }
            
            return sum_prod / (sum_sq1.sqrt() * sum_sq2.sqrt());
        }
        
        // Fallback to standard implementation
        use crate::correlation::Correlation;
        self.pearson_correlation(other)
    }
}

#[cfg(feature = "parallel")]
impl ParallelStats<f64> for VectorF64 {
    fn mean_parallel(&self) -> f64 {
        let data = self.as_slice_unchecked();
        if data.is_empty() {
            return 0.0;
        }
        
        // Use adaptive implementation which may choose parallel or not
        if data.len() >= thresholds::PARALLEL_VECTOR_THRESHOLD {
            data.par_iter().sum::<f64>() / data.len() as f64
        } else {
            // Fallback to serial for small datasets
            self.mean()
        }
    }
    
    fn var_parallel(&self, ddof: Option<usize>) -> f64 {
        let data = self.as_slice_unchecked();
        let ddof = ddof.unwrap_or(1);
        
        if data.len() <= ddof {
            return 0.0;
        }
        
        // Only use parallel for large datasets
        if data.len() >= thresholds::PARALLEL_VECTOR_THRESHOLD {
            let mean = self.mean_parallel();
            
            let sum_sq_diff: f64 = data.par_iter()
                .map(|&x| (x - mean).powi(2))
                .sum();
            
            sum_sq_diff / (data.len() - ddof) as f64
        } else {
            // Fallback to serial for small datasets
            self.var(Some(ddof))
        }
    }
    
    fn correlation_parallel(&self, other: &Self) -> f64 {
        let data1 = self.as_slice_unchecked();
        let data2 = other.as_slice_unchecked();
        
        if data1.len() != data2.len() {
            panic!("Vector lengths must match for correlation");
        }
        
        if data1.is_empty() {
            return f64::NAN;
        }
        
        let mean1 = self.mean_parallel();
        let mean2 = other.mean_parallel();
        
        let (sum_prod, sum_sq1, sum_sq2): (f64, f64, f64) = data1.par_iter()
            .zip(data2.par_iter())
            .map(|(&x1, &x2)| {
                let dx1 = x1 - mean1;
                let dx2 = x2 - mean2;
                (dx1 * dx2, dx1 * dx1, dx2 * dx2)
            })
            .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        
        if sum_sq1 == 0.0 || sum_sq2 == 0.0 {
            return f64::NAN;
        }
        
        sum_prod / (sum_sq1.sqrt() * sum_sq2.sqrt())
    }
    
    fn mean_axis_parallel(&self, axis: Axis) -> Self {
        // For vectors, axis doesn't apply - return self
        self.clone()
    }
    
    fn zscore_parallel(&self, ddof: Option<usize>) -> Self {
        let data = self.as_slice_unchecked();
        
        if data.is_empty() {
            panic!("Cannot standardize empty vector");
        }
        
        let mean = self.mean_parallel();
        let std = self.var_parallel(ddof).sqrt();
        
        if std == 0.0 {
            panic!("Cannot standardize vector with zero standard deviation");
        }
        
        let standardized: Vec<f64> = data.par_iter()
            .map(|&x| (x - mean) / std)
            .collect();
        
        VectorF64::from_slice(&standardized)
    }
}

#[cfg(feature = "parallel")]
impl ParallelStats<f64> for ArrayF64 {
    fn mean_parallel(&self) -> f64 {
        let total_elements = self.nrows() * self.ncols();
        let mut data = Vec::with_capacity(total_elements);
        
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                data.push(self.get(i, j).unwrap());
            }
        }
        
        data.par_iter().sum::<f64>() / data.len() as f64
    }
    
    fn var_parallel(&self, ddof: Option<usize>) -> f64 {
        let ddof = ddof.unwrap_or(1);
        let total_elements = self.nrows() * self.ncols();
        
        if total_elements <= ddof {
            return 0.0;
        }
        
        let mut data = Vec::with_capacity(total_elements);
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                data.push(self.get(i, j).unwrap());
            }
        }
        
        let mean = self.mean_parallel();
        
        let sum_sq_diff: f64 = data.par_iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        
        sum_sq_diff / (total_elements - ddof) as f64
    }
    
    fn correlation_parallel(&self, other: &Self) -> f64 {
        if self.nrows() != other.nrows() || self.ncols() != other.ncols() {
            panic!("Array dimensions must match for correlation");
        }
        
        let total_elements = self.nrows() * self.ncols();
        let mut data1 = Vec::with_capacity(total_elements);
        let mut data2 = Vec::with_capacity(total_elements);
        
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                data1.push(self.get(i, j).unwrap());
                data2.push(other.get(i, j).unwrap());
            }
        }
        
        let mean1 = data1.par_iter().sum::<f64>() / data1.len() as f64;
        let mean2 = data2.par_iter().sum::<f64>() / data2.len() as f64;
        
        let (sum_prod, sum_sq1, sum_sq2): (f64, f64, f64) = data1.par_iter()
            .zip(data2.par_iter())
            .map(|(&x1, &x2)| {
                let dx1 = x1 - mean1;
                let dx2 = x2 - mean2;
                (dx1 * dx2, dx1 * dx1, dx2 * dx2)
            })
            .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        
        if sum_sq1 == 0.0 || sum_sq2 == 0.0 {
            return f64::NAN;
        }
        
        sum_prod / (sum_sq1.sqrt() * sum_sq2.sqrt())
    }
    
    fn mean_axis_parallel(&self, axis: Axis) -> Self {
        let (nrows, ncols) = (self.nrows(), self.ncols());
        
        match axis {
            Axis::Rows => {
                // Compute mean for each column (result has 1 row, ncols columns)
                let col_means: Vec<f64> = (0..ncols).into_par_iter()
                    .map(|j| {
                        let col_sum: f64 = (0..nrows).map(|i| self.get(i, j).unwrap()).sum();
                        col_sum / nrows as f64
                    })
                    .collect();
                
                ArrayF64::from_slice(&col_means, 1, ncols).unwrap()
            },
            Axis::Cols => {
                // Compute mean for each row (result has nrows rows, 1 column)
                let row_means: Vec<f64> = (0..nrows).into_par_iter()
                    .map(|i| {
                        let row_sum: f64 = (0..ncols).map(|j| self.get(i, j).unwrap()).sum();
                        row_sum / ncols as f64
                    })
                    .collect();
                
                ArrayF64::from_slice(&row_means, nrows, 1).unwrap()
            }
        }
    }
    
    fn zscore_parallel(&self, ddof: Option<usize>) -> Self {
        let mean = self.mean_parallel();
        let std = self.var_parallel(ddof).sqrt();
        
        if std == 0.0 {
            panic!("Cannot standardize array with zero standard deviation");
        }
        
        let (nrows, ncols) = (self.nrows(), self.ncols());
        let standardized_data: Vec<f64> = (0..nrows * ncols).into_par_iter()
            .map(|idx| {
                let i = idx / ncols;
                let j = idx % ncols;
                let value = self.get(i, j).unwrap();
                (value - mean) / std
            })
            .collect();
        
        ArrayF64::from_slice(&standardized_data, nrows, ncols).unwrap()
    }
}

/// Benchmarking utilities for performance testing
pub mod benchmarks {
    use super::*;
    use std::time::{Duration, Instant};
    
    /// Benchmark result containing timing information
    #[derive(Debug, Clone)]
    pub struct BenchmarkResult {
        /// Name of the benchmarked operation
        pub name: String,
        /// Duration taken for the operation
        pub duration: Duration,
        /// Operations performed per second
        pub operations_per_second: f64,
        /// Size of the data processed
        pub data_size: usize,
    }
    
    impl BenchmarkResult {
        /// Create a new benchmark result
        pub fn new(name: String, duration: Duration, data_size: usize) -> Self {
            let ops_per_sec = if duration.as_secs_f64() > 0.0 {
                data_size as f64 / duration.as_secs_f64()
            } else {
                f64::INFINITY
            };
            
            Self {
                name,
                duration,
                operations_per_second: ops_per_sec,
                data_size,
            }
        }
    }
    
    /// Benchmark a statistical operation
    pub fn benchmark_operation<F, T>(name: &str, data_size: usize, mut operation: F) -> BenchmarkResult 
    where 
        F: FnMut() -> T,
    {
        // Warm up
        for _ in 0..3 {
            let _ = operation();
        }
        
        let start = Instant::now();
        let _result = operation();
        let duration = start.elapsed();
        
        BenchmarkResult::new(name.to_string(), duration, data_size)
    }
    
    /// Compare two implementations and return performance ratio
    pub fn compare_implementations<F1, F2, T>(
        name1: &str, 
        name2: &str, 
        data_size: usize,
        impl1: F1, 
        impl2: F2
    ) -> (BenchmarkResult, BenchmarkResult, f64)
    where 
        F1: FnMut() -> T,
        F2: FnMut() -> T,
    {
        let result1 = benchmark_operation(name1, data_size, impl1);
        let result2 = benchmark_operation(name2, data_size, impl2);
        
        let speedup = result1.duration.as_secs_f64() / result2.duration.as_secs_f64();
        
        (result1, result2, speedup)
    }
    
    /// Run comprehensive benchmarks on vector operations
    pub fn benchmark_vector_ops(size: usize) -> Vec<BenchmarkResult> {
        use crate::advanced::quantiles::Quantiles;
        use crate::normalization::Normalization;
        
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let vector = VectorF64::from_slice(&data);
        
        let mut results = Vec::new();
        
        // Benchmark mean
        results.push(benchmark_operation("mean", size, || vector.mean()));
        
        // Benchmark variance
        results.push(benchmark_operation("variance", size, || vector.var(None)));
        
        // Benchmark median
        results.push(benchmark_operation("median", size, || vector.median()));
        
        // Benchmark zscore
        results.push(benchmark_operation("zscore", size, || vector.zscore(None)));
        
        // Benchmark zero-copy mean
        results.push(benchmark_operation("mean_zero_copy", size, || {
            f64::mean_slice(vector.as_slice_unchecked())
        }));
        
        #[cfg(feature = "parallel")]
        {
            // Benchmark parallel mean
            results.push(benchmark_operation("mean_parallel", size, || vector.mean_parallel()));
            
            // Benchmark parallel variance
            results.push(benchmark_operation("var_parallel", size, || vector.var_parallel(None)));
        }
        
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::{vec64, ArrayF64};
    
    #[test]
    fn test_zero_copy_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = f64::mean_slice(&data);
        assert!((mean - 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_zero_copy_variance() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let var = f64::var_slice(&data, 1);
        assert!((var - 2.5).abs() < 1e-10); // Sample variance
    }
    
    #[test]
    fn test_zero_copy_minmax() {
        let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let (min, max) = f64::minmax_slice(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 9.0);
    }
    
    #[test]
    fn test_quickselect_median() {
        let mut data = [3.0, 1.0, 4.0, 1.0, 5.0];
        let median = f64::quantile_slice_fast(&mut data, 0.5);
        // For [1, 1, 3, 4, 5], median should be 3
        assert_eq!(median, 3.0);
    }
    
    #[test]
    fn test_streaming_stats() {
        let mut stats = StreamingStats::<f64>::new();
        
        // Add values incrementally
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        for &value in &values {
            stats.update(value);
        }
        
        assert_eq!(stats.count(), 5);
        assert!((stats.mean() - 3.0).abs() < 1e-10);
        assert!((stats.variance() - 2.5).abs() < 1e-10); // Sample variance
        assert_eq!(stats.minmax(), (1.0, 5.0));
    }
    
    #[test]
    fn test_streaming_stats_merge() {
        let mut stats1 = StreamingStats::<f64>::new();
        let mut stats2 = StreamingStats::<f64>::new();
        
        // Add values to first accumulator
        for &value in &[1.0, 2.0, 3.0] {
            stats1.update(value);
        }
        
        // Add values to second accumulator  
        for &value in &[4.0, 5.0] {
            stats2.update(value);
        }
        
        // Merge
        stats1.merge(&stats2);
        
        assert_eq!(stats1.count(), 5);
        assert!((stats1.mean() - 3.0).abs() < 1e-10);
        assert!((stats1.variance() - 2.5).abs() < 1e-10);
    }
    
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_mean() {
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean_serial = data.mean();
        let mean_parallel = data.mean_parallel();
        
        assert!((mean_serial - mean_parallel).abs() < 1e-10);
    }
    
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_variance() {
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let var_serial = data.var(None);
        let var_parallel = data.var_parallel(None);
        
        assert!((var_serial - var_parallel).abs() < 1e-10);
    }
    
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_correlation() {
        let data1 = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec64![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr_parallel = data1.correlation_parallel(&data2);
        
        // Should be perfect positive correlation
        assert!((corr_parallel - 1.0).abs() < 1e-10);
    }
    
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_array_mean_axis() {
        let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        // Array:
        // [1.0, 2.0, 3.0]
        // [4.0, 5.0, 6.0]
        
        let col_means = arr.mean_axis_parallel(Axis::Rows);
        // Column means: [2.5, 3.5, 4.5]
        
        assert_eq!(col_means.nrows(), 1);
        assert_eq!(col_means.ncols(), 3);
        assert!((col_means.get(0, 0).unwrap() - 2.5).abs() < 1e-10);
        assert!((col_means.get(0, 1).unwrap() - 3.5).abs() < 1e-10);
        assert!((col_means.get(0, 2).unwrap() - 4.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_benchmark_operation() {
        let result = benchmarks::benchmark_operation("test_op", 1000, || {
            let data = vec![1.0f64; 1000];
            data.iter().sum::<f64>()
        });
        
        assert_eq!(result.name, "test_op");
        assert_eq!(result.data_size, 1000);
        assert!(result.duration.as_nanos() > 0);
        assert!(result.operations_per_second > 0.0);
    }
    
    #[test]
    fn test_benchmark_vector_ops() {
        let results = benchmarks::benchmark_vector_ops(1000);
        
        // Should have at least basic operations
        assert!(results.len() >= 4);
        
        let mean_result = results.iter().find(|r| r.name == "mean").unwrap();
        assert_eq!(mean_result.data_size, 1000);
        assert!(mean_result.duration.as_nanos() > 0);
    }
    
    #[test]
    fn test_f32_zero_copy_operations() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        
        let mean = f32::mean_slice(&data);
        assert!((mean - 3.0).abs() < 1e-5);
        
        let var = f32::var_slice(&data, 1);
        assert!((var - 2.5).abs() < 1e-5);
        
        let (min, max) = f32::minmax_slice(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
    }
}