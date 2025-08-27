//! # List Comprehension with Automatic Parallelism
//! 
//! Advanced list comprehension capabilities with complexity-aware automatic parallelism,
//! providing NumPy/Julia-style vectorization for Rust with optimal performance.
//!
//! ## Key Features
//! 
//! - **Cost-Based Parallelism**: Uses total computational cost (complexity × size) for decisions
//! - **Zero-Overhead Options**: `serial:` mode for guaranteed zero-overhead serial execution
//! - **Reference-Based Processing**: Use `&data` to avoid cloning overhead
//! - **Adaptive Measurement**: Profiles unknown functions to determine complexity
//! - **Memory Efficient**: Chunked processing for huge datasets
//! - **Type Safe**: Compile-time dimension checking
//!
//! ## Parallelization Strategy
//! 
//! The system calculates total cost as `complexity_factor × number_of_elements` and
//! parallelizes when total cost exceeds 500,000:
//! 
//! - **Trivial** (factor 1): Requires 500,000+ elements
//! - **Simple** (factor 10): Requires 50,000+ elements  
//! - **Moderate** (factor 100): Requires 5,000+ elements
//! - **Complex** (factor 10,000): Requires only 50+ elements
//!
//! ## Usage Examples
//!
//! ```rust
//! use rustlab_math::{vectorize, VectorF64, Complexity};
//! 
//! // Zero-overhead serial execution
//! let data = vec![1.0, 2.0, 3.0];
//! let result = vectorize![serial: x * 2.0, for x in &data];
//! 
//! // Automatic complexity detection (avoids cloning with &data)
//! let normalized = vectorize![x.sin() * x.cos(), for x in &data];
//! 
//! // Force parallel for expensive operations
//! let results = vectorize![
//!     complex: expensive_function(x),
//!     for x in &expensive_data
//! ];
//! 
//! // Adaptive complexity profiling
//! let adaptive = vectorize![
//!     adaptive: unknown_function(*x),
//!     for x in data
//! ];
//! ```

use std::time::Instant;
use rayon::prelude::*;
use crate::{VectorF64, ArrayF64, Result, MathError};

/// Complexity levels for operations, determining parallelization thresholds
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Complexity {
    /// Simple arithmetic operations (x + 1, x * 2)
    /// Cost factor: 1 (requires 500,000+ elements for parallelization)
    Trivial,
    
    /// Mathematical functions (sqrt, sin, cos)
    /// Cost factor: 10 (requires 50,000+ elements for parallelization)
    Simple,
    
    /// Matrix operations, FFT, moderate algorithms
    /// Cost factor: 100 (requires 5,000+ elements for parallelization)
    Moderate,
    
    /// Optimization, simulation, neural networks
    /// Cost factor: 10,000 (requires only 50+ elements for parallelization)
    Complex,
}

impl Complexity {
    /// Calculate if operation should be parallelized based on size and complexity
    /// Uses total computational cost (complexity factor × size) to decide
    #[inline(always)]
    pub fn should_parallelize(&self, size: usize) -> bool {
        // Calculate total computational cost
        let total_cost = self.cost_factor() * size;
        
        // Parallelize when total cost exceeds threshold
        // This accounts for both operation complexity and data size
        // Increased threshold to avoid parallelizing small workloads
        const PARALLELIZATION_THRESHOLD: usize = 500_000;
        
        total_cost >= PARALLELIZATION_THRESHOLD
    }
    
    /// Get the cost factor for this complexity level
    #[inline(always)]
    pub fn cost_factor(&self) -> usize {
        match self {
            Complexity::Trivial => 1,      // Basic arithmetic
            Complexity::Simple => 10,      // Math functions like sin, cos
            Complexity::Moderate => 100,   // Matrix ops, FFT
            Complexity::Complex => 10000,  // Simulations, ML models
        }
    }
}

/// Cost model for measuring operation complexity
pub struct CostModel;

impl CostModel {
    /// Measure actual execution time to determine complexity
    /// 
    /// Samples the first few elements to estimate operation cost
    pub fn measure_complexity<F, T, R>(f: &F, samples: &[T]) -> Complexity 
    where 
        F: Fn(&T) -> R,
        T: Sync,
    {
        if samples.is_empty() {
            return Complexity::Simple;
        }
        
        // Take up to 10 samples for measurement
        let n_samples = samples.len().min(10);
        let measure_samples = &samples[..n_samples];
        
        // Warm-up run
        for sample in measure_samples.iter().take(2) {
            let _ = f(sample);
        }
        
        // Actual measurement
        let start = Instant::now();
        for sample in measure_samples {
            let _ = f(sample);
        }
        let elapsed_nanos = start.elapsed().as_nanos() / n_samples as u128;
        
        // Classify based on nanoseconds per operation
        match elapsed_nanos {
            0..=100 => Complexity::Trivial,
            101..=1_000 => Complexity::Simple,
            1_001..=10_000 => Complexity::Moderate,
            _ => Complexity::Complex,
        }
    }
}

/// Trait for types that can be computed in comprehensions
pub trait Computable {
    /// The complexity of this computation
    const COMPLEXITY: Complexity = Complexity::Simple;
    
    /// The output type of the computation
    type Output;
    
    /// Perform the computation
    fn compute(self) -> Self::Output;
}

/// Core vectorization function with automatic parallelism
/// 
/// Takes ownership of data and applies function with complexity-aware parallelism.
/// Parallelizes when `complexity.cost_factor() * data.len() >= 500,000`.
pub fn vectorize_with_complexity<T, F, R>(
    data: Vec<T>,
    complexity: Complexity,
    f: F,
) -> Vec<R>
where
    T: Send + Sync,
    F: Fn(T) -> R + Send + Sync,
    R: Send,
{
    if complexity.should_parallelize(data.len()) {
        data.into_par_iter().map(f).collect()
    } else {
        data.into_iter().map(f).collect()
    }
}

/// Reference-based vectorization for efficient processing without cloning
/// 
/// Borrows data instead of taking ownership, avoiding clone overhead.
/// Ideal for when you need to keep the original data.
/// Parallelizes when `complexity.cost_factor() * data.len() >= 500,000`.
#[inline(always)]
pub fn vectorize_with_complexity_ref<T, F, R>(
    data: &[T],
    complexity: Complexity,
    f: F,
) -> Vec<R>
where
    T: Send + Sync,
    F: Fn(&T) -> R + Send + Sync,
    R: Send,
{
    if complexity.should_parallelize(data.len()) {
        data.par_iter().map(f).collect()
    } else {
        data.iter().map(f).collect()
    }
}

/// Zero-overhead serial vectorization
/// 
/// Guaranteed serial execution with no parallelism checks.
/// Use when you know parallelism isn't beneficial.
/// Expands directly to `data.iter().map(f).collect()`.
#[inline(always)]
pub fn vectorize_serial<T, F, R>(
    data: &[T],
    f: F,
) -> Vec<R>
where
    F: Fn(&T) -> R,
{
    data.iter().map(f).collect()
}

/// Adaptive vectorization that measures complexity
pub fn vectorize_adaptive<T, F, R>(
    data: Vec<T>,
    f: F,
) -> Vec<R>
where
    T: Send + Sync,
    F: Fn(&T) -> R + Send + Sync,
    R: Send + Clone,
{
    // Measure complexity on first few elements
    let complexity = if data.len() > 0 {
        CostModel::measure_complexity(&f, &data[..data.len().min(10)])
    } else {
        Complexity::Simple
    };
    
    // Apply function with determined complexity
    if complexity.should_parallelize(data.len()) {
        data.par_iter().map(f).collect()
    } else {
        data.iter().map(f).collect()
    }
}

/// Chunked processing for memory-efficient parallel computation
pub fn vectorize_chunked<T, F, R>(
    data: Vec<T>,
    chunk_size: usize,
    f: F,
) -> Vec<R>
where
    T: Send + Sync,
    F: Fn(&[T]) -> Vec<R> + Send + Sync,
    R: Send,
{
    data.par_chunks(chunk_size)
        .flat_map(f)
        .collect()
}

/// Extensions for VectorF64 with comprehension support
impl VectorF64 {
    /// Apply a function to each element with automatic parallelism
    /// 
    /// # Example
    /// ```rust
    /// let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// let squared = vec.apply_with_complexity(|x| x * x, Complexity::Trivial);
    /// ```
    pub fn apply_with_complexity<F>(&self, f: F, complexity: Complexity) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        let data: Vec<f64> = self.to_vec();
        let result = vectorize_with_complexity(data, complexity, f);
        VectorF64::from_slice(&result)
    }
    
    /// Apply a function with adaptive complexity detection
    pub fn apply_adaptive<F>(&self, f: F) -> Self
    where
        F: Fn(&f64) -> f64 + Send + Sync + Clone,
    {
        let data: Vec<f64> = self.to_vec();
        let result = vectorize_adaptive(data, f);
        VectorF64::from_slice(&result)
    }
    
    /// Zip with another vector and apply a binary function
    pub fn zip_with_complexity<F>(&self, other: &Self, f: F, complexity: Complexity) -> Result<Self>
    where
        F: Fn(f64, f64) -> f64 + Send + Sync,
    {
        if self.len() != other.len() {
            return Err(MathError::InvalidSliceLength {
                expected: self.len(),
                actual: other.len(),
            });
        }
        
        let pairs: Vec<(f64, f64)> = self.to_vec()
            .into_iter()
            .zip(other.to_vec())
            .collect();
            
        let result = vectorize_with_complexity(pairs, complexity, |(a, b)| f(a, b));
        Ok(VectorF64::from_slice(&result))
    }
}

/// Generate coordinate grids (meshgrid functionality)
pub fn meshgrid(x: &VectorF64, y: &VectorF64) -> (ArrayF64, ArrayF64) {
    let nx = x.len();
    let ny = y.len();
    
    let mut X = ArrayF64::zeros(ny, nx);
    let mut Y = ArrayF64::zeros(ny, nx);
    
    // Serial population for all grids
    for i in 0..ny {
        for j in 0..nx {
            let _ = X.set(i, j, x[j]);
            let _ = Y.set(i, j, y[i]);
        }
    }
    
    (X, Y)
}

/// Macro for list comprehension with automatic parallelism
/// 
/// Provides NumPy/Julia-style list comprehensions with intelligent parallelism.
/// 
/// # Syntax Options
/// 
/// - `vectorize![serial: expr, for x in &data]` - Zero-overhead serial execution
/// - `vectorize![expr, for x in &data]` - Auto-decides based on Simple complexity
/// - `vectorize![complex: expr, for x in &data]` - Forces Complex complexity (low threshold)
/// - `vectorize![adaptive: expr, for x in data]` - Measures complexity at runtime
/// 
/// # Performance Tips
/// 
/// - Use `&data` to avoid cloning when possible
/// - Use `serial:` when you know parallelism won't help
/// - Use `complex:` for expensive operations like simulations
/// - Default mode uses Simple complexity (50,000+ elements for parallelism)
#[macro_export]
macro_rules! vectorize {
    // Serial mode - zero overhead, expands directly to iterator chain
    [serial: $expr:expr, for $var:ident in &$iter:expr] => {{
        $iter.iter().map(|$var| $expr).collect::<Vec<_>>()
    }};
    
    // Serial mode - owned version
    [serial: $expr:expr, for $var:ident in $iter:expr] => {{
        $iter.into_iter().map(|$var| $expr).collect::<Vec<_>>()
    }};
    
    // Complex operations - low threshold (by reference to avoid clone)
    [complex: $expr:expr, for $var:ident in &$iter:expr] => {{
        $crate::comprehension::vectorize_with_complexity_ref(
            &$iter[..],
            $crate::comprehension::Complexity::Complex,
            |$var| $expr
        )
    }};
    
    // Complex operations - low threshold (owned)
    [complex: $expr:expr, for $var:ident in $iter:expr] => {{
        let data: Vec<_> = $iter.into_iter().collect();
        $crate::comprehension::vectorize_with_complexity(
            data,
            $crate::comprehension::Complexity::Complex,
            |$var| $expr
        )
    }};
    
    // Adaptive complexity detection
    [adaptive: $expr:expr, for $var:ident in $iter:expr] => {{
        let data: Vec<_> = $iter.into_iter().collect();
        $crate::comprehension::vectorize_adaptive(
            data,
            |$var| $expr
        )
    }};
    
    // User-specified complexity
    [$expr:expr, for $var:ident in $iter:expr, complexity = $comp:expr] => {{
        let data: Vec<_> = $iter.into_iter().collect();
        $crate::comprehension::vectorize_with_complexity(
            data,
            $comp,
            |$var| $expr
        )
    }};
    
    // Default case - simple complexity (by reference to avoid clone)
    [$expr:expr, for $var:ident in &$iter:expr] => {{
        $crate::comprehension::vectorize_with_complexity_ref(
            &$iter[..],
            $crate::comprehension::Complexity::Simple,
            |$var| $expr
        )
    }};
    
    // Default case - simple complexity (owned)
    [$expr:expr, for $var:ident in $iter:expr] => {{
        let data: Vec<_> = $iter.into_iter().collect();
        $crate::comprehension::vectorize_with_complexity(
            data,
            $crate::comprehension::Complexity::Simple,
            |$var| $expr
        )
    }};
}

/// Macro for creating coordinate grids
#[macro_export]
macro_rules! meshgrid {
    (x: $x:expr, y: $y:expr) => {{
        $crate::comprehension::meshgrid(&$x, &$y)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_complexity_thresholds() {
        // With threshold of 500,000:
        // Trivial (factor 1): needs 500,000+ elements
        assert!(!Complexity::Trivial.should_parallelize(100_000));
        assert!(!Complexity::Trivial.should_parallelize(499_999));
        assert!(Complexity::Trivial.should_parallelize(500_000));
        
        // Simple (factor 10): needs 50,000+ elements
        assert!(!Complexity::Simple.should_parallelize(10_000));
        assert!(!Complexity::Simple.should_parallelize(49_999));
        assert!(Complexity::Simple.should_parallelize(50_000));
        
        // Moderate (factor 100): needs 5,000+ elements  
        assert!(!Complexity::Moderate.should_parallelize(1_000));
        assert!(!Complexity::Moderate.should_parallelize(4_999));
        assert!(Complexity::Moderate.should_parallelize(5_000));
        
        // Complex (factor 10,000): needs 50+ elements
        assert!(!Complexity::Complex.should_parallelize(49));
        assert!(Complexity::Complex.should_parallelize(50));
        assert!(Complexity::Complex.should_parallelize(100));
    }
    
    #[test]
    fn test_vectorize_simple() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = vectorize_with_complexity(data, Complexity::Trivial, |x| x * 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }
    
    #[test]
    fn test_vectorize_macro_basic() {
        let data = vec![1.0, 2.0, 3.0];
        let result: Vec<f64> = vectorize![x * 2.0, for x in data];
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }
    
    #[test]
    fn test_vector_apply() {
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
        let squared = vec.apply_with_complexity(|x| x * x, Complexity::Trivial);
        assert_eq!(squared.to_vec(), vec![1.0, 4.0, 9.0]);
    }
    
    #[test]
    fn test_vector_zip_with() {
        let v1 = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
        let v2 = VectorF64::from_slice(&[4.0, 5.0, 6.0]);
        let result = v1.zip_with_complexity(&v2, |a, b| a + b, Complexity::Trivial).unwrap();
        assert_eq!(result.to_vec(), vec![5.0, 7.0, 9.0]);
    }
    
    #[test]
    fn test_meshgrid() {
        let x = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
        let y = VectorF64::from_slice(&[4.0, 5.0]);
        
        let (X, Y) = meshgrid(&x, &y);
        
        // Check X grid
        assert_eq!(X.get(0, 0), Some(1.0));
        assert_eq!(X.get(0, 1), Some(2.0));
        assert_eq!(X.get(0, 2), Some(3.0));
        assert_eq!(X.get(1, 0), Some(1.0));
        assert_eq!(X.get(1, 1), Some(2.0));
        assert_eq!(X.get(1, 2), Some(3.0));
        
        // Check Y grid
        assert_eq!(Y.get(0, 0), Some(4.0));
        assert_eq!(Y.get(0, 1), Some(4.0));
        assert_eq!(Y.get(0, 2), Some(4.0));
        assert_eq!(Y.get(1, 0), Some(5.0));
        assert_eq!(Y.get(1, 1), Some(5.0));
        assert_eq!(Y.get(1, 2), Some(5.0));
    }
}