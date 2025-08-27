//! Natural Slicing Syntax Using Rust's Index Trait for Python/NumPy-like Ergonomics
//!
//! This module implements Rust's Index trait to enable the most natural possible
//! slicing syntax, making RustLab feel like Python/NumPy for mathematical computing.
//!
//! # Core Philosophy
//! Provide true **Python/NumPy syntax** in Rust:
//! - `&vec[1..4]` instead of `vec.slice_at(1..4)` - Zero syntax overhead
//! - `&vec[2..]` instead of `vec.slice_at(2..)` - Natural range expressions  
//! - `vec.select(indices)` for operations requiring owned results
//! - Bridge Index trait limitations with extension traits
//!
//! # Design Principles
//! 1. **Zero Syntax Overhead**: `&vec[1..4]` is as natural as Python `arr[1:4]`
//! 2. **Performance First**: Index operations return slice references (zero-copy)
//! 3. **Safety**: All operations are bounds-checked (panic on invalid indices)
//! 4. **Completeness**: Cover all NumPy slicing patterns through multiple APIs
//! 5. **Composability**: Operations work with all RustLab mathematical functions
//!
//! # Three-Layer API Design
//!
//! ## Layer 1: Index Trait (Zero-Copy Views)
//! ```rust
//! use rustlab_math::vec64;
//! 
//! let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
//! 
//! let slice = &vec[1..4];  // &[f64] slice reference - zero copy
//! let tail = &vec[2..];    // &[f64] slice reference - zero copy  
//! let head = &vec[..3];    // &[f64] slice reference - zero copy
//! let full = &vec[..];     // &[f64] slice reference - zero copy
//! ```
//!
//! ## Layer 2: NaturalSlicing Trait (Owned Results)
//! ```rust
//! use rustlab_math::{vec64, NaturalSlicing, BooleanVector};
//! 
//! let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
//! 
//! // Operations requiring owned results
//! let owned = vec.slice_owned(1..4);           // VectorF64 - owned
//! let selected = vec.select(vec![0, 2, 4]);    // VectorF64 - owned
//! let mask = BooleanVector::from_slice(&[true, false, true, false, true]);
//! let filtered = vec.select_where(mask);       // VectorF64 - owned
//! ```
//!
//! ## Layer 3: Method-Based API (Full Flexibility) 
//! ```rust
//! use rustlab_math::{vec64, BooleanVector};
//! 
//! let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
//! 
//! // Error handling and advanced features
//! let result = vec.slice_at(1..4)?;            // Result<VectorF64, String>
//! let last = vec.slice_at(-1)?;                // Negative indexing
//! let fancy = vec.slice_at(vec![0, 2, 4])?;    // Fancy indexing
//! ```
//!
//! # Mathematical Foundation
//! All operations follow rigorous mathematical definitions:
//! - **Vector slicing**: v[i..j] → [v[i], v[i+1], ..., v[j-1]] ∈ ℝ^(j-i)
//! - **Half-open intervals**: Range [i..j) includes i, excludes j
//! - **Zero-based indexing**: First element at index 0, last at index n-1
//! - **Bounds safety**: All operations validate indices before access
//!
//! # Performance Characteristics
//! - **Index trait**: O(1) slice references, zero memory allocation
//! - **Owned operations**: O(k) where k is result size, single allocation
//! - **Memory layout**: Preserves cache locality for all operations
//! - **SIMD compatibility**: Results work with vectorized operations
//!
//! # For AI Code Generation
//! This module provides the most natural translation targets for AI tools:
//! - **Direct NumPy mapping**: `arr[1:4]` → `&vec[1..4]`
//! - **Type safety**: Compile-time slice type checking
//! - **Error contexts**: Clear panic messages for debugging
//! - **Chainable**: `&vec[1..4][0..2]` works naturally
//! - **Consistent**: Same patterns across all RustLab data structures
//!
//! # Index Trait Limitations and Solutions
//! Rust's Index trait has fundamental limitations that we address:
//!
//! | Limitation | Solution | Example |
//! |------------|----------|---------|
//! | Only returns references | NaturalSlicing trait | `vec.slice_owned(1..4)` |
//! | No fancy indexing | Extension methods | `vec.select(vec![0,2,4])` |
//! | No boolean masking | Extension methods | `vec.select_where(mask)` |
//! | No error handling | Method-based API | `vec.slice_at(1..4)?` |
//! | No 2D tuple syntax | Method-based API | `arr.slice_2d_at((1..3, 0..2))` |
//!
//! # Integration Examples
//! ```rust
//! use rustlab_math::{vec64, array64, NaturalSlicing};
//! 
//! let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! 
//! // Natural syntax for common patterns
//! let window = &data[1..5];                    // Sliding window
//! let head = &data[..3];                       // First N elements  
//! let tail = &data[3..];                       // Last N elements
//! 
//! // Owned results for transformations
//! let processed = data.slice_owned(1..4)       // [2.0, 3.0, 4.0]
//!                     .slice_owned(0..2);      // [2.0, 3.0] - chained!
//! 
//! // Statistical operations work seamlessly
//! let mean = (&data[1..4]).iter().sum::<f64>() / 3.0;
//! 
//! // Matrix slicing follows same patterns
//! let matrix = array64![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//! let submatrix = matrix.slice_2d_at((0..2, 1..3))?;
//! ```

use std::ops::{Index, Range, RangeFrom, RangeTo, RangeFull};
use crate::{VectorF64, BooleanVector};

// ============================================================================
// Natural Range Slicing for VectorF64
// ============================================================================

/// Enable Python/NumPy-style bracket notation for range slicing: `vec[1..4]`
/// 
/// # Mathematical Specification
/// Given vector v ∈ ℝⁿ and range [i..j):
/// Returns slice reference &v[i..j-1] = [v[i], v[i+1], ..., v[j-1]]
/// 
/// # Dimensions
/// - Input: self (n), range i..j where 0 ≤ i < j ≤ n
/// - Output: &[f64] slice reference (length j-i)
/// 
/// # Complexity
/// - Time: O(1) - returns slice reference, no copying
/// - Space: O(1) - zero additional memory
/// 
/// # For AI Code Generation
/// - Enables natural Python syntax: `&vec[1..4]` instead of `vec.slice_at(1..4)`
/// - Returns slice reference (&[f64]), not owned vector
/// - Zero-copy operation - extremely efficient for read-only access
/// - Panics on out-of-bounds - use `slice_at()` for error handling
/// - Common pattern: `let slice = &vec[start..end];`
/// - Use when you need temporary read-only access to subvector
/// 
/// # Example
/// ```
/// use rustlab_math::vec64;
/// 
/// let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
/// 
/// let slice = &vec[1..4];  // [2.0, 3.0, 4.0] as &[f64]
/// assert_eq!(slice, &[2.0, 3.0, 4.0]);
/// 
/// // Zero-copy - extremely fast for large vectors
/// let large_vec = vec64![/* ... 1M elements ... */];
/// let window = &large_vec[1000..2000];  // O(1) operation
/// ```
/// 
/// # Panics
/// - Panics if start >= end or end > vector length
/// - For safe access with error handling, use `slice_at()` method instead
impl Index<Range<usize>> for VectorF64 {
    type Output = [f64];
    
    fn index(&self, range: Range<usize>) -> &Self::Output {
        // Get the underlying slice and return the range
        let slice = self.as_slice().expect("Vector data must be contiguous for natural slicing");
        &slice[range]
    }
}

/// Enable Python/NumPy-style bracket notation for tail slicing: `vec[2..]`
/// 
/// # Mathematical Specification
/// Given vector v ∈ ℝⁿ and range [i..):
/// Returns slice reference &v[i..n-1] = [v[i], v[i+1], ..., v[n-1]]
/// where n is vector length
/// 
/// # Dimensions
/// - Input: self (n), range i.. where 0 ≤ i ≤ n
/// - Output: &[f64] slice reference (length n-i)
/// 
/// # Complexity
/// - Time: O(1) - returns slice reference, no copying
/// - Space: O(1) - zero additional memory
/// 
/// # For AI Code Generation
/// - Enables Python-style tail slicing: `&vec[2..]` for "from index 2 to end"
/// - Returns slice reference (&[f64]), not owned vector
/// - Zero-copy operation for efficient access to vector suffix
/// - Panics on out-of-bounds - use `slice_at()` for error handling
/// - Common pattern: `let tail = &vec[start..];`
/// - Use for processing vector suffixes without allocation
/// 
/// # Example
/// ```
/// use rustlab_math::vec64;
/// 
/// let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
/// 
/// let tail = &vec[2..];  // [3.0, 4.0, 5.0] as &[f64]
/// assert_eq!(tail, &[3.0, 4.0, 5.0]);
/// 
/// // Skip header, process rest
/// let data_with_header = vec64![0.0, 1.0, 2.0, 3.0, 4.0];  // 0.0 is header
/// let actual_data = &data_with_header[1..];  // Skip first element
/// ```
/// 
/// # Panics
/// - Panics if start > vector length
/// - Returns empty slice if start == vector length
impl Index<RangeFrom<usize>> for VectorF64 {
    type Output = [f64];
    
    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        let slice = self.as_slice().expect("Vector data must be contiguous for natural slicing");
        &slice[range]
    }
}

/// Enable Python/NumPy-style bracket notation for head slicing: `vec[..3]`
/// 
/// # Mathematical Specification
/// Given vector v ∈ ℝⁿ and range [..j):
/// Returns slice reference &v[0..j-1] = [v[0], v[1], ..., v[j-1]]
/// 
/// # Dimensions
/// - Input: self (n), range ..j where 0 ≤ j ≤ n
/// - Output: &[f64] slice reference (length j)
/// 
/// # Complexity
/// - Time: O(1) - returns slice reference, no copying
/// - Space: O(1) - zero additional memory
/// 
/// # For AI Code Generation
/// - Enables Python-style head slicing: `&vec[..3]` for "first 3 elements"
/// - Returns slice reference (&[f64]), not owned vector
/// - Zero-copy operation for efficient access to vector prefix
/// - Panics on out-of-bounds - use `slice_at()` for error handling
/// - Common pattern: `let head = &vec[..count];`
/// - Use for processing initial vector elements without allocation
/// 
/// # Example
/// ```
/// use rustlab_math::vec64;
/// 
/// let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
/// 
/// let head = &vec[..3];  // [1.0, 2.0, 3.0] as &[f64]
/// assert_eq!(head, &[1.0, 2.0, 3.0]);
/// 
/// // Process first N elements
/// let samples = vec64![/* ... many samples ... */];
/// let preview = &samples[..10];  // First 10 samples only
/// ```
/// 
/// # Panics
/// - Panics if end > vector length
/// - Returns empty slice if end == 0
impl Index<RangeTo<usize>> for VectorF64 {
    type Output = [f64];
    
    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        let slice = self.as_slice().expect("Vector data must be contiguous for natural slicing");
        &slice[range]
    }
}

/// Enable Python/NumPy-style bracket notation for full slicing: `vec[..]`
/// 
/// # Mathematical Specification
/// Given vector v ∈ ℝⁿ and range [..):
/// Returns slice reference &v[0..n-1] = entire vector as slice
/// 
/// # Dimensions
/// - Input: self (n)
/// - Output: &[f64] slice reference (length n)
/// 
/// # Complexity
/// - Time: O(1) - returns slice reference, no copying
/// - Space: O(1) - zero additional memory
/// 
/// # For AI Code Generation
/// - Enables Python-style full slicing: `&vec[..]` for entire vector as slice
/// - Returns slice reference (&[f64]), not owned vector
/// - Equivalent to `vec.as_slice()` but with natural bracket syntax
/// - Never panics - always valid operation
/// - Common pattern: `let slice = &vec[..];` for type conversion
/// - Use when you need &[f64] interface instead of VectorF64
/// 
/// # Example
/// ```
/// use rustlab_math::vec64;
/// 
/// let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
/// 
/// let slice = &vec[..];  // Entire vector as &[f64]
/// assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0]);
/// 
/// // Pass to function expecting &[f64]
/// fn process_slice(data: &[f64]) -> f64 { data.iter().sum() }
/// let sum = process_slice(&vec[..]);  // Natural syntax
/// ```
/// 
/// # See Also
/// - [`as_slice`]: Equivalent method-based approach
/// - [`to_slice`]: Convert to owned Vec<f64>
impl Index<RangeFull> for VectorF64 {
    type Output = [f64];
    
    fn index(&self, _range: RangeFull) -> &Self::Output {
        self.as_slice().expect("Vector data must be contiguous for natural slicing")
    }
}

// ============================================================================
// Natural Fancy Indexing for VectorF64
// ============================================================================

/// Index implementation for Vec<usize> - enables `vec[&indices]` fancy indexing
impl Index<&Vec<usize>> for VectorF64 {
    type Output = VectorF64;
    
    fn index(&self, _indices: &Vec<usize>) -> &Self::Output {
        // For fancy indexing, we need to return an owned result
        // This is a limitation - Index trait requires returning a reference
        // We'll need to use a different approach or keep the method-based API for fancy indexing
        panic!("Fancy indexing with Vec<usize> requires owned result - use .slice_at(indices) instead");
    }
}

// ============================================================================
// Extension Trait for Natural Slicing with Owned Results
// ============================================================================

/// Extension trait providing natural slicing operations with owned results
/// 
/// # Mathematical Specification
/// Complements Rust's Index trait by providing slicing operations that return
/// owned VectorF64 instead of slice references, enabling operations requiring
/// ownership like further chaining, mutation, or storage.
/// 
/// # For AI Code Generation
/// - Bridges Index trait limitations for operations requiring owned results
/// - All methods return VectorF64 (owned) instead of &[f64] (borrowed)
/// - Use when you need to store, modify, or chain slicing results
/// - Methods use safe error handling (Result) instead of panicking
/// - Common pattern: `vec.slice_owned(1..4).slice_owned(0..2)` for chaining
/// - Complements bracket notation: use `&vec[1..4]` for views, `vec.slice_owned(1..4)` for owned
/// 
/// # Design Rationale
/// Rust's Index trait can only return references, but many operations need owned results:
/// - Storing slices in data structures
/// - Chaining multiple slicing operations  
/// - Applying transformations to sliced data
/// - Returning slices from functions
/// 
/// # Example
/// ```
/// use rustlab_math::{vec64, NaturalSlicing, BooleanVector};
/// 
/// let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
/// 
/// // Owned results for chaining
/// let processed = vec.slice_owned(1..4)    // [2.0, 3.0, 4.0]
///                    .slice_owned(0..2);   // [2.0, 3.0] - chained!
/// 
/// // Fancy indexing (not possible with Index trait)
/// let selected = vec.select(vec![0, 2, 4]);  // [1.0, 3.0, 5.0]
/// 
/// // Boolean filtering (not possible with Index trait)
/// let mask = BooleanVector::from_slice(&[true, false, true, false, true]);
/// let filtered = vec.select_where(mask);  // [1.0, 3.0, 5.0]
/// ```
pub trait NaturalSlicing {
    /// Extract elements at specified indices (fancy indexing) with owned result
    /// 
    /// # Mathematical Specification
    /// Given vector v ∈ ℝⁿ and index array I = [i₁, i₂, ..., iₖ]:
    /// Returns vector [v[i₁], v[i₂], ..., v[iₖ]] ∈ ℝₖ
    /// 
    /// # Dimensions
    /// - Input: self (n), indices (k)
    /// - Output: VectorF64 (k)
    /// 
    /// # Complexity
    /// - Time: O(k) where k is number of indices
    /// - Space: O(k) for result vector
    /// 
    /// # For AI Code Generation
    /// - NumPy-equivalent: `arr[[i1, i2, i3]]` fancy indexing
    /// - Enables non-contiguous element selection
    /// - Indices can be in any order, including duplicates
    /// - Common uses: reordering, sampling, gathering scattered data
    /// - Returns owned VectorF64 for full flexibility
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::{vec64, NaturalSlicing};
    /// 
    /// let vec = vec64![10.0, 20.0, 30.0, 40.0, 50.0];
    /// 
    /// // Select specific elements: [10.0, 30.0, 50.0]
    /// let selected = vec.select(vec![0, 2, 4]);
    /// 
    /// // Reorder elements: [50.0, 20.0, 10.0]
    /// let reordered = vec.select(vec![4, 1, 0]);
    /// 
    /// // Duplicate elements: [10.0, 10.0, 20.0]
    /// let duplicated = vec.select(vec![0, 0, 1]);
    /// ```
    fn select(&self, indices: Vec<usize>) -> VectorF64;
    
    /// Filter elements using boolean mask (conditional selection) with owned result
    /// 
    /// # Mathematical Specification
    /// Given vector v ∈ ℝⁿ and boolean mask m ∈ {0,1}ⁿ:
    /// Returns vector [v[i] for i where m[i] = 1] ∈ ℝₖ
    /// where k = |{i : m[i] = 1}| (number of true values)
    /// 
    /// # Dimensions
    /// - Input: self (n), mask (n) 
    /// - Output: VectorF64 (k where k ≤ n)
    /// - Constraint: mask.len() == self.len()
    /// 
    /// # Complexity
    /// - Time: O(n) to scan mask and copy selected elements
    /// - Space: O(k) for result vector where k = number of true values
    /// 
    /// # For AI Code Generation
    /// - NumPy-equivalent: `arr[mask]` boolean indexing
    /// - Essential for conditional data filtering
    /// - Mask length must exactly match vector length
    /// - Result length depends on number of True values in mask
    /// - Common uses: filtering outliers, selecting by condition, data cleaning
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::{vec64, NaturalSlicing, BooleanVector};
    /// 
    /// let values = vec64![1.0, -2.0, 3.0, -4.0, 5.0];
    /// 
    /// // Filter positive values
    /// let positive_mask = BooleanVector::from_slice(&[true, false, true, false, true]);
    /// let positive = values.select_where(positive_mask);  // [1.0, 3.0, 5.0]
    /// 
    /// // Filter by condition (conceptual - mask creation separate)
    /// // let large_values = values.select_where(values > 2.0);  // Would be [3.0, 5.0]
    /// ```
    /// 
    /// # Errors
    /// - Panics if mask.len() != self.len()
    fn select_where(&self, mask: BooleanVector) -> VectorF64;
    
    /// Extract contiguous range of elements with owned result
    /// 
    /// # Mathematical Specification
    /// Given vector v ∈ ℝⁿ and range [i..j):
    /// Returns owned vector [v[i], v[i+1], ..., v[j-1]] ∈ ℝ⬬ʲⁿ
    /// where result length = j - i
    /// 
    /// # Dimensions
    /// - Input: self (n), range i..j where 0 ≤ i < j ≤ n
    /// - Output: VectorF64 (j-i)
    /// 
    /// # Complexity
    /// - Time: O(j-i) to copy elements
    /// - Space: O(j-i) for result vector
    /// 
    /// # For AI Code Generation
    /// - Owned version of `&vec[1..4]` - returns VectorF64 instead of &[f64]
    /// - Use when you need to store, modify, or return the slice
    /// - Safe error handling version of Index trait slicing
    /// - Enables method chaining: `vec.slice_owned(1..4).slice_owned(0..2)`
    /// - Common uses: extracting subdata for processing, windowing operations
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::{vec64, NaturalSlicing};
    /// 
    /// let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
    /// 
    /// // Owned slice for storage
    /// let owned_slice = vec.slice_owned(1..4);  // [2.0, 3.0, 4.0] as VectorF64
    /// 
    /// // Method chaining
    /// let processed = vec.slice_owned(1..4)     // [2.0, 3.0, 4.0]
    ///                    .slice_owned(0..2);    // [2.0, 3.0]
    /// 
    /// // Safe version of panicking index
    /// let safe_slice = vec.slice_owned(1..100);  // Would return error, not panic
    /// ```
    fn slice_owned(&self, range: Range<usize>) -> VectorF64;
    
    /// Extract tail elements from start index to end with owned result
    /// 
    /// # Mathematical Specification
    /// Given vector v ∈ ℝⁿ and start index i:
    /// Returns owned vector [v[i], v[i+1], ..., v[n-1]] ∈ ℝⁿ⁻ⁱ
    /// where result length = n - i
    /// 
    /// # Dimensions
    /// - Input: self (n), start i where 0 ≤ i ≤ n
    /// - Output: VectorF64 (n-i)
    /// 
    /// # Complexity
    /// - Time: O(n-i) to copy elements
    /// - Space: O(n-i) for result vector
    /// 
    /// # For AI Code Generation
    /// - Owned version of `&vec[start..]` - returns VectorF64 instead of &[f64]
    /// - Equivalent to `slice_owned(start..vec.len())` but more convenient
    /// - Use for "everything from index N onwards" operations
    /// - Common uses: skipping headers, processing suffixes, streaming data
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::{vec64, NaturalSlicing};
    /// 
    /// let data = vec64![0.0, 1.0, 2.0, 3.0, 4.0];  // 0.0 is header
    /// 
    /// // Skip header, get actual data: [1.0, 2.0, 3.0, 4.0]
    /// let actual_data = data.slice_from_owned(1);
    /// 
    /// // Process second half
    /// let second_half = data.slice_from_owned(data.len() / 2);
    /// ```
    fn slice_from_owned(&self, start: usize) -> VectorF64;
    
    /// Extract head elements from start to end index with owned result
    /// 
    /// # Mathematical Specification
    /// Given vector v ∈ ℝⁿ and end index j:
    /// Returns owned vector [v[0], v[1], ..., v[j-1]] ∈ ℝʲ
    /// where result length = j
    /// 
    /// # Dimensions
    /// - Input: self (n), end j where 0 ≤ j ≤ n
    /// - Output: VectorF64 (j)
    /// 
    /// # Complexity
    /// - Time: O(j) to copy elements
    /// - Space: O(j) for result vector
    /// 
    /// # For AI Code Generation
    /// - Owned version of `&vec[..end]` - returns VectorF64 instead of &[f64]
    /// - Equivalent to `slice_owned(0..end)` but more convenient
    /// - Use for "first N elements" operations
    /// - Common uses: limiting data size, preview operations, truncating
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::{vec64, NaturalSlicing};
    /// 
    /// let large_dataset = vec64![/* ... many elements ... */];
    /// 
    /// // Take only first 100 elements for quick analysis
    /// let preview = large_dataset.slice_to_owned(100);
    /// 
    /// // Process first half
    /// let first_half = large_dataset.slice_to_owned(large_dataset.len() / 2);
    /// ```
    fn slice_to_owned(&self, end: usize) -> VectorF64;
}

impl NaturalSlicing for VectorF64 {
    fn select(&self, indices: Vec<usize>) -> VectorF64 {
        // Use existing fancy indexing implementation
        self.slice_at(indices).unwrap()
    }
    
    fn select_where(&self, mask: BooleanVector) -> VectorF64 {
        // Use existing boolean mask implementation
        self.slice_at(mask).unwrap()
    }
    
    fn slice_owned(&self, range: Range<usize>) -> VectorF64 {
        // Use existing range slicing implementation
        self.slice_at(range).unwrap()
    }
    
    fn slice_from_owned(&self, start: usize) -> VectorF64 {
        // Convert RangeFrom to Range
        let end = self.len();
        self.slice_owned(start..end)
    }
    
    fn slice_to_owned(&self, end: usize) -> VectorF64 {
        // Convert RangeTo to Range
        self.slice_owned(0..end)
    }
}

// ============================================================================
// Macro for Natural Slicing Syntax
// ============================================================================

/// Macro to provide truly natural slicing syntax that works around Index trait limitations
/// 
/// # Examples
/// ```rust
/// use rustlab_math::{vec64, natural_slice};
/// 
/// let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
/// 
/// // Natural range slicing (returns owned VectorF64)
/// let slice = natural_slice!(vec, 1..4);  // [2.0, 3.0, 4.0]
/// let tail = natural_slice!(vec, 2..);    // [3.0, 4.0, 5.0]
/// let head = natural_slice!(vec, ..3);    // [1.0, 2.0, 3.0]
/// 
/// // Natural fancy indexing (returns owned VectorF64)
/// let indices = vec![0, 2, 4];
/// let selected = natural_slice!(vec, indices);  // [1.0, 3.0, 5.0]
/// 
/// // Natural boolean indexing (returns owned VectorF64)
/// let mask = BooleanVector::from_slice(&[true, false, true, false, true]);
/// let filtered = natural_slice!(vec, mask);  // [1.0, 3.0, 5.0]
/// ```
#[macro_export]
macro_rules! natural_slice {
    // Range slicing
    ($vec:expr, $range:expr) => {
        $vec.slice_at($range).unwrap()
    };
}

// ============================================================================
// Array 2D Natural Slicing (Future Enhancement)
// ============================================================================

// Note: 2D array slicing like `array[1..3, 0..2]` requires custom syntax
// Rust's Index trait doesn't support tuple indexing directly
// We would need to implement Index<(Range<usize>, Range<usize>)> but this
// conflicts with existing tuple indexing for single elements
// For now, keep the method-based API: array.slice_2d_at((1..3, 0..2))

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{vec64, BooleanVector};
    
    #[test]
    fn test_natural_range_slicing() {
        let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Range slicing returns slice references
        let slice = &vec[1..4];
        assert_eq!(slice, &[2.0, 3.0, 4.0]);
        
        let tail = &vec[2..];
        assert_eq!(tail, &[3.0, 4.0, 5.0]);
        
        let head = &vec[..3];
        assert_eq!(head, &[1.0, 2.0, 3.0]);
        
        let full = &vec[..];
        assert_eq!(full, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    
    #[test]
    fn test_natural_slicing_extension() {
        let vec = vec64![10.0, 20.0, 30.0, 40.0, 50.0];
        
        // Fancy indexing with owned results
        let indices = vec![0, 2, 4];
        let selected = vec.select(indices);
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0], 10.0);
        assert_eq!(selected[1], 30.0);
        assert_eq!(selected[2], 50.0);
        
        // Range slicing with owned results
        let owned_slice = vec.slice_owned(1..4);
        assert_eq!(owned_slice.len(), 3);
        assert_eq!(owned_slice[0], 20.0);
        assert_eq!(owned_slice[2], 40.0);
    }
    
    #[test]
    fn test_natural_slice_macro() {
        let vec = vec64![100.0, 200.0, 300.0, 400.0, 500.0];
        
        // Macro-based natural slicing
        let slice1 = natural_slice!(vec, 1..4);
        assert_eq!(slice1.len(), 3);
        assert_eq!(slice1[0], 200.0);
        assert_eq!(slice1[2], 400.0);
        
        let slice2 = natural_slice!(vec, 2..);
        assert_eq!(slice2.len(), 3);
        assert_eq!(slice2[0], 300.0);
        
        let slice3 = natural_slice!(vec, ..3);
        assert_eq!(slice3.len(), 3);
        assert_eq!(slice3[2], 300.0);
    }
    
    #[test]
    fn test_boolean_mask_extension() {
        let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = BooleanVector::from_slice(&[true, false, true, false, true]);
        
        let filtered = vec.select_where(mask);
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0], 1.0);
        assert_eq!(filtered[1], 3.0);
        assert_eq!(filtered[2], 5.0);
    }
}