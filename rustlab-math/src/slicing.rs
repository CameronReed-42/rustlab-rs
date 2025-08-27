//! Range-Based Slicing Operations with AI-Optimized Documentation
//!
//! This module provides comprehensive NumPy-style range slicing capabilities with
//! zero-copy views and efficient memory management for scientific computing workflows.
//!
//! # Core Slicing Capabilities  
//! - **Vector slicing**: `vec[1..4]`, `vec[..3]`, `vec[2..]` - Contiguous element extraction
//! - **Array slicing**: `array[1..3, 2..5]`, `array[.., 1..3]` - 2D submatrix operations
//! - **Zero-copy views**: Memory-efficient reference-based slicing when possible
//! - **Owned results**: Conversion to owned containers when needed for flexibility
//!
//! # For AI Code Generation
//! This module complements [`ergonomic_slicing`] and [`natural_slicing`] by providing
//! the foundational slicing infrastructure. Key patterns:
//!
//! ```rust
//! use rustlab_math::{VectorF64, ArrayF64, vec64, array64};
//!
//! let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
//! let matrix = array64![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//!
//! // Range slicing with views (zero-copy)
//! let slice_view = data.slice(1..4)?;      // View: [2.0, 3.0, 4.0]
//! let matrix_view = matrix.slice((0..2, 1..3))?; // 2D view
//!
//! // Convert views to owned containers  
//! let owned_slice = slice_view.to_vector();     // Owned VectorF64
//! let owned_matrix = matrix_view.to_array();    // Owned ArrayF64
//! ```
//!
//! # Performance Characteristics
//! - **Views**: O(1) creation, zero memory allocation
//! - **Owned conversion**: O(k) where k is slice size, single allocation
//! - **Memory layout**: Preserves cache locality for sliced operations
//! - **SIMD compatibility**: Sliced data works with vectorized operations
//!
//! # Cross-Module Integration
//! - Provides views for [`ergonomic_slicing`] advanced indexing operations
//! - Works with [`natural_slicing`] Index trait implementations
//! - Compatible with [`Array`] and [`Vector`] mathematical operations
//! - Integrates with [`broadcasting`] for sliced data normalization

use crate::{Array, Vector};
use faer_entity::Entity;
use faer_traits::ComplexField;
use std::ops::{Range, RangeTo, RangeFrom, RangeFull};

/// Trait for types that can be used as slice indices with AI-optimized documentation
/// 
/// # For AI Code Generation
/// - Unified interface for different slice index types: Range, RangeTo, RangeFrom, etc.
/// - Enables generic slicing operations across all range types
/// - Provides type-safe slicing with compile-time output type checking
/// - Used internally by slicing infrastructure - rarely called directly
/// - Pattern: implement this trait for custom index types
/// 
/// # Mathematical Specification
/// For container C and index I:
/// slice(I, C) → C' where C' is a view or subset of C based on index specification
/// 
/// # Common Index Types
/// - `Range<usize>`: Explicit start..end bounds
/// - `RangeTo<usize>`: From beginning ..end  
/// - `RangeFrom<usize>`: From start.. to end
/// - `RangeFull`: Entire container ..
/// 
/// # Example
/// ```rust
/// // All these use SliceIndex internally:
/// let slice1 = vector.slice(1..4)?;    // Range<usize>
/// let slice2 = vector.slice(..3)?;     // RangeTo<usize>
/// let slice3 = vector.slice(2..)?;     // RangeFrom<usize>
/// ```
pub trait SliceIndex<T> {
    /// The output type when slicing
    type Output;
    
    /// Perform the slicing operation
    /// 
    /// # For AI Code Generation
    /// - Core slicing method that creates views or owned results
    /// - Input validation and bounds checking performed here
    /// - Returns appropriate container type based on slice requirements
    /// - May return view (zero-copy) or owned (allocated) depending on implementation
    fn slice(self, target: T) -> Self::Output;
}

/// A sliced vector view that references a portion of the original vector with AI-optimized documentation
/// 
/// # For AI Code Generation
/// - Zero-copy view into existing vector data (no allocation)
/// - Lifetime-bound to original vector - cannot outlive source data
/// - Provides safe element access with bounds checking
/// - Efficient for temporary operations on subvectors
/// - Convert to owned `Vector<T>` when permanence needed
/// - Common pattern: slice → process view → convert to owned if storing
/// 
/// # Mathematical Specification
/// Given vector v ∈ ℝⁿ and slice parameters (start, len):
/// View represents subvector v[start..start+len] with zero additional memory
/// 
/// # Complexity
/// - Creation: O(1) - just stores references and bounds
/// - Element access: O(1) - direct indexing with bounds check
/// - Conversion to owned: O(len) - single allocation and copy
/// 
/// # Lifetime Management
/// - Borrows from source vector - source must remain valid
/// - Cannot modify source while view exists (Rust borrowing rules)
/// - Use owned operations when concurrent modification needed
/// 
/// # Example
/// ```rust
/// use rustlab_math::VectorF64;
/// 
/// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// {
///     let view = data.slice(1..4)?;        // Zero-copy view: [2.0, 3.0, 4.0]
///     let sum = view.as_slice().iter().sum::<f64>();  // Process efficiently
///     println!("Sum of slice: {}", sum);    // 9.0
/// } // view dropped here
/// 
/// // Convert to owned for storage
/// let permanent = data.slice(1..4)?.to_vector();  // Owned copy
/// ```
#[derive(Debug)]
pub struct SlicedVectorView<'a, T: Entity> {
    /// Reference to the original vector data
    data: &'a [T],
    /// Start index in the original vector
    start: usize,
    /// Length of the slice
    len: usize,
}

impl<'a, T: Entity> SlicedVectorView<'a, T> {
    /// Create a new sliced vector view
    pub fn new(data: &'a [T], start: usize, len: usize) -> Self {
        Self { data, start, len }
    }
    
    /// Get the length of the slice
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if the slice is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get element at index (bounds checked)
    /// 
    /// # For AI Code Generation
    /// - Safe element access with bounds checking
    /// - Returns None for out-of-bounds access (no panic)
    /// - Index is relative to slice start (0-based within slice)
    /// - Use for safe iteration and element inspection
    /// - Preferred over direct indexing for safety
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let slice = data.slice(1..3).unwrap();  // [2.0, 3.0]
    /// assert_eq!(slice.get(0), Some(&2.0));   // First element of slice
    /// assert_eq!(slice.get(1), Some(&3.0));   // Second element of slice
    /// assert_eq!(slice.get(2), None);         // Out of bounds
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            Some(&self.data[self.start + index])
        } else {
            None
        }
    }
    
    /// Get the slice as a standard Rust slice
    pub fn as_slice(&self) -> &[T] {
        &self.data[self.start..self.start + self.len]
    }
    
    /// Convert to owned vector
    pub fn to_vec(&self) -> Vec<T> 
    where 
        T: Clone 
    {
        self.as_slice().to_vec()
    }
    
    /// Convert to owned Vector<T>
    pub fn to_vector(&self) -> Vector<T>
    where
        T: Clone + ComplexField,
    {
        Vector::from_slice(self.as_slice())
    }
}

/// A sliced mutable vector view that references a portion of the original vector
#[derive(Debug)]
pub struct SlicedVectorViewMut<'a, T: Entity> {
    /// Mutable reference to the original vector data
    data: &'a mut [T],
    /// Start index in the original vector
    start: usize,
    /// Length of the slice
    len: usize,
}

impl<'a, T: Entity> SlicedVectorViewMut<'a, T> {
    /// Create a new mutable sliced vector view
    pub fn new(data: &'a mut [T], start: usize, len: usize) -> Self {
        Self { data, start, len }
    }
    
    /// Get the length of the slice
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if the slice is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get element at index (bounds checked)
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            Some(&self.data[self.start + index])
        } else {
            None
        }
    }
    
    /// Get mutable element at index (bounds checked)
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            Some(&mut self.data[self.start + index])
        } else {
            None
        }
    }
    
    /// Get the slice as a standard Rust slice
    pub fn as_slice(&self) -> &[T] {
        &self.data[self.start..self.start + self.len]
    }
    
    /// Get the slice as a mutable standard Rust slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[self.start..self.start + self.len]
    }
    
    /// Convert to owned vector
    pub fn to_vec(&self) -> Vec<T> 
    where 
        T: Clone 
    {
        self.as_slice().to_vec()
    }
    
    /// Convert to owned Vector<T>
    pub fn to_vector(&self) -> Vector<T>
    where
        T: Clone + ComplexField,
    {
        Vector::from_slice(self.as_slice())
    }
}

// Implement range slicing for vectors

impl<T: Entity> Vector<T> {
    /// Slice the vector with a range
    /// 
    /// # Mathematical Specification
    /// For vector v ∈ ℝⁿ and range [start, end):
    /// slice(v, start..end) = [v[start], v[start+1], ..., v[end-1]]
    /// Creates a view with length = end - start
    /// 
    /// # Complexity
    /// - Time: O(1) - creates view without copying data
    /// - Space: O(1) - only stores reference + metadata
    /// 
    /// # For AI Code Generation
    /// - Creates zero-copy view of vector subsequence
    /// - Range is half-open: includes start, excludes end
    /// - Essential for data processing: train/test splits, batching, windowing
    /// - Bounds checking prevents runtime panics
    /// - Returns Result for safe error handling
    /// - View shares lifetime with original vector
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// // Data preprocessing: extract feature subsets
    /// let features = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let subset = features.slice(1..4).unwrap();  // [2.0, 3.0, 4.0]
    /// 
    /// // Train/test split
    /// let train_size = (0.8 * features.len() as f64) as usize;
    /// let train = features.slice(0..train_size).unwrap();
    /// let test = features.slice(train_size..features.len()).unwrap();
    /// ```
    /// 
    /// # Errors
    /// Returns error if:
    /// - start > end (invalid range)
    /// - end > vector.len() (out of bounds)
    /// - Vector data is not contiguous (rare case)
    /// 
    /// # See Also
    /// - [`slice_to`]: Slice from start to specified end
    /// - [`slice_from`]: Slice from specified start to end
    /// - [`Array::slice_2d`]: 2D matrix slicing
    pub fn slice(&self, range: Range<usize>) -> Result<SlicedVectorView<'_, T>, String> {
        if let Some(data) = self.as_slice() {
            if range.start <= range.end && range.end <= data.len() {
                Ok(SlicedVectorView::new(data, range.start, range.end - range.start))
            } else {
                Err(format!("Range {:?} is out of bounds for vector of length {}", range, data.len()))
            }
        } else {
            Err("Vector data is not contiguous - cannot create slice view".to_string())
        }
    }
    
    /// Slice the vector with a range (mutable)
    pub fn slice_mut(&mut self, range: Range<usize>) -> Result<SlicedVectorViewMut<'_, T>, String> {
        let len = self.len();
        if let Some(data) = self.as_mut_slice() {
            if range.start <= range.end && range.end <= len {
                Ok(SlicedVectorViewMut::new(data, range.start, range.end - range.start))
            } else {
                Err(format!("Range {:?} is out of bounds for vector of length {}", range, len))
            }
        } else {
            Err("Vector data is not contiguous - cannot create mutable slice view".to_string())
        }
    }
    
    /// Slice from start to end (exclusive)
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `slice(0..end)` - convenience method
    /// - Creates view from vector start up to (but not including) end
    /// - Common pattern: get first N elements
    /// - Use for head/prefix operations, data truncation
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let head = data.slice_to(3).unwrap();  // [1.0, 2.0, 3.0]
    /// 
    /// // Common ML pattern: get first batch
    /// let batch_size = 64;
    /// let first_batch = data.slice_to(batch_size.min(data.len())).unwrap();
    /// ```
    pub fn slice_to(&self, end: usize) -> Result<SlicedVectorView<'_, T>, String> {
        self.slice(0..end)
    }
    
    /// Slice from start to end (mutable)
    pub fn slice_to_mut(&mut self, end: usize) -> Result<SlicedVectorViewMut<'_, T>, String> {
        self.slice_mut(0..end)
    }
    
    /// Slice from start to the end of the vector
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `slice(start..vector.len())` - convenience method
    /// - Creates view from specified start to vector end
    /// - Common pattern: skip first N elements, get remainder
    /// - Use for tail/suffix operations, data skipping
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let tail = data.slice_from(2).unwrap();  // [3.0, 4.0, 5.0]
    /// 
    /// // Skip header row in data processing
    /// let skip_header = 1;
    /// let content = data.slice_from(skip_header).unwrap();
    /// ```
    pub fn slice_from(&self, start: usize) -> Result<SlicedVectorView<'_, T>, String> {
        let len = self.len();
        self.slice(start..len)
    }
    
    /// Slice from start to the end of the vector (mutable)
    pub fn slice_from_mut(&mut self, start: usize) -> Result<SlicedVectorViewMut<'_, T>, String> {
        let len = self.len();
        self.slice_mut(start..len)
    }
}

// Index trait for Range slicing is now implemented in natural_slicing.rs for VectorF64 specifically
// Generic Index implementation for Vector<T> is complex due to lifetime issues with views
// For now, use method-based slicing: .slice_range(), .slice_range_from(), etc.

// For now, let's implement method-based slicing rather than Index trait
// because the Index trait requires returning references, which is complex with views

/// Trait for creating vector slices with different range types
pub trait VectorSlicing<T: Entity> {
    /// Slice with a Range<usize>
    fn slice_range(&self, range: Range<usize>) -> Result<SlicedVectorView<'_, T>, String>;
    
    /// Slice with a RangeFrom<usize> (start..)
    fn slice_range_from(&self, range: RangeFrom<usize>) -> Result<SlicedVectorView<'_, T>, String>;
    
    /// Slice with a RangeTo<usize> (..end)
    fn slice_range_to(&self, range: RangeTo<usize>) -> Result<SlicedVectorView<'_, T>, String>;
    
    /// Slice with a RangeFull (..)
    fn slice_range_full(&self, range: RangeFull) -> Result<SlicedVectorView<'_, T>, String>;
}

impl<T: Entity> VectorSlicing<T> for Vector<T> {
    fn slice_range(&self, range: Range<usize>) -> Result<SlicedVectorView<'_, T>, String> {
        self.slice(range)
    }
    
    fn slice_range_from(&self, range: RangeFrom<usize>) -> Result<SlicedVectorView<'_, T>, String> {
        let len = self.len();
        self.slice(range.start..len)
    }
    
    fn slice_range_to(&self, range: RangeTo<usize>) -> Result<SlicedVectorView<'_, T>, String> {
        self.slice(0..range.end)
    }
    
    fn slice_range_full(&self, _range: RangeFull) -> Result<SlicedVectorView<'_, T>, String> {
        let len = self.len();
        self.slice(0..len)
    }
}

// Type aliases for common slice types
/// Immutable f64 vector slice
pub type VectorSliceF64<'a> = SlicedVectorView<'a, f64>;
/// Mutable f64 vector slice  
pub type VectorSliceMutF64<'a> = SlicedVectorViewMut<'a, f64>;
/// Immutable f32 vector slice
pub type VectorSliceF32<'a> = SlicedVectorView<'a, f32>;
/// Mutable f32 vector slice
pub type VectorSliceMutF32<'a> = SlicedVectorViewMut<'a, f32>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorF64;
    
    #[test]
    fn test_vector_range_slicing() {
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Test range slicing
        let slice = vec.slice(1..4).unwrap();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.get(0), Some(&2.0));
        assert_eq!(slice.get(1), Some(&3.0));
        assert_eq!(slice.get(2), Some(&4.0));
        assert_eq!(slice.get(3), None);
        
        // Test slice_to
        let slice_to = vec.slice_to(3).unwrap();
        assert_eq!(slice_to.len(), 3);
        assert_eq!(slice_to.as_slice(), &[1.0, 2.0, 3.0]);
        
        // Test slice_from
        let slice_from = vec.slice_from(2).unwrap();
        assert_eq!(slice_from.len(), 3);
        assert_eq!(slice_from.as_slice(), &[3.0, 4.0, 5.0]);
    }
    
    #[test]
    fn test_vector_mutable_slicing() {
        let mut vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Test mutable slicing
        {
            let mut slice = vec.slice_mut(1..4).unwrap();
            assert_eq!(slice.len(), 3);
            
            // Modify through slice
            *slice.get_mut(0).unwrap() = 20.0;
            *slice.get_mut(2).unwrap() = 40.0;
        }
        
        // Verify changes
        let slice = vec.slice(0..5).unwrap();
        assert_eq!(slice.as_slice(), &[1.0, 20.0, 3.0, 40.0, 5.0]);
    }
    
    #[test]
    fn test_slicing_error_handling() {
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
        
        // Test out of bounds
        assert!(vec.slice(0..5).is_err());
        assert!(vec.slice(2..1).is_err());
        assert!(vec.slice_to(5).is_err());
        assert!(vec.slice_from(5).is_err());
    }
    
    #[test]
    fn test_vector_slicing_trait() {
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Test different range types
        let range_slice = vec.slice_range(1..3).unwrap();
        assert_eq!(range_slice.as_slice(), &[2.0, 3.0]);
        
        let from_slice = vec.slice_range_from(2..).unwrap();
        assert_eq!(from_slice.as_slice(), &[3.0, 4.0, 5.0]);
        
        let to_slice = vec.slice_range_to(..3).unwrap();
        assert_eq!(to_slice.as_slice(), &[1.0, 2.0, 3.0]);
        
        let full_slice = vec.slice_range_full(..).unwrap();
        assert_eq!(full_slice.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}

// ========== ARRAY (2D) SLICING ==========

/// A sliced array view that references a submatrix of the original array
#[derive(Debug)]
pub struct SlicedArrayView<'a, T: Entity> {
    /// Reference to the original array
    array: &'a Array<T>,
    /// Row range
    row_range: Range<usize>,
    /// Column range
    col_range: Range<usize>,
}

impl<'a, T: Entity> SlicedArrayView<'a, T> {
    /// Create a new sliced array view
    pub fn new(array: &'a Array<T>, row_range: Range<usize>, col_range: Range<usize>) -> Self {
        Self { array, row_range, col_range }
    }
    
    /// Get the shape of the slice (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.row_range.len(), self.col_range.len())
    }
    
    /// Get the number of rows in the slice
    pub fn nrows(&self) -> usize {
        self.row_range.len()
    }
    
    /// Get the number of columns in the slice
    pub fn ncols(&self) -> usize {
        self.col_range.len()
    }
    
    /// Get element at position (row, col) within the slice
    /// 
    /// # For AI Code Generation
    /// - Safe element access with bounds checking on slice dimensions
    /// - Coordinates are relative to slice origin (0-based within slice)
    /// - Automatically translates to original matrix coordinates
    /// - Returns None for out-of-bounds access (no panic)
    /// - Use for safe element inspection and iteration
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    /// let slice = matrix.slice_2d(0..2, 1..2).unwrap();  // Middle column
    /// 
    /// assert_eq!(slice.get(0, 0), Some(2.0));  // matrix[0, 1]
    /// assert_eq!(slice.get(1, 0), Some(5.0));  // matrix[1, 1]
    /// assert_eq!(slice.get(0, 1), None);       // Out of slice bounds
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Option<T>
    where
        T: Clone + ComplexField,
    {
        if row < self.nrows() && col < self.ncols() {
            let actual_row = self.row_range.start + row;
            let actual_col = self.col_range.start + col;
            self.array.get(actual_row, actual_col)
        } else {
            None
        }
    }
    
    /// Convert slice to owned Array<T>
    pub fn to_array(&self) -> Array<T>
    where
        T: Clone + ComplexField,
    {
        Array::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.get(i, j).unwrap()
        })
    }
    
    /// Get a row from the slice as a vector
    pub fn row(&self, row: usize) -> Option<Vector<T>>
    where
        T: Clone + ComplexField,
    {
        if row < self.nrows() {
            let mut data = Vec::with_capacity(self.ncols());
            for col in 0..self.ncols() {
                data.push(self.get(row, col).unwrap());
            }
            Some(Vector::from_slice(&data))
        } else {
            None
        }
    }
    
    /// Get a column from the slice as a vector
    pub fn col(&self, col: usize) -> Option<Vector<T>>
    where
        T: Clone + ComplexField,
    {
        if col < self.ncols() {
            let mut data = Vec::with_capacity(self.nrows());
            for row in 0..self.nrows() {
                data.push(self.get(row, col).unwrap());
            }
            Some(Vector::from_slice(&data))
        } else {
            None
        }
    }
}

/// A mutable sliced array view that references a submatrix of the original array
#[derive(Debug)]
pub struct SlicedArrayViewMut<'a, T: Entity> {
    /// Mutable reference to the original array
    array: &'a mut Array<T>,
    /// Row range
    row_range: Range<usize>,
    /// Column range
    col_range: Range<usize>,
}

impl<'a, T: Entity> SlicedArrayViewMut<'a, T> {
    /// Create a new mutable sliced array view
    pub fn new(array: &'a mut Array<T>, row_range: Range<usize>, col_range: Range<usize>) -> Self {
        Self { array, row_range, col_range }
    }
    
    /// Get the shape of the slice (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.row_range.len(), self.col_range.len())
    }
    
    /// Get the number of rows in the slice
    pub fn nrows(&self) -> usize {
        self.row_range.len()
    }
    
    /// Get the number of columns in the slice
    pub fn ncols(&self) -> usize {
        self.col_range.len()
    }
    
    /// Get element at position (row, col) within the slice
    pub fn get(&self, row: usize, col: usize) -> Option<T>
    where
        T: Clone + ComplexField,
    {
        if row < self.nrows() && col < self.ncols() {
            let actual_row = self.row_range.start + row;
            let actual_col = self.col_range.start + col;
            self.array.get(actual_row, actual_col)
        } else {
            None
        }
    }
    
    /// Set element at position (row, col) within the slice
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), String>
    where
        T: Clone + ComplexField,
    {
        if row < self.nrows() && col < self.ncols() {
            let actual_row = self.row_range.start + row;
            let actual_col = self.col_range.start + col;
            self.array.set(actual_row, actual_col, value)
                .map_err(|e| format!("Error setting element: {:?}", e))
        } else {
            Err(format!("Index ({}, {}) out of bounds for slice of shape ({}, {})", 
                       row, col, self.nrows(), self.ncols()))
        }
    }
    
    /// Convert slice to owned Array<T>
    pub fn to_array(&self) -> Array<T>
    where
        T: Clone + ComplexField,
    {
        Array::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.get(i, j).unwrap()
        })
    }
}

impl<T: Entity> Array<T> {
    /// Slice the array with row and column ranges
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ and ranges [r₁, r₂), [c₁, c₂):
    /// slice_2d(A, r₁..r₂, c₁..c₂) = submatrix A[r₁:r₂, c₁:c₂]
    /// Result shape: (r₂-r₁) × (c₂-c₁)
    /// 
    /// # Complexity
    /// - Time: O(1) - creates view without copying data
    /// - Space: O(1) - only stores references + range metadata
    /// 
    /// # For AI Code Generation
    /// - Creates zero-copy 2D view into matrix data
    /// - Both ranges are half-open: [start, end)
    /// - Essential for data processing: feature selection, sample selection
    /// - Bounds checking prevents runtime panics
    /// - View shares lifetime with original matrix
    /// - Most flexible slicing method - specify both dimensions
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// // Feature and sample selection
    /// let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    /// // Select first 2 rows, last 2 columns
    /// let subset = data.slice_2d(0..2, 1..3).unwrap();
    /// assert_eq!(subset.shape(), (2, 2));
    /// 
    /// // Train/validation split with feature selection
    /// let train_samples = 80;
    /// let feature_start = 1;
    /// let feature_end = 10;
    /// let train_features = data.slice_2d(0..train_samples, feature_start..feature_end).unwrap();
    /// ```
    /// 
    /// # Errors
    /// Returns error if ranges are invalid or out of bounds
    /// 
    /// # See Also
    /// - [`slice_rows`]: Select specific rows (all columns)
    /// - [`slice_cols`]: Select specific columns (all rows)
    pub fn slice_2d(&self, row_range: Range<usize>, col_range: Range<usize>) -> Result<SlicedArrayView<'_, T>, String> {
        let (nrows, ncols) = self.shape();
        
        if row_range.start <= row_range.end && row_range.end <= nrows &&
           col_range.start <= col_range.end && col_range.end <= ncols {
            Ok(SlicedArrayView::new(self, row_range, col_range))
        } else {
            Err(format!("Range ({:?}, {:?}) is out of bounds for array of shape ({}, {})", 
                       row_range, col_range, nrows, ncols))
        }
    }
    
    /// Slice the array with row and column ranges (mutable)
    pub fn slice_2d_mut(&mut self, row_range: Range<usize>, col_range: Range<usize>) -> Result<SlicedArrayViewMut<'_, T>, String> {
        let (nrows, ncols) = self.shape();
        
        if row_range.start <= row_range.end && row_range.end <= nrows &&
           col_range.start <= col_range.end && col_range.end <= ncols {
            Ok(SlicedArrayViewMut::new(self, row_range, col_range))
        } else {
            Err(format!("Range ({:?}, {:?}) is out of bounds for array of shape ({}, {})", 
                       row_range, col_range, nrows, ncols))
        }
    }
    
    /// Slice specific rows (all columns)
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `slice_2d(row_range, 0..ncols)` - convenience method
    /// - Selects subset of samples/observations, keeps all features
    /// - Most common pattern in ML: sample selection, train/test splits
    /// - Zero-copy operation - no data duplication
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let dataset = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    /// 
    /// // Sample selection: get specific rows
    /// let selected_samples = dataset.slice_rows(0..1).unwrap();  // First row only
    /// 
    /// // Train/test split
    /// let total_samples = dataset.nrows();
    /// let train_size = (0.8 * total_samples as f64) as usize;
    /// let train_set = dataset.slice_rows(0..train_size).unwrap();
    /// let test_set = dataset.slice_rows(train_size..total_samples).unwrap();
    /// ```
    pub fn slice_rows(&self, row_range: Range<usize>) -> Result<SlicedArrayView<'_, T>, String> {
        let ncols = self.ncols();
        self.slice_2d(row_range, 0..ncols)
    }
    
    /// Slice specific rows (all columns, mutable)
    pub fn slice_rows_mut(&mut self, row_range: Range<usize>) -> Result<SlicedArrayViewMut<'_, T>, String> {
        let ncols = self.ncols();
        self.slice_2d_mut(row_range, 0..ncols)
    }
    
    /// Slice specific columns (all rows)
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `slice_2d(0..nrows, col_range)` - convenience method
    /// - Selects subset of features, keeps all samples
    /// - Common in ML: feature selection, dimensionality reduction
    /// - Zero-copy operation - efficient for large datasets
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// // Dataset with features and labels
    /// let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    /// 
    /// // Feature selection
    /// let features = data.slice_cols(0..2).unwrap();   // First 2 columns as features
    /// let labels = data.slice_cols(2..3).unwrap();     // Last column as labels
    /// 
    /// // Dimensionality reduction: select top k features
    /// let k_best_features = 5;
    /// let reduced = data.slice_cols(0..k_best_features).unwrap();
    /// ```
    pub fn slice_cols(&self, col_range: Range<usize>) -> Result<SlicedArrayView<'_, T>, String> {
        let nrows = self.nrows();
        self.slice_2d(0..nrows, col_range)
    }
    
    /// Slice specific columns (all rows, mutable)
    pub fn slice_cols_mut(&mut self, col_range: Range<usize>) -> Result<SlicedArrayViewMut<'_, T>, String> {
        let nrows = self.nrows();
        self.slice_2d_mut(0..nrows, col_range)
    }
}

/// Trait for creating array slices with different range types
pub trait ArraySlicing<T: Entity> {
    /// Slice with two Range<usize> for rows and columns
    fn slice_ranges(&self, row_range: Range<usize>, col_range: Range<usize>) -> Result<SlicedArrayView<'_, T>, String>;
    
    /// Slice rows with RangeFrom (..)
    fn slice_rows_from(&self, row_range: RangeFrom<usize>) -> Result<SlicedArrayView<'_, T>, String>;
    
    /// Slice rows with RangeTo (..)
    fn slice_rows_to(&self, row_range: RangeTo<usize>) -> Result<SlicedArrayView<'_, T>, String>;
    
    /// Slice columns with RangeFrom (..)
    fn slice_cols_from(&self, col_range: RangeFrom<usize>) -> Result<SlicedArrayView<'_, T>, String>;
    
    /// Slice columns with RangeTo (..)
    fn slice_cols_to(&self, col_range: RangeTo<usize>) -> Result<SlicedArrayView<'_, T>, String>;
}

impl<T: Entity> ArraySlicing<T> for Array<T> {
    fn slice_ranges(&self, row_range: Range<usize>, col_range: Range<usize>) -> Result<SlicedArrayView<'_, T>, String> {
        self.slice_2d(row_range, col_range)
    }
    
    fn slice_rows_from(&self, row_range: RangeFrom<usize>) -> Result<SlicedArrayView<'_, T>, String> {
        let nrows = self.nrows();
        self.slice_rows(row_range.start..nrows)
    }
    
    fn slice_rows_to(&self, row_range: RangeTo<usize>) -> Result<SlicedArrayView<'_, T>, String> {
        self.slice_rows(0..row_range.end)
    }
    
    fn slice_cols_from(&self, col_range: RangeFrom<usize>) -> Result<SlicedArrayView<'_, T>, String> {
        let ncols = self.ncols();
        self.slice_cols(col_range.start..ncols)
    }
    
    fn slice_cols_to(&self, col_range: RangeTo<usize>) -> Result<SlicedArrayView<'_, T>, String> {
        self.slice_cols(0..col_range.end)
    }
}

// Additional type aliases for array slices
/// Immutable f64 array slice
pub type ArraySliceF64<'a> = SlicedArrayView<'a, f64>;
/// Mutable f64 array slice  
pub type ArraySliceMutF64<'a> = SlicedArrayViewMut<'a, f64>;
/// Immutable f32 array slice
pub type ArraySliceF32<'a> = SlicedArrayView<'a, f32>;
/// Mutable f32 array slice
pub type ArraySliceMutF32<'a> = SlicedArrayViewMut<'a, f32>;

// Tests for array slicing
#[cfg(test)]
mod array_tests {
    use super::*;
    use crate::ArrayF64;
    
    #[test]
    fn test_array_2d_slicing() {
        // Create a 3x4 array: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 4).unwrap();
        
        // Test 2D slicing
        let slice = arr.slice_2d(1..3, 1..3).unwrap(); // rows 1-2, cols 1-2
        assert_eq!(slice.shape(), (2, 2));
        assert_eq!(slice.get(0, 0), Some(6.0)); // arr[1,1]
        assert_eq!(slice.get(0, 1), Some(7.0)); // arr[1,2]
        assert_eq!(slice.get(1, 0), Some(10.0)); // arr[2,1]
        assert_eq!(slice.get(1, 1), Some(11.0)); // arr[2,2]
        
        // Test row slicing
        let row_slice = arr.slice_rows(1..2).unwrap(); // middle row
        assert_eq!(row_slice.shape(), (1, 4));
        assert_eq!(row_slice.get(0, 0), Some(5.0));
        assert_eq!(row_slice.get(0, 3), Some(8.0));
        
        // Test column slicing
        let col_slice = arr.slice_cols(2..4).unwrap(); // last 2 columns
        assert_eq!(col_slice.shape(), (3, 2));
        assert_eq!(col_slice.get(0, 0), Some(3.0)); // arr[0,2]
        assert_eq!(col_slice.get(2, 1), Some(12.0)); // arr[2,3]
    }
    
    #[test]
    fn test_array_mutable_slicing() {
        let mut arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        
        // Test mutable slicing
        {
            let mut slice = arr.slice_2d_mut(0..2, 1..2).unwrap(); // middle column
            assert_eq!(slice.shape(), (2, 1));
            
            // Modify through slice
            slice.set(0, 0, 20.0).unwrap(); // arr[0,1] = 20.0
            slice.set(1, 0, 60.0).unwrap(); // arr[1,1] = 60.0
        }
        
        // Verify changes
        assert_eq!(arr.get(0, 1), Some(20.0));
        assert_eq!(arr.get(1, 1), Some(60.0));
        // Other elements unchanged
        assert_eq!(arr.get(0, 0), Some(1.0));
        assert_eq!(arr.get(0, 2), Some(3.0));
    }
    
    #[test]
    fn test_slice_to_array_conversion() {
        let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();
        
        // Create a slice and convert to owned array
        let slice = arr.slice_2d(1..3, 0..2).unwrap(); // bottom-left 2x2 submatrix
        let owned = slice.to_array();
        
        assert_eq!(owned.shape(), (2, 2));
        assert_eq!(owned.get(0, 0), Some(4.0)); // arr[1,0]
        assert_eq!(owned.get(0, 1), Some(5.0)); // arr[1,1]
        assert_eq!(owned.get(1, 0), Some(7.0)); // arr[2,0]
        assert_eq!(owned.get(1, 1), Some(8.0)); // arr[2,1]
    }
    
    #[test]
    fn test_slice_row_and_col_extraction() {
        let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let slice = arr.slice_2d(0..2, 1..3).unwrap(); // columns 1-2
        
        // Extract row from slice
        let row = slice.row(0).unwrap();
        assert_eq!(row.len(), 2);
        assert_eq!(row.get(0), Some(2.0)); // arr[0,1]
        assert_eq!(row.get(1), Some(3.0)); // arr[0,2]
        
        // Extract column from slice
        let col = slice.col(1).unwrap();
        assert_eq!(col.len(), 2);
        assert_eq!(col.get(0), Some(3.0)); // arr[0,2]
        assert_eq!(col.get(1), Some(6.0)); // arr[1,2]
    }
    
    #[test]
    fn test_array_slicing_error_handling() {
        let arr = ArrayF64::zeros(3, 3);
        
        // Test out of bounds
        assert!(arr.slice_2d(0..5, 0..2).is_err()); // row range too large
        assert!(arr.slice_2d(0..2, 0..5).is_err()); // col range too large
        assert!(arr.slice_2d(2..1, 0..2).is_err()); // invalid range
        assert!(arr.slice_rows(0..5).is_err()); // row range too large
        assert!(arr.slice_cols(0..5).is_err()); // col range too large
    }
    
    #[test]
    fn test_array_slicing_trait() {
        let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).unwrap();
        
        // Test different range types
        let range_slice = arr.slice_ranges(1..3, 1..3).unwrap();
        assert_eq!(range_slice.shape(), (2, 2));
        
        let rows_from = arr.slice_rows_from(1..).unwrap();
        assert_eq!(rows_from.shape(), (2, 3)); // rows 1-2
        
        let rows_to = arr.slice_rows_to(..2).unwrap();
        assert_eq!(rows_to.shape(), (2, 3)); // rows 0-1
        
        let cols_from = arr.slice_cols_from(1..).unwrap();
        assert_eq!(cols_from.shape(), (3, 2)); // cols 1-2
        
        let cols_to = arr.slice_cols_to(..2).unwrap();
        assert_eq!(cols_to.shape(), (3, 2)); // cols 0-1
    }
}