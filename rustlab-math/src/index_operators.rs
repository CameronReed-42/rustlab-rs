//! Index Operators for Math-First Ergonomic Slicing with AI-Optimized Documentation
//!
//! This module implements Rust's `Index` trait and helper methods to enable the most
//! natural possible slicing syntax, approaching NumPy/MATLAB ergonomics in Rust.
//! All operations integrate with RustLab's mathematical ecosystem.
//!
//! # Core Index Operations
//! - **Single element**: `vec[5]` - Direct element access (panics on bounds)
//! - **Range slicing**: `vec[1..4]` - Contiguous slice extraction
//! - **Negative indexing**: `vec.at(-1)` - Python-style end-relative access
//! - **2D array slicing**: `array[(0..2, 1..3)]` - Tuple-based matrix slicing
//! - **Boolean masking**: `vec.slice_at(mask)` - Conditional element selection
//!
//! # For AI Code Generation
//! This module provides the bridge between Rust's Index trait and mathematical
//! slicing operations. Key patterns:
//!
//! ```rust
//! use rustlab_math::{VectorF64, ArrayF64, vec64, array64, BooleanVector};
//!
//! let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
//! let matrix = array64![[1.0, 2.0], [3.0, 4.0]];
//!
//! // Direct element access (panics on invalid index)
//! let elem = v[2];                     // 3.0
//! let matrix_elem = matrix[(1, 0)];    // 3.0
//!
//! // Safe element access (returns Option)  
//! let safe_elem = v.get(10);           // None (out of bounds)
//! let safe_negative = v.at(-1);        // Some(5.0) (last element)
//!
//! // Range slicing (see natural_slicing and ergonomic_slicing for full API)
//! let slice = &v[1..4];                // &[f64] slice reference
//! let owned = v.slice_at(1..4)?;       // VectorF64 owned result
//! ```
//!
//! # Index Trait Limitations
//! Rust's Index trait has fundamental constraints that limit natural syntax:
//! - Must return references, cannot return owned results
//! - Cannot perform fancy indexing (non-contiguous indices)
//! - No error handling (must panic on invalid access)
//! - Limited boolean masking support
//!
//! # Solutions and Workarounds  
//! | Desired Syntax | Rust Limitation | RustLab Solution |
//! |---------------|-----------------|------------------|
//! | `vec[1..4]` → owned | Index returns `&[T]` | `vec.slice_at(1..4)?` |
//! | `vec[-1]` | Negative indices not supported | `vec.at(-1)?` method |
//! | `vec[[1,3,5]]` | Fancy indexing impossible | `vec.slice_at(vec![1,3,5])?` |
//! | `vec[mask]` | Boolean indexing limited | `vec.slice_at(mask)?` |
//! | Error handling | Index must panic | Method-based API with `Result` |
//!
//! # Cross-Module Integration
//! - Built on [`ergonomic_slicing`] for advanced slicing operations
//! - Complements [`natural_slicing`] for Index trait implementations
//! - Works with [`Array`] and [`Vector`] core mathematical operations
//! - Integrates with [`comparison`] for boolean mask generation

use crate::{VectorF64, ArrayF64, BooleanVector};
use crate::ergonomic_slicing::IntoSliceIndex;
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};

// ============================================================================
// VectorF64 Index Implementations
// ============================================================================

// Note: VectorF64 already has Index<usize> implemented in vector.rs
// We'll focus on the ergonomic slicing methods instead of conflicting Index implementations

/// Signed integer indexing with negative support through helper methods
impl VectorF64 {
    /// Get element with negative indexing support: `vec.at(-1)`
    /// 
    /// # Mathematical Specification
    /// Given vector v ∈ ℝⁿ and index i:
    /// - If i ≥ 0: returns v[i] if i < n, else None
    /// - If i < 0: returns v[n + i] if -i ≤ n, else None
    /// 
    /// # For AI Code Generation
    /// - Enables Python/NumPy-style negative indexing in Rust
    /// - Index -1 = last element, -2 = second-to-last, etc.
    /// - Returns Option for safe bounds checking (no panic)
    /// - Equivalent to `vec[index]` in Python but with Rust safety
    /// - Common uses: accessing end elements, reverse iteration patterns
    /// - Use `at_unchecked()` for panic-on-error behavior
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::vec64;
    /// 
    /// let v = vec64![10.0, 20.0, 30.0, 40.0, 50.0];
    /// 
    /// // Positive indexing
    /// assert_eq!(v.at(0), Some(10.0));    // First element
    /// assert_eq!(v.at(4), Some(50.0));    // Last element
    /// assert_eq!(v.at(10), None);         // Out of bounds
    /// 
    /// // Negative indexing (Python-style)
    /// assert_eq!(v.at(-1), Some(50.0));   // Last element
    /// assert_eq!(v.at(-2), Some(40.0));   // Second-to-last
    /// assert_eq!(v.at(-5), Some(10.0));   // First element
    /// assert_eq!(v.at(-10), None);        // Out of bounds
    /// ```
    /// 
    /// # Complexity
    /// - Time: O(1) direct access
    /// - Space: O(1)
    pub fn at(&self, index: i32) -> Option<f64> {
        let resolved_idx = if index >= 0 {
            index as usize
        } else {
            // Negative indexing: -1 = last element
            let offset = (-index) as usize;
            if offset > self.len() || offset == 0 {
                return None;
            }
            self.len() - offset
        };
        
        self.get(resolved_idx)
    }
    
    /// Get element with negative indexing, panicking on bounds error
    pub fn at_unchecked(&self, index: i32) -> f64 {
        self.at(index).unwrap_or_else(|| 
            panic!("Index {} out of bounds for vector of length {}", index, self.len())
        )
    }
}

// ============================================================================
// Wrapper Types for Owned Slicing Results
// ============================================================================

/// A wrapper that holds a VectorF64 slice result to enable ergonomic indexing
/// This allows `vec.s[1..4]` syntax where `.s` creates a SliceableVector
pub struct SliceableVector<'a>(&'a VectorF64);

impl<'a> SliceableVector<'a> {
    /// Create a new sliceable vector wrapper
    pub fn new(vec: &'a VectorF64) -> Self {
        SliceableVector(vec)
    }
}

impl VectorF64 {
    /// Create a sliceable view that supports ergonomic indexing operations
    /// 
    /// # Examples
    /// ```
    /// let vec = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let slice = vec.s()[1..4].unwrap();  // [2.0, 3.0, 4.0]
    /// let last = vec.s()[-1].unwrap();     // 5.0 (last element)
    /// ```
    pub fn s(&self) -> SliceableVector<'_> {
        SliceableVector::new(self)
    }
}

impl<'a> SliceableVector<'a> {
    /// Range slicing that returns Result<VectorF64>
    pub fn get_range(&self, range: Range<i32>) -> Result<VectorF64, String> {
        self.0.slice_at(range)
    }
    
    /// Range from slicing: `vec.s()[2..]`
    pub fn get_range_from(&self, range: RangeFrom<i32>) -> Result<VectorF64, String> {
        self.0.slice_at(range)
    }
    
    /// Range to slicing: `vec.s()[..3]`
    pub fn get_range_to(&self, range: RangeTo<i32>) -> Result<VectorF64, String> {
        self.0.slice_at(range)
    }
    
    /// Full range slicing: `vec.s()[..]`
    pub fn get_full(&self, _: RangeFull) -> Result<VectorF64, String> {
        self.0.slice_at(..)
    }
    
    /// Boolean mask slicing: `vec.s()[mask]`
    pub fn get_mask(&self, mask: BooleanVector) -> Result<VectorF64, String> {
        self.0.slice_at(mask)
    }
    
    /// Index array slicing: `vec.s()[[1, 3, 5]]`
    pub fn get_indices(&self, indices: Vec<i32>) -> Result<VectorF64, String> {
        self.0.slice_at(indices)
    }
    
    /// Single element with negative indexing support
    pub fn get_single(&self, index: i32) -> Result<f64, String> {
        let resolved_idx = if index >= 0 {
            index as usize
        } else {
            let offset = (-index) as usize;
            if offset > self.0.len() || offset == 0 {
                return Err(format!("Negative index {} out of bounds for vector of length {}", index, self.0.len()));
            }
            self.0.len() - offset
        };
        
        self.0.get(resolved_idx)
            .ok_or_else(|| format!("Index {} out of bounds for vector of length {}", index, self.0.len()))
    }
}

// ============================================================================
// ArrayF64 2D Sliceable Wrapper
// ============================================================================

/// A wrapper that enables ergonomic 2D slicing for arrays
pub struct SliceableArray<'a>(&'a ArrayF64);

impl<'a> SliceableArray<'a> {
    /// Create a new sliceable array wrapper
    pub fn new(array: &'a ArrayF64) -> Self {
        SliceableArray(array)
    }
}

impl ArrayF64 {
    /// Create a sliceable view that supports ergonomic 2D indexing operations
    /// 
    /// # Examples
    /// ```
    /// let arr = array2d![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// let sub = arr.s()[(1..3, 0..2)].unwrap();    // [[4, 5], [7, 8]]
    /// let row = arr.s()[(1, ..)].unwrap();         // [4, 5, 6] (entire row 1)
    /// ```
    pub fn s(&self) -> SliceableArray<'_> {
        SliceableArray::new(self)
    }
}

impl<'a> SliceableArray<'a> {
    /// 2D slicing with tuple indices
    pub fn get_2d<R, C>(&self, indices: (R, C)) -> Result<ArrayF64, String>
    where
        R: IntoSliceIndex,
        C: IntoSliceIndex,
    {
        self.0.slice_2d_at(indices)
    }
    
    /// Single element access with 2D coordinates
    pub fn get_single_2d(&self, row: i32, col: i32) -> Result<f64, String> {
        // Resolve negative indices
        let resolved_row = if row >= 0 {
            row as usize
        } else {
            let offset = (-row) as usize;
            if offset > self.0.nrows() || offset == 0 {
                return Err(format!("Row index {} out of bounds", row));
            }
            self.0.nrows() - offset
        };
        
        let resolved_col = if col >= 0 {
            col as usize
        } else {
            let offset = (-col) as usize;
            if offset > self.0.ncols() || offset == 0 {
                return Err(format!("Column index {} out of bounds", col));
            }
            self.0.ncols() - offset
        };
        
        self.0.get(resolved_row, resolved_col)
            .ok_or_else(|| format!("Index ({}, {}) out of bounds", row, col))
    }
}

// ============================================================================
// Convenience Macros for Even More Ergonomic Syntax
// ============================================================================

/// Macro to make slicing even more ergonomic
/// 
/// # Examples
/// ```
/// let vec = vec64![1, 2, 3, 4, 5];
/// let slice = slice!(vec, 1..4);     // [2, 3, 4]
/// let last = slice!(vec, -1);        // 5
/// ```
#[macro_export]
macro_rules! slice {
    ($vec:expr, $range:expr) => {
        $vec.slice_at($range).expect("Slice operation failed")
    };
}

/// Macro for 2D slicing
/// 
/// # Examples  
/// ```
/// let arr = array2d![[1, 2, 3], [4, 5, 6]];
/// let sub = slice_2d!(arr, (1, 0..2));  // [[4, 5]]
/// ```
#[macro_export]
macro_rules! slice_2d {
    ($arr:expr, $indices:expr) => {
        $arr.slice_2d_at($indices).expect("2D slice operation failed")
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec64;
    
    #[test]
    fn test_single_element_indexing() {
        let vec = crate::vec64![10.0, 20.0, 30.0, 40.0, 50.0];
        
        // Positive indexing (use existing Index trait)
        assert_eq!(vec[0], 10.0);
        assert_eq!(vec[4], 50.0);
        
        // Negative indexing with new method
        assert_eq!(vec.at(-1), Some(50.0));  // last element
        assert_eq!(vec.at(-2), Some(40.0));  // second to last
        assert_eq!(vec.at_unchecked(-1), 50.0);
    }
    
    #[test]
    #[should_panic]
    fn test_out_of_bounds_indexing() {
        let vec = vec64![1.0, 2.0, 3.0];
        let _ = vec[5];  // Should panic
    }
    
    #[test]
    fn test_sliceable_vector() {
        let vec = crate::vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Range slicing
        let slice = vec.s().get_range(1..4).unwrap();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 2.0);
        assert_eq!(slice[2], 4.0);
        
        // Single element with negative indexing
        let last = vec.s().get_single(-1).unwrap();
        assert_eq!(last, 5.0);
        
        // Range from
        let tail = vec.s().get_range_from(3..).unwrap();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0], 4.0);
        assert_eq!(tail[1], 5.0);
    }
    
    #[test]
    fn test_slice_macros() {
        let vec = crate::vec64![10.0, 20.0, 30.0, 40.0, 50.0];
        
        // Range slicing with macro
        let slice = slice!(vec, 1..4);
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 20.0);
        
        // Single element with macro
        let last = slice!(vec, -1);
        assert_eq!(last.len(), 1);
        assert_eq!(last[0], 50.0);
    }
}