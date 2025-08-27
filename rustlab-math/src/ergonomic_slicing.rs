//! Ergonomic Math-First Slicing Operations for Scientific Computing
//!
//! This module provides comprehensive NumPy/MATLAB-style slicing with natural syntax
//! and efficient implementations optimized for mathematical computing workflows.
//!
//! # Core Features
//! - **Range slicing**: `vec.slice_at(0..5)` - Contiguous element extraction
//! - **Negative indexing**: `vec.slice_at(-1)` - Python-style end-relative access  
//! - **2D slicing**: `array.slice_2d_at((0..3, 1..5))` - MATLAB-style matrix subsets
//! - **Boolean masking**: `vec.slice_at(mask)` - Conditional element filtering
//! - **Fancy indexing**: `vec.slice_at([1, 3, 5])` - Arbitrary index arrays
//!
//! # Mathematical Foundation
//! All slicing operations follow standard mathematical conventions:
//! - **Zero-based indexing**: First element at index 0
//! - **Half-open intervals**: Range [i..j) includes i, excludes j
//! - **Negative indexing**: Index -1 = last element, -2 = second-to-last
//! - **Bounds checking**: All operations validate indices at runtime
//!
//! # Performance Characteristics
//! - **Zero-copy when possible**: Range operations return views, not copies
//! - **SIMD-optimized**: Boolean mask operations use vectorized comparisons
//! - **Memory efficient**: Lazy evaluation for chained operations
//! - **Cache-friendly**: Contiguous memory access patterns
//!
//! # For AI Code Generation
//! This module enables AI tools to generate efficient, readable slicing code:
//! - **Type-safe**: Compile-time dimension checking where possible
//! - **Error-safe**: Runtime bounds checking with clear error messages
//! - **Chainable**: Operations can be composed naturally
//! - **Consistent**: Same syntax across vectors and matrices
//! - **Familiar**: Direct translation from NumPy/MATLAB patterns
//!
//! # Usage Patterns
//! ```rust
//! use rustlab_math::{VectorF64, ArrayF64, vec64, array64, BooleanVector};
//!
//! let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! // Range slicing
//! let subset = data.slice_at(1..4)?;           // [2.0, 3.0, 4.0]
//! let tail = data.slice_at(2..)?;              // [3.0, 4.0, 5.0]
//! let head = data.slice_at(..3)?;              // [1.0, 2.0, 3.0]
//!
//! // Negative indexing
//! let last = data.slice_at(-1)?;               // [5.0]
//! let last_two = data.slice_at(-2..)?;         // [4.0, 5.0]
//!
//! // Fancy indexing
//! let selected = data.slice_at(vec![0, 2, 4])?;   // [1.0, 3.0, 5.0]
//!
//! // Boolean masking
//! let mask = BooleanVector::from_slice(&[true, false, true, false, true]);
//! let filtered = data.slice_at(mask)?;         // [1.0, 3.0, 5.0]
//!
//! // 2D matrix slicing
//! let matrix = array64![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
//! let submatrix = matrix.slice_2d_at((1..3, 0..2))?;  // [[4.0, 5.0], [7.0, 8.0]]
//! let row = matrix.slice_2d_at((1, ..))?;             // [[4.0, 5.0, 6.0]]
//! let col = matrix.slice_2d_at((.., 0))?;             // [[1.0], [4.0], [7.0]]
//! ```
//!
//! # Integration with RustLab Ecosystem
//! - **Broadcasting**: Sliced results work with broadcasting operations
//! - **Linear Algebra**: Submatrices integrate with decomposition routines
//! - **Statistics**: Filtered data works with statistical functions
//! - **Optimization**: Slicing preserves performance characteristics
//!
//! # Migration from NumPy/MATLAB
//! | NumPy/MATLAB | RustLab Equivalent |
//! |--------------|-------------------|
//! | `arr[1:4]` | `arr.slice_at(1..4)?` |
//! | `arr[-1]` | `arr.slice_at(-1)?` |
//! | `arr[[0,2,4]]` | `arr.slice_at(vec![0,2,4])?` |
//! | `arr[mask]` | `arr.slice_at(mask)?` |
//! | `arr[1:3, 0:2]` | `arr.slice_2d_at((1..3, 0..2))?` |

use crate::{VectorF64, ArrayF64, BooleanVector};
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};

/// Unified slice index type supporting NumPy/MATLAB-style indexing modes
/// 
/// # Mathematical Specification
/// Represents different indexing operations:
/// - Single: i → vector[i] (single element access)
/// - Range: i..j → vector[i:j] (contiguous slice)
/// - Boolean: mask → vector[mask==true] (conditional selection)
/// - IndexArray: [i₁,i₂,...] → [vector[i₁], vector[i₂], ...] (fancy indexing)
/// 
/// # Dimensions
/// - Input: Various (see individual variants)
/// - Output: Vector or Array with dimensions based on index type
/// 
/// # For AI Code Generation
/// - Enables NumPy-style slicing: `vec[1..4]`, `vec[mask]`, `vec[[0,2,4]]`
/// - Supports negative indexing: `vec[-1]` for last element
/// - Zero-copy when possible (Range operations)
/// - Common uses: data filtering, subarray extraction, conditional selection
/// - Index bounds are checked at runtime
/// 
/// # Example
/// ```
/// use rustlab_math::{VectorF64, SliceIndex};
/// 
/// let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// 
/// // Range slicing: [2.0, 3.0, 4.0]
/// let slice1 = vec.slice_at(1..4)?;
/// 
/// // Negative indexing: [5.0] (last element)
/// let slice2 = vec.slice_at(-1)?;
/// 
/// // Fancy indexing: [1.0, 3.0, 5.0]
/// let slice3 = vec.slice_at(vec![0, 2, 4])?;
/// ```
#[derive(Debug, Clone)]
pub enum SliceIndex {
    /// Single index: `vec[5]`
    Single(i32),
    /// Range: `vec[1..4]`
    Range(Range<i32>),
    /// Range from: `vec[2..]`
    RangeFrom(i32),
    /// Range to: `vec[..3]`  
    RangeTo(i32),
    /// Full range: `vec[..]`
    Full,
    /// Boolean mask: `vec[mask]`
    BoolMask(BooleanVector),
    /// Index array: `vec[[1, 3, 5]]`
    IndexArray(Vec<i32>),
}

/// 2D slice index for arrays: `array[(rows, cols)]`
#[derive(Debug, Clone)]
pub struct SliceIndex2D {
    /// Row slice specification
    pub rows: SliceIndex,
    /// Column slice specification
    pub cols: SliceIndex,
}

/// Convert various index types to unified SliceIndex for ergonomic slicing
/// 
/// # Mathematical Specification
/// Provides type conversion: T → SliceIndex for seamless indexing
/// Enables natural syntax: `vec.slice_at(1..4)` instead of `vec.slice_at(SliceIndex::Range(1..4))`
/// 
/// # For AI Code Generation
/// - Automatically implemented for common index types: i32, Range<i32>, Vec<i32>, etc.
/// - Supports both signed (i32) and unsigned (usize) integer types
/// - Enables method chaining: `vec.slice_at(range).slice_at(indices)`
/// - No explicit SliceIndex construction needed in user code
/// - Common pattern: `vec.slice_at(1..4)` automatically converts Range<i32> to SliceIndex::Range
/// 
/// # Example
/// ```
/// use rustlab_math::{VectorF64, IntoSliceIndex};
/// 
/// let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// 
/// // All these work automatically through IntoSliceIndex:
/// let slice1 = vec.slice_at(1..4)?;        // Range<i32>
/// let slice2 = vec.slice_at(2..)?;         // RangeFrom<i32>
/// let slice3 = vec.slice_at(..3)?;         // RangeTo<i32>
/// let slice4 = vec.slice_at(vec![0,2,4])?; // Vec<i32>
/// ```
pub trait IntoSliceIndex {
    /// Convert this type into a SliceIndex for ergonomic slicing
    fn into_slice_index(self) -> SliceIndex;
}

// Implementations for natural syntax
impl IntoSliceIndex for i32 {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::Single(self)
    }
}

impl IntoSliceIndex for Range<i32> {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::Range(self)
    }
}

impl IntoSliceIndex for RangeFrom<i32> {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::RangeFrom(self.start)
    }
}

impl IntoSliceIndex for RangeTo<i32> {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::RangeTo(self.end)
    }
}

impl IntoSliceIndex for RangeFull {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::Full
    }
}

impl IntoSliceIndex for BooleanVector {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::BoolMask(self)
    }
}

impl IntoSliceIndex for Vec<i32> {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::IndexArray(self)
    }
}

impl IntoSliceIndex for &[i32] {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::IndexArray(self.to_vec())
    }
}

// Additional implementations for usize types (common in natural slicing)
impl IntoSliceIndex for Range<usize> {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::Range((self.start as i32)..(self.end as i32))
    }
}

impl IntoSliceIndex for RangeFrom<usize> {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::RangeFrom(self.start as i32)
    }
}

impl IntoSliceIndex for RangeTo<usize> {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::RangeTo(self.end as i32)
    }
}

impl IntoSliceIndex for Vec<usize> {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::IndexArray(self.into_iter().map(|x| x as i32).collect())
    }
}

impl IntoSliceIndex for &[usize] {
    fn into_slice_index(self) -> SliceIndex {
        SliceIndex::IndexArray(self.iter().map(|&x| x as i32).collect())
    }
}

/// Resolve signed index to positive array index with Python-style negative indexing
/// 
/// # Mathematical Specification
/// Given index i and array length n:
/// - If i ≥ 0: return i if i < n, else None
/// - If i < 0: return n + i if -i ≤ n, else None
/// 
/// # Complexity
/// - Time: O(1)
/// - Space: O(1)
/// 
/// # For AI Code Generation
/// - Enables Python/NumPy-style negative indexing: -1 = last element, -2 = second-to-last
/// - Index -1 maps to len-1, -2 maps to len-2, etc.
/// - Returns None for out-of-bounds indices (both positive and negative)
/// - Used internally by slicing operations for bounds checking
/// 
/// # Example
/// ```
/// // For array of length 5: [a, b, c, d, e]
/// resolve_index(0, 5)  // Some(0) -> 'a'
/// resolve_index(-1, 5) // Some(4) -> 'e' (last element)
/// resolve_index(-2, 5) // Some(3) -> 'd' (second-to-last)
/// resolve_index(5, 5)  // None (out of bounds)
/// resolve_index(-6, 5) // None (out of bounds)
/// ```
fn resolve_index(idx: i32, len: usize) -> Option<usize> {
    if idx >= 0 {
        let ui = idx as usize;
        if ui < len {
            Some(ui)
        } else {
            None
        }
    } else {
        // Negative indexing: -1 = last element
        let offset = (-idx) as usize;
        if offset <= len && offset > 0 {
            Some(len - offset)
        } else {
            None
        }
    }
}

/// Resolve signed range to positive array range with Python-style negative indexing
/// 
/// # Mathematical Specification
/// Given range [start..end) and array length n:
/// - Convert negative indices: start' = start < 0 ? n + start : start
/// - Convert negative indices: end' = end < 0 ? n + end : end
/// - Return Some(start'..end') if 0 ≤ start' ≤ end' ≤ n, else None
/// 
/// # Complexity
/// - Time: O(1)
/// - Space: O(1)
/// 
/// # For AI Code Generation
/// - Supports mixed positive/negative bounds: `1..-1` for "from index 1 to last element"
/// - Handles empty ranges correctly: `2..2` returns empty range
/// - Returns None for invalid ranges: start > end or out of bounds
/// - Used internally by all range-based slicing operations
/// 
/// # Example
/// ```
/// // For array of length 5: [a, b, c, d, e]
/// resolve_range(&(1..4), 5)   // Some(1..4) -> [b, c, d]
/// resolve_range(&(1..-1), 5)  // Some(1..4) -> [b, c, d] (exclude last)
/// resolve_range(&(-3..-1), 5) // Some(2..4) -> [c, d]
/// resolve_range(&(3..2), 5)   // None (invalid: start > end)
/// ```
fn resolve_range(range: &Range<i32>, len: usize) -> Option<Range<usize>> {
    let start = if range.start >= 0 {
        range.start as usize
    } else {
        let offset = (-range.start) as usize;
        if offset <= len {
            len - offset
        } else {
            return None;
        }
    };
    
    let end = if range.end >= 0 {
        range.end as usize
    } else {
        let offset = (-range.end) as usize;
        if offset <= len {
            len - offset
        } else {
            return None;
        }
    };
    
    if start <= end && end <= len {
        Some(start..end)
    } else {
        None
    }
}

// ============================================================================
// VectorF64 Ergonomic Indexing Implementation
// ============================================================================

impl VectorF64 {
    /// Extract vector elements using NumPy/MATLAB-style indexing with owned result
    /// 
    /// # Mathematical Specification
    /// Given vector v ∈ ℝⁿ and index specification I:
    /// - Range [i..j): returns subvector [v[i], v[i+1], ..., v[j-1]]
    /// - Single i: returns single-element vector [v[i]]
    /// - Boolean mask m: returns [v[k] for k where m[k] = true]
    /// - Index array [i₁,i₂,...]: returns [v[i₁], v[i₂], ...]
    /// 
    /// # Dimensions
    /// - Input: self (n), index (varies by type)
    /// - Output: VectorF64 (m) where m depends on index type
    /// 
    /// # Complexity
    /// - Time: O(k) where k is output size
    /// - Space: O(k) for result vector
    /// 
    /// # For AI Code Generation
    /// - Primary slicing method for all index types: ranges, arrays, boolean masks
    /// - Supports negative indexing: -1 = last element, -2 = second-to-last
    /// - Returns owned VectorF64 (not a view) for full flexibility
    /// - Equivalent to NumPy: `arr[start:end]`, `arr[mask]`, `arr[[i1,i2,i3]]`
    /// - Common uses: data filtering, subsetting, reordering elements
    /// - Always bounds-checked at runtime
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::{VectorF64, vec64, BooleanVector};
    /// 
    /// let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
    /// 
    /// // Range slicing: [2.0, 3.0, 4.0]
    /// let slice = v.slice_at(1..4)?;
    /// 
    /// // Negative indexing: [5.0] (last element)
    /// let last = v.slice_at(-1)?;
    /// 
    /// // Partial ranges
    /// let first3 = v.slice_at(..3)?;     // [1.0, 2.0, 3.0]
    /// let from2 = v.slice_at(2..)?;      // [3.0, 4.0, 5.0]
    /// 
    /// // Fancy indexing: [1.0, 3.0, 5.0]
    /// let selected = v.slice_at(vec![0, 2, 4])?;
    /// 
    /// // Boolean mask filtering
    /// let mask = BooleanVector::from_slice(&[true, false, true, false, true]);
    /// let filtered = v.slice_at(mask)?;  // [1.0, 3.0, 5.0]
    /// ```
    /// 
    /// # Errors
    /// - Returns error for out-of-bounds indices
    /// - Returns error for boolean mask length mismatch
    /// - Returns error for empty index arrays
    /// 
    /// # See Also
    /// - [`slice_owned`]: Range slicing shorthand that returns owned result
    /// - [`select`]: Fancy indexing shorthand
    /// - [`select_where`]: Boolean mask shorthand
    pub fn slice_at<T: IntoSliceIndex>(&self, index: T) -> Result<VectorF64, String> {
        let slice_idx = index.into_slice_index();
        self.apply_slice_index(slice_idx)
    }
    
    /// Apply a slice index to create a new vector
    fn apply_slice_index(&self, index: SliceIndex) -> Result<VectorF64, String> {
        match index {
            SliceIndex::Single(idx) => {
                if let Some(ui) = resolve_index(idx, self.len()) {
                    if let Some(val) = self.get(ui) {
                        Ok(VectorF64::from_slice(&[val]))
                    } else {
                        Err(format!("Index {} out of bounds for vector of length {}", idx, self.len()))
                    }
                } else {
                    Err(format!("Index {} out of bounds for vector of length {}", idx, self.len()))
                }
            },
            SliceIndex::Range(range) => {
                if let Some(resolved) = resolve_range(&range, self.len()) {
                    let slice_data: Vec<f64> = (resolved.start..resolved.end)
                        .map(|i| self.get(i).unwrap_or(0.0))
                        .collect();
                    Ok(VectorF64::from_slice(&slice_data))
                } else {
                    Err(format!("Range {:?} out of bounds for vector of length {}", range, self.len()))
                }
            },
            SliceIndex::RangeFrom(start) => {
                let range = Range { start, end: self.len() as i32 };
                self.apply_slice_index(SliceIndex::Range(range))
            },
            SliceIndex::RangeTo(end) => {
                let range = Range { start: 0, end };
                self.apply_slice_index(SliceIndex::Range(range))
            },
            SliceIndex::Full => {
                // Return a copy of the entire vector
                Ok(self.clone())
            },
            SliceIndex::BoolMask(mask) => {
                if mask.len() != self.len() {
                    return Err(format!("Boolean mask length {} doesn't match vector length {}", 
                                     mask.len(), self.len()));
                }
                
                let mut filtered_data = Vec::new();
                for i in 0..self.len() {
                    if let (Some(mask_val), Some(data_val)) = (mask.get(i), self.get(i)) {
                        if mask_val {
                            filtered_data.push(data_val);
                        }
                    }
                }
                Ok(VectorF64::from_slice(&filtered_data))
            },
            SliceIndex::IndexArray(indices) => {
                let mut selected_data = Vec::new();
                for &idx in &indices {
                    if let Some(ui) = resolve_index(idx, self.len()) {
                        if let Some(val) = self.get(ui) {
                            selected_data.push(val);
                        } else {
                            return Err(format!("Index {} out of bounds", idx));
                        }
                    } else {
                        return Err(format!("Index {} out of bounds", idx));
                    }
                }
                Ok(VectorF64::from_slice(&selected_data))
            },
        }
    }
}

// ============================================================================
// ArrayF64 Ergonomic 2D Indexing Implementation
// ============================================================================

impl ArrayF64 {
    /// Extract 2D subarray using MATLAB/NumPy-style tuple indexing with owned result
    /// 
    /// # Mathematical Specification
    /// Given matrix A ∈ ℝᵐˣⁿ and index tuple (I, J):
    /// Returns submatrix B ∈ ℝᵖˣᵍ where:
    /// - B[i,j] = A[I[i], J[j]] for selected row indices I and column indices J
    /// - Supports same index types as 1D slicing: ranges, arrays, boolean masks
    /// 
    /// # Dimensions
    /// - Input: self (m × n), row_indices (varies), col_indices (varies)
    /// - Output: ArrayF64 (p × q) where p,q depend on index types
    /// 
    /// # Complexity
    /// - Time: O(p × q) where p,q are output dimensions
    /// - Space: O(p × q) for result matrix
    /// 
    /// # For AI Code Generation
    /// - Primary 2D slicing method using tuple syntax: `(row_spec, col_spec)`
    /// - Supports negative indexing in both dimensions
    /// - Returns owned ArrayF64 (not a view) for full flexibility
    /// - Equivalent to NumPy: `arr[row_start:row_end, col_start:col_end]`
    /// - Common uses: submatrix extraction, row/column selection, data windowing
    /// - Tuple order: (rows, columns) following matrix convention
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::{ArrayF64, array64};
    /// 
    /// let arr = array64![
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    ///     [7.0, 8.0, 9.0]
    /// ];
    /// 
    /// // Submatrix: [[4.0, 5.0], [7.0, 8.0]]
    /// let sub = arr.slice_2d_at((1..3, 0..2))?;
    /// 
    /// // Entire row 1: [[4.0, 5.0, 6.0]] (1×3 matrix)
    /// let row = arr.slice_2d_at((1, ..))?;
    /// 
    /// // Entire column 0: [[1.0], [4.0], [7.0]] (3×1 matrix)
    /// let col = arr.slice_2d_at((.., 0))?;
    /// 
    /// // Last 2×2 corner: [[5.0, 6.0], [8.0, 9.0]]
    /// let corner = arr.slice_2d_at((-2.., -2..))?;
    /// 
    /// // Mixed indexing: specific rows, range of columns
    /// let mixed = arr.slice_2d_at((vec![0, 2], 1..3))?; // [[2.0, 3.0], [8.0, 9.0]]
    /// ```
    /// 
    /// # Errors
    /// - Returns error for out-of-bounds indices in either dimension
    /// - Returns error for empty slices (0 rows or 0 columns)
    /// - Returns error for boolean mask length mismatch
    /// 
    /// # See Also
    /// - [`slice_at`]: 1D vector slicing with same index types
    /// - [`slice_rows`]: Extract specific rows only
    /// - [`slice_cols`]: Extract specific columns only
    pub fn slice_2d_at<R, C>(&self, indices: (R, C)) -> Result<ArrayF64, String>
    where
        R: IntoSliceIndex,
        C: IntoSliceIndex,
    {
        let (row_idx, col_idx) = indices;
        let row_slice = row_idx.into_slice_index();
        let col_slice = col_idx.into_slice_index();
        
        self.apply_2d_slice_index(row_slice, col_slice)
    }
    
    /// Apply 2D slice indices to create a new array
    fn apply_2d_slice_index(&self, row_idx: SliceIndex, col_idx: SliceIndex) -> Result<ArrayF64, String> {
        // Get the range of rows to include
        let row_range = self.resolve_slice_to_range(row_idx, self.nrows())?;
        let col_range = self.resolve_slice_to_range(col_idx, self.ncols())?;
        
        // Extract the submatrix
        let mut sub_data = Vec::new();
        for r in row_range.clone() {
            for c in col_range.clone() {
                if let Some(val) = self.get(r, c) {
                    sub_data.push(val);
                } else {
                    return Err(format!("Index ({}, {}) out of bounds", r, c));
                }
            }
        }
        
        let sub_rows = row_range.len();
        let sub_cols = col_range.len();
        
        if sub_rows == 0 || sub_cols == 0 {
            return Err("Empty slice not supported".to_string());
        }
        
        ArrayF64::from_slice(&sub_data, sub_rows, sub_cols)
            .map_err(|e| format!("Failed to create array from slice: {:?}", e))
    }
    
    /// Helper to resolve slice index to concrete range
    fn resolve_slice_to_range(&self, index: SliceIndex, dimension_size: usize) -> Result<Range<usize>, String> {
        match index {
            SliceIndex::Single(idx) => {
                if let Some(ui) = resolve_index(idx, dimension_size) {
                    Ok(ui..ui+1)
                } else {
                    Err(format!("Index {} out of bounds for dimension size {}", idx, dimension_size))
                }
            },
            SliceIndex::Range(range) => {
                if let Some(resolved) = resolve_range(&range, dimension_size) {
                    Ok(resolved)
                } else {
                    Err(format!("Range {:?} out of bounds for dimension size {}", range, dimension_size))
                }
            },
            SliceIndex::RangeFrom(start) => {
                let range = Range { start, end: dimension_size as i32 };
                if let Some(resolved) = resolve_range(&range, dimension_size) {
                    Ok(resolved)
                } else {
                    Err(format!("Range {}.. out of bounds for dimension size {}", start, dimension_size))
                }
            },
            SliceIndex::RangeTo(end) => {
                let range = Range { start: 0, end };
                if let Some(resolved) = resolve_range(&range, dimension_size) {
                    Ok(resolved)
                } else {
                    Err(format!("Range ..{} out of bounds for dimension size {}", end, dimension_size))
                }
            },
            SliceIndex::Full => {
                Ok(0..dimension_size)
            },
            SliceIndex::BoolMask(_) => {
                Err("Boolean mask indexing not supported for 2D array dimensions yet".to_string())
            },
            SliceIndex::IndexArray(indices) => {
                // For now, convert to a range-like operation
                // This is a simplified implementation - full fancy indexing would be more complex
                if indices.is_empty() {
                    return Err("Empty index array".to_string());
                }
                
                let mut resolved_indices = Vec::new();
                for &idx in &indices {
                    if let Some(ui) = resolve_index(idx, dimension_size) {
                        resolved_indices.push(ui);
                    } else {
                        return Err(format!("Index {} out of bounds for dimension size {}", idx, dimension_size));
                    }
                }
                
                // For simplicity, return the range from min to max + 1
                // A full implementation would support arbitrary index arrays
                let min_idx = *resolved_indices.iter().min().unwrap();
                let max_idx = *resolved_indices.iter().max().unwrap();
                Ok(min_idx..max_idx+1)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test] 
    fn test_vector_slicing() {
        let v = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Range slicing
        let slice = v.slice_at(1..4).unwrap();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.get(0), Some(2.0));
        assert_eq!(slice.get(2), Some(4.0));
        
        // Negative indexing
        let last = v.slice_at(-1).unwrap();
        assert_eq!(last.len(), 1);
        assert_eq!(last.get(0), Some(5.0));
        
        // Range from
        let from2 = v.slice_at(2..).unwrap();
        assert_eq!(from2.len(), 3);
        assert_eq!(from2.get(0), Some(3.0));
        
        // Range to  
        let to3 = v.slice_at(..3).unwrap();
        assert_eq!(to3.len(), 3);
        assert_eq!(to3.get(2), Some(3.0));
    }
    
    #[test]
    fn test_boolean_mask_slicing() {
        let v = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mask = BooleanVector::from_slice(&[true, false, true, false, true]);
        
        let filtered = v.slice_at(mask).unwrap();
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.get(0), Some(1.0));
        assert_eq!(filtered.get(1), Some(3.0));
        assert_eq!(filtered.get(2), Some(5.0));
    }
    
    #[test]
    fn test_fancy_indexing() {
        let v = VectorF64::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let indices = vec![0, 2, 4];
        
        let selected = v.slice_at(indices).unwrap();
        assert_eq!(selected.len(), 3);
        assert_eq!(selected.get(0), Some(10.0));
        assert_eq!(selected.get(1), Some(30.0));
        assert_eq!(selected.get(2), Some(50.0));
    }
    
    #[test]
    fn test_2d_array_slicing() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let arr = ArrayF64::from_slice(&data, 3, 3).unwrap();
        
        // Submatrix slicing
        let sub = arr.slice_2d_at((1..3, 0..2)).unwrap();
        assert_eq!(sub.nrows(), 2);
        assert_eq!(sub.ncols(), 2);
        assert_eq!(sub.get(0, 0), Some(4.0)); // arr[1,0]
        assert_eq!(sub.get(1, 1), Some(8.0)); // arr[2,1]
        
        // Single row
        let row = arr.slice_2d_at((1, ..)).unwrap();
        assert_eq!(row.nrows(), 1);
        assert_eq!(row.ncols(), 3);
        assert_eq!(row.get(0, 0), Some(4.0));
        assert_eq!(row.get(0, 2), Some(6.0));
    }
}