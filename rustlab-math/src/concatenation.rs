//! Concatenation operations for arrays and vectors with AI-optimized documentation
//! 
//! This module provides efficient concatenation operations that leverage
//! faer's memory layout for optimal performance. All operations preserve
//! numerical precision and integrate with RustLab's operator ecosystem.
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::{ArrayF64, VectorF64, concatenation::*};
//! 
//! // Matrix concatenation
//! let A = ArrayF64::ones(2, 3);
//! let B = ArrayF64::zeros(2, 2);
//! let combined = A.hstack(&B)?;  // 2×5 matrix
//! 
//! // Vector concatenation
//! let v1 = VectorF64::from_slice(&[1.0, 2.0]);
//! let v2 = VectorF64::from_slice(&[3.0, 4.0]);
//! let combined = v1.append(&v2)?;  // [1, 2, 3, 4]
//! 
//! // Use in machine learning: combine feature matrices
//! let features = A ^ &v1;  // Apply to vector
//! ```

use crate::{ArrayF64, VectorF64};
use faer::{Mat, Col};

/// Trait for concatenation operations on arrays
pub trait Concatenate {
    /// The output type of the concatenation operation
    type Output;
    
    /// Concatenate horizontally (along columns) - side by side joining
    /// 
    /// # Mathematical Specification
    /// For matrices A ∈ ℝᵐˣⁿ¹, B ∈ ℝᵐˣⁿ²:
    /// hstack(A, B) = C ∈ ℝᵐˣ⁽ⁿ¹⁺ⁿ²⁾
    /// Where C[:, 0:n1] = A and C[:, n1:n1+n2] = B
    /// 
    /// # For AI Code Generation
    /// - Requires same number of rows: A.nrows() == B.nrows()
    /// - Output columns = A.ncols() + B.ncols()
    /// - Common uses: feature concatenation, augmented matrices
    /// - Equivalent to NumPy's np.hstack() or np.concatenate(axis=1)
    /// - Returns Result for dimension mismatch error handling
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, concatenation::Concatenate};
    /// 
    /// let A = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let B = ArrayF64::from_slice(&[5.0, 6.0], 2, 1).unwrap();
    /// let C = A.hstack(&B).unwrap();  // 2×3 result
    /// // C = [[1.0, 2.0, 5.0],
    /// //      [3.0, 4.0, 6.0]]
    /// ```
    /// 
    /// # Errors
    /// Returns error if row counts don't match
    fn hstack(&self, other: &Self) -> Result<Self::Output, String>;
    
    /// Concatenate vertically (along rows) - stacked on top of each other
    /// 
    /// # Mathematical Specification
    /// For matrices A ∈ ℝᵐ¹ˣⁿ, B ∈ ℝᵐ²ˣⁿ:
    /// vstack(A, B) = C ∈ ℝ⁽ᵐ¹⁺ᵐ²⁾ˣⁿ
    /// Where C[0:m1, :] = A and C[m1:m1+m2, :] = B
    /// 
    /// # For AI Code Generation
    /// - Requires same number of columns: A.ncols() == B.ncols()
    /// - Output rows = A.nrows() + B.nrows()
    /// - Common uses: data batching, time series extension, sample stacking
    /// - Equivalent to NumPy's np.vstack() or np.concatenate(axis=0)
    /// - Returns Result for dimension mismatch error handling
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, concatenation::Concatenate};
    /// 
    /// let A = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let B = ArrayF64::from_slice(&[5.0, 6.0], 1, 2).unwrap();
    /// let C = A.vstack(&B).unwrap();  // 3×2 result
    /// // C = [[1.0, 2.0],
    /// //      [3.0, 4.0],
    /// //      [5.0, 6.0]]
    /// ```
    /// 
    /// # Errors
    /// Returns error if column counts don't match
    fn vstack(&self, other: &Self) -> Result<Self::Output, String>;
}

/// Trait for vector concatenation operations
pub trait VectorConcatenate {
    /// The output type of the concatenation operation
    type Output;
    
    /// Append another vector to this one (concatenate end-to-end)
    /// 
    /// # Mathematical Specification
    /// For vectors u ∈ ℝⁿ¹, v ∈ ℝⁿ²:
    /// append(u, v) = w ∈ ℝⁿ¹⁺ⁿ²
    /// Where w = [u₁, u₂, ..., uₙ₁, v₁, v₂, ..., vₙ₂]
    /// 
    /// # For AI Code Generation
    /// - No dimension restrictions (vectors are 1D)
    /// - Output length = u.len() + v.len()
    /// - Common uses: time series extension, feature combination, data joining
    /// - Equivalent to NumPy's np.append() or np.concatenate()
    /// - Returns Result for consistency with array operations
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, concatenation::VectorConcatenate};
    /// 
    /// let u = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// let v = VectorF64::from_slice(&[4.0, 5.0]);
    /// let w = u.append(&v).unwrap();
    /// // w = [1.0, 2.0, 3.0, 4.0, 5.0]
    /// 
    /// // Time series extension
    /// let historical = VectorF64::from_slice(&[100.0, 105.0, 110.0]);
    /// let new_data = VectorF64::from_slice(&[108.0, 112.0]);
    /// let complete = historical.append(&new_data).unwrap();
    /// ```
    fn append(&self, other: &Self) -> Result<Self::Output, String>;
    
    /// Concatenate multiple vectors (variadic append)
    fn concat(vectors: &[&Self]) -> Result<Self::Output, String>;
}

/// Implementation of Concatenate trait for ArrayF64
impl Concatenate for ArrayF64 {
    type Output = ArrayF64;
    
    /// Horizontally concatenate two arrays (side by side)
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::{ArrayF64, Concatenate};
    /// let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let b = ArrayF64::from_slice(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
    /// let result = a.hstack(&b).unwrap();
    /// // result is [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]
    /// ```
    fn hstack(&self, other: &Self) -> Result<Self::Output, String> {
        if self.nrows() != other.nrows() {
            return Err(format!(
                "Cannot hstack: left has {} rows, right has {} rows",
                self.nrows(), other.nrows()
            ));
        }
        
        let rows = self.nrows();
        let cols = self.ncols() + other.ncols();
        let result = Mat::from_fn(rows, cols, |i, j| {
            if j < self.ncols() {
                unsafe { *self.inner.get_unchecked(i, j) }
            } else {
                unsafe { *other.inner.get_unchecked(i, j - self.ncols()) }
            }
        });
        
        Ok(ArrayF64 { inner: result })
    }
    
    /// Vertically concatenate two arrays (stack vertically)
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::{ArrayF64, Concatenate};
    /// let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let b = ArrayF64::from_slice(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
    /// let result = a.vstack(&b).unwrap();
    /// // result is [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    /// ```
    fn vstack(&self, other: &Self) -> Result<Self::Output, String> {
        if self.ncols() != other.ncols() {
            return Err(format!(
                "Cannot vstack: left has {} columns, right has {} columns",
                self.ncols(), other.ncols()
            ));
        }
        
        let rows = self.nrows() + other.nrows();
        let cols = self.ncols();
        let result = Mat::from_fn(rows, cols, |i, j| {
            if i < self.nrows() {
                unsafe { *self.inner.get_unchecked(i, j) }
            } else {
                unsafe { *other.inner.get_unchecked(i - self.nrows(), j) }
            }
        });
        
        Ok(ArrayF64 { inner: result })
    }
}

/// Implementation of array block matrix construction
impl ArrayF64 {
    /// Construct a block matrix from a 2D array of arrays
    /// 
    /// # Example
    /// ```
    /// use rustlab_math::ArrayF64;
    /// let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let b = ArrayF64::from_slice(&[5.0, 6.0], 2, 1).unwrap();
    /// let c = ArrayF64::from_slice(&[7.0, 8.0], 1, 2).unwrap();
    /// let d = ArrayF64::from_slice(&[9.0], 1, 1).unwrap();
    /// 
    /// let block = ArrayF64::block(vec![
    ///     vec![&a, &b],
    ///     vec![&c, &d]
    /// ]).unwrap();
    /// // Creates [[1, 2, 5], [3, 4, 6], [7, 8, 9]]
    /// ```
    pub fn block(blocks: Vec<Vec<&ArrayF64>>) -> Result<ArrayF64, String> {
        if blocks.is_empty() || blocks[0].is_empty() {
            return Err("Block matrix cannot be empty".to_string());
        }
        
        let block_rows = blocks.len();
        let block_cols = blocks[0].len();
        
        // Verify rectangular shape
        for row in &blocks {
            if row.len() != block_cols {
                return Err("Block matrix must be rectangular".to_string());
            }
        }
        
        // Calculate row heights and verify consistency
        let mut row_heights = vec![0; block_rows];
        for i in 0..block_rows {
            row_heights[i] = blocks[i][0].nrows();
            // Verify all blocks in row have same height
            for j in 1..block_cols {
                if blocks[i][j].nrows() != row_heights[i] {
                    return Err(format!(
                        "Block at ({},{}) has incorrect height: expected {}, got {}",
                        i, j, row_heights[i], blocks[i][j].nrows()
                    ));
                }
            }
        }
        
        // Calculate column widths and verify consistency
        let mut col_widths = vec![0; block_cols];
        for j in 0..block_cols {
            col_widths[j] = blocks[0][j].ncols();
            // Verify all blocks in column have same width
            for i in 1..block_rows {
                if blocks[i][j].ncols() != col_widths[j] {
                    return Err(format!(
                        "Block at ({},{}) has incorrect width: expected {}, got {}",
                        i, j, col_widths[j], blocks[i][j].ncols()
                    ));
                }
            }
        }
        
        let total_rows: usize = row_heights.iter().sum();
        let total_cols: usize = col_widths.iter().sum();
        
        // Build the result matrix
        let result = Mat::from_fn(total_rows, total_cols, |i, j| {
            // Find which block contains this element
            let mut block_row = 0;
            let mut row_offset = i;
            for (idx, &height) in row_heights.iter().enumerate() {
                if row_offset < height {
                    block_row = idx;
                    break;
                }
                row_offset -= height;
            }
            
            let mut block_col = 0;
            let mut col_offset = j;
            for (idx, &width) in col_widths.iter().enumerate() {
                if col_offset < width {
                    block_col = idx;
                    break;
                }
                col_offset -= width;
            }
            
            unsafe { *blocks[block_row][block_col].inner.get_unchecked(row_offset, col_offset) }
        });
        
        Ok(ArrayF64 { inner: result })
    }
}

/// Implementation of VectorConcatenate trait for VectorF64
impl VectorConcatenate for VectorF64 {
    type Output = VectorF64;
    
    /// Append another vector to create a new concatenated vector
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::{VectorF64, VectorConcatenate};
    /// 
    /// let v1 = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// let v2 = VectorF64::from_slice(&[4.0, 5.0]);
    /// let result = v1.append(&v2).unwrap();
    /// 
    /// assert_eq!(result.len(), 5);
    /// // result contains [1.0, 2.0, 3.0, 4.0, 5.0]
    /// ```
    fn append(&self, other: &Self) -> Result<Self::Output, String> {
        let total_len = self.len() + other.len();
        
        if total_len == 0 {
            return Ok(VectorF64::zeros(0));
        }
        
        // Create new vector with combined length
        let combined = Col::from_fn(total_len, |i| {
            if i < self.len() {
                unsafe { *self.inner.get_unchecked(i) }
            } else {
                unsafe { *other.inner.get_unchecked(i - self.len()) }
            }
        });
        
        Ok(VectorF64 { inner: combined })
    }
    
    /// Concatenate multiple vectors into a single vector
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::{VectorF64, VectorConcatenate};
    /// 
    /// let v1 = VectorF64::from_slice(&[1.0, 2.0]);
    /// let v2 = VectorF64::from_slice(&[3.0]);
    /// let v3 = VectorF64::from_slice(&[4.0, 5.0, 6.0]);
    /// 
    /// let result = VectorF64::concat(&[&v1, &v2, &v3]).unwrap();
    /// assert_eq!(result.len(), 6);
    /// // result contains [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    /// ```
    fn concat(vectors: &[&Self]) -> Result<Self::Output, String> {
        if vectors.is_empty() {
            return Ok(VectorF64::zeros(0));
        }
        
        // Calculate total length
        let total_len: usize = vectors.iter().map(|v| v.len()).sum();
        
        if total_len == 0 {
            return Ok(VectorF64::zeros(0));
        }
        
        // Create concatenated vector
        let combined = Col::from_fn(total_len, |i| {
            // Find which vector this index belongs to
            let mut current_offset = 0;
            for vector in vectors {
                if i < current_offset + vector.len() {
                    return unsafe { *vector.inner.get_unchecked(i - current_offset) };
                }
                current_offset += vector.len();
            }
            // This should never happen given our length calculation
            unreachable!("Index out of bounds in vector concatenation")
        });
        
        Ok(VectorF64 { inner: combined })
    }
}

/// Extension methods for VectorF64 to provide convenient concatenation
impl VectorF64 {
    /// Append multiple vectors to this one
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let v1 = VectorF64::from_slice(&[1.0, 2.0]);
    /// let v2 = VectorF64::from_slice(&[3.0, 4.0]);
    /// let v3 = VectorF64::from_slice(&[5.0]);
    /// 
    /// let result = v1.append_multiple(&[&v2, &v3]).unwrap();
    /// assert_eq!(result.len(), 5);
    /// ```
    pub fn append_multiple(&self, others: &[&VectorF64]) -> Result<VectorF64, String> {
        let mut all_vectors = vec![self];
        all_vectors.extend(others);
        VectorF64::concat(&all_vectors)
    }
    
    /// Create a new vector by repeating this vector n times
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let v = VectorF64::from_slice(&[1.0, 2.0]);
    /// let repeated = v.repeat(3).unwrap();
    /// 
    /// assert_eq!(repeated.len(), 6);
    /// // repeated contains [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    /// ```
    pub fn repeat(&self, n: usize) -> Result<VectorF64, String> {
        if n == 0 {
            return Ok(VectorF64::zeros(0));
        }
        
        if self.is_empty() {
            return Ok(VectorF64::zeros(0));
        }
        
        let total_len = self.len() * n;
        let repeated = Col::from_fn(total_len, |i| {
            let idx = i % self.len();
            unsafe { *self.inner.get_unchecked(idx) }
        });
        
        Ok(VectorF64 { inner: repeated })
    }
    
    /// Split this vector into n equally-sized chunks (if possible)
    /// 
    /// Returns an error if the vector length is not divisible by n.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let v = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let chunks = v.split_into(3).unwrap();
    /// 
    /// assert_eq!(chunks.len(), 3);
    /// assert_eq!(chunks[0].len(), 2);
    /// ```
    pub fn split_into(&self, n: usize) -> Result<Vec<VectorF64>, String> {
        if n == 0 {
            return Err("Cannot split into 0 chunks".to_string());
        }
        
        if self.len() % n != 0 {
            return Err(format!(
                "Vector length {} is not divisible by {}", self.len(), n
            ));
        }
        
        let chunk_size = self.len() / n;
        let mut chunks = Vec::with_capacity(n);
        
        for i in 0..n {
            let start = i * chunk_size;
            
            let chunk = Col::from_fn(chunk_size, |j| {
                unsafe { *self.inner.get_unchecked(start + j) }
            });
            
            chunks.push(VectorF64 { inner: chunk });
        }
        
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_hstack_operation() {
        let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = ArrayF64::from_slice(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        
        let result = a.hstack(&b).unwrap();
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 4);
        
        // Check values
        assert_relative_eq!(result.get(0, 0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 1).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 2).unwrap(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 3).unwrap(), 6.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 0).unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 1).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 2).unwrap(), 7.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 3).unwrap(), 8.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_vstack_operation() {
        let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = ArrayF64::from_slice(&[5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        
        let result = a.vstack(&b).unwrap();
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        
        // Check values
        assert_relative_eq!(result.get(0, 0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(0, 1).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 0).unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 1).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(2, 0).unwrap(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(2, 1).unwrap(), 6.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(3, 0).unwrap(), 7.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(3, 1).unwrap(), 8.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_block_matrix() {
        let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = ArrayF64::from_slice(&[5.0, 6.0], 2, 1).unwrap();
        let c = ArrayF64::from_slice(&[7.0, 8.0], 1, 2).unwrap();
        let d = ArrayF64::from_slice(&[9.0], 1, 1).unwrap();
        
        let block = ArrayF64::block(vec![
            vec![&a, &b],
            vec![&c, &d]
        ]).unwrap();
        
        assert_eq!(block.nrows(), 3);
        assert_eq!(block.ncols(), 3);
        
        // Check top-left block (a)
        assert_relative_eq!(block.get(0, 0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(block.get(0, 1).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(block.get(1, 0).unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(block.get(1, 1).unwrap(), 4.0, epsilon = 1e-10);
        
        // Check top-right block (b)
        assert_relative_eq!(block.get(0, 2).unwrap(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(block.get(1, 2).unwrap(), 6.0, epsilon = 1e-10);
        
        // Check bottom-left block (c)
        assert_relative_eq!(block.get(2, 0).unwrap(), 7.0, epsilon = 1e-10);
        assert_relative_eq!(block.get(2, 1).unwrap(), 8.0, epsilon = 1e-10);
        
        // Check bottom-right block (d)
        assert_relative_eq!(block.get(2, 2).unwrap(), 9.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = ArrayF64::from_slice(&[5.0, 6.0, 7.0], 1, 3).unwrap(); // Wrong dimensions
        
        assert!(a.hstack(&b).is_err());
        assert!(a.vstack(&b).is_err());
    }
    
    // =============================================================================
    // Vector Concatenation Tests
    // =============================================================================
    
    #[test]
    fn test_vector_append() {
        let v1 = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
        let v2 = VectorF64::from_slice(&[4.0, 5.0]);
        
        let result = v1.append(&v2).unwrap();
        assert_eq!(result.len(), 5);
        
        // Check values
        for i in 0..3 {
            assert_relative_eq!(result.get(i).unwrap(), v1.get(i).unwrap(), epsilon = 1e-10);
        }
        for i in 0..2 {
            assert_relative_eq!(result.get(i + 3).unwrap(), v2.get(i).unwrap(), epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_vector_concat_multiple() {
        let v1 = VectorF64::from_slice(&[1.0, 2.0]);
        let v2 = VectorF64::from_slice(&[3.0]);
        let v3 = VectorF64::from_slice(&[4.0, 5.0, 6.0]);
        
        let result = VectorF64::concat(&[&v1, &v2, &v3]).unwrap();
        assert_eq!(result.len(), 6);
        
        // Check values
        assert_relative_eq!(result.get(0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(2).unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(3).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(4).unwrap(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(5).unwrap(), 6.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_vector_append_multiple() {
        let v1 = VectorF64::from_slice(&[1.0, 2.0]);
        let v2 = VectorF64::from_slice(&[3.0, 4.0]);
        let v3 = VectorF64::from_slice(&[5.0]);
        
        let result = v1.append_multiple(&[&v2, &v3]).unwrap();
        assert_eq!(result.len(), 5);
        
        // Check values
        assert_relative_eq!(result.get(0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(1).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(2).unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(3).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(result.get(4).unwrap(), 5.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_vector_repeat() {
        let v = VectorF64::from_slice(&[1.0, 2.0]);
        let repeated = v.repeat(3).unwrap();
        
        assert_eq!(repeated.len(), 6);
        
        // Check pattern repeats correctly
        assert_relative_eq!(repeated.get(0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(repeated.get(1).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(repeated.get(2).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(repeated.get(3).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(repeated.get(4).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(repeated.get(5).unwrap(), 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_vector_split_into() {
        let v = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let chunks = v.split_into(3).unwrap();
        
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[1].len(), 2);
        assert_eq!(chunks[2].len(), 2);
        
        // Check values
        assert_relative_eq!(chunks[0].get(0).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(chunks[0].get(1).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(chunks[1].get(0).unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(chunks[1].get(1).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(chunks[2].get(0).unwrap(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(chunks[2].get(1).unwrap(), 6.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_vector_empty_cases() {
        let empty = VectorF64::zeros(0);
        let v = VectorF64::from_slice(&[1.0, 2.0]);
        
        // Append to empty
        let result1 = empty.append(&v).unwrap();
        assert_eq!(result1.len(), 2);
        
        // Append empty
        let result2 = v.append(&empty).unwrap();
        assert_eq!(result2.len(), 2);
        
        // Concat empty vector list
        let result3 = VectorF64::concat(&[]).unwrap();
        assert_eq!(result3.len(), 0);
        
        // Repeat zero times
        let result4 = v.repeat(0).unwrap();
        assert_eq!(result4.len(), 0);
    }
    
    #[test]
    fn test_vector_split_error_cases() {
        let v = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
        
        // Cannot split into 0 chunks
        assert!(v.split_into(0).is_err());
        
        // Length not divisible by chunk count
        assert!(v.split_into(2).is_err());
    }
}