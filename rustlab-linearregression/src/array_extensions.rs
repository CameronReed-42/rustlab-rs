//! Extension methods for Array to add missing functionality

use rustlab_math::{ArrayF64, VectorF64};

/// Extension trait to add missing methods to Array
pub trait ArrayExtensions {
    /// Get a column as a Vector (zero-copy when possible)
    fn col(&self, col_idx: usize) -> Option<VectorF64>;
}

impl ArrayExtensions for ArrayF64 {
    fn col(&self, col_idx: usize) -> Option<VectorF64> {
        if col_idx >= self.ncols() {
            return None;
        }
        
        // Create a slice for the single column (all rows, one column)
        if let Ok(col_slice) = self.slice_cols(col_idx..col_idx + 1) {
            // Extract the column from the slice
            col_slice.col(0)
        } else {
            None
        }
    }
}