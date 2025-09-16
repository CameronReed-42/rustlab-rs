//! Advanced multidimensional array statistics with axis-wise operations
//!
//! This module provides comprehensive statistical operations for multidimensional arrays,
//! enabling sophisticated analysis along specific axes (rows, columns). It fills a critical
//! gap in the scientific computing ecosystem by bringing NumPy-style axis-wise statistics
//! to Rust with type safety and performance.
//!
//! # Axis-Wise Statistics Explained
//!
//! ## Conceptual Framework
//!
//! For a 2D array with shape (nrows, ncols), axis-wise operations compute statistics
//! along specified dimensions:
//!
//! ```text
//! Array shape: (4, 3)
//! ┌─────────────┐
//! │  1   2   3  │  ← Row 0
//! │  4   5   6  │  ← Row 1  
//! │  7   8   9  │  ← Row 2
//! │ 10  11  12  │  ← Row 3
//! └─────────────┘
//!   ↑   ↑   ↑
//!  Col0 Col1 Col2
//! ```
//!
//! ## Axis Specification
//!
//! ### Axis::Rows (Axis 0)
//! - **Operation**: Compute statistics across rows (down columns)
//! - **Input**: Each column becomes a data vector
//! - **Output**: Vector with length = ncols (one result per column)
//! - **Example**: Column means, column medians, column standard deviations
//!
//! ### Axis::Cols (Axis 1)  
//! - **Operation**: Compute statistics across columns (along rows)
//! - **Input**: Each row becomes a data vector
//! - **Output**: Vector with length = nrows (one result per row)
//! - **Example**: Row means, row medians, row standard deviations
//!
//! # Mathematical Operations
//!
//! ## Quantile-Based Statistics
//!
//! All quantile operations (median, IQR, percentiles) work axis-wise:
//!
//! **Median Along Axis**
//! ```text
//! For Axis::Rows: median([col_0_values, col_1_values, ...]) → [med_0, med_1, ...]
//! For Axis::Cols: median([row_0_values, row_1_values, ...]) → [med_0, med_1, ...]
//! ```
//!
//! **Advantages**:
//! - Robust to outliers in individual rows/columns
//! - Distribution-free (no normality assumptions)
//! - Meaningful for ordinal data
//!
//! ## Robust Variability Measures
//!
//! **Median Absolute Deviation (MAD)**
//! ```text
//! MAD = median(|X - median(X)|)
//! ```
//! - 50% breakdown point (robust to 50% outliers)
//! - Scale-equivariant (MAD(aX) = |a| × MAD(X))
//! - Efficient robust scale estimator
//!
//! **Interquartile Range (IQR)**
//! ```text
//! IQR = Q3 - Q1 (75th percentile - 25th percentile)
//! ```
//! - Contains middle 50% of data
//! - Basis for outlier detection (1.5 × IQR rule)
//! - Robust alternative to standard deviation
//!
//! ## Distribution Analysis
//!
//! **Mode Detection**
//! - Most frequently occurring value in each axis
//! - Handles multimodal distributions
//! - For ties, returns smallest modal value (deterministic behavior)
//!
//! **Range Analysis**
//! ```text
//! Range = max - min
//! ```
//! - Simple spread measure
//! - Sensitive to outliers (useful for outlier detection)
//! - Computationally efficient O(n) operation
//!
//! # Advanced Use Cases
//!
//! ## Time Series Analysis
//!
//! ```rust
//! use rustlab_stats::prelude::*;
//! use rustlab_math::ArrayF64;
//!
//! // Time series data: rows = time points, columns = variables
//! let ts_data = ArrayF64::from_slice(&[
//!     100.0, 50.0,  // t=0: stock_price, volume
//!     102.0, 52.0,  // t=1
//!     98.0,  48.0,  // t=2
//!     105.0, 55.0,  // t=3
//! ], 4, 2).unwrap();
//!
//! // Variable-wise statistics (across time)
//! let variable_medians = ts_data.median_axis(Axis::Rows);  // [median_price, median_volume]
//! let variable_ranges = ts_data.range_axis(Axis::Rows);    // [price_range, volume_range]
//!
//! // Time-point statistics (across variables)
//! let timepoint_medians = ts_data.median_axis(Axis::Cols); // Median per time point
//! ```
//!
//! ## Feature Analysis in Machine Learning
//!
//! ```rust
//! // Dataset: rows = samples, columns = features
//! let feature_data = ArrayF64::from_slice(&[
//!     /* sample data */
//! ], n_samples, n_features).unwrap();
//!
//! // Feature-wise analysis (across samples)
//! let feature_medians = feature_data.median_axis(Axis::Rows);
//! let feature_mad = feature_data.mad_axis(Axis::Rows);      // Robust scale per feature
//! let feature_iqr = feature_data.iqr_axis(Axis::Rows);      // Feature variability
//!
//! // Outlier detection per feature
//! for (i, &mad_val) in feature_mad.as_slice_unchecked().iter().enumerate() {
//!     let threshold = 3.0 * mad_val;  // 3-MAD rule for outlier detection
//!     println!("Feature {} outlier threshold: {:.3}", i, threshold);
//! }
//! ```
//!
//! ## Image Processing Statistics
//!
//! ```rust
//! // Image data: rows = height, columns = width  
//! let image_data = ArrayF64::from_slice(&[
//!     /* pixel intensities */
//! ], height, width).unwrap();
//!
//! // Row-wise analysis (horizontal statistics)
//! let row_medians = image_data.median_axis(Axis::Cols);    // Median intensity per row
//! let row_ranges = image_data.range_axis(Axis::Cols);      // Dynamic range per row
//!
//! // Column-wise analysis (vertical statistics)
//! let col_medians = image_data.median_axis(Axis::Rows);   // Median intensity per column
//! let col_iqr = image_data.iqr_axis(Axis::Rows);          // Variability per column
//! ```
//!
//! # Performance Characteristics
//!
//! ## Computational Complexity
//!
//! For an array of shape (m, n):
//!
//! | Operation | Axis::Rows | Axis::Cols | Notes |
//! |-----------|------------|------------|-------|
//! | Median | O(n × m log m) | O(m × n log n) | Sorting required |
//! | Quantiles | O(n × m log m) | O(m × n log n) | Sorting required |
//! | MAD | O(n × m log m) | O(m × n log n) | Two sorts per axis |
//! | IQR | O(n × m log m) | O(m × n log n) | Quartile computation |
//! | Mode | O(n × m log m) | O(m × n log n) | Frequency analysis |
//! | Range | O(n × m) | O(m × n) | Linear scan |
//!
//! ## Memory Efficiency
//!
//! - **Temporary allocation**: O(max(m, n)) for axis data extraction
//! - **Result storage**: O(m) for Axis::Cols, O(n) for Axis::Rows
//! - **In-place sorting**: When possible to minimize memory usage
//!
//! ## Optimization Opportunities
//!
//! - **Parallel processing**: Each axis can be processed independently
//! - **Cache efficiency**: Column-major vs row-major access patterns
//! - **SIMD acceleration**: For range and basic statistics operations
//!
//! # Integration Patterns
//!
//! ## With Data Preprocessing
//!
//! ```rust
//! // Robust feature scaling per column
//! let feature_medians = data.median_axis(Axis::Rows);
//! let feature_mad = data.mad_axis(Axis::Rows);
//!
//! // Apply robust scaling column-wise
//! let scaled_data = data.robust_scale_axis(Axis::Rows);
//! ```
//!
//! ## With Statistical Testing
//!
//! ```rust
//! // Test each feature for normality
//! for col_idx in 0..n_features {
//!     let column_data = extract_axis_data_f64(&data, Axis::Rows, col_idx);
//!     let skewness = column_data.skewness();
//!     let kurtosis = column_data.kurtosis();
//!     
//!     if skewness.abs() > 1.0 || kurtosis.abs() > 1.0 {
//!         println!("Feature {} appears non-normal", col_idx);
//!     }
//! }
//! ```
//!
//! # Best Practices
//!
//! ## Choosing the Right Axis
//! - **Row-wise analysis**: When columns represent different variables/features
//! - **Column-wise analysis**: When rows represent different samples/observations
//! - **Domain context**: Consider what makes statistical sense for your data
//!
//! ## Robust vs Classical Statistics
//! - Use robust statistics (median, MAD, IQR) when outliers are expected
//! - Use classical statistics (mean, std) when data is well-behaved and normal
//! - Always compare both to understand data characteristics
//!
//! ## Performance Optimization
//! - For repeated operations, consider caching frequently-used axis extractions
//! - Use appropriate data layouts (row-major vs column-major) for your access patterns
//! - Consider parallel processing for large arrays with independent axes

use rustlab_math::{ArrayF64, ArrayF32, VectorF64, VectorF32};
use rustlab_math::reductions::Axis;
use crate::advanced::quantiles::{Quantiles, QuantileMethod};
use crate::advanced::descriptive::AdvancedDescriptive;

/// Trait for advanced array statistics along axes
pub trait AdvancedArrayStatistics<T> {
    /// Array output type for keepdims operations
    type ArrayOutput;
    /// Vector output type for axis reductions
    type VectorOutput;
    
    /// Compute median along specified axis
    /// 
    /// # Arguments
    /// * `axis` - Axis::Rows (0) computes median along rows, Axis::Cols (1) along columns
    /// 
    /// # Returns
    /// * For Axis::Rows: Vector with length = ncols (column medians)
    /// * For Axis::Cols: Vector with length = nrows (row medians)
    fn median_axis(&self, axis: Axis) -> Self::VectorOutput;
    
    /// Compute quantile along specified axis
    /// 
    /// # Arguments
    /// * `q` - Quantile value between 0.0 and 1.0
    /// * `axis` - Axis specification
    /// * `method` - Quantile computation method
    fn quantile_axis(&self, q: T, axis: Axis, method: Option<QuantileMethod>) -> Self::VectorOutput;
    
    /// Compute interquartile range along specified axis
    fn iqr_axis(&self, axis: Axis) -> Self::VectorOutput;
    
    /// Compute median absolute deviation along specified axis
    fn mad_axis(&self, axis: Axis) -> Self::VectorOutput;
    
    /// Compute mode along specified axis
    /// For ties, returns the smallest value among the modes
    fn mode_axis(&self, axis: Axis) -> Self::VectorOutput;
    
    /// Compute range (max - min) along specified axis
    fn range_axis(&self, axis: Axis) -> Self::VectorOutput;
}

// Helper function to extract a row or column as a vector
fn extract_axis_data_f64(array: &ArrayF64, axis: Axis, index: usize) -> VectorF64 {
    match axis {
        Axis::Rows => {
            // Extract column `index`
            let mut data = Vec::with_capacity(array.nrows());
            for i in 0..array.nrows() {
                data.push(array.get(i, index).unwrap());
            }
            VectorF64::from_slice(&data)
        },
        Axis::Cols => {
            // Extract row `index`
            let mut data = Vec::with_capacity(array.ncols());
            for j in 0..array.ncols() {
                data.push(array.get(index, j).unwrap());
            }
            VectorF64::from_slice(&data)
        }
    }
}

fn extract_axis_data_f32(array: &ArrayF32, axis: Axis, index: usize) -> VectorF32 {
    match axis {
        Axis::Rows => {
            // Extract column `index`
            let mut data = Vec::with_capacity(array.nrows());
            for i in 0..array.nrows() {
                data.push(array.get(i, index).unwrap());
            }
            VectorF32::from_slice(&data)
        },
        Axis::Cols => {
            // Extract row `index`
            let mut data = Vec::with_capacity(array.ncols());
            for j in 0..array.ncols() {
                data.push(array.get(index, j).unwrap());
            }
            VectorF32::from_slice(&data)
        }
    }
}

impl AdvancedArrayStatistics<f64> for ArrayF64 {
    type ArrayOutput = ArrayF64;
    type VectorOutput = VectorF64;
    
    fn median_axis(&self, axis: Axis) -> VectorF64 {
        match axis {
            Axis::Rows => {
                // Compute median for each column
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f64(self, axis, j);
                    results.push(col_data.median());
                }
                VectorF64::from_slice(&results)
            },
            Axis::Cols => {
                // Compute median for each row
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f64(self, axis, i);
                    results.push(row_data.median());
                }
                VectorF64::from_slice(&results)
            }
        }
    }
    
    fn quantile_axis(&self, q: f64, axis: Axis, method: Option<QuantileMethod>) -> VectorF64 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f64(self, axis, j);
                    results.push(col_data.quantile(q, method));
                }
                VectorF64::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f64(self, axis, i);
                    results.push(row_data.quantile(q, method));
                }
                VectorF64::from_slice(&results)
            }
        }
    }
    
    fn iqr_axis(&self, axis: Axis) -> VectorF64 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f64(self, axis, j);
                    results.push(col_data.iqr());
                }
                VectorF64::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f64(self, axis, i);
                    results.push(row_data.iqr());
                }
                VectorF64::from_slice(&results)
            }
        }
    }
    
    fn mad_axis(&self, axis: Axis) -> VectorF64 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f64(self, axis, j);
                    results.push(col_data.mad());
                }
                VectorF64::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f64(self, axis, i);
                    results.push(row_data.mad());
                }
                VectorF64::from_slice(&results)
            }
        }
    }
    
    fn mode_axis(&self, axis: Axis) -> VectorF64 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f64(self, axis, j);
                    results.push(col_data.mode());
                }
                VectorF64::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f64(self, axis, i);
                    results.push(row_data.mode());
                }
                VectorF64::from_slice(&results)
            }
        }
    }
    
    fn range_axis(&self, axis: Axis) -> VectorF64 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f64(self, axis, j);
                    results.push(col_data.range());
                }
                VectorF64::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f64(self, axis, i);
                    results.push(row_data.range());
                }
                VectorF64::from_slice(&results)
            }
        }
    }
}

impl AdvancedArrayStatistics<f32> for ArrayF32 {
    type ArrayOutput = ArrayF32;
    type VectorOutput = VectorF32;
    
    fn median_axis(&self, axis: Axis) -> VectorF32 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f32(self, axis, j);
                    results.push(col_data.median());
                }
                VectorF32::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f32(self, axis, i);
                    results.push(row_data.median());
                }
                VectorF32::from_slice(&results)
            }
        }
    }
    
    fn quantile_axis(&self, q: f32, axis: Axis, method: Option<QuantileMethod>) -> VectorF32 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f32(self, axis, j);
                    results.push(col_data.quantile(q, method));
                }
                VectorF32::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f32(self, axis, i);
                    results.push(row_data.quantile(q, method));
                }
                VectorF32::from_slice(&results)
            }
        }
    }
    
    fn iqr_axis(&self, axis: Axis) -> VectorF32 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f32(self, axis, j);
                    results.push(col_data.iqr());
                }
                VectorF32::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f32(self, axis, i);
                    results.push(row_data.iqr());
                }
                VectorF32::from_slice(&results)
            }
        }
    }
    
    fn mad_axis(&self, axis: Axis) -> VectorF32 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f32(self, axis, j);
                    results.push(col_data.mad());
                }
                VectorF32::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f32(self, axis, i);
                    results.push(row_data.mad());
                }
                VectorF32::from_slice(&results)
            }
        }
    }
    
    fn mode_axis(&self, axis: Axis) -> VectorF32 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f32(self, axis, j);
                    results.push(col_data.mode());
                }
                VectorF32::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f32(self, axis, i);
                    results.push(row_data.mode());
                }
                VectorF32::from_slice(&results)
            }
        }
    }
    
    fn range_axis(&self, axis: Axis) -> VectorF32 {
        match axis {
            Axis::Rows => {
                let mut results = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let col_data = extract_axis_data_f32(self, axis, j);
                    results.push(col_data.range());
                }
                VectorF32::from_slice(&results)
            },
            Axis::Cols => {
                let mut results = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let row_data = extract_axis_data_f32(self, axis, i);
                    results.push(row_data.range());
                }
                VectorF32::from_slice(&results)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::ArrayF64;
    
    #[test]
    fn test_median_axis_rows() {
        // Create a 2x3 array
        let arr = ArrayF64::from_slice(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3).unwrap();
        // Array looks like:
        // [1.0, 4.0, 2.0]
        // [5.0, 3.0, 6.0]
        
        let col_medians = arr.median_axis(Axis::Rows);
        let expected = vec![3.0, 3.5, 4.0]; // median of each column [1,5], [4,3], [2,6]
        
        for i in 0..expected.len() {
            assert!((col_medians.get(i).unwrap() - expected[i]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_median_axis_cols() {
        let arr = ArrayF64::from_slice(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3).unwrap();
        // Array looks like:
        // [1.0, 4.0, 2.0]
        // [5.0, 3.0, 6.0]
        
        let row_medians = arr.median_axis(Axis::Cols);
        let expected = vec![2.0, 5.0]; // median of each row [1,4,2], [5,3,6]
        
        for i in 0..expected.len() {
            assert!((row_medians.get(i).unwrap() - expected[i]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_iqr_axis() {
        let arr = ArrayF64::from_slice(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], 4, 2).unwrap();
        // Array looks like:
        // [1.0, 3.0]
        // [10.0, 30.0]
        // [2.0, 4.0] 
        // [20.0, 40.0]
        
        let col_iqrs = arr.iqr_axis(Axis::Rows);
        assert_eq!(col_iqrs.len(), 2);
        assert!(col_iqrs.get(0).unwrap() > 0.0);
        assert!(col_iqrs.get(1).unwrap() > 0.0);
    }
    
    #[test]
    fn test_quantile_axis() {
        let arr = ArrayF64::from_slice(&[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], 4, 2).unwrap();
        
        let q25 = arr.quantile_axis(0.25, Axis::Rows, None);
        let q75 = arr.quantile_axis(0.75, Axis::Rows, None);
        
        assert_eq!(q25.len(), 2);
        assert_eq!(q75.len(), 2);
        
        // Q75 should be greater than Q25 for each column
        for i in 0..2 {
            assert!(q75.get(i).unwrap() > q25.get(i).unwrap());
        }
    }
    
    #[test]  
    fn test_range_axis() {
        let arr = ArrayF64::from_slice(&[1.0, 5.0, 10.0, 8.0], 2, 2).unwrap();
        // Array: [1.0, 5.0]
        //        [10.0, 8.0]
        
        let col_ranges = arr.range_axis(Axis::Rows);
        let expected = vec![9.0, 3.0]; // [10-1, 8-5]
        
        for i in 0..expected.len() {
            assert!((col_ranges.get(i).unwrap() - expected[i]).abs() < 1e-10);
        }
    }
}