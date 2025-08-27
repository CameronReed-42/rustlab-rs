//! Math-first macro syntax for axis reductions
//! 
//! Provides NumPy-like syntax for axis reductions using macros:
//! - `sum![matrix, axis=0]` - sum along axis 0
//! - `sum![matrix, axis=1, keep=true]` - sum along axis 1 with keepdims
//! - `mean![matrix, axis=0]` - mean along axis 0
//! - `min![matrix, axis=1]` - min along axis 1
//! - `max![matrix, axis=0]` - max along axis 0
//! - `std![matrix, axis=1]` - standard deviation along axis 1
//! - `var![matrix, axis=0]` - variance along axis 0

use crate::reductions::Axis;

// Internal helper to convert axis number to Axis enum
#[doc(hidden)]
pub fn axis_from_number(axis: usize) -> Axis {
    match axis {
        0 => Axis::Rows,
        1 => Axis::Cols,
        _ => panic!("Invalid axis: {}. Must be 0 or 1", axis),
    }
}

/// Math-first sum operation with axis specification
/// 
/// # Syntax
/// - `sum![matrix, axis=0]` - sum along axis 0 (rows), returns column sums
/// - `sum![matrix, axis=1]` - sum along axis 1 (columns), returns row sums  
/// - `sum![matrix, axis=0, keep=true]` - sum with keepdims for broadcasting
/// - `sum![matrix, axis=1, keep=false]` - sum without keepdims (default)
/// 
/// # Examples
/// ```rust
/// use rustlab_math::*;
/// 
/// let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
/// 
/// // Sum along rows (column sums)
/// let col_sums = sum![matrix, axis=0];  // Vector with 2 elements
/// 
/// // Sum along columns (row sums) with keepdims
/// let row_sums = sum![matrix, axis=1, keep=true];  // Array with shape (2, 1)
/// ```
#[macro_export]
macro_rules! sum {
    // sum![matrix, axis=N] - without keepdims
    ($matrix:expr, axis = $axis:expr) => {{
        use $crate::math_first_reductions::axis_from_number;
        use $crate::reductions::AxisReductions;
        $matrix.sum_axis(axis_from_number($axis))
    }};
    
    // sum![matrix, axis=N, keep=true] - with keepdims
    ($matrix:expr, axis = $axis:expr, keep = true) => {{
        use $crate::math_first_reductions::axis_from_number;
        use $crate::reductions::AxisReductions;
        $matrix.sum_axis_keepdims(axis_from_number($axis))
    }};
    
    // sum![matrix, axis=N, keep=false] - explicitly without keepdims
    ($matrix:expr, axis = $axis:expr, keep = false) => {{
        use $crate::math_first_reductions::axis_from_number;
        use $crate::reductions::AxisReductions;
        $matrix.sum_axis(axis_from_number($axis))
    }};
}

/// Math-first mean operation with axis specification
/// 
/// # Syntax
/// - `mean![matrix, axis=0]` - mean along axis 0
/// - `mean![matrix, axis=1, keep=true]` - mean with keepdims
/// 
/// # Examples
/// ```rust
/// use rustlab_math::*;
/// 
/// let matrix = ArrayF64::from_slice(&[2.0, 4.0, 6.0, 8.0], 2, 2).unwrap();
/// let col_means = mean![matrix, axis=0];  // [4.0, 6.0]
/// ```
#[macro_export]
macro_rules! mean {
    // mean![matrix, axis=N] - without keepdims
    ($matrix:expr, axis = $axis:expr) => {{
        use $crate::math_first_reductions::axis_from_number;
        use $crate::reductions::AxisReductions;
        $matrix.mean_axis(axis_from_number($axis))
    }};
    
    // mean![matrix, axis=N, keep=true] - with keepdims
    ($matrix:expr, axis = $axis:expr, keep = true) => {{
        use $crate::math_first_reductions::axis_from_number;
        use $crate::reductions::AxisReductions;
        $matrix.mean_axis_keepdims(axis_from_number($axis))
    }};
    
    // mean![matrix, axis=N, keep=false] - explicitly without keepdims
    ($matrix:expr, axis = $axis:expr, keep = false) => {{
        use $crate::math_first_reductions::axis_from_number;
        use $crate::reductions::AxisReductions;
        $matrix.mean_axis(axis_from_number($axis))
    }};
}

/// Math-first min operation with axis specification
/// 
/// # Syntax
/// - `min![matrix, axis=0]` - min along axis 0
/// - `min![matrix, axis=1]` - min along axis 1
/// 
/// # Examples
/// ```rust
/// use rustlab_math::*;
/// 
/// let matrix = ArrayF64::from_slice(&[1.0, 5.0, 2.0, 4.0], 2, 2).unwrap();
/// let col_mins = min![matrix, axis=0];  // [1.0, 4.0]
/// ```
#[macro_export]
macro_rules! min {
    ($matrix:expr, axis = $axis:expr) => {{
        use $crate::math_first_reductions::axis_from_number;
        use $crate::reductions::AxisReductions;
        $matrix.min_axis(axis_from_number($axis))
    }};
}

/// Math-first max operation with axis specification
/// 
/// # Syntax
/// - `max![matrix, axis=0]` - max along axis 0
/// - `max![matrix, axis=1]` - max along axis 1
/// 
/// # Examples
/// ```rust
/// use rustlab_math::*;
/// 
/// let matrix = ArrayF64::from_slice(&[1.0, 5.0, 2.0, 4.0], 2, 2).unwrap();
/// let col_maxs = max![matrix, axis=0];  // [2.0, 5.0]
/// ```
#[macro_export]
macro_rules! max {
    ($matrix:expr, axis = $axis:expr) => {{
        use $crate::math_first_reductions::axis_from_number;
        use $crate::reductions::AxisReductions;
        $matrix.max_axis(axis_from_number($axis))
    }};
}

/// Math-first standard deviation operation with axis specification
/// 
/// # Syntax
/// - `std![matrix, axis=0]` - std along axis 0
/// - `std![matrix, axis=1]` - std along axis 1
/// 
/// # Examples
/// ```rust
/// use rustlab_math::*;
/// 
/// let matrix = ArrayF64::from_slice(&[1.0, 3.0, 2.0, 4.0], 2, 2).unwrap();
/// let col_stds = std![matrix, axis=0];
/// ```
#[macro_export]
macro_rules! std {
    ($matrix:expr, axis = $axis:expr) => {{
        use $crate::math_first_reductions::axis_from_number;
        use $crate::reductions::AxisReductions;
        $matrix.std_axis(axis_from_number($axis))
    }};
}

/// Math-first variance operation with axis specification
/// 
/// # Syntax
/// - `var![matrix, axis=0]` - variance along axis 0
/// - `var![matrix, axis=1]` - variance along axis 1
/// 
/// # Examples
/// ```rust
/// use rustlab_math::*;
/// 
/// let matrix = ArrayF64::from_slice(&[1.0, 3.0, 2.0, 4.0], 2, 2).unwrap();
/// let col_vars = var![matrix, axis=0];
/// ```
#[macro_export]
macro_rules! var {
    ($matrix:expr, axis = $axis:expr) => {{
        use $crate::math_first_reductions::axis_from_number;
        use $crate::reductions::AxisReductions;
        $matrix.var_axis(axis_from_number($axis))
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayF64, VectorF64};
    
    #[test]
    fn test_sum_macro() {
        let matrix = ArrayF64::from_slice(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ], 2, 3).unwrap();
        
        // Sum along rows (column sums)
        let col_sums = sum![matrix, axis=0].unwrap();
        assert_eq!(col_sums.len(), 3);
        assert_eq!(col_sums.get(0), Some(5.0));  // 1 + 4
        assert_eq!(col_sums.get(1), Some(7.0));  // 2 + 5  
        assert_eq!(col_sums.get(2), Some(9.0));  // 3 + 6
        
        // Sum along columns (row sums)
        let row_sums = sum![matrix, axis=1].unwrap();
        assert_eq!(row_sums.len(), 2);
        assert_eq!(row_sums.get(0), Some(6.0));   // 1 + 2 + 3
        assert_eq!(row_sums.get(1), Some(15.0));  // 4 + 5 + 6
        
        // Sum with keepdims
        let col_sums_keepdims = sum![matrix, axis=0, keep=true].unwrap();
        assert_eq!(col_sums_keepdims.shape(), (1, 3));
        
        let row_sums_keepdims = sum![matrix, axis=1, keep=true].unwrap();
        assert_eq!(row_sums_keepdims.shape(), (2, 1));
        
        // Explicit keep=false
        let col_sums_explicit = sum![matrix, axis=0, keep=false].unwrap();
        assert_eq!(col_sums_explicit.len(), 3);
    }
    
    #[test]
    fn test_mean_macro() {
        let matrix = ArrayF64::from_slice(&[
            2.0, 4.0,
            6.0, 8.0
        ], 2, 2).unwrap();
        
        // Mean along rows (column means)
        let col_means = mean![matrix, axis=0].unwrap();
        assert_eq!(col_means.get(0), Some(4.0)); // (2 + 6) / 2
        assert_eq!(col_means.get(1), Some(6.0)); // (4 + 8) / 2
        
        // Mean along columns (row means)
        let row_means = mean![matrix, axis=1].unwrap();
        assert_eq!(row_means.get(0), Some(3.0)); // (2 + 4) / 2
        assert_eq!(row_means.get(1), Some(7.0)); // (6 + 8) / 2
        
        // Mean with keepdims
        let col_means_keepdims = mean![matrix, axis=0, keep=true].unwrap();
        assert_eq!(col_means_keepdims.shape(), (1, 2));
    }
    
    #[test]
    fn test_min_max_macros() {
        let matrix = ArrayF64::from_slice(&[
            1.0, 5.0, 2.0,
            4.0, 2.0, 6.0
        ], 2, 3).unwrap();
        
        // Min along rows (column mins)
        let col_mins = min![matrix, axis=0].unwrap();
        assert_eq!(col_mins.get(0), Some(1.0)); // min(1, 4)
        assert_eq!(col_mins.get(1), Some(2.0)); // min(5, 2)
        assert_eq!(col_mins.get(2), Some(2.0)); // min(2, 6)
        
        // Max along columns (row maxes)
        let row_maxs = max![matrix, axis=1].unwrap();
        assert_eq!(row_maxs.get(0), Some(5.0)); // max(1, 5, 2)
        assert_eq!(row_maxs.get(1), Some(6.0)); // max(4, 2, 6)
        
        // Max along rows (column maxes)
        let col_maxs = max![matrix, axis=0].unwrap();
        assert_eq!(col_maxs.get(0), Some(4.0)); // max(1, 4)
        assert_eq!(col_maxs.get(1), Some(5.0)); // max(5, 2)
        assert_eq!(col_maxs.get(2), Some(6.0)); // max(2, 6)
    }
    
    #[test]
    fn test_std_var_macros() {
        let matrix = ArrayF64::from_slice(&[
            1.0, 4.0,
            3.0, 6.0
        ], 2, 2).unwrap();
        
        // Variance along rows (column variances)
        let col_vars = var![matrix, axis=0].unwrap();
        assert_eq!(col_vars.len(), 2);
        // For column 0: variance of [1, 3] = ((1-2)^2 + (3-2)^2) / 1 = 2
        assert_eq!(col_vars.get(0), Some(2.0));
        // For column 1: variance of [4, 6] = ((4-5)^2 + (6-5)^2) / 1 = 2  
        assert_eq!(col_vars.get(1), Some(2.0));
        
        // Standard deviation along rows
        let col_stds = std![matrix, axis=0].unwrap();
        assert_eq!(col_stds.len(), 2);
        // std = sqrt(variance) = sqrt(2) ≈ 1.414
        assert!((col_stds.get(0).unwrap() - 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((col_stds.get(1).unwrap() - 2.0_f64.sqrt()).abs() < 1e-10);
    }
    
    
    #[test]
    fn test_real_world_example() {
        // Student grades matrix from the demo
        let grades = ArrayF64::from_slice(&[
            85.0, 92.0, 78.0,  // Student 1
            90.0, 88.0, 85.0,  // Student 2  
            75.0, 95.0, 82.0,  // Student 3
            88.0, 91.0, 89.0,  // Student 4
        ], 4, 3).unwrap();
        
        // Subject averages using math-first syntax
        let subject_averages = mean![grades, axis=0].unwrap();
        assert_eq!(subject_averages.len(), 3);
        
        // Math: (85+90+75+88)/4 = 84.5
        assert_eq!(subject_averages.get(0), Some(84.5));
        // Science: (92+88+95+91)/4 = 91.5  
        assert_eq!(subject_averages.get(1), Some(91.5));
        // English: (78+85+82+89)/4 = 83.5
        assert_eq!(subject_averages.get(2), Some(83.5));
        
        // Student averages
        let student_averages = mean![grades, axis=1].unwrap();
        assert_eq!(student_averages.len(), 4);
        
        // Student 1: (85+92+78)/3 = 85.0
        assert_eq!(student_averages.get(0), Some(85.0));
        
        // Highest scores per subject
        let subject_highs = max![grades, axis=0].unwrap();
        assert_eq!(subject_highs.get(0), Some(90.0)); // Math max
        assert_eq!(subject_highs.get(1), Some(95.0)); // Science max
        assert_eq!(subject_highs.get(2), Some(89.0)); // English max
        
        // Lowest scores per subject
        let subject_lows = min![grades, axis=0].unwrap();
        assert_eq!(subject_lows.get(0), Some(75.0)); // Math min
        assert_eq!(subject_lows.get(1), Some(88.0)); // Science min
        assert_eq!(subject_lows.get(2), Some(78.0)); // English min
    }
}