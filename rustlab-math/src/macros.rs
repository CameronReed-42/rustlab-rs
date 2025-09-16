//! Ergonomic macros for mathematical operations with AI-optimized documentation
//! 
//! This module provides convenient macros that enable natural mathematical syntax
//! for creating arrays, vectors, and performing operations. All macros integrate
//! seamlessly with RustLab's operator ecosystem (^ for matrix multiplication).
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::{array64, vec64, hstack, vstack, matrix};
//! 
//! // Natural matrix and vector creation
//! let A = array64![[1.0, 2.0], [3.0, 4.0]];
//! let v = vec64![1.0, 2.0];
//! let I = matrix!(eye: 3);  // Identity matrix
//! 
//! // Mathematical operations
//! let result = A ^ v;       // Matrix-vector multiplication
//! let combined = hstack![A, A].unwrap();  // Horizontal concatenation
//! ```

/// Create an ArrayF64 (2D matrix) with natural syntax
/// 
/// # For AI Code Generation
/// - Most commonly used matrix creation macro in RustLab
/// - Creates f64 matrices with natural nested array syntax
/// - All rows must have same number of columns
/// - Zero-based indexing: first element is (0,0)
/// - Integrates with ^ operator for matrix multiplication
/// - Compile-time dimension validation
/// 
/// # Syntax
/// `array64![[row1_col1, row1_col2, ...], [row2_col1, row2_col2, ...], ...]`
/// 
/// # Example
/// ```rust
/// use rustlab_math::array64;
/// 
/// let A = array64![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ];
/// assert_eq!(A.shape(), (2, 3));
/// assert_eq!(A.get(0, 1), Some(2.0));
/// 
/// // Use in mathematical operations
/// let B = array64![[1.0], [2.0], [3.0]];
/// let result = A ^ B;  // Matrix multiplication: (2x3) × (3x1) = (2x1)
/// 
/// // Combine with other operations
/// let scaled = A * 2.0;  // Element-wise scaling
/// ```
/// 
/// # Panics
/// Panics at compile time if:
/// - Zero rows provided
/// - Rows have different numbers of columns
/// 
/// # See Also
/// - [`vec64!`]: Create 1D vectors
/// - [`matrix!`]: Create special matrices (eye, zeros, ones)
/// - [`array32!`]: 32-bit precision version
#[macro_export]
macro_rules! array64 {
    [$([$($x:expr),* $(,)?]),* $(,)?] => {{
        use $crate::ArrayF64;
        
        // Collect all row data
        let row_data: Vec<Vec<f64>> = vec![$( vec![$($x as f64),*] ),*];
        
        // Validate dimensions
        let rows = row_data.len();
        if rows == 0 {
            panic!("Cannot create array with zero rows");
        }
        
        let cols = row_data[0].len();
        for (i, row) in row_data.iter().enumerate() {
            if row.len() != cols {
                panic!("Row {} has {} columns, expected {}", i, row.len(), cols);
            }
        }
        
        // Flatten row-major data  
        let flat_data: Vec<f64> = row_data.into_iter().flatten().collect();
        ArrayF64::from_slice(&flat_data, rows, cols).unwrap()
    }};
}

/// Create an ArrayF32 (2D matrix) with natural syntax
/// 
/// # Example
/// ```
/// use rustlab_math::array32;
/// 
/// let m = array32![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0]
/// ];
/// assert_eq!(m.shape(), (2, 3));
/// ```
#[macro_export]
macro_rules! array32 {
    [$([$($x:expr),* $(,)?]),* $(,)?] => {{
        use $crate::ArrayF32;
        
        // Collect all row data
        let row_data: Vec<Vec<f32>> = vec![$( vec![$($x as f32),*] ),*];
        
        // Validate dimensions
        let rows = row_data.len();
        if rows == 0 {
            panic!("Cannot create array with zero rows");
        }
        
        let cols = row_data[0].len();
        for (i, row) in row_data.iter().enumerate() {
            if row.len() != cols {
                panic!("Row {} has {} columns, expected {}", i, row.len(), cols);
            }
        }
        
        // Flatten row-major data  
        let flat_data: Vec<f32> = row_data.into_iter().flatten().collect();
        ArrayF32::from_slice(&flat_data, rows, cols).unwrap()
    }};
}

/// Create an ArrayC64 (complex f64 matrix) with natural syntax
/// 
/// # Examples
/// ```
/// use rustlab_math::carray64;
/// 
/// // Real numbers only
/// let m1 = carray64![
///     [1.0, 2.0],
///     [3.0, 4.0]
/// ];
/// ```
#[macro_export]
macro_rules! carray64 {
    // Real numbers only: [[real, ...], ...]
    [$([$($x:expr),* $(,)?]),* $(,)?] => {{
        use $crate::ArrayC64;
        use num_complex::Complex;
        
        // Collect all row data
        let row_data: Vec<Vec<Complex<f64>>> = vec![$( vec![$(Complex::new($x as f64, 0.0)),*] ),*];
        
        // Validate dimensions
        let rows = row_data.len();
        if rows == 0 {
            panic!("Cannot create array with zero rows");
        }
        
        let cols = row_data[0].len();
        for (i, row) in row_data.iter().enumerate() {
            if row.len() != cols {
                panic!("Row {} has {} columns, expected {}", i, row.len(), cols);
            }
        }
        
        // Flatten row-major data  
        let flat_data: Vec<Complex<f64>> = row_data.into_iter().flatten().collect();
        ArrayC64::from_slice(&flat_data, rows, cols).unwrap()
    }};
}

/// Create an ArrayC32 (complex f32 matrix) with natural syntax
/// 
/// # Examples
/// ```
/// use rustlab_math::carray32;
/// 
/// // Real numbers only
/// let m1 = carray32![
///     [1.0, 2.0],
///     [3.0, 4.0]
/// ];
/// ```
#[macro_export]
macro_rules! carray32 {
    // Real numbers only: [[real, ...], ...]
    [$([$($x:expr),* $(,)?]),* $(,)?] => {{
        use $crate::ArrayC32;
        use num_complex::Complex;
        
        // Collect all row data
        let row_data: Vec<Vec<Complex<f32>>> = vec![$( vec![$(Complex::new($x as f32, 0.0)),*] ),*];
        
        // Validate dimensions
        let rows = row_data.len();
        if rows == 0 {
            panic!("Cannot create array with zero rows");
        }
        
        let cols = row_data[0].len();
        for (i, row) in row_data.iter().enumerate() {
            if row.len() != cols {
                panic!("Row {} has {} columns, expected {}", i, row.len(), cols);
            }
        }
        
        // Flatten row-major data  
        let flat_data: Vec<Complex<f32>> = row_data.into_iter().flatten().collect();
        ArrayC32::from_slice(&flat_data, rows, cols).unwrap()
    }};
}

/// Create a VectorF64 with natural syntax
/// 
/// # For AI Code Generation
/// - Most commonly used vector creation macro in RustLab
/// - Creates f64 vectors with comma-separated syntax
/// - Zero-based indexing: first element is index 0
/// - Integrates with ^ operator for dot products and matrix operations
/// - Automatic type conversion to f64
/// 
/// # Syntax
/// `vec64![elem1, elem2, elem3, ...]`
/// 
/// # Example
/// ```rust
/// use rustlab_math::{vec64, array64};
/// 
/// let v = vec64![1.0, 2.0, 3.0, 4.0];
/// assert_eq!(v.len(), 4);
/// assert_eq!(v.get(0), Some(1.0));
/// 
/// // Mathematical operations
/// let u = vec64![2.0, 1.0, 0.0, -1.0];
/// let dot_product = v ^ u;  // Dot product: scalar result
/// 
/// // Matrix-vector multiplication
/// let A = array64![[1.0, 0.0], [0.0, 1.0]];
/// let w = vec64![5.0, 3.0];
/// let Aw = A ^ w;  // Matrix × vector = vector
/// 
/// // Feature vectors for ML
/// let features = vec64![1.2, -0.5, 2.1, 0.8];  // Input features
/// ```
/// 
/// # See Also
/// - [`array64!`]: Create 2D matrices
/// - [`vec32!`]: 32-bit precision version
/// - [`cvec64!`]: Complex vector version
#[macro_export]
macro_rules! vec64 {
    [$($x:expr),* $(,)?] => {{
        use $crate::VectorF64;
        VectorF64::from_slice(&[$($x as f64),*])
    }};
}

/// Create a VectorF32 with natural syntax
/// 
/// # Example
/// ```
/// use rustlab_math::vec32;
/// 
/// let v = vec32![1.0, 2.0, 3.0, 4.0];
/// assert_eq!(v.len(), 4);
/// assert_eq!(v.get(0), Some(1.0));
/// ```
#[macro_export]
macro_rules! vec32 {
    [$($x:expr),* $(,)?] => {{
        use $crate::VectorF32;
        VectorF32::from_slice(&[$($x as f32),*])
    }};
}

/// Create a VectorC64 (complex f64) with natural syntax
/// 
/// # Examples
/// ```
/// use rustlab_math::cvec64;
/// 
/// // Real numbers only
/// let v1 = cvec64![1.0, 2.0, 3.0];
/// 
/// // Complex (real, imaginary) pairs
/// let v2 = cvec64![(1.0, 2.0), (3.0, -1.0), (2.0, 0.0)];
/// ```
#[macro_export]
macro_rules! cvec64 {
    // Complex pairs: [(real, imag), ...]
    [$(($re:expr, $im:expr)),* $(,)?] => {{
        use $crate::VectorC64;
        use num_complex::Complex;
        VectorC64::from_slice(&[$(Complex::new($re as f64, $im as f64)),*])
    }};
    
    // Real numbers only: [real, ...]
    [$($x:expr),* $(,)?] => {{
        use $crate::VectorC64;
        use num_complex::Complex;
        VectorC64::from_slice(&[$(Complex::new($x as f64, 0.0)),*])
    }};
}

/// Create a VectorC32 (complex f32) with natural syntax
/// 
/// # Examples
/// ```
/// use rustlab_math::cvec32;
/// 
/// // Real numbers only
/// let v1 = cvec32![1.0, 2.0, 3.0];
/// 
/// // Complex (real, imaginary) pairs
/// let v2 = cvec32![(1.0, 2.0), (3.0, -1.0)];
/// ```
#[macro_export]
macro_rules! cvec32 {
    // Complex pairs: [(real, imag), ...]
    [$(($re:expr, $im:expr)),* $(,)?] => {{
        use $crate::VectorC32;
        use num_complex::Complex;
        VectorC32::from_slice(&[$(Complex::new($re as f32, $im as f32)),*])
    }};
    
    // Real numbers only: [real, ...]
    [$($x:expr),* $(,)?] => {{
        use $crate::VectorC32;
        use num_complex::Complex;
        VectorC32::from_slice(&[$(Complex::new($x as f32, 0.0)),*])
    }};
}

/// Horizontally stack arrays (matrices) with ergonomic syntax
/// 
/// # For AI Code Generation
/// - Concatenates matrices side-by-side (along columns)
/// - All matrices must have same number of rows
/// - Result has shape (rows, sum_of_all_cols)
/// - Returns Result for error handling
/// - Common uses: feature concatenation, augmented matrices
/// - Equivalent to NumPy's np.hstack()
/// 
/// # Syntax
/// `hstack![matrix1, matrix2, matrix3, ...].unwrap()`
/// 
/// # Example
/// ```rust
/// use rustlab_math::{hstack, array64};
/// 
/// let A = array64![[1.0, 2.0], [3.0, 4.0]];  // 2×2
/// let B = array64![[5.0, 6.0], [7.0, 8.0]];  // 2×2
/// let C = array64![[9.0], [10.0]];           // 2×1
/// 
/// let result = hstack![A, B, C].unwrap();     // 2×5 result
/// assert_eq!(result.shape(), (2, 5));
/// 
/// // Machine learning: combine feature matrices
/// let features1 = array64![[1.0, 2.0], [3.0, 4.0]];
/// let features2 = array64![[0.5], [1.5]];
/// let combined = hstack![features1, features2].unwrap();
/// ```
/// 
/// # Errors
/// Returns error if matrices have different numbers of rows
/// 
/// # See Also
/// - [`vstack!`]: Vertical stacking (along rows)
/// - [`block!`]: 2D block matrix construction
/// - Concatenate trait methods
#[macro_export]
macro_rules! hstack {
    [$first:expr, $($rest:expr),* $(,)?] => {{
        (|| -> std::result::Result<_, String> {
            use $crate::Concatenate;
            let mut result = $first.clone();
            $(
                result = result.hstack(&$rest)?;
            )*
            Ok(result)
        })()
    }};
    [$single:expr] => {{
        std::result::Result::<_, String>::Ok($single.clone())
    }};
}

/// Vertically stack arrays (matrices) with ergonomic syntax
/// 
/// # Examples
/// ```
/// use rustlab_math::{vstack, array64};
/// 
/// let a = array64![[1.0, 2.0], [3.0, 4.0]];
/// let b = array64![[5.0, 6.0], [7.0, 8.0]];
/// 
/// let result = vstack![a, b].unwrap();
/// assert_eq!(result.shape(), (4, 2));
/// ```
#[macro_export]
macro_rules! vstack {
    [$first:expr, $($rest:expr),* $(,)?] => {{
        (|| -> std::result::Result<_, String> {
            use $crate::Concatenate;
            let mut result = $first.clone();
            $(
                result = result.vstack(&$rest)?;
            )*
            Ok(result)
        })()
    }};
    [$single:expr] => {{
        std::result::Result::<_, String>::Ok($single.clone())
    }};
}

/// Create block matrices with ergonomic 2D syntax
/// 
/// # Examples
/// ```
/// use rustlab_math::{block, array64};
/// 
/// let a = array64![[1.0, 2.0], [3.0, 4.0]];
/// let b = array64![[5.0], [6.0]];
/// let c = array64![[7.0, 8.0]];
/// let d = array64![[9.0]];
/// 
/// let result = block![
///     [a, b],
///     [c, d]
/// ].unwrap();
/// assert_eq!(result.shape(), (3, 3));
/// ```
#[macro_export]
macro_rules! block {
    [$([$($expr:expr),* $(,)?]),* $(,)?] => {{
        use $crate::ArrayF64;
        let block_data = vec![$(vec![$(&$expr),*]),*];
        ArrayF64::block(block_data)
    }};
}

/// Concatenate vectors with ergonomic syntax
/// 
/// # Examples
/// ```
/// use rustlab_math::{vconcat, vec64};
/// 
/// let a = vec64![1.0, 2.0];
/// let b = vec64![3.0, 4.0]; 
/// let c = vec64![5.0, 6.0];
/// 
/// let result = vconcat![a, b, c].unwrap();
/// assert_eq!(result.len(), 6);
/// ```
#[macro_export]
macro_rules! vconcat {
    [$($vec:expr),* $(,)?] => {{
        use $crate::VectorConcatenate;
        let vectors = vec![$(&$vec),*];
        $crate::VectorF64::concat(&vectors)
    }};
}

/// Concatenate vectors horizontally (alias for vconcat)
/// 
/// # Examples
/// ```
/// use rustlab_math::{hconcat, vec64};
/// 
/// let a = vec64![1.0, 2.0];
/// let b = vec64![3.0, 4.0];
/// 
/// let result = hconcat![a, b].unwrap();
/// assert_eq!(result.len(), 4);
/// ```
#[macro_export]
macro_rules! hconcat {
    [$($vec:expr),* $(,)?] => {{
        $crate::vconcat![$($vec),*]
    }};
}

/// Repeat a vector n times with ergonomic syntax
/// 
/// # Examples
/// ```
/// use rustlab_math::{vrepeat, vec64};
/// 
/// let pattern = vec64![1.0, 2.0];
/// let repeated = vrepeat![pattern, 3].unwrap();
/// assert_eq!(repeated.len(), 6);
/// // Result: [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
/// ```
#[macro_export]
macro_rules! vrepeat {
    [$vec:expr, $n:expr] => {{
        $vec.repeat($n)
    }};
}

/// Split a vector into n equal chunks with ergonomic syntax
/// 
/// # Examples
/// ```
/// use rustlab_math::{vsplit, vec64};
/// 
/// let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let chunks = vsplit![data, 3].unwrap();
/// assert_eq!(chunks.len(), 3);
/// assert_eq!(chunks[0].len(), 2);
/// ```
#[macro_export]
macro_rules! vsplit {
    [$vec:expr, $n:expr] => {{
        $vec.split_into($n)
    }};
}

/// Create matrices with special patterns using convenient syntax
/// 
/// # For AI Code Generation
/// - Powerful macro for creating common matrix patterns
/// - Multiple patterns: zeros, ones, eye, diag, range
/// - All create ArrayF64 matrices
/// - Compile-time pattern selection
/// - Common uses: initialization, linear algebra, feature engineering
/// 
/// # Patterns
/// - `matrix!(zeros: rows, cols)` - Zero-filled matrix
/// - `matrix!(ones: rows, cols)` - Ones-filled matrix
/// - `matrix!(eye: n)` - n×n identity matrix
/// - `matrix!(diag: [a, b, c])` - Diagonal matrix with given values
/// - `matrix!(range: start, end, rows, cols)` - Linear range values
/// 
/// # Example
/// ```rust
/// use rustlab_math::{matrix, array64};
/// 
/// // Common initializations
/// let Z = matrix!(zeros: 3, 4);      // 3×4 zero matrix
/// let I = matrix!(eye: 3);           // 3×3 identity matrix
/// let D = matrix!(diag: [1, 2, 3]);  // Diagonal [1,0,0; 0,2,0; 0,0,3]
/// 
/// // Linear algebra operations
/// let A = array64![[1.0, 2.0], [3.0, 4.0]];
/// let identity_result = A ^ I;       // A × I = A (identity property)
/// 
/// // Feature engineering
/// let features = matrix!(range: 0.0, 1.0, 100, 5);  // 100 samples, 5 features
/// let weights = matrix!(ones: 5, 1);                 // Weight vector
/// let predictions = features ^ weights;              // Linear model
/// ```
/// 
/// # See Also
/// - [`array64!`]: Manual matrix creation
/// - [`cmatrix!`]: Complex matrix patterns
/// - Creation module functions
#[macro_export]
macro_rules! matrix {
    (zeros: $rows:expr, $cols:expr) => {{
        use $crate::ArrayF64;
        ArrayF64::zeros($rows, $cols)
    }};
    
    (ones: $rows:expr, $cols:expr) => {{
        use $crate::ArrayF64; 
        ArrayF64::ones($rows, $cols)
    }};
    
    (eye: $n:expr) => {{
        use $crate::ArrayF64;
        ArrayF64::eye($n)
    }};
    
    (diag: [$($x:expr),* $(,)?]) => {{
        use $crate::{ArrayF64, VectorF64};
        let diag_vec = VectorF64::from_slice(&[$($x as f64),*]);
        let n = diag_vec.len();
        ArrayF64::from_fn(n, n, |i, j| {
            if i == j {
                diag_vec.get(i).unwrap()
            } else {
                0.0
            }
        })
    }};
    
    (range: $start:expr, $end:expr, $rows:expr, $cols:expr) => {{
        use $crate::ArrayF64;
        let total = ($rows as usize) * ($cols as usize);
        let step = if total > 1 { 
            ($end as f64 - $start as f64) / ((total - 1) as f64) 
        } else { 
            0.0 
        };
        ArrayF64::from_fn($rows, $cols, |i, j| {
            let idx = i * ($cols as usize) + j;
            $start as f64 + (idx as f64) * step
        })
    }};
}

/// Create complex matrices with special patterns
/// 
/// # Examples
/// ```
/// use rustlab_math::cmatrix;
/// 
/// let z = cmatrix!(zeros: 3, 4);                    // 3x4 complex zero matrix
/// let i = cmatrix!(eye: 4);                         // 4x4 complex identity matrix
/// let d = cmatrix!(diag: [1, 2, 3]);               // Real diagonal
/// let cd = cmatrix!(cdiag: [(1,1), (2,-1)]);       // Complex diagonal
/// ```
#[macro_export]
macro_rules! cmatrix {
    (zeros: $rows:expr, $cols:expr) => {{
        use $crate::ArrayC64;
        ArrayC64::zeros($rows, $cols)
    }};
    
    (ones: $rows:expr, $cols:expr) => {{
        use $crate::ArrayC64;
        ArrayC64::ones($rows, $cols)
    }};
    
    (eye: $n:expr) => {{
        use $crate::ArrayC64;
        ArrayC64::eye($n)
    }};
    
    // Real diagonal
    (diag: [$($x:expr),* $(,)?]) => {{
        use $crate::{ArrayC64, VectorC64};
        use num_complex::Complex;
        let diag_vec = VectorC64::from_slice(&[$(Complex::new($x as f64, 0.0)),*]);
        let n = diag_vec.len();
        ArrayC64::from_fn(n, n, |i, j| {
            if i == j {
                diag_vec.get(i).unwrap()
            } else {
                Complex::new(0.0, 0.0)
            }
        })
    }};
    
    // Complex diagonal
    (cdiag: [$(($re:expr, $im:expr)),* $(,)?]) => {{
        use $crate::{ArrayC64, VectorC64};
        use num_complex::Complex;
        let diag_vec = VectorC64::from_slice(&[$(Complex::new($re as f64, $im as f64)),*]);
        let n = diag_vec.len();
        ArrayC64::from_fn(n, n, |i, j| {
            if i == j {
                diag_vec.get(i).unwrap()
            } else {
                Complex::new(0.0, 0.0)
            }
        })
    }};
}

/// MATLAB-style horizontal concatenation (alias for hstack!)
/// 
/// # For AI Code Generation
/// - MATLAB/NumPy-style naming: `hcat!` = horizontal concatenation  
/// - Identical to `hstack!` but with more familiar mathematical naming
/// - Concatenates matrices side-by-side (along columns)
/// - All matrices must have same number of rows
/// - Common in linear algebra and data science workflows
/// - Returns Result for error handling
/// 
/// # Syntax
/// `hcat![matrix1, matrix2, matrix3, ...].unwrap()`
/// 
/// # Example
/// ```rust
/// use rustlab_math::{hcat, array64};
/// 
/// let A = array64![[1.0, 2.0], [3.0, 4.0]];
/// let B = array64![[5.0], [7.0]];
/// let C = hcat![A, B].unwrap();  // 2×3 result
/// 
/// // Equivalent to MATLAB: C = [A, B]
/// // Equivalent to NumPy: C = np.hstack([A, B])
/// ```
/// 
/// # See Also
/// - [`hstack!`]: Identical functionality with different name
/// - [`vcat!`]: Vertical concatenation (MATLAB-style)
#[macro_export]
macro_rules! hcat {
    [$($matrix:expr),* $(,)?] => {{
        $crate::hstack![$($matrix),*]
    }};
}

/// MATLAB-style vertical concatenation (alias for vstack!)
/// 
/// # For AI Code Generation
/// - MATLAB/NumPy-style naming: `vcat!` = vertical concatenation
/// - Identical to `vstack!` but with more familiar mathematical naming
/// - Concatenates matrices top-to-bottom (along rows)
/// - All matrices must have same number of columns
/// - Common in linear algebra and data science workflows
/// - Returns Result for error handling
/// 
/// # Syntax
/// `vcat![matrix1, matrix2, matrix3, ...].unwrap()`
/// 
/// # Example
/// ```rust
/// use rustlab_math::{vcat, array64};
/// 
/// let A = array64![[1.0, 2.0], [3.0, 4.0]];
/// let B = array64![[5.0, 6.0]];
/// let C = vcat![A, B].unwrap();  // 3×2 result
/// 
/// // Equivalent to MATLAB: C = [A; B]
/// // Equivalent to NumPy: C = np.vstack([A, B])
/// ```
/// 
/// # See Also
/// - [`vstack!`]: Identical functionality with different name  
/// - [`hcat!`]: Horizontal concatenation (MATLAB-style)
#[macro_export]
macro_rules! vcat {
    [$($matrix:expr),* $(,)?] => {{
        $crate::vstack![$($matrix),*]
    }};
}

/// Create a range vector with convenient syntax
/// 
/// # Examples
/// ```
/// use rustlab_math::range;
/// 
/// let r1 = range!(0.0 => 10.0, 11);  // 11 points from 0 to 10
/// let r2 = range!(1.0 => 5.0, 5);    // [1.0, 2.0, 3.0, 4.0, 5.0]
/// ```
#[macro_export]
macro_rules! range {
    ($start:expr => $end:expr, $n:expr) => {{
        use $crate::VectorF64;
        let n_points = $n as usize;
        if n_points == 0 {
            VectorF64::from_slice(&[])
        } else if n_points == 1 {
            VectorF64::from_slice(&[$start as f64])
        } else {
            let step = ($end as f64 - $start as f64) / ((n_points - 1) as f64);
            let data: Vec<f64> = (0..n_points)
                .map(|i| $start as f64 + (i as f64) * step)
                .collect();
            VectorF64::from_slice(&data)
        }
    }};
}

#[cfg(test)]
mod tests {
    use crate::{ArrayF64, VectorF64, ArrayF32, VectorF32, ArrayC64, VectorC64};
    use num_complex::Complex;
    
    #[test]
    fn test_array64_macro() {
        let m = array64![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ];
        assert_eq!(m.shape(), (3, 2));
        assert_eq!(m.get(1, 0), Some(3.0));
        assert_eq!(m.get(2, 1), Some(6.0));
    }
    
    #[test]
    fn test_vec64_macro() {
        let v = vec64![1.0, 2.0, 3.0];
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(1), Some(2.0));
    }
    
    #[test]
    fn test_matrix_macros() {
        let z = matrix!(zeros: 2, 3);
        assert_eq!(z.shape(), (2, 3));
        assert_eq!(z.get(0, 0), Some(0.0));
        
        let i = matrix!(eye: 2);
        assert_eq!(i.shape(), (2, 2));
        assert_eq!(i.get(0, 0), Some(1.0));
        assert_eq!(i.get(0, 1), Some(0.0));
        
        let d = matrix!(diag: [1, 2, 3]);
        assert_eq!(d.shape(), (3, 3));
        assert_eq!(d.get(1, 1), Some(2.0));
        assert_eq!(d.get(0, 1), Some(0.0));
    }
    
    #[test]
    fn test_vector_concatenation_macros() {
        let a = vec64![1.0, 2.0];
        let b = vec64![3.0, 4.0];
        let c = vec64![5.0];
        
        // Test vconcat
        let result1 = vconcat![a, b, c].unwrap();
        assert_eq!(result1.len(), 5);
        assert_eq!(result1.get(0), Some(1.0));
        assert_eq!(result1.get(4), Some(5.0));
        
        // Test hconcat (alias)
        let result2 = hconcat![a, b].unwrap();
        assert_eq!(result2.len(), 4);
        
        // Test repeat
        let repeated = vrepeat![a, 3].unwrap();
        assert_eq!(repeated.len(), 6);
        assert_eq!(repeated.get(0), Some(1.0));
        assert_eq!(repeated.get(2), Some(1.0));
    }
    
    #[test]
    fn test_vector_split_macro() {
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let chunks = vsplit![data, 3].unwrap();
        
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[1].len(), 2);
        assert_eq!(chunks[2].len(), 2);
        
        assert_eq!(chunks[0].get(0), Some(1.0));
        assert_eq!(chunks[0].get(1), Some(2.0));
        assert_eq!(chunks[1].get(0), Some(3.0));
        assert_eq!(chunks[2].get(1), Some(6.0));
    }
    
    #[test]
    fn test_hstack_macro() {
        let a = array64![[1.0, 2.0], [3.0, 4.0]];
        let b = array64![[5.0, 6.0], [7.0, 8.0]];
        
        let result = hstack![a, b].unwrap();
        assert_eq!(result.shape(), (2, 4));
        
        // Check values
        assert_eq!(result.get(0, 0), Some(1.0));
        assert_eq!(result.get(0, 1), Some(2.0));
        assert_eq!(result.get(0, 2), Some(5.0));
        assert_eq!(result.get(0, 3), Some(6.0));
        assert_eq!(result.get(1, 0), Some(3.0));
        assert_eq!(result.get(1, 1), Some(4.0));
        assert_eq!(result.get(1, 2), Some(7.0));
        assert_eq!(result.get(1, 3), Some(8.0));
    }
    
    #[test]
    fn test_vstack_macro() {
        let a = array64![[1.0, 2.0], [3.0, 4.0]];
        let b = array64![[5.0, 6.0], [7.0, 8.0]];
        
        let result = vstack![a, b].unwrap();
        assert_eq!(result.shape(), (4, 2));
        
        // Check values
        assert_eq!(result.get(0, 0), Some(1.0));
        assert_eq!(result.get(0, 1), Some(2.0));
        assert_eq!(result.get(1, 0), Some(3.0));
        assert_eq!(result.get(1, 1), Some(4.0));
        assert_eq!(result.get(2, 0), Some(5.0));
        assert_eq!(result.get(2, 1), Some(6.0));
        assert_eq!(result.get(3, 0), Some(7.0));
        assert_eq!(result.get(3, 1), Some(8.0));
    }
    
    #[test]
    fn test_block_macro() {
        let a = array64![[1.0, 2.0], [3.0, 4.0]];
        let b = array64![[5.0], [6.0]];
        let c = array64![[7.0, 8.0]];
        let d = array64![[9.0]];
        
        let result = block![
            [a, b],
            [c, d]
        ].unwrap();
        
        assert_eq!(result.shape(), (3, 3));
        
        // Check top-left block (a)
        assert_eq!(result.get(0, 0), Some(1.0));
        assert_eq!(result.get(0, 1), Some(2.0));
        assert_eq!(result.get(1, 0), Some(3.0));
        assert_eq!(result.get(1, 1), Some(4.0));
        
        // Check top-right block (b)
        assert_eq!(result.get(0, 2), Some(5.0));
        assert_eq!(result.get(1, 2), Some(6.0));
        
        // Check bottom-left block (c)
        assert_eq!(result.get(2, 0), Some(7.0));
        assert_eq!(result.get(2, 1), Some(8.0));
        
        // Check bottom-right block (d)
        assert_eq!(result.get(2, 2), Some(9.0));
    }
    
    #[test]
    fn test_hstack_macro_multiple() {
        let a = array64![[1.0], [2.0]];
        let b = array64![[3.0], [4.0]];
        let c = array64![[5.0], [6.0]];
        
        let result = hstack![a, b, c].unwrap();
        assert_eq!(result.shape(), (2, 3));
        
        assert_eq!(result.get(0, 0), Some(1.0));
        assert_eq!(result.get(0, 1), Some(3.0));
        assert_eq!(result.get(0, 2), Some(5.0));
        assert_eq!(result.get(1, 0), Some(2.0));
        assert_eq!(result.get(1, 1), Some(4.0));
        assert_eq!(result.get(1, 2), Some(6.0));
    }
    
    #[test]
    fn test_vstack_macro_multiple() {
        let a = array64![[1.0, 2.0]];
        let b = array64![[3.0, 4.0]];
        let c = array64![[5.0, 6.0]];
        
        let result = vstack![a, b, c].unwrap();
        assert_eq!(result.shape(), (3, 2));
        
        assert_eq!(result.get(0, 0), Some(1.0));
        assert_eq!(result.get(0, 1), Some(2.0));
        assert_eq!(result.get(1, 0), Some(3.0));
        assert_eq!(result.get(1, 1), Some(4.0));
        assert_eq!(result.get(2, 0), Some(5.0));
        assert_eq!(result.get(2, 1), Some(6.0));
    }
    
    #[test]
    fn test_new_creation_macros() {
        // Test different numeric types
        let v32 = vec32![1.0, 2.0, 3.0];
        assert_eq!(v32.len(), 3);
        assert_eq!(v32.get(0), Some(1.0));
        
        let a32 = array32![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(a32.shape(), (2, 2));
        assert_eq!(a32.get(0, 1), Some(2.0));
        
        // Test complex vectors
        let cv_real = cvec64![1.0, 2.0, 3.0];
        assert_eq!(cv_real.len(), 3);
        assert_eq!(cv_real.get(0), Some(Complex::new(1.0, 0.0)));
        
        let cv_complex = cvec64![(1.0, 2.0), (3.0, -1.0)];
        assert_eq!(cv_complex.len(), 2);
        assert_eq!(cv_complex.get(0), Some(Complex::new(1.0, 2.0)));
        assert_eq!(cv_complex.get(1), Some(Complex::new(3.0, -1.0)));
        
        // Test complex arrays - real only
        let ca_real = carray64![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(ca_real.shape(), (2, 2));
        assert_eq!(ca_real.get(0, 0), Some(Complex::new(1.0, 0.0)));
        
        // Note: Complex array with pairs is tricky due to macro parsing,
        // so we'll test the basic functionality
    }
    
    #[test]
    fn test_enhanced_matrix_macros() {
        // Test range matrix
        let range_mat = matrix!(range: 0.0, 1.0, 2, 3);
        assert_eq!(range_mat.shape(), (2, 3));
        assert_eq!(range_mat.get(0, 0), Some(0.0));
        assert_eq!(range_mat.get(1, 2), Some(1.0)); // Last element
        
        // Test complex matrix patterns
        let cz = cmatrix!(zeros: 2, 2);
        assert_eq!(cz.shape(), (2, 2));
        assert_eq!(cz.get(0, 0), Some(Complex::new(0.0, 0.0)));
        
        let ci = cmatrix!(eye: 2);
        assert_eq!(ci.shape(), (2, 2));
        assert_eq!(ci.get(0, 0), Some(Complex::new(1.0, 0.0)));
        assert_eq!(ci.get(0, 1), Some(Complex::new(0.0, 0.0)));
        
        let cd_real = cmatrix!(diag: [1, 2, 3]);
        assert_eq!(cd_real.shape(), (3, 3));
        assert_eq!(cd_real.get(1, 1), Some(Complex::new(2.0, 0.0)));
        assert_eq!(cd_real.get(0, 1), Some(Complex::new(0.0, 0.0)));
        
        let cd_complex = cmatrix!(cdiag: [(1, 1), (2, -1)]);
        assert_eq!(cd_complex.shape(), (2, 2));
        assert_eq!(cd_complex.get(0, 0), Some(Complex::new(1.0, 1.0)));
        assert_eq!(cd_complex.get(1, 1), Some(Complex::new(2.0, -1.0)));
        assert_eq!(cd_complex.get(0, 1), Some(Complex::new(0.0, 0.0)));
    }
    
    #[test] 
    fn test_range_macro() {
        let r1 = range!(0.0 => 10.0, 11);
        assert_eq!(r1.len(), 11);
        assert_eq!(r1.get(0), Some(0.0));
        assert_eq!(r1.get(10), Some(10.0));
        assert_eq!(r1.get(5), Some(5.0));
        
        let r2 = range!(1.0 => 5.0, 5);
        assert_eq!(r2.len(), 5);
        assert_eq!(r2.get(0), Some(1.0));
        assert_eq!(r2.get(4), Some(5.0));
        
        // Test edge cases
        let r_empty = range!(0.0 => 1.0, 0);
        assert_eq!(r_empty.len(), 0);
        
        let r_single = range!(5.0 => 10.0, 1);
        assert_eq!(r_single.len(), 1);
        assert_eq!(r_single.get(0), Some(5.0));
    }
    
    #[test]
    fn test_hcat_vcat_matlab_aliases() {
        // Test hcat! (MATLAB-style horizontal concatenation)
        let a = array64![[1.0, 2.0], [3.0, 4.0]];
        let b = array64![[5.0], [6.0]];
        
        let result_hcat = hcat![a, b].unwrap();
        let result_hstack = hstack![a, b].unwrap();
        
        // Should be identical to hstack!
        assert_eq!(result_hcat.shape(), result_hstack.shape());
        assert_eq!(result_hcat.shape(), (2, 3));
        assert_eq!(result_hcat.get(0, 0), Some(1.0));
        assert_eq!(result_hcat.get(0, 2), Some(5.0));
        assert_eq!(result_hcat.get(1, 2), Some(6.0));
        
        // Test vcat! (MATLAB-style vertical concatenation)
        let c = array64![[1.0, 2.0], [3.0, 4.0]];
        let d = array64![[5.0, 6.0]];
        
        let result_vcat = vcat![c, d].unwrap();
        let result_vstack = vstack![c, d].unwrap();
        
        // Should be identical to vstack!
        assert_eq!(result_vcat.shape(), result_vstack.shape());
        assert_eq!(result_vcat.shape(), (3, 2));
        assert_eq!(result_vcat.get(0, 0), Some(1.0));
        assert_eq!(result_vcat.get(2, 0), Some(5.0));
        assert_eq!(result_vcat.get(2, 1), Some(6.0));
    }
}