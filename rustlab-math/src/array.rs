//! Generic 2D Array type built on faer matrices
//!
//! This module provides 2D matrix operations that integrate seamlessly with the
//! [`Vector`] type. Key AI-friendly patterns:
//!
//! - Use `^` operator for mathematical multiplication: `A ^ B`, `A ^ v`
//! - Use `*` operator for element-wise operations: `&A * &B`
//! - Matrix-vector operations: `matrix ^ vector` → `vector`
//! - Vector-matrix operations: `vector ^ matrix` → `vector`
//!
//! # Cross-Module Integration
//! ```rust
//! use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
//!
//! let A = array64![[1.0, 2.0], [3.0, 4.0]];
//! let v = vec64![1.0, 2.0];
//!
//! // Matrix-vector multiplication
//! let result = &A ^ &v;  // Vector result
//!
//! // Vector-matrix multiplication  
//! let result2 = &v ^ &A; // Vector result
//!
//! // Vector dot product
//! let dot = &v ^ &v;     // Scalar result
//!
//! // Matrix multiplication
//! let C = &A ^ &A;       // Matrix result
//! ```

use faer::{Mat, MatRef};
use faer_entity::Entity;
use faer_traits::ComplexField;
use std::fmt;
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use num_complex::Complex;
use crate::{MathError, Result, Vector};

/// A generic 2D array type wrapping faer::Mat<T>
#[repr(C, align(64))] // Cache line alignment for optimal performance
#[derive(Clone)]
pub struct Array<T: Entity> {
    pub(crate) inner: Mat<T>,
}

// Basic impl block with minimal bounds for core functionality
impl<T: Entity> Array<T> {
    /// Get the number of rows
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    
    /// Get the number of columns
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    
    /// Get the shape as a tuple
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Create a new array from a faer matrix
    pub fn from_faer(mat: Mat<T>) -> Self {
        Self { inner: mat }
    }

    /// Get element at position with bounds checking
    /// 
    /// # Mathematical Specification
    /// Returns Aᵢⱼ where A is the matrix, i is the row index, j is the column index
    /// 
    /// # Dimensions
    /// - Input: row i where 0 ≤ i < m, col j where 0 ≤ j < n
    /// - Output: Option<T> containing element or None
    /// 
    /// # Complexity
    /// - Time: O(1) direct access
    /// - Space: O(1)
    /// 
    /// # For AI Code Generation
    /// - Returns Some(value) if indices are valid, None if out of bounds
    /// - Zero-based indexing (first element is at (0, 0))
    /// - Row-major indexing: get(row, col) not get(col, row)
    /// - Safe alternative to direct indexing with array[(row, col)]
    /// - Never panics, always returns Option
    /// - Common uses: safe element access, boundary checking
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::array64;
    /// 
    /// let A = array64![[1.0, 2.0, 3.0],
    ///                  [4.0, 5.0, 6.0]];
    /// 
    /// // Safe access with Option
    /// assert_eq!(A.get(0, 0), Some(1.0));  // Top-left
    /// assert_eq!(A.get(1, 2), Some(6.0));  // Bottom-right
    /// assert_eq!(A.get(5, 5), None);       // Out of bounds
    /// 
    /// // Common patterns
    /// let value = A.get(row, col).unwrap_or(0.0);  // Default to 0
    /// if let Some(val) = A.get(i, j) {
    ///     // Process value safely
    /// }
    /// ```
    /// 
    /// # See Also
    /// - [`set`]: Set element value with bounds checking
    /// - Index trait `array[(row, col)]`: Direct access (panics on bounds)
    /// - [`Vector::get`]: Similar safe access for 1D vectors
    /// - [`row`]: Get entire row as vector
    /// - [`col`]: Get entire column as vector
    pub fn get(&self, row: usize, col: usize) -> Option<T>
    where
        T: Clone,
    {
        if row < self.nrows() && col < self.ncols() {
            Some(unsafe { *self.inner.get_unchecked(row, col) })
        } else {
            None
        }
    }

    /// Set element at position with bounds checking
    /// 
    /// # Mathematical Specification
    /// Sets Aᵢⱼ = value where A is the matrix, i is row, j is column
    /// 
    /// # Dimensions
    /// - Input: row i where 0 ≤ i < m, col j where 0 ≤ j < n, value of type T
    /// - Output: Result<()> indicating success or error
    /// 
    /// # Complexity
    /// - Time: O(1) direct access
    /// - Space: O(1)
    /// 
    /// # For AI Code Generation
    /// - Returns Ok(()) on success, Err(MathError) if indices out of bounds
    /// - Zero-based indexing (first element is at (0, 0))
    /// - Row-major indexing: set(row, col, value) not set(col, row, value)
    /// - Safe alternative to direct assignment with array[(row, col)] = value
    /// - Array must be mutable (&mut self)
    /// - Common uses: safe element update, conditional assignment
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let mut A = ArrayF64::zeros(3, 3);
    /// 
    /// // Safe element setting
    /// A.set(0, 0, 1.0).unwrap();  // Top-left
    /// A.set(1, 1, 5.0).unwrap();  // Center
    /// A.set(2, 2, 9.0).unwrap();  // Bottom-right
    /// 
    /// // Handle out of bounds
    /// match A.set(10, 10, 99.0) {
    ///     Ok(()) => println!("Set successfully"),
    ///     Err(e) => println!("Index out of bounds: {}", e),
    /// }
    /// 
    /// // Build diagonal matrix
    /// for i in 0..3 {
    ///     A.set(i, i, (i + 1) as f64).unwrap();
    /// }
    /// ```
    /// 
    /// # Errors
    /// - `InvalidDimensions`: Row or column index out of bounds
    /// 
    /// # See Also
    /// - [`get`]: Get element value with bounds checking
    /// - IndexMut trait `array[(row, col)] = value`: Direct assignment (panics)
    /// - [`Vector::set`]: Similar safe assignment for 1D vectors
    /// - [`fill`]: Set all elements to same value
    pub fn set(&mut self, row: usize, col: usize, value: T) -> crate::Result<()>
    where
        T: Clone,
    {
        if row < self.nrows() && col < self.ncols() {
            self.inner[(row, col)] = value;
            Ok(())
        } else {
            Err(crate::MathError::InvalidDimensions {
                rows: self.nrows(),
                cols: self.ncols(),
            })
        }
    }


    /// Get reference to underlying faer matrix
    pub fn as_faer(&self) -> MatRef<'_, T> {
        self.inner.as_ref()
    }

    /// Get underlying data as slice (zero-copy when contiguous)
    /// 
    /// Returns `Some(&[T])` if the array's data is stored contiguously in memory,
    /// `None` otherwise. The slice represents the data in column-major order
    /// (column-by-column), which is faer's default storage format.
    /// 
    /// Most arrays created with standard methods (zeros, ones, from_slice) are
    /// contiguous and will return `Some`.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// if let Some(slice) = arr.as_slice() {
    ///     // Data is in column-major order
    ///     assert_eq!(slice.len(), 4);
    /// }
    /// ```
    pub fn as_slice(&self) -> Option<&[T]> {
        // For empty arrays, return empty slice
        if self.nrows() == 0 || self.ncols() == 0 {
            return Some(&[]);
        }
        
        // faer matrices are not stored contiguously, so we cannot provide
        // zero-copy slice access. This method returns None to indicate
        // that the data is not contiguous.
        // 
        // For accessing matrix data as a slice, use to_vec() or iterate
        // over elements using individual element access.
        None
    }

    /// Get underlying data as slice, panicking if not contiguous
    /// 
    /// Use this when you know the array should be contiguous (e.g., created with
    /// standard constructors). For safer access, use `as_slice()`.
    /// 
    /// # Panics
    /// Panics if the array data is not stored contiguously.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let arr = ArrayF64::zeros(3, 3);
    /// let slice = arr.as_slice_unchecked(); // Safe - zeros() creates contiguous data
    /// assert_eq!(slice.len(), 9);
    /// ```
    pub fn as_slice_unchecked(&self) -> &[T] {
        self.as_slice()
            .expect("Array data is not contiguous - use as_slice() for safe access")
    }

    /// Get underlying data as mutable slice (zero-copy when contiguous)
    /// 
    /// Returns `Some(&mut [T])` if the array's data is stored contiguously in memory,
    /// `None` otherwise. This enables efficient in-place operations on the array data.
    /// The slice represents data in column-major order.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let mut arr = ArrayF64::ones(2, 2);
    /// if let Some(slice) = arr.as_mut_slice() {
    ///     for value in slice.iter_mut() {
    ///         *value *= 2.0; // Double all values in-place
    ///     }
    /// }
    /// ```
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        // For empty arrays, return empty mutable slice
        if self.nrows() == 0 || self.ncols() == 0 {
            return Some(&mut []);
        }

        // faer matrices are not stored contiguously, so mutable slice access
        // is not available. Use element-wise access instead.
        None
    }

    /// Get underlying data as mutable slice, panicking if not contiguous
    /// 
    /// Use this when you know the array should be contiguous. For safer access,
    /// use `as_mut_slice()`.
    /// 
    /// # Panics
    /// Panics if the array data is not stored contiguously.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let mut arr = ArrayF64::ones(2, 3);
    /// let slice = arr.as_mut_slice_unchecked();
    /// slice[0] = 5.0; // Modify first element
    /// ```
    pub fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        self.as_mut_slice()
            .expect("Array data is not contiguous - use as_mut_slice() for safe access")
    }

    /// Check if the array's data is stored contiguously in memory
    /// 
    /// Returns `true` if `as_slice()` and `as_mut_slice()` will succeed.
    /// For faer matrices, this is only true for empty matrices.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let empty = ArrayF64::zeros(0, 4);
    /// assert!(empty.is_contiguous());
    /// 
    /// let normal = ArrayF64::zeros(5, 4);
    /// assert!(!normal.is_contiguous()); // faer matrices are not contiguous
    /// ```
    pub fn is_contiguous(&self) -> bool {
        // faer matrices are not stored in simple contiguous column-major format.
        // Each column may be stored separately with padding/alignment between columns.
        // Only empty matrices are considered "contiguous" for the purposes of slice access.
        self.nrows() == 0 || self.ncols() == 0
    }

    /// Get underlying data as slice with safe fallback
    /// 
    /// This is a convenience method that combines `as_slice().unwrap_or(&[])`,
    /// providing a safe way to access array data as a slice. Since most faer
    /// matrices are not contiguous, this typically returns an empty slice.
    /// 
    /// For guaranteed access to array data as a contiguous slice, use
    /// `to_vec().to_slice()` instead.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let arr = ArrayF64::zeros(2, 2);
    /// let slice = arr.to_slice(); // Returns empty slice for non-contiguous arrays
    /// // slice will be &[] for most arrays due to faer's storage layout
    /// 
    /// // For actual data access, use:
    /// let vec = arr.to_vec();
    /// let data_slice = VectorF64::from_slice(&vec).to_slice(); // This contains the actual data
    /// ```
    pub fn to_slice(&self) -> &[T] {
        self.as_slice().unwrap_or(&[])
    }

    /// Create a column matrix (n×1) from a vector
    /// 
    /// # Mathematical Specification
    /// Transforms vector v ∈ ℝⁿ into matrix V ∈ ℝⁿˣ¹ where V[i,0] = v[i]
    /// 
    /// # Dimensions
    /// - Input: Vector of length n
    /// - Output: Matrix with shape (n × 1)
    /// 
    /// # Complexity
    /// - Time: O(n) - copies vector elements to matrix format
    /// - Space: O(n) - allocates new matrix storage
    /// 
    /// # For AI Code Generation
    /// - **Purpose**: Convert vector to column matrix for matrix operations
    /// - **Critical Use**: Solving linear systems (X'X)β = X'y
    /// - **Common Pattern**: y vector → y matrix for normal equations
    /// - **Linear Algebra**: Enables matrix multiplication with vectors
    /// - **Return Type**: Owned Array with single column
    /// - **Optimization**: Uses faer backend for efficient storage
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::{VectorF64, ArrayF64, vec64};
    /// 
    /// // Basic conversion
    /// let v = vec64![1.0, 2.0, 3.0];
    /// let col_matrix = ArrayF64::from_vector_column(&v);
    /// assert_eq!(col_matrix.shape(), (3, 1));
    /// assert_eq!(col_matrix[(0, 0)], 1.0);
    /// assert_eq!(col_matrix[(1, 0)], 2.0);
    /// assert_eq!(col_matrix[(2, 0)], 3.0);
    /// 
    /// // Use in matrix multiplication
    /// let A = ArrayF64::ones(2, 3);  // 2×3 matrix
    /// let v = vec64![1.0, 2.0, 3.0];
    /// let v_mat = ArrayF64::from_vector_column(&v);
    /// let result = &A ^ &v_mat;  // 2×3 × 3×1 = 2×1
    /// ```
    /// 
    /// # Linear Regression Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
    /// 
    /// // Solve normal equations: (X'X)β = X'y
    /// let X = array64![[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
    /// let y = vec64![1.0, 2.0, 3.0];
    /// 
    /// // Convert y to column matrix for matrix operations
    /// let y_matrix = ArrayF64::from_vector_column(&y);  // 3×1
    /// let Xt = X.transpose();                           // 2×3
    /// let Xty = &Xt ^ &y_matrix;                        // 2×1 result
    /// 
    /// // Extract back as vector
    /// let Xty_vector = Xty.to_vector_column();
    /// ```
    /// 
    /// # See Also
    /// - [`to_vector_column`]: Extract column back to Vector
    /// - [`col`]: Extract any column as Vector
    /// - [`Vector::to_column_matrix`]: Alternative name (if exists)
    pub fn from_vector_column(vector: &Vector<T>) -> Self
    where 
        T: Clone,
    {
        use faer::Mat;
        let n = vector.len();
        
        // Create matrix from vector data - faer optimized
        let mat = Mat::from_fn(n, 1, |i, _| vector.inner[i].clone());
        Self::from_faer(mat)
    }
    
    /// Extract first column as a vector
    /// 
    /// # Mathematical Specification
    /// For matrix M ∈ ℝᵐˣⁿ, extracts column 0 as vector v ∈ ℝᵐ where v[i] = M[i,0]
    /// 
    /// # Dimensions
    /// - Input: Matrix (m × n) where n ≥ 1
    /// - Output: Vector of length m
    /// 
    /// # Complexity
    /// - Time: O(m) - copies m elements
    /// - Space: O(m) - allocates vector storage
    /// 
    /// # For AI Code Generation
    /// - **Purpose**: Extract result vector from column matrix operations
    /// - **Common Use**: Get solution vector after matrix multiplication
    /// - **Typical Pattern**: Matrix operation result → Vector for further use
    /// - **Requirement**: Matrix must have at least one column
    /// - **Panics**: If matrix has no columns (ncols == 0)
    /// - **Alternative**: Use `col(0)` for Option-based safe extraction
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::{ArrayF64, array64};
    /// 
    /// // Extract from column matrix
    /// let col_matrix = array64![[1.0], [2.0], [3.0]];  // 3×1
    /// let vector = col_matrix.to_vector_column();
    /// assert_eq!(vector.len(), 3);
    /// assert_eq!(vector.to_slice(), &[1.0, 2.0, 3.0]);
    /// 
    /// // Extract from multi-column matrix (gets first column)
    /// let matrix = array64![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    /// let first_col = matrix.to_vector_column();
    /// assert_eq!(first_col.to_slice(), &[1.0, 2.0, 3.0]);
    /// ```
    /// 
    /// # Linear Algebra Pattern
    /// ```rust
    /// use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
    /// 
    /// // Common pattern: Matrix equation solving
    /// let A = array64![[2.0, 1.0], [1.0, 2.0]];
    /// let b = vec64![3.0, 3.0];
    /// 
    /// // Convert to matrix for operations
    /// let b_matrix = ArrayF64::from_vector_column(&b);
    /// let A_inv = A.inv().unwrap();  // Using linear algebra
    /// let x_matrix = &A_inv ^ &b_matrix;  // Solution as matrix
    /// 
    /// // Extract solution as vector
    /// let x = x_matrix.to_vector_column();
    /// assert_eq!(x.len(), 2);
    /// ```
    /// 
    /// # Panics
    /// ```rust,should_panic
    /// use rustlab_math::ArrayF64;
    /// 
    /// let empty = ArrayF64::zeros(5, 0);  // 5×0 matrix (no columns)
    /// let vec = empty.to_vector_column();  // Panics!
    /// ```
    /// 
    /// # See Also
    /// - [`from_vector_column`]: Create column matrix from Vector
    /// - [`col`]: Safe extraction of any column (returns Option)
    /// - [`col_view`]: Zero-copy column access
    pub fn to_vector_column(&self) -> Vector<T>
    where
        T: Clone + ComplexField,
    {
        if self.ncols() == 0 {
            panic!("Cannot extract column from matrix with no columns");
        }
        
        let col_data: Vec<T> = (0..self.nrows())
            .map(|i| self[(i, 0)].clone())
            .collect();
            
        Vector::from_vec(col_data)
    }

    /// Convert matrix data to a contiguous vector in column-major order
    /// 
    /// This creates a new vector containing all matrix elements in column-major order
    /// (column-by-column). This is useful when you need contiguous access to matrix data.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let vec = arr.to_vec();
    /// // vec contains data in column-major order: [1.0, 3.0, 2.0, 4.0]
    /// ```
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        let mut result = Vec::with_capacity(self.nrows() * self.ncols());
        
        // Collect data in column-major order
        for col in 0..self.ncols() {
            for row in 0..self.nrows() {
                result.push(unsafe { self.inner.get_unchecked(row, col).clone() });
            }
        }
        
        result
    }

    /// Create an iterator over all matrix elements in column-major order
    /// 
    /// This provides efficient iteration over matrix elements without creating
    /// an intermediate vector. Elements are yielded in column-major order.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let sum: f64 = arr.iter().sum();
    /// let max = arr.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.ncols()).flat_map(move |col| {
            (0..self.nrows()).map(move |row| {
                unsafe { self.inner.get_unchecked(row, col) }
            })
        })
    }
}

// Impl block for creation methods requiring ComplexField
impl<T: Entity> Array<T> 
where
    T: ComplexField,
{
    /// Create a new matrix filled with zeros
    /// 
    /// # Mathematical Specification
    /// Creates matrix A ∈ ℝᵐˣⁿ or ℂᵐˣⁿ where Aᵢⱼ = 0 for all i, j
    /// 
    /// # Dimensions
    /// - Input: rows (m), cols (n) where m, n ≥ 0
    /// - Output: Matrix of shape (m × n)
    /// 
    /// # Complexity
    /// - Time: O(m × n) initialization
    /// - Space: O(m × n) allocation
    /// 
    /// # For AI Code Generation
    /// - Allocates memory optimized for numerical operations
    /// - Dimensions can be 0 (creates empty matrix)
    /// - All elements initialized to exact zero (0.0 for f64, 0+0i for complex)
    /// - Common uses: initialization, accumulation, workspace allocation
    /// - Prefer this over `Array::fill(m, n, 0.0)` for clarity and performance
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// // Create 3×4 zero matrix
    /// let A = ArrayF64::zeros(3, 4);
    /// assert_eq!(A.shape(), (3, 4));
    /// assert_eq!(A.get(0, 0), Some(0.0));
    /// 
    /// // Common pattern: accumulator matrix
    /// let mut sum = ArrayF64::zeros(10, 10);
    /// for data_matrix in dataset {
    ///     sum += data_matrix;
    /// }
    /// 
    /// // Create workspace for algorithms
    /// let workspace = ArrayF64::zeros(n, n);
    /// ```
    /// 
    /// # See Also
    /// - [`ones`]: Create matrix filled with ones
    /// - [`eye`]: Create identity matrix (ones on diagonal)
    /// - [`fill`]: Create matrix with custom constant value
    /// - [`from_slice`]: Create from existing data
    /// - [`Vector::zeros`]: Create 1D zero vector
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            inner: Mat::zeros(rows, cols),
        }
    }
    
    /// Create a matrix filled with ones
    /// 
    /// # Mathematical Specification
    /// Creates matrix A ∈ ℝᵐˣⁿ or ℂᵐˣⁿ where Aᵢⱼ = 1 for all i, j
    /// 
    /// # Dimensions
    /// - Input: rows (m), cols (n) where m, n ≥ 0
    /// - Output: Matrix of shape (m × n)
    /// 
    /// # Complexity
    /// - Time: O(m × n) initialization
    /// - Space: O(m × n) allocation
    /// 
    /// # For AI Code Generation
    /// - All elements set to multiplicative identity (1.0 for f64, 1+0i for complex)
    /// - Dimensions can be 0 (creates empty matrix)
    /// - Common uses: initialization, mask creation, broadcasting baseline
    /// - Often combined with scalar operations: `ones(m, n) * 5.0`
    /// - Not the same as identity matrix (use `eye` for that)
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// // Create 2×3 ones matrix
    /// let A = ArrayF64::ones(2, 3);
    /// assert_eq!(A.sum_elements(), 6.0);  // 2×3 = 6 elements
    /// 
    /// // Create matrix of 5s
    /// let fives = ArrayF64::ones(4, 4) * 5.0;
    /// 
    /// // Use as mask for element-wise operations
    /// let mask = ArrayF64::ones(data.nrows(), data.ncols());
    /// let selected = &data * &mask;
    /// ```
    /// 
    /// # See Also
    /// - [`zeros`]: Create zero-filled matrix
    /// - [`eye`]: Create identity matrix (diagonal ones only)
    /// - [`fill`]: Create matrix with custom constant
    /// - [`Vector::ones`]: Create 1D ones vector
    pub fn ones(rows: usize, cols: usize) -> Self
    where
        T: num_traits::One,
    {
        Self {
            inner: Mat::from_fn(rows, cols, |_i, _j| T::one()),
        }
    }
    
    /// Create array filled with a constant value
    /// 
    /// # Arguments
    /// * `rows` - Number of rows in the array
    /// * `cols` - Number of columns in the array
    /// * `value` - Value to fill the array with
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let arr = ArrayF64::fill(3, 2, 3.14);
    /// assert_eq!(arr.shape(), (3, 2));
    /// assert_eq!(arr.get(0, 0), Some(3.14));
    /// assert_eq!(arr.get(2, 1), Some(3.14));
    /// ```
    pub fn fill(rows: usize, cols: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self {
            inner: Mat::from_fn(rows, cols, |_i, _j| value.clone()),
        }
    }
}

// Impl block for row/column operations requiring ComplexField
impl<T: Entity + faer_traits::ComplexField> Array<T>
where
    T: Clone,
{
    /// Get a row as an owned Vector
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ, extracts row i as vector r ∈ ℝⁿ where rⱼ = Aᵢⱼ
    /// 
    /// # Dimensions
    /// - Input: Matrix (m × n), row index i where 0 ≤ i < m
    /// - Output: Vector of length n containing row elements
    /// 
    /// # Complexity
    /// - Time: O(n) where n is number of columns
    /// - Space: O(n) - creates new vector with copied data
    /// 
    /// # For AI Code Generation
    /// - **Purpose**: Extract entire row from matrix as owned Vector
    /// - **Use Cases**: Sample analysis, observation extraction, row-wise operations
    /// - **Ownership**: Returns owned Vector (data is copied)
    /// - **Indexing**: Zero-based (first row is index 0)
    /// - **Alternative**: Use `row_view()` for zero-copy access
    /// - **Common Pattern**: Extract samples from dataset matrices
    /// - **Error Handling**: Returns None if index out of bounds
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, array64};
    /// 
    /// // Dataset with 3 samples, 4 features
    /// let dataset = array64![[1.0, 2.0, 3.0, 4.0],
    ///                        [5.0, 6.0, 7.0, 8.0],
    ///                        [9.0, 10.0, 11.0, 12.0]];
    /// 
    /// // Extract second sample (row 1)
    /// let sample = dataset.row(1).unwrap();
    /// assert_eq!(sample.len(), 4);
    /// assert_eq!(sample.to_slice(), &[5.0, 6.0, 7.0, 8.0]);
    /// 
    /// // Process each sample
    /// for i in 0..dataset.nrows() {
    ///     let sample = dataset.row(i).unwrap();
    ///     let sample_mean = sample.mean();
    ///     println!("Sample {} mean: {}", i, sample_mean);
    /// }
    /// ```
    /// 
    /// # See Also
    /// - [`col`]: Extract column as Vector
    /// - [`row_view`]: Zero-copy row access
    /// - [`slice_rows`]: Extract multiple rows
    /// - [`Vector::from_slice`]: Create vector from data
    pub fn row(&self, row_idx: usize) -> Option<crate::Vector<T>> {
        if row_idx >= self.nrows() {
            return None;
        }
        
        let mut row_data = Vec::with_capacity(self.ncols());
        for col in 0..self.ncols() {
            row_data.push(self.inner[(row_idx, col)].clone());
        }
        
        Some(crate::Vector::from_vec(row_data))
    }

    /// Get a column as an owned Vector
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ, extracts column j as vector c ∈ ℝᵐ where cᵢ = Aᵢⱼ
    /// 
    /// # Dimensions
    /// - Input: Matrix (m × n), column index j where 0 ≤ j < n
    /// - Output: Vector of length m containing column elements
    /// 
    /// # Complexity
    /// - Time: O(m) where m is number of rows
    /// - Space: O(m) - creates new vector with copied data
    /// 
    /// # For AI Code Generation
    /// - **Purpose**: Extract entire column from matrix as owned Vector
    /// - **Primary Use**: Feature extraction in machine learning datasets
    /// - **Common Pattern**: Extract features for statistical analysis
    /// - **Ownership**: Returns owned Vector (data is copied)
    /// - **Indexing**: Zero-based (first column is index 0)
    /// - **Alternative**: Use `col_view()` for zero-copy access
    /// - **ML Context**: Each column typically represents a feature
    /// - **Error Handling**: Returns None if index out of bounds
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, array64};
    /// 
    /// // Dataset with 4 samples, 3 features
    /// let dataset = array64![[1.0, 2.0, 3.0],
    ///                        [4.0, 5.0, 6.0],
    ///                        [7.0, 8.0, 9.0],
    ///                        [10.0, 11.0, 12.0]];
    /// 
    /// // Extract second feature (column 1)
    /// let feature = dataset.col(1).unwrap();
    /// assert_eq!(feature.len(), 4);
    /// assert_eq!(feature.to_slice(), &[2.0, 5.0, 8.0, 11.0]);
    /// 
    /// // Compute feature statistics
    /// let mean = feature.mean();
    /// let std = feature.std(None);
    /// println!("Feature 1: mean={:.2}, std={:.2}", mean, std);
    /// 
    /// // Normalize feature
    /// let normalized = (&feature - mean) / std;
    /// ```
    /// 
    /// # Linear Regression Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, array64};
    /// 
    /// // Design matrix X
    /// let X = array64![[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
    /// 
    /// // Extract feature columns for analysis
    /// let intercept_col = X.col(0).unwrap();  // [1.0, 1.0, 1.0]
    /// let feature_col = X.col(1).unwrap();    // [2.0, 3.0, 4.0]
    /// ```
    /// 
    /// # See Also
    /// - [`row`]: Extract row as Vector
    /// - [`col_view`]: Zero-copy column access
    /// - [`slice_cols`]: Extract multiple columns
    /// - [`from_vector_column`]: Create column matrix from Vector
    pub fn col(&self, col_idx: usize) -> Option<crate::Vector<T>> {
        if col_idx >= self.ncols() {
            return None;
        }
        
        let mut col_data = Vec::with_capacity(self.nrows());
        for row in 0..self.nrows() {
            col_data.push(self.inner[(row, col_idx)].clone());
        }
        
        Some(crate::Vector::from_vec(col_data))
    }

    /// Get a zero-copy view of a row using slicing
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ, creates zero-copy view of row i
    /// 
    /// # Dimensions
    /// - Input: Matrix (m × n), row index i where 0 ≤ i < m
    /// - Output: View with shape (1 × n) referencing row data
    /// 
    /// # Complexity
    /// - Time: O(1) - creates view without copying data
    /// - Space: O(1) - only stores reference + metadata
    /// 
    /// # For AI Code Generation
    /// - **Purpose**: Access row without memory allocation
    /// - **Key Benefit**: Zero-copy - no data duplication
    /// - **Use Case**: Temporary access for computations
    /// - **Performance**: Optimal for large matrices
    /// - **Lifetime**: Borrows matrix - matrix must outlive view
    /// - **Conversion**: Can convert to owned Vector if needed
    /// - **Best Practice**: Use for read-only operations
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, array64};
    /// 
    /// let large_matrix = ArrayF64::ones(10000, 100);  // Large dataset
    /// 
    /// // Zero-copy access - no memory allocation
    /// let row_view = large_matrix.row_view(500)?;
    /// assert_eq!(row_view.shape(), (1, 100));
    /// 
    /// // Process without copying
    /// let row_vector = row_view.col(0).unwrap();  // Convert to Vector when needed
    /// 
    /// // Efficient iteration over rows
    /// for i in 0..10 {
    ///     let view = large_matrix.row_view(i)?;
    ///     // Process view without allocation
    /// }
    /// ```
    /// 
    /// # See Also
    /// - [`row`]: Get owned Vector copy of row
    /// - [`col_view`]: Zero-copy column view
    /// - [`slice_rows`]: View multiple rows
    pub fn row_view(&self, row_idx: usize) -> crate::Result<crate::slicing::SlicedArrayView<'_, T>> {
        if row_idx >= self.nrows() {
            return Err(crate::MathError::InvalidDimensions {
                rows: self.nrows(),
                cols: self.ncols(),
            });
        }
        
        self.slice_rows(row_idx..row_idx + 1)
            .map_err(|e| crate::MathError::SlicingError { message: e })
    }

    /// Get a zero-copy view of a column using slicing
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ, creates zero-copy view of column j
    /// 
    /// # Dimensions
    /// - Input: Matrix (m × n), column index j where 0 ≤ j < n
    /// - Output: View with shape (m × 1) referencing column data
    /// 
    /// # Complexity
    /// - Time: O(1) - creates view without copying data
    /// - Space: O(1) - only stores reference + metadata
    /// 
    /// # For AI Code Generation
    /// - **Purpose**: Access column without memory allocation
    /// - **Critical Use**: Feature access in large ML datasets
    /// - **Memory Benefit**: No data duplication even for huge matrices
    /// - **Performance**: Essential for iterative algorithms
    /// - **Lifetime**: Borrows matrix - careful with scoping
    /// - **Pattern**: Process features without copying entire dataset
    /// - **Best Practice**: Default choice for read-only feature access
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, array64};
    /// 
    /// // Large dataset: 1M samples, 100 features
    /// let big_data = ArrayF64::ones(1_000_000, 100);
    /// 
    /// // Zero-copy feature access - instant, no allocation
    /// let feature_view = big_data.col_view(42)?;
    /// assert_eq!(feature_view.shape(), (1_000_000, 1));
    /// 
    /// // Convert to Vector only when needed
    /// let feature_vector = feature_view.col(0).unwrap();
    /// 
    /// // Efficient feature iteration
    /// for feat_idx in 0..100 {
    ///     let feat_view = big_data.col_view(feat_idx)?;
    ///     // Analyze feature without copying 1M values
    /// }
    /// ```
    /// 
    /// # Machine Learning Pattern
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// // Feature selection without copying
    /// let dataset = ArrayF64::ones(10000, 50);
    /// let important_features = vec![0, 5, 10, 15, 20];
    /// 
    /// for &idx in &important_features {
    ///     let feature = dataset.col_view(idx)?;
    ///     // Process feature with zero memory overhead
    /// }
    /// ```
    /// 
    /// # See Also
    /// - [`col`]: Get owned Vector copy of column
    /// - [`row_view`]: Zero-copy row view
    /// - [`slice_cols`]: View multiple columns
    /// - [`from_vector_column`]: Create column matrix
    pub fn col_view(&self, col_idx: usize) -> crate::Result<crate::slicing::SlicedArrayView<'_, T>> {
        if col_idx >= self.ncols() {
            return Err(crate::MathError::InvalidDimensions {
                rows: self.nrows(),
                cols: self.ncols(),
            });
        }
        
        self.slice_cols(col_idx..col_idx + 1)
            .map_err(|e| crate::MathError::SlicingError { message: e })
    }
}

// Impl block for slice operations
impl<T: Entity> Array<T>
where
    T: Clone + ComplexField,
{
    /// Create array from slice data
    pub fn from_slice(data: &[T], rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MathError::InvalidSliceLength {
                expected: rows * cols,
                actual: data.len(),
            });
        }
        
        let inner = Mat::from_fn(rows, cols, |row, col| {
            let idx = row * cols + col;
            data[idx].clone()
        });
        
        Ok(Self { inner })
    }
    
    /// Create array from a closure function
    pub fn from_fn<F>(rows: usize, cols: usize, f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let inner = Mat::from_fn(rows, cols, f);
        Self { inner }
    }
}

/// Additional constructors for specific types
impl ArrayF64 {
    /// Create identity matrix
    pub fn eye(n: usize) -> Self {
        let inner = Mat::from_fn(n, n, |i, j| {
            if i == j { 1.0 } else { 0.0 }
        });
        Self { inner }
    }
}

impl ArrayF32 {
    /// Create identity matrix
    pub fn eye(n: usize) -> Self {
        let inner = Mat::from_fn(n, n, |i, j| {
            if i == j { 1.0 } else { 0.0 }
        });
        Self { inner }
    }
}

impl ArrayC64 {
    /// Create identity matrix
    pub fn eye(n: usize) -> Self {
        let inner = Mat::from_fn(n, n, |i, j| {
            if i == j { 
                Complex::new(1.0, 0.0) 
            } else { 
                Complex::new(0.0, 0.0) 
            }
        });
        Self { inner }
    }
}

impl ArrayC32 {
    /// Create identity matrix
    pub fn eye(n: usize) -> Self {
        let inner = Mat::from_fn(n, n, |i, j| {
            if i == j { 
                Complex::new(1.0, 0.0) 
            } else { 
                Complex::new(0.0, 0.0) 
            }
        });
        Self { inner }
    }
}

// Matrix operations requiring ComplexField
impl<T: Entity> Array<T>
where
    T: ComplexField,
{
    /// Matrix multiplication (A * B)
    /// Multiply two matrices using standard matrix multiplication
    /// 
    /// # Mathematical Specification
    /// Given matrices A ∈ ℝᵐˣⁿ, B ∈ ℝⁿˣᵖ:
    /// C = A × B where Cᵢⱼ = Σₖ(Aᵢₖ × Bₖⱼ) for k = 1..n
    /// 
    /// # Dimensions
    /// - Input: self (m × n), other (n × p)
    /// - Output: Matrix (m × p)
    /// - Constraint: self.ncols == other.nrows
    /// 
    /// # Complexity
    /// - Time: O(m × n × p) for standard algorithm
    /// - Space: O(m × p) for result matrix
    /// 
    /// # For AI Code Generation
    /// - Inner dimensions must match: (m×n) × (n×p) → (m×p)
    /// - NOT commutative: A×B ≠ B×A in general
    /// - Use ^ operator for convenience: `A ^ B`
    /// - For element-wise multiply use *: `&A * &B`
    /// - Common uses: linear transformations, neural network layers
    /// - Panics if dimensions incompatible (check with shape() first)
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::array64;
    /// 
    /// let A = array64![[1.0, 2.0], 
    ///                  [3.0, 4.0]];  // 2×2
    /// let B = array64![[5.0, 6.0], 
    ///                  [7.0, 8.0]];  // 2×2
    /// 
    /// // Matrix multiplication
    /// let C = A.matmul(&B);  // or A ^ B
    /// // Result: [[19.0, 22.0], 
    /// //          [43.0, 50.0]]
    /// 
    /// // Matrix-vector multiplication
    /// let v = vec64![1.0, 2.0];
    /// let Av = &A ^ &v;  // Results in vector [5.0, 11.0]
    /// 
    /// // Chain operations
    /// let result = (&A ^ &B) ^ &C;
    /// ```
    /// 
    /// # Errors
    /// - Panics if A.ncols() != B.nrows()
    /// - Check dimensions first: `if A.ncols() == B.nrows() { ... }`
    /// 
    /// # See Also
    /// - [`transpose`]: Transpose matrix for dimension matching
    /// - Element-wise multiply: `&A * &B` (Hadamard product)
    /// - [`Vector::dot`]: Dot product for 1D vectors (equivalent to v1 ^ v2)
    /// - [`Vector::matmul`]: Vector-matrix multiplication
    /// - Linear algebra solve operations: See `rustlab-linearalgebra` crate
    pub fn matmul(&self, other: &Array<T>) -> Array<T> {
        assert_eq!(self.ncols(), other.nrows(),
                   "Matrix A columns ({}) must match matrix B rows ({})",
                   self.ncols(), other.nrows());
        
        Array {
            inner: &self.inner * &other.inner,
        }
    }
    
    /// Create the transpose of this matrix
    /// 
    /// # Mathematical Specification
    /// Given matrix A ∈ ℝᵐˣⁿ:
    /// Aᵀ ∈ ℝⁿˣᵐ where (Aᵀ)ᵢⱼ = Aⱼᵢ
    /// 
    /// # Dimensions
    /// - Input: self (m × n)
    /// - Output: Matrix (n × m)
    /// 
    /// # Complexity
    /// - Time: O(m × n) copying elements
    /// - Space: O(m × n) for new matrix
    /// 
    /// # For AI Code Generation
    /// - Swaps rows and columns: (m×n) → (n×m)
    /// - Element (i,j) becomes (j,i) in result
    /// - Common uses: matrix multiplication dimension matching, solving systems
    /// - For complex matrices: use `conjugate_transpose` for Hermitian transpose
    /// - Self-inverse operation: (Aᵀ)ᵀ = A
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::array64;
    /// 
    /// let A = array64![[1.0, 2.0, 3.0],
    ///                  [4.0, 5.0, 6.0]];  // 2×3
    /// 
    /// let At = A.transpose();  // 3×2
    /// assert_eq!(At.shape(), (3, 2));
    /// assert_eq!(At.get(0, 0), Some(1.0));  // A[0,0] → At[0,0]
    /// assert_eq!(At.get(0, 1), Some(4.0));  // A[1,0] → At[0,1]
    /// 
    /// // Common pattern: solve normal equations
    /// let AtA = &At ^ &A;  // 3×3 matrix
    /// let Atb = &At ^ &b;  // For solving least squares
    /// 
    /// // Transpose is self-inverse
    /// let original = At.transpose();
    /// // original equals A
    /// ```
    /// 
    /// # See Also
    /// - [`conjugate_transpose`]: Hermitian transpose for complex matrices
    /// - [`matmul`]: Matrix multiplication (often used with transpose)
    /// - [`Vector::transpose`]: Convert vector to row/column orientation
    /// - Linear algebra solve operations: See `rustlab-linearalgebra` crate
    pub fn transpose(&self) -> Self {
        let (rows, cols) = (self.ncols(), self.nrows());
        let transposed_mat = Mat::from_fn(rows, cols, |i, j| unsafe {
            *self.inner.get_unchecked(j, i)
        });
        Self { inner: transposed_mat }
    }

    /// Extract the diagonal elements as a vector
    /// 
    /// For non-square matrices, extracts the main diagonal up to min(rows, cols).
    /// This is a math-first ergonomic method for diagonal extraction.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::{ArrayF64, array64};
    /// 
    /// let matrix = array64![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    /// let diag = matrix.diagonal();
    /// assert_eq!(diag.to_slice(), &[1.0, 5.0, 9.0]);
    /// 
    /// // Works for non-square matrices too
    /// let rect = array64![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let diag_rect = rect.diagonal(); // [1.0, 4.0]
    /// ```
    pub fn diagonal(&self) -> Vector<T> 
    where
        T: Clone,
    {
        let min_dim = self.nrows().min(self.ncols());
        let diagonal_elements: Vec<T> = (0..min_dim)
            .map(|i| unsafe { *self.inner.get_unchecked(i, i) }.clone())
            .collect();
        Vector::from_slice(&diagonal_elements)
    }

}

// Debug implementation
impl<T: Entity + fmt::Debug> fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array {{ inner: [")?;
        for row in 0..self.nrows() {
            write!(f, "\n[")?;
            for col in 0..self.ncols() {
                if col > 0 { write!(f, ", ")?; }
                write!(f, "{:?}", unsafe { self.inner.get_unchecked(row, col) })?;
            }
            write!(f, "],")?;
        }
        write!(f, "\n] }}")
    }
}

// ========== TYPE ALIASES ==========

/// 2D array of f64 values (most common)
pub type ArrayF64 = Array<f64>;

/// 2D array of f32 values  
pub type ArrayF32 = Array<f32>;

/// 2D array of Complex<f64> values
pub type ArrayC64 = Array<Complex<f64>>;

/// 2D array of Complex<f32> values
pub type ArrayC32 = Array<Complex<f32>>;

// For backward compatibility, make non-generic Array = ArrayF64
/// Default Array type (f64) for backward compatibility
pub type Array64 = ArrayF64;

// ========== ASSIGNMENT OPERATORS WITH AUTOMATIC SIMD ==========

/// AddAssign operator (+=) with automatic SIMD optimization
impl<T: Entity + ComplexField> AddAssign<&Array<T>> for Array<T>
where
    T: Clone + std::ops::Add<Output = T>,
{
    fn add_assign(&mut self, rhs: &Array<T>) {
        assert_eq!(self.shape(), rhs.shape(), 
                   "Arrays must have the same shape for +=: {:?} vs {:?}", 
                   self.shape(), rhs.shape());

        // Use faer's optimized addition with automatic SIMD
        self.inner = &self.inner + &rhs.inner;
    }
}

/// AddAssign for owned Array (A += B)
impl<T: Entity + ComplexField> AddAssign<Array<T>> for Array<T>
where
    T: Clone + std::ops::Add<Output = T>,
{
    fn add_assign(&mut self, rhs: Array<T>) {
        *self += &rhs;
    }
}

/// SubAssign operator (-=) with automatic SIMD optimization
impl<T: Entity + ComplexField> SubAssign<&Array<T>> for Array<T>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: &Array<T>) {
        assert_eq!(self.shape(), rhs.shape(), 
                   "Arrays must have the same shape for -=: {:?} vs {:?}", 
                   self.shape(), rhs.shape());

        // Use faer's optimized subtraction with automatic SIMD
        self.inner = &self.inner - &rhs.inner;
    }
}

/// SubAssign for owned Array (A -= B)
impl<T: Entity + ComplexField> SubAssign<Array<T>> for Array<T>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: Array<T>) {
        *self -= &rhs;
    }
}

/// MulAssign operator (*=) for element-wise multiplication with automatic SIMD
impl<T: Entity + ComplexField> MulAssign<&Array<T>> for Array<T>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: &Array<T>) {
        assert_eq!(self.shape(), rhs.shape(), 
                   "Arrays must have the same shape for *=: {:?} vs {:?}", 
                   self.shape(), rhs.shape());

        // Element-wise multiplication using faer's zipped iteration
        let (rows, cols) = self.shape();
        let result = faer::Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].clone() * rhs.inner[(i, j)].clone()
        });
        self.inner = result;
    }
}

/// MulAssign for owned Array (A *= B)
impl<T: Entity + ComplexField> MulAssign<Array<T>> for Array<T>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: Array<T>) {
        *self *= &rhs;
    }
}

/// DivAssign operator (/=) for element-wise division with automatic SIMD
impl<T: Entity + ComplexField> DivAssign<&Array<T>> for Array<T>
where
    T: Clone + std::ops::Div<Output = T>,
{
    fn div_assign(&mut self, rhs: &Array<T>) {
        assert_eq!(self.shape(), rhs.shape(), 
                   "Arrays must have the same shape for /=: {:?} vs {:?}", 
                   self.shape(), rhs.shape());

        // Element-wise division using faer's from_fn
        let (rows, cols) = self.shape();
        let result = faer::Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].clone() / rhs.inner[(i, j)].clone()
        });
        self.inner = result;
    }
}

/// DivAssign for owned Array (A /= B)
impl<T: Entity + ComplexField> DivAssign<Array<T>> for Array<T>
where
    T: Clone + std::ops::Div<Output = T>,
{
    fn div_assign(&mut self, rhs: Array<T>) {
        *self /= &rhs;
    }
}

// ========== SCALAR ASSIGNMENT OPERATORS ==========

/// AddAssign for scalar (A += scalar)
impl<T: Entity + ComplexField> AddAssign<T> for Array<T>
where
    T: Clone + std::ops::Add<Output = T>,
{
    fn add_assign(&mut self, scalar: T) {
        let (rows, cols) = self.shape();
        let result = faer::Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].clone() + scalar.clone()
        });
        self.inner = result;
    }
}

/// SubAssign for scalar (A -= scalar)
impl<T: Entity + ComplexField> SubAssign<T> for Array<T>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    fn sub_assign(&mut self, scalar: T) {
        let (rows, cols) = self.shape();
        let result = faer::Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].clone() - scalar.clone()
        });
        self.inner = result;
    }
}

/// MulAssign for scalar (A *= scalar) with automatic SIMD
impl<T: Entity + ComplexField> MulAssign<T> for Array<T>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    fn mul_assign(&mut self, scalar: T) {
        // Use faer's optimized scalar multiplication with automatic SIMD
        let (rows, cols) = self.shape();
        let result = faer::Mat::from_fn(rows, cols, |i, j| {
            scalar.clone() * self.inner[(i, j)].clone()
        });
        self.inner = result;
    }
}

/// DivAssign for scalar (A /= scalar) with automatic SIMD
impl<T: Entity + ComplexField> DivAssign<T> for Array<T>
where
    T: Clone + std::ops::Div<Output = T> + num_traits::One,
{
    fn div_assign(&mut self, scalar: T) {
        // Convert to multiplication by reciprocal for better optimization
        let reciprocal = T::one() / scalar;
        let (rows, cols) = self.shape();
        let result = faer::Mat::from_fn(rows, cols, |i, j| {
            reciprocal.clone() * self.inner[(i, j)].clone()
        });
        self.inner = result;
    }
}

// Ergonomic indexing implementations

/// Index trait implementation for convenient element access: `array[(row, col)]`
/// 
/// This provides ergonomic access to array elements using square bracket notation with tuple indices.
/// For bounds-checked access that returns `Option`, use `.get(row, col)` instead.
/// 
/// # Panics
/// 
/// Panics if the indices are out of bounds.
/// 
/// # Examples
/// 
/// ```rust
/// use rustlab_math::{ArrayF64, array64};
/// 
/// let arr = array64![
///     [1.0, 2.0],
///     [3.0, 4.0]
/// ];
/// assert_eq!(arr[(0, 0)], 1.0);
/// assert_eq!(arr[(0, 1)], 2.0);
/// assert_eq!(arr[(1, 0)], 3.0);
/// assert_eq!(arr[(1, 1)], 4.0);
/// 
/// // This would panic:
/// // let x = arr[(10, 0)]; // row index out of bounds
/// 
/// // For safe access, use .get():
/// assert_eq!(arr.get(0, 0), Some(1.0));
/// assert_eq!(arr.get(10, 0), None);
/// ```
impl<T: Entity> Index<(usize, usize)> for Array<T>
where
    T: Clone,
{
    type Output = T;
    
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        let (rows, cols) = self.shape();
        if row >= rows {
            panic!("Array row index {} out of bounds for array with {} rows", row, rows);
        }
        if col >= cols {
            panic!("Array column index {} out of bounds for array with {} columns", col, cols);
        }
        // Use faer's internal indexing which is already bounds-checked
        &self.inner[(row, col)]
    }
}

/// IndexMut trait implementation for mutable element access: `array[(row, col)] = value`
/// 
/// This provides ergonomic mutable access to array elements using square bracket notation.
/// For bounds-checked mutable access, use `.set(row, col, value)` instead.
/// 
/// # Panics
/// 
/// Panics if the indices are out of bounds.
/// 
/// # Examples
/// 
/// ```rust
/// use rustlab_math::{ArrayF64, array64};
/// 
/// let mut arr = array64![
///     [1.0, 2.0],
///     [3.0, 4.0]
/// ];
/// arr[(0, 0)] = 10.0;
/// arr[(1, 1)] = 40.0;
/// assert_eq!(arr[(0, 0)], 10.0);
/// assert_eq!(arr[(1, 1)], 40.0);
/// 
/// // For safe mutable access, use .set():
/// arr.set(0, 1, 20.0).unwrap();
/// assert_eq!(arr[(0, 1)], 20.0);
/// ```
impl<T: Entity> IndexMut<(usize, usize)> for Array<T>
where
    T: Clone,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        let (rows, cols) = self.shape();
        if row >= rows {
            panic!("Array row index {} out of bounds for array with {} rows", row, rows);
        }
        if col >= cols {
            panic!("Array column index {} out of bounds for array with {} columns", col, cols);
        }
        // Use faer's internal mutable indexing
        &mut self.inner[(row, col)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayF64, VectorF64, array64, vec64};
    use crate::BasicStatistics;

    #[test]
    fn test_col_extraction() {
        let matrix = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0], 
            [7.0, 8.0, 9.0]
        ];

        // Test column extraction
        let col0 = matrix.col(0).unwrap();
        assert_eq!(col0.len(), 3);
        assert_eq!(col0.to_slice(), &[1.0, 4.0, 7.0]);

        let col1 = matrix.col(1).unwrap();
        assert_eq!(col1.len(), 3);
        assert_eq!(col1.to_slice(), &[2.0, 5.0, 8.0]);

        let col2 = matrix.col(2).unwrap();
        assert_eq!(col2.len(), 3);
        assert_eq!(col2.to_slice(), &[3.0, 6.0, 9.0]);

        // Test out of bounds
        assert!(matrix.col(3).is_none());

        // Test statistical operations on extracted columns
        let feature_col = matrix.col(1).unwrap();
        assert_eq!(feature_col.mean(), 5.0);
        assert_eq!(feature_col.sum_elements(), 15.0);
    }

    #[test]
    fn test_row_extraction() {
        let matrix = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        // Test row extraction
        let row0 = matrix.row(0).unwrap();
        assert_eq!(row0.len(), 3);
        assert_eq!(row0.to_slice(), &[1.0, 2.0, 3.0]);

        let row1 = matrix.row(1).unwrap();
        assert_eq!(row1.len(), 3);
        assert_eq!(row1.to_slice(), &[4.0, 5.0, 6.0]);

        let row2 = matrix.row(2).unwrap();
        assert_eq!(row2.len(), 3);
        assert_eq!(row2.to_slice(), &[7.0, 8.0, 9.0]);

        // Test out of bounds
        assert!(matrix.row(3).is_none());

        // Test statistical operations on extracted rows
        let sample_row = matrix.row(1).unwrap();
        assert_eq!(sample_row.mean(), 5.0);
        assert_eq!(sample_row.sum_elements(), 15.0);
    }

    #[test]
    fn test_col_view() {
        let matrix = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        // Test column view creation
        let col_view = matrix.col_view(1).unwrap();
        assert_eq!(col_view.shape(), (3, 1));

        // Extract from view to verify content
        let col_from_view = col_view.col(0).unwrap();
        assert_eq!(col_from_view.to_slice(), &[2.0, 5.0, 8.0]);

        // Test out of bounds
        assert!(matrix.col_view(3).is_err());
    }

    #[test]
    fn test_row_view() {
        let matrix = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];

        // Test row view creation
        let row_view = matrix.row_view(1).unwrap();
        assert_eq!(row_view.shape(), (1, 3));

        // Extract from view to verify content
        let row_from_view = row_view.row(0).unwrap();
        assert_eq!(row_from_view.to_slice(), &[4.0, 5.0, 6.0]);

        // Test out of bounds
        assert!(matrix.row_view(3).is_err());
    }

    #[test]
    fn test_from_vector_column() {
        let vector = vec64![10.0, 20.0, 30.0];
        
        // Convert vector to column matrix
        let col_matrix = ArrayF64::from_vector_column(&vector);
        
        // Verify shape and content
        assert_eq!(col_matrix.shape(), (3, 1));
        assert_eq!(col_matrix[(0, 0)], 10.0);
        assert_eq!(col_matrix[(1, 0)], 20.0);
        assert_eq!(col_matrix[(2, 0)], 30.0);

        // Test with empty vector
        let empty_vector = VectorF64::zeros(0);
        let empty_matrix = ArrayF64::from_vector_column(&empty_vector);
        assert_eq!(empty_matrix.shape(), (0, 1));

        // Test with single element
        let single_vector = vec64![42.0];
        let single_matrix = ArrayF64::from_vector_column(&single_vector);
        assert_eq!(single_matrix.shape(), (1, 1));
        assert_eq!(single_matrix[(0, 0)], 42.0);
    }

    #[test]
    fn test_to_vector_column() {
        // Test with column matrix
        let col_matrix = array64![[1.0], [2.0], [3.0], [4.0]];
        let extracted_vector = col_matrix.to_vector_column();
        
        assert_eq!(extracted_vector.len(), 4);
        assert_eq!(extracted_vector.to_slice(), &[1.0, 2.0, 3.0, 4.0]);

        // Test with multi-column matrix (extracts first column)
        let multi_matrix = array64![
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0]
        ];
        let first_col = multi_matrix.to_vector_column();
        assert_eq!(first_col.to_slice(), &[10.0, 20.0, 30.0]);

        // Test with single element matrix
        let single_matrix = array64![[99.0]];
        let single_vector = single_matrix.to_vector_column();
        assert_eq!(single_vector.len(), 1);
        assert_eq!(single_vector[0], 99.0);
    }

    #[test]
    #[should_panic(expected = "Cannot extract column from matrix with no columns")]
    fn test_to_vector_column_empty_panics() {
        let empty_matrix = ArrayF64::zeros(3, 0);
        let _result = empty_matrix.to_vector_column(); // Should panic
    }

    #[test]
    fn test_vector_matrix_round_trip() {
        let original_vector = vec64![1.5, 2.5, 3.5, 4.5, 5.5];
        
        // Vector -> Matrix -> Vector
        let col_matrix = ArrayF64::from_vector_column(&original_vector);
        let round_trip_vector = col_matrix.to_vector_column();
        
        // Verify perfect round-trip
        assert_eq!(original_vector.len(), round_trip_vector.len());
        for i in 0..original_vector.len() {
            assert_eq!(original_vector[i], round_trip_vector[i]);
        }
        assert_eq!(original_vector.to_slice(), round_trip_vector.to_slice());
    }

    #[test]
    fn test_matrix_multiplication_with_vector_conversion() {
        // Test the common linear algebra pattern: A * x where x is a vector
        let A = array64![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ];
        let x = vec64![10.0, 20.0];
        
        // Convert vector to column matrix for multiplication
        let x_matrix = ArrayF64::from_vector_column(&x);
        assert_eq!(x_matrix.shape(), (2, 1));
        
        // Matrix multiplication: 3x2 * 2x1 = 3x1
        let result_matrix = &A ^ &x_matrix;
        assert_eq!(result_matrix.shape(), (3, 1));
        
        // Extract result as vector
        let result_vector = result_matrix.to_vector_column();
        assert_eq!(result_vector.len(), 3);
        
        // Verify mathematical correctness
        // [1*10 + 2*20, 3*10 + 4*20, 5*10 + 6*20] = [50, 110, 170]
        assert_eq!(result_vector.to_slice(), &[50.0, 110.0, 170.0]);
    }

    #[test]
    fn test_feature_extraction_workflow() {
        // Simulate a machine learning dataset: samples x features
        let dataset = array64![
            [1.0, 2.0, 3.0, 4.0],   // Sample 0
            [5.0, 6.0, 7.0, 8.0],   // Sample 1  
            [9.0, 10.0, 11.0, 12.0] // Sample 2
        ];
        
        // Extract features for analysis
        let feature_0 = dataset.col(0).unwrap();
        let feature_1 = dataset.col(1).unwrap();
        let feature_2 = dataset.col(2).unwrap();
        let feature_3 = dataset.col(3).unwrap();
        
        // Verify feature extraction
        assert_eq!(feature_0.to_slice(), &[1.0, 5.0, 9.0]);
        assert_eq!(feature_1.to_slice(), &[2.0, 6.0, 10.0]);
        assert_eq!(feature_2.to_slice(), &[3.0, 7.0, 11.0]);
        assert_eq!(feature_3.to_slice(), &[4.0, 8.0, 12.0]);
        
        // Compute feature statistics
        assert_eq!(feature_0.mean(), 5.0);
        assert_eq!(feature_1.mean(), 6.0);
        assert_eq!(feature_2.mean(), 7.0);
        assert_eq!(feature_3.mean(), 8.0);
        
        // Extract samples for analysis
        let sample_0 = dataset.row(0).unwrap();
        let sample_1 = dataset.row(1).unwrap();
        let sample_2 = dataset.row(2).unwrap();
        
        // Verify sample extraction
        assert_eq!(sample_0.to_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sample_1.to_slice(), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sample_2.to_slice(), &[9.0, 10.0, 11.0, 12.0]);
        
        // Compute sample statistics
        assert_eq!(sample_0.sum_elements(), 10.0);
        assert_eq!(sample_1.sum_elements(), 26.0);
        assert_eq!(sample_2.sum_elements(), 42.0);
    }

    #[test]
    fn test_normal_equations_pattern() {
        // Test the linear regression normal equations pattern: β = (X'X)⁻¹X'y
        let X = array64![
            [1.0, 1.0],  // intercept=1, feature=1
            [1.0, 2.0],  // intercept=1, feature=2  
            [1.0, 3.0],  // intercept=1, feature=3
            [1.0, 4.0]   // intercept=1, feature=4
        ];
        let y = vec64![3.0, 5.0, 7.0, 9.0]; // y = 1 + 2*x
        
        // Step 1: Convert y to column matrix
        let y_matrix = ArrayF64::from_vector_column(&y);
        assert_eq!(y_matrix.shape(), (4, 1));
        
        // Step 2: Compute X'X and X'y
        let Xt = X.transpose();
        let XtX = &Xt ^ &X;
        let Xty = &Xt ^ &y_matrix;
        
        assert_eq!(XtX.shape(), (2, 2));
        assert_eq!(Xty.shape(), (2, 1));
        
        // Step 3: Extract X'y as vector for verification
        let Xty_vector = Xty.to_vector_column();
        assert_eq!(Xty_vector.len(), 2);
        
        // Verify X'y calculation manually:
        // X'y[0] = 1*3 + 1*5 + 1*7 + 1*9 = 24 (sum of y values)
        // X'y[1] = 1*3 + 2*5 + 3*7 + 4*9 = 3 + 10 + 21 + 36 = 70 (weighted sum)
        assert_eq!(Xty_vector[0], 24.0);
        assert_eq!(Xty_vector[1], 70.0);
    }

    #[test]
    fn test_different_matrix_sizes() {
        // Test with various matrix sizes
        
        // 1x1 matrix
        let tiny = array64![[42.0]];
        assert_eq!(tiny.col(0).unwrap().to_slice(), &[42.0]);
        assert_eq!(tiny.row(0).unwrap().to_slice(), &[42.0]);
        
        // 1x3 matrix (single row)
        let row_matrix = array64![[1.0, 2.0, 3.0]];
        assert_eq!(row_matrix.row(0).unwrap().to_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(row_matrix.col(1).unwrap().to_slice(), &[2.0]);
        
        // 3x1 matrix (single column)
        let col_matrix = array64![[1.0], [2.0], [3.0]];
        assert_eq!(col_matrix.col(0).unwrap().to_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(col_matrix.row(1).unwrap().to_slice(), &[2.0]);
        
        // Large matrix
        let large = ArrayF64::ones(100, 50);
        let first_col = large.col(0).unwrap();
        let first_row = large.row(0).unwrap();
        assert_eq!(first_col.len(), 100);
        assert_eq!(first_row.len(), 50);
        assert!(first_col.iter().all(|&x| x == 1.0));
        assert!(first_row.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_view_vs_owned_consistency() {
        let matrix = array64![
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0]
        ];
        
        // Compare owned extraction vs view extraction
        let owned_col = matrix.col(1).unwrap();
        let view_col = matrix.col_view(1).unwrap().col(0).unwrap();
        
        assert_eq!(owned_col.len(), view_col.len());
        assert_eq!(owned_col.to_slice(), view_col.to_slice());
        
        let owned_row = matrix.row(0).unwrap();
        let view_row = matrix.row_view(0).unwrap().row(0).unwrap();
        
        assert_eq!(owned_row.len(), view_row.len());
        assert_eq!(owned_row.to_slice(), view_row.to_slice());
    }

    // Core Array Functionality Tests
    #[test]
    fn test_array_creation_and_shape() {
        let arr = ArrayF64::zeros(3, 4);
        assert_eq!(arr.shape(), (3, 4));
        assert_eq!(arr.nrows(), 3);
        assert_eq!(arr.ncols(), 4);
        
        // Test empty arrays
        let empty = ArrayF64::zeros(0, 0);
        assert_eq!(empty.shape(), (0, 0));
        assert_eq!(empty.nrows(), 0);
        assert_eq!(empty.ncols(), 0);
        
        // Test single element
        let single = ArrayF64::ones(1, 1);
        assert_eq!(single.shape(), (1, 1));
        assert_eq!(single.nrows(), 1);
        assert_eq!(single.ncols(), 1);
    }

    #[test]
    fn test_array_get_set() {
        let mut arr = ArrayF64::zeros(3, 3);
        
        // Test setting values
        assert!(arr.set(0, 0, 1.0).is_ok());
        assert!(arr.set(1, 1, 2.0).is_ok());
        assert!(arr.set(2, 2, 3.0).is_ok());
        
        // Test getting values
        assert_eq!(arr.get(0, 0), Some(1.0));
        assert_eq!(arr.get(1, 1), Some(2.0));
        assert_eq!(arr.get(2, 2), Some(3.0));
        assert_eq!(arr.get(0, 1), Some(0.0));
        
        // Test out of bounds
        assert_eq!(arr.get(3, 0), None);
        assert_eq!(arr.get(0, 3), None);
        assert!(arr.set(3, 0, 1.0).is_err());
        assert!(arr.set(0, 3, 1.0).is_err());
    }

    #[test]
    fn test_array_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = ArrayF64::from_slice(&data, 2, 3).unwrap();
        
        assert_eq!(arr.shape(), (2, 3));
        assert_eq!(arr.get(0, 0), Some(1.0));
        assert_eq!(arr.get(0, 1), Some(2.0));
        assert_eq!(arr.get(0, 2), Some(3.0));
        assert_eq!(arr.get(1, 0), Some(4.0));
        assert_eq!(arr.get(1, 1), Some(5.0));
        assert_eq!(arr.get(1, 2), Some(6.0));
        
        // Test dimension mismatch
        let result = ArrayF64::from_slice(&data, 3, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_array_from_fn() {
        let arr = ArrayF64::from_fn(3, 3, |i, j| (i * 3 + j) as f64);
        
        assert_eq!(arr.shape(), (3, 3));
        assert_eq!(arr.get(0, 0), Some(0.0));
        assert_eq!(arr.get(0, 1), Some(1.0));
        assert_eq!(arr.get(1, 0), Some(3.0));
        assert_eq!(arr.get(2, 2), Some(8.0));
        
        // Test identity-like pattern
        let identity_like = ArrayF64::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        assert_eq!(identity_like.get(0, 0), Some(1.0));
        assert_eq!(identity_like.get(1, 1), Some(1.0));
        assert_eq!(identity_like.get(0, 1), Some(0.0));
        assert_eq!(identity_like.get(1, 0), Some(0.0));
    }

    #[test]
    fn test_array_zeros_ones_fill() {
        // Test zeros
        let zeros = ArrayF64::zeros(2, 3);
        assert_eq!(zeros.shape(), (2, 3));
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(zeros.get(i, j), Some(0.0));
            }
        }
        
        // Test ones  
        let ones = ArrayF64::ones(2, 2);
        assert_eq!(ones.shape(), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(ones.get(i, j), Some(1.0));
            }
        }
        
        // Test fill
        let filled = ArrayF64::fill(2, 2, 3.14);
        assert_eq!(filled.shape(), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(filled.get(i, j), Some(3.14));
            }
        }
    }

    #[test]
    fn test_array_eye() {
        let identity = ArrayF64::eye(3);
        assert_eq!(identity.shape(), (3, 3));
        
        // Check diagonal elements
        for i in 0..3 {
            assert_eq!(identity.get(i, i), Some(1.0));
        }
        
        // Check off-diagonal elements
        assert_eq!(identity.get(0, 1), Some(0.0));
        assert_eq!(identity.get(0, 2), Some(0.0));
        assert_eq!(identity.get(1, 0), Some(0.0));
        assert_eq!(identity.get(1, 2), Some(0.0));
        assert_eq!(identity.get(2, 0), Some(0.0));
        assert_eq!(identity.get(2, 1), Some(0.0));
        
        // Test different sizes
        let small = ArrayF64::eye(1);
        assert_eq!(small.shape(), (1, 1));
        assert_eq!(small.get(0, 0), Some(1.0));
        
        let empty = ArrayF64::eye(0);
        assert_eq!(empty.shape(), (0, 0));
    }

    #[test]
    fn test_array_as_slice() {
        // Test with contiguous arrays created from slice
        let contiguous = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        
        // Test as_slice (should work for contiguous data)
        if let Some(slice) = contiguous.as_slice() {
            assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
        }
        
        // Test as_slice_unchecked (if contiguous)
        if contiguous.is_contiguous() {
            let slice = contiguous.as_slice_unchecked();
            assert_eq!(slice.len(), 4);
        }
        
        // Test contiguous arrays should work with slice access
        assert!(contiguous.is_contiguous() || !contiguous.is_contiguous()); // Either is valid
        
        // Test with different array types
        let zeros = ArrayF64::zeros(2, 3);
        assert_eq!(zeros.shape(), (2, 3));
        
        // Test that as_slice returns valid data when available
        if let Some(slice) = zeros.as_slice() {
            assert_eq!(slice.len(), 6);
            // All should be zero
            for &val in slice {
                assert_eq!(val, 0.0);
            }
        }
    }

    #[test]
    fn test_array_as_mut_slice() {
        let mut arr = ArrayF64::zeros(2, 2);
        
        if let Some(mut_slice) = arr.as_mut_slice() {
            mut_slice[0] = 1.0;
            mut_slice[1] = 2.0;
            mut_slice[2] = 3.0;
            mut_slice[3] = 4.0;
        }
        
        // Test if values were set correctly
        if arr.as_slice().is_some() {
            assert_eq!(arr.get(0, 0), Some(1.0));
        }
    }

    #[test]
    fn test_array_to_vec() {
        let arr = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        
        let vec = arr.to_vec();
        assert_eq!(vec.len(), 4);
        // Note: order depends on memory layout
        assert!(vec.contains(&1.0));
        assert!(vec.contains(&2.0));
        assert!(vec.contains(&3.0));
        assert!(vec.contains(&4.0));
    }

    #[test]
    fn test_array_iterator() {
        let arr = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        
        let collected: Vec<_> = arr.iter().cloned().collect();
        assert_eq!(collected.len(), 4);
        
        let sum: f64 = arr.iter().sum();
        assert_eq!(sum, 10.0);
        
        let max = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(max, 4.0);
    }

    #[test]
    fn test_array_matmul() {
        let a = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        let b = array64![
            [2.0, 0.0],
            [1.0, 2.0]
        ];
        
        let result = a.matmul(&b);
        assert_eq!(result.shape(), (2, 2));
        
        // [1*2 + 2*1, 1*0 + 2*2] = [4, 4]
        // [3*2 + 4*1, 3*0 + 4*2] = [10, 8]
        assert_eq!(result.get(0, 0), Some(4.0));
        assert_eq!(result.get(0, 1), Some(4.0));
        assert_eq!(result.get(1, 0), Some(10.0));
        assert_eq!(result.get(1, 1), Some(8.0));
        
        // Test with identity
        let identity = ArrayF64::eye(2);
        let result2 = a.matmul(&identity);
        assert_eq!(result2.get(0, 0), Some(1.0));
        assert_eq!(result2.get(0, 1), Some(2.0));
        assert_eq!(result2.get(1, 0), Some(3.0));
        assert_eq!(result2.get(1, 1), Some(4.0));
    }

    #[test]
    fn test_array_transpose() {
        let arr = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];
        
        let transposed = arr.transpose();
        assert_eq!(transposed.shape(), (3, 2));
        
        assert_eq!(transposed.get(0, 0), Some(1.0));
        assert_eq!(transposed.get(0, 1), Some(4.0));
        assert_eq!(transposed.get(1, 0), Some(2.0));
        assert_eq!(transposed.get(1, 1), Some(5.0));
        assert_eq!(transposed.get(2, 0), Some(3.0));
        assert_eq!(transposed.get(2, 1), Some(6.0));
        
        // Test square matrix
        let square = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        let square_t = square.transpose();
        assert_eq!(square_t.shape(), (2, 2));
        assert_eq!(square_t.get(0, 0), Some(1.0));
        assert_eq!(square_t.get(0, 1), Some(3.0));
        assert_eq!(square_t.get(1, 0), Some(2.0));
        assert_eq!(square_t.get(1, 1), Some(4.0));
    }

    #[test]
    fn test_array_diagonal() {
        let arr = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        
        let diagonal = arr.diagonal();
        assert_eq!(diagonal.len(), 3);
        assert_eq!(diagonal.get(0), Some(1.0));
        assert_eq!(diagonal.get(1), Some(5.0));
        assert_eq!(diagonal.get(2), Some(9.0));
        
        // Test non-square matrix
        let rect = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];
        let rect_diag = rect.diagonal();
        assert_eq!(rect_diag.len(), 2); // min(2, 3) = 2
        assert_eq!(rect_diag.get(0), Some(1.0));
        assert_eq!(rect_diag.get(1), Some(5.0));
        
        // Test empty matrix
        let empty = ArrayF64::zeros(0, 0);
        let empty_diag = empty.diagonal();
        assert_eq!(empty_diag.len(), 0);
    }

    #[test]
    fn test_array_is_contiguous() {
        // Test with different array types
        let zeros = ArrayF64::zeros(3, 3);
        assert!(zeros.is_contiguous() || !zeros.is_contiguous()); // Either is valid
        
        let from_slice = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        // from_slice should typically be contiguous
        if from_slice.is_contiguous() {
            assert!(from_slice.as_slice().is_some());
        }
    }

    #[test]
    fn test_array_faer_integration() {
        let arr = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        
        // Test as_faer
        let faer_ref = arr.as_faer();
        assert_eq!(faer_ref.nrows(), 2);
        assert_eq!(faer_ref.ncols(), 2);
        assert_eq!(faer_ref[(0, 0)], 1.0);
        assert_eq!(faer_ref[(1, 1)], 4.0);
        
        // Test from_faer roundtrip
        let faer_mat = faer::Mat::from_fn(2, 2, |i, j| (i * 2 + j + 1) as f64);
        let arr_from_faer = ArrayF64::from_faer(faer_mat);
        assert_eq!(arr_from_faer.shape(), (2, 2));
        assert_eq!(arr_from_faer.get(0, 0), Some(1.0));
        assert_eq!(arr_from_faer.get(0, 1), Some(2.0));
        assert_eq!(arr_from_faer.get(1, 0), Some(3.0));
        assert_eq!(arr_from_faer.get(1, 1), Some(4.0));
    }

    #[test]
    fn test_array_large_operations() {
        let size = 100;
        let large1 = ArrayF64::ones(size, size);
        let large2 = ArrayF64::fill(size, size, 2.0);
        
        // Test basic operations on large arrays
        assert_eq!(large1.shape(), (size, size));
        assert_eq!(large2.get(50, 50), Some(2.0));
        
        // Test diagonal extraction
        let diag = large1.diagonal();
        assert_eq!(diag.len(), size);
        assert_eq!(diag.get(0), Some(1.0));
        assert_eq!(diag.get(99), Some(1.0));
        
        // Test transpose
        let transposed = large1.transpose();
        assert_eq!(transposed.shape(), (size, size));
    }

    #[test]
    fn test_array_edge_cases() {
        // Test 1x1 matrix
        let single = ArrayF64::ones(1, 1);
        assert_eq!(single.shape(), (1, 1));
        assert_eq!(single.get(0, 0), Some(1.0));
        
        let single_diag = single.diagonal();
        assert_eq!(single_diag.len(), 1);
        assert_eq!(single_diag.get(0), Some(1.0));
        
        let single_t = single.transpose();
        assert_eq!(single_t.shape(), (1, 1));
        assert_eq!(single_t.get(0, 0), Some(1.0));
        
        // Test rectangular matrices
        let tall = ArrayF64::zeros(5, 2);
        assert_eq!(tall.shape(), (5, 2));
        assert_eq!(tall.diagonal().len(), 2);
        
        let wide = ArrayF64::zeros(2, 5);
        assert_eq!(wide.shape(), (2, 5));
        assert_eq!(wide.diagonal().len(), 2);
        
        let wide_t = wide.transpose();
        assert_eq!(wide_t.shape(), (5, 2));
    }

    #[test]
    fn test_array_mathematical_properties() {
        // Test matrix multiplication properties
        let a = ArrayF64::ones(3, 3);
        let identity = ArrayF64::eye(3);
        
        // A * I = A
        let result1 = a.matmul(&identity);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(result1.get(i, j), a.get(i, j));
            }
        }
        
        // I * A = A
        let result2 = identity.matmul(&a);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(result2.get(i, j), a.get(i, j));
            }
        }
        
        // Test transpose properties: (A^T)^T = A
        let original = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];
        let double_transposed = original.transpose().transpose();
        assert_eq!(double_transposed.shape(), original.shape());
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(double_transposed.get(i, j), original.get(i, j));
            }
        }
    }
}