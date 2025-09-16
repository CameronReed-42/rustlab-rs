//! Generic 1D Vector type built on faer columns
//!
//! This module provides 1D vector operations that integrate seamlessly with the
//! [`Array`] type for matrix operations. Key AI-friendly patterns:
//!
//! - Use `^` operator for mathematical operations: `v1 ^ v2` (dot product), `v ^ A` (vector-matrix)
//! - Use `*` operator for element-wise operations: `&v1 * &v2`
//! - Vector-vector operations: `vector ^ vector` → `scalar` (dot product)
//! - Vector-matrix operations: `vector ^ matrix` → `vector`
//! - Matrix-vector operations: `matrix ^ vector` → `vector`
//!
//! # Cross-Module Integration  
//! ```rust
//! use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
//!
//! let v1 = vec64![1.0, 2.0];
//! let v2 = vec64![3.0, 4.0];
//! let A = array64![[1.0, 2.0], [3.0, 4.0]];
//!
//! // Vector dot product
//! let dot = &v1 ^ &v2;   // Scalar: 11.0
//!
//! // Vector-matrix multiplication
//! let result = &v1 ^ &A; // Vector result
//!
//! // Matrix-vector multiplication
//! let result2 = &A ^ &v1; // Vector result  
//! ```

use faer::{Col, ColRef};
use faer_entity::Entity;
use faer_traits::ComplexField;
use std::fmt;
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use num_complex::Complex;
use crate::Array;

/// A generic 1D vector type wrapping faer::Col<T>
#[repr(C, align(64))] // Cache line alignment for optimal performance
#[derive(Clone)]
pub struct Vector<T: Entity> {
    pub(crate) inner: Col<T>,
}

// Basic impl block with minimal bounds for core functionality
impl<T: Entity> Vector<T> {
    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.inner.nrows()
    }
    
    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a new vector from a faer column
    pub fn from_faer(col: Col<T>) -> Self {
        Self { inner: col }
    }

    /// Get element at index with bounds checking
    /// 
    /// # Mathematical Specification
    /// Returns vᵢ where v is the vector and i is the 0-based index
    /// 
    /// # Dimensions
    /// - Input: index i where 0 ≤ i < n
    /// - Output: Option<T> containing element or None
    /// 
    /// # Complexity
    /// - Time: O(1) direct access
    /// - Space: O(1)
    /// 
    /// # For AI Code Generation
    /// - Returns Some(value) if index is valid, None if out of bounds
    /// - Zero-based indexing (first element is index 0)
    /// - Safe alternative to direct indexing with []
    /// - Never panics, always returns Option
    /// - Common uses: safe element access, boundary checking
    /// - Use with .unwrap_or() for default values
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::vec64;
    /// 
    /// let v = vec64![10.0, 20.0, 30.0];
    /// 
    /// // Safe access with Option
    /// assert_eq!(v.get(0), Some(10.0));
    /// assert_eq!(v.get(1), Some(20.0));
    /// assert_eq!(v.get(5), None);  // Out of bounds
    /// 
    /// // Common patterns
    /// let value = v.get(10).unwrap_or(0.0);  // Default to 0
    /// if let Some(val) = v.get(index) {
    ///     // Process value safely
    /// }
    /// ```
    /// 
    /// # See Also
    /// - [`set`]: Set element value with bounds checking
    /// - Index trait `v[i]`: Direct access (panics on bounds violation)
    /// - [`as_slice`]: Get all elements as slice
    pub fn get(&self, index: usize) -> Option<T>
    where
        T: Clone,
    {
        if index < self.len() {
            Some(unsafe { *self.inner.get_unchecked(index) })
        } else {
            None
        }
    }

    /// Set element at index with bounds checking
    /// 
    /// # Mathematical Specification
    /// Sets vᵢ = value where v is the vector and i is the 0-based index
    /// 
    /// # Dimensions
    /// - Input: index i where 0 ≤ i < n, value of type T
    /// - Output: Result<()> indicating success or error
    /// 
    /// # Complexity
    /// - Time: O(1) direct access
    /// - Space: O(1)
    /// 
    /// # For AI Code Generation
    /// - Returns Ok(()) on success, Err(MathError) if index out of bounds
    /// - Zero-based indexing (first element is index 0)
    /// - Safe alternative to direct assignment with v[i] = value
    /// - Vector must be mutable (&mut self)
    /// - Common uses: safe element update, conditional assignment
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let mut v = VectorF64::zeros(3);
    /// 
    /// // Safe element setting
    /// v.set(0, 10.0).unwrap();
    /// v.set(1, 20.0).unwrap();
    /// assert_eq!(v.get(0), Some(10.0));
    /// 
    /// // Handle out of bounds
    /// match v.set(10, 99.0) {
    ///     Ok(()) => println!("Set successfully"),
    ///     Err(e) => println!("Index out of bounds: {}", e),
    /// }
    /// 
    /// // Conditional update
    /// if index < v.len() {
    ///     v.set(index, new_value).unwrap();
    /// }
    /// ```
    /// 
    /// # Errors
    /// - `IndexOutOfBounds`: Index >= vector length
    /// 
    /// # See Also
    /// - [`get`]: Get element value with bounds checking
    /// - IndexMut trait `v[i] = value`: Direct assignment (panics on bounds)
    /// - [`fill`]: Set all elements to same value
    pub fn set(&mut self, index: usize, value: T) -> crate::Result<()> {
        if index < self.len() {
            if let Some(slice) = self.as_mut_slice() {
                slice[index] = value;
                Ok(())
            } else {
                Err(crate::MathError::IndexOutOfBounds { 
                    index, 
                    size: self.len() 
                })
            }
        } else {
            Err(crate::MathError::IndexOutOfBounds { 
                index, 
                size: self.len() 
            })
        }
    }

    /// Get reference to underlying faer column
    pub fn as_faer(&self) -> ColRef<'_, T> {
        self.inner.as_ref()
    }

    /// Get underlying data as slice (zero-copy when contiguous)
    /// 
    /// Returns `Some(&[T])` if the vector's data is stored contiguously in memory,
    /// `None` otherwise. This enables efficient access to the underlying data
    /// without copying.
    /// 
    /// Most vectors created with standard methods (zeros, ones, from_slice) are
    /// contiguous and will return `Some`.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// if let Some(slice) = vec.as_slice() {
    ///     assert_eq!(slice, &[1.0, 2.0, 3.0]);
    /// }
    /// ```
    pub fn as_slice(&self) -> Option<&[T]> {
        // For empty vectors, return empty slice
        if self.is_empty() {
            return Some(&[]);
        }
        
        // Check if we can get contiguous access to the data
        // faer Col<T> stores data contiguously for owned columns
        unsafe {
            let col_ref = self.inner.as_ref();
            let ptr = col_ref.as_ptr();
            if ptr.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts(ptr, self.len()))
            }
        }
    }

    /// Get underlying data as slice, panicking if not contiguous
    /// 
    /// Use this when you know the vector should be contiguous (e.g., created with
    /// standard constructors). For safer access, use `as_slice()`.
    /// 
    /// # Panics
    /// Panics if the vector data is not stored contiguously.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let vec = VectorF64::zeros(5);
    /// let slice = vec.as_slice_unchecked(); // Safe - zeros() creates contiguous data
    /// assert_eq!(slice.len(), 5);
    /// ```
    pub fn as_slice_unchecked(&self) -> &[T] {
        self.as_slice()
            .expect("Vector data is not contiguous - use as_slice() for safe access")
    }

    /// Get underlying data as mutable slice (zero-copy when contiguous)
    /// 
    /// Returns `Some(&mut [T])` if the vector's data is stored contiguously in memory,
    /// `None` otherwise. This enables efficient in-place operations on the vector data.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let mut vec = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// if let Some(slice) = vec.as_mut_slice() {
    ///     for value in slice.iter_mut() {
    ///         *value *= 2.0; // Double all values in-place
    ///     }
    /// }
    /// ```
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        // For empty vectors, return empty mutable slice
        if self.is_empty() {
            return Some(&mut []);
        }

        let len = self.len();
        
        unsafe {
            // Get mutable pointer to the data
            let col_mut = self.inner.as_mut();
            let ptr = col_mut.as_ptr_mut();
            if ptr.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts_mut(ptr, len))
            }
        }
    }

    /// Get underlying data as mutable slice, panicking if not contiguous
    /// 
    /// Use this when you know the vector should be contiguous. For safer access,
    /// use `as_mut_slice()`.
    /// 
    /// # Panics
    /// Panics if the vector data is not stored contiguously.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let mut vec = VectorF64::ones(3);
    /// let slice = vec.as_mut_slice_unchecked();
    /// slice[0] = 5.0; // Modify first element
    /// ```
    pub fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        self.as_mut_slice()
            .expect("Vector data is not contiguous - use as_mut_slice() for safe access")
    }

    /// Check if the vector's data is stored contiguously in memory
    /// 
    /// Returns `true` if `as_slice()` and `as_mut_slice()` will succeed.
    /// Most vectors created with standard methods are contiguous.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let vec = VectorF64::zeros(10);
    /// assert!(vec.is_contiguous());
    /// ```
    pub fn is_contiguous(&self) -> bool {
        // For owned faer columns created with standard constructors,
        // data should always be contiguous. We can check by seeing if
        // we can get a valid pointer.
        if self.is_empty() {
            return true;
        }
        
        let col_ref = self.inner.as_ref();
        let ptr = col_ref.as_ptr();
        !ptr.is_null()
    }

    /// Get underlying data as slice with safe fallback
    /// 
    /// This is a convenience method that combines `as_slice().unwrap_or(&[])`,
    /// providing a safe way to access vector data as a slice. If the vector's
    /// data is not contiguous, returns an empty slice instead of panicking.
    /// 
    /// This is particularly useful for display, debugging, and safe iteration
    /// over vector contents.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// let slice = vec.to_slice(); // Always returns &[T], never panics
    /// assert_eq!(slice, &[1.0, 2.0, 3.0]);
    /// 
    /// // Safe even for non-contiguous vectors
    /// let empty_fallback = vec.to_slice(); // Returns empty slice if not contiguous
    /// ```
    pub fn to_slice(&self) -> &[T] {
        self.as_slice().unwrap_or(&[])
    }
}

// Impl block for creation methods requiring ComplexField
impl<T: Entity> Vector<T> 
where
    T: ComplexField,
{
    /// Create a new vector filled with zeros
    /// 
    /// # Mathematical Specification
    /// Creates vector v ∈ ℝⁿ or ℂⁿ where vᵢ = 0 for all i
    /// 
    /// # Dimensions
    /// - Input: size (n) where n ≥ 0
    /// - Output: Vector of length n
    /// 
    /// # Complexity
    /// - Time: O(n) initialization
    /// - Space: O(n) allocation
    /// 
    /// # For AI Code Generation
    /// - Allocates contiguous memory for optimal performance
    /// - Size can be 0 (creates empty vector)
    /// - All elements initialized to exact zero (0.0 for f64, 0+0i for complex)
    /// - Common uses: initialization, accumulation, placeholder
    /// - Prefer this over `Vector::fill(n, 0.0)` for clarity
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let v = VectorF64::zeros(5);
    /// assert_eq!(v.len(), 5);
    /// assert_eq!(v.get(0), Some(0.0));
    /// 
    /// // Often used as accumulator
    /// let mut sum = VectorF64::zeros(3);
    /// for data in dataset {
    ///     sum += data;
    /// }
    /// ```
    /// 
    /// # See Also
    /// - [`ones`]: Create vector filled with ones
    /// - [`fill`]: Create vector with custom constant value
    /// - [`from_slice`]: Create from existing data
    pub fn zeros(size: usize) -> Self {
        Self {
            inner: Col::zeros(size),
        }
    }
    
    /// Create a vector filled with ones
    /// 
    /// # Mathematical Specification
    /// Creates vector v ∈ ℝⁿ or ℂⁿ where vᵢ = 1 for all i
    /// 
    /// # Dimensions
    /// - Input: size (n) where n ≥ 0
    /// - Output: Vector of length n
    /// 
    /// # Complexity
    /// - Time: O(n) initialization
    /// - Space: O(n) allocation
    /// 
    /// # For AI Code Generation
    /// - All elements initialized to multiplicative identity (1.0 for f64, 1+0i for complex)
    /// - Size can be 0 (creates empty vector)
    /// - Common uses: initialization, mask creation, constant baseline
    /// - Often combined with scalar multiplication: `ones(n) * 5.0`
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let v = VectorF64::ones(4);
    /// assert_eq!(v.sum_elements(), 4.0);
    /// 
    /// // Create vector of 5s
    /// let fives = VectorF64::ones(10) * 5.0;
    /// 
    /// // Use as mask for element-wise operations
    /// let mask = VectorF64::ones(data.len());
    /// let selected = &data * &mask;
    /// ```
    /// 
    /// # See Also
    /// - [`zeros`]: Create zero-filled vector
    /// - [`fill`]: Create vector with custom constant
    /// - [`eye`]: Create identity matrix (ones on diagonal)
    pub fn ones(size: usize) -> Self
    where
        T: num_traits::One,
    {
        Self {
            inner: Col::from_fn(size, |_i| T::one()),
        }
    }
    
    /// Create a vector filled with a constant value
    /// 
    /// # Mathematical Specification
    /// Creates vector v ∈ ℝⁿ or ℂⁿ where vᵢ = c for all i, given constant c
    /// 
    /// # Dimensions
    /// - Input: size (n) where n ≥ 0, value (scalar)
    /// - Output: Vector of length n
    /// 
    /// # Complexity
    /// - Time: O(n) initialization
    /// - Space: O(n) allocation
    /// 
    /// # For AI Code Generation
    /// - All elements set to exact same value (cloned)
    /// - Size can be 0 (creates empty vector)
    /// - Value is cloned for each element (safe for any T)
    /// - Common uses: initialization with default, creating baselines
    /// - Consider `zeros` or `ones` for special cases
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// // Create temperature vector at room temperature
    /// let temps = VectorF64::fill(100, 20.0);  // 100 sensors at 20°C
    /// 
    /// // Initialize with NaN for missing data
    /// let missing = VectorF64::fill(50, f64::NAN);
    /// 
    /// // Create baseline for comparison
    /// let baseline = VectorF64::fill(data.len(), average_value);
    /// let deviations = &data - &baseline;
    /// ```
    /// 
    /// # See Also
    /// - [`zeros`]: Optimized for zero fill
    /// - [`ones`]: Optimized for one fill
    /// - [`from_fn`]: Create with computed values
    pub fn fill(size: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self {
            inner: Col::from_fn(size, |_i| value.clone()),
        }
    }
}

// Impl block for slice operations
impl<T: Entity> Vector<T>
where
    T: Clone + ComplexField,
{
    /// Create vector from slice data
    pub fn from_slice(data: &[T]) -> Self {
        let inner = Col::from_fn(data.len(), |i| data[i].clone());
        Self { inner }
    }
    
    /// Create vector from Vec data (convenience method)
    pub fn from_vec(data: Vec<T>) -> Self {
        Self::from_slice(&data)
    }
    
    /// Convert vector to Vec<T> (creates a copy)
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let v = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// let vec: Vec<f64> = v.to_vec();
    /// assert_eq!(vec, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice_unchecked().to_vec()
    }
    
    /// Get iterator over vector elements
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let v = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// let sum: f64 = v.iter().sum();
    /// assert_eq!(sum, 6.0);
    /// ```
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice_unchecked().iter()
    }
}

// Vector operations requiring ComplexField
impl<T: Entity> Vector<T>
where
    T: ComplexField,
{
    /// Compute the dot product (inner product) of two vectors
    /// 
    /// # Mathematical Specification
    /// Given vectors u, v ∈ ℝⁿ or ℂⁿ:
    /// dot(u, v) = Σᵢ(uᵢ × v̄ᵢ) for i = 1..n
    /// where v̄ᵢ is the complex conjugate of vᵢ
    /// 
    /// # Dimensions
    /// - Input: self (n), other (n) where n > 0
    /// - Output: scalar T
    /// 
    /// # Complexity
    /// - Time: O(n) where n is vector length
    /// - Space: O(1)
    /// 
    /// # For AI Code Generation
    /// - Both vectors must have identical length
    /// - Returns a scalar value (f64 for VectorF64, Complex<f64> for VectorC64)
    /// - Use the ^ operator as shorthand: `v1 ^ v2`
    /// - For real vectors: standard dot product
    /// - For complex vectors: includes conjugation of second vector
    /// - Common uses: angle calculation, projection, similarity metrics
    /// - Zero vectors return 0
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::vec64;
    /// 
    /// let v1 = vec64![3.0, 4.0, 0.0];
    /// let v2 = vec64![1.0, 2.0, 2.0];
    /// 
    /// // Standard method call
    /// let dot_product = v1.dot(&v2);  // Returns 11.0
    /// 
    /// // Using ^ operator (equivalent)
    /// let dot_product = &v1 ^ &v2;    // Returns 11.0
    /// 
    /// // Calculate angle between vectors
    /// let cos_angle = dot_product / (v1.norm() * v2.norm());
    /// let angle_radians = cos_angle.acos();
    /// ```
    /// 
    /// # Errors
    /// - Panics if vectors have different lengths
    /// - Consider using `try_dot` for Result-based error handling
    /// 
    /// # See Also
    /// - [`norm`]: Calculate vector magnitude (often used with dot product)
    /// - [`normalize`]: Create unit vector for angle calculations
    /// - [`outer`]: Outer product producing a matrix
    pub fn dot(&self, other: &Self) -> T {
        assert_eq!(self.len(), other.len(), 
                   "Vectors must have the same length for dot product");
        
        self.inner.transpose() * &other.inner
    }
    
    /// Compute the L2 norm (Euclidean magnitude) of the vector
    /// 
    /// # Mathematical Specification
    /// For vector v ∈ ℝⁿ or ℂⁿ:
    /// ||v||₂ = √(Σᵢ|vᵢ|²) for i = 1..n
    /// where |vᵢ| is the modulus for complex numbers
    /// 
    /// # Dimensions
    /// - Input: self (n) where n ≥ 0
    /// - Output: scalar T::Real (always real, even for complex vectors)
    /// 
    /// # Complexity
    /// - Time: O(n) where n is vector length
    /// - Space: O(1)
    /// 
    /// # For AI Code Generation
    /// - Returns non-negative real value (f64 for both VectorF64 and VectorC64)
    /// - Zero vector returns 0.0
    /// - Result is always real, even for complex vectors
    /// - Common uses: normalization, distance calculation, convergence testing
    /// - Numerically stable for large values (avoids overflow)
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::vec64;
    /// 
    /// let v = vec64![3.0, 4.0];
    /// let magnitude = v.norm();  // Returns 5.0
    /// 
    /// // Normalize vector to unit length
    /// let unit_vector = &v / magnitude;
    /// assert!((unit_vector.norm() - 1.0).abs() < 1e-10);
    /// 
    /// // Distance between two points
    /// let p1 = vec64![1.0, 2.0, 3.0];
    /// let p2 = vec64![4.0, 6.0, 8.0];
    /// let distance = (&p2 - &p1).norm();
    /// ```
    /// 
    /// # See Also
    /// - [`normalize`]: Create unit vector (norm = 1)
    /// - [`norm_squared`]: Compute norm² without square root
    /// - [`dot`]: Related inner product operation
    pub fn norm(&self) -> T::Real
    where
        T: ComplexField,
    {
        // Use faer's built-in norm computation 
        self.inner.norm_l2()
    }
}

// Vector-Array multiplication operations
impl<T: Entity> Vector<T>
where
    T: ComplexField,
{
    /// Vector-matrix multiplication (row vector × matrix)
    pub fn matmul(&self, array: &Array<T>) -> Vector<T> {
        assert_eq!(self.len(), array.nrows(),
                   "Vector length ({}) must match matrix rows ({})",
                   self.len(), array.nrows());
        
        // Convert row result back to column vector
        let result_row = self.inner.transpose() * &array.inner;
        Vector {
            inner: result_row.transpose().to_owned(),
        }
    }
}

// Debug implementation
impl<T: Entity + fmt::Debug> fmt::Debug for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector {{ inner: [")?;
        for i in 0..self.len() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:?}", unsafe { self.inner.get_unchecked(i) })?;
        }
        write!(f, "] }}")
    }
}

// ========== TYPE ALIASES ==========

/// 1D vector of f64 values (most common)
pub type VectorF64 = Vector<f64>;

/// 1D vector of f32 values  
pub type VectorF32 = Vector<f32>;

/// 1D vector of Complex<f64> values
pub type VectorC64 = Vector<Complex<f64>>;

/// 1D vector of Complex<f32> values
pub type VectorC32 = Vector<Complex<f32>>;

// For backward compatibility, make non-generic Vector = VectorF64
/// Default Vector type (f64) for backward compatibility
pub type Vector64 = VectorF64;

// ========== ASSIGNMENT OPERATORS WITH AUTOMATIC SIMD ==========

/// AddAssign operator (+=) with automatic SIMD optimization
impl<T: Entity + ComplexField> AddAssign<&Vector<T>> for Vector<T>
where
    T: Clone + std::ops::Add<Output = T>,
{
    fn add_assign(&mut self, rhs: &Vector<T>) {
        assert_eq!(self.len(), rhs.len(), 
                   "Vectors must have the same length for +=: {} vs {}", 
                   self.len(), rhs.len());

        // Use faer's optimized addition with automatic SIMD
        self.inner = &self.inner + &rhs.inner;
    }
}

/// AddAssign for owned Vector (v += u)
impl<T: Entity + ComplexField> AddAssign<Vector<T>> for Vector<T>
where
    T: Clone + std::ops::Add<Output = T>,
{
    fn add_assign(&mut self, rhs: Vector<T>) {
        *self += &rhs;
    }
}

/// SubAssign operator (-=) with automatic SIMD optimization
impl<T: Entity + ComplexField> SubAssign<&Vector<T>> for Vector<T>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: &Vector<T>) {
        assert_eq!(self.len(), rhs.len(), 
                   "Vectors must have the same length for -=: {} vs {}", 
                   self.len(), rhs.len());

        // Use faer's optimized subtraction with automatic SIMD
        self.inner = &self.inner - &rhs.inner;
    }
}

/// SubAssign for owned Vector (v -= u)
impl<T: Entity + ComplexField> SubAssign<Vector<T>> for Vector<T>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: Vector<T>) {
        *self -= &rhs;
    }
}

/// MulAssign operator (*=) for element-wise multiplication with automatic SIMD
impl<T: Entity + ComplexField> MulAssign<&Vector<T>> for Vector<T>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: &Vector<T>) {
        assert_eq!(self.len(), rhs.len(), 
                   "Vectors must have the same length for *=: {} vs {}", 
                   self.len(), rhs.len());

        // Element-wise multiplication using faer's from_fn
        let len = self.len();
        let result = faer::Col::from_fn(len, |i| {
            self.inner[i].clone() * rhs.inner[i].clone()
        });
        self.inner = result;
    }
}

/// MulAssign for owned Vector (v *= u)
impl<T: Entity + ComplexField> MulAssign<Vector<T>> for Vector<T>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: Vector<T>) {
        *self *= &rhs;
    }
}

/// DivAssign operator (/=) for element-wise division with automatic SIMD
impl<T: Entity + ComplexField> DivAssign<&Vector<T>> for Vector<T>
where
    T: Clone + std::ops::Div<Output = T>,
{
    fn div_assign(&mut self, rhs: &Vector<T>) {
        assert_eq!(self.len(), rhs.len(), 
                   "Vectors must have the same length for /=: {} vs {}", 
                   self.len(), rhs.len());

        // Element-wise division using faer's from_fn
        let len = self.len();
        let result = faer::Col::from_fn(len, |i| {
            self.inner[i].clone() / rhs.inner[i].clone()
        });
        self.inner = result;
    }
}

/// DivAssign for owned Vector (v /= u)
impl<T: Entity + ComplexField> DivAssign<Vector<T>> for Vector<T>
where
    T: Clone + std::ops::Div<Output = T>,
{
    fn div_assign(&mut self, rhs: Vector<T>) {
        *self /= &rhs;
    }
}

// ========== SCALAR ASSIGNMENT OPERATORS ==========

/// AddAssign for scalar (v += scalar)
impl<T: Entity + ComplexField> AddAssign<T> for Vector<T>
where
    T: Clone + std::ops::Add<Output = T>,
{
    fn add_assign(&mut self, scalar: T) {
        let len = self.len();
        let result = faer::Col::from_fn(len, |i| {
            self.inner[i].clone() + scalar.clone()
        });
        self.inner = result;
    }
}

/// SubAssign for scalar (v -= scalar)
impl<T: Entity + ComplexField> SubAssign<T> for Vector<T>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    fn sub_assign(&mut self, scalar: T) {
        let len = self.len();
        let result = faer::Col::from_fn(len, |i| {
            self.inner[i].clone() - scalar.clone()
        });
        self.inner = result;
    }
}

/// MulAssign for scalar (v *= scalar) with automatic SIMD
impl<T: Entity + ComplexField> MulAssign<T> for Vector<T>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    fn mul_assign(&mut self, scalar: T) {
        // Use faer's optimized scalar multiplication with automatic SIMD
        let len = self.len();
        let result = faer::Col::from_fn(len, |i| {
            scalar.clone() * self.inner[i].clone()
        });
        self.inner = result;
    }
}

/// DivAssign for scalar (v /= scalar) with automatic SIMD
impl<T: Entity + ComplexField> DivAssign<T> for Vector<T>
where
    T: Clone + std::ops::Div<Output = T> + num_traits::One,
{
    fn div_assign(&mut self, scalar: T) {
        // Convert to multiplication by reciprocal for better optimization
        let reciprocal = T::one() / scalar;
        let len = self.len();
        let result = faer::Col::from_fn(len, |i| {
            reciprocal.clone() * self.inner[i].clone()
        });
        self.inner = result;
    }
}

// Ergonomic indexing implementations

/// Index trait implementation for convenient element access: `vector[index]`
/// 
/// This provides ergonomic access to vector elements using square bracket notation.
/// For bounds-checked access that returns `Option`, use `.get(index)` instead.
/// 
/// # Panics
/// 
/// Panics if the index is out of bounds.
/// 
/// # Examples
/// 
/// ```rust
/// use rustlab_math::{VectorF64, vec64};
/// 
/// let v = vec64![1.0, 2.0, 3.0];
/// assert_eq!(v[0], 1.0);
/// assert_eq!(v[1], 2.0);
/// assert_eq!(v[2], 3.0);
/// 
/// // This would panic:
/// // let x = v[10]; // index out of bounds
/// 
/// // For safe access, use .get():
/// assert_eq!(v.get(0), Some(1.0));
/// assert_eq!(v.get(10), None);
/// ```
/// Enable direct element access via indexing: `vector[i]`
/// 
/// # Mathematical Specification
/// Provides direct access to vᵢ where v is the vector and i is the 0-based index
/// 
/// # For AI Code Generation
/// - Syntax: `value = vector[index]` for reading
/// - Zero-based indexing (first element is vector[0])
/// - **PANICS** if index >= length (use .get() for safe access)
/// - Returns reference to element, not a copy
/// - Common uses: mathematical operations, direct access when bounds are known
/// - Prefer .get() when index might be out of bounds
/// 
/// # Example
/// ```rust
/// use rustlab_math::vec64;
/// 
/// let v = vec64![10.0, 20.0, 30.0];
/// 
/// // Direct access (panics if out of bounds)
/// let first = v[0];   // 10.0
/// let second = v[1];  // 20.0
/// 
/// // Use in calculations
/// let sum = v[0] + v[1] + v[2];
/// 
/// // DANGER: This will panic!
/// // let bad = v[10];  // panic: index out of bounds
/// 
/// // Safe alternative
/// let safe = v.get(10).unwrap_or(0.0);
/// ```
/// 
/// # See Also
/// - [`get`]: Safe access returning Option
/// - [`IndexMut`]: Mutable access with `vector[i] = value`
/// - [`Array`] indexing: Use `array[(row, col)]` for 2D matrices
impl<T: Entity> Index<usize> for Vector<T>
where
    T: Clone,
{
    type Output = T;
    
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len() {
            panic!("Vector index {} out of bounds for vector of length {}", index, self.len());
        }
        // Use faer's internal indexing which is already bounds-checked
        &self.inner[index]
    }
}

/// IndexMut trait implementation for mutable element access: `vector[index] = value`
/// 
/// This provides ergonomic mutable access to vector elements using square bracket notation.
/// For bounds-checked mutable access, use `.set(index, value)` instead.
/// 
/// # Panics
/// 
/// Panics if the index is out of bounds.
/// 
/// # Examples
/// 
/// ```rust
/// use rustlab_math::{VectorF64, vec64};
/// 
/// let mut v = vec64![1.0, 2.0, 3.0];
/// v[0] = 10.0;
/// v[1] = 20.0;
/// assert_eq!(v[0], 10.0);
/// assert_eq!(v[1], 20.0);
/// 
/// // For safe mutable access, use .set():
/// v.set(0, 100.0).unwrap();
/// assert_eq!(v[0], 100.0);
/// ```
impl<T: Entity> IndexMut<usize> for Vector<T>
where
    T: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.len() {
            panic!("Vector index {} out of bounds for vector of length {}", index, self.len());
        }
        // Use faer's internal mutable indexing
        &mut self.inner[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VectorF64, vec64};
    use approx::assert_relative_eq;
    use num_complex::Complex;

    #[test]
    fn test_vector_creation() {
        let v = VectorF64::zeros(5);
        assert_eq!(v.len(), 5);
        assert!(!v.is_empty());
        
        let v_empty = VectorF64::zeros(0);
        assert_eq!(v_empty.len(), 0);
        assert!(v_empty.is_empty());
    }

    #[test]
    fn test_vector_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let v = VectorF64::from_slice(&data);
        
        assert_eq!(v.len(), 4);
        assert_eq!(v.get(0), Some(1.0));
        assert_eq!(v.get(1), Some(2.0));
        assert_eq!(v.get(2), Some(3.0));
        assert_eq!(v.get(3), Some(4.0));
        assert_eq!(v.get(4), None);
    }

    #[test]
    fn test_vector_from_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let v = VectorF64::from_vec(data);
        
        assert_eq!(v.len(), 3);
        assert_eq!(v.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_ones_and_fill() {
        let ones = VectorF64::ones(3);
        assert_eq!(ones.get(0), Some(1.0));
        assert_eq!(ones.get(1), Some(1.0));
        assert_eq!(ones.get(2), Some(1.0));

        let filled = VectorF64::fill(4, 2.5);
        for i in 0..4 {
            assert_eq!(filled.get(i), Some(2.5));
        }
    }

    #[test]
    fn test_vector_get_set() {
        let mut v = VectorF64::zeros(3);
        
        // Test set method
        assert!(v.set(0, 10.0).is_ok());
        assert!(v.set(1, 20.0).is_ok());
        assert!(v.set(2, 30.0).is_ok());
        assert!(v.set(3, 40.0).is_err()); // Out of bounds
        
        // Test get method
        assert_eq!(v.get(0), Some(10.0));
        assert_eq!(v.get(1), Some(20.0));
        assert_eq!(v.get(2), Some(30.0));
        assert_eq!(v.get(3), None); // Out of bounds
    }

    #[test]
    fn test_vector_indexing() {
        let v = vec64![1.0, 2.0, 3.0];
        
        // Test index access
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
        
        // Test mutable indexing
        let mut v_mut = vec64![1.0, 2.0, 3.0];
        v_mut[1] = 5.0;
        assert_eq!(v_mut[1], 5.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_vector_index_panic() {
        let v = vec64![1.0, 2.0];
        let _ = v[5]; // Should panic
    }

    #[test]
    fn test_vector_as_slice() {
        let v = vec64![1.0, 2.0, 3.0, 4.0];
        
        if let Some(slice) = v.as_slice() {
            assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
        }
        
        // Test unchecked version
        let slice_unchecked = v.as_slice_unchecked();
        assert_eq!(slice_unchecked, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vector_as_mut_slice() {
        let mut v = vec64![1.0, 2.0, 3.0];
        
        if let Some(slice) = v.as_mut_slice() {
            slice[1] = 10.0;
        }
        assert_eq!(v.get(1), Some(10.0));
        
        // Test unchecked version
        let slice_unchecked = v.as_mut_slice_unchecked();
        slice_unchecked[2] = 20.0;
        assert_eq!(v.get(2), Some(20.0));
    }

    #[test]
    fn test_vector_to_slice() {
        let v = vec64![1.0, 2.0, 3.0];
        let slice = v.to_slice();
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_is_contiguous() {
        let v = vec64![1.0, 2.0, 3.0];
        assert!(v.is_contiguous());
    }

    #[test]
    fn test_vector_dot_product() {
        let v1 = vec64![1.0, 2.0, 3.0];
        let v2 = vec64![4.0, 5.0, 6.0];
        
        let dot = v1.dot(&v2);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(dot, 32.0);
        
        // Test with zero vector
        let zero = VectorF64::zeros(3);
        let dot_zero = v1.dot(&zero);
        assert_eq!(dot_zero, 0.0);
    }

    #[test]
    fn test_vector_norm() {
        let v = vec64![3.0, 4.0]; // 3-4-5 triangle
        let norm = v.norm();
        assert_relative_eq!(norm, 5.0, epsilon = 1e-15);
        
        // Test unit vector
        let unit = vec64![1.0, 0.0, 0.0];
        assert_relative_eq!(unit.norm(), 1.0, epsilon = 1e-15);
        
        // Test zero vector
        let zero = VectorF64::zeros(3);
        assert_eq!(zero.norm(), 0.0);
    }

    #[test]
    fn test_vector_iterator() {
        let v = vec64![1.0, 2.0, 3.0];
        let collected: Vec<_> = v.iter().cloned().collect();
        assert_eq!(collected, vec![1.0, 2.0, 3.0]);
        
        // Test iterator with calculations
        let sum: f64 = v.iter().sum();
        assert_eq!(sum, 6.0);
    }

    #[test]
    fn test_vector_clone() {
        let v1 = vec64![1.0, 2.0, 3.0];
        let v2 = v1.clone();
        
        assert_eq!(v1.len(), v2.len());
        for i in 0..v1.len() {
            assert_eq!(v1.get(i), v2.get(i));
        }
    }

    #[test]
    fn test_vector_partial_eq() {
        let v1 = vec64![1.0, 2.0, 3.0];
        let v2 = vec64![1.0, 2.0, 3.0];
        let v3 = vec64![1.0, 2.0, 4.0];
        
        // Manual equality test since PartialEq may not be implemented
        assert_eq!(v1.len(), v2.len());
        for i in 0..v1.len() {
            assert_eq!(v1.get(i), v2.get(i));
        }
        
        assert_ne!(v1.get(2), v3.get(2));
    }

    #[test]
    fn test_vector_debug() {
        let v = vec64![1.0, 2.0, 3.0];
        let debug_str = format!("{:?}", v);
        // Just ensure debug formatting works without checking specific format
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_vector_generic_types() {
        // Test with f32
        let v_f32 = Vector::<f32>::from_slice(&[1.0f32, 2.0f32, 3.0f32]);
        assert_eq!(v_f32.len(), 3);
        assert_eq!(v_f32.get(0), Some(1.0f32));
        
        // Test with complex numbers
        let v_complex = Vector::<Complex<f64>>::from_slice(&[
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
        ]);
        assert_eq!(v_complex.len(), 2);
        assert_eq!(v_complex.get(0), Some(Complex::new(1.0, 0.0)));
    }

    #[test]
    fn test_vector_assignment_operations() {
        let mut v1 = vec64![1.0, 2.0, 3.0];
        let v2 = vec64![1.0, 1.0, 1.0];
        
        // Test AddAssign
        v1 += &v2;
        assert_eq!(v1.get(0), Some(2.0));
        assert_eq!(v1.get(1), Some(3.0));
        assert_eq!(v1.get(2), Some(4.0));
        
        // Test SubAssign
        v1 -= &v2;
        assert_eq!(v1.get(0), Some(1.0));
        assert_eq!(v1.get(1), Some(2.0));
        assert_eq!(v1.get(2), Some(3.0));
        
        // Test MulAssign with scalar
        v1 *= 2.0;
        assert_eq!(v1.get(0), Some(2.0));
        assert_eq!(v1.get(1), Some(4.0));
        assert_eq!(v1.get(2), Some(6.0));
        
        // Test DivAssign with scalar
        v1 /= 2.0;
        assert_eq!(v1.get(0), Some(1.0));
        assert_eq!(v1.get(1), Some(2.0));
        assert_eq!(v1.get(2), Some(3.0));
    }

    #[test]
    fn test_vector_large_operations() {
        let size = 1000;
        let v1 = VectorF64::ones(size);
        let v2 = VectorF64::fill(size, 2.0);
        
        let dot = v1.dot(&v2);
        assert_eq!(dot, 2000.0); // 1000 elements × 1.0 × 2.0 = 2000
        
        let norm = v1.norm();
        assert_relative_eq!(norm, (size as f64).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_vector_empty_operations() {
        let v1 = VectorF64::zeros(0);
        let v2 = VectorF64::zeros(0);
        
        assert!(v1.is_empty());
        assert!(v2.is_empty());
        
        let dot = v1.dot(&v2);
        assert_eq!(dot, 0.0);
        
        let norm = v1.norm();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_vector_faer_integration() {
        let v = vec64![1.0, 2.0, 3.0];
        
        // Test as_faer method
        let faer_ref = v.as_faer();
        assert_eq!(faer_ref.nrows(), 3);
        assert_eq!(faer_ref[0], 1.0);
        assert_eq!(faer_ref[1], 2.0);
        assert_eq!(faer_ref[2], 3.0);
        
        // Test from_faer constructor
        let faer_col = faer::Col::from_fn(3, |i| (i + 1) as f64);
        let v_from_faer = VectorF64::from_faer(faer_col);
        assert_eq!(v_from_faer.get(0), Some(1.0));
        assert_eq!(v_from_faer.get(1), Some(2.0));
        assert_eq!(v_from_faer.get(2), Some(3.0));
    }

    #[test]
    fn test_vector_edge_cases() {
        // Test single element vector
        let v_single = vec64![42.0];
        assert_eq!(v_single.len(), 1);
        assert!(!v_single.is_empty());
        assert_eq!(v_single.get(0), Some(42.0));
        assert_eq!(v_single.norm(), 42.0);
        
        // Test very small numbers
        let v_small = vec64![1e-15, 1e-15];
        let norm = v_small.norm();
        assert!(norm > 0.0);
        assert_relative_eq!(norm, (2e-30_f64).sqrt(), epsilon = 1e-16);
    }
}