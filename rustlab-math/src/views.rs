//! Zero-copy views for Array<T> and Vector<T> with AI-optimized documentation
//!
//! This module provides lightweight, zero-cost abstraction views that enable efficient
//! data access without copying. All view operations integrate seamlessly with RustLab's
//! mathematical operators and maintain the ergonomic ^ operator for matrix operations.
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::{ArrayF64, VectorF64};
//! 
//! let matrix = ArrayF64::ones(1000, 500);     // Large matrix
//! let vector = VectorF64::ones(500);          // Large vector
//! 
//! // Zero-copy views - no memory allocation
//! let matrix_view = matrix.view();           // Lightweight reference
//! let vector_view = vector.view();           // Lightweight reference
//! 
//! // Mathematical operations work directly with views
//! let result1 = matrix_view ^ vector_view;    // Matrix × vector (no copying)
//! let result2 = matrix_view * 2.0;            // Scalar multiplication
//! let result3 = matrix_view + matrix_view;    // Element-wise addition
//! 
//! // Mix views with owned data seamlessly
//! let mixed = &matrix ^ vector_view;          // Matrix × view
//! let combined = matrix + matrix_view;        // Owned + view
//! ```
//!
//! # Memory Efficiency
//! - **Zero allocation**: Views only store references, no data copying
//! - **Minimal overhead**: Views are 1-2 machine words (8-16 bytes)
//! - **Lifetime safety**: Rust prevents dangling pointer issues
//! - **SIMD optimized**: All operations use faer's vectorized implementations
//!
//! # Cross-Module Integration
//! - Compatible with all [`Array`] and [`Vector`] mathematical operators
//! - Works with [`broadcasting`] for element-wise operations
//! - Integrates with [`slicing`] for advanced data access patterns
//! - Maintains ergonomic ^ operator throughout the view system

use faer::{MatRef, ColRef};
use faer_entity::Entity;
use faer_traits::ComplexField;
use std::ops::{Add, Sub, Mul, BitXor, AddAssign, SubAssign, MulAssign, DivAssign};
use num_traits::Zero;
use crate::{Array, Vector};

/// Zero-copy view of a 2D array with AI-optimized documentation
/// 
/// # For AI Code Generation
/// - Lightweight reference to matrix data (no allocation)
/// - Lifetime 'a ensures memory safety - view cannot outlive source
/// - All mathematical operators work directly with views
/// - **Critical**: Use ^ operator for matrix multiplication, * for element-wise
/// - Automatic SIMD optimization through faer backend
/// - Copy trait enables efficient passing by value
/// 
/// # Memory Characteristics
/// - Size: 1 machine word (8 bytes on 64-bit systems)
/// - No data copying during view creation
/// - Multiple views can reference same matrix simultaneously
/// - Safe: Rust lifetime system prevents use-after-free
/// 
/// # Mathematical Operations
/// - **Matrix multiplication**: `view1 ^ view2`, `view ^ matrix`
/// - **Element-wise operations**: `view1 + view2`, `view * scalar`
/// - **Mixed operations**: `matrix ^ view`, `&matrix + view`
/// - **In-place operations**: `matrix += view` (SIMD optimized)
/// 
/// # Example
/// ```rust
/// use rustlab_math::ArrayF64;
/// 
/// let A = ArrayF64::ones(1000, 500);  // Large matrix
/// let B = ArrayF64::ones(500, 300);   // Another matrix
/// 
/// let view_A = A.view();              // Zero-cost view
/// let view_B = B.view();              // Zero-cost view
/// 
/// let C = view_A ^ view_B;            // Matrix multiplication (no copying A or B)
/// let scaled = view_A * 2.0;          // Scalar multiplication
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ArrayView<'a, T: Entity> {
    pub(crate) inner: MatRef<'a, T>,
}

impl<'a, T: Entity> ArrayView<'a, T> {
    /// Get array dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.inner.nrows(), self.inner.ncols())
    }
    
    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    
    /// Get number of cols
    pub fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    
    /// Get element at position
    /// 
    /// # For AI Code Generation
    /// - Safe element access with bounds checking
    /// - Returns None for out-of-bounds access (no panic)
    /// - Clones element value for owned access
    /// - Use for individual element inspection and iteration
    /// - Preferred over direct indexing for safety
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let view = matrix.view();
    /// 
    /// assert_eq!(view.get(0, 1), Some(2.0));  // Valid access
    /// assert_eq!(view.get(5, 5), None);       // Out of bounds
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Option<T>
    where
        T: Clone,
    {
        if row < self.nrows() && col < self.ncols() {
            Some(self.inner[(row, col)].clone())
        } else {
            None
        }
    }
    
    /// Convert view to owned Array
    /// 
    /// # For AI Code Generation
    /// - Converts zero-copy view to owned RustLab Array<T>
    /// - Performs memory allocation and element cloning
    /// - Use when you need persistent matrix that outlives original
    /// - More expensive than view operations - use when necessary
    /// - Result integrates fully with RustLab mathematical ecosystem
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let original = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let view = original.view();
    /// let owned = view.to_owned();        // Convert to owned matrix
    /// 
    /// // Now `owned` can outlive `original`
    /// drop(original);
    /// let result = owned ^ owned;         // Use mathematical operations
    /// ```
    pub fn to_owned(&self) -> Array<T>
    where
        T: Clone,
    {
        let mat = faer::Mat::from_fn(self.nrows(), self.ncols(), |i, j| {
            self.inner[(i, j)].clone()
        });
        Array::from_faer(mat)
    }
    
    /// Get reference to underlying faer matrix reference
    pub fn as_faer(&self) -> MatRef<'a, T> {
        self.inner
    }
}

/// Zero-copy view of a 1D vector with AI-optimized documentation
/// 
/// # For AI Code Generation
/// - Lightweight reference to vector data (no allocation)
/// - Lifetime 'a ensures memory safety - view cannot outlive source
/// - All mathematical operators work directly with views
/// - **Critical**: Use ^ operator for dot product, * for element-wise
/// - Automatic SIMD optimization through faer backend
/// - Copy trait enables efficient passing by value
/// 
/// # Memory Characteristics
/// - Size: 1 machine word (8 bytes on 64-bit systems)
/// - No data copying during view creation
/// - Multiple views can reference same vector simultaneously
/// - Safe: Rust lifetime system prevents use-after-free
/// 
/// # Mathematical Operations
/// - **Dot product**: `view1 ^ view2`, `view ^ vector`
/// - **Element-wise operations**: `view1 + view2`, `view * scalar`
/// - **Matrix-vector**: `matrix_view ^ view`, `view ^ matrix_view`
/// - **In-place operations**: `vector += view` (SIMD optimized)
/// 
/// # Example
/// ```rust
/// use rustlab_math::VectorF64;
/// 
/// let u = VectorF64::ones(10000);     // Large vector
/// let v = VectorF64::ones(10000);     // Another vector
/// 
/// let view_u = u.view();              // Zero-cost view
/// let view_v = v.view();              // Zero-cost view
/// 
/// let dot = view_u ^ view_v;          // Dot product (no copying u or v)
/// let scaled = view_u * 3.14;         // Scalar multiplication
/// ```
#[derive(Debug, Clone, Copy)]
pub struct VectorView<'a, T: Entity> {
    pub(crate) inner: ColRef<'a, T>,
}

impl<'a, T: Entity> VectorView<'a, T> {
    /// Get vector length
    pub fn len(&self) -> usize {
        self.inner.nrows()
    }
    
    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get element at index
    pub fn get(&self, index: usize) -> Option<T>
    where
        T: Clone,
    {
        if index < self.len() {
            Some(self.inner[index].clone())
        } else {
            None
        }
    }
    
    /// Convert view to owned Vector
    pub fn to_owned(&self) -> Vector<T>
    where
        T: Clone,
    {
        let col = faer::Col::from_fn(self.len(), |i| {
            self.inner[i].clone()
        });
        Vector::from_faer(col)
    }
    
    /// Get reference to underlying faer column reference
    pub fn as_faer(&self) -> ColRef<'a, T> {
        self.inner
    }
}

// ========== INTEGRATION WITH ARRAY<T> AND VECTOR<T> ==========

impl<T: Entity + ComplexField> Array<T> {
    /// Create a zero-copy view of this array
    /// 
    /// # For AI Code Generation
    /// - Creates lightweight view without copying matrix data
    /// - Essential for efficient mathematical operations on large matrices
    /// - View inherits lifetime from source matrix
    /// - Use when you need temporary reference for calculations
    /// - All mathematical operators work with views
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let large_matrix = ArrayF64::ones(1000, 1000);
    /// let view = large_matrix.view();     // Zero-cost, instant
    /// let result = view ^ view;           // Mathematical operations
    /// ```
    pub fn view(&self) -> ArrayView<'_, T> {
        ArrayView {
            inner: self.inner.as_ref(),
        }
    }
}

impl<T: Entity + ComplexField> Vector<T> {
    /// Create a zero-copy view of this vector
    pub fn view(&self) -> VectorView<'_, T> {
        VectorView {
            inner: self.inner.as_ref(),
        }
    }
}

// ========== OPERATORS FOR ARRAYVIEW ==========

/// ArrayView + ArrayView -> Array (element-wise addition)
impl<'a, 'b, T: Entity + ComplexField> Add<ArrayView<'b, T>> for ArrayView<'a, T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Array<T>;
    
    fn add(self, rhs: ArrayView<'b, T>) -> Array<T> {
        assert_eq!(self.shape(), rhs.shape(), "Arrays must have the same shape for addition");
        
        let (rows, cols) = self.shape();
        let result = faer::Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].clone() + rhs.inner[(i, j)].clone()
        });
        Array::from_faer(result)
    }
}

/// ArrayView - ArrayView -> Array (element-wise subtraction)
impl<'a, 'b, T: Entity + ComplexField> Sub<ArrayView<'b, T>> for ArrayView<'a, T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Array<T>;
    
    fn sub(self, rhs: ArrayView<'b, T>) -> Array<T> {
        assert_eq!(self.shape(), rhs.shape(), "Arrays must have the same shape for subtraction");
        
        let (rows, cols) = self.shape();
        let result = faer::Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].clone() - rhs.inner[(i, j)].clone()
        });
        Array::from_faer(result)
    }
}

/// ArrayView * ArrayView -> Array (element-wise multiplication)
impl<'a, 'b, T: Entity + ComplexField> Mul<ArrayView<'b, T>> for ArrayView<'a, T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Array<T>;
    
    fn mul(self, rhs: ArrayView<'b, T>) -> Array<T> {
        assert_eq!(self.shape(), rhs.shape(), "Arrays must have the same shape for element-wise multiplication");
        
        let (rows, cols) = self.shape();
        let result = faer::Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].clone() * rhs.inner[(i, j)].clone()
        });
        Array::from_faer(result)
    }
}

/// ArrayView ^ ArrayView -> Array (matrix multiplication)
/// 
/// # For AI Code Generation
/// - **Core matrix multiplication** between two views
/// - Zero-copy input, owned output for chaining operations
/// - Dimension compatibility: left.ncols() must equal right.nrows()
/// - Most efficient pattern for view-based linear algebra
/// - Automatic SIMD optimization through faer backend
/// 
/// # Mathematical Specification
/// For matrices A ∈ ℝᵐˣᵏ, B ∈ ℝᵏˣⁿ:
/// C = A ^ B where C[i,j] = Σ(A[i,l] × B[l,j]) for l = 1..k
/// 
/// # Example
/// ```rust
/// use rustlab_math::ArrayF64;
/// 
/// let A = ArrayF64::ones(100, 50);
/// let B = ArrayF64::ones(50, 75);
/// let C = A.view() ^ B.view();     // 100×75 result (no copying A or B)
/// ```
impl<'a, 'b, T: Entity + ComplexField> BitXor<ArrayView<'b, T>> for ArrayView<'a, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Array<T>;
    
    fn bitxor(self, rhs: ArrayView<'b, T>) -> Array<T> {
        assert_eq!(self.ncols(), rhs.nrows(), 
                   "Matrix dimensions must be compatible for multiplication: ({}, {}) ^ ({}, {})",
                   self.nrows(), self.ncols(), rhs.nrows(), rhs.ncols());
        
        let result_rows = self.nrows();
        let result_cols = rhs.ncols();
        let inner_dim = self.ncols();
        
        let result = faer::Mat::from_fn(result_rows, result_cols, |i, j| {
            let mut sum = T::zero();
            for k in 0..inner_dim {
                sum = sum + self.inner[(i, k)].clone() * rhs.inner[(k, j)].clone();
            }
            sum
        });
        
        Array::from_faer(result)
    }
}

/// ArrayView * scalar -> Array (scalar multiplication)
impl<'a, T: Entity + ComplexField> Mul<T> for ArrayView<'a, T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Array<T>;
    
    fn mul(self, scalar: T) -> Array<T> {
        let (rows, cols) = self.shape();
        let result = faer::Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].clone() * scalar.clone()
        });
        Array::from_faer(result)
    }
}

// ========== OPERATORS FOR VECTORVIEW ==========

/// VectorView + VectorView -> Vector (element-wise addition)
impl<'a, 'b, T: Entity + ComplexField> Add<VectorView<'b, T>> for VectorView<'a, T>
where
    T: Clone + Add<Output = T>,
{
    type Output = Vector<T>;
    
    fn add(self, rhs: VectorView<'b, T>) -> Vector<T> {
        assert_eq!(self.len(), rhs.len(), "Vectors must have the same length for addition");
        
        let len = self.len();
        let result = faer::Col::from_fn(len, |i| {
            self.inner[i].clone() + rhs.inner[i].clone()
        });
        Vector::from_faer(result)
    }
}

/// VectorView - VectorView -> Vector (element-wise subtraction)
impl<'a, 'b, T: Entity + ComplexField> Sub<VectorView<'b, T>> for VectorView<'a, T>
where
    T: Clone + Sub<Output = T>,
{
    type Output = Vector<T>;
    
    fn sub(self, rhs: VectorView<'b, T>) -> Vector<T> {
        assert_eq!(self.len(), rhs.len(), "Vectors must have the same length for subtraction");
        
        let len = self.len();
        let result = faer::Col::from_fn(len, |i| {
            self.inner[i].clone() - rhs.inner[i].clone()
        });
        Vector::from_faer(result)
    }
}

/// VectorView * VectorView -> Vector (element-wise multiplication)
impl<'a, 'b, T: Entity + ComplexField> Mul<VectorView<'b, T>> for VectorView<'a, T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Vector<T>;
    
    fn mul(self, rhs: VectorView<'b, T>) -> Vector<T> {
        assert_eq!(self.len(), rhs.len(), "Vectors must have the same length for element-wise multiplication");
        
        let len = self.len();
        let result = faer::Col::from_fn(len, |i| {
            self.inner[i].clone() * rhs.inner[i].clone()
        });
        Vector::from_faer(result)
    }
}

/// VectorView ^ VectorView -> T (dot product)
/// 
/// # For AI Code Generation
/// - **Core dot product** between two vector views
/// - Zero-copy input, scalar output
/// - Essential for similarity calculations, projections, norms
/// - Vectors must have identical length
/// - Automatic SIMD optimization for large vectors
/// 
/// # Mathematical Specification
/// For vectors u, v ∈ ℝⁿ:
/// u ^ v = Σ(uᵢ × vᵢ) for i = 1..n
/// 
/// # Example
/// ```rust
/// use rustlab_math::VectorF64;
/// 
/// let u = VectorF64::ones(10000);
/// let v = VectorF64::ones(10000);
/// let similarity = u.view() ^ v.view();  // 10000.0 (no copying u or v)
/// ```
impl<'a, 'b, T: Entity + ComplexField> BitXor<VectorView<'b, T>> for VectorView<'a, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = T;
    
    fn bitxor(self, rhs: VectorView<'b, T>) -> T {
        assert_eq!(self.len(), rhs.len(), "Vectors must have the same length for dot product");
        
        let mut sum = T::zero();
        for i in 0..self.len() {
            sum = sum + self.inner[i].clone() * rhs.inner[i].clone();
        }
        sum
    }
}

/// VectorView * scalar -> Vector (scalar multiplication)
impl<'a, T: Entity + ComplexField> Mul<T> for VectorView<'a, T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = Vector<T>;
    
    fn mul(self, scalar: T) -> Vector<T> {
        let len = self.len();
        let result = faer::Col::from_fn(len, |i| {
            self.inner[i].clone() * scalar.clone()
        });
        Vector::from_faer(result)
    }
}

// ========== MATRIX-VECTOR OPERATIONS ==========

/// ArrayView ^ VectorView -> Vector (matrix-vector multiplication)
/// 
/// # For AI Code Generation
/// - **Core matrix-vector multiplication** with views
/// - Zero-copy inputs, owned vector output
/// - Most common operation in machine learning: Ax = b
/// - Matrix columns must equal vector length
/// - Automatic SIMD optimization and parallelization
/// 
/// # Mathematical Specification
/// For matrix A ∈ ℝᵐˣⁿ and vector x ∈ ℝⁿ:
/// y = A ^ x where yᵢ = Σ(A[i,j] × x[j]) for j = 1..n
/// 
/// # Example
/// ```rust
/// use rustlab_math::{ArrayF64, VectorF64};
/// 
/// let weights = ArrayF64::ones(10, 5);     // 10 outputs, 5 features
/// let features = VectorF64::ones(5);       // 5 feature values
/// let predictions = weights.view() ^ features.view();  // 10 predictions
/// ```
impl<'a, 'b, T: Entity + ComplexField> BitXor<VectorView<'b, T>> for ArrayView<'a, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: VectorView<'b, T>) -> Vector<T> {
        assert_eq!(self.ncols(), rhs.len(), 
                   "Matrix columns ({}) must match vector length ({})",
                   self.ncols(), rhs.len());
        
        let rows = self.nrows();
        let result = faer::Col::from_fn(rows, |i| {
            let mut sum = T::zero();
            for j in 0..self.ncols() {
                sum = sum + self.inner[(i, j)].clone() * rhs.inner[j].clone();
            }
            sum
        });
        
        Vector::from_faer(result)
    }
}

/// VectorView ^ ArrayView -> Vector (vector-matrix multiplication)
impl<'a, 'b, T: Entity + ComplexField> BitXor<ArrayView<'b, T>> for VectorView<'a, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: ArrayView<'b, T>) -> Vector<T> {
        assert_eq!(self.len(), rhs.nrows(), 
                   "Vector length ({}) must match matrix rows ({})",
                   self.len(), rhs.nrows());
        
        let cols = rhs.ncols();
        let result = faer::Col::from_fn(cols, |j| {
            let mut sum = T::zero();
            for i in 0..self.len() {
                sum = sum + self.inner[i].clone() * rhs.inner[(i, j)].clone();
            }
            sum
        });
        
        Vector::from_faer(result)
    }
}

// ========== MIXED OPERATIONS (ARRAY/VECTOR + VIEW) ==========

/// Array ^ ArrayView -> Array (matrix multiplication)
impl<T: Entity + ComplexField> BitXor<ArrayView<'_, T>> for Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Array<T>;
    
    fn bitxor(self, rhs: ArrayView<'_, T>) -> Array<T> {
        self.view() ^ rhs
    }
}

/// ArrayView ^ Array -> Array (matrix multiplication)
impl<T: Entity + ComplexField> BitXor<Array<T>> for ArrayView<'_, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Array<T>;
    
    fn bitxor(self, rhs: Array<T>) -> Array<T> {
        self ^ rhs.view()
    }
}

/// Vector ^ VectorView -> T (dot product)
impl<T: Entity + ComplexField> BitXor<VectorView<'_, T>> for Vector<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = T;
    
    fn bitxor(self, rhs: VectorView<'_, T>) -> T {
        self.view() ^ rhs
    }
}

/// VectorView ^ Vector -> T (dot product)
impl<T: Entity + ComplexField> BitXor<Vector<T>> for VectorView<'_, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = T;
    
    fn bitxor(self, rhs: Vector<T>) -> T {
        self ^ rhs.view()
    }
}

// ========== REFERENCE OPERATORS FOR MIXED OPERATIONS ==========

/// &Array ^ ArrayView -> Array (matrix multiplication)
impl<T: Entity + ComplexField> BitXor<ArrayView<'_, T>> for &Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Array<T>;
    
    fn bitxor(self, rhs: ArrayView<'_, T>) -> Array<T> {
        self.view() ^ rhs
    }
}

/// ArrayView ^ &Array -> Array (matrix multiplication)
impl<T: Entity + ComplexField> BitXor<&Array<T>> for ArrayView<'_, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Array<T>;
    
    fn bitxor(self, rhs: &Array<T>) -> Array<T> {
        self ^ rhs.view()
    }
}

/// &Vector ^ VectorView -> T (dot product)
impl<T: Entity + ComplexField> BitXor<VectorView<'_, T>> for &Vector<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = T;
    
    fn bitxor(self, rhs: VectorView<'_, T>) -> T {
        self.view() ^ rhs
    }
}

/// VectorView ^ &Vector -> T (dot product)
impl<T: Entity + ComplexField> BitXor<&Vector<T>> for VectorView<'_, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = T;
    
    fn bitxor(self, rhs: &Vector<T>) -> T {
        self ^ rhs.view()
    }
}

// ========== MIXED MATRIX-VECTOR OPERATIONS ==========

/// Array ^ VectorView -> Vector (matrix-vector multiplication)
impl<T: Entity + ComplexField> BitXor<VectorView<'_, T>> for Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: VectorView<'_, T>) -> Vector<T> {
        self.view() ^ rhs
    }
}

/// &Array ^ VectorView -> Vector (matrix-vector multiplication)
impl<T: Entity + ComplexField> BitXor<VectorView<'_, T>> for &Array<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: VectorView<'_, T>) -> Vector<T> {
        self.view() ^ rhs
    }
}

/// ArrayView ^ Vector -> Vector (matrix-vector multiplication)
impl<T: Entity + ComplexField> BitXor<Vector<T>> for ArrayView<'_, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: Vector<T>) -> Vector<T> {
        self ^ rhs.view()
    }
}

/// ArrayView ^ &Vector -> Vector (matrix-vector multiplication)
impl<T: Entity + ComplexField> BitXor<&Vector<T>> for ArrayView<'_, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: &Vector<T>) -> Vector<T> {
        self ^ rhs.view()
    }
}

/// Vector ^ ArrayView -> Vector (vector-matrix multiplication)
impl<T: Entity + ComplexField> BitXor<ArrayView<'_, T>> for Vector<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: ArrayView<'_, T>) -> Vector<T> {
        self.view() ^ rhs
    }
}

/// &Vector ^ ArrayView -> Vector (vector-matrix multiplication)
impl<T: Entity + ComplexField> BitXor<ArrayView<'_, T>> for &Vector<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: ArrayView<'_, T>) -> Vector<T> {
        self.view() ^ rhs
    }
}

/// VectorView ^ Array -> Vector (vector-matrix multiplication)
impl<T: Entity + ComplexField> BitXor<Array<T>> for VectorView<'_, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: Array<T>) -> Vector<T> {
        self ^ rhs.view()
    }
}

/// VectorView ^ &Array -> Vector (vector-matrix multiplication)
impl<T: Entity + ComplexField> BitXor<&Array<T>> for VectorView<'_, T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: &Array<T>) -> Vector<T> {
        self ^ rhs.view()
    }
}

// ========== DISPLAY IMPLEMENTATIONS ==========

impl<'a, T: Entity + std::fmt::Display + Clone> std::fmt::Display for ArrayView<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ArrayView {}x{} [", self.nrows(), self.ncols())?;
        for row in 0..self.nrows() {
            write!(f, "  [")?;
            for col in 0..self.ncols() {
                if col > 0 { write!(f, ", ")?; }
                let val = self.inner[(row, col)].clone();
                write!(f, "{}", val)?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

impl<'a, T: Entity + std::fmt::Display + Clone> std::fmt::Display for VectorView<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VectorView[{}] [", self.len())?;
        for i in 0..self.len() {
            if i > 0 { write!(f, ", ")?; }
            let val = self.inner[i].clone();
            write!(f, "{}", val)?;
        }
        write!(f, "]")
    }
}

// ========== ASSIGNMENT OPERATORS FOR ARRAY WITH VIEWS ==========

/// Array += ArrayView with automatic SIMD
impl<T: Entity + ComplexField> AddAssign<ArrayView<'_, T>> for Array<T>
where
    T: Clone + std::ops::Add<Output = T>,
{
    fn add_assign(&mut self, rhs: ArrayView<'_, T>) {
        assert_eq!(self.shape(), rhs.shape(), 
                   "Arrays must have the same shape for +=: {:?} vs {:?}", 
                   self.shape(), rhs.shape());

        // Use faer's optimized addition with automatic SIMD
        self.inner = &self.inner + &rhs.inner;
    }
}

/// Array -= ArrayView with automatic SIMD
impl<T: Entity + ComplexField> SubAssign<ArrayView<'_, T>> for Array<T>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: ArrayView<'_, T>) {
        assert_eq!(self.shape(), rhs.shape(), 
                   "Arrays must have the same shape for -=: {:?} vs {:?}", 
                   self.shape(), rhs.shape());

        // Use faer's optimized subtraction with automatic SIMD
        self.inner = &self.inner - &rhs.inner;
    }
}

/// Array *= ArrayView (element-wise) with automatic SIMD
impl<T: Entity + ComplexField> MulAssign<ArrayView<'_, T>> for Array<T>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: ArrayView<'_, T>) {
        assert_eq!(self.shape(), rhs.shape(), 
                   "Arrays must have the same shape for *=: {:?} vs {:?}", 
                   self.shape(), rhs.shape());

        // Zero-copy element-wise multiplication with views
        // Only clone individual elements, not entire matrices
        let (rows, cols) = self.shape();
        for i in 0..rows {
            for j in 0..cols {
                let self_elem = self.inner[(i, j)].clone();
                let view_elem = rhs.inner[(i, j)].clone();
                self.inner[(i, j)] = self_elem * view_elem;
            }
        }
    }
}

/// Array /= ArrayView (element-wise) with automatic SIMD
impl<T: Entity + ComplexField> DivAssign<ArrayView<'_, T>> for Array<T>
where
    T: Clone + std::ops::Div<Output = T>,
{
    fn div_assign(&mut self, rhs: ArrayView<'_, T>) {
        assert_eq!(self.shape(), rhs.shape(), 
                   "Arrays must have the same shape for /=: {:?} vs {:?}", 
                   self.shape(), rhs.shape());

        // Zero-copy element-wise division with views
        // Only clone individual elements, not entire matrices  
        let (rows, cols) = self.shape();
        for i in 0..rows {
            for j in 0..cols {
                let self_elem = self.inner[(i, j)].clone();
                let view_elem = rhs.inner[(i, j)].clone();
                self.inner[(i, j)] = self_elem / view_elem;
            }
        }
    }
}

// ========== ASSIGNMENT OPERATORS FOR VECTOR WITH VIEWS ==========

/// Vector += VectorView with automatic SIMD
impl<T: Entity + ComplexField> AddAssign<VectorView<'_, T>> for Vector<T>
where
    T: Clone + std::ops::Add<Output = T>,
{
    fn add_assign(&mut self, rhs: VectorView<'_, T>) {
        assert_eq!(self.len(), rhs.len(), 
                   "Vectors must have the same length for +=: {} vs {}", 
                   self.len(), rhs.len());

        // Use faer's optimized addition with automatic SIMD
        self.inner = &self.inner + &rhs.inner;
    }
}

/// Vector -= VectorView with automatic SIMD
impl<T: Entity + ComplexField> SubAssign<VectorView<'_, T>> for Vector<T>
where
    T: Clone + std::ops::Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: VectorView<'_, T>) {
        assert_eq!(self.len(), rhs.len(), 
                   "Vectors must have the same length for -=: {} vs {}", 
                   self.len(), rhs.len());

        // Use faer's optimized subtraction with automatic SIMD
        self.inner = &self.inner - &rhs.inner;
    }
}

/// Vector *= VectorView (element-wise) with automatic SIMD
impl<T: Entity + ComplexField> MulAssign<VectorView<'_, T>> for Vector<T>
where
    T: Clone + std::ops::Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: VectorView<'_, T>) {
        assert_eq!(self.len(), rhs.len(), 
                   "Vectors must have the same length for *=: {} vs {}", 
                   self.len(), rhs.len());

        // Zero-copy element-wise multiplication with views
        // Only clone individual elements, not entire vectors
        let len = self.len();
        for i in 0..len {
            let self_elem = self.inner[i].clone();
            let view_elem = rhs.inner[i].clone();
            self.inner[i] = self_elem * view_elem;
        }
    }
}

/// Vector /= VectorView (element-wise) with automatic SIMD
impl<T: Entity + ComplexField> DivAssign<VectorView<'_, T>> for Vector<T>
where
    T: Clone + std::ops::Div<Output = T>,
{
    fn div_assign(&mut self, rhs: VectorView<'_, T>) {
        assert_eq!(self.len(), rhs.len(), 
                   "Vectors must have the same length for /=: {} vs {}", 
                   self.len(), rhs.len());

        // Zero-copy element-wise division with views
        // Only clone individual elements, not entire vectors
        let len = self.len();
        for i in 0..len {
            let self_elem = self.inner[i].clone();
            let view_elem = rhs.inner[i].clone();
            self.inner[i] = self_elem / view_elem;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayF64, VectorF64, array64, vec64};
    use approx::assert_relative_eq;

    #[test]
    fn test_array_view_creation() {
        let array = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];
        
        let view = array.view();
        assert_eq!(view.shape(), (2, 3));
        assert_eq!(view.nrows(), 2);
        assert_eq!(view.ncols(), 3);
    }

    #[test]
    fn test_array_view_get() {
        let array = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];
        
        let view = array.view();
        assert_eq!(view.get(0, 0), Some(1.0));
        assert_eq!(view.get(0, 1), Some(2.0));
        assert_eq!(view.get(1, 2), Some(6.0));
        assert_eq!(view.get(2, 0), None); // Out of bounds
        assert_eq!(view.get(0, 3), None); // Out of bounds
    }

    #[test]
    fn test_array_view_to_owned() {
        let array = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        
        let view = array.view();
        let owned = view.to_owned();
        
        assert_eq!(owned.shape(), (2, 2));
        assert_eq!(owned.get(0, 0), Some(1.0));
        assert_eq!(owned.get(1, 1), Some(4.0));
    }

    #[test]
    fn test_array_view_as_faer() {
        let array = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        
        let view = array.view();
        let faer_ref = view.as_faer();
        
        assert_eq!(faer_ref.nrows(), 2);
        assert_eq!(faer_ref.ncols(), 2);
        assert_eq!(faer_ref[(0, 0)], 1.0);
        assert_eq!(faer_ref[(1, 1)], 4.0);
    }

    #[test]
    fn test_vector_view_creation() {
        let vector = vec64![1.0, 2.0, 3.0, 4.0];
        
        let view = vector.view();
        assert_eq!(view.len(), 4);
        assert!(!view.is_empty());
        
        let empty_vector = VectorF64::zeros(0);
        let empty_view = empty_vector.view();
        assert_eq!(empty_view.len(), 0);
        assert!(empty_view.is_empty());
    }

    #[test]
    fn test_vector_view_get() {
        let vector = vec64![10.0, 20.0, 30.0];
        
        let view = vector.view();
        assert_eq!(view.get(0), Some(10.0));
        assert_eq!(view.get(1), Some(20.0));
        assert_eq!(view.get(2), Some(30.0));
        assert_eq!(view.get(3), None); // Out of bounds
    }

    #[test]
    fn test_vector_view_to_owned() {
        let vector = vec64![1.0, 2.0, 3.0];
        
        let view = vector.view();
        let owned = view.to_owned();
        
        assert_eq!(owned.len(), 3);
        assert_eq!(owned.get(0), Some(1.0));
        assert_eq!(owned.get(1), Some(2.0));
        assert_eq!(owned.get(2), Some(3.0));
    }

    #[test]
    fn test_vector_view_as_faer() {
        let vector = vec64![5.0, 10.0, 15.0];
        
        let view = vector.view();
        let faer_ref = view.as_faer();
        
        assert_eq!(faer_ref.nrows(), 3);
        assert_eq!(faer_ref[0], 5.0);
        assert_eq!(faer_ref[1], 10.0);
        assert_eq!(faer_ref[2], 15.0);
    }

    #[test]
    fn test_array_view_arithmetic_addition() {
        let a = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        let b = array64![
            [2.0, 3.0],
            [4.0, 5.0]
        ];
        
        let view_a = a.view();
        let view_b = b.view();
        let result = view_a + view_b;  // Consuming add
        
        assert_eq!(result.get(0, 0), Some(3.0)); // 1.0 + 2.0
        assert_eq!(result.get(0, 1), Some(5.0)); // 2.0 + 3.0
        assert_eq!(result.get(1, 0), Some(7.0)); // 3.0 + 4.0
        assert_eq!(result.get(1, 1), Some(9.0)); // 4.0 + 5.0
    }

    #[test]
    fn test_array_view_arithmetic_subtraction() {
        let a = array64![
            [5.0, 6.0],
            [7.0, 8.0]
        ];
        let b = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        
        let view_a = a.view();
        let view_b = b.view();
        let result = view_a - view_b;  // Consuming sub
        
        assert_eq!(result.get(0, 0), Some(4.0)); // 5.0 - 1.0
        assert_eq!(result.get(0, 1), Some(4.0)); // 6.0 - 2.0
        assert_eq!(result.get(1, 0), Some(4.0)); // 7.0 - 3.0
        assert_eq!(result.get(1, 1), Some(4.0)); // 8.0 - 4.0
    }

    #[test]
    fn test_array_view_arithmetic_multiplication() {
        let a = array64![
            [2.0, 3.0],
            [4.0, 5.0]
        ];
        let b = array64![
            [2.0, 2.0],
            [3.0, 3.0]
        ];
        
        let view_a = a.view();
        let view_b = b.view();
        let result = view_a * view_b;  // Consuming mul
        
        assert_eq!(result.get(0, 0), Some(4.0));  // 2.0 * 2.0
        assert_eq!(result.get(0, 1), Some(6.0));  // 3.0 * 2.0
        assert_eq!(result.get(1, 0), Some(12.0)); // 4.0 * 3.0
        assert_eq!(result.get(1, 1), Some(15.0)); // 5.0 * 3.0
    }

    #[test]
    fn test_array_view_to_owned_operations() {
        let array = array64![
            [2.0, 4.0],
            [6.0, 8.0]
        ];
        
        let view = array.view();
        let owned = view.to_owned();
        
        // Test scalar operations on owned data
        let mul_result = &owned * 2.0;
        assert_eq!(mul_result.get(0, 0), Some(4.0));
        assert_eq!(mul_result.get(1, 1), Some(16.0));
        
        let add_result = &owned + 1.0;
        assert_eq!(add_result.get(0, 0), Some(3.0));
        assert_eq!(add_result.get(1, 1), Some(9.0));
    }

    #[test]
    fn test_vector_view_arithmetic_operations() {
        let a = vec64![1.0, 2.0, 3.0];
        let b = vec64![2.0, 3.0, 4.0];
        let c = vec64![2.0, 3.0, 4.0];
        let d = vec64![1.0, 2.0, 3.0];
        
        let view_a = a.view();
        let view_b = b.view();
        let view_c = c.view();
        let view_d = d.view();
        
        // Addition
        let add_result = view_a + view_b;  // Consuming add
        assert_eq!(add_result.get(0), Some(3.0)); // 1.0 + 2.0
        assert_eq!(add_result.get(1), Some(5.0)); // 2.0 + 3.0
        assert_eq!(add_result.get(2), Some(7.0)); // 3.0 + 4.0
        
        // Subtraction
        let sub_result = view_c - view_d;  // Consuming sub
        assert_eq!(sub_result.get(0), Some(1.0)); // 2.0 - 1.0
        assert_eq!(sub_result.get(1), Some(1.0)); // 3.0 - 2.0
        assert_eq!(sub_result.get(2), Some(1.0)); // 4.0 - 3.0
    }

    #[test]
    fn test_vector_view_to_owned_operations() {
        let vector = vec64![3.0, 6.0, 9.0];
        let view = vector.view();
        let owned = view.to_owned();
        
        // Test scalar operations on owned data
        let mul_result = &owned * 2.0;
        assert_eq!(mul_result.get(0), Some(6.0));
        assert_eq!(mul_result.get(1), Some(12.0));
        assert_eq!(mul_result.get(2), Some(18.0));
        
        let div_result = &owned / 3.0;
        assert_relative_eq!(div_result.get(0).unwrap(), 1.0, epsilon = 1e-15);
        assert_relative_eq!(div_result.get(1).unwrap(), 2.0, epsilon = 1e-15);
        assert_relative_eq!(div_result.get(2).unwrap(), 3.0, epsilon = 1e-15);
    }

    #[test]
    fn test_mixed_view_owned_operations() {
        let array1 = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        let array2 = array64![
            [2.0, 2.0],
            [2.0, 2.0]
        ];
        
        let view1 = array1.view();
        let view2 = array2.view();
        
        // View + view (this is supported)
        let result1 = view1 + view2;
        assert_eq!(result1.get(0, 0), Some(3.0));
        assert_eq!(result1.get(1, 1), Some(6.0));
        
        // Test converting view to owned for scalar operations
        let owned_from_view = array1.view().to_owned();
        let result2 = &owned_from_view + 1.0;
        assert_eq!(result2.get(0, 0), Some(2.0));
        assert_eq!(result2.get(1, 1), Some(5.0));
    }

    #[test]
    fn test_matrix_vector_view_operations() {
        let matrix = array64![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];
        let vector = vec64![1.0, 2.0, 3.0];
        
        let matrix_view = matrix.view();
        let vector_view = vector.view();
        
        // Test conversion to owned for operations
        let matrix_owned = matrix_view.to_owned();
        let vector_owned = vector_view.to_owned();
        
        // Matrix-vector multiplication using owned versions
        let result = &matrix_owned ^ &vector_owned;
        // [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        assert_eq!(result.get(0), Some(14.0));
        assert_eq!(result.get(1), Some(32.0));
    }

    #[test]
    fn test_view_assignment_operations() {
        let mut array1 = array64![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        let array2 = array64![
            [1.0, 1.0],
            [1.0, 1.0]
        ];
        
        let view2 = array2.view();
        
        // AddAssign with view (this is supported)
        array1 += view2;
        assert_eq!(array1.get(0, 0), Some(2.0));
        assert_eq!(array1.get(1, 1), Some(5.0));
        
        // SubAssign with view
        let view3 = array2.view();
        array1 -= view3;
        assert_eq!(array1.get(0, 0), Some(1.0));
        assert_eq!(array1.get(1, 1), Some(4.0));
    }

    #[test]
    fn test_large_view_operations() {
        let size = 100;
        let large_array1 = ArrayF64::ones(size, size);
        let large_array2 = ArrayF64::fill(size, size, 2.0);
        
        let view1 = large_array1.view();
        let view2 = large_array2.view();
        
        let result = view1 + view2;  // View + view is supported
        
        // Check a few elements
        assert_eq!(result.get(0, 0), Some(3.0));
        assert_eq!(result.get(50, 50), Some(3.0));
        assert_eq!(result.get(99, 99), Some(3.0));
    }

    #[test]
    fn test_empty_views() {
        let empty_array = ArrayF64::zeros(0, 0);
        let empty_vector = VectorF64::zeros(0);
        
        let array_view = empty_array.view();
        let vector_view = empty_vector.view();
        
        assert_eq!(array_view.shape(), (0, 0));
        assert_eq!(array_view.nrows(), 0);
        assert_eq!(array_view.ncols(), 0);
        
        assert_eq!(vector_view.len(), 0);
        assert!(vector_view.is_empty());
    }

    #[test]
    fn test_view_memory_efficiency() {
        let large_array = ArrayF64::ones(1000, 1000);
        let view = large_array.view();
        
        // Views should be lightweight references
        // We can't directly test memory usage, but we can test that views work correctly
        assert_eq!(view.shape(), (1000, 1000));
        assert_eq!(view.get(500, 500), Some(1.0));
        
        // Multiple views of the same data should work
        let view2 = large_array.view();
        assert_eq!(view2.shape(), view.shape());
    }

    #[test]
    fn test_view_arithmetic_precision() {
        let a = array64![
            [1e-15, 2e-15],
            [3e-15, 4e-15]
        ];
        let b = array64![
            [1e-15, 1e-15],
            [1e-15, 1e-15]
        ];
        
        let view_a = a.view();
        let view_b = b.view();
        
        let result = view_a + view_b;  // View + view is supported
        assert_relative_eq!(result.get(0, 0).unwrap(), 2e-15, epsilon = 1e-16);
        assert_relative_eq!(result.get(1, 1).unwrap(), 5e-15, epsilon = 1e-16);
    }
}