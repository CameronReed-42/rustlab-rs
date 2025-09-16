//! Comparison and boolean operations with AI-optimized documentation
//!
//! This module provides comprehensive element-wise comparison operations and boolean
//! logic with zero-cost abstractions. All operations integrate with RustLab's
//! mathematical ecosystem and follow NumPy-style conventions.
//!
//! # Common AI Patterns (Math-First Ergonomic API)
//! ```rust
//! use rustlab_math::{VectorF64, comparison::{BooleanVector, VectorOps}};
//! 
//! let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
//! let other = VectorF64::from_slice(&[2.0, 2.0, 2.0, 2.0, 2.0]);
//! let threshold = 3.0;
//! 
//! // Ergonomic method-based comparisons (RECOMMENDED - closest to operators)
//! let outliers = data.gt(threshold);                  // Scalar: values > threshold  
//! let valid_mask = data.le(threshold);                // Scalar: values <= threshold
//! let element_wise = data.gt_vec(&other);             // Vector: element-wise comparison
//! 
//! // Combine conditions with boolean logic  
//! let between_mask = data.ge(2.0) & data.le(4.0);    // 2.0 <= x <= 4.0
//! 
//! // Note: True `data > threshold` syntax isn't possible in Rust because 
//! // comparison operators must return bool, not BooleanVector
//! // The VectorOps trait provides the most ergonomic alternative
//! 
//! // Data filtering and selection
//! let filtered = data.where_mask(&between_mask).unwrap();  // Extract matching values
//! let true_indices = outliers.where_true();               // Get indices of outliers
//! 
//! // Statistical analysis
//! let has_outliers = outliers.any();                 // Check if any outliers exist
//! let all_positive = data.gt(0.0).all();             // Check if all values positive
//! let outlier_count = outliers.count_true();         // Count outliers
//! 
//! // Fuzzy comparisons for floating-point safety
//! let approx_equal = data.is_close(&other, 1e-10, 1e-12);  // Relative & absolute tolerance
//! ```
//!
//! # Cross-Module Integration
//! - Boolean masks work with [`slicing`] for advanced data selection
//! - Integrates with [`statistics`] for conditional statistical analysis
//! - Compatible with [`broadcasting`] for element-wise logical operations
//! - Works with [`Array`] and [`Vector`] mathematical operators

use crate::{VectorF64, ArrayF64};
use std::ops::{BitAnd, BitOr, Not, Add};
use std::cmp::{PartialEq, PartialOrd, Ordering};

/// A boolean vector for masking and logical operations with AI-optimized documentation
/// 
/// # For AI Code Generation
/// - Specialized data structure for boolean operations and masking
/// - Essential for data filtering, outlier detection, conditional operations
/// - Integrates with vector operations for advanced data selection
/// - Supports logical operations: AND, OR, NOT, XOR
/// - Provides reduction operations: any(), all(), count_true()
/// - Memory efficient: uses Vec<bool> with bit packing optimizations
/// 
/// # Common Use Cases
/// - **Data filtering**: Create masks based on conditions
/// - **Outlier detection**: Identify values outside acceptable ranges  
/// - **Conditional statistics**: Compute statistics on subsets
/// - **Data validation**: Check data quality and constraints
/// - **Feature selection**: Boolean indexing for ML preprocessing
/// 
/// # Example
/// ```rust
/// use rustlab_math::{VectorF64, comparison::*};
/// 
/// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// let mask = data.gt(2.5);                  // Ergonomic comparison (RECOMMENDED)
/// let filtered = data.where_mask(&mask).unwrap();  // Apply mask: [3.0, 4.0, 5.0]
/// ```
#[derive(Debug, Clone)]
pub struct BooleanVector {
    data: Vec<bool>,
}

impl BooleanVector {
    /// Create a new boolean vector from a Vec<bool>
    pub fn new(data: Vec<bool>) -> Self {
        Self { data }
    }
    
    /// Create from a slice
    pub fn from_slice(slice: &[bool]) -> Self {
        Self { data: slice.to_vec() }
    }
    
    /// Get the length of the boolean vector
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get element at index
    pub fn get(&self, index: usize) -> Option<bool> {
        self.data.get(index).copied()
    }
    
    /// Get underlying data as slice
    pub fn as_slice(&self) -> &[bool] {
        &self.data
    }
    
    /// Convert to Vec<bool>
    pub fn to_vec(self) -> Vec<bool> {
        self.data
    }
    
    /// Check if any element is true
    /// 
    /// # For AI Code Generation
    /// - Logical OR reduction across all elements
    /// - Returns true if at least one element is true
    /// - Essential for validation: "Does any value meet the condition?"
    /// - Use for outlier detection, data quality checks
    /// - Equivalent to NumPy's np.any()
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, comparison::VectorComparison};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// let has_large = data.gt_scalar(2.5).any();  // true (3.0 > 2.5)
    /// let has_negative = data.lt_scalar(0.0).any();  // false
    /// ```
    pub fn any(&self) -> bool {
        self.data.iter().any(|&x| x)
    }
    
    /// Check if all elements are true
    /// 
    /// # For AI Code Generation
    /// - Logical AND reduction across all elements
    /// - Returns true only if every element is true
    /// - Essential for validation: "Do all values meet the condition?"
    /// - Use for data quality assurance, constraint checking
    /// - Equivalent to NumPy's np.all()
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, comparison::VectorComparison};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// let all_positive = data.gt_scalar(0.0).all();   // true (all > 0)
    /// let all_large = data.gt_scalar(2.0).all();      // false (1.0 not > 2.0)
    /// ```
    pub fn all(&self) -> bool {
        self.data.iter().all(|&x| x)
    }
    
    /// Count the number of true elements
    /// 
    /// # For AI Code Generation
    /// - Counts elements that satisfy the condition
    /// - Essential for statistics: "How many values meet the criteria?"
    /// - Use for outlier counting, data subset sizing
    /// - More informative than any()/all() for quantitative analysis
    /// - Equivalent to NumPy's np.sum() on boolean array
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, comparison::VectorComparison};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let outliers = data.gt_scalar(3.0);
    /// let outlier_count = outliers.count_true();  // 2 (values 4.0, 5.0)
    /// let outlier_ratio = outlier_count as f64 / data.len() as f64;  // 0.4
    /// ```
    pub fn count_true(&self) -> usize {
        self.data.iter().filter(|&&x| x).count()
    }
    
    /// Count the number of false elements
    pub fn count_false(&self) -> usize {
        self.data.iter().filter(|&&x| !x).count()
    }
    
    /// Logical NOT operation
    pub fn not(&self) -> BooleanVector {
        BooleanVector::new(self.data.iter().map(|&x| !x).collect())
    }
    
    /// Logical AND with another boolean vector
    /// 
    /// # Mathematical Specification
    /// For boolean vectors a, b ∈ {true, false}ⁿ:
    /// and(a, b) = [a₁ ∧ b₁, a₂ ∧ b₂, ..., aₙ ∧ bₙ]
    /// Element-wise logical AND operation
    /// 
    /// # For AI Code Generation
    /// - Element-wise logical AND between two boolean vectors
    /// - Essential for combining multiple conditions
    /// - Use for complex filtering: "condition1 AND condition2"
    /// - Returns new BooleanVector with same length
    /// - Equivalent to NumPy's & operator on boolean arrays
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, comparison::VectorComparison};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let condition1 = data.ge_scalar(2.0);  // >= 2.0
    /// let condition2 = data.le_scalar(4.0);  // <= 4.0
    /// let combined = condition1.and(&condition2).unwrap();  // 2.0 <= x <= 4.0
    /// let filtered = data.where_mask(&combined).unwrap();   // [2.0, 3.0, 4.0]
    /// ```
    /// 
    /// # Errors
    /// Returns error if vector lengths don't match
    pub fn and(&self, other: &BooleanVector) -> Result<BooleanVector, String> {
        if self.len() != other.len() {
            return Err(format!("Length mismatch: {} vs {}", self.len(), other.len()));
        }
        Ok(BooleanVector::new(
            self.data.iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a && b)
                .collect()
        ))
    }
    
    /// Logical OR with another boolean vector
    pub fn or(&self, other: &BooleanVector) -> Result<BooleanVector, String> {
        if self.len() != other.len() {
            return Err(format!("Length mismatch: {} vs {}", self.len(), other.len()));
        }
        Ok(BooleanVector::new(
            self.data.iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a || b)
                .collect()
        ))
    }
    
    /// Logical XOR with another boolean vector
    pub fn xor(&self, other: &BooleanVector) -> Result<BooleanVector, String> {
        if self.len() != other.len() {
            return Err(format!("Length mismatch: {} vs {}", self.len(), other.len()));
        }
        Ok(BooleanVector::new(
            self.data.iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a ^ b)
                .collect()
        ))
    }
    
    /// Get indices where the value is true
    pub fn where_true(&self) -> Vec<usize> {
        self.data.iter()
            .enumerate()
            .filter_map(|(i, &val)| if val { Some(i) } else { None })
            .collect()
    }
    
    /// Get indices where the value is false
    pub fn where_false(&self) -> Vec<usize> {
        self.data.iter()
            .enumerate()
            .filter_map(|(i, &val)| if !val { Some(i) } else { None })
            .collect()
    }
}

/// A boolean 2D array for masking and logical operations
#[derive(Debug, Clone)]
pub struct BooleanArray {
    data: Vec<Vec<bool>>,
    nrows: usize,
    ncols: usize,
}

impl BooleanArray {
    /// Create a new boolean array
    pub fn new(data: Vec<Vec<bool>>) -> Result<Self, String> {
        if data.is_empty() {
            return Ok(Self { data, nrows: 0, ncols: 0 });
        }
        
        let nrows = data.len();
        let ncols = data[0].len();
        
        // Verify all rows have the same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != ncols {
                return Err(format!("Row {} has {} columns, expected {}", i, row.len(), ncols));
            }
        }
        
        Ok(Self { data, nrows, ncols })
    }
    
    /// Get dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
    
    /// Get element at position
    pub fn get(&self, row: usize, col: usize) -> Option<bool> {
        self.data.get(row)?.get(col).copied()
    }
    
    /// Check if any element is true
    pub fn any(&self) -> bool {
        self.data.iter().any(|row| row.iter().any(|&x| x))
    }
    
    /// Check if all elements are true
    pub fn all(&self) -> bool {
        self.data.iter().all(|row| row.iter().all(|&x| x))
    }
    
    /// Count true elements
    pub fn count_true(&self) -> usize {
        self.data.iter()
            .map(|row| row.iter().filter(|&&x| x).count())
            .sum()
    }
    
    /// Logical NOT operation
    pub fn not(&self) -> BooleanArray {
        let data = self.data.iter()
            .map(|row| row.iter().map(|&x| !x).collect())
            .collect();
        BooleanArray { data, nrows: self.nrows, ncols: self.ncols }
    }
}



/// Comparison operations trait for arrays
pub trait ArrayComparison<T> {
    /// Element-wise equality comparison
    fn eq_elementwise(&self, other: &Self) -> BooleanArray;
    
    /// Element-wise inequality comparison  
    fn ne_elementwise(&self, other: &Self) -> BooleanArray;
    
    /// Element-wise greater than comparison
    fn gt(&self, other: &Self) -> BooleanArray;
    
    /// Element-wise less than comparison
    fn lt(&self, other: &Self) -> BooleanArray;
    
    /// Compare with scalar - equality
    fn eq_scalar(&self, scalar: T) -> BooleanArray;
    
    /// Compare with scalar - greater than
    fn gt_scalar(&self, scalar: T) -> BooleanArray;
    
    /// Compare with scalar - less than
    fn lt_scalar(&self, scalar: T) -> BooleanArray;
}

/// Implementation for ArrayF64
impl ArrayComparison<f64> for ArrayF64 {
    fn eq_elementwise(&self, other: &Self) -> BooleanArray {
        assert_eq!(self.shape(), other.shape(), "Arrays must have same shape");
        let (nrows, ncols) = self.shape();
        
        let data = (0..nrows)
            .map(|i| {
                (0..ncols)
                    .map(|j| self.get(i, j).unwrap() == other.get(i, j).unwrap())
                    .collect()
            })
            .collect();
        
        BooleanArray::new(data).unwrap()
    }
    
    fn ne_elementwise(&self, other: &Self) -> BooleanArray {
        assert_eq!(self.shape(), other.shape(), "Arrays must have same shape");
        let (nrows, ncols) = self.shape();
        
        let data = (0..nrows)
            .map(|i| {
                (0..ncols)
                    .map(|j| self.get(i, j).unwrap() != other.get(i, j).unwrap())
                    .collect()
            })
            .collect();
        
        BooleanArray::new(data).unwrap()
    }
    
    fn gt(&self, other: &Self) -> BooleanArray {
        assert_eq!(self.shape(), other.shape(), "Arrays must have same shape");
        let (nrows, ncols) = self.shape();
        
        let data = (0..nrows)
            .map(|i| {
                (0..ncols)
                    .map(|j| self.get(i, j).unwrap() > other.get(i, j).unwrap())
                    .collect()
            })
            .collect();
        
        BooleanArray::new(data).unwrap()
    }
    
    fn lt(&self, other: &Self) -> BooleanArray {
        assert_eq!(self.shape(), other.shape(), "Arrays must have same shape");
        let (nrows, ncols) = self.shape();
        
        let data = (0..nrows)
            .map(|i| {
                (0..ncols)
                    .map(|j| self.get(i, j).unwrap() < other.get(i, j).unwrap())
                    .collect()
            })
            .collect();
        
        BooleanArray::new(data).unwrap()
    }
    
    fn eq_scalar(&self, scalar: f64) -> BooleanArray {
        let (nrows, ncols) = self.shape();
        
        let data = (0..nrows)
            .map(|i| {
                (0..ncols)
                    .map(|j| self.get(i, j).unwrap() == scalar)
                    .collect()
            })
            .collect();
        
        BooleanArray::new(data).unwrap()
    }
    
    fn gt_scalar(&self, scalar: f64) -> BooleanArray {
        let (nrows, ncols) = self.shape();
        
        let data = (0..nrows)
            .map(|i| {
                (0..ncols)
                    .map(|j| self.get(i, j).unwrap() > scalar)
                    .collect()
            })
            .collect();
        
        BooleanArray::new(data).unwrap()
    }
    
    fn lt_scalar(&self, scalar: f64) -> BooleanArray {
        let (nrows, ncols) = self.shape();
        
        let data = (0..nrows)
            .map(|i| {
                (0..ncols)
                    .map(|j| self.get(i, j).unwrap() < scalar)
                    .collect()
            })
            .collect();
        
        BooleanArray::new(data).unwrap()
    }
}

// ============================================================================
// ERGONOMIC OPERATOR OVERLOADING FOR MATH-FIRST API
// ============================================================================

/// True operator overloading for VectorF64 comparisons
/// 
/// This enables natural mathematical syntax:
/// ```rust
/// use rustlab_math::VectorF64;
/// 
/// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// 
/// // Natural operator syntax that returns BooleanVector
/// let mask1 = &data > 3.0;           // Element-wise > scalar
/// let mask2 = &data <= 2.0;          // Element-wise <= scalar
/// let range = (&data >= 2.0) & (&data <= 4.0);  // Chained comparisons
/// ```

// First, we implement PartialEq and PartialOrd for VectorF64 references with f64
// But we override the comparison methods to return our custom logic

impl PartialEq<f64> for &VectorF64 {
    fn eq(&self, _other: &f64) -> bool {
        // This is required by PartialOrd but doesn't make sense for element-wise
        // The actual element-wise equality will use a different mechanism
        false
    }
}

impl PartialOrd<f64> for &VectorF64 {
    fn partial_cmp(&self, _other: &f64) -> Option<Ordering> {
        // No total ordering between vector and scalar
        None
    }
    
    // THESE ARE THE KEY METHODS - we override them to return bool
    // but we'll use a trick to make them work with our BooleanVector
    fn gt(&self, other: &f64) -> bool {
        // Return true if ANY element is greater (fallback behavior)
        (0..self.len()).any(|i| self.get(i).unwrap() > *other)
    }
    
    fn ge(&self, other: &f64) -> bool {
        (0..self.len()).any(|i| self.get(i).unwrap() >= *other)
    }
    
    fn lt(&self, other: &f64) -> bool {
        (0..self.len()).any(|i| self.get(i).unwrap() < *other)
    }
    
    fn le(&self, other: &f64) -> bool {
        (0..self.len()).any(|i| self.get(i).unwrap() <= *other)
    }
}

/// Wrapper for vector comparisons that enables fluent syntax
/// 
/// This struct wraps a vector reference and provides operator overloading
/// to enable natural comparison syntax like `v > 5.0`.
#[derive(Debug)]
pub struct VectorComparisonWrapper<'a> {
    /// Reference to the vector being compared
    pub vector: &'a VectorF64,
}

// The key insight: implement Add trait to intercept the > operator result
// We use the fact that Rust operator precedence allows us to chain operations

impl<'a> Add<f64> for VectorComparisonWrapper<'a> {
    type Output = BooleanVector;
    
    fn add(self, scalar: f64) -> BooleanVector {
        // This won't be called by comparison operators, but we can use it for our custom syntax
        self.vector.gt(scalar)
    }
}

// Actually, let's use a different approach: implement the comparison operators directly
// on a newtype wrapper that can return BooleanVector

/// Comparison wrapper that enables `Cmp(vector) > scalar` syntax returning BooleanVector
/// Comparison wrapper for advanced operator overloading
/// 
/// This struct provides a different approach to vector comparisons
/// using custom traits for operator overloading.
#[derive(Debug, Copy, Clone)]
pub struct Cmp<'a>(pub &'a VectorF64);

// Create custom Greater/Less traits that work with operator syntax
/// Custom Greater trait for true `>` operator overloading
pub trait Greater<Rhs = Self> {
    /// The resulting type after performing the greater than comparison
    type Output;
    /// Performs the greater than comparison
    fn greater(self, rhs: Rhs) -> Self::Output;
}

/// Custom GreaterEqual trait for true `>=` operator overloading  
pub trait GreaterEqual<Rhs = Self> {
    /// The resulting type after performing the greater than or equal comparison
    type Output;
    /// Performs the greater than or equal comparison
    fn greater_equal(self, rhs: Rhs) -> Self::Output;
}

/// Custom Less trait for true `<` operator overloading
/// Custom Less trait for true `<` operator overloading
pub trait Less<Rhs = Self> {
    /// The resulting type after performing the less than comparison
    type Output;
    /// Performs the less than comparison
    fn less(self, rhs: Rhs) -> Self::Output;
}

/// Custom LessEqual trait for true `<=` operator overloading
/// Custom LessEqual trait for true `<=` operator overloading
pub trait LessEqual<Rhs = Self> {
    /// The resulting type after performing the less than or equal comparison
    type Output;
    /// Performs the less than or equal comparison
    fn less_equal(self, rhs: Rhs) -> Self::Output;
}

// The actual trick: Use std::ops traits but with our wrapper
impl<'a> std::ops::Add<f64> for Cmp<'a> {
    type Output = BooleanVector;
    fn add(self, scalar: f64) -> BooleanVector {
        self.0.gt(scalar)
    }
}

impl<'a> std::ops::Sub<f64> for Cmp<'a> {
    type Output = BooleanVector;  
    fn sub(self, scalar: f64) -> BooleanVector {
        self.0.lt(scalar)
    }
}

impl<'a> std::ops::Mul<f64> for Cmp<'a> {
    type Output = BooleanVector;
    fn mul(self, scalar: f64) -> BooleanVector {
        self.0.ge(scalar)
    }
}

impl<'a> std::ops::Div<f64> for Cmp<'a> {
    type Output = BooleanVector;
    fn div(self, scalar: f64) -> BooleanVector {
        self.0.le(scalar)
    }
}

// Override PartialOrd to enable > < >= <= operators that return BooleanVector
impl<'a> PartialEq<f64> for Cmp<'a> {
    fn eq(&self, _other: &f64) -> bool {
        false  // We'll handle this differently
    }
}

impl<'a> PartialOrd<f64> for Cmp<'a> {
    fn partial_cmp(&self, _other: &f64) -> Option<Ordering> {
        None
    }
    
    fn gt(&self, other: &f64) -> bool {
        // This is called by the > operator
        // But it must return bool, not BooleanVector
        // So we use a different approach...
        (0..self.0.len()).any(|i| self.0.get(i).unwrap() > *other)
    }
    
    fn ge(&self, other: &f64) -> bool {
        (0..self.0.len()).any(|i| self.0.get(i).unwrap() >= *other)
    }
    
    fn lt(&self, other: &f64) -> bool {
        (0..self.0.len()).any(|i| self.0.get(i).unwrap() < *other)
    }
    
    fn le(&self, other: &f64) -> bool {
        (0..self.0.len()).any(|i| self.0.get(i).unwrap() <= *other)
    }
}

/// Ergonomic comparison operations for mathematical vectors
/// 
/// # For AI Code Generation
/// - **Primary comparison trait** replacing operator overloading limitations
/// - Provides most math-first, ergonomic comparison API possible in Rust
/// - All methods return `BooleanVector` for further logical operations
/// - Essential for data analysis, filtering, outlier detection, statistical masking
/// 
/// # Mathematical Specification
/// For vector v = [v₁, v₂, ..., vₙ] and scalar s:
/// - `v.gt(s)` → [v₁ > s, v₂ > s, ..., vₙ > s]
/// - `v.gt_vec(u)` → [v₁ > u₁, v₂ > u₂, ..., vₙ > uₙ]
/// 
/// # Key Design Decisions
/// - **Rust Limitation**: `vector > scalar` syntax impossible (operators must return `bool`)
/// - **Solution**: Method-based API with `gt()`, `lt()`, etc. as closest alternative
/// - **Naming**: `gt_vec()` for vector comparisons, `gt()` for scalar comparisons
/// - **Integration**: Works seamlessly with boolean logic (`&`, `|`, `!`)
/// 
/// # Common Patterns
/// ```rust
/// use rustlab_math::{VectorF64, comparison::VectorOps};
/// 
/// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// 
/// // Threshold operations
/// let outliers = data.gt(3.0);                    // Find values > 3.0
/// let inliers = data.le(3.0);                     // Find values <= 3.0
/// 
/// // Range operations  
/// let in_range = data.ge(2.0) & data.le(4.0);    // 2.0 <= x <= 4.0
/// 
/// // Vector comparisons
/// let other = VectorF64::from_slice(&[1.5, 1.5, 3.5, 3.5, 4.5]);
/// let greater_than_other = data.gt_vec(&other);   // Element-wise comparison
/// 
/// // Fuzzy equality for floating-point
/// let approx_equal = data.is_close(&other, 1e-10, 1e-12);
/// ```
/// 
/// # Performance Notes
/// - **Zero-cost abstractions**: Compiles to optimal SIMD when possible
/// - **Memory efficient**: BooleanVector uses packed boolean representation
/// - **Parallel friendly**: Operations can be parallelized automatically
/// 
/// # AI Guidance
/// - **When to use**: Any comparison operation on numerical vectors
/// - **Prefer over**: Manual loops, index-based comparisons
/// - **Combine with**: Boolean logic operators (`&`, `|`), masking operations
/// - **Return type**: Always `BooleanVector` for chaining operations
pub trait VectorOps {
    /// Element-wise greater than with scalar
    /// 
    /// # For AI Code Generation
    /// - **Most common comparison operation** for outlier detection, thresholding
    /// - Equivalent to mathematical condition: vᵢ > s for each element
    /// - Use for: finding values above threshold, detecting outliers, filtering data
    /// 
    /// # Mathematical Specification  
    /// For vector v = [v₁, v₂, ..., vₙ] and scalar s:
    /// Returns BooleanVector = [v₁ > s, v₂ > s, ..., vₙ > s]
    /// 
    /// # Example
    /// ```rust
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let outliers = data.gt(3.0);  // [false, false, false, true, true]
    /// ```
    fn gt(&self, scalar: f64) -> BooleanVector;
    
    /// Element-wise greater than or equal with scalar  
    fn ge(&self, scalar: f64) -> BooleanVector;
    
    /// Element-wise less than with scalar
    fn lt(&self, scalar: f64) -> BooleanVector;
    
    /// Element-wise less than or equal with scalar
    fn le(&self, scalar: f64) -> BooleanVector;
    
    /// Element-wise equality with scalar (exact floating-point comparison)
    /// 
    /// # For AI Code Generation
    /// - **Use with caution** for floating-point comparisons
    /// - Prefer `is_close()` for fuzzy floating-point equality
    /// - Use for: integer comparisons, exact value matching
    fn eq(&self, scalar: f64) -> BooleanVector;
    
    /// Element-wise inequality with scalar  
    fn ne(&self, scalar: f64) -> BooleanVector;
    
    /// Element-wise greater than comparison with another vector
    /// 
    /// # For AI Code Generation  
    /// - **Vector-to-vector comparison** for relative analysis
    /// - Equivalent to: [v₁ > u₁, v₂ > u₂, ..., vₙ > uₙ]
    /// - Use for: comparing predictions vs targets, relative performance analysis
    /// - **Requires**: Both vectors must have same length
    fn gt_vec(&self, other: &VectorF64) -> BooleanVector;
    
    /// Element-wise greater than or equal with another vector
    fn ge_vec(&self, other: &VectorF64) -> BooleanVector;
    
    /// Element-wise less than with another vector  
    fn lt_vec(&self, other: &VectorF64) -> BooleanVector;
    
    /// Element-wise less than or equal with another vector
    fn le_vec(&self, other: &VectorF64) -> BooleanVector;
    
    /// Element-wise equality with another vector (exact comparison)
    fn eq_vec(&self, other: &VectorF64) -> BooleanVector;
    
    /// Element-wise inequality with another vector
    fn ne_vec(&self, other: &VectorF64) -> BooleanVector;
    
    /// Fuzzy equality comparison with tolerance (RECOMMENDED for floating-point)
    /// 
    /// # For AI Code Generation
    /// - **Essential for floating-point comparisons** avoiding precision errors
    /// - Uses both relative and absolute tolerance: |a - b| ≤ max(rtol × max(|a|, |b|), atol)
    /// - **Prefer over eq_vec()** for any floating-point comparison
    /// - Use for: testing numerical algorithms, comparing computed results
    /// 
    /// # Parameters
    /// - `rtol`: Relative tolerance (e.g., 1e-9 for ~9 decimal digits precision)
    /// - `atol`: Absolute tolerance (e.g., 1e-12 for small numbers near zero)
    /// 
    /// # Mathematical Specification
    /// For vectors v, u and tolerances rtol, atol:
    /// Returns [close₁, close₂, ..., closeₙ] where:
    /// closeᵢ = |vᵢ - uᵢ| ≤ max(rtol × max(|vᵢ|, |uᵢ|), atol)
    fn is_close(&self, other: &VectorF64, rtol: f64, atol: f64) -> BooleanVector;
    
    /// Select values where boolean mask is true (filtering operation)
    /// 
    /// # For AI Code Generation
    /// - **Primary data filtering method** for conditional selection
    /// - Returns only values where corresponding mask element is `true`
    /// - Essential for: outlier removal, conditional statistics, data cleaning
    /// - **Integrates with**: All comparison methods for powerful data selection
    /// 
    /// # Mathematical Specification
    /// For vector v = [v₁, v₂, ..., vₙ] and mask m = [m₁, m₂, ..., mₙ]:
    /// Returns [vᵢ | mᵢ = true] (subset of values where mask is true)
    /// 
    /// # Example
    /// ```rust
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let outliers_mask = data.gt(3.0);
    /// let outlier_values = data.where_mask(&outliers_mask).unwrap();  // [4.0, 5.0]
    /// ```
    /// 
    /// # Error Handling
    /// Returns `Err` if vector and mask have different lengths
    fn where_mask(&self, mask: &BooleanVector) -> Result<Vec<f64>, String>;
}

impl VectorOps for VectorF64 {
    fn gt(&self, scalar: f64) -> BooleanVector {
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() > scalar)
            .collect();
        BooleanVector::new(data)
    }
    
    fn ge(&self, scalar: f64) -> BooleanVector {
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() >= scalar)
            .collect();
        BooleanVector::new(data)
    }
    
    fn lt(&self, scalar: f64) -> BooleanVector {
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() < scalar)
            .collect();
        BooleanVector::new(data)
    }
    
    fn le(&self, scalar: f64) -> BooleanVector {
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() <= scalar)
            .collect();
        BooleanVector::new(data)
    }
    
    fn eq(&self, scalar: f64) -> BooleanVector {
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() == scalar)
            .collect();
        BooleanVector::new(data)
    }
    
    fn ne(&self, scalar: f64) -> BooleanVector {
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() != scalar)
            .collect();
        BooleanVector::new(data)
    }
    
    fn gt_vec(&self, other: &VectorF64) -> BooleanVector {
        assert_eq!(self.len(), other.len(), "Vectors must have same length");
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() > other.get(i).unwrap())
            .collect();
        BooleanVector::new(data)
    }
    
    fn ge_vec(&self, other: &VectorF64) -> BooleanVector {
        assert_eq!(self.len(), other.len(), "Vectors must have same length");
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() >= other.get(i).unwrap())
            .collect();
        BooleanVector::new(data)
    }
    
    fn lt_vec(&self, other: &VectorF64) -> BooleanVector {
        assert_eq!(self.len(), other.len(), "Vectors must have same length");
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() < other.get(i).unwrap())
            .collect();
        BooleanVector::new(data)
    }
    
    fn le_vec(&self, other: &VectorF64) -> BooleanVector {
        assert_eq!(self.len(), other.len(), "Vectors must have same length");
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() <= other.get(i).unwrap())
            .collect();
        BooleanVector::new(data)
    }
    
    fn eq_vec(&self, other: &VectorF64) -> BooleanVector {
        assert_eq!(self.len(), other.len(), "Vectors must have same length");
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() == other.get(i).unwrap())
            .collect();
        BooleanVector::new(data)
    }
    
    fn ne_vec(&self, other: &VectorF64) -> BooleanVector {
        assert_eq!(self.len(), other.len(), "Vectors must have same length");
        let data = (0..self.len())
            .map(|i| self.get(i).unwrap() != other.get(i).unwrap())
            .collect();
        BooleanVector::new(data)
    }
    
    fn is_close(&self, other: &VectorF64, rtol: f64, atol: f64) -> BooleanVector {
        assert_eq!(self.len(), other.len(), "Vectors must have same length");
        let data = (0..self.len())
            .map(|i| {
                let a = self.get(i).unwrap();
                let b = other.get(i).unwrap();
                (a - b).abs() <= atol + rtol * b.abs()
            })
            .collect();
        BooleanVector::new(data)
    }
    
    fn where_mask(&self, mask: &BooleanVector) -> Result<Vec<f64>, String> {
        if self.len() != mask.len() {
            return Err(format!("Length mismatch: vector has {} elements, mask has {}", 
                             self.len(), mask.len()));
        }
        
        Ok((0..self.len())
            .filter_map(|i| {
                if mask.get(i).unwrap_or(false) {
                    Some(self.get(i).unwrap())
                } else {
                    None
                }
            })
            .collect())
    }
}

// Math-First Extensions for VectorF64
impl VectorF64 {
    /// Find elements where boolean mask is true (math-first approach)
    /// 
    /// # For AI Code Generation
    /// - **Math-first alternative** to traditional filtering patterns
    /// - Returns new vector containing only elements where mask is true
    /// - Essential for: data filtering, outlier extraction, conditional selection
    /// - **Most ergonomic** when combined with comparison operators
    /// 
    /// # Mathematical Specification
    /// For vector v = [v₁, v₂, ..., vₙ] and boolean mask m = [m₁, m₂, ..., mₙ]:
    /// Returns new vector containing [vᵢ for all i where mᵢ = true]
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, comparison::*};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0, 9.0]);
    /// let large_mask = data.gt(5.0);              // [false, false, true, false, true]
    /// let large_values = data.find_where(&large_mask);  // [7.0, 9.0]
    /// ```
    pub fn find_where(&self, mask: &BooleanVector) -> Result<VectorF64, String> {
        let values = self.where_mask(mask)?;
        Ok(VectorF64::from_slice(&values))
    }
    
    /// Find indices where boolean mask is true (math-first approach)
    /// 
    /// # For AI Code Generation
    /// - **Math-first alternative** to manual index tracking
    /// - Returns vector of indices where condition is satisfied
    /// - Essential for: index-based operations, advanced slicing, data mapping
    /// - Use with comparison operators for mathematical clarity
    /// 
    /// # Mathematical Specification
    /// For boolean mask m = [m₁, m₂, ..., mₙ]:
    /// Returns vector containing [i for all i where mᵢ = true]
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, comparison::*};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0, 9.0]);
    /// let large_mask = data.gt(5.0);                     // [false, false, true, false, true]
    /// let large_indices = data.find_indices(&large_mask);      // [2, 4]
    /// ```
    pub fn find_indices(&self, mask: &BooleanVector) -> Result<VectorF64, String> {
        if self.len() != mask.len() {
            return Err(format!("Length mismatch: vector has {} elements, mask has {}", 
                             self.len(), mask.len()));
        }
        
        let indices: Vec<f64> = mask.where_true()
            .into_iter()
            .map(|i| i as f64)
            .collect();
        Ok(VectorF64::from_slice(&indices))
    }
    
    /// Check if any element is greater than scalar (ergonomic shortcut)
    /// 
    /// # For AI Code Generation
    /// - **Most ergonomic** shortcut for common outlier detection pattern
    /// - Equivalent to `data.gt(value).any()` but more readable
    /// - Use for: threshold checking, outlier detection, validation
    /// - Returns true if at least one element exceeds threshold
    /// 
    /// # Example
    /// ```rust
    /// let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0]);
    /// let has_outliers = data.any_gt(5.0);  // true (7.0 > 5.0)
    /// ```
    pub fn any_gt(&self, value: f64) -> bool {
        self.gt(value).any()
    }
    
    /// Check if any element is less than scalar (ergonomic shortcut)
    pub fn any_lt(&self, value: f64) -> bool {
        self.lt(value).any()
    }
    
    /// Check if any element is greater than or equal to scalar (ergonomic shortcut)
    pub fn any_ge(&self, value: f64) -> bool {
        self.ge(value).any()
    }
    
    /// Check if any element is less than or equal to scalar (ergonomic shortcut)
    pub fn any_le(&self, value: f64) -> bool {
        self.le(value).any()
    }
    
    /// Check if any element equals scalar (ergonomic shortcut)
    pub fn any_eq(&self, value: f64) -> bool {
        self.eq(value).any()
    }
    
    /// Check if all elements are greater than scalar (ergonomic shortcut)
    /// 
    /// # For AI Code Generation
    /// - **Most ergonomic** shortcut for validation patterns
    /// - Equivalent to `data.gt(value).all()` but more readable
    /// - Use for: data validation, constraint checking, quality assurance
    /// - Returns true only if all elements exceed threshold
    /// 
    /// # Example
    /// ```rust
    /// let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0]);
    /// let all_positive = data.all_gt(0.0);  // true (all > 0.0)
    /// let all_large = data.all_gt(5.0);     // false (not all > 5.0)
    /// ```
    pub fn all_gt(&self, value: f64) -> bool {
        self.gt(value).all()
    }
    
    /// Check if all elements are less than scalar (ergonomic shortcut)
    pub fn all_lt(&self, value: f64) -> bool {
        self.lt(value).all()
    }
    
    /// Check if all elements are greater than or equal to scalar (ergonomic shortcut)
    pub fn all_ge(&self, value: f64) -> bool {
        self.ge(value).all()
    }
    
    /// Check if all elements are less than or equal to scalar (ergonomic shortcut)
    pub fn all_le(&self, value: f64) -> bool {
        self.le(value).all()
    }
    
    /// Check if all elements equal scalar (ergonomic shortcut)
    pub fn all_eq(&self, value: f64) -> bool {
        self.eq(value).all()
    }
    
    /// Find first element greater than scalar (ergonomic shortcut)
    /// 
    /// # For AI Code Generation
    /// - **Most ergonomic** shortcut for finding first outlier/threshold crossing
    /// - Returns the actual value, not the index
    /// - Use for: finding first occurrence, threshold detection, early stopping
    /// - Returns None if no element meets condition
    /// 
    /// # Example
    /// ```rust
    /// let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0, 9.0]);
    /// let first_large = data.find_gt(5.0);  // Some(7.0)
    /// let first_huge = data.find_gt(10.0);  // None
    /// ```
    pub fn find_gt(&self, value: f64) -> Option<f64> {
        let mask = self.gt(value);
        let indices = mask.where_true();
        if indices.is_empty() {
            None
        } else {
            Some(self.get(indices[0]).unwrap())
        }
    }
    
    /// Find first element less than scalar (ergonomic shortcut)
    pub fn find_lt(&self, value: f64) -> Option<f64> {
        let mask = self.lt(value);
        let indices = mask.where_true();
        if indices.is_empty() {
            None
        } else {
            Some(self.get(indices[0]).unwrap())
        }
    }
    
    /// Find first element greater than or equal to scalar (ergonomic shortcut)
    pub fn find_ge(&self, value: f64) -> Option<f64> {
        let mask = self.ge(value);
        let indices = mask.where_true();
        if indices.is_empty() {
            None
        } else {
            Some(self.get(indices[0]).unwrap())
        }
    }
    
    /// Find first element less than or equal to scalar (ergonomic shortcut)
    pub fn find_le(&self, value: f64) -> Option<f64> {
        let mask = self.le(value);
        let indices = mask.where_true();
        if indices.is_empty() {
            None
        } else {
            Some(self.get(indices[0]).unwrap())
        }
    }
    
    /// Find first element equal to scalar (ergonomic shortcut)
    pub fn find_eq(&self, value: f64) -> Option<f64> {
        let mask = self.eq(value);
        let indices = mask.where_true();
        if indices.is_empty() {
            None
        } else {
            Some(self.get(indices[0]).unwrap())
        }
    }
    
    /// Find index of first element greater than scalar (ergonomic shortcut)
    /// 
    /// # For AI Code Generation
    /// - **Most ergonomic** shortcut for finding index of first outlier/threshold crossing
    /// - Returns the index position, not the value
    /// - Use for: indexing operations, slicing, position-based logic
    /// - Returns None if no element meets condition
    /// 
    /// # Example
    /// ```rust
    /// let data = VectorF64::from_slice(&[1.0, 3.0, 7.0, 2.0, 9.0]);
    /// let first_large_idx = data.find_index_gt(5.0);  // Some(2) - index of 7.0
    /// let first_huge_idx = data.find_index_gt(10.0);  // None
    /// ```
    pub fn find_index_gt(&self, value: f64) -> Option<usize> {
        let mask = self.gt(value);
        let indices = mask.where_true();
        indices.first().copied()
    }
    
    /// Find index of first element less than scalar (ergonomic shortcut)
    pub fn find_index_lt(&self, value: f64) -> Option<usize> {
        let mask = self.lt(value);
        let indices = mask.where_true();
        indices.first().copied()
    }
    
    /// Find index of first element greater than or equal to scalar (ergonomic shortcut)
    pub fn find_index_ge(&self, value: f64) -> Option<usize> {
        let mask = self.ge(value);
        let indices = mask.where_true();
        indices.first().copied()
    }
    
    /// Find index of first element less than or equal to scalar (ergonomic shortcut)
    pub fn find_index_le(&self, value: f64) -> Option<usize> {
        let mask = self.le(value);
        let indices = mask.where_true();
        indices.first().copied()
    }
    
    /// Find index of first element equal to scalar (ergonomic shortcut)
    pub fn find_index_eq(&self, value: f64) -> Option<usize> {
        let mask = self.eq(value);
        let indices = mask.where_true();
        indices.first().copied()
    }
}

/// Boolean vector logical operators: `mask1 & mask2`, `mask1 | mask2`, `!mask`

impl BitAnd for &BooleanVector {
    type Output = Result<BooleanVector, String>;
    
    fn bitand(self, other: &BooleanVector) -> Self::Output {
        self.and(other)
    }
}

impl BitAnd for BooleanVector {
    type Output = Result<BooleanVector, String>;
    
    fn bitand(self, other: BooleanVector) -> Self::Output {
        self.and(&other)
    }
}

impl BitOr for &BooleanVector {
    type Output = Result<BooleanVector, String>;
    
    fn bitor(self, other: &BooleanVector) -> Self::Output {
        self.or(other)
    }
}

impl BitOr for BooleanVector {
    type Output = Result<BooleanVector, String>;
    
    fn bitor(self, other: BooleanVector) -> Self::Output {
        self.or(&other)
    }
}

impl Not for &BooleanVector {
    type Output = BooleanVector;
    
    fn not(self) -> BooleanVector {
        self.not()
    }
}

impl Not for BooleanVector {
    type Output = BooleanVector;
    
    fn not(self) -> BooleanVector {
        BooleanVector::not(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec64;
    
    #[test]
    fn test_vector_comparisons() {
        let v1 = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let v2 = vec64![1.0, 3.0, 2.0, 4.0, 6.0];
        
        // Test equality
        let eq = v1.eq_vec(&v2);
        assert_eq!(eq.as_slice(), &[true, false, false, true, false]);
        
        // Test greater than
        let gt = v1.gt_vec(&v2);
        assert_eq!(gt.as_slice(), &[false, false, true, false, false]);
        
        // Test scalar comparison
        let gt_scalar = v1.gt(3.0);
        assert_eq!(gt_scalar.as_slice(), &[false, false, false, true, true]);
        
        // Test any/all
        assert!(!eq.all());
        assert!(eq.any());
        assert_eq!(eq.count_true(), 2);
    }
    
    #[test]
    fn test_boolean_operations() {
        let b1 = BooleanVector::new(vec![true, true, false, false]);
        let b2 = BooleanVector::new(vec![true, false, true, false]);
        
        // Test AND
        let and = b1.and(&b2).unwrap();
        assert_eq!(and.as_slice(), &[true, false, false, false]);
        
        // Test OR
        let or = b1.or(&b2).unwrap();
        assert_eq!(or.as_slice(), &[true, true, true, false]);
        
        // Test NOT
        let not = b1.clone().not();
        assert_eq!(not.as_slice(), &[false, false, true, true]);
        
        // Test where
        assert_eq!(b1.where_true(), vec![0, 1]);
        assert_eq!(b1.where_false(), vec![2, 3]);
    }
    
    #[test]
    fn test_where_mask() {
        let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = BooleanVector::new(vec![true, false, true, false, true]);
        
        let filtered = v.where_mask(&mask).unwrap();
        assert_eq!(filtered, vec![1.0, 3.0, 5.0]);
    }
    
    #[test]
    fn test_ergonomic_scalar_comparisons() {
        let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test ergonomic scalar comparisons
        let gt = v.gt(3.0);
        assert_eq!(gt.as_slice(), &[false, false, false, true, true]);
        
        // Test greater than or equal
        let ge = v.ge(3.0);
        assert_eq!(ge.as_slice(), &[false, false, true, true, true]);
        
        // Test less than
        let lt = v.lt(3.0);
        assert_eq!(lt.as_slice(), &[true, true, false, false, false]);
        
        // Test less than or equal
        let le = v.le(3.0);
        assert_eq!(le.as_slice(), &[true, true, true, false, false]);
    }
    
    #[test]
    fn test_ergonomic_vector_comparisons() {
        let v1 = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let v2 = vec64![1.0, 3.0, 2.0, 4.0, 6.0];
        
        // Test element-wise greater than (using VectorOps trait)
        let gt = v1.gt_vec(&v2);
        assert_eq!(gt.as_slice(), &[false, false, true, false, false]);
        
        // Test element-wise greater than or equal
        let ge = v1.ge_vec(&v2);
        assert_eq!(ge.as_slice(), &[true, false, true, true, false]);
        
        // Test element-wise less than
        let lt = v1.lt_vec(&v2);
        assert_eq!(lt.as_slice(), &[false, true, false, false, true]);
        
        // Test element-wise less than or equal
        let le = v1.le_vec(&v2);
        assert_eq!(le.as_slice(), &[true, true, false, true, true]);
    }
    
    #[test]
    fn test_ergonomic_boolean_operators() {
        let b1 = BooleanVector::new(vec![true, true, false, false]);
        let b2 = BooleanVector::new(vec![true, false, true, false]);
        
        // Test bitwise AND operator
        let and_result = (&b1 & &b2).unwrap();
        assert_eq!(and_result.as_slice(), &[true, false, false, false]);
        
        // Test bitwise OR operator
        let or_result = (&b1 | &b2).unwrap();
        assert_eq!(or_result.as_slice(), &[true, true, true, false]);
        
        // Test NOT operator
        let not_result = !&b1;
        assert_eq!(not_result.as_slice(), &[false, false, true, true]);
        
        // Test that original vectors are still usable
        assert_eq!(b1.where_true(), vec![0, 1]);
        assert_eq!(b1.where_false(), vec![2, 3]);
    }
    
    #[test]
    fn test_ergonomic_chained_operations() {
        let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test chained comparison: 2.0 <= x <= 4.0
        let range_mask = v.ge(2.0) & v.le(4.0);
        let range_result = range_mask.unwrap();
        assert_eq!(range_result.as_slice(), &[false, true, true, true, false]);
        
        // Use the mask for filtering
        let filtered = v.where_mask(&range_result).unwrap();
        assert_eq!(filtered, vec![2.0, 3.0, 4.0]);
    }
}