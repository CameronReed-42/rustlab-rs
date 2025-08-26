//! Axis-specific reduction operations with AI-optimized documentation
//!
//! This module provides efficient reduction operations along specific axes following
//! NumPy-style conventions. All operations use zero-cost abstractions and integrate
//! with RustLab's mathematical operators for consistent AI code generation.
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::{ArrayF64, reductions::{AxisReductions, Axis}};
//! 
//! let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
//! let col_means = data.mean_axis(Axis::Rows).unwrap();    // Feature means
//! let row_sums = data.sum_axis(Axis::Cols).unwrap();      // Sample totals
//! 
//! // Feature normalization using axis reductions
//! let normalized = (&data - &col_means.broadcast_to_matrix(data.shape())?)?;
//! 
//! // Statistical analysis
//! let feature_stds = data.std_axis(Axis::Rows).unwrap();  // Feature standard deviations
//! ```
//!
//! # Cross-Module Integration
//! - Compatible with [`broadcasting`] for normalization patterns
//! - Integrates with [`statistics`] for comprehensive statistical analysis
//! - Works with [`Array`] and [`Vector`] core operations

use crate::{Array, Vector, Result};
use faer_entity::Entity;
use faer_traits::ComplexField;
use num_traits::{Zero, One, FromPrimitive, Float};
use std::f64;

/// Axis specification for reduction operations with AI-optimized documentation
/// 
/// # For AI Code Generation
/// - `Axis::Rows` (0): Reduces along rows, produces column statistics (most common in ML)
/// - `Axis::Cols` (1): Reduces along columns, produces row statistics (less common)
/// - Follows NumPy convention: axis=0 for rows, axis=1 for columns
/// - Used with all reduction functions: sum_axis, mean_axis, std_axis, etc.
/// 
/// # Common ML Usage
/// - **Feature statistics**: Use `Axis::Rows` to get per-feature statistics
/// - **Sample statistics**: Use `Axis::Cols` to get per-sample statistics
/// - **Data normalization**: `mean_axis(Axis::Rows)` for feature means
/// 
/// # Dimension Rules
/// - Input: (m×n) matrix
/// - `Axis::Rows`: Output length = n (columns)
/// - `Axis::Cols`: Output length = m (rows)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Axis {
    /// Reduce along rows (axis 0) - results in column vector
    /// Most common: compute statistics per feature/column
    Rows = 0,
    /// Reduce along columns (axis 1) - results in row vector
    /// Less common: compute statistics per sample/row
    Cols = 1,
}

/// Trait for axis-specific reduction operations with AI-optimized documentation
/// 
/// This trait provides NumPy-style axis reductions with comprehensive mathematical
/// specifications for AI code generation. All methods follow consistent patterns
/// and integrate with RustLab's broadcasting and operator systems.
/// 
/// # For AI Code Generation
/// - All methods return `Result<Vector<T>>` for error handling
/// - Axis parameter determines reduction direction
/// - Methods with `_keepdims` preserve matrix structure for broadcasting
/// - Follows NumPy naming and behavior conventions
/// - Compatible with zero-cost abstractions
/// 
/// # Common Patterns
/// - Feature normalization: `mean_axis(Axis::Rows)` + `std_axis(Axis::Rows)`
/// - Data aggregation: `sum_axis(Axis::Cols)` for sample totals
/// - Statistical analysis: `var_axis(Axis::Rows)` for feature variance
pub trait AxisReductions<T: Entity + ComplexField> {
    /// Sum along specified axis
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ:
    /// - `Axis::Rows`: Σᵢ(A[i,j]) for i = 1..m → vector ∈ ℝⁿ (column sums)
    /// - `Axis::Cols`: Σⱼ(A[i,j]) for j = 1..n → vector ∈ ℝᵐ (row sums)
    /// 
    /// # Complexity
    /// - Time: O(m×n) single pass through matrix
    /// - Space: O(max(m,n)) for result vector
    /// 
    /// # For AI Code Generation
    /// - `Axis::Rows`: Most common, produces feature sums (length = ncols)
    /// - `Axis::Cols`: Produces sample sums (length = nrows)
    /// - Equivalent to NumPy's `np.sum(axis=0)` and `np.sum(axis=1)`
    /// - Result type: Vector with length matching remaining dimension
    /// - Use for data aggregation, feature analysis, totals computation
    fn sum_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: Zero + One + Clone;
    
    /// Sum along axis with keepdims option for broadcasting compatibility
    /// 
    /// # Mathematical Specification
    /// Same as sum_axis but preserves matrix structure:
    /// - `Axis::Rows`: (m×n) → (1×n) matrix (row vector)
    /// - `Axis::Cols`: (m×n) → (m×1) matrix (column vector)
    /// 
    /// # For AI Code Generation
    /// - Maintains matrix structure for broadcasting operations
    /// - Equivalent to NumPy's `np.sum(axis=0, keepdims=True)`
    /// - Use when you need to broadcast result back to original matrix
    /// - Common in normalization: `(data - mean_keepdims) / std_keepdims`
    /// - Result can be used directly with broadcasting operators
    fn sum_axis_keepdims(&self, axis: Axis) -> Result<Array<T>>
    where
        T: Zero + One + Clone;
    
    /// Mean along specified axis
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ:
    /// - `Axis::Rows`: μⱼ = (1/m) × Σᵢ(A[i,j]) → vector ∈ ℝⁿ (column means)
    /// - `Axis::Cols`: μᵢ = (1/n) × Σⱼ(A[i,j]) → vector ∈ ℝᵐ (row means)
    /// 
    /// # Complexity
    /// - Time: O(m×n) for sum + O(1) for division
    /// - Space: O(max(m,n)) for result vector
    /// 
    /// # For AI Code Generation
    /// - `Axis::Rows`: Feature means (most common in ML preprocessing)
    /// - `Axis::Cols`: Sample means (less common)
    /// - Equivalent to NumPy's `np.mean(axis=0)` and `np.mean(axis=1)`
    /// - Essential for data normalization and centering
    /// - Often combined with broadcasting for mean centering: `data - mean`
    /// - Use with std_axis for z-score normalization
    fn mean_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: Zero + One + Clone + FromPrimitive + std::ops::Div<Output = T>;
    
    /// Mean along axis with keepdims option
    /// 
    /// # For AI Code Generation
    /// - Same as mean_axis but preserves matrix structure for broadcasting
    /// - `Axis::Rows`: (m×n) → (1×n) matrix of feature means
    /// - `Axis::Cols`: (m×n) → (m×1) matrix of sample means
    /// - Critical for normalization: `(data - mean_keepdims) / std_keepdims`
    /// - Equivalent to NumPy's `np.mean(axis=0, keepdims=True)`
    fn mean_axis_keepdims(&self, axis: Axis) -> Result<Array<T>>
    where
        T: Zero + One + Clone + FromPrimitive + std::ops::Div<Output = T>;
    
    /// Minimum along specified axis
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ:
    /// - `Axis::Rows`: min_j = minᵢ(A[i,j]) → vector ∈ ℝⁿ (column minimums)
    /// - `Axis::Cols`: min_i = minⱼ(A[i,j]) → vector ∈ ℝᵐ (row minimums)
    /// 
    /// # Complexity
    /// - Time: O(m×n) single pass with comparison
    /// - Space: O(max(m,n)) for result vector
    /// 
    /// # For AI Code Generation
    /// - `Axis::Rows`: Feature minimums (per column)
    /// - `Axis::Cols`: Sample minimums (per row)
    /// - Equivalent to NumPy's `np.min(axis=0)` and `np.min(axis=1)`
    /// - Use for data range analysis, outlier detection, bounds checking
    /// - Often combined with max_axis for range calculations
    fn min_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: PartialOrd + Clone;
    
    /// Maximum along specified axis
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ:
    /// - `Axis::Rows`: max_j = maxᵢ(A[i,j]) → vector ∈ ℝⁿ (column maximums)
    /// - `Axis::Cols`: max_i = maxⱼ(A[i,j]) → vector ∈ ℝᵐ (row maximums)
    /// 
    /// # Complexity
    /// - Time: O(m×n) single pass with comparison
    /// - Space: O(max(m,n)) for result vector
    /// 
    /// # For AI Code Generation
    /// - `Axis::Rows`: Feature maximums (per column)
    /// - `Axis::Cols`: Sample maximums (per row)
    /// - Equivalent to NumPy's `np.max(axis=0)` and `np.max(axis=1)`
    /// - Use for data range analysis, normalization bounds, feature scaling
    /// - Combined with min_axis: `(data - min) / (max - min)` for min-max scaling
    fn max_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: PartialOrd + Clone;
    
    /// Standard deviation along specified axis
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ with means μ:
    /// - `Axis::Rows`: σⱼ = √((1/(m-1)) × Σᵢ(A[i,j] - μⱼ)²) → vector ∈ ℝⁿ
    /// - `Axis::Cols`: σᵢ = √((1/(n-1)) × Σⱼ(A[i,j] - μᵢ)²) → vector ∈ ℝᵐ
    /// Uses sample standard deviation (Bessel's correction, ddof=1)
    /// 
    /// # Complexity
    /// - Time: O(m×n) for mean + O(m×n) for variance + O(k) for sqrt
    /// - Space: O(max(m,n)) for result vector
    /// 
    /// # For AI Code Generation
    /// - `Axis::Rows`: Feature standard deviations (most common for normalization)
    /// - `Axis::Cols`: Sample standard deviations (less common)
    /// - Equivalent to NumPy's `np.std(axis=0, ddof=1)` (sample std dev)
    /// - Essential for z-score normalization: `(data - mean) / std`
    /// - Use with mean_axis for standardization in machine learning
    /// - Always non-negative values
    fn std_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + Zero + One + Clone + FromPrimitive + Float;
    
    /// Variance along specified axis
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ with means μ:
    /// - `Axis::Rows`: σ²ⱼ = (1/(m-1)) × Σᵢ(A[i,j] - μⱼ)² → vector ∈ ℝⁿ
    /// - `Axis::Cols`: σ²ᵢ = (1/(n-1)) × Σⱼ(A[i,j] - μᵢ)² → vector ∈ ℝᵐ
    /// Uses sample variance (Bessel's correction, ddof=1)
    /// 
    /// # Complexity
    /// - Time: O(m×n) for mean + O(m×n) for variance calculation
    /// - Space: O(max(m,n)) for result vector
    /// 
    /// # For AI Code Generation
    /// - `Axis::Rows`: Feature variances (measure of feature spread)
    /// - `Axis::Cols`: Sample variances (measure of sample diversity)
    /// - Equivalent to NumPy's `np.var(axis=0, ddof=1)` (sample variance)
    /// - Use sqrt() to get standard deviation
    /// - Common in feature selection: high variance features are more informative
    /// - Always non-negative values
    fn var_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + Zero + One + Clone + FromPrimitive;
}

impl<T: Entity + ComplexField> AxisReductions<T> for Array<T> {
    /// Sum along specified axis with AI-optimized implementation
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, reductions::{AxisReductions, Axis}};
    /// 
    /// // Feature aggregation (most common pattern)
    /// let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    /// let feature_totals = data.sum_axis(Axis::Rows).unwrap();  // [5.0, 7.0, 9.0]
    /// 
    /// // Sample aggregation
    /// let sample_totals = data.sum_axis(Axis::Cols).unwrap();   // [6.0, 15.0]
    /// 
    /// // Combined with broadcasting for analysis
    /// let total_sum = feature_totals.sum_elements();  // Grand total
    /// ```
    /// 
    /// # See Also
    /// - [`sum_axis_keepdims`]: Version that preserves matrix structure
    /// - [`mean_axis`]: Average values along axis
    /// - [`broadcasting`]: For normalizing against sums
    fn sum_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: Zero + One + Clone,
    {
        match axis {
            Axis::Rows => {
                // Sum along rows (axis 0) -> column sums
                let mut sums = vec![T::zero(); self.ncols()];
                for j in 0..self.ncols() {
                    for i in 0..self.nrows() {
                        sums[j] = sums[j].clone() + self.get(i, j).unwrap().clone();
                    }
                }
                Ok(Vector::from_slice(&sums))
            },
            Axis::Cols => {
                // Sum along columns (axis 1) -> row sums
                let mut sums = vec![T::zero(); self.nrows()];
                for i in 0..self.nrows() {
                    for j in 0..self.ncols() {
                        sums[i] = sums[i].clone() + self.get(i, j).unwrap().clone();
                    }
                }
                Ok(Vector::from_slice(&sums))
            }
        }
    }
    
    /// Sum along axis preserving dimensions for broadcasting
    fn sum_axis_keepdims(&self, axis: Axis) -> Result<Array<T>>
    where
        T: Zero + One + Clone,
    {
        let sums = self.sum_axis(axis)?;
        
        match axis {
            Axis::Rows => {
                // Column sums -> shape (1, ncols)
                let data: Vec<T> = (0..sums.len()).map(|i| sums.get(i).unwrap().clone()).collect();
                Array::from_slice(&data, 1, self.ncols())
            },
            Axis::Cols => {
                // Row sums -> shape (nrows, 1)
                let data: Vec<T> = (0..sums.len()).map(|i| sums.get(i).unwrap().clone()).collect();
                Array::from_slice(&data, self.nrows(), 1)
            }
        }
    }
    
    /// Mean along specified axis
    fn mean_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: Zero + One + Clone + FromPrimitive + std::ops::Div<Output = T>,
    {
        let sums = self.sum_axis(axis)?;
        let count = match axis {
            Axis::Rows => T::from_f64(self.nrows() as f64).unwrap(),
            Axis::Cols => T::from_f64(self.ncols() as f64).unwrap(),
        };
        
        let means: Vec<T> = (0..sums.len())
            .map(|i| sums.get(i).unwrap().clone() / count.clone())
            .collect();
        
        Ok(Vector::from_slice(&means))
    }
    
    /// Mean along axis with keepdims
    fn mean_axis_keepdims(&self, axis: Axis) -> Result<Array<T>>
    where
        T: Zero + One + Clone + FromPrimitive + std::ops::Div<Output = T>,
    {
        let means = self.mean_axis(axis)?;
        
        match axis {
            Axis::Rows => {
                // Column means -> shape (1, ncols)
                let data: Vec<T> = (0..means.len()).map(|i| means.get(i).unwrap().clone()).collect();
                Array::from_slice(&data, 1, self.ncols())
            },
            Axis::Cols => {
                // Row means -> shape (nrows, 1)
                let data: Vec<T> = (0..means.len()).map(|i| means.get(i).unwrap().clone()).collect();
                Array::from_slice(&data, self.nrows(), 1)
            }
        }
    }
    
    /// Minimum along specified axis
    fn min_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: PartialOrd,
    {
        match axis {
            Axis::Rows => {
                // Min along rows -> column mins
                let mut mins = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let mut col_min = self.get(0, j).unwrap().clone();
                    for i in 1..self.nrows() {
                        let val = self.get(i, j).unwrap().clone();
                        if val < col_min {
                            col_min = val;
                        }
                    }
                    mins.push(col_min);
                }
                Ok(Vector::from_slice(&mins))
            },
            Axis::Cols => {
                // Min along columns -> row mins
                let mut mins = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let mut row_min = self.get(i, 0).unwrap().clone();
                    for j in 1..self.ncols() {
                        let val = self.get(i, j).unwrap().clone();
                        if val < row_min {
                            row_min = val;
                        }
                    }
                    mins.push(row_min);
                }
                Ok(Vector::from_slice(&mins))
            }
        }
    }
    
    /// Maximum along specified axis
    fn max_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: PartialOrd,
    {
        match axis {
            Axis::Rows => {
                // Max along rows -> column maxes
                let mut maxs = Vec::with_capacity(self.ncols());
                for j in 0..self.ncols() {
                    let mut col_max = self.get(0, j).unwrap().clone();
                    for i in 1..self.nrows() {
                        let val = self.get(i, j).unwrap().clone();
                        if val > col_max {
                            col_max = val;
                        }
                    }
                    maxs.push(col_max);
                }
                Ok(Vector::from_slice(&maxs))
            },
            Axis::Cols => {
                // Max along columns -> row maxes
                let mut maxs = Vec::with_capacity(self.nrows());
                for i in 0..self.nrows() {
                    let mut row_max = self.get(i, 0).unwrap().clone();
                    for j in 1..self.ncols() {
                        let val = self.get(i, j).unwrap().clone();
                        if val > row_max {
                            row_max = val;
                        }
                    }
                    maxs.push(row_max);
                }
                Ok(Vector::from_slice(&maxs))
            }
        }
    }
    
    /// Variance along specified axis
    fn var_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + Zero + One + Clone + FromPrimitive,
    {
        let means = self.mean_axis(axis)?;
        let count = match axis {
            Axis::Rows => T::from_f64((self.nrows() - 1) as f64).unwrap(), // Bessel's correction
            Axis::Cols => T::from_f64((self.ncols() - 1) as f64).unwrap(),
        };
        
        let vars: Vec<T> = match axis {
            Axis::Rows => {
                // Variance along rows -> column variances
                (0..self.ncols()).map(|j| {
                    let mean = means.get(j).unwrap().clone();
                    let mut var_sum = T::zero();
                    for i in 0..self.nrows() {
                        let val = self.get(i, j).unwrap().clone();
                        let diff = val - mean.clone();
                        var_sum = var_sum + diff.clone() * diff;
                    }
                    var_sum / count.clone()
                }).collect()
            },
            Axis::Cols => {
                // Variance along columns -> row variances
                (0..self.nrows()).map(|i| {
                    let mean = means.get(i).unwrap().clone();
                    let mut var_sum = T::zero();
                    for j in 0..self.ncols() {
                        let val = self.get(i, j).unwrap().clone();
                        let diff = val - mean.clone();
                        var_sum = var_sum + diff.clone() * diff;
                    }
                    var_sum / count.clone()
                }).collect()
            }
        };
        
        Ok(Vector::from_slice(&vars))
    }
    
    /// Standard deviation along specified axis
    fn std_axis(&self, axis: Axis) -> Result<Vector<T>>
    where
        T: std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + Zero + One + Clone + FromPrimitive + Float,
    {
        let vars = self.var_axis(axis)?;
        let stds: Vec<T> = (0..vars.len())
            .map(|i| {
                let var = vars.get(i).unwrap().clone();
                // For now, we assume T has a sqrt method - this works for f64/f32
                // For complex types, this would need to be handled differently
                var.sqrt()
            })
            .collect();
        
        Ok(Vector::from_slice(&stds))
    }
}

// Convenience methods for common reductions with AI-optimized documentation
/// 
/// These methods provide NumPy-style axis specification using numeric indices
/// for compatibility with AI code generators familiar with Python conventions.
impl<T: Entity + ComplexField> Array<T> {
    /// Sum along columns (axis 0) - NumPy-style convenience method
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `sum_axis(Axis::Rows)`
    /// - Follows NumPy convention: axis=0 reduces along rows
    /// - Result length = number of columns (features)
    /// - Most common reduction in machine learning
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let col_sums = data.sum_axis_0().unwrap();  // [4.0, 6.0] (feature sums)
    /// ```
    pub fn sum_axis_0(&self) -> Result<Vector<T>>
    where
        T: Zero + One + Clone,
    {
        self.sum_axis(Axis::Rows)
    }
    
    /// Sum along rows (axis 1) - NumPy-style convenience method
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `sum_axis(Axis::Cols)`
    /// - Follows NumPy convention: axis=1 reduces along columns
    /// - Result length = number of rows (samples)
    /// - Less common than axis=0 in typical ML workflows
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let row_sums = data.sum_axis_1().unwrap();  // [3.0, 7.0] (sample sums)
    /// ```
    pub fn sum_axis_1(&self) -> Result<Vector<T>>
    where
        T: Zero + One + Clone,
    {
        self.sum_axis(Axis::Cols)
    }
    
    /// Mean along columns (axis 0) - Feature means for ML preprocessing
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `mean_axis(Axis::Rows)`
    /// - Produces feature means (most critical for ML normalization)
    /// - Use with broadcasting for mean centering: `data - mean_axis_0()`
    /// - Combined with std_axis_0 for z-score normalization
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let features = ArrayF64::from_slice(&[1.0, 4.0, 2.0, 6.0], 2, 2).unwrap();
    /// let feature_means = features.mean_axis_0().unwrap();  // [1.5, 5.0]
    /// 
    /// // Mean centering (critical preprocessing step)
    /// let centered = &features - &feature_means.broadcast_to_matrix(features.shape())?;
    /// ```
    pub fn mean_axis_0(&self) -> Result<Vector<T>>
    where
        T: Zero + One + Clone + FromPrimitive + std::ops::Div<Output = T>,
    {
        self.mean_axis(Axis::Rows)
    }
    
    /// Mean along rows (axis 1) - Sample means
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `mean_axis(Axis::Cols)`
    /// - Produces sample means (less common than feature means)
    /// - Result length = number of rows (samples)
    /// - Use for per-sample analysis or sample-wise normalization
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let samples = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 8.0, 9.0, 10.0], 2, 3).unwrap();
    /// let sample_means = samples.mean_axis_1().unwrap();  // [2.0, 9.0]
    /// ```
    pub fn mean_axis_1(&self) -> Result<Vector<T>>
    where
        T: Zero + One + Clone + FromPrimitive + std::ops::Div<Output = T>,
    {
        self.mean_axis(Axis::Cols)
    }
    
    /// Min along columns (axis 0) - Feature minimums
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `min_axis(Axis::Rows)`
    /// - Produces minimum value for each feature/column
    /// - Use for data range analysis and min-max normalization
    /// - Combined with max_axis_0 for range calculations
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let data = ArrayF64::from_slice(&[1.0, 5.0, 2.0, 3.0], 2, 2).unwrap();
    /// let feature_mins = data.min_axis_0().unwrap();  // [1.0, 3.0]
    /// let feature_maxs = data.max_axis_0().unwrap();  // [2.0, 5.0]
    /// 
    /// // Min-max normalization
    /// let range = &feature_maxs - &feature_mins;
    /// let normalized = (&data - &feature_mins) / &range;
    /// ```
    pub fn min_axis_0(&self) -> Result<Vector<T>>
    where
        T: PartialOrd + Clone,
    {
        self.min_axis(Axis::Rows)
    }
    
    /// Max along columns (axis 0) - Feature maximums
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `max_axis(Axis::Rows)`
    /// - Produces maximum value for each feature/column
    /// - Essential for min-max scaling and data range analysis
    /// - Often paired with min_axis_0 for normalization
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let features = ArrayF64::from_slice(&[0.0, 100.0, 50.0, 200.0], 2, 2).unwrap();
    /// let feature_maxs = features.max_axis_0().unwrap();  // [50.0, 200.0]
    /// 
    /// // Scale to [0,1] range
    /// let scaled = &features / &feature_maxs.broadcast_to_matrix(features.shape())?;
    /// ```
    pub fn max_axis_0(&self) -> Result<Vector<T>>
    where
        T: PartialOrd + Clone,
    {
        self.max_axis(Axis::Rows)
    }
    
    /// Variance along columns (axis 0) - Feature variances
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `var_axis(Axis::Rows)`
    /// - Produces variance for each feature/column (measure of spread)
    /// - Use for feature selection: higher variance often means more information
    /// - Combined with mean_axis_0 for statistical analysis
    /// - Use sqrt() to get standard deviations
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let features = ArrayF64::from_slice(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2).unwrap();
    /// let feature_vars = features.var_axis_0().unwrap();  // Feature variances
    /// let feature_stds = feature_vars.map(|x| x.sqrt());  // Feature std devs
    /// 
    /// // Identify high-variance features for selection
    /// let high_var_threshold = 5.0;
    /// ```
    pub fn var_axis_0(&self) -> Result<Vector<T>>
    where
        T: std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + Zero + One + Clone + FromPrimitive,
    {
        self.var_axis(Axis::Rows)
    }
    
    /// Standard deviation along columns (axis 0) - Feature standard deviations
    /// 
    /// # For AI Code Generation
    /// - Equivalent to `std_axis(Axis::Rows)`
    /// - Produces standard deviation for each feature/column
    /// - **Critical for z-score normalization**: `(data - mean) / std`
    /// - Most commonly used axis reduction in machine learning preprocessing
    /// - Combined with mean_axis_0 for standardization
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let features = ArrayF64::from_slice(&[1.0, 10.0, 3.0, 30.0, 5.0, 50.0], 3, 2).unwrap();
    /// let feature_means = features.mean_axis_0().unwrap();
    /// let feature_stds = features.std_axis_0().unwrap();
    /// 
    /// // Z-score normalization (standardization)
    /// let standardized = (&features - &feature_means.broadcast_to_matrix(features.shape())?) 
    ///                   / &feature_stds.broadcast_to_matrix(features.shape())?;
    /// // Result: mean=0, std=1 for each feature
    /// ```
    pub fn std_axis_0(&self) -> Result<Vector<T>>
    where
        T: std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + Zero + One + Clone + FromPrimitive + Float,
    {
        self.std_axis(Axis::Rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayF64, VectorF64};
    
    #[test]
    fn test_sum_axis() {
        let m = ArrayF64::from_slice(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ], 2, 3).unwrap();
        
        // Sum along rows (column sums)
        let col_sums = m.sum_axis(Axis::Rows).unwrap();
        assert_eq!(col_sums.len(), 3);
        assert_eq!(col_sums.get(0), Some(5.0)); // 1 + 4
        assert_eq!(col_sums.get(1), Some(7.0)); // 2 + 5
        assert_eq!(col_sums.get(2), Some(9.0)); // 3 + 6
        
        // Sum along columns (row sums)
        let row_sums = m.sum_axis(Axis::Cols).unwrap();
        assert_eq!(row_sums.len(), 2);
        assert_eq!(row_sums.get(0), Some(6.0));  // 1 + 2 + 3
        assert_eq!(row_sums.get(1), Some(15.0)); // 4 + 5 + 6
    }
    
    #[test]
    fn test_mean_axis() {
        let m = ArrayF64::from_slice(&[
            2.0, 4.0,
            6.0, 8.0
        ], 2, 2).unwrap();
        
        // Mean along rows (column means)
        let col_means = m.mean_axis(Axis::Rows).unwrap();
        assert_eq!(col_means.get(0), Some(4.0)); // (2 + 6) / 2
        assert_eq!(col_means.get(1), Some(6.0)); // (4 + 8) / 2
        
        // Mean along columns (row means)
        let row_means = m.mean_axis(Axis::Cols).unwrap();
        assert_eq!(row_means.get(0), Some(3.0)); // (2 + 4) / 2
        assert_eq!(row_means.get(1), Some(7.0)); // (6 + 8) / 2
    }
    
    #[test]
    fn test_keepdims() {
        let m = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        
        // Sum with keepdims
        let col_sums = m.sum_axis_keepdims(Axis::Rows).unwrap();
        assert_eq!(col_sums.shape(), (1, 2));
        
        let row_sums = m.sum_axis_keepdims(Axis::Cols).unwrap();
        assert_eq!(row_sums.shape(), (2, 1));
    }
    
    #[test]
    fn test_min_max_axis() {
        let m = ArrayF64::from_slice(&[
            1.0, 5.0, 2.0,
            4.0, 2.0, 6.0
        ], 2, 3).unwrap();
        
        // Min along rows (column mins)
        let col_mins = m.min_axis(Axis::Rows).unwrap();
        assert_eq!(col_mins.get(0), Some(1.0)); // min(1, 4)
        assert_eq!(col_mins.get(1), Some(2.0)); // min(5, 2)
        assert_eq!(col_mins.get(2), Some(2.0)); // min(2, 6)
        
        // Max along columns (row maxes)
        let row_maxs = m.max_axis(Axis::Cols).unwrap();
        assert_eq!(row_maxs.get(0), Some(5.0)); // max(1, 5, 2)
        assert_eq!(row_maxs.get(1), Some(6.0)); // max(4, 2, 6)
    }
    
    #[test]
    fn test_convenience_methods() {
        let m = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        
        let col_sums = m.sum_axis_0().unwrap();
        assert_eq!(col_sums.get(0), Some(4.0)); // 1 + 3
        assert_eq!(col_sums.get(1), Some(6.0)); // 2 + 4
        
        let col_means = m.mean_axis_0().unwrap();
        assert_eq!(col_means.get(0), Some(2.0)); // (1 + 3) / 2
        assert_eq!(col_means.get(1), Some(3.0)); // (2 + 4) / 2
    }
}