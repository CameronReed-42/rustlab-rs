//! Array and Vector creation utilities with AI-optimized documentation
//!
//! This module provides NumPy-style creation functions for arrays and vectors.
//! All functions use zero-cost abstractions and integrate seamlessly with RustLab's
//! mathematical operators (^ for matrix multiplication, * for element-wise).
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::creation::*;
//! 
//! // Create and initialize arrays
//! let A = zeros(3, 3);              // 3×3 zero matrix
//! let v = linspace(0.0, 1.0, 100);   // 100 points from 0 to 1
//! let I = eye(5);                    // 5×5 identity matrix
//! 
//! // Use in calculations
//! let result = A ^ v;                // Matrix-vector multiplication
//! ```

use crate::{Array, Vector};
use faer_entity::Entity;
use faer_traits::ComplexField;

/// Create 2D array filled with zeros
/// 
/// # Mathematical Specification
/// Creates matrix A ∈ ℝᵐˣⁿ or ℂᵐˣⁿ where Aᵢⱼ = 0 for all i, j
/// 
/// # Dimensions
/// - Input: rows (m), cols (n) where m, n ≥ 0
/// - Output: Array<T> of shape (m × n)
/// 
/// # Complexity
/// - Time: O(m × n) initialization
/// - Space: O(m × n) allocation
/// 
/// # For AI Code Generation
/// - Generic function works with f64, f32, Complex<f64>, Complex<f32>
/// - Dimensions can be 0 (creates empty matrix)
/// - All elements initialized to exact zero
/// - Common uses: matrix initialization, workspace allocation, accumulator
/// - Equivalent to `Array::zeros(rows, cols)` but more convenient for imports
/// 
/// # Example
/// ```rust
/// use rustlab_math::creation::zeros;
/// 
/// let A = zeros::<f64>(3, 4);  // 3×4 matrix of zeros
/// assert_eq!(A.shape(), (3, 4));
/// assert_eq!(A.get(0, 0), Some(0.0));
/// 
/// // Type inference often works
/// let B = zeros(2, 2);  // Inferred from context
/// let result = B + 1.0; // Now B is known to be ArrayF64
/// ```
/// 
/// # See Also
/// - [`ones`]: Create matrix filled with ones
/// - [`eye`]: Create identity matrix
/// - [`Array::zeros`]: Method version on Array type
/// - [`zeros_vec`]: Create 1D zero vector
pub fn zeros<T: Entity + ComplexField>(rows: usize, cols: usize) -> Array<T> {
    Array::zeros(rows, cols)
}

/// Create 2D array filled with ones
pub fn ones<T: Entity + ComplexField + num_traits::One>(rows: usize, cols: usize) -> Array<T> {
    Array::ones(rows, cols)
}

/// Create array with same shape as another, filled with zeros
pub fn zeros_like<T: Entity + ComplexField>(other: &Array<T>) -> Array<T> {
    let (rows, cols) = other.shape();
    zeros(rows, cols)
}

/// Create array with same shape as another, filled with ones  
pub fn ones_like<T: Entity + ComplexField + num_traits::One>(other: &Array<T>) -> Array<T> {
    let (rows, cols) = other.shape();
    ones(rows, cols)
}

/// Create 1D vector filled with zeros
pub fn zeros_vec<T: Entity + ComplexField>(size: usize) -> Vector<T> {
    Vector::zeros(size)
}

/// Create 1D vector filled with ones
pub fn ones_vec<T: Entity + ComplexField + num_traits::One>(size: usize) -> Vector<T> {
    Vector::ones(size)
}

/// Create vector with same length as another, filled with zeros
pub fn zeros_like_vec<T: Entity + ComplexField>(other: &Vector<T>) -> Vector<T> {
    zeros_vec(other.len())
}

/// Create vector with same length as another, filled with ones
pub fn ones_like_vec<T: Entity + ComplexField + num_traits::One>(other: &Vector<T>) -> Vector<T> {
    ones_vec(other.len())
}

/// Create 1D vector filled with a constant value
pub fn fill_vec<T: Entity + ComplexField + Clone>(size: usize, value: T) -> Vector<T> {
    Vector::fill(size, value)
}

/// Create 2D array filled with a constant value
pub fn fill<T: Entity + ComplexField + Clone>(rows: usize, cols: usize, value: T) -> Array<T> {
    Array::fill(rows, cols, value)
}

/// Create identity matrix
/// 
/// # Mathematical Specification
/// Creates matrix I ∈ ℝⁿˣⁿ or ℂⁿˣⁿ where:
/// Iᵢⱼ = 1 if i = j, 0 otherwise
/// The multiplicative identity for matrix operations
/// 
/// # Dimensions
/// - Input: size (n) where n ≥ 0
/// - Output: Array<T> of shape (n × n)
/// 
/// # Complexity
/// - Time: O(n²) initialization
/// - Space: O(n²) allocation
/// 
/// # For AI Code Generation
/// - Square matrix only (n × n)
/// - Main diagonal filled with ones, rest zeros
/// - Matrix multiplication identity: A ^ I = I ^ A = A
/// - Common uses: initialization, matrix inversion, linear systems
/// - Generic over numeric types (f64, f32, Complex)
/// - Size 0 creates empty matrix
/// 
/// # Example
/// ```rust
/// use rustlab_math::creation::eye;
/// 
/// let I = eye::<f64>(3);  // 3×3 identity matrix
/// assert_eq!(I.get(0, 0), Some(1.0));  // Diagonal = 1
/// assert_eq!(I.get(0, 1), Some(0.0));  // Off-diagonal = 0
/// 
/// // Identity property: A * I = A
/// let A = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
/// let I2 = eye::<f64>(2);
/// let result = A ^ I2;  // Should equal A
/// 
/// // Linear system solving initialization
/// let mut solver_matrix = eye::<f64>(n);
/// // Modify solver_matrix for specific problem...
/// ```
/// 
/// # See Also
/// - [`zeros`]: Create zero-filled matrix
/// - [`ones`]: Create ones-filled matrix
/// - [`Array::eye`]: Type-specific method versions
/// - Linear algebra solve operations: See `rustlab-linearalgebra` crate
pub fn eye<T: Entity + ComplexField + num_traits::One>(size: usize) -> Array<T> {
    let mut mat = zeros(size, size);
    for i in 0..size {
        mat.set(i, i, T::one()).unwrap();
    }
    mat
}

/// Create linearly spaced vector
/// 
/// # Mathematical Specification
/// Creates vector v ∈ ℝⁿ where:
/// vᵢ = start + i × (stop - start)/(n-1) for i = 0..n-1
/// Evenly spaced points from start to stop (inclusive)
/// 
/// # Dimensions
/// - Input: start, stop (scalars), num (count ≥ 0)
/// - Output: Vector<f64> of length num
/// 
/// # Complexity
/// - Time: O(n) generation
/// - Space: O(n) allocation
/// 
/// # For AI Code Generation
/// - Both endpoints are included: linspace(0, 1, 3) = [0, 0.5, 1]
/// - num=0 creates empty vector, num=1 creates [start]
/// - Always returns f64 vector (most common for numerical work)
/// - Common uses: time series, plotting grids, sampling ranges
/// - Equivalent to NumPy's linspace function
/// - Step size is automatically calculated
/// 
/// # Example
/// ```rust
/// use rustlab_math::creation::linspace;
/// 
/// // Create 5 points from 0 to 1
/// let points = linspace(0.0, 1.0, 5);
/// // Result: [0.0, 0.25, 0.5, 0.75, 1.0]
/// 
/// // Time series for signal processing
/// let t = linspace(0.0, 2.0, 1000);  // 1000 time points
/// let signal = t.sin();               // sin(t) for each point
/// 
/// // Mesh grid for plotting
/// let x_axis = linspace(-5.0, 5.0, 100);
/// let y_values = x_axis.square() * 2.0; // y = 2x²
/// ```
/// 
/// # See Also
/// - [`arange`]: Integer sequence vector
/// - [`arange_step`]: Custom step size
/// - [`zeros_vec`]: Create zero-filled vector
/// - [`ones_vec`]: Create ones-filled vector
pub fn linspace(start: f64, stop: f64, num: usize) -> Vector<f64> {
    if num == 0 {
        return Vector::zeros(0);
    }
    if num == 1 {
        return Vector::from_slice(&[start]);
    }
    
    let step = (stop - start) / (num - 1) as f64;
    let data: Vec<f64> = (0..num)
        .map(|i| start + i as f64 * step)
        .collect();
    
    Vector::from_slice(&data)
}

/// Create vector with values from 0 to n-1
/// 
/// # Mathematical Specification
/// Creates vector v ∈ ℝⁿ where:
/// vᵢ = i for i = 0, 1, 2, ..., n-1
/// Sequential integer values as floats
/// 
/// # Dimensions
/// - Input: n (count ≥ 0)
/// - Output: Vector<f64> of length n
/// 
/// # Complexity
/// - Time: O(n) generation
/// - Space: O(n) allocation
/// 
/// # For AI Code Generation
/// - Always starts from 0, ends at n-1
/// - Returns f64 vector for numerical compatibility
/// - Common uses: indexing, enumeration, sequence generation
/// - Equivalent to Python's range(n) or NumPy's arange(n)
/// - Zero-based indexing (first element is 0)
/// 
/// # Example
/// ```rust
/// use rustlab_math::creation::arange;
/// 
/// let indices = arange(5);
/// // Result: [0.0, 1.0, 2.0, 3.0, 4.0]
/// 
/// // Use for indexing or enumeration
/// let weights = indices / 10.0;  // [0.0, 0.1, 0.2, 0.3, 0.4]
/// 
/// // Polynomial terms: x⁰, x¹, x², ...
/// let x = 2.0;
/// let powers = arange(6);         // [0, 1, 2, 3, 4, 5]
/// let terms = powers.exp();       // [e⁰, e¹, e², e³, e⁴, e⁵]
/// ```
/// 
/// # See Also
/// - [`arange_step`]: Custom start, stop, and step
/// - [`linspace`]: Evenly spaced points between endpoints
/// - [`fill_vec`]: Create vector with constant value
pub fn arange(n: usize) -> Vector<f64> {
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    Vector::from_slice(&data)
}

/// Create vector with evenly spaced values in range
pub fn arange_step(start: f64, stop: f64, step: f64) -> Vector<f64> {
    let mut data = Vec::new();
    let mut current = start;
    
    while current < stop {
        data.push(current);
        current += step;
    }
    
    Vector::from_slice(&data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayF64, VectorF64};
    use approx::assert_relative_eq;
    use num_complex::Complex;

    #[test]
    fn test_zeros_function() {
        let arr = zeros::<f64>(2, 3);
        assert_eq!(arr.shape(), (2, 3));
        
        // Check all elements are zero
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(arr.get(i, j), Some(0.0));
            }
        }
        
        // Test with different sizes
        let small = zeros::<f64>(1, 1);
        assert_eq!(small.shape(), (1, 1));
        assert_eq!(small.get(0, 0), Some(0.0));
        
        // Test empty matrix
        let empty = zeros::<f64>(0, 0);
        assert_eq!(empty.shape(), (0, 0));
    }

    #[test]
    fn test_ones_function() {
        let arr = ones::<f64>(2, 2);
        assert_eq!(arr.shape(), (2, 2));
        
        // Check all elements are one
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(arr.get(i, j), Some(1.0));
            }
        }
        
        // Test with f32
        let arr_f32 = ones::<f32>(3, 1);
        assert_eq!(arr_f32.shape(), (3, 1));
        assert_eq!(arr_f32.get(0, 0), Some(1.0f32));
        assert_eq!(arr_f32.get(1, 0), Some(1.0f32));
        assert_eq!(arr_f32.get(2, 0), Some(1.0f32));
    }

    #[test]
    fn test_zeros_like() {
        let original = ArrayF64::ones(3, 4);
        let zeros_copy = zeros_like(&original);
        
        assert_eq!(zeros_copy.shape(), original.shape());
        
        // Check all elements are zero
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(zeros_copy.get(i, j), Some(0.0));
            }
        }
    }

    #[test]
    fn test_ones_like() {
        let original = ArrayF64::zeros(2, 5);
        let ones_copy = ones_like(&original);
        
        assert_eq!(ones_copy.shape(), original.shape());
        
        // Check all elements are one
        for i in 0..2 {
            for j in 0..5 {
                assert_eq!(ones_copy.get(i, j), Some(1.0));
            }
        }
    }

    #[test]
    fn test_zeros_vec() {
        let vec = zeros_vec::<f64>(5);
        assert_eq!(vec.len(), 5);
        
        // Check all elements are zero
        for i in 0..5 {
            assert_eq!(vec.get(i), Some(0.0));
        }
        
        // Test empty vector
        let empty_vec = zeros_vec::<f64>(0);
        assert_eq!(empty_vec.len(), 0);
        assert!(empty_vec.is_empty());
    }

    #[test]
    fn test_ones_vec() {
        let vec = ones_vec::<f64>(4);
        assert_eq!(vec.len(), 4);
        
        // Check all elements are one
        for i in 0..4 {
            assert_eq!(vec.get(i), Some(1.0));
        }
        
        // Test with complex numbers
        let complex_vec = ones_vec::<Complex<f64>>(2);
        assert_eq!(complex_vec.len(), 2);
        assert_eq!(complex_vec.get(0), Some(Complex::new(1.0, 0.0)));
        assert_eq!(complex_vec.get(1), Some(Complex::new(1.0, 0.0)));
    }

    #[test]
    fn test_zeros_like_vec() {
        let original = VectorF64::ones(3);
        let zeros_copy = zeros_like_vec(&original);
        
        assert_eq!(zeros_copy.len(), original.len());
        
        // Check all elements are zero
        for i in 0..3 {
            assert_eq!(zeros_copy.get(i), Some(0.0));
        }
    }

    #[test]
    fn test_ones_like_vec() {
        let original = VectorF64::zeros(6);
        let ones_copy = ones_like_vec(&original);
        
        assert_eq!(ones_copy.len(), original.len());
        
        // Check all elements are one
        for i in 0..6 {
            assert_eq!(ones_copy.get(i), Some(1.0));
        }
    }

    #[test]
    fn test_fill_vec() {
        let vec = fill_vec(4, 3.14);
        assert_eq!(vec.len(), 4);
        
        // Check all elements are the fill value
        for i in 0..4 {
            assert_eq!(vec.get(i), Some(3.14));
        }
        
        // Test with negative values
        let neg_vec = fill_vec(3, -1.5);
        for i in 0..3 {
            assert_eq!(neg_vec.get(i), Some(-1.5));
        }
    }

    #[test]
    fn test_fill() {
        let arr = fill(3, 2, 2.718);
        assert_eq!(arr.shape(), (3, 2));
        
        // Check all elements are the fill value
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(arr.get(i, j), Some(2.718));
            }
        }
        
        // Test with negative values
        let neg_arr = fill(2, 2, -5.5);
        assert_eq!(neg_arr.get(0, 0), Some(-5.5));
        assert_eq!(neg_arr.get(1, 1), Some(-5.5));
    }

    #[test]
    fn test_eye() {
        let identity = eye::<f64>(3);
        assert_eq!(identity.shape(), (3, 3));
        
        // Check diagonal elements are 1
        for i in 0..3 {
            assert_eq!(identity.get(i, i), Some(1.0));
        }
        
        // Check off-diagonal elements are 0
        assert_eq!(identity.get(0, 1), Some(0.0));
        assert_eq!(identity.get(0, 2), Some(0.0));
        assert_eq!(identity.get(1, 0), Some(0.0));
        assert_eq!(identity.get(1, 2), Some(0.0));
        assert_eq!(identity.get(2, 0), Some(0.0));
        assert_eq!(identity.get(2, 1), Some(0.0));
        
        // Test with different size
        let small_identity = eye::<f64>(1);
        assert_eq!(small_identity.shape(), (1, 1));
        assert_eq!(small_identity.get(0, 0), Some(1.0));
        
        // Test empty identity
        let empty_identity = eye::<f64>(0);
        assert_eq!(empty_identity.shape(), (0, 0));
    }

    #[test]
    fn test_linspace() {
        // Test basic linspace
        let points = linspace(0.0, 1.0, 5);
        assert_eq!(points.len(), 5);
        
        // Check values
        assert_eq!(points.get(0), Some(0.0));
        assert_eq!(points.get(1), Some(0.25));
        assert_eq!(points.get(2), Some(0.5));
        assert_eq!(points.get(3), Some(0.75));
        assert_eq!(points.get(4), Some(1.0));
        
        // Test with different range
        let neg_points = linspace(-1.0, 1.0, 3);
        assert_eq!(neg_points.len(), 3);
        assert_eq!(neg_points.get(0), Some(-1.0));
        assert_eq!(neg_points.get(1), Some(0.0));
        assert_eq!(neg_points.get(2), Some(1.0));
        
        // Test edge cases
        let empty = linspace(0.0, 1.0, 0);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        
        let single = linspace(5.0, 10.0, 1);
        assert_eq!(single.len(), 1);
        assert_eq!(single.get(0), Some(5.0)); // Should return start value
        
        // Test with large number of points
        let large = linspace(0.0, 100.0, 101);
        assert_eq!(large.len(), 101);
        assert_eq!(large.get(0), Some(0.0));
        assert_eq!(large.get(100), Some(100.0));
        assert_relative_eq!(large.get(50).unwrap(), 50.0, epsilon = 1e-14);
    }

    #[test]
    fn test_arange() {
        // Test basic arange
        let sequence = arange(5);
        assert_eq!(sequence.len(), 5);
        
        // Check values
        assert_eq!(sequence.get(0), Some(0.0));
        assert_eq!(sequence.get(1), Some(1.0));
        assert_eq!(sequence.get(2), Some(2.0));
        assert_eq!(sequence.get(3), Some(3.0));
        assert_eq!(sequence.get(4), Some(4.0));
        
        // Test empty range
        let empty = arange(0);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        
        // Test single element
        let single = arange(1);
        assert_eq!(single.len(), 1);
        assert_eq!(single.get(0), Some(0.0));
        
        // Test large range
        let large = arange(100);
        assert_eq!(large.len(), 100);
        assert_eq!(large.get(0), Some(0.0));
        assert_eq!(large.get(99), Some(99.0));
    }

    #[test]
    fn test_arange_step() {
        // Test basic step
        let stepped = arange_step(0.0, 5.0, 1.0);
        assert_eq!(stepped.len(), 5); // 0, 1, 2, 3, 4
        assert_eq!(stepped.get(0), Some(0.0));
        assert_eq!(stepped.get(1), Some(1.0));
        assert_eq!(stepped.get(4), Some(4.0));
        
        // Test fractional step
        let fractional = arange_step(0.0, 1.0, 0.2);
        assert_eq!(fractional.len(), 5); // 0, 0.2, 0.4, 0.6, 0.8
        assert_relative_eq!(fractional.get(0).unwrap(), 0.0, epsilon = 1e-15);
        assert_relative_eq!(fractional.get(1).unwrap(), 0.2, epsilon = 1e-15);
        assert_relative_eq!(fractional.get(4).unwrap(), 0.8, epsilon = 1e-15);
        
        // Test larger step
        let large_step = arange_step(0.0, 10.0, 3.0);
        assert_eq!(large_step.len(), 4); // 0, 3, 6, 9
        assert_eq!(large_step.get(0), Some(0.0));
        assert_eq!(large_step.get(1), Some(3.0));
        assert_eq!(large_step.get(2), Some(6.0));
        assert_eq!(large_step.get(3), Some(9.0));
        
        // Test negative range
        let negative = arange_step(-2.0, 1.0, 1.0);
        assert_eq!(negative.len(), 3); // -2, -1, 0
        assert_eq!(negative.get(0), Some(-2.0));
        assert_eq!(negative.get(1), Some(-1.0));
        assert_eq!(negative.get(2), Some(0.0));
        
        // Test step larger than range
        let large_step_small_range = arange_step(0.0, 0.5, 1.0);
        assert_eq!(large_step_small_range.len(), 1); // Only 0
        assert_eq!(large_step_small_range.get(0), Some(0.0));
    }

    #[test]
    fn test_complex_number_support() {
        // Test zeros with complex numbers
        let complex_zeros = zeros::<Complex<f64>>(2, 2);
        assert_eq!(complex_zeros.get(0, 0), Some(Complex::new(0.0, 0.0)));
        assert_eq!(complex_zeros.get(1, 1), Some(Complex::new(0.0, 0.0)));
        
        // Test ones with complex numbers
        let complex_ones = ones::<Complex<f64>>(2, 2);
        assert_eq!(complex_ones.get(0, 0), Some(Complex::new(1.0, 0.0)));
        assert_eq!(complex_ones.get(1, 1), Some(Complex::new(1.0, 0.0)));
        
        // Test identity with complex numbers
        let complex_eye = eye::<Complex<f64>>(2);
        assert_eq!(complex_eye.get(0, 0), Some(Complex::new(1.0, 0.0)));
        assert_eq!(complex_eye.get(0, 1), Some(Complex::new(0.0, 0.0)));
        assert_eq!(complex_eye.get(1, 0), Some(Complex::new(0.0, 0.0)));
        assert_eq!(complex_eye.get(1, 1), Some(Complex::new(1.0, 0.0)));
    }

    #[test]
    fn test_generic_types() {
        // Test with f32
        let f32_zeros = zeros::<f32>(2, 2);
        assert_eq!(f32_zeros.get(0, 0), Some(0.0f32));
        
        let f32_ones = ones::<f32>(2, 2);
        assert_eq!(f32_ones.get(0, 0), Some(1.0f32));
        
        let f32_eye = eye::<f32>(2);
        assert_eq!(f32_eye.get(0, 0), Some(1.0f32));
        assert_eq!(f32_eye.get(0, 1), Some(0.0f32));
        
        // Test vectors with f32
        let f32_vec_zeros = zeros_vec::<f32>(3);
        assert_eq!(f32_vec_zeros.get(0), Some(0.0f32));
        
        let f32_vec_ones = ones_vec::<f32>(3);
        assert_eq!(f32_vec_ones.get(0), Some(1.0f32));
    }

    #[test]
    fn test_integration_with_arrays() {
        // Test that created arrays work with mathematical operations
        let zeros_arr = zeros::<f64>(2, 2);
        let ones_arr = ones::<f64>(2, 2);
        
        // Test addition
        let sum = &zeros_arr + &ones_arr;
        assert_eq!(sum.get(0, 0), Some(1.0));
        assert_eq!(sum.get(1, 1), Some(1.0));
        
        // Test with identity matrix
        let identity = eye::<f64>(2);
        let result = &ones_arr ^ &identity; // Matrix multiplication
        assert_eq!(result.get(0, 0), Some(1.0));
        assert_eq!(result.get(1, 1), Some(1.0));
    }

    #[test]
    fn test_edge_cases() {
        // Test zero-sized arrays and vectors
        let zero_arr = zeros::<f64>(0, 5);
        assert_eq!(zero_arr.shape(), (0, 5));
        
        let zero_vec = zeros_vec::<f64>(0);
        assert_eq!(zero_vec.len(), 0);
        
        // Test very small arrays
        let tiny_arr = ones::<f64>(1, 1);
        assert_eq!(tiny_arr.shape(), (1, 1));
        assert_eq!(tiny_arr.get(0, 0), Some(1.0));
        
        // Test linspace edge cases
        let reversed = linspace(1.0, 0.0, 3);
        assert_eq!(reversed.get(0), Some(1.0));
        assert_eq!(reversed.get(1), Some(0.5));
        assert_eq!(reversed.get(2), Some(0.0));
    }

    #[test]
    fn test_mathematical_properties() {
        // Test identity matrix properties
        let identity = eye::<f64>(3);
        let test_matrix = ones::<f64>(3, 3);
        
        // I * A = A
        let result = &identity ^ &test_matrix;
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(result.get(i, j).unwrap(), test_matrix.get(i, j).unwrap(), epsilon = 1e-15);
            }
        }
        
        // Test linspace mathematical properties
        let points = linspace(0.0, 10.0, 11);
        let step = points.get(1).unwrap() - points.get(0).unwrap();
        
        // Check uniform spacing
        for i in 1..points.len() {
            let current_step = points.get(i).unwrap() - points.get(i-1).unwrap();
            assert_relative_eq!(current_step, step, epsilon = 1e-14);
        }
    }
}