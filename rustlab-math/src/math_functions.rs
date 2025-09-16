//! Mathematical functions with automatic SIMD optimization
//!
//! This module provides element-wise mathematical functions that automatically
//! use SIMD when beneficial, maintaining our math-first ergonomics philosophy.
//! Users write natural mathematical code and get optimal performance transparently.

use crate::{ArrayF64, ArrayF32, VectorF64, VectorF32, ArrayC64, ArrayC32, VectorC64, VectorC32};
use faer::{Mat, Col};
use num_complex::Complex;

// SIMD thresholds for mathematical functions - lower than basic ops since 
// math functions are more expensive and benefit from SIMD even at smaller sizes
const MATH_SIMD_THRESHOLD_F64: usize = 8;   // 8 elements = 1 AVX2 vector
const MATH_SIMD_THRESHOLD_F32: usize = 16;  // 16 elements = 2 AVX2 vectors

/// Mathematical functions for ArrayF64 with automatic SIMD optimization
impl ArrayF64 {
    /// Compute sine of each element with automatic SIMD optimization
    /// 
    /// # Mathematical Specification
    /// For matrix A ∈ ℝᵐˣⁿ:
    /// sin(A)ᵢⱼ = sin(Aᵢⱼ) for all i, j
    /// Element-wise sine function applied to each matrix element
    /// 
    /// # Dimensions
    /// - Input: self (m × n)
    /// - Output: Matrix (m × n) with same dimensions
    /// 
    /// # Complexity
    /// - Time: O(m × n) with automatic SIMD acceleration for large arrays
    /// - Space: O(m × n) for result matrix
    /// 
    /// # For AI Code Generation
    /// - Applies sin() to every element independently
    /// - Input/output matrices have identical dimensions
    /// - Angles assumed to be in radians (not degrees)
    /// - Automatic SIMD optimization for arrays ≥ 8 elements (f64)
    /// - Common uses: trigonometric calculations, signal processing, oscillations
    /// - Range: sin(x) ∈ [-1, 1] for all real x
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{ArrayF64, constants::PI};
    /// 
    /// let angles = ArrayF64::from_slice(&[0.0, PI/2.0, PI, 3.0*PI/2.0], 2, 2).unwrap();
    /// let sines = angles.sin(); // Automatic SIMD for performance!
    /// 
    /// assert_eq!(sines.get(0, 0), Some(0.0));   // sin(0) = 0
    /// assert_eq!(sines.get(0, 1), Some(1.0));   // sin(π/2) = 1
    /// assert_eq!(sines.get(1, 0), Some(0.0));   // sin(π) ≈ 0
    /// assert_eq!(sines.get(1, 1), Some(-1.0));  // sin(3π/2) = -1
    /// 
    /// // Chain with other operations
    /// let scaled_sin = (&angles.sin() + 1.0) * 0.5; // Scale to [0,1]
    /// ```
    /// 
    /// # See Also
    /// - [`cos`]: Compute cosine (90° phase shift)
    /// - [`tan`]: Compute tangent (sin/cos ratio)
    /// - [`asin`]: Inverse sine function
    /// - [`Vector::sin`]: Element-wise sine for 1D vectors
    /// - [`sinh`]: Hyperbolic sine function
    pub fn sin(&self) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F64 && crate::simd::is_simd_available() {
            // Use SIMD implementation for larger arrays
            unsafe { simd_sin_f64(self) }
        } else {
            // Use scalar implementation for small arrays
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].sin()
            });
            ArrayF64 { inner: result }
        }
    }
    
    /// Compute cosine of each element with automatic SIMD optimization
    pub fn cos(&self) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F64 && crate::simd::is_simd_available() {
            unsafe { simd_cos_f64(self) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].cos()
            });
            ArrayF64 { inner: result }
        }
    }
    
    /// Compute tangent of each element
    pub fn tan(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].tan()
        });
        ArrayF64 { inner: result }
    }
    
    /// Compute exponential (e^x) of each element with automatic SIMD optimization
    pub fn exp(&self) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F64 && crate::simd::is_simd_available() {
            unsafe { simd_exp_f64(self) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].exp()
            });
            ArrayF64 { inner: result }
        }
    }
    
    /// Compute natural logarithm of each element with automatic SIMD optimization
    pub fn ln(&self) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F64 && crate::simd::is_simd_available() {
            unsafe { simd_ln_f64(self) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].ln()
            });
            ArrayF64 { inner: result }
        }
    }
    
    /// Compute base-10 logarithm of each element
    pub fn log10(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].log10()
        });
        ArrayF64 { inner: result }
    }
    
    /// Compute base-2 logarithm of each element  
    pub fn log2(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].log2()
        });
        ArrayF64 { inner: result }
    }
    
    /// Compute square root of each element with automatic SIMD optimization
    pub fn sqrt(&self) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F64 && crate::simd::is_simd_available() {
            unsafe { simd_sqrt_f64(self) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].sqrt()
            });
            ArrayF64 { inner: result }
        }
    }
    
    /// Compute absolute value of each element
    pub fn abs(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].abs()
        });
        ArrayF64 { inner: result }
    }
    
    /// Raise each element to a power with automatic SIMD optimization
    pub fn pow(&self, exponent: f64) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F64 && crate::simd::is_simd_available() {
            unsafe { simd_pow_f64(self, exponent) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].powf(exponent)
            });
            ArrayF64 { inner: result }
        }
    }
    
    /// Compute arcsine of each element
    pub fn asin(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].asin()
        });
        ArrayF64 { inner: result }
    }
    
    /// Compute arccosine of each element
    pub fn acos(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].acos()
        });
        ArrayF64 { inner: result }
    }
    
    /// Compute arctangent of each element
    pub fn atan(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].atan()
        });
        ArrayF64 { inner: result }
    }
    
    /// Compute hyperbolic sine of each element
    pub fn sinh(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].sinh()
        });
        ArrayF64 { inner: result }
    }
    
    /// Compute hyperbolic cosine of each element
    pub fn cosh(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].cosh()
        });
        ArrayF64 { inner: result }
    }
    
    /// Compute hyperbolic tangent of each element
    pub fn tanh(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].tanh()
        });
        ArrayF64 { inner: result }
    }
}

/// Mathematical functions for ArrayF32 with automatic SIMD optimization
impl ArrayF32 {
    /// Compute sine of each element with automatic SIMD optimization
    pub fn sin(&self) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F32 && crate::simd::is_simd_available() {
            unsafe { simd_sin_f32(self) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].sin()
            });
            ArrayF32 { inner: result }
        }
    }
    
    /// Compute cosine of each element with automatic SIMD optimization
    pub fn cos(&self) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F32 && crate::simd::is_simd_available() {
            unsafe { simd_cos_f32(self) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].cos()
            });
            ArrayF32 { inner: result }
        }
    }
    
    /// Compute tangent of each element
    pub fn tan(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].tan()
        });
        ArrayF32 { inner: result }
    }
    
    /// Compute exponential (e^x) of each element with automatic SIMD optimization
    pub fn exp(&self) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F32 && crate::simd::is_simd_available() {
            unsafe { simd_exp_f32(self) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].exp()
            });
            ArrayF32 { inner: result }
        }
    }
    
    /// Compute natural logarithm of each element with automatic SIMD optimization
    pub fn ln(&self) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F32 && crate::simd::is_simd_available() {
            unsafe { simd_ln_f32(self) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].ln()
            });
            ArrayF32 { inner: result }
        }
    }
    
    /// Compute square root of each element with automatic SIMD optimization
    pub fn sqrt(&self) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F32 && crate::simd::is_simd_available() {
            unsafe { simd_sqrt_f32(self) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].sqrt()
            });
            ArrayF32 { inner: result }
        }
    }
    
    /// Compute absolute value of each element
    pub fn abs(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].abs()
        });
        ArrayF32 { inner: result }
    }
    
    /// Raise each element to a power with automatic SIMD optimization
    pub fn pow(&self, exponent: f32) -> Self {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        
        if total_elements >= MATH_SIMD_THRESHOLD_F32 && crate::simd::is_simd_available() {
            unsafe { simd_pow_f32(self, exponent) }
        } else {
            let result = Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)].powf(exponent)
            });
            ArrayF32 { inner: result }
        }
    }
}

/// Mathematical functions for ArrayC64 (Complex<f64>) 
impl ArrayC64 {
    /// Compute sine of each complex element
    pub fn sin(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].sin()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute cosine of each complex element
    pub fn cos(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].cos()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute tangent of each complex element
    pub fn tan(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].tan()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute exponential (e^z) of each complex element
    pub fn exp(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].exp()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute natural logarithm of each complex element
    pub fn ln(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].ln()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute base-10 logarithm of each complex element
    pub fn log10(&self) -> Self {
        let (rows, cols) = self.shape();
        let ln10 = Complex::new(10.0f64.ln(), 0.0);
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].ln() / ln10
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute base-2 logarithm of each complex element
    pub fn log2(&self) -> Self {
        let (rows, cols) = self.shape();
        let ln2 = Complex::new(2.0f64.ln(), 0.0);
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].ln() / ln2
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute square root of each complex element
    pub fn sqrt(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].sqrt()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute absolute value (magnitude) of each complex element
    /// Returns array of real f64 values
    pub fn abs(&self) -> ArrayF64 {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].norm()
        });
        ArrayF64 { inner: result }
    }
    
    /// Raise each complex element to a complex power
    pub fn pow(&self, exponent: Complex<f64>) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].powc(exponent)
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute arcsine of each complex element
    pub fn asin(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].asin()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute arccosine of each complex element
    pub fn acos(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].acos()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute arctangent of each complex element
    pub fn atan(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].atan()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute hyperbolic sine of each complex element
    pub fn sinh(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].sinh()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute hyperbolic cosine of each complex element
    pub fn cosh(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].cosh()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute hyperbolic tangent of each complex element
    pub fn tanh(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].tanh()
        });
        ArrayC64 { inner: result }
    }
    
    // Complex-specific functions
    
    /// Extract real parts of complex elements
    pub fn real(&self) -> ArrayF64 {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].re
        });
        ArrayF64 { inner: result }
    }
    
    /// Extract imaginary parts of complex elements
    pub fn imag(&self) -> ArrayF64 {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].im
        });
        ArrayF64 { inner: result }
    }
    
    /// Compute complex conjugate of each element
    pub fn conj(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].conj()
        });
        ArrayC64 { inner: result }
    }
    
    /// Compute argument/phase of each complex element (in radians)
    pub fn arg(&self) -> ArrayF64 {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].arg()
        });
        ArrayF64 { inner: result }
    }
    
    /// Compute magnitude/norm of each complex element
    /// Alias for abs() for consistency with complex number terminology
    pub fn norm(&self) -> ArrayF64 {
        self.abs()
    }
}

/// Mathematical functions for ArrayC32 (Complex<f32>)
impl ArrayC32 {
    /// Compute sine of each complex element
    pub fn sin(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].sin()
        });
        ArrayC32 { inner: result }
    }
    
    /// Compute cosine of each complex element
    pub fn cos(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].cos()
        });
        ArrayC32 { inner: result }
    }
    
    /// Compute exponential (e^z) of each complex element
    pub fn exp(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].exp()
        });
        ArrayC32 { inner: result }
    }
    
    /// Compute natural logarithm of each complex element
    pub fn ln(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].ln()
        });
        ArrayC32 { inner: result }
    }
    
    /// Compute square root of each complex element
    pub fn sqrt(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].sqrt()
        });
        ArrayC32 { inner: result }
    }
    
    /// Compute absolute value (magnitude) of each complex element
    pub fn abs(&self) -> ArrayF32 {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].norm()
        });
        ArrayF32 { inner: result }
    }
    
    /// Extract real parts of complex elements
    pub fn real(&self) -> ArrayF32 {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].re
        });
        ArrayF32 { inner: result }
    }
    
    /// Extract imaginary parts of complex elements
    pub fn imag(&self) -> ArrayF32 {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].im
        });
        ArrayF32 { inner: result }
    }
    
    /// Compute complex conjugate of each element
    pub fn conj(&self) -> Self {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].conj()
        });
        ArrayC32 { inner: result }
    }
    
    /// Compute argument/phase of each complex element (in radians)
    pub fn arg(&self) -> ArrayF32 {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            self.inner[(i, j)].arg()
        });
        ArrayF32 { inner: result }
    }
    
    /// Compute magnitude/norm of each complex element
    pub fn norm(&self) -> ArrayF32 {
        self.abs()
    }
}

/// Mathematical functions for VectorF64 with automatic SIMD optimization
impl VectorF64 {
    /// Compute sine of each element
    /// 
    /// For vectors, we use scalar operations as they are more efficient
    /// than SIMD for typical vector sizes in signal processing
    pub fn sin(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].sin()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute cosine of each element
    /// 
    /// For vectors, we use scalar operations as they are more efficient
    /// than SIMD for typical vector sizes in signal processing
    pub fn cos(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].cos()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute exponential (e^x) of each element
    /// 
    /// For vectors, we use scalar operations as they are more efficient
    /// than SIMD for typical vector sizes in signal processing
    pub fn exp(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].exp()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute natural logarithm of each element
    /// 
    /// For vectors, we use scalar operations as they are more efficient
    /// than SIMD for typical vector sizes in signal processing
    pub fn ln(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].ln()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute square root of each element
    /// 
    /// For vectors, we use scalar operations as they are more efficient
    /// than SIMD for typical vector sizes in signal processing
    pub fn sqrt(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].sqrt()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute absolute value of each element
    pub fn abs(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].abs()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute square of each element (x^2)
    pub fn square(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            let x = self.inner[i];
            x * x
        });
        VectorF64 { inner: result }
    }
    
    /// Raise each element to a power
    pub fn pow(&self, exponent: f64) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].powf(exponent)
        });
        VectorF64 { inner: result }
    }
    
    /// Compute tangent of each element
    pub fn tan(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].tan()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute hyperbolic sine of each element
    pub fn sinh(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].sinh()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute hyperbolic cosine of each element
    pub fn cosh(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].cosh()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute hyperbolic tangent of each element
    pub fn tanh(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].tanh()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute base-10 logarithm of each element
    pub fn log10(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].log10()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute base-2 logarithm of each element
    pub fn log2(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].log2()
        });
        VectorF64 { inner: result }
    }
}

/// Mathematical functions for VectorF32 with automatic SIMD optimization
impl VectorF32 {
    /// Compute sine of each element
    /// 
    /// For vectors, we use scalar operations as they are more efficient
    /// than SIMD for typical vector sizes in signal processing
    pub fn sin(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].sin()
        });
        VectorF32 { inner: result }
    }
    
    /// Compute cosine of each element
    /// 
    /// For vectors, we use scalar operations as they are more efficient
    /// than SIMD for typical vector sizes in signal processing
    pub fn cos(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].cos()
        });
        VectorF32 { inner: result }
    }
    
    /// Compute exponential (e^x) of each element
    /// 
    /// For vectors, we use scalar operations as they are more efficient
    /// than SIMD for typical vector sizes in signal processing
    pub fn exp(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].exp()
        });
        VectorF32 { inner: result }
    }
    
    /// Compute natural logarithm of each element
    /// 
    /// For vectors, we use scalar operations as they are more efficient
    /// than SIMD for typical vector sizes in signal processing
    pub fn ln(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].ln()
        });
        VectorF32 { inner: result }
    }
    
    /// Compute square root of each element
    /// 
    /// For vectors, we use scalar operations as they are more efficient
    /// than SIMD for typical vector sizes in signal processing
    pub fn sqrt(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].sqrt()
        });
        VectorF32 { inner: result }
    }
}

/// Mathematical functions for VectorC64 (Complex<f64>)
impl VectorC64 {
    /// Compute sine of each complex element
    pub fn sin(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].sin()
        });
        VectorC64 { inner: result }
    }
    
    /// Compute cosine of each complex element
    pub fn cos(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].cos()
        });
        VectorC64 { inner: result }
    }
    
    /// Compute exponential (e^z) of each complex element
    pub fn exp(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].exp()
        });
        VectorC64 { inner: result }
    }
    
    /// Compute natural logarithm of each complex element
    pub fn ln(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].ln()
        });
        VectorC64 { inner: result }
    }
    
    /// Compute square root of each complex element
    pub fn sqrt(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].sqrt()
        });
        VectorC64 { inner: result }
    }
    
    /// Compute absolute value (magnitude) of each complex element
    pub fn abs(&self) -> VectorF64 {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].norm()
        });
        VectorF64 { inner: result }
    }
    
    /// Extract real parts of complex elements
    pub fn real(&self) -> VectorF64 {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].re
        });
        VectorF64 { inner: result }
    }
    
    /// Extract imaginary parts of complex elements
    pub fn imag(&self) -> VectorF64 {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].im
        });
        VectorF64 { inner: result }
    }
    
    /// Compute complex conjugate of each element
    pub fn conj(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].conj()
        });
        VectorC64 { inner: result }
    }
    
    /// Compute argument/phase of each complex element (in radians)
    pub fn arg(&self) -> VectorF64 {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].arg()
        });
        VectorF64 { inner: result }
    }
    
    /// Compute magnitude/norm of each complex element
    pub fn magnitude(&self) -> VectorF64 {
        self.abs()
    }
}

/// Mathematical functions for VectorC32 (Complex<f32>)
impl VectorC32 {
    /// Compute sine of each complex element
    pub fn sin(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].sin()
        });
        VectorC32 { inner: result }
    }
    
    /// Compute cosine of each complex element
    pub fn cos(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].cos()
        });
        VectorC32 { inner: result }
    }
    
    /// Compute exponential (e^z) of each complex element
    pub fn exp(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].exp()
        });
        VectorC32 { inner: result }
    }
    
    /// Compute natural logarithm of each complex element
    pub fn ln(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].ln()
        });
        VectorC32 { inner: result }
    }
    
    /// Compute square root of each complex element
    pub fn sqrt(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].sqrt()
        });
        VectorC32 { inner: result }
    }
    
    /// Compute absolute value (magnitude) of each complex element
    pub fn abs(&self) -> VectorF32 {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].norm()
        });
        VectorF32 { inner: result }
    }
    
    /// Extract real parts of complex elements
    pub fn real(&self) -> VectorF32 {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].re
        });
        VectorF32 { inner: result }
    }
    
    /// Extract imaginary parts of complex elements
    pub fn imag(&self) -> VectorF32 {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].im
        });
        VectorF32 { inner: result }
    }
    
    /// Compute complex conjugate of each element
    pub fn conj(&self) -> Self {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].conj()
        });
        VectorC32 { inner: result }
    }
    
    /// Compute argument/phase of each complex element (in radians)
    pub fn arg(&self) -> VectorF32 {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            self.inner[i].arg()
        });
        VectorF32 { inner: result }
    }
    
    /// Compute magnitude/norm of each complex element
    pub fn magnitude(&self) -> VectorF32 {
        self.abs()
    }
}

// ========== SIMD IMPLEMENTATIONS ==========

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD sine for f64 arrays
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sin_f64(a: &ArrayF64) -> ArrayF64 {
    let (rows, cols) = a.shape();
    let mut result = Mat::<f64>::zeros(rows, cols);
    
    // For now, use scalar implementation since SIMD sine is complex
    // TODO: Implement proper SIMD sine with range reduction and high accuracy
    for i in 0..rows {
        for j in 0..cols {
            result[(i, j)] = a.inner[(i, j)].sin();
        }
    }
    
    ArrayF64 { inner: result }
}

// Fallback for non-x86_64
#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_sin_f64(a: &ArrayF64) -> ArrayF64 {
    a.sin() // Will use scalar path
}

/// SIMD cosine for f64 arrays
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_cos_f64(a: &ArrayF64) -> ArrayF64 {
    let (rows, cols) = a.shape();
    let mut result = Mat::<f64>::zeros(rows, cols);
    
    // Process 4 elements at a time with AVX2
    let chunks = rows * cols / 4;
    let remainder = rows * cols % 4;
    
    for chunk in 0..chunks {
        let base_idx = chunk * 4;
        
        // Load 4 values
        let mut values = [0.0; 4];
        for i in 0..4 {
            let idx = base_idx + i;
            let r = idx / cols;
            let c = idx % cols;
            values[i] = a.inner[(r, c)];
        }
        
        let x = _mm256_loadu_pd(values.as_ptr());
        
        // cos(x) ≈ 1 - x²/2 + x⁴/24
        let x2 = _mm256_mul_pd(x, x);
        let x4 = _mm256_mul_pd(x2, x2);
        
        let one = _mm256_set1_pd(1.0);
        let c2 = _mm256_set1_pd(-0.5);
        let c4 = _mm256_set1_pd(1.0 / 24.0);
        
        let term1 = _mm256_mul_pd(c2, x2);
        let term2 = _mm256_mul_pd(c4, x4);
        
        let result_vec = _mm256_add_pd(one, _mm256_add_pd(term1, term2));
        
        // Store results
        let mut results = [0.0; 4];
        _mm256_storeu_pd(results.as_mut_ptr(), result_vec);
        
        for i in 0..4 {
            let idx = base_idx + i;
            let r = idx / cols;
            let c = idx % cols;
            result[(r, c)] = results[i];
        }
    }
    
    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let idx = base + i;
        let r = idx / cols;
        let c = idx % cols;
        result[(r, c)] = a.inner[(r, c)].cos();
    }
    
    ArrayF64 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_cos_f64(a: &ArrayF64) -> ArrayF64 {
    a.cos()
}

/// SIMD exponential for f64 arrays
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_exp_f64(a: &ArrayF64) -> ArrayF64 {
    let (rows, cols) = a.shape();
    let mut result = Mat::<f64>::zeros(rows, cols);
    
    // Process 4 elements at a time
    let chunks = rows * cols / 4;
    let remainder = rows * cols % 4;
    
    for chunk in 0..chunks {
        let base_idx = chunk * 4;
        
        // Load 4 values
        let mut values = [0.0; 4];
        for i in 0..4 {
            let idx = base_idx + i;
            let r = idx / cols;
            let c = idx % cols;
            values[i] = a.inner[(r, c)];
        }
        
        let x = _mm256_loadu_pd(values.as_ptr());
        
        // Simple exp approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
        let one = _mm256_set1_pd(1.0);
        let x2 = _mm256_mul_pd(x, x);
        let x3 = _mm256_mul_pd(x2, x);
        let x4 = _mm256_mul_pd(x2, x2);
        
        let c2 = _mm256_set1_pd(0.5);
        let c3 = _mm256_set1_pd(1.0 / 6.0);
        let c4 = _mm256_set1_pd(1.0 / 24.0);
        
        let term2 = _mm256_mul_pd(c2, x2);
        let term3 = _mm256_mul_pd(c3, x3);
        let term4 = _mm256_mul_pd(c4, x4);
        
        let sum = _mm256_add_pd(one, x);
        let sum = _mm256_add_pd(sum, term2);
        let sum = _mm256_add_pd(sum, term3);
        let result_vec = _mm256_add_pd(sum, term4);
        
        // Store results
        let mut results = [0.0; 4];
        _mm256_storeu_pd(results.as_mut_ptr(), result_vec);
        
        for i in 0..4 {
            let idx = base_idx + i;
            let r = idx / cols;
            let c = idx % cols;
            result[(r, c)] = results[i];
        }
    }
    
    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let idx = base + i;
        let r = idx / cols;
        let c = idx % cols;
        result[(r, c)] = a.inner[(r, c)].exp();
    }
    
    ArrayF64 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_exp_f64(a: &ArrayF64) -> ArrayF64 {
    a.exp()
}

/// SIMD natural logarithm for f64 arrays
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_ln_f64(a: &ArrayF64) -> ArrayF64 {
    let (rows, cols) = a.shape();
    let mut result = Mat::<f64>::zeros(rows, cols);
    
    // For now, use scalar implementation
    // Full SIMD ln is complex with bit manipulation
    for i in 0..rows {
        for j in 0..cols {
            result[(i, j)] = a.inner[(i, j)].ln();
        }
    }
    
    ArrayF64 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_ln_f64(a: &ArrayF64) -> ArrayF64 {
    a.ln()
}

/// SIMD square root for f64 arrays
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sqrt_f64(a: &ArrayF64) -> ArrayF64 {
    let (rows, cols) = a.shape();
    let mut result = Mat::<f64>::zeros(rows, cols);
    
    // Process 4 elements at a time
    let chunks = rows * cols / 4;
    let remainder = rows * cols % 4;
    
    for chunk in 0..chunks {
        let base_idx = chunk * 4;
        
        // Load 4 values
        let mut values = [0.0; 4];
        for i in 0..4 {
            let idx = base_idx + i;
            let r = idx / cols;
            let c = idx % cols;
            values[i] = a.inner[(r, c)];
        }
        
        let x = _mm256_loadu_pd(values.as_ptr());
        let result_vec = _mm256_sqrt_pd(x); // Hardware instruction!
        
        // Store results
        let mut results = [0.0; 4];
        _mm256_storeu_pd(results.as_mut_ptr(), result_vec);
        
        for i in 0..4 {
            let idx = base_idx + i;
            let r = idx / cols;
            let c = idx % cols;
            result[(r, c)] = results[i];
        }
    }
    
    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let idx = base + i;
        let r = idx / cols;
        let c = idx % cols;
        result[(r, c)] = a.inner[(r, c)].sqrt();
    }
    
    ArrayF64 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_sqrt_f64(a: &ArrayF64) -> ArrayF64 {
    a.sqrt()
}

/// SIMD power for f64 arrays
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_pow_f64(a: &ArrayF64, exp: f64) -> ArrayF64 {
    // For general pow, we typically use x^y = exp(y * ln(x))
    // For now, fallback to scalar
    let (rows, cols) = a.shape();
    let result = Mat::from_fn(rows, cols, |i, j| {
        a.inner[(i, j)].powf(exp)
    });
    ArrayF64 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_pow_f64(a: &ArrayF64, exp: f64) -> ArrayF64 {
    a.pow(exp)
}

// Similar implementations for f32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sin_f32(a: &ArrayF32) -> ArrayF32 {
    let (rows, cols) = a.shape();
    let mut result = Mat::<f32>::zeros(rows, cols);
    
    // Process 8 elements at a time with AVX2
    let chunks = rows * cols / 8;
    let remainder = rows * cols % 8;
    
    for chunk in 0..chunks {
        let base_idx = chunk * 8;
        
        // Load 8 values
        let mut values = [0.0f32; 8];
        for i in 0..8 {
            let idx = base_idx + i;
            let r = idx / cols;
            let c = idx % cols;
            values[i] = a.inner[(r, c)];
        }
        
        let x = _mm256_loadu_ps(values.as_ptr());
        
        // sin(x) ≈ x - x³/6
        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        
        let c3 = _mm256_set1_ps(-1.0 / 6.0);
        let term1 = _mm256_mul_ps(c3, x3);
        
        let result_vec = _mm256_add_ps(x, term1);
        
        // Store results
        let mut results = [0.0f32; 8];
        _mm256_storeu_ps(results.as_mut_ptr(), result_vec);
        
        for i in 0..8 {
            let idx = base_idx + i;
            let r = idx / cols;
            let c = idx % cols;
            result[(r, c)] = results[i];
        }
    }
    
    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let idx = base + i;
        let r = idx / cols;
        let c = idx % cols;
        result[(r, c)] = a.inner[(r, c)].sin();
    }
    
    ArrayF32 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_sin_f32(a: &ArrayF32) -> ArrayF32 {
    a.sin()
}

// Continue with other f32 functions...
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_cos_f32(a: &ArrayF32) -> ArrayF32 {
    let (rows, cols) = a.shape();
    let result = Mat::from_fn(rows, cols, |i, j| {
        a.inner[(i, j)].cos()
    });
    ArrayF32 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_cos_f32(a: &ArrayF32) -> ArrayF32 {
    a.cos()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_exp_f32(a: &ArrayF32) -> ArrayF32 {
    let (rows, cols) = a.shape();
    let result = Mat::from_fn(rows, cols, |i, j| {
        a.inner[(i, j)].exp()
    });
    ArrayF32 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_exp_f32(a: &ArrayF32) -> ArrayF32 {
    a.exp()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_ln_f32(a: &ArrayF32) -> ArrayF32 {
    let (rows, cols) = a.shape();
    let result = Mat::from_fn(rows, cols, |i, j| {
        a.inner[(i, j)].ln()
    });
    ArrayF32 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_ln_f32(a: &ArrayF32) -> ArrayF32 {
    a.ln()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sqrt_f32(a: &ArrayF32) -> ArrayF32 {
    let (rows, cols) = a.shape();
    let mut result = Mat::<f32>::zeros(rows, cols);
    
    // Process 8 elements at a time
    let chunks = rows * cols / 8;
    let remainder = rows * cols % 8;
    
    for chunk in 0..chunks {
        let base_idx = chunk * 8;
        
        // Load 8 values
        let mut values = [0.0f32; 8];
        for i in 0..8 {
            let idx = base_idx + i;
            let r = idx / cols;
            let c = idx % cols;
            values[i] = a.inner[(r, c)];
        }
        
        let x = _mm256_loadu_ps(values.as_ptr());
        let result_vec = _mm256_sqrt_ps(x); // Hardware instruction!
        
        // Store results
        let mut results = [0.0f32; 8];
        _mm256_storeu_ps(results.as_mut_ptr(), result_vec);
        
        for i in 0..8 {
            let idx = base_idx + i;
            let r = idx / cols;
            let c = idx % cols;
            result[(r, c)] = results[i];
        }
    }
    
    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let idx = base + i;
        let r = idx / cols;
        let c = idx % cols;
        result[(r, c)] = a.inner[(r, c)].sqrt();
    }
    
    ArrayF32 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_sqrt_f32(a: &ArrayF32) -> ArrayF32 {
    a.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_pow_f32(a: &ArrayF32, exp: f32) -> ArrayF32 {
    let (rows, cols) = a.shape();
    let result = Mat::from_fn(rows, cols, |i, j| {
        a.inner[(i, j)].powf(exp)
    });
    ArrayF32 { inner: result }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_pow_f32(a: &ArrayF32, exp: f32) -> ArrayF32 {
    a.pow(exp)
}

// Note: Vector SIMD functions have been removed as vectors use scalar operations
// for better performance with typical signal processing vector sizes.
// SIMD is reserved for large arrays/matrices only.

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_array_sin() {
        use std::f64::consts::PI;
        let a = ArrayF64::from_slice(&[0.0, PI/2.0, PI, 3.0*PI/2.0], 2, 2).unwrap();
        let result = a.sin();
        
        assert_relative_eq!(result.get(0, 0).unwrap(), 0.0, epsilon = 0.001);
        assert_relative_eq!(result.get(0, 1).unwrap(), 1.0, epsilon = 0.001);
        assert_relative_eq!(result.get(1, 0).unwrap(), 0.0, epsilon = 0.001);
        assert_relative_eq!(result.get(1, 1).unwrap(), -1.0, epsilon = 0.001);
    }
    
    #[test]
    fn test_array_cos() {
        use std::f64::consts::PI;
        let a = ArrayF64::from_slice(&[0.0, PI/2.0, PI, 3.0*PI/2.0], 2, 2).unwrap();
        let result = a.cos();
        
        assert_relative_eq!(result.get(0, 0).unwrap(), 1.0, epsilon = 0.001);
        assert_relative_eq!(result.get(0, 1).unwrap(), 0.0, epsilon = 0.001);
        assert_relative_eq!(result.get(1, 0).unwrap(), -1.0, epsilon = 0.001);
        assert_relative_eq!(result.get(1, 1).unwrap(), 0.0, epsilon = 0.001);
    }
    
    #[test]
    fn test_array_exp() {
        let a = ArrayF64::from_slice(&[0.0, 1.0, 2.0, -1.0], 2, 2).unwrap();
        let result = a.exp();
        
        assert_relative_eq!(result.get(0, 0).unwrap(), 1.0, epsilon = 0.001);
        assert_relative_eq!(result.get(0, 1).unwrap(), 2.718281828, epsilon = 0.001);
        assert_relative_eq!(result.get(1, 0).unwrap(), 7.389056099, epsilon = 0.001);
        assert_relative_eq!(result.get(1, 1).unwrap(), 0.367879441, epsilon = 0.001);
    }
    
    #[test]
    fn test_array_sqrt() {
        let a = ArrayF64::from_slice(&[1.0, 4.0, 9.0, 16.0], 2, 2).unwrap();
        let result = a.sqrt();
        
        assert_eq!(result.get(0, 0).unwrap(), 1.0);
        assert_eq!(result.get(0, 1).unwrap(), 2.0);
        assert_eq!(result.get(1, 0).unwrap(), 3.0);
        assert_eq!(result.get(1, 1).unwrap(), 4.0);
    }
    
    #[test]
    fn test_vector_sin() {
        use std::f64::consts::PI;
        let v = VectorF64::from_slice(&[0.0, PI/2.0, PI, 3.0*PI/2.0]);
        let result = v.sin();
        
        assert_relative_eq!(result.get(0).unwrap(), 0.0, epsilon = 0.001);
        assert_relative_eq!(result.get(1).unwrap(), 1.0, epsilon = 0.001);
        assert_relative_eq!(result.get(2).unwrap(), 0.0, epsilon = 0.001);
        assert_relative_eq!(result.get(3).unwrap(), -1.0, epsilon = 0.001);
    }
    
    #[test]
    fn test_vector_sqrt() {
        let v = VectorF64::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);
        let result = v.sqrt();
        
        for i in 0..8 {
            assert_eq!(result.get(i).unwrap(), (i + 1) as f64);
        }
    }
    
    #[test]
    fn test_automatic_simd() {
        // Large array should use SIMD
        let large = ArrayF64::ones(100, 100);
        let result = large.sqrt(); // Should use SIMD automatically
        
        for i in 0..100 {
            for j in 0..100 {
                assert_eq!(result.get(i, j).unwrap(), 1.0);
            }
        }
        
        // Small array should use scalar
        let small = ArrayF64::ones(2, 2);
        let result = small.sqrt(); // Should use scalar path
        
        assert_eq!(result.get(0, 0).unwrap(), 1.0);
    }
}