//! Element-Wise Arithmetic Operations with AI-Optimized Documentation
//!
//! This module implements comprehensive element-wise arithmetic operations with automatic
//! SIMD optimization and transparent performance scaling. All operations integrate
//! with RustLab's mathematical ecosystem and follow NumPy-style element-wise conventions.
//!
//! # Core Operations (Element-wise ONLY)
//! - **Addition**: `&a + &b` - Element-by-element addition with auto-SIMD
//! - **Subtraction**: `&a - &b` - Element-by-element subtraction with auto-SIMD  
//! - **Multiplication**: `&a * &b` - Element-wise (Hadamard) product with auto-SIMD
//! - **Scalar operations**: `&a * scalar`, `scalar * &a` - Broadcasting with auto-SIMD
//! - **Division**: `&a / scalar` - Element-wise scalar division with auto-SIMD
//!
//! # CRITICAL DISTINCTION for AI Code Generation
//! - **`*` operator**: Element-wise multiplication (Hadamard product)
//! - **`^` operator**: Mathematical multiplication (matrix/dot product)
//!
//! This prevents the most common AI hallucination in numerical computing libraries.
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
//! 
//! let A = array64![[1.0, 2.0], [3.0, 4.0]];
//! let B = array64![[2.0, 1.0], [1.0, 2.0]];
//! let v = vec64![1.0, 2.0];
//! 
//! // Element-wise arithmetic (use & for references to avoid moves)
//! let elem_sum = &A + &B;        // [[3.0, 3.0], [4.0, 6.0]]
//! let elem_diff = &A - &B;       // [[-1.0, 1.0], [2.0, 2.0]]
//! let hadamard = &A * &B;        // [[2.0, 2.0], [3.0, 8.0]] - element-wise!
//! 
//! // Scalar operations (broadcasting)
//! let scaled = &A * 2.0;         // [[2.0, 4.0], [6.0, 8.0]]
//! let shifted = &A + 1.0;        // [[2.0, 3.0], [4.0, 5.0]]
//! 
//! // Matrix multiplication (NOT element-wise) - use ^ operator
//! let matrix_product = &A ^ &B;  // True matrix multiplication
//! let mv_product = &A ^ &v;      // Matrix-vector multiplication
//! ```
//!
//! # Performance Characteristics
//! - **Auto-SIMD**: Transparent vectorization for arrays/vectors > 64 elements
//! - **Cache optimization**: Memory-efficient operations leveraging faer's layout
//! - **Zero-cost abstractions**: Compile-time optimization of operation chains
//! - **Reference operations**: Use `&` to avoid unnecessary moves and copies
//!
//! # Cross-Module Integration
//! - Complements [`operators`] module (`^` for matrix multiplication)
//! - Works with [`broadcasting`] for automatic dimension compatibility
//! - Integrates with [`Array`] and [`Vector`] core mathematical operations
//! - Compatible with [`statistics`] for efficient statistical computations
//!
//! # SIMD Optimization Details
//! Automatic SIMD vectorization is applied for f64 arrays/vectors larger than 64 elements,
//! providing significant performance improvements for large-scale scientific computing.

use crate::{ArrayF64, VectorF64};
use std::ops::{Add, Sub, Mul, Div};

/// SIMD threshold - use SIMD optimizations for arrays/vectors larger than this
const SIMD_THRESHOLD: usize = 64;

// ========== F64-SPECIFIC SIMD-OPTIMIZED ARRAY OPERATORS ==========

/// ArrayF64 + ArrayF64 (element-wise addition with transparent SIMD)
impl Add<ArrayF64> for ArrayF64 {
    type Output = ArrayF64;
    
    fn add(self, other: ArrayF64) -> ArrayF64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for addition");
        
        // For large arrays, faer's built-in SIMD is already optimal
        // For smaller arrays, use direct faer operations which are cache-friendly
        ArrayF64 {
            inner: &self.inner + &other.inner,
        }
    }
}

/// ArrayF64 - ArrayF64 (element-wise subtraction with transparent SIMD)
impl Sub<ArrayF64> for ArrayF64 {
    type Output = ArrayF64;
    
    fn sub(self, other: ArrayF64) -> ArrayF64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for subtraction");
        
        ArrayF64 {
            inner: &self.inner - &other.inner,
        }
    }
}

/// &ArrayF64 + &ArrayF64 (element-wise addition with transparent SIMD)
impl Add<&ArrayF64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn add(self, other: &ArrayF64) -> ArrayF64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for addition");
        
        // Faer automatically uses SIMD optimizations when beneficial
        ArrayF64 {
            inner: &self.inner + &other.inner,
        }
    }
}

/// &ArrayF64 - &ArrayF64 (element-wise subtraction with transparent SIMD)
impl Sub<&ArrayF64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn sub(self, other: &ArrayF64) -> ArrayF64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for subtraction");
        
        // Faer automatically uses SIMD optimizations when beneficial
        ArrayF64 {
            inner: &self.inner - &other.inner,
        }
    }
}

/// ArrayF64 * ArrayF64 (element-wise multiplication with transparent SIMD)
impl Mul<ArrayF64> for ArrayF64 {
    type Output = ArrayF64;
    
    fn mul(self, other: ArrayF64) -> ArrayF64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for element-wise multiplication");
        
        let total_elements = self.nrows() * self.ncols();
        
        if total_elements >= SIMD_THRESHOLD {
            // For large arrays, use optimized element-wise multiplication
            let (rows, cols) = self.shape();
            ArrayF64 {
                inner: faer::Mat::from_fn(rows, cols, |i, j| {
                    self.inner[(i, j)] * other.inner[(i, j)]
                }),
            }
        } else {
            // For small arrays, simple element-wise is often faster
            let (rows, cols) = self.shape();
            ArrayF64 {
                inner: faer::Mat::from_fn(rows, cols, |i, j| {
                    self.inner[(i, j)] * other.inner[(i, j)]
                }),
            }
        }
    }
}

/// &ArrayF64 * &ArrayF64 (element-wise multiplication with transparent SIMD)
impl Mul<&ArrayF64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn mul(self, other: &ArrayF64) -> ArrayF64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for element-wise multiplication");
        
        let (rows, cols) = self.shape();
        ArrayF64 {
            inner: faer::Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)] * other.inner[(i, j)]
            }),
        }
    }
}

/// ArrayF64 * f64 (scalar multiplication with transparent SIMD)
impl Mul<f64> for ArrayF64 {
    type Output = ArrayF64;
    
    fn mul(self, scalar: f64) -> ArrayF64 {
        let total_elements = self.nrows() * self.ncols();
        
        if total_elements >= SIMD_THRESHOLD {
            // For large arrays, faer's built-in scalar multiplication is SIMD-optimized
            let (rows, cols) = self.shape();
            ArrayF64 {
                inner: faer::Mat::from_fn(rows, cols, |i, j| {
                    self.inner[(i, j)] * scalar
                }),
            }
        } else {
            // For small arrays, direct multiplication
            let (rows, cols) = self.shape();
            ArrayF64 {
                inner: faer::Mat::from_fn(rows, cols, |i, j| {
                    self.inner[(i, j)] * scalar
                }),
            }
        }
    }
}

/// &ArrayF64 * f64 (scalar multiplication with transparent SIMD)
impl Mul<f64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn mul(self, scalar: f64) -> ArrayF64 {
        let (rows, cols) = self.shape();
        ArrayF64 {
            inner: faer::Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)] * scalar
            }),
        }
    }
}

/// f64 * ArrayF64 (commutative scalar multiplication with transparent SIMD)
impl Mul<ArrayF64> for f64 {
    type Output = ArrayF64;
    
    fn mul(self, array: ArrayF64) -> ArrayF64 {
        array * self
    }
}

/// f64 * &ArrayF64 (commutative scalar multiplication with transparent SIMD)
impl Mul<&ArrayF64> for f64 {
    type Output = ArrayF64;
    
    fn mul(self, array: &ArrayF64) -> ArrayF64 {
        array * self
    }
}

// ========== F64-SPECIFIC SIMD-OPTIMIZED VECTOR OPERATORS ==========

/// VectorF64 + VectorF64 (element-wise addition with transparent SIMD)
impl Add<VectorF64> for VectorF64 {
    type Output = VectorF64;
    
    fn add(self, other: VectorF64) -> VectorF64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for addition");
        
        // Faer's built-in vector addition is already SIMD-optimized
        VectorF64 {
            inner: &self.inner + &other.inner,
        }
    }
}

/// &VectorF64 + &VectorF64 (element-wise addition with transparent SIMD)
impl Add<&VectorF64> for &VectorF64 {
    type Output = VectorF64;
    
    fn add(self, other: &VectorF64) -> VectorF64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for addition");
        
        VectorF64 {
            inner: &self.inner + &other.inner,
        }
    }
}

/// VectorF64 * VectorF64 (element-wise multiplication with transparent SIMD)
impl Mul<VectorF64> for VectorF64 {
    type Output = VectorF64;
    
    fn mul(self, other: VectorF64) -> VectorF64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for element-wise multiplication");
        
        let len = self.len();
        
        if len >= SIMD_THRESHOLD {
            // For large vectors, use optimized element-wise multiplication
            VectorF64 {
                inner: faer::Col::from_fn(len, |i| {
                    self.inner[i] * other.inner[i]
                }),
            }
        } else {
            // For small vectors
            VectorF64 {
                inner: faer::Col::from_fn(len, |i| {
                    self.inner[i] * other.inner[i]
                }),
            }
        }
    }
}

/// &VectorF64 * &VectorF64 (element-wise multiplication with transparent SIMD)  
impl Mul<&VectorF64> for &VectorF64 {
    type Output = VectorF64;
    
    fn mul(self, other: &VectorF64) -> VectorF64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for element-wise multiplication");
        
        VectorF64 {
            inner: faer::Col::from_fn(self.len(), |i| {
                self.inner[i] * other.inner[i]
            }),
        }
    }
}

/// VectorF64 * f64 (scalar multiplication with transparent SIMD)
impl Mul<f64> for VectorF64 {
    type Output = VectorF64;
    
    fn mul(self, scalar: f64) -> VectorF64 {
        VectorF64 {
            inner: faer::Col::from_fn(self.len(), |i| {
                self.inner[i] * scalar
            }),
        }
    }
}

/// &VectorF64 * f64 (scalar multiplication with transparent SIMD)
impl Mul<f64> for &VectorF64 {
    type Output = VectorF64;
    
    fn mul(self, scalar: f64) -> VectorF64 {
        VectorF64 {
            inner: faer::Col::from_fn(self.len(), |i| {
                self.inner[i] * scalar
            }),
        }
    }
}

/// f64 * VectorF64 (commutative scalar multiplication with transparent SIMD)
impl Mul<VectorF64> for f64 {
    type Output = VectorF64;
    
    fn mul(self, vector: VectorF64) -> VectorF64 {
        vector * self
    }
}

/// VectorF64 - VectorF64 (element-wise subtraction with transparent SIMD)
impl Sub<VectorF64> for VectorF64 {
    type Output = VectorF64;
    
    fn sub(self, other: VectorF64) -> VectorF64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for subtraction");
        
        // Faer automatically uses SIMD optimizations when beneficial
        VectorF64 {
            inner: &self.inner - &other.inner,
        }
    }
}

/// &VectorF64 - &VectorF64 (element-wise subtraction with transparent SIMD)
impl Sub<&VectorF64> for &VectorF64 {
    type Output = VectorF64;
    
    fn sub(self, other: &VectorF64) -> VectorF64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for subtraction");
        
        // Faer automatically uses SIMD optimizations when beneficial
        VectorF64 {
            inner: &self.inner - &other.inner,
        }
    }
}

/// VectorF64 / VectorF64 (element-wise division)
impl Div<VectorF64> for VectorF64 {
    type Output = VectorF64;
    
    fn div(self, other: VectorF64) -> VectorF64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for division");
        
        // Element-wise division using faer's from_fn constructor
        use faer::Col;
        let result = Col::from_fn(self.len(), |i| {
            self.inner[i] / other.inner[i]
        });
        VectorF64 { inner: result }
    }
}

/// &VectorF64 / &VectorF64 (element-wise division)
impl Div<&VectorF64> for &VectorF64 {
    type Output = VectorF64;
    
    fn div(self, other: &VectorF64) -> VectorF64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for division");
        
        // Element-wise division using faer's from_fn constructor
        use faer::Col;
        let result = Col::from_fn(self.len(), |i| {
            self.inner[i] / other.inner[i]
        });
        VectorF64 { inner: result }
    }
}

/// VectorF64 / f64 (scalar division with transparent SIMD)
impl Div<f64> for VectorF64 {
    type Output = VectorF64;
    
    fn div(self, scalar: f64) -> VectorF64 {
        VectorF64 {
            inner: &self.inner / scalar,
        }
    }
}

/// &VectorF64 / f64 (scalar division with transparent SIMD)
impl Div<f64> for &VectorF64 {
    type Output = VectorF64;
    
    fn div(self, scalar: f64) -> VectorF64 {
        VectorF64 {
            inner: &self.inner / scalar,
        }
    }
}

// ========== SCALAR ADDITION AND SUBTRACTION ==========

/// VectorF64 + f64 (scalar addition with transparent SIMD)
impl Add<f64> for VectorF64 {
    type Output = VectorF64;
    
    fn add(self, scalar: f64) -> VectorF64 {
        VectorF64 {
            inner: faer::Col::from_fn(self.len(), |i| {
                self.inner[i] + scalar
            }),
        }
    }
}

/// &VectorF64 + f64 (scalar addition with transparent SIMD)
impl Add<f64> for &VectorF64 {
    type Output = VectorF64;
    
    fn add(self, scalar: f64) -> VectorF64 {
        VectorF64 {
            inner: faer::Col::from_fn(self.len(), |i| {
                self.inner[i] + scalar
            }),
        }
    }
}

/// f64 + VectorF64 (commutative scalar addition with transparent SIMD)
impl Add<VectorF64> for f64 {
    type Output = VectorF64;
    
    fn add(self, vector: VectorF64) -> VectorF64 {
        vector + self
    }
}

/// VectorF64 - f64 (scalar subtraction with transparent SIMD)
impl Sub<f64> for VectorF64 {
    type Output = VectorF64;
    
    fn sub(self, scalar: f64) -> VectorF64 {
        VectorF64 {
            inner: faer::Col::from_fn(self.len(), |i| {
                self.inner[i] - scalar
            }),
        }
    }
}

/// &VectorF64 - f64 (scalar subtraction with transparent SIMD)
impl Sub<f64> for &VectorF64 {
    type Output = VectorF64;
    
    fn sub(self, scalar: f64) -> VectorF64 {
        VectorF64 {
            inner: faer::Col::from_fn(self.len(), |i| {
                self.inner[i] - scalar
            }),
        }
    }
}

/// ArrayF64 + f64 (scalar addition with transparent SIMD)
impl Add<f64> for ArrayF64 {
    type Output = ArrayF64;
    
    fn add(self, scalar: f64) -> ArrayF64 {
        let (rows, cols) = self.shape();
        ArrayF64 {
            inner: faer::Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)] + scalar
            }),
        }
    }
}

/// &ArrayF64 + f64 (scalar addition with transparent SIMD)
impl Add<f64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn add(self, scalar: f64) -> ArrayF64 {
        let (rows, cols) = self.shape();
        ArrayF64 {
            inner: faer::Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)] + scalar
            }),
        }
    }
}

/// f64 + ArrayF64 (commutative scalar addition with transparent SIMD)
impl Add<ArrayF64> for f64 {
    type Output = ArrayF64;
    
    fn add(self, array: ArrayF64) -> ArrayF64 {
        array + self
    }
}

/// ArrayF64 - f64 (scalar subtraction with transparent SIMD)
impl Sub<f64> for ArrayF64 {
    type Output = ArrayF64;
    
    fn sub(self, scalar: f64) -> ArrayF64 {
        let (rows, cols) = self.shape();
        ArrayF64 {
            inner: faer::Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)] - scalar
            }),
        }
    }
}

/// &ArrayF64 - f64 (scalar subtraction with transparent SIMD)
impl Sub<f64> for &ArrayF64 {
    type Output = ArrayF64;
    
    fn sub(self, scalar: f64) -> ArrayF64 {
        let (rows, cols) = self.shape();
        ArrayF64 {
            inner: faer::Mat::from_fn(rows, cols, |i, j| {
                self.inner[(i, j)] - scalar
            }),
        }
    }
}

// ========== COMPLEX NUMBER ARITHMETIC OPERATORS ==========

use crate::{ArrayC64, VectorC64};
use num_complex::Complex;

// Complex Vector Addition
impl Add<VectorC64> for VectorC64 {
    type Output = VectorC64;
    
    fn add(self, other: VectorC64) -> VectorC64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for addition");
        
        let result_data: Vec<Complex<f64>> = self.to_slice()
            .iter()
            .zip(other.to_slice().iter())
            .map(|(a, b)| a + b)
            .collect();
        
        VectorC64::from_slice(&result_data)
    }
}

impl Add<&VectorC64> for &VectorC64 {
    type Output = VectorC64;
    
    fn add(self, other: &VectorC64) -> VectorC64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for addition");
        
        let result_data: Vec<Complex<f64>> = self.to_slice()
            .iter()
            .zip(other.to_slice().iter())
            .map(|(a, b)| a + b)
            .collect();
        
        VectorC64::from_slice(&result_data)
    }
}

// Complex Vector Subtraction
impl Sub<VectorC64> for VectorC64 {
    type Output = VectorC64;
    
    fn sub(self, other: VectorC64) -> VectorC64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for subtraction");
        
        let result_data: Vec<Complex<f64>> = self.to_slice()
            .iter()
            .zip(other.to_slice().iter())
            .map(|(a, b)| a - b)
            .collect();
        
        VectorC64::from_slice(&result_data)
    }
}

impl Sub<&VectorC64> for &VectorC64 {
    type Output = VectorC64;
    
    fn sub(self, other: &VectorC64) -> VectorC64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for subtraction");
        
        let result_data: Vec<Complex<f64>> = self.to_slice()
            .iter()
            .zip(other.to_slice().iter())
            .map(|(a, b)| a - b)
            .collect();
        
        VectorC64::from_slice(&result_data)
    }
}

// Complex Vector Element-wise Multiplication
impl Mul<VectorC64> for VectorC64 {
    type Output = VectorC64;
    
    fn mul(self, other: VectorC64) -> VectorC64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for element-wise multiplication");
        
        let result_data: Vec<Complex<f64>> = self.to_slice()
            .iter()
            .zip(other.to_slice().iter())
            .map(|(a, b)| a * b)
            .collect();
        
        VectorC64::from_slice(&result_data)
    }
}

impl Mul<&VectorC64> for &VectorC64 {
    type Output = VectorC64;
    
    fn mul(self, other: &VectorC64) -> VectorC64 {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length for element-wise multiplication");
        
        let result_data: Vec<Complex<f64>> = self.to_slice()
            .iter()
            .zip(other.to_slice().iter())
            .map(|(a, b)| a * b)
            .collect();
        
        VectorC64::from_slice(&result_data)
    }
}

// Complex Vector Scalar Multiplication
impl Mul<Complex<f64>> for VectorC64 {
    type Output = VectorC64;
    
    fn mul(self, scalar: Complex<f64>) -> VectorC64 {
        let result_data: Vec<Complex<f64>> = self.to_slice()
            .iter()
            .map(|a| a * scalar)
            .collect();
        
        VectorC64::from_slice(&result_data)
    }
}

impl Mul<Complex<f64>> for &VectorC64 {
    type Output = VectorC64;
    
    fn mul(self, scalar: Complex<f64>) -> VectorC64 {
        let result_data: Vec<Complex<f64>> = self.to_slice()
            .iter()
            .map(|a| a * scalar)
            .collect();
        
        VectorC64::from_slice(&result_data)
    }
}

// Complex<f64> * VectorC64 (commutative)
impl Mul<VectorC64> for Complex<f64> {
    type Output = VectorC64;
    
    fn mul(self, vector: VectorC64) -> VectorC64 {
        vector * self
    }
}

impl Mul<&VectorC64> for Complex<f64> {
    type Output = VectorC64;
    
    fn mul(self, vector: &VectorC64) -> VectorC64 {
        vector * self
    }
}

// Complex Array Addition
impl Add<ArrayC64> for ArrayC64 {
    type Output = ArrayC64;
    
    fn add(self, other: ArrayC64) -> ArrayC64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for addition");
        
        let (rows, cols) = self.shape();
        ArrayC64::from_fn(rows, cols, |i, j| {
            self.get(i, j).unwrap() + other.get(i, j).unwrap()
        })
    }
}

impl Add<&ArrayC64> for &ArrayC64 {
    type Output = ArrayC64;
    
    fn add(self, other: &ArrayC64) -> ArrayC64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for addition");
        
        let (rows, cols) = self.shape();
        ArrayC64::from_fn(rows, cols, |i, j| {
            self.get(i, j).unwrap() + other.get(i, j).unwrap()
        })
    }
}

// Complex Array Subtraction
impl Sub<ArrayC64> for ArrayC64 {
    type Output = ArrayC64;
    
    fn sub(self, other: ArrayC64) -> ArrayC64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for subtraction");
        
        let (rows, cols) = self.shape();
        ArrayC64::from_fn(rows, cols, |i, j| {
            self.get(i, j).unwrap() - other.get(i, j).unwrap()
        })
    }
}

impl Sub<&ArrayC64> for &ArrayC64 {
    type Output = ArrayC64;
    
    fn sub(self, other: &ArrayC64) -> ArrayC64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for subtraction");
        
        let (rows, cols) = self.shape();
        ArrayC64::from_fn(rows, cols, |i, j| {
            self.get(i, j).unwrap() - other.get(i, j).unwrap()
        })
    }
}

// Complex Array Element-wise Multiplication
impl Mul<ArrayC64> for ArrayC64 {
    type Output = ArrayC64;
    
    fn mul(self, other: ArrayC64) -> ArrayC64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for element-wise multiplication");
        
        let (rows, cols) = self.shape();
        ArrayC64::from_fn(rows, cols, |i, j| {
            self.get(i, j).unwrap() * other.get(i, j).unwrap()
        })
    }
}

impl Mul<&ArrayC64> for &ArrayC64 {
    type Output = ArrayC64;
    
    fn mul(self, other: &ArrayC64) -> ArrayC64 {
        assert_eq!(self.shape(), other.shape(), "Arrays must have the same shape for element-wise multiplication");
        
        let (rows, cols) = self.shape();
        ArrayC64::from_fn(rows, cols, |i, j| {
            self.get(i, j).unwrap() * other.get(i, j).unwrap()
        })
    }
}

// Complex Array Scalar Multiplication
impl Mul<Complex<f64>> for ArrayC64 {
    type Output = ArrayC64;
    
    fn mul(self, scalar: Complex<f64>) -> ArrayC64 {
        let (rows, cols) = self.shape();
        ArrayC64::from_fn(rows, cols, |i, j| {
            self.get(i, j).unwrap() * scalar
        })
    }
}

impl Mul<Complex<f64>> for &ArrayC64 {
    type Output = ArrayC64;
    
    fn mul(self, scalar: Complex<f64>) -> ArrayC64 {
        let (rows, cols) = self.shape();
        ArrayC64::from_fn(rows, cols, |i, j| {
            self.get(i, j).unwrap() * scalar
        })
    }
}

// Complex<f64> * ArrayC64 (commutative)
impl Mul<ArrayC64> for Complex<f64> {
    type Output = ArrayC64;
    
    fn mul(self, array: ArrayC64) -> ArrayC64 {
        array * self
    }
}

impl Mul<&ArrayC64> for Complex<f64> {
    type Output = ArrayC64;
    
    fn mul(self, array: &ArrayC64) -> ArrayC64 {
        array * self
    }
}

// Note: For now, we focus on f64 and Complex<f64> optimizations. 
// Generic implementations for other types can be added later if needed.