//! RustLab Math - Mathematical Computing with AI-Optimized Documentation
//! 
//! A high-performance numerical computing library designed for intuitive mathematical
//! syntax with comprehensive AI code generation support. Built on faer for optimal
//! performance and scientific computing workflows.
//!
//! # CRITICAL for AI Code Generation
//! 
//! **Operator Distinction** (Most Important Rule):
//! - **`^` operator**: Mathematical multiplication (matrix/dot product)
//! - **`*` operator**: Element-wise multiplication (Hadamard product)
//!
//! This prevents the most common AI hallucination in numerical computing.
//!
//! # Quick Start for AI
//!
//! ```rust
//! use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
//!
//! // Create data structures
//! let A = array64![[1.0, 2.0], [3.0, 4.0]];  // 2×2 matrix
//! let v = vec64![1.0, 2.0];                   // 2D vector
//!
//! // Mathematical operations (use ^ for matrix math)
//! let result = &A ^ &v;          // Matrix-vector multiplication
//! let dot_prod = &v ^ &v;        // Vector dot product (scalar)
//! let mat_mult = &A ^ &A;        // Matrix multiplication
//!
//! // Element-wise operations (use * for element-wise)
//! let elem_mult = &A * &A;       // Element-wise (Hadamard) product
//! let scaled = &A * 2.0;         // Scalar multiplication
//! let sum = &A + &A;             // Element-wise addition
//!
//! // Natural slicing (zero-copy views)
//! let slice = &v[1..2];          // Slice reference
//! let owned = v.slice_at(1..2)?; // Owned slice
//!
//! // Advanced slicing with numpy-style indexing
//! let indices = vec![0, 1]; 
//! let selected = v.slice_at(indices)?;    // Fancy indexing
//! ```
//!
//! # Core Design Principles
//!
//! 1. **Math-First Syntax**: Natural mathematical notation (`A ^ B`, `&v + &u`)
//! 2. **AI-Friendly**: Comprehensive documentation preventing hallucinations
//! 3. **Zero-Cost Abstractions**: High-level ergonomics with optimal performance
//! 4. **NumPy-Style Operations**: Familiar patterns for scientific computing
//! 5. **Type Safety**: Compile-time dimension checking where possible
//! 6. **Memory Efficient**: Zero-copy views and lazy evaluation
//!
//! # Module Organization
//!
//! ## Core Data Structures
//! - [`Array`] - Generic 2D matrices with mathematical operations
//! - [`Vector`] - Generic 1D vectors with mathematical operations
//! - Type aliases: [`ArrayF64`], [`VectorF64`], [`ArrayF32`], [`VectorF32`]
//!
//! ## Mathematical Operations
//! - [`operators`] - The critical `^` operator for matrix multiplication
//! - [`arithmetic`] - Element-wise operations (`+`, `-`, `*`, `/`)
//! - [`broadcasting`] - NumPy-style broadcasting for mixed operations
//! - [`statistics`] - Statistical functions (mean, std, var)
//! - [`reductions`] - Axis-specific reductions (sum, mean along dimensions)
//!
//! ## Data Creation and Manipulation
//! - [`creation`] - Matrix/vector creation (`zeros`, `ones`, `linspace`, `eye`)
//! - [`macros`] - Convenient creation macros (`array64!`, `vec64!`)
//! - [`concatenation`] - Joining operations (`hstack`, `vstack`, `hcat!`, `vcat!`, `block!`)
//! - [`slicing`] - Advanced slicing infrastructure
//!
//! ## Advanced Features
//! - [`ergonomic_slicing`] - NumPy/MATLAB-style advanced indexing
//! - [`natural_slicing`] - Rust Index trait for `vec[1..4]` syntax
//! - [`comparison`] - Boolean operations and masking
//! - [`functional`] - Functional programming operations (map, filter, fold)
//!
//! # Dimension Rules for AI
//!
//! | Operation | Input Dimensions | Output | Description |
//! |-----------|------------------|---------|-------------|
//! | `A ^ B` | (m×n) × (n×p) | (m×p) | Matrix multiplication |
//! | `A ^ v` | (m×n) × (n) | (m) | Matrix-vector multiplication |  
//! | `u ^ v` | (n) × (n) | scalar | Vector dot product |
//! | `&A + &B` | (m×n) + (m×n) | (m×n) | Element-wise addition |
//! | `&A * &B` | (m×n) × (m×n) | (m×n) | Element-wise multiplication |
//!
//! **Key Rule**: For `A ^ B`, inner dimensions must match: (m×**n**) × (**n**×p)
//!
//! # Performance Features
//!
//! - **Automatic SIMD**: Vectorization for arrays > 64 elements
//! - **Memory Layout**: Cache-friendly operations via faer backend  
//! - **Zero-Copy Views**: Slicing without allocation where possible
//! - **Lazy Evaluation**: Chained operations optimized into single loops
//! - **Reference Operations**: Use `&` to avoid unnecessary moves

#![warn(missing_docs)]

pub mod array;
pub mod vector;
pub mod views;
pub mod error;
pub mod simd;
pub mod operators;
pub mod arithmetic;
pub mod broadcasting;
pub mod broadcasting_ops;
pub mod concatenation;
pub mod comparison;
pub mod creation;
pub mod macros;
pub mod math_functions;
pub mod slicing;
pub mod ergonomic_slicing;
pub mod index_operators;
pub mod natural_slicing;
pub mod functional;
pub mod reductions;
pub mod math_first_reductions;
pub mod constants;
pub mod statistics;
pub mod io;
pub mod comprehension;

// Re-export main generic types
pub use array::Array;
pub use vector::Vector;
pub use views::{ArrayView, VectorView};
pub use error::{MathError, Result};
pub use statistics::BasicStatistics;

// Re-export convenient type aliases
pub use array::{ArrayF64, ArrayF32, ArrayC64, ArrayC32};
pub use vector::{VectorF64, VectorF32, VectorC64, VectorC32};

// Note: ^ operator for matrix multiplication is automatically available

// Re-export SIMD utilities
pub use simd::{is_simd_available, simd_info, simd_benchmark_demo};

// Re-export broadcasting utilities
pub use broadcasting::{BroadcastType, BroadcastOp, Shape, broadcast_compatibility};

// Re-export concatenation utilities
pub use concatenation::{Concatenate, VectorConcatenate};

// Re-export comparison utilities
pub use comparison::{BooleanVector, BooleanArray, VectorOps, ArrayComparison};

// Re-export creation utilities
pub use creation::{linspace, arange, arange_step, zeros_vec, ones_vec, fill_vec, eye, fill, zeros, ones};

// Note: vec64, array64, matrix, hstack, vstack, hcat, vcat, block, vectorize, meshgrid macros are available via #[macro_export]

// Re-export mathematical constants for ergonomic math-first coding
pub use constants::{
    // Core mathematical constants
    PI, E, TAU, PHI, EULER_GAMMA,
    // Square roots
    SQRT_2, SQRT_3, FRAC_1_SQRT_2,
    // Pi fractions (common angles)
    PI_2, PI_3, PI_4, PI_6, PI_8,
    // Pi reciprocals
    FRAC_1_PI, FRAC_2_PI, FRAC_2_SQRT_PI,
    // Logarithms
    LN_2, LN_10, LOG2_E, LOG2_10, LOG10_E, LOG10_2,
    // Angle conversions
    DEG_TO_RAD, RAD_TO_DEG, DEGREES_PER_RADIAN, RADIANS_PER_DEGREE,
    // Physical constants (optional, for scientific computing)
    C, H, HBAR, K_B, N_A, E_CHARGE, M_E, M_P, ALPHA
};

// Re-export slicing utilities
pub use slicing::{
    SlicedVectorView, SlicedVectorViewMut, VectorSlicing, 
    VectorSliceF64, VectorSliceMutF64, VectorSliceF32, VectorSliceMutF32,
    SlicedArrayView, SlicedArrayViewMut, ArraySlicing,
    ArraySliceF64, ArraySliceMutF64, ArraySliceF32, ArraySliceMutF32
};

// Re-export ergonomic slicing utilities for NumPy/MATLAB-like syntax
pub use ergonomic_slicing::{
    SliceIndex, SliceIndex2D, IntoSliceIndex
};

// Re-export index operators for natural slicing syntax
pub use index_operators::{
    SliceableVector, SliceableArray
};

// Re-export natural slicing for Python-like syntax
pub use natural_slicing::{NaturalSlicing};

// Re-export functional programming utilities
pub use functional::{
    FunctionalMap, FunctionalReduce, FunctionalZip,
    ZippedArrays, ZippedVectors
};

// Re-export axis reduction utilities
pub use reductions::{Axis, AxisReductions};

// Re-export the clean I/O API
pub use io::MathIO;

// Re-export comprehension utilities for list comprehension with automatic parallelism
pub use comprehension::{
    Complexity, CostModel, Computable,
    vectorize_with_complexity, vectorize_adaptive, vectorize_chunked,
    meshgrid
};

// Note: arithmetic module is imported above as `pub mod arithmetic;`
// All trait implementations in that module are automatically available

// For backward compatibility, provide non-generic defaults
/// Default 2D array type (f64) for backward compatibility  
pub type DefaultArray = ArrayF64;

/// Default 1D vector type (f64) for backward compatibility
pub type DefaultVector = VectorF64;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use crate::io::MathIO;

    #[test]
    fn test_array_basic_ops() {
        let a = ArrayF64::zeros(2, 2);
        let b = ArrayF64::ones(2, 2);
        
        // Test addition
        let result = a + b;
        assert_eq!(result.get(0, 0), Some(1.0));
        assert_eq!(result.get(1, 1), Some(1.0));
        
        // Test scalar multiplication
        let c = ArrayF64::ones(2, 2);
        let result = c * 2.0;
        assert_eq!(result.get(0, 0), Some(2.0));
        assert_eq!(result.get(1, 1), Some(2.0));
    }

    #[test]
    fn test_vector_basic_ops() {
        let a = VectorF64::zeros(3);
        let b = VectorF64::ones(3);
        
        // Test addition
        let result = a + b;
        assert_eq!(result.get(0), Some(1.0));
        assert_eq!(result.get(2), Some(1.0));
        
        // Test scalar multiplication
        let c = VectorF64::ones(3);
        let result = c * 3.0;
        assert_eq!(result.get(0), Some(3.0));
        assert_eq!(result.get(2), Some(3.0));
        
        // Test dot product
        let x = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
        let y = VectorF64::from_slice(&[4.0, 5.0, 6.0]);
        let dot = x.dot(&y);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_chained_operations() {
        // Test the immediate operator chaining like the example in the docs
        let a = ArrayF64::zeros(2, 2);
        let b = ArrayF64::ones(2, 2);
        let result = a + b * 2.0;
        assert_eq!(result.get(0, 0), Some(2.0));
        assert_eq!(result.get(1, 1), Some(2.0));
    }

    #[test]
    fn test_natural_slicing_basic() {
        // Test basic natural slicing with Index trait
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Range slicing
        let slice = &vec[1..4];
        assert_eq!(slice, &[2.0, 3.0, 4.0]);
        
        let tail = &vec[2..];
        assert_eq!(tail, &[3.0, 4.0, 5.0]);
        
        let head = &vec[..3];
        assert_eq!(head, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_natural_slicing_advanced() {
        // Test method-based ergonomic slicing
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Range slicing  
        let slice = vec.slice_at(1..4).unwrap();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 2.0);
        assert_eq!(slice[2], 4.0);
        
        // Negative indexing
        let last = vec.at(-1);
        assert_eq!(last, Some(5.0));
        let second_last = vec.at(-2);
        assert_eq!(second_last, Some(4.0));
        
        // Error handling
        assert_eq!(vec.at(-10), None); // Out of bounds
        assert_eq!(vec.at(10), None);  // Out of bounds
    }

    #[test]
    fn test_io_module_integration() {
        // Test basic I/O functionality with new consolidated API
        let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        
        // Create a temporary file for testing
        let temp_file = NamedTempFile::new().unwrap();
        
        // Save using new MathIO trait
        data.save(temp_file.path()).unwrap();
        
        // Load it back
        let loaded = ArrayF64::load(temp_file.path()).unwrap();
        
        // Verify dimensions
        assert_eq!(loaded.shape(), (2, 3));
        
        // Verify content
        let original_vec = data.to_vec();
        let loaded_vec = loaded.to_vec();
        
        for (orig, load) in original_vec.iter().zip(loaded_vec.iter()) {
            assert!((orig - load).abs() < 1e-15, "Values should match exactly");
        }
        
        println!("✓ I/O module integration test passed");
    }
}