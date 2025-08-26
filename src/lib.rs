//! # RustLab - Comprehensive Scientific Computing Library
//!
//! RustLab is a modern, high-performance scientific computing library for Rust that provides
//! comprehensive functionality for numerical analysis, linear algebra, optimization, statistics,
//! and visualization.
//!
//! ## Features
//!
//! - **Mathematical Operations**: Core mathematical types and operations through `rustlab_math`
//! - **Linear Algebra**: Advanced linear algebra operations via `rustlab_linearalgebra`  
//! - **Optimization**: Curve fitting and optimization algorithms in `rustlab_optimize`
//! - **Statistics**: Statistical functions and distributions through `rustlab_stats` and `rustlab_distributions`
//! - **Numerical Methods**: Integration, differentiation, and root finding via `rustlab_numerical`
//! - **Special Functions**: Mathematical special functions through `rustlab_special`
//! - **Plotting**: Cross-platform plotting with Jupyter support via `rustlab_plotting`
//!
//! ## Quick Start
//!
//! ```rust
//! use rustlab_math::{VectorF64, vec64};
//! use rustlab_math::functional::FunctionalMap;
//!
//! // Create vectors and arrays
//! let x = vec64![1.0, 2.0, 3.0, 4.0];
//! let y = x.map(|val| val * 2.0);
//!
//! assert_eq!(y.get(0), Some(2.0));
//! assert_eq!(y.get(3), Some(8.0));
//! ```
//!
//! ## Jupyter Notebook Support
//!
//! RustLab provides first-class Jupyter notebook support with cross-platform compatibility.
//! In Jupyter notebooks, use this dependency setup:
//!
//! ```text
//! :dep rustlab-math = { path = "../rustlab-math" }
//! ```
//!
//! Then import and use RustLab functions normally.

#![warn(missing_docs)]

// Re-export all core functionality
// Math is always available (it's not optional)
pub use rustlab_math as math;

// Conditionally re-export optional crates
#[cfg(feature = "linalg")]
pub use rustlab_linearalgebra as linalg;

#[cfg(feature = "optimization")]
pub use rustlab_optimize as optimization;

#[cfg(feature = "stats")]
pub use rustlab_stats as stats;

#[cfg(feature = "distributions")]
pub use rustlab_distributions as distributions;

#[cfg(feature = "numerical")]
pub use rustlab_numerical as numerical;

#[cfg(feature = "special")]
pub use rustlab_special as special;

#[cfg(feature = "plotting")]
pub use rustlab_plotting as plotting;

// Re-export common types and functions for convenience
pub use rustlab_math::{VectorF64, ArrayF64, VectorF32, ArrayF32, vec64, array64, vconcat, hconcat};

#[cfg(feature = "plotting")]
pub use rustlab_plotting::Plot;

#[cfg(feature = "plotting")]
pub use rustlab_plotting::plot::Scale;

/// Convenient prelude that imports the most commonly used items
pub mod prelude {
    //! Prelude module for convenient imports
    //! 
    //! ```rust
    //! use rustlab_math::{VectorF64, vec64};
    //! use rustlab_math::functional::FunctionalMap;
    //! 
    //! let x = vec64![1.0, 2.0, 3.0];
    //! let y = x.map(|val| val * 2.0);
    //! assert_eq!(y.get(0), Some(2.0));
    //! ```
    
    // Core mathematical types and macros
    pub use crate::math::{
        VectorF64, ArrayF64, VectorF32, ArrayF32,
        vec64, array64, vec32, array32, vconcat, hconcat,
        linspace, arange, zeros_vec, ones_vec, eye,
        PI, E, TAU
    };
    
    // Functional programming traits (essential for map, filter, etc.)
    pub use crate::math::functional::{FunctionalMap, FunctionalReduce, FunctionalZip};
    
    // Concatenation traits (essential for concat, append operations)
    pub use crate::math::concatenation::{Concatenate, VectorConcatenate};
    
    // Plotting functionality (if available)
    #[cfg(feature = "plotting")]
    pub use crate::plotting::Plot;
    #[cfg(feature = "plotting")]
    pub use crate::plotting::plot::Scale;
    
    // Statistical functions (if available)
    #[cfg(feature = "stats")]
    pub use crate::stats::prelude::*;
    
    // Linear algebra essentials (if available)
    #[cfg(feature = "linalg")]
    pub use crate::linalg::{decompositions, eigenvalues};
    
    // Optimization (if available)
    #[cfg(feature = "optimization")]
    pub use crate::optimization::prelude::*;
    
    // Special functions (if available)
    #[cfg(feature = "special")]
    pub use crate::special::{gamma, bessel_j0, bessel_j1, erf, erfc};
    
    // Numerical methods (if available)
    #[cfg(feature = "numerical")]
    pub use crate::numerical::{integration, differentiation, roots};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_basic_functionality() {
        // Test vector creation and operations
        let x = vec64![1.0, 2.0, 3.0];
        let y = x.map(|val| val * 2.0);
        
        assert_eq!(y.get(0), Some(2.0));
        assert_eq!(y.get(1), Some(4.0));
        assert_eq!(y.get(2), Some(6.0));
    }
    
    #[test]
    fn test_array_functionality() {
        // Test array creation and operations
        let a = array64![[1.0, 2.0], [3.0, 4.0]];
        let b = a.map(|val| val + 1.0);
        
        assert_eq!(b.get(0, 0), Some(2.0));
        assert_eq!(b.get(1, 1), Some(5.0));
    }
}