//! Advanced sampling algorithms and random number generation utilities
//!
//! This module provides a comprehensive suite of sampling algorithms and random number
//! generation utilities optimized for statistical computing and Monte Carlo methods.
//! It includes both classical and modern sampling techniques designed for efficiency,
//! numerical stability, and variance reduction.
//!
//! ## Module Structure
//!
//! - **algorithms**: Core sampling algorithms (Box-Muller, Marsaglia, Ziggurat)
//! - **rng**: Random number generator wrappers and utilities
//!
//! ## Key Features
//!
//! ### High-Performance Sampling Algorithms
//! - **Box-Muller Transform**: Efficient normal distribution sampling
//! - **Marsaglia's Method**: Fast gamma distribution sampling
//! - **Ziggurat Algorithm**: Ultra-fast normal sampling for production use
//! - **Polar Method**: Trigonometry-free alternative to Box-Muller
//!
//! ### Variance Reduction Techniques
//! - **Stratified Sampling**: Reduces variance in Monte Carlo integration
//! - **Antithetic Variables**: Negatively correlated samples for variance reduction
//! - **Latin Hypercube Sampling**: Better space coverage in high dimensions
//!
//! ### Quasi-Random Sequences
//! - **Sobol Sequences**: Low-discrepancy sequences for uniform coverage
//! - **Linear Congruential Generators**: Reproducible sequences for testing
//!
//! ## Examples
//!
//! ```rust
//! use rustlab_distributions::sampling::{
//!     box_muller_samples, marsaglia_gamma_samples, ZigguratNormal,
//!     thread_rng_wrapper, stratified_uniform_samples
//! };
//! use rand::thread_rng;
//!
//! let mut rng = thread_rng();
//!
//! // High-performance normal sampling
//! # #[cfg(feature = "integration")]
//! let normal_samples = box_muller_samples(1000, &mut rng);
//!
//! // Gamma distribution sampling
//! # #[cfg(feature = "integration")]
//! let gamma_samples = marsaglia_gamma_samples(500, 2.0, 1.5, &mut rng);
//!
//! // Ultra-fast Ziggurat method
//! let ziggurat = ZigguratNormal::new();
//! let fast_normal = ziggurat.sample(&mut rng);
//!
//! // Variance reduction with stratified sampling
//! # #[cfg(feature = "integration")]
//! let stratified = stratified_uniform_samples(100, &mut rng);
//! ```
//!
//! ## Performance Considerations
//!
//! - **Ziggurat**: Fastest for repeated normal sampling (~2.5x faster than Box-Muller)
//! - **Box-Muller**: Good balance of speed and simplicity
//! - **Polar Method**: Avoids trigonometric functions, good for some architectures
//! - **Marsaglia**: Highly efficient for gamma distributions with α ≥ 1
//!
//! ## Mathematical Background
//!
//! The algorithms implement well-established mathematical transformations:
//! - **Box-Muller**: Uses inverse transform of Rayleigh distribution
//! - **Marsaglia**: Based on squeeze acceptance for gamma distributions
//! - **Ziggurat**: Rectangle-based rejection sampling with precomputed tables
//! - **LHS**: Ensures exactly one sample per stratum in each dimension

pub mod algorithms;
pub mod rng;

// Re-export commonly used sampling algorithms
/// Core sampling algorithms that are commonly used across the library
pub use algorithms::{
    // Single gamma sample using Marsaglia's method
    marsaglia_gamma_sample, 
    // High-precision inverse normal CDF for uniform-to-normal conversion
    inverse_normal_cdf,
    // High-performance Ziggurat algorithm for normal distribution
    ZigguratNormal
};

/// Vector-based sampling algorithms (available with "integration" feature)
#[cfg(feature = "integration")]
pub use algorithms::{
    // Box-Muller transform for batch normal sampling
    box_muller_transform, 
    // Generate multiple normal samples using Box-Muller
    box_muller_samples, 
    // Generate multiple gamma samples using Marsaglia's method
    marsaglia_gamma_samples, 
    // Polar method for normal sampling without trigonometry
    polar_method_samples
};

/// Random number generation utilities and wrappers
pub use rng::{
    // Enhanced RNG wrapper with additional sampling methods
    RngWrapper, 
    // Create thread-local RNG wrapper
    thread_rng_wrapper, 
    // Create seeded RNG wrapper for reproducibility
    seeded_rng_wrapper,
    // Quasi-random Sobol sequence generator
    SobolSequence, 
    // Simple LCG for reproducible testing
    SimpleLCG
};

/// Variance reduction techniques (available with "integration" feature)
#[cfg(feature = "integration")]
pub use rng::{
    // Stratified sampling for variance reduction in Monte Carlo methods
    stratified_uniform_samples, 
    // Antithetic variables for negatively correlated samples
    antithetic_uniform_samples,
    // Latin Hypercube Sampling for better high-dimensional coverage
    latin_hypercube_samples
};