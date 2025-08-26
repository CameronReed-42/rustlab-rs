//! # RustLab Distributions
//! 
//! A comprehensive probability distributions library for Rust, part of the RustLab ecosystem.
//! 
//! This crate provides:
//! - Continuous distributions (Normal, Uniform, Exponential, Gamma, Beta, etc.)
//! - Discrete distributions (Binomial, Poisson, Bernoulli, Geometric)
//! - Multivariate distributions (Multivariate Normal, Dirichlet)
//! - High-precision calculations using rustlab-special functions
//! - Efficient sampling algorithms
//! 
//! ## Example
//! 
//! ```rust,ignore
//! use rustlab_distributions::{Normal, Distribution, Sampling};
//! use rand::thread_rng;
//! 
//! // Create a normal distribution
//! let normal = Normal::new(0.0, 1.0).unwrap();
//! 
//! // Compute PDF and CDF
//! let pdf = normal.pdf(1.0);
//! let cdf = normal.cdf(1.0);
//! 
//! // Sample from the distribution
//! let mut rng = thread_rng();
//! let sample = normal.sample(&mut rng);
//! ```

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod traits;
pub mod continuous;
pub mod discrete;
pub mod multivariate;
pub mod sampling;
pub mod utils;
pub mod error;
pub mod enhanced_api;
pub mod fitting;

// Integration module (feature-gated)
#[cfg(feature = "integration")]
pub mod integration;

// Re-export main types and traits
pub use traits::{Distribution, ContinuousDistribution, DiscreteDistribution, Sampling};
pub use error::{DistributionError, Result};

// Re-export commonly used distributions
pub use continuous::{Normal, Uniform, Gamma, Exponential};
pub use discrete::{Bernoulli, Binomial, Poisson};

// Re-export enhanced API for ergonomic usage
pub use enhanced_api::{EnhancedNormal, NormalBuilder};

// Re-export fitting functionality
pub use fitting::{Fittable, FitDistribution, FittingResult, FittingMethod, BestFitResult};

// Re-export integration functionality when feature is enabled
#[cfg(feature = "integration")]
pub use integration::{
    DistributionVectorF64, DiscreteDistributionVectorF64,
    DistributionArrayF64, DiscreteDistributionArrayF64,
    normal_samples, uniform_samples, exponential_samples, gamma_samples,
    bernoulli_samples, binomial_samples, poisson_samples,
    normal_array, uniform_array, standard_normal, standard_uniform,
    random_walk, geometric_brownian_motion, random_symmetric_matrix,
    random_positive_definite_matrix, random_sparse_vector,
    random_time_series, ar1_series, random_correlation_matrix,
    multivariate_normal_samples
};