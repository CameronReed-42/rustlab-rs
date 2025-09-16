//! Core types and utilities for rustlab-optimize

pub mod error;
pub mod result;

pub use error::{Error, Result};
pub use result::{OptimizationResult, Algorithm, ConvergenceStatus};