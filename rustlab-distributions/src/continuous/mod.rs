//! Continuous probability distributions

pub mod normal;
pub mod uniform;
pub mod gamma;
pub mod exponential;
pub mod student_t;
pub mod chi_squared;
pub mod fisher_f;
pub mod beta;

// Re-export distributions
pub use normal::Normal;
pub use uniform::Uniform;
pub use gamma::Gamma;
pub use exponential::Exponential;
pub use student_t::StudentT;
pub use chi_squared::ChiSquared;
pub use fisher_f::FisherF;
pub use beta::Beta;