//! Incomplete gamma functions
//!
//! High-precision implementations of incomplete gamma functions and their regularized versions.
//! These functions are fundamental in statistics, probability theory, and many areas of mathematics.
//!
//! ## Functions Provided
//!
//! - **Lower incomplete gamma**: γ(s, x) = ∫₀ˣ t^(s-1) e^(-t) dt
//! - **Upper incomplete gamma**: Γ(s, x) = ∫ₓ^∞ t^(s-1) e^(-t) dt
//! - **Regularized incomplete gamma**: P(s, x) = γ(s, x) / Γ(s)
//! - **Complementary regularized gamma**: Q(s, x) = Γ(s, x) / Γ(s)
//!
//! ## Mathematical Properties
//!
//! - γ(s, x) + Γ(s, x) = Γ(s)
//! - P(s, x) + Q(s, x) = 1
//! - P(s, 0) = 0, Q(s, 0) = 1
//! - P(s, ∞) = 1, Q(s, ∞) = 0
//!
//! ## Applications
//!
//! - Chi-square distribution CDF: P(ν/2, x/2)
//! - Poisson distribution CDF: Q(k+1, λ)
//! - Gamma distribution CDF: P(α, x/β)

use crate::gamma_functions::gamma;
// Mathematical constants would go here if needed

// Convergence tolerance for series and continued fractions
const TOLERANCE: f64 = 1e-14;
const MAX_ITERATIONS: usize = 1000;

// ===== Public Interface =====

/// Lower incomplete gamma function: γ(s, x) = ∫₀ˣ t^(s-1) e^(-t) dt
///
/// Computes the lower incomplete gamma function for s > 0 and x ≥ 0.
///
/// # Arguments
/// * `s` - Shape parameter (must be positive)
/// * `x` - Upper limit of integration (must be non-negative)
///
/// # Returns
/// The value of γ(s, x)
///
/// # Examples
/// ```rust
/// use rustlab_special::gamma_lower;
/// 
/// // γ(1, 1) = 1 - e^(-1) ≈ 0.6321
/// let result = gamma_lower(1.0, 1.0);
/// assert!((result - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
/// ```
pub fn gamma_lower(s: f64, x: f64) -> f64 {
    if s <= 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    
    // Use series expansion for x < s + 1, continued fractions otherwise
    if x < s + 1.0 {
        gamma_lower_series(s, x)
    } else {
        // γ(s, x) = Γ(s) - Γ(s, x)
        let gamma_s = gamma(s);
        gamma_s - gamma_upper_continued_fraction(s, x)
    }
}

/// Upper incomplete gamma function: Γ(s, x) = ∫ₓ^∞ t^(s-1) e^(-t) dt
///
/// Computes the upper incomplete gamma function for s > 0 and x ≥ 0.
///
/// # Arguments
/// * `s` - Shape parameter (must be positive)
/// * `x` - Lower limit of integration (must be non-negative)
///
/// # Returns
/// The value of Γ(s, x)
///
/// # Examples
/// ```rust
/// use rustlab_special::{gamma_upper, gamma};
/// 
/// // Γ(s, 0) = Γ(s)
/// let s = 2.5;
/// let result = gamma_upper(s, 0.0);
/// assert!((result - gamma(s)).abs() < 1e-12);
/// ```
pub fn gamma_upper(s: f64, x: f64) -> f64 {
    if s <= 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return gamma(s);
    }
    
    // Use continued fractions for x > s + 1, series otherwise
    if x > s + 1.0 {
        gamma_upper_continued_fraction(s, x)
    } else {
        // Γ(s, x) = Γ(s) - γ(s, x)
        let gamma_s = gamma(s);
        gamma_s - gamma_lower_series(s, x)
    }
}

/// Regularized lower incomplete gamma function: P(s, x) = γ(s, x) / Γ(s)
///
/// This is the cumulative distribution function of the gamma distribution.
///
/// # Arguments
/// * `s` - Shape parameter (must be positive)
/// * `x` - Upper limit (must be non-negative)
///
/// # Returns
/// The value of P(s, x) ∈ [0, 1]
///
/// # Examples
/// ```rust
/// use rustlab_special::gamma_p;
/// 
/// // P(s, 0) = 0, P(s, ∞) = 1
/// assert_eq!(gamma_p(2.0, 0.0), 0.0);
/// assert!((gamma_p(2.0, 100.0) - 1.0).abs() < 1e-10);
/// ```
pub fn gamma_p(s: f64, x: f64) -> f64 {
    if s <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    
    let gamma_s = gamma(s);
    gamma_lower(s, x) / gamma_s
}

/// Regularized upper incomplete gamma function: Q(s, x) = Γ(s, x) / Γ(s)
///
/// This is the complementary cumulative distribution function of the gamma distribution.
/// Note: Q(s, x) = 1 - P(s, x)
///
/// # Arguments
/// * `s` - Shape parameter (must be positive)
/// * `x` - Lower limit (must be non-negative)
///
/// # Returns
/// The value of Q(s, x) ∈ [0, 1]
///
/// # Examples
/// ```rust
/// use rustlab_special::{gamma_p, gamma_q};
/// 
/// // P(s, x) + Q(s, x) = 1
/// let s = 3.0;
/// let x = 2.0;
/// assert!((gamma_p(s, x) + gamma_q(s, x) - 1.0).abs() < 1e-14);
/// ```
pub fn gamma_q(s: f64, x: f64) -> f64 {
    if s <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    
    let gamma_s = gamma(s);
    gamma_upper(s, x) / gamma_s
}

// ===== Implementation Details =====

fn gamma_lower_series(s: f64, x: f64) -> f64 {
    // Series expansion: γ(s, x) = x^s * e^(-x) * Σ(n=0 to ∞) x^n / Γ(s + n + 1)
    // Rearranged: γ(s, x) = x^s * e^(-x) / Γ(s) * Σ(n=0 to ∞) x^n * Γ(s) / Γ(s + n + 1)
    // Using Γ(s + n + 1) = (s + n) * Γ(s + n), we get a recurrence
    
    let ln_x = x.ln();
    let log_term = s * ln_x - x;
    
    // Handle potential overflow/underflow
    if log_term < -700.0 {
        return 0.0;
    }
    
    let prefactor = log_term.exp();
    let mut sum = 1.0;
    let mut term = 1.0;
    
    for n in 1..MAX_ITERATIONS {
        term *= x / (s + n as f64);
        sum += term;
        
        if term.abs() < TOLERANCE * sum.abs() {
            break;
        }
    }
    
    prefactor * sum / gamma(s)
}

fn gamma_upper_continued_fraction(s: f64, x: f64) -> f64 {
    // Continued fraction: Γ(s, x) = x^s * e^(-x) * (1 / (x + 1-s - 1/(x + 3-s - 2/(x + 5-s - ...))))
    
    let ln_x = x.ln();
    let log_term = s * ln_x - x;
    
    // Handle potential overflow/underflow
    if log_term < -700.0 {
        return 0.0;
    }
    
    let prefactor = log_term.exp();
    
    // Evaluate continued fraction using modified Lentz's method
    let mut b = x + 1.0 - s;
    let mut c = 1e30;
    let mut d = 1.0 / b;
    let mut h = d;
    
    for i in 1..MAX_ITERATIONS {
        let a = -(i as f64) * (i as f64 - s);
        b += 2.0;
        d = a * d + b;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = b + a / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        
        if (del - 1.0).abs() < TOLERANCE {
            break;
        }
    }
    
    prefactor * h
}

#[cfg(test)]
mod tests {
    use super::*;
    
    const TEST_TOLERANCE: f64 = 1e-12;
    
    #[test]
    fn test_gamma_lower_special_values() {
        // γ(1, x) = 1 - e^(-x)
        let x = 2.0;
        let expected = 1.0 - (-x as f64).exp();
        let result = gamma_lower(1.0, x);
        assert!((result - expected).abs() < TEST_TOLERANCE);
        
        // γ(s, 0) = 0
        assert_eq!(gamma_lower(2.5, 0.0), 0.0);
    }
    
    #[test]
    fn test_gamma_upper_special_values() {
        // Γ(1, x) = e^(-x)
        let x = 1.5;
        let expected = (-x as f64).exp();
        let result = gamma_upper(1.0, x);
        assert!((result - expected).abs() < TEST_TOLERANCE);
        
        // Γ(s, 0) = Γ(s)
        let s = 3.5;
        let result = gamma_upper(s, 0.0);
        let expected = gamma(s);
        assert!((result - expected).abs() < TEST_TOLERANCE);
    }
    
    #[test]
    fn test_completeness_relation() {
        // γ(s, x) + Γ(s, x) = Γ(s)
        let test_cases = [(1.5, 0.5), (2.0, 1.0), (0.5, 2.0), (5.0, 3.0)];
        
        for &(s, x) in &test_cases {
            let lower = gamma_lower(s, x);
            let upper = gamma_upper(s, x);
            let total = lower + upper;
            let expected = gamma(s);
            
            assert!((total - expected).abs() < TEST_TOLERANCE,
                   "Completeness failed for s={}, x={}: {} + {} = {}, expected {}",
                   s, x, lower, upper, total, expected);
        }
    }
    
    #[test]
    fn test_regularized_functions() {
        // P(s, x) + Q(s, x) = 1
        let test_cases = [(1.0, 1.0), (2.0, 0.5), (0.5, 3.0)];
        
        for &(s, x) in &test_cases {
            let p = gamma_p(s, x);
            let q = gamma_q(s, x);
            
            assert!((p + q - 1.0).abs() < TEST_TOLERANCE,
                   "P + Q != 1 for s={}, x={}: {} + {} = {}",
                   s, x, p, q, p + q);
            
            // Check bounds with small tolerance for floating point precision
            assert!(p >= -1e-15 && p <= 1.0 + 1e-15, "P out of bounds: {}", p);
            assert!(q >= -1e-15 && q <= 1.0 + 1e-15, "Q out of bounds: {}", q);
        }
    }
    
    #[test]
    fn test_boundary_values() {
        let s = 2.0;
        
        // P(s, 0) = 0, Q(s, 0) = 1
        assert_eq!(gamma_p(s, 0.0), 0.0);
        assert_eq!(gamma_q(s, 0.0), 1.0);
        
        // P(s, ∞) ≈ 1, Q(s, ∞) ≈ 0
        let large_x = 50.0;
        assert!((gamma_p(s, large_x) - 1.0).abs() < 1e-10);
        assert!(gamma_q(s, large_x) < 1e-10);
    }
}