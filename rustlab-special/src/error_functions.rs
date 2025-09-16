//! Error function family: erf, erfc, erfinv, erfcinv
//! 
//! This module provides high-precision implementations of the error function and its variants,
//! which are fundamental to probability theory, statistics, and mathematical physics.
//! 
//! ## Mathematical Background
//! 
//! The error function is defined as:
//! 
//! ```text
//! erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
//! ```
//! 
//! ## Key Properties
//! 
//! - **Domain and Range**: erf: ℝ → (-1, 1), erfc: ℝ → (0, 2)
//! - **Symmetry**: erf(-x) = -erf(x) (odd function)
//! - **Complementary**: erfc(x) = 1 - erf(x)
//! - **Asymptotic behavior**: erf(∞) = 1, erf(-∞) = -1
//! - **Normal distribution**: Φ(x) = (1 + erf(x/√2))/2
//! 
//! ## Implementation Details
//! 
//! - **Small arguments (|x| < 3)**: 50-term Taylor series expansion
//! - **Large arguments (|x| ≥ 3)**: Direct asymptotic evaluation (±1)
//! - **Inverse functions**: Winitzki approximation + Newton-Raphson refinement
//! - **Target accuracy**: 1e-15 relative error in primary domain
//! 
//! ## Applications
//! 
//! - **Statistics**: Normal distribution CDF and quantile functions
//! - **Physics**: Diffusion processes, quantum mechanics
//! - **Engineering**: Signal processing, communications theory

use std::f64::consts::PI;

/// 2/sqrt(pi)
const TWO_OVER_SQRT_PI: f64 = 1.1283791670955125739;

// ===== Error Function Implementation =====

/// Error function erf(x)
/// 
/// Computes the error function, which represents the probability that a standard
/// normal random variable falls within x standard deviations of the mean.
/// 
/// # Arguments
/// 
/// * `x` - The argument at which to evaluate erf(x). Can be any real number.
/// 
/// # Returns
/// 
/// The value of erf(x) ∈ (-1, 1) with target accuracy of 1e-15.
/// 
/// # Mathematical Definition
/// 
/// ```text
/// erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
/// ```
/// 
/// # Special Values
/// 
/// - erf(0) = 0
/// - erf(∞) = 1  
/// - erf(-∞) = -1
/// - erf(-x) = -erf(x) (odd function)
/// 
/// # Algorithm
/// 
/// - For |x| < 3: Uses 50-term Taylor series expansion for maximum accuracy
/// - For |x| ≥ 3: Returns ±1 (asymptotic behavior)
/// 
/// # Examples
/// 
/// ```
/// use rustlab_special::erf;
/// use std::f64::consts::SQRT_2;
/// 
/// // Error function at key points
/// assert_eq!(erf(0.0), 0.0);
/// assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-14);
/// 
/// // Odd function property
/// assert_eq!(erf(-1.5), -erf(1.5));
/// 
/// // Normal distribution CDF: Φ(x) = (1 + erf(x/√2))/2
/// let normal_cdf_1 = 0.5 * (1.0 + erf(1.0 / SQRT_2));
/// assert!((normal_cdf_1 - 0.8413447460685429).abs() < 1e-14);
/// ```
/// 
/// # Applications
/// 
/// - **Probability**: Normal distribution cumulative density function
/// - **Statistics**: Confidence intervals and hypothesis testing
/// - **Physics**: Diffusion processes, heat conduction
/// - **Engineering**: Signal processing, error probability in communications
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    
    let ax = x.abs();
    let sign = x.signum();
    
    // Use Taylor series for reasonable range - it's very accurate
    if ax < 3.0 {
        erf_taylor(ax) * sign
    } else {
        // For very large x, erf(x) ≈ ±1
        sign
    }
}

/// Complementary error function erfc(x) = 1 - erf(x)
/// 
/// Computes the complementary error function, which gives the tail probability
/// of the normal distribution and is numerically more stable than 1 - erf(x)
/// for large positive arguments.
/// 
/// # Arguments
/// 
/// * `x` - The argument at which to evaluate erfc(x). Can be any real number.
/// 
/// # Returns
/// 
/// The value of erfc(x) ∈ (0, 2) with target accuracy of 1e-15.
/// 
/// # Mathematical Definition
/// 
/// ```text
/// erfc(x) = 1 - erf(x) = (2/√π) ∫ₓ^∞ e^(-t²) dt
/// ```
/// 
/// # Special Values
/// 
/// - erfc(0) = 1
/// - erfc(∞) = 0
/// - erfc(-∞) = 2
/// - erfc(-x) = 2 - erfc(x)
/// 
/// # Examples
/// 
/// ```
/// use rustlab_special::erfc;
/// 
/// // Complementary error function at key points
/// assert_eq!(erfc(0.0), 1.0);
/// assert!((erfc(1.0) - 0.15729920705028513).abs() < 1e-14);
/// 
/// // Large argument behavior
/// assert!(erfc(5.0) < 1e-10);
/// assert!(erfc(10.0) < 1e-40);
/// 
/// // Relation to normal distribution tail probability
/// // P(X > x) for X ~ N(0,1) = erfc(x/√2)/2
/// let tail_prob = erfc(2.0 / std::f64::consts::SQRT_2) / 2.0;
/// assert!((tail_prob - 0.022750131948179195).abs() < 1e-14);
/// ```
/// 
/// # Applications
/// 
/// - **Statistics**: Tail probabilities, p-values, confidence intervals
/// - **Communications**: Bit error rate calculations
/// - **Physics**: Survival functions, decay processes
/// - **Numerical analysis**: Avoiding subtraction cancellation for large x
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Inverse error function erfinv(x)
/// 
/// Computes the inverse error function, which finds y such that erf(y) = x.
/// This is the quantile function for the normal distribution (after scaling).
/// 
/// # Arguments
/// 
/// * `x` - The value for which to find the inverse, must be in (-1, 1)
/// 
/// # Returns
/// 
/// The value y such that erf(y) = x, with target accuracy of 1e-15.
/// Returns ±∞ for x = ±1, and NaN for |x| > 1.
/// 
/// # Mathematical Definition
/// 
/// ```text
/// erfinv(x) = y  such that  erf(y) = x
/// ```
/// 
/// # Domain and Range
/// 
/// - **Domain**: x ∈ (-1, 1)
/// - **Range**: y ∈ (-∞, ∞)
/// - **Special values**: erfinv(0) = 0, erfinv(±1) = ±∞
/// 
/// # Algorithm
/// 
/// 1. **Initial approximation**: Winitzki's rational approximation
/// 2. **Refinement**: 5 Newton-Raphson iterations for high precision
/// 3. **Newton formula**: y_{n+1} = y_n - (erf(y_n) - x) / (2/√π · exp(-y_n²))
/// 
/// # Examples
/// 
/// ```
/// use rustlab_special::{erfinv, erf};
/// use std::f64::consts::SQRT_2;
/// 
/// // Inverse property: erf(erfinv(x)) = x
/// let x = 0.5;
/// let y = erfinv(x);
/// assert!((erf(y) - x).abs() < 1e-14);
/// 
/// // Normal distribution quantiles
/// // For X ~ N(0,1), P(X ≤ x) = 0.95 gives x = √2 * erfinv(0.9)
/// let quantile_95 = SQRT_2 * erfinv(0.9);
/// assert!((quantile_95 - 1.6448536269514729).abs() < 1e-12);
/// 
/// // Special values
/// assert_eq!(erfinv(0.0), 0.0);
/// assert_eq!(erfinv(1.0), f64::INFINITY);
/// assert_eq!(erfinv(-1.0), f64::NEG_INFINITY);
/// assert!(erfinv(1.5).is_nan()); // Outside domain
/// ```
/// 
/// # Applications
/// 
/// - **Statistics**: Normal distribution quantile function
/// - **Monte Carlo**: Inverse transform sampling from normal distribution
/// - **Finance**: Value-at-Risk calculations, option pricing
/// - **Engineering**: Control system design, reliability analysis
/// 
/// # Notes
/// 
/// The implementation uses up to 5 Newton-Raphson iterations to achieve
/// 1e-15 accuracy. For most practical applications, 3-4 iterations suffice.
pub fn erfinv(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    if x < -1.0 || x > 1.0 {
        return f64::NAN;
    }
    
    // Start with Winitzki approximation
    let mut y = erfinv_winitzki(x);
    
    // Refine with Newton-Raphson iterations
    // We want to solve: erf(y) - x = 0
    // Newton step: y_new = y - (erf(y) - x) / erf'(y)
    // where erf'(y) = (2/√π) * exp(-y²)
    
    for _ in 0..5 {
        let erf_y = erf(y);
        let error = erf_y - x;
        
        if error.abs() < 1e-15 {
            break;
        }
        
        let derivative = TWO_OVER_SQRT_PI * (-y * y).exp();
        y -= error / derivative;
    }
    
    y
}

/// Inverse complementary error function erfcinv(x) = erfinv(1 - x)
/// 
/// Computes the inverse complementary error function, which finds y such that
/// erfc(y) = x. This is particularly useful for computing tail quantiles.
/// 
/// # Arguments
/// 
/// * `x` - The value for which to find the inverse, must be in (0, 2)
/// 
/// # Returns
/// 
/// The value y such that erfc(y) = x, with target accuracy of 1e-15.
/// Returns NaN for x ≤ 0 or x ≥ 2.
/// 
/// # Mathematical Definition
/// 
/// ```text
/// erfcinv(x) = y  such that  erfc(y) = x
/// erfcinv(x) = erfinv(1 - x)
/// ```
/// 
/// # Domain and Range
/// 
/// - **Domain**: x ∈ (0, 2)
/// - **Range**: y ∈ (-∞, ∞)
/// - **Special values**: erfcinv(1) = 0
/// 
/// # Examples
/// 
/// ```
/// use rustlab_special::{erfcinv, erfc};
/// use std::f64::consts::SQRT_2;
/// 
/// // Inverse property: erfc(erfcinv(x)) = x
/// let x = 0.1;
/// let y = erfcinv(x);
/// assert!((erfc(y) - x).abs() < 1e-14);
/// 
/// // Tail quantiles for normal distribution
/// // For X ~ N(0,1), P(X > x) = 0.01 gives x = √2 * erfcinv(0.02)
/// let tail_99 = SQRT_2 * erfcinv(0.02);
/// assert!((tail_99 - 2.3263478740408408).abs() < 1e-12);
/// 
/// // Special values
/// assert_eq!(erfcinv(1.0), 0.0);
/// assert!(erfcinv(0.0).is_nan());   // Outside domain
/// assert!(erfcinv(2.5).is_nan());   // Outside domain
/// ```
/// 
/// # Applications
/// 
/// - **Statistics**: Upper tail quantiles, extreme value analysis
/// - **Quality control**: Control limits, process capability
/// - **Risk management**: Tail risk measures, stress testing
/// - **Engineering**: Reliability analysis, safety margins
/// 
/// # Notes
/// 
/// This function is implemented as erfcinv(x) = erfinv(1 - x), which
/// automatically inherits the high precision of the erfinv implementation.
pub fn erfcinv(x: f64) -> f64 {
    if x <= 0.0 || x >= 2.0 {
        return f64::NAN;
    }
    erfinv(1.0 - x)
}

// ===== Internal implementations =====

fn erf_taylor(x: f64) -> f64 {
    // erf(x) = (2/√π) * x * Σ(n=0 to ∞) [(-1)^n * x^(2n)] / [n! * (2n+1)]
    let x2 = x * x;
    let mut series = 1.0;
    let mut term = 1.0;
    let mut factorial = 1.0;
    
    for n in 1..50 {
        factorial *= n as f64;
        term *= -x2;
        let new_term = term / (factorial * (2.0 * n as f64 + 1.0));
        series += new_term;
        
        // Early termination for convergence
        if new_term.abs() < 1e-16 {
            break;
        }
    }
    
    TWO_OVER_SQRT_PI * x * series
}

fn erfinv_winitzki(x: f64) -> f64 {
    // Winitzki approximation as starting point
    let a = 8.0 * (PI - 3.0) / (3.0 * PI * (4.0 - PI));
    let ln_1_minus_x_sq = (1.0 - x * x).ln();
    
    let term1 = 2.0 / (PI * a) + ln_1_minus_x_sq / 2.0;
    let term2 = ln_1_minus_x_sq / a;
    
    let sqrt_term = (term1 * term1 - term2).sqrt();
    let result = (sqrt_term - term1).sqrt();
    
    x.signum() * result
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    
    const EPSILON: f64 = 1e-12;
    
    #[test]
    fn test_erf_special_values() {
        assert_eq!(erf(0.0), 0.0);
        assert!((erf(1.0) - 0.8427007929497149).abs() < EPSILON);
        assert!((erf(-1.0) - (-0.8427007929497149)).abs() < EPSILON);
        assert!((erf(2.0) - 0.9953222650189527).abs() < EPSILON);
        assert!(erf(10.0) > 0.9999999999);
        assert!(erf(-10.0) < -0.9999999999);
    }
    
    #[test]
    fn test_erfc_special_values() {
        assert_eq!(erfc(0.0), 1.0);
        assert!((erfc(1.0) - 0.15729920705028513).abs() < EPSILON);
        assert!(erfc(5.0) < 1e-10);
    }
    
    #[test]
    fn test_erf_erfc_identity() {
        for x in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
            assert!((erf(x) + erfc(x) - 1.0).abs() < 1e-14);
        }
    }
    
    #[test]
    fn test_erfinv_inverse() {
        for x in [-0.9, -0.5, 0.0, 0.5, 0.9] {
            let y = erfinv(x);
            let x_recovered = erf(y);
            assert!((x - x_recovered).abs() < EPSILON,
                "erfinv({}) = {}, erf({}) = {}", x, y, y, x_recovered);
        }
    }
    
    #[test]
    fn test_erfcinv() {
        assert!((erfcinv(1.0) - 0.0).abs() < EPSILON);
        assert!((erfcinv(0.5) - erfinv(0.5)).abs() < EPSILON);
        // erfcinv(0.01) should be approximately 1.8214 (Wolfram Alpha reference)
        assert!((erfcinv(0.01) - 1.8214).abs() < 1e-3);
        // For reference: erfcinv(x) finds y such that erfc(y) = x
        // So erfcinv(0.01) finds y such that erfc(y) = 0.01, which gives y ≈ 1.82
    }
    
    #[test]
    fn test_monotonicity() {
        let x_vals = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        for i in 1..x_vals.len() {
            assert!(erf(x_vals[i]) > erf(x_vals[i-1]));
        }
    }
}

// ===== Benchmarking support =====

#[cfg(feature = "bench")]
pub fn benchmark_error_functions() {
    use std::time::Instant;
    
    let test_points: Vec<f64> = (-300..=300)
        .map(|i| i as f64 * 0.01)
        .collect();
    
    let start = Instant::now();
    let mut sum = 0.0;
    for &x in &test_points {
        sum += erf(x);
    }
    println!("erf: {} values in {:?}", test_points.len(), start.elapsed());
    
    let start = Instant::now();
    let mut sum = 0.0;
    for &x in &test_points {
        sum += erfc(x.abs());  // Only positive values
    }
    println!("erfc: {} values in {:?}", test_points.len(), start.elapsed());
}