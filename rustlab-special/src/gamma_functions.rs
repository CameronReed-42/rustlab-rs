//! Gamma function family: gamma, lgamma, digamma, beta, lbeta
//!
//! This module provides high-precision implementations of the gamma function and
//! its related functions, which are fundamental to many areas of mathematics and
//! statistics including probability distributions, combinatorics, and special
//! function theory.
//!
//! ## Mathematical Background
//!
//! The gamma function extends the factorial function to real and complex numbers:
//!
//! ```text
//! Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt    for Re(x) > 0
//! ```
//!
//! ## Key Properties
//!
//! - **Factorial extension**: Γ(n) = (n-1)! for positive integers n
//! - **Functional equation**: Γ(x+1) = x·Γ(x)
//! - **Reflection formula**: Γ(x)Γ(1-x) = π/sin(πx)
//! - **Duplication formula**: Γ(x)Γ(x+1/2) = √π/2^(2x-1) Γ(2x)
//! - **Special values**: Γ(1/2) = √π, Γ(1) = 1, Γ(2) = 1
//!
//! ## Implementation Strategy
//!
//! - **Small positive x (0.5 ≤ x ≤ 20)**: 9-term Lanczos approximation (g=7)
//! - **Large positive x (x > 20)**: Stirling's asymptotic expansion
//! - **Negative x**: Reflection formula using Γ(x)Γ(1-x) = π/sin(πx)
//! - **Logarithmic forms**: Computed directly to avoid overflow
//! - **Target accuracy**: 1e-14 relative error in primary domains
//!
//! ## Applications
//!
//! - **Statistics**: Beta, chi-squared, F, and t distributions
//! - **Combinatorics**: Generalized factorials and binomial coefficients
//! - **Number theory**: Riemann zeta function, prime number theorem
//! - **Physics**: Quantum mechanics, statistical mechanics, field theory

use std::f64::consts::{PI};

/// Euler-Mascheroni constant
#[allow(dead_code)]
const EULER_GAMMA: f64 = 0.5772156649015328606065120900824024;



/// Natural log of pi
const LN_PI: f64 = 1.1447298858494001741434273513530587;

/// Natural log of sqrt(2*pi)
const LN_SQRT_2PI: f64 = 0.9189385332046727417803297364056176;

// ===== Lanczos Coefficients =====

/// Lanczos g parameter
const LANCZOS_G: f64 = 7.000000000;

/// Lanczos coefficients for g = 7
const LANCZOS_COEFFS: [f64; 9] = [
    0.9999999999998099,
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.98436957801957e-6,
    1.5056327351493114e-7,
];

// ===== Stirling Expansion Coefficients =====

/// Stirling expansion coefficients for log(gamma)
const STIRLING_COEFFS: [f64; 6] = [
    0.08333333333333333333333333333333333333,  // B_{2}/(2*1*(2*1-1))
    -0.00277777777777777777777777777777777778,  // B_{4}/(2*2*(2*2-1))
    0.00079365079365079365079365079365079365,  // B_{6}/(2*3*(2*3-1))
    -0.00059523809523809523809523809523809524,  // B_{8}/(2*4*(2*4-1))
    0.00084175084175084175084175084175084175,  // B_{10}/(2*5*(2*5-1))
    -0.00191752691752691752691752691752691753,  // B_{12}/(2*6*(2*6-1))
];

// ===== Digamma Coefficients =====

/// Asymptotic expansion coefficients for digamma
const DIGAMMA_ASYMP_COEFFS: [f64; 6] = [
    0.08333333333333333333333333333333333333,  // B_{2}/(2*1)
    -0.00833333333333333333333333333333333333,  // B_{4}/(2*2)
    0.00396825396825396825396825396825396825,  // B_{6}/(2*3)
    -0.00416666666666666666666666666666666667,  // B_{8}/(2*4)
    0.00757575757575757575757575757575757576,  // B_{10}/(2*5)
    -0.02109279609279609279609279609279609279,  // B_{12}/(2*6)
];

// ===== Gamma Function Implementation =====

/// Gamma function Γ(x)
///
/// Computes the gamma function, which extends the factorial function to real
/// numbers and is fundamental to many probability distributions and mathematical
/// functions.
///
/// # Arguments
///
/// * `x` - The argument at which to evaluate Γ(x). Can be any real number except
///         non-positive integers.
///
/// # Returns
///
/// The value of Γ(x) with target accuracy of 1e-14. Returns:
/// - `+∞` for x = 0
/// - `NaN` for negative integers
/// - Finite values for all other real x
///
/// # Mathematical Definition
///
/// ```text
/// Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt    for Re(x) > 0
/// ```
///
/// # Special Values
///
/// - Γ(1) = 1
/// - Γ(2) = 1  
/// - Γ(3) = 2
/// - Γ(4) = 6
/// - Γ(1/2) = √π ≈ 1.7724538509055159
/// - Γ(n) = (n-1)! for positive integers n
///
/// # Algorithm Selection
///
/// - **Positive x ∈ [0.5, 20]**: 9-term Lanczos approximation for optimal accuracy
/// - **Large x > 20**: Stirling's asymptotic expansion for efficiency
/// - **Negative x**: Reflection formula Γ(x) = π / [sin(πx) · Γ(1-x)]
///
/// # Examples
///
/// ```
/// use rustlab_special::gamma;
/// use std::f64::consts::PI;
///
/// // Factorial property: Γ(n) = (n-1)!
/// assert!((gamma(5.0) - 24.0).abs() < 1e-14);  // Γ(5) = 4! = 24
///
/// // Half-integer: Γ(1/2) = √π
/// assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-14);
///
/// // Functional equation: Γ(x+1) = x·Γ(x)
/// let x = 2.3;
/// assert!((gamma(x + 1.0) - x * gamma(x)).abs() < 1e-13);
///
/// // Special cases
/// assert_eq!(gamma(1.0), 1.0);
/// assert_eq!(gamma(2.0), 1.0);
/// assert!(gamma(0.0).is_infinite());
/// assert!(gamma(-2.0).is_nan());  // Negative integer
/// ```
///
/// # Applications
///
/// - **Statistics**: Gamma, chi-squared, beta, t-distributions
/// - **Combinatorics**: Generalized binomial coefficients
/// - **Number theory**: Analytic continuation of factorial
/// - **Physics**: Partition functions, quantum field theory
///
/// # Notes
///
/// The implementation handles the full real line except for poles at
/// non-positive integers. For very large arguments (|x| > 170), consider
/// using `lgamma` to avoid overflow.
pub fn gamma(x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY;
    }
    
    // For negative integers, gamma is undefined
    if x < 0.0 && x == x.floor() {
        return f64::NAN;
    }
    
    if x < 0.5 {
        // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else if x > 20.0 {
        // Use Stirling's approximation for large x
        stirling_gamma(x)
    } else {
        // Lanczos approximation
        lanczos_gamma(x)
    }
}

/// Natural logarithm of gamma function ln(Γ(x))
///
/// Computes ln(Γ(x)) directly without computing Γ(x) first, which prevents
/// overflow for large arguments and provides better numerical stability.
///
/// # Arguments
///
/// * `x` - The argument at which to evaluate ln(Γ(x)). Can be any real number
///         except non-positive integers.
///
/// # Returns
///
/// The value of ln(Γ(x)) with target accuracy of 1e-14. Returns:
/// - `+∞` for x = 0
/// - `NaN` for negative integers
/// - Finite values for all other real x
///
/// # Algorithm Selection
///
/// - **Positive x ∈ [0.5, 20]**: Lanczos approximation computed directly
/// - **Large x > 20**: Stirling's asymptotic expansion for ln(Γ(x))
/// - **Negative x**: Reflection formula using ln|Γ(x)| = ln(π) - ln|sin(πx)| - ln(Γ(1-x))
///
/// # Examples
///
/// ```
/// use rustlab_special::{lgamma, gamma};
/// use std::f64::consts::PI;
///
/// // Should match log of gamma function (when finite)
/// let x = 3.5;
/// assert!((lgamma(x) - gamma(x).ln()).abs() < 1e-14);
///
/// // Handles large arguments without overflow
/// let large_x = 100.0;
/// let lg_100 = lgamma(large_x);
/// assert!(lg_100.is_finite());
/// // gamma(100) would overflow, but lgamma works fine
///
/// // Stirling approximation check for large x:
/// // ln(Γ(x)) ≈ (x-0.5)ln(x) - x + 0.5ln(2π) + O(1/x)
/// let x = 50.0;
/// let stirling_approx = (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln();
/// assert!((lgamma(x) - stirling_approx).abs() < 0.01);
///
/// // Special values
/// assert!((lgamma(1.0) - 0.0).abs() < 1e-14);     // ln(Γ(1)) = ln(1) = 0
/// assert!((lgamma(0.5) - (PI.ln() / 2.0)).abs() < 1e-14); // ln(Γ(1/2)) = ln(√π)
/// ```
///
/// # Applications
///
/// - **Large factorials**: ln(n!) = ln(Γ(n+1)) without overflow
/// - **Statistical computations**: Log-likelihood functions
/// - **Numerical stability**: Ratios of gamma functions
/// - **Physics**: Statistical mechanics with large particle numbers
///
/// # Notes
///
/// This function is essential when working with large arguments where
/// Γ(x) would overflow. It's also more accurate than computing ln(gamma(x))
/// due to direct implementation of logarithmic forms.
pub fn lgamma(x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY;
    }
    
    if x < 0.0 && x == x.floor() {
        return f64::NAN;
    }
    
    if x < 0.5 {
        // log|Gamma(x)| = log(pi) - log|sin(pi*x)| - lgamma(1-x)
        LN_PI - (PI * x).sin().abs().ln() - lgamma(1.0 - x)
    } else if x > 20.0 {
        // Stirling's approximation
        stirling_lgamma(x)
    } else {
        // Lanczos approximation
        lanczos_lgamma(x)
    }
}

/// Digamma function ψ(x) = d/dx ln(Γ(x))
///
/// Computes the digamma function, which is the logarithmic derivative of the
/// gamma function. It appears in many statistical distributions and mathematical
/// physics applications.
///
/// # Arguments
///
/// * `x` - The argument at which to evaluate ψ(x). Can be any real number except
///         non-positive integers.
///
/// # Returns
///
/// The value of ψ(x) with target accuracy of 1e-14. Returns:
/// - `-∞` for x = 0
/// - `NaN` for negative integers
/// - Finite values for all other real x
///
/// # Mathematical Definition
///
/// ```text
/// ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
/// ```
///
/// # Algorithm Selection
///
/// - **Small positive x < 1**: Recurrence relation ψ(x) = ψ(x+1) - 1/x
/// - **Moderate positive x (1 ≤ x < 10)**: Upward recurrence to x ≥ 10
/// - **Large positive x ≥ 10**: 6-term asymptotic expansion
/// - **Negative x**: Reflection formula ψ(1-x) - ψ(x) = π cot(πx)
///
/// # Key Properties
///
/// - **Recurrence**: ψ(x+1) = ψ(x) + 1/x
/// - **Reflection**: ψ(1-x) - ψ(x) = π cot(πx)
/// - **Asymptotic**: ψ(x) ~ ln(x) - 1/(2x) - 1/(12x²) + ... for large x
///
/// # Examples
///
/// ```
/// use rustlab_special::digamma;
/// use std::f64::consts::PI;
///
/// // Recurrence relation: ψ(x+1) = ψ(x) + 1/x
/// let x = 2.5;
/// assert!((digamma(x + 1.0) - digamma(x) - 1.0/x).abs() < 1e-14);
///
/// // Special values (from mathematical tables)
/// assert!((digamma(1.0) - (-0.5772156649015329)).abs() < 1e-14); // -γ (Euler-Mascheroni)
/// assert!((digamma(0.5) - (-1.9635100260214235)).abs() < 1e-13); // -γ - 2ln(2)
///
/// // Large argument asymptotic behavior: ψ(x) ≈ ln(x)
/// let large_x = 100.0;
/// assert!((digamma(large_x) - large_x.ln()).abs() < 0.01);
///
/// // Integer values: ψ(n) = -γ + Σ_{k=1}^{n-1} 1/k for n ≥ 1
/// let psi_3 = digamma(3.0);
/// let harmonic_2 = 1.0 + 0.5; // H_2 = 1 + 1/2
/// let expected = -0.5772156649015329 + harmonic_2;
/// assert!((psi_3 - expected).abs() < 1e-14);
/// ```
///
/// # Applications
///
/// - **Statistics**: Beta and Dirichlet distributions, maximum likelihood estimation
/// - **Physics**: Quantum field theory, statistical mechanics
/// - **Number theory**: Harmonic numbers, Riemann zeta function
/// - **Special functions**: Polygamma functions, Hurwitz zeta function
///
/// # Notes
///
/// The digamma function has a rich connection to harmonic numbers:
/// ψ(n) = -γ + H_{n-1} for positive integers n, where H_k is the k-th harmonic number.
pub fn digamma(x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    
    if x < 0.0 && x == x.floor() {
        return f64::NAN;
    }
    
    if x < 0.0 {
        // Reflection formula: psi(1-x) - psi(x) = pi * cot(pi*x)
        digamma(1.0 - x) - PI / (PI * x).tan()
    } else if x < 1.0 {
        // Use recurrence: psi(x) = psi(x+1) - 1/x
        digamma(x + 1.0) - 1.0 / x
    } else if x < 10.0 {
        // Use recurrence to get to range where asymptotic works well
        let mut result = 0.0;
        let mut z = x;
        while z < 10.0 {
            result -= 1.0 / z;
            z += 1.0;
        }
        result + digamma_asymptotic(z)
    } else {
        // Asymptotic expansion
        digamma_asymptotic(x)
    }
}

/// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
///
/// Computes the beta function, which is closely related to the binomial coefficient
/// and appears in many probability distributions, particularly the beta distribution.
///
/// # Arguments
///
/// * `a` - First parameter, must be positive
/// * `b` - Second parameter, must be positive
///
/// # Returns
///
/// The value of B(a,b) with target accuracy of 1e-14.
/// Returns `+∞` or `NaN` for invalid arguments.
///
/// # Mathematical Definition
///
/// ```text
/// B(a,b) = Γ(a)Γ(b)/Γ(a+b) = ∫₀¹ t^(a-1)(1-t)^(b-1) dt
/// ```
///
/// # Key Properties
///
/// - **Symmetry**: B(a,b) = B(b,a)
/// - **Binomial relation**: B(n,k) = 1/[(n+k-1) × C(n+k-2,k-1)] for integers
/// - **Integral representation**: B(a,b) = ∫₀¹ t^(a-1)(1-t)^(b-1) dt
/// - **Recurrence**: B(a,b+1) = b/(a+b) × B(a,b)
///
/// # Algorithm
///
/// Computed as exp(lgamma(a) + lgamma(b) - lgamma(a+b)) to avoid overflow
/// and maximize numerical stability.
///
/// # Examples
///
/// ```
/// use rustlab_special::beta;
///
/// // Symmetry property
/// let a = 2.5;
/// let b = 3.7;
/// assert!((beta(a, b) - beta(b, a)).abs() < 1e-15);
///
/// // Integer cases: B(m,n) = (m-1)!(n-1)!/(m+n-1)!
/// // B(3,4) = 2!×3!/6! = 2×6/720 = 1/60
/// assert!((beta(3.0, 4.0) - 1.0/60.0).abs() < 1e-14);
///
/// // Special case: B(1,1) = 1
/// assert!((beta(1.0, 1.0) - 1.0).abs() < 1e-14);
///
/// // Half-integer case: B(1/2, 1/2) = π
/// assert!((beta(0.5, 0.5) - std::f64::consts::PI).abs() < 1e-14);
///
/// // Connection to binomial coefficient: B(n,k+1) = 1/[n × C(n-1,k)]
/// let n = 10.0;
/// let k = 3.0;
/// let beta_val = beta(n, k + 1.0);
/// // C(9,3) = 84, so B(10,4) = 1/(10×84) = 1/840
/// assert!((beta_val - 1.0/840.0).abs() < 1e-14);
/// ```
///
/// # Applications
///
/// - **Statistics**: Beta distribution, Dirichlet distribution
/// - **Bayesian statistics**: Prior and posterior distributions
/// - **Combinatorics**: Generalized binomial coefficients
/// - **Physics**: Statistical mechanics, partition functions
/// - **Engineering**: Reliability analysis, quality control
///
/// # Notes
///
/// For large arguments, this function uses logarithmic computation to prevent
/// overflow, making it stable even when individual gamma function values
/// would be too large to represent.
pub fn beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        if a == a.floor() || b == b.floor() {
            return f64::INFINITY;
        }
        return f64::NAN;
    }
    (lgamma(a) + lgamma(b) - lgamma(a + b)).exp()
}

/// Natural logarithm of beta function ln(B(a,b))
///
/// Computes ln(B(a,b)) directly without computing B(a,b) first, providing
/// numerical stability for large arguments and preventing overflow.
///
/// # Arguments
///
/// * `a` - First parameter, must be positive
/// * `b` - Second parameter, must be positive
///
/// # Returns
///
/// The value of ln(B(a,b)) with target accuracy of 1e-14.
///
/// # Mathematical Definition
///
/// ```text
/// ln(B(a,b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
/// ```
///
/// # Algorithm
///
/// Computed as lgamma(a) + lgamma(b) - lgamma(a+b), leveraging the
/// high-precision lgamma implementation.
///
/// # Examples
///
/// ```
/// use rustlab_special::{lbeta, beta};
///
/// // Should match log of beta function (when finite)
/// let a = 2.3;
/// let b = 4.7;
/// assert!((lbeta(a, b) - beta(a, b).ln()).abs() < 1e-14);
///
/// // Handles large arguments without overflow
/// let large_a = 100.0;
/// let large_b = 200.0;
/// let lbeta_val = lbeta(large_a, large_b);
/// assert!(lbeta_val.is_finite());
/// // beta(100, 200) would underflow, but lbeta works fine
///
/// // Symmetry
/// assert!((lbeta(a, b) - lbeta(b, a)).abs() < 1e-15);
///
/// // Special case: ln(B(1,1)) = ln(1) = 0
/// assert!((lbeta(1.0, 1.0) - 0.0).abs() < 1e-14);
/// ```
///
/// # Applications
///
/// - **Statistics**: Log-likelihood functions for beta distributions
/// - **Bayesian inference**: Evidence computation, model comparison
/// - **Numerical analysis**: Stable computation of beta function ratios
/// - **Machine learning**: Dirichlet-multinomial models
///
/// # Notes
///
/// This function is essential for statistical computations involving the
/// beta distribution when the parameters are large, as it avoids the
/// numerical issues that would arise from computing beta(a,b) and then
/// taking its logarithm.
pub fn lbeta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        if a == a.floor() || b == b.floor() {
            return f64::INFINITY;
        }
        return f64::NAN;
    }
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

// ===== Internal implementations =====

fn lanczos_gamma(x: f64) -> f64 {
    lanczos_lgamma(x).exp()
}

fn lanczos_lgamma(x: f64) -> f64 {
    let z = x - 1.0;
    
    // Compute series sum
    let mut sum = LANCZOS_COEFFS[0];
    for i in 1..LANCZOS_COEFFS.len() {
        sum += LANCZOS_COEFFS[i] / (z + i as f64);
    }
    
    let tmp = z + LANCZOS_G + 0.5;
    LN_SQRT_2PI + (z + 0.5) * tmp.ln() - tmp + sum.ln()
}

fn stirling_gamma(x: f64) -> f64 {
    stirling_lgamma(x).exp()
}

fn stirling_lgamma(x: f64) -> f64 {
    // Stirling's approximation:
    // log(Gamma(x)) ~ (x-1/2)*log(x) - x + log(2*pi)/2 + sum(B_2n/(2n*(2n-1)*x^(2n-1)))
    
    let mut sum = 0.0;
    let x2 = x * x;
    let mut x_pow = x;
    
    for &coeff in &STIRLING_COEFFS {
        sum += coeff / x_pow;
        x_pow *= x2;
    }
    
    (x - 0.5) * x.ln() - x + LN_SQRT_2PI + sum
}

fn digamma_asymptotic(x: f64) -> f64 {
    // Asymptotic expansion:
    // psi(x) ~ log(x) - 1/(2x) - sum(B_2n/(2n*x^(2n)))
    
    let mut sum = x.ln() - 0.5 / x;
    let x2 = x * x;
    let mut x2n = x2;
    
    for &coeff in &DIGAMMA_ASYMP_COEFFS {
        sum -= coeff / x2n;
        x2n *= x2;
    }
    
    sum
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    
    const EPSILON: f64 = 1e-12;  // Relaxed slightly for numerical precision
    
    #[test]
    fn test_gamma_special_values() {
        assert!((gamma(1.0) - 1.0).abs() < EPSILON);
        assert!((gamma(2.0) - 1.0).abs() < EPSILON);
        assert!((gamma(3.0) - 2.0).abs() < EPSILON);
        assert!((gamma(4.0) - 6.0).abs() < EPSILON);
        assert!((gamma(0.5) - PI.sqrt()).abs() < EPSILON);
    }
    
    #[test]
    fn test_lgamma_special_values() {
        assert!(lgamma(1.0).abs() < EPSILON);
        assert!(lgamma(2.0).abs() < EPSILON);
        assert!((lgamma(3.0) - 2.0_f64.ln()).abs() < EPSILON);
        assert!((lgamma(0.5) - PI.sqrt().ln()).abs() < EPSILON);
    }
    
    #[test]
    fn test_gamma_reflection() {
        for x in [0.3, 0.7, 1.3, 2.7] {
            let product = gamma(x) * gamma(1.0 - x);
            let expected = PI / (PI * x).sin();
            assert!((product - expected).abs() < EPSILON * expected.abs());
        }
    }
    
    #[test]
    fn test_digamma_special_values() {
        assert!((digamma(1.0) - (-EULER_GAMMA)).abs() < EPSILON);
        assert!((digamma(2.0) - (1.0 - EULER_GAMMA)).abs() < EPSILON);
        // For digamma(0.5) = -γ - 2*ln(2)
        let expected = -EULER_GAMMA - 2.0 * 2.0_f64.ln();
        assert!((digamma(0.5) - expected).abs() < EPSILON);
    }
    
    #[test]
    fn test_beta_function() {
        // B(a,b) = B(b,a)
        assert!((beta(2.0, 3.0) - beta(3.0, 2.0)).abs() < EPSILON);
        
        // B(1,1) = 1
        assert!((beta(1.0, 1.0) - 1.0).abs() < EPSILON);
        
        // B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
        let a = 2.5;
        let b = 3.5;
        let expected = gamma(a) * gamma(b) / gamma(a + b);
        assert!((beta(a, b) - expected).abs() < EPSILON * expected);
    }
    
    #[test]
    fn test_digamma_recurrence() {
        // psi(x+1) = psi(x) + 1/x
        for x in [0.5, 1.5, 2.5, 3.5] {
            let diff = digamma(x + 1.0) - digamma(x);
            assert!((diff - 1.0 / x).abs() < EPSILON);
        }
    }
}

#[cfg(feature = "bench")]
pub fn benchmark_gamma_functions() {
    use std::time::Instant;
    
    let test_points: Vec<f64> = (1..=1000)
        .map(|i| i as f64 * 0.1)
        .collect();
    
    let start = Instant::now();
    let mut sum = 0.0;
    for &x in &test_points {
        sum += gamma(x);
    }
    println!("gamma: {} values in {:?}", test_points.len(), start.elapsed());
    
    let start = Instant::now();
    let mut sum = 0.0;
    for &x in &test_points {
        sum += lgamma(x);
    }
    println!("lgamma: {} values in {:?}", test_points.len(), start.elapsed());
    
    let start = Instant::now();
    let mut sum = 0.0;
    for &x in &test_points {
        sum += digamma(x);
    }
    println!("digamma: {} values in {:?}", test_points.len(), start.elapsed());
}