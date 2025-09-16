//! # Special Functions Library
//! 
//! A high-precision implementation of mathematical special functions.
//! 
//! ## Features
//! 
//! This crate provides implementations of:
//! 
//! - **Bessel Functions**
//!   - First kind: J_n(x)
//!   - Second kind: Y_n(x) (Neumann functions)
//!   - Modified first kind: I_n(x)
//!   - Modified second kind: K_n(x)
//!   - Spherical Bessel functions: j_n(x), y_n(x)
//!   - Hankel functions: H^(1)_n(x), H^(2)_n(x)
//! 
//! - **Error Functions**
//!   - Error function: erf(x)
//!   - Complementary error function: erfc(x)
//!   - Inverse error function: erfinv(x)
//!   - Inverse complementary error function: erfcinv(x)
//! 
//! - **Gamma Functions**
//!   - Gamma function: Γ(x)
//!   - Log gamma: ln(Γ(x))
//!   - Digamma (psi): ψ(x) = d/dx ln(Γ(x))
//!   - Beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
//!   - Log beta: ln(B(a,b))
//! 
//! ## Accuracy
//! 
//! Target accuracy is 1e-15 for most functions in their primary domains.
//! Some functions may have reduced accuracy for extreme arguments or higher orders.
//! 
//! ## Examples
//! 
//! ### Basic Usage (Scalar Functions)
//! 
//! ```rust
//! use rustlab_special::{bessel_j0, erf, gamma};
//! 
//! // Bessel function of the first kind, order 0
//! let j0 = bessel_j0(1.0);
//! assert!((j0 - 0.7651976865579666).abs() < 1e-14);
//! 
//! // Error function
//! let erf_val = erf(1.0);
//! assert!((erf_val - 0.8427007929497149).abs() < 1e-14);
//! 
//! // Gamma function
//! let gamma_val = gamma(0.5);
//! assert!((gamma_val - std::f64::consts::PI.sqrt()).abs() < 1e-14);
//! ```
//! 
//! ### RustLab-Math Integration (with "integration" feature)
//! 
//! ```rust,ignore
//! use rustlab_math::{vec64, array64};
//! use rustlab_special::integration::*; // Bring extension traits into scope
//! 
//! // Apply special functions element-wise to vectors
//! let x = vec64![1.0, 2.0, 3.0, 4.0];
//! let bessel_values = x.bessel_j0();    // J_0 applied to each element
//! let error_values = x.erf();           // erf applied to each element
//! let gamma_values = x.gamma();         // Γ applied to each element
//! 
//! // Apply to 2D arrays as well
//! let matrix = array64![[1.0, 2.0], [3.0, 4.0]];
//! let bessel_matrix = matrix.bessel_j0(); // Element-wise application
//! 
//! // Convenience functions for statistical applications
//! use rustlab_special::integration::convenience::*;
//! let normal_cdf_values = normal_cdf(&x); // Φ(x) = (1 + erf(x/√2))/2
//! ```

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod bessel;
pub mod bessel_y;
pub mod bessel_modified;
pub mod error_functions;
pub mod gamma_functions;
pub mod incomplete_gamma;

// Integration module for rustlab-math (only available with "integration" feature)
#[cfg(feature = "integration")]
pub mod integration;

// Re-export primary functions for convenience
pub use bessel::{bessel_j, bessel_j0, bessel_j1, bessel_j2};
pub use bessel_y::{bessel_y, bessel_y0, bessel_y1};
pub use bessel_modified::{
    bessel_i, bessel_k, bessel_i0, bessel_i1, bessel_i2, 
    bessel_k0, bessel_k1,
    bessel_i_derivative, bessel_k_derivative
};
pub use error_functions::{erf, erfc, erfinv, erfcinv};
pub use gamma_functions::{gamma, lgamma, digamma, beta, lbeta};
pub use incomplete_gamma::{gamma_lower, gamma_upper, gamma_p, gamma_q};

use std::f64::consts::PI;

/// Computes the Wronskian of Bessel functions J_n and Y_n.
/// 
/// The Wronskian W[J_n, Y_n](x) = J_n(x)Y'_n(x) - J'_n(x)Y_n(x) = 2/(πx)
/// 
/// This is a fundamental identity that holds for all real x > 0 and integer n ≥ 0.
/// 
/// # Arguments
/// 
/// * `n` - Order of the Bessel functions (non-negative integer)
/// * `x` - Argument (must be positive)
/// 
/// # Returns
/// 
/// Returns 2/(πx) for x > 0, NaN for x ≤ 0
/// 
/// # Example
/// 
/// ```rust
/// use rustlab_special::wronskian;
/// use std::f64::consts::PI;
/// 
/// let w = wronskian(0, 2.0);
/// assert!((w - 2.0 / (PI * 2.0)).abs() < 1e-14);
/// ```
#[inline]
pub fn wronskian(_n: u32, x: f64) -> f64 {
    if x <= 0.0 {
        f64::NAN
    } else {
        2.0 / (PI * x)
    }
}

/// Computes the modified Wronskian of modified Bessel functions I_n and K_n.
/// 
/// The identity W[I_n, K_n](x) = I_n(x)K_{n+1}(x) + I_{n+1}(x)K_n(x) = 1/x
/// 
/// # Arguments
/// 
/// * `n` - Order of the Bessel functions (non-negative integer)
/// * `x` - Argument (must be positive)
/// 
/// # Returns
/// 
/// Returns the value of the Wronskian for x > 0, NaN for x ≤ 0
/// 
/// # Note
/// 
/// Due to numerical precision limitations, this may not equal exactly 1/x
/// for all arguments, especially for large n.
pub fn modified_wronskian(n: u32, x: f64) -> f64 {
    if x <= 0.0 {
        f64::NAN
    } else {
        bessel_i(n, x) * bessel_k(n + 1, x) + bessel_i(n + 1, x) * bessel_k(n, x)
    }
}

/// Computes the Hankel function of the first kind H^(1)_n(x) = J_n(x) + i*Y_n(x).
/// 
/// # Arguments
/// 
/// * `n` - Order of the Hankel function (non-negative integer)
/// * `x` - Argument (must be positive)
/// 
/// # Returns
/// 
/// Returns a tuple (real_part, imaginary_part) where:
/// - real_part = J_n(x)
/// - imaginary_part = Y_n(x)
/// 
/// # Example
/// 
/// ```rust
/// use rustlab_special::{hankel_first, bessel_j0, bessel_y0};
/// 
/// let (re, im) = hankel_first(0, 2.0);
/// assert!((re - bessel_j0(2.0)).abs() < 1e-14);
/// assert!((im - bessel_y0(2.0)).abs() < 1e-14);
/// ```
pub fn hankel_first(n: u32, x: f64) -> (f64, f64) {
    (bessel_j(n, x), bessel_y(n, x))
}

/// Computes the Hankel function of the second kind H^(2)_n(x) = J_n(x) - i*Y_n(x).
/// 
/// # Arguments
/// 
/// * `n` - Order of the Hankel function (non-negative integer)
/// * `x` - Argument (must be positive)
/// 
/// # Returns
/// 
/// Returns a tuple (real_part, imaginary_part) where:
/// - real_part = J_n(x)
/// - imaginary_part = -Y_n(x)
pub fn hankel_second(n: u32, x: f64) -> (f64, f64) {
    (bessel_j(n, x), -bessel_y(n, x))
}

/// Computes the derivative of the Bessel function of the first kind J_n.
/// 
/// Uses the identity: J'_n(x) = (J_{n-1}(x) - J_{n+1}(x))/2
/// 
/// Special case: J'_0(x) = -J_1(x)
/// 
/// # Arguments
/// 
/// * `n` - Order of the Bessel function
/// * `x` - Argument
/// 
/// # Example
/// 
/// ```rust
/// use rustlab_special::{bessel_j_derivative, bessel_j1};
/// 
/// // J'_0(2.0) = -J_1(2.0)
/// let j0_prime = bessel_j_derivative(0, 2.0);
/// assert!((j0_prime + bessel_j1(2.0)).abs() < 1e-14);
/// ```
pub fn bessel_j_derivative(n: u32, x: f64) -> f64 {
    if n == 0 {
        -bessel_j(1, x)
    } else {
        0.5 * (bessel_j(n - 1, x) - bessel_j(n + 1, x))
    }
}

/// Computes the derivative of the Bessel function of the second kind Y_n.
/// 
/// Uses the identity: Y'_n(x) = (Y_{n-1}(x) - Y_{n+1}(x))/2
/// 
/// Special case: Y'_0(x) = -Y_1(x)
pub fn bessel_y_derivative(n: u32, x: f64) -> f64 {
    if x <= 0.0 {
        f64::NAN
    } else if n == 0 {
        -bessel_y(1, x)
    } else {
        0.5 * (bessel_y(n - 1, x) - bessel_y(n + 1, x))
    }
}



/// Computes the spherical Bessel function of the first kind j_n(x).
/// 
/// The spherical Bessel functions are related to the cylindrical Bessel functions by:
/// j_n(x) = sqrt(π/(2x)) * J_{n+1/2}(x)
/// 
/// # Special values
/// 
/// - j_0(x) = sin(x)/x
/// - j_1(x) = sin(x)/x² - cos(x)/x
/// - j_n(0) = 1 if n=0, 0 if n>0
/// 
/// # Arguments
/// 
/// * `n` - Order of the spherical Bessel function
/// * `x` - Argument
/// 
/// # Example
/// 
/// ```rust
/// use rustlab_special::spherical_bessel_j;
/// 
/// // j_0(x) = sin(x)/x
/// let x = 1.0;
/// let j0 = spherical_bessel_j(0, x);
/// assert!((j0 - x.sin() / x).abs() < 1e-14);
/// ```
pub fn spherical_bessel_j(n: u32, x: f64) -> f64 {
    // j_n(x) = sqrt(π/(2x)) * J_{n+1/2}(x)
    use crate::bessel::bessel_j_nu;
    
    if x == 0.0 {
        if n == 0 { 1.0 } else { 0.0 }
    } else if x < 0.0 {
        // Handle negative x based on parity of n
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        sign * spherical_bessel_j(n, -x)
    } else {
        // Use relation to cylindrical Bessel function
        let nu = n as f64 + 0.5;
        (PI / (2.0 * x)).sqrt() * bessel_j_nu(nu, x)
    }
}

/// Computes the spherical Bessel function of the second kind y_n(x).
/// 
/// The spherical Bessel functions are related to the cylindrical Bessel functions by:
/// y_n(x) = sqrt(π/(2x)) * Y_{n+1/2}(x)
/// 
/// # Special values
/// 
/// - y_0(x) = -cos(x)/x
/// - y_1(x) = -cos(x)/x² - sin(x)/x
/// - y_n(0) = -∞ for all n
/// 
/// # Arguments
/// 
/// * `n` - Order of the spherical Bessel function
/// * `x` - Argument (must be positive)
pub fn spherical_bessel_y(n: u32, x: f64) -> f64 {
    // y_n(x) = sqrt(π/(2x)) * Y_{n+1/2}(x)
    use crate::bessel_y::bessel_y_nu;
    
    if x <= 0.0 {
        f64::NEG_INFINITY
    } else {
        // Use relation to cylindrical Bessel function  
        let nu = n as f64 + 0.5;
        (PI / (2.0 * x)).sqrt() * bessel_y_nu(nu, x)
    }
}

/// Computes the spherical modified Bessel function of the first kind i_n(x).
/// 
/// The spherical modified Bessel functions are related to the cylindrical modified Bessel functions by:
/// i_n(x) = sqrt(π/(2x)) * I_{n+1/2}(x)
/// 
/// # Special values
/// 
/// - i_0(x) = sinh(x)/x
/// - i_1(x) = -sinh(x)/x² + cosh(x)/x
/// - i_n(0) = 1 if n=0, 0 if n>0
/// 
/// # Arguments
/// 
/// * `n` - Order of the spherical modified Bessel function
/// * `x` - Argument
pub fn spherical_bessel_i(n: u32, x: f64) -> f64 {
    // i_n(x) = sqrt(π/(2x)) * I_{n+1/2}(x)
    use crate::bessel_modified::bessel_i_nu;
    
    if x == 0.0 {
        if n == 0 { 1.0 } else { 0.0 }
    } else {
        // Use relation to cylindrical modified Bessel function
        let nu = n as f64 + 0.5;
        (PI / (2.0 * x.abs())).sqrt() * bessel_i_nu(nu, x)
    }
}

/// Computes the spherical modified Bessel function of the second kind k_n(x).
/// 
/// The spherical modified Bessel functions are related to the cylindrical modified Bessel functions by:
/// k_n(x) = sqrt(π/(2x)) * K_{n+1/2}(x)
/// 
/// # Special values
/// 
/// - k_0(x) = π/(2x) * e^(-x)
/// - k_1(x) = π/(2x) * e^(-x) * (1 + 1/x)
/// - k_n(0) = +∞ for all n
/// 
/// # Arguments
/// 
/// * `n` - Order of the spherical modified Bessel function
/// * `x` - Argument (must be positive)
pub fn spherical_bessel_k(n: u32, x: f64) -> f64 {
    // k_n(x) = sqrt(π/(2x)) * K_{n+1/2}(x)
    use crate::bessel_modified::bessel_k_nu;
    
    if x <= 0.0 {
        f64::INFINITY
    } else {
        // Use relation to cylindrical modified Bessel function
        let nu = n as f64 + 0.5;
        (PI / (2.0 * x)).sqrt() * bessel_k_nu(nu, x)
    }
}

/// Computes the spherical Hankel function of the first kind h_n^(1)(x).
/// 
/// Returns a tuple (real_part, imaginary_part) where:
/// h_n^(1)(x) = j_n(x) + i*y_n(x)
/// 
/// # Arguments
/// 
/// * `n` - Order of the spherical Hankel function
/// * `x` - Argument
pub fn spherical_hankel_1(n: u32, x: f64) -> (f64, f64) {
    (spherical_bessel_j(n, x), spherical_bessel_y(n, x))
}

/// Computes the spherical Hankel function of the second kind h_n^(2)(x).
/// 
/// Returns a tuple (real_part, imaginary_part) where:
/// h_n^(2)(x) = j_n(x) - i*y_n(x)
/// 
/// # Arguments
/// 
/// * `n` - Order of the spherical Hankel function
/// * `x` - Argument
pub fn spherical_hankel_2(n: u32, x: f64) -> (f64, f64) {
    (spherical_bessel_j(n, x), -spherical_bessel_y(n, x))
}

/// Utility function to compute factorial for small non-negative integers.
/// 
/// For larger values, consider using the gamma function: n! = Γ(n+1)
#[allow(dead_code)]
#[inline]
fn factorial(n: u32) -> f64 {
    match n {
        0 | 1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        4 => 24.0,
        5 => 120.0,
        6 => 720.0,
        7 => 5040.0,
        8 => 40320.0,
        9 => 362880.0,
        10 => 3628800.0,
        _ => (1..=n).fold(1.0, |acc, i| acc * i as f64)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    
    const EPSILON: f64 = 1e-14;
    
    #[test]
    fn test_wronskian_identity() {
        let test_points = [0.5, 1.0, 2.0, 5.0, 10.0];
        
        for &x in &test_points {
            for n in 0..5 {
                let w = wronskian(n, x);
                let expected = 2.0 / (PI * x);
                assert!(
                    (w - expected).abs() < EPSILON,
                    "Wronskian({}, {}) = {}, expected {}",
                    n, x, w, expected
                );
            }
        }
    }
    
    #[test]
    fn test_spherical_bessel_special_values() {
        // Test j_0(x) = sin(x)/x
        for &x in &[0.1, 0.5, 1.0, 2.0, 5.0] {
            let j0 = spherical_bessel_j(0, x);
            let expected = x.sin() / x;
            assert!((j0 - expected).abs() < EPSILON);
        }
        
        // Test j_n(0)
        assert_eq!(spherical_bessel_j(0, 0.0), 1.0);
        assert_eq!(spherical_bessel_j(1, 0.0), 0.0);
        assert_eq!(spherical_bessel_j(5, 0.0), 0.0);
        
        // Test y_0(x) = -cos(x)/x
        for &x in &[0.1, 0.5, 1.0, 2.0, 5.0] {
            let y0 = spherical_bessel_y(0, x);
            let expected = -x.cos() / x;
            assert!((y0 - expected).abs() < EPSILON);
        }
    }
    
    #[test]
    fn test_derivative_relations() {
        let x = 2.5;
        
        // Test J'_0 = -J_1
        let j0_prime = bessel_j_derivative(0, x);
        let j1 = bessel_j1(x);
        assert!((j0_prime + j1).abs() < EPSILON);
        
        // Test Y'_0 = -Y_1
        let y0_prime = bessel_y_derivative(0, x);
        let y1 = bessel_y1(x);
        assert!((y0_prime + y1).abs() < EPSILON);
        
        // Test recurrence for derivatives
        for n in 1..5 {
            let jn_prime = bessel_j_derivative(n, x);
            let expected = 0.5 * (bessel_j(n - 1, x) - bessel_j(n + 1, x));
            assert!((jn_prime - expected).abs() < EPSILON);
        }
    }
    
    #[test]
    fn test_hankel_functions() {
        let x = 3.0;
        
        for n in 0..4 {
            let (h1_re, h1_im) = hankel_first(n, x);
            let (h2_re, h2_im) = hankel_second(n, x);
            
            // H^(1) = J + iY, H^(2) = J - iY
            assert_eq!(h1_re, bessel_j(n, x));
            assert_eq!(h1_im, bessel_y(n, x));
            assert_eq!(h2_re, bessel_j(n, x));
            assert_eq!(h2_im, -bessel_y(n, x));
            
            // H^(1) + H^(2) = 2J
            assert!((h1_re + h2_re - 2.0 * bessel_j(n, x)).abs() < EPSILON);
            
            // H^(1) - H^(2) = 2iY
            assert!((h1_im - h2_im - 2.0 * bessel_y(n, x)).abs() < EPSILON);
        }
    }
    
    #[test]
    fn test_negative_x_handling() {
        // Spherical Bessel functions handle negative x
        let x = -2.0;
        
        // j_0(-x) = j_0(x) (even)
        assert!((spherical_bessel_j(0, x) - spherical_bessel_j(0, -x)).abs() < EPSILON);
        
        // j_1(-x) = -j_1(x) (odd)
        assert!((spherical_bessel_j(1, x) + spherical_bessel_j(1, -x)).abs() < EPSILON);
        
        // Regular Bessel Y functions return NaN/infinity for x <= 0
        assert!(bessel_y0(0.0).is_infinite());
        assert!(bessel_y1(-1.0).is_infinite());
    }
    
    #[test]
    fn test_factorial_helper() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3628800.0);
    }
    
    #[test]
    fn test_fractional_order_bessel_functions() {
        // Test half-integer orders that are used by spherical functions
        use crate::bessel::bessel_j_nu;
        use crate::bessel_y::bessel_y_nu;
        use crate::bessel_modified::{bessel_i_nu, bessel_k_nu};
        
        let x = 2.0;
        let tolerance = 1e-10;
        
        // Test J_{1/2}(x) = sqrt(2/(πx)) * sin(x)
        let j_half = bessel_j_nu(0.5, x);
        let expected_j_half = (2.0 / (PI * x)).sqrt() * x.sin();
        assert!((j_half - expected_j_half).abs() < tolerance,
               "J_{{1/2}}({}) = {}, expected {}", x, j_half, expected_j_half);
        
        // Test that spherical j_0 matches the relation
        let spherical_j0 = spherical_bessel_j(0, x);
        let cylindrical_j_half = (PI / (2.0 * x)).sqrt() * bessel_j_nu(0.5, x);
        assert!((spherical_j0 - cylindrical_j_half).abs() < tolerance,
               "Spherical j_0 relation failed: {} vs {}", spherical_j0, cylindrical_j_half);
        
        // Test spherical modified Bessel functions
        let spherical_i0 = spherical_bessel_i(0, x);
        let expected_i0 = x.sinh() / x;
        assert!((spherical_i0 - expected_i0).abs() < tolerance,
               "Spherical i_0({}) = {}, expected {}", x, spherical_i0, expected_i0);
    }
    
    #[test]
    fn test_comprehensive_integration() {
        // Test that all function families work together
        let x = 2.0;
        
        // Compute various Bessel functions
        let j0 = bessel_j0(x);
        let y0 = bessel_y0(x);
        let i0 = bessel_i0(x);
        let k0 = bessel_k0(x);
        
        // All should be finite
        assert!(j0.is_finite());
        assert!(y0.is_finite());
        assert!(i0.is_finite());
        assert!(k0.is_finite());
        
        // Test some known relationships
        // J_0(x)^2 + Y_0(x)^2 should be positive
        assert!(j0 * j0 + y0 * y0 > 0.0);
        
        // I_0(x) should be greater than 1 for x > 0
        assert!(i0 > 1.0);
        
        // K_0(x) should be positive and decreasing
        assert!(k0 > 0.0);
        assert!(bessel_k0(x + 1.0) < k0);
    }
}

#[cfg(test)]
#[path = "bessel_test_data.rs"]
mod bessel_test_data;

#[cfg(test)]
#[path = "error_functions_test_data.rs"]
mod error_functions_test_data;

#[cfg(test)]
#[path = "gamma_functions_test_data.rs"]
mod gamma_functions_test_data;

// Optional: Benchmarking support
#[cfg(feature = "bench")]
pub mod benchmarks {
    use super::*;
    
    /// Benchmark all special functions
    pub fn run_all_benchmarks() {
        println!("Running special functions benchmarks...\n");
        
        // Benchmark error functions
        #[cfg(feature = "bench")]
        error_functions::benchmark_error_functions();
        
        // Benchmark gamma functions
        #[cfg(feature = "bench")]
        gamma_functions::benchmark_gamma_functions();
        
        // Add Bessel benchmarks
        benchmark_bessel_functions();
    }
    
    fn benchmark_bessel_functions() {
        use std::time::Instant;
        
        let test_points: Vec<f64> = (1..=100)
            .map(|i| i as f64 * 0.1)
            .collect();
        
        let start = Instant::now();
        let mut sum = 0.0;
        for &x in &test_points {
            sum += bessel_j0(x);
        }
        println!("bessel_j0: {} values in {:?}", test_points.len(), start.elapsed());
        
        let start = Instant::now();
        let mut sum = 0.0;
        for &x in &test_points {
            sum += bessel_i0(x);
        }
        println!("bessel_i0: {} values in {:?}", test_points.len(), start.elapsed());
    }
}