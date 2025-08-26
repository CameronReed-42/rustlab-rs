//! Integration module for rustlab-math array operations
//! 
//! This module provides convenient methods to apply special functions element-wise
//! to rustlab-math arrays and vectors. It is only available when the "integration"
//! feature is enabled.
//! 
//! # Examples
//! 
//! ```rust,ignore
//! use rustlab_math::{vec64, array64};
//! use rustlab_special::integration::*; // Brings extension traits into scope
//! 
//! let x = vec64![1.0, 2.0, 3.0];
//! let bessel_values = x.bessel_j0();  // Apply J_0(x) element-wise
//! let erf_values = x.erf();           // Apply erf(x) element-wise
//! 
//! let matrix = array64![[1.0, 2.0], [3.0, 4.0]];
//! let gamma_matrix = matrix.gamma();  // Apply Γ(x) element-wise
//! ```

#[cfg(feature = "integration")]
use rustlab_math::{ArrayF64, VectorF64, ArrayF32, VectorF32};

/// Trait providing special functions for f64 vectors
#[cfg(feature = "integration")]
pub trait SpecialFunctionsVectorF64 {
    /// Apply Bessel function J_0(x) element-wise
    fn bessel_j0(&self) -> VectorF64;
    /// Apply Bessel function J_1(x) element-wise
    fn bessel_j1(&self) -> VectorF64;
    /// Apply Bessel function J_2(x) element-wise
    fn bessel_j2(&self) -> VectorF64;
    /// Apply Bessel function Y_0(x) element-wise
    fn bessel_y0(&self) -> VectorF64;
    /// Apply Bessel function Y_1(x) element-wise
    fn bessel_y1(&self) -> VectorF64;
    /// Apply modified Bessel function I_0(x) element-wise
    fn bessel_i0(&self) -> VectorF64;
    /// Apply modified Bessel function I_1(x) element-wise
    fn bessel_i1(&self) -> VectorF64;
    /// Apply modified Bessel function I_2(x) element-wise
    fn bessel_i2(&self) -> VectorF64;
    /// Apply modified Bessel function K_0(x) element-wise
    fn bessel_k0(&self) -> VectorF64;
    /// Apply modified Bessel function K_1(x) element-wise
    fn bessel_k1(&self) -> VectorF64;
    
    /// Apply error function erf(x) element-wise
    fn erf(&self) -> VectorF64;
    /// Apply complementary error function erfc(x) element-wise
    fn erfc(&self) -> VectorF64;
    /// Apply inverse error function erfinv(x) element-wise
    fn erfinv(&self) -> VectorF64;
    /// Apply inverse complementary error function erfcinv(x) element-wise
    fn erfcinv(&self) -> VectorF64;
    
    /// Apply gamma function Γ(x) element-wise
    fn gamma(&self) -> VectorF64;
    /// Apply log gamma function ln(Γ(x)) element-wise
    fn lgamma(&self) -> VectorF64;
    /// Apply digamma function ψ(x) element-wise
    fn digamma(&self) -> VectorF64;
    
    /// Apply spherical Bessel function j_n(x) element-wise
    fn spherical_bessel_j(&self, n: u32) -> VectorF64;
    /// Apply spherical Bessel function y_n(x) element-wise
    fn spherical_bessel_y(&self, n: u32) -> VectorF64;
}

/// Trait providing special functions for f64 arrays
#[cfg(feature = "integration")]
pub trait SpecialFunctionsArrayF64 {
    /// Apply Bessel function J_0(x) element-wise
    fn bessel_j0(&self) -> ArrayF64;
    /// Apply Bessel function J_1(x) element-wise
    fn bessel_j1(&self) -> ArrayF64;
    /// Apply Bessel function J_2(x) element-wise
    fn bessel_j2(&self) -> ArrayF64;
    /// Apply Bessel function Y_0(x) element-wise
    fn bessel_y0(&self) -> ArrayF64;
    /// Apply Bessel function Y_1(x) element-wise
    fn bessel_y1(&self) -> ArrayF64;
    /// Apply modified Bessel function I_0(x) element-wise
    fn bessel_i0(&self) -> ArrayF64;
    /// Apply modified Bessel function I_1(x) element-wise
    fn bessel_i1(&self) -> ArrayF64;
    /// Apply modified Bessel function I_2(x) element-wise
    fn bessel_i2(&self) -> ArrayF64;
    /// Apply modified Bessel function K_0(x) element-wise
    fn bessel_k0(&self) -> ArrayF64;
    /// Apply modified Bessel function K_1(x) element-wise
    fn bessel_k1(&self) -> ArrayF64;
    
    /// Apply error function erf(x) element-wise
    fn erf(&self) -> ArrayF64;
    /// Apply complementary error function erfc(x) element-wise
    fn erfc(&self) -> ArrayF64;
    /// Apply inverse error function erfinv(x) element-wise
    fn erfinv(&self) -> ArrayF64;
    /// Apply inverse complementary error function erfcinv(x) element-wise
    fn erfcinv(&self) -> ArrayF64;
    
    /// Apply gamma function Γ(x) element-wise
    fn gamma(&self) -> ArrayF64;
    /// Apply log gamma function ln(Γ(x)) element-wise
    fn lgamma(&self) -> ArrayF64;
    /// Apply digamma function ψ(x) element-wise
    fn digamma(&self) -> ArrayF64;
    
    /// Apply spherical Bessel function j_n(x) element-wise
    fn spherical_bessel_j(&self, n: u32) -> ArrayF64;
    /// Apply spherical Bessel function y_n(x) element-wise
    fn spherical_bessel_y(&self, n: u32) -> ArrayF64;
}

#[cfg(feature = "integration")]
impl SpecialFunctionsVectorF64 for VectorF64 {
    fn bessel_j0(&self) -> VectorF64 {
        self.map(|x| crate::bessel_j0(x))
    }
    
    fn bessel_j1(&self) -> VectorF64 {
        self.map(|x| crate::bessel_j1(x))
    }
    
    fn bessel_j2(&self) -> VectorF64 {
        self.map(|x| crate::bessel_j2(x))
    }
    
    fn bessel_y0(&self) -> VectorF64 {
        self.map(|x| crate::bessel_y0(x))
    }
    
    fn bessel_y1(&self) -> VectorF64 {
        self.map(|x| crate::bessel_y1(x))
    }
    
    fn bessel_i0(&self) -> VectorF64 {
        self.map(|x| crate::bessel_i0(x))
    }
    
    fn bessel_i1(&self) -> VectorF64 {
        self.map(|x| crate::bessel_i1(x))
    }
    
    fn bessel_i2(&self) -> VectorF64 {
        self.map(|x| crate::bessel_i2(x))
    }
    
    fn bessel_k0(&self) -> VectorF64 {
        self.map(|x| crate::bessel_k0(x))
    }
    
    fn bessel_k1(&self) -> VectorF64 {
        self.map(|x| crate::bessel_k1(x))
    }
    
    fn erf(&self) -> VectorF64 {
        self.map(|x| crate::erf(x))
    }
    
    fn erfc(&self) -> VectorF64 {
        self.map(|x| crate::erfc(x))
    }
    
    fn erfinv(&self) -> VectorF64 {
        self.map(|x| crate::erfinv(x))
    }
    
    fn erfcinv(&self) -> VectorF64 {
        self.map(|x| crate::erfcinv(x))
    }
    
    fn gamma(&self) -> VectorF64 {
        self.map(|x| crate::gamma(x))
    }
    
    fn lgamma(&self) -> VectorF64 {
        self.map(|x| crate::lgamma(x))
    }
    
    fn digamma(&self) -> VectorF64 {
        self.map(|x| crate::digamma(x))
    }
    
    fn spherical_bessel_j(&self, n: u32) -> VectorF64 {
        self.map(|x| crate::spherical_bessel_j(n, x))
    }
    
    fn spherical_bessel_y(&self, n: u32) -> VectorF64 {
        self.map(|x| crate::spherical_bessel_y(n, x))
    }
}

#[cfg(feature = "integration")]
impl SpecialFunctionsArrayF64 for ArrayF64 {
    fn bessel_j0(&self) -> ArrayF64 {
        self.map_elements(|x| crate::bessel_j0(x))
    }
    
    fn bessel_j1(&self) -> ArrayF64 {
        self.map_elements(|x| crate::bessel_j1(x))
    }
    
    fn bessel_j2(&self) -> ArrayF64 {
        self.map_elements(|x| crate::bessel_j2(x))
    }
    
    fn bessel_y0(&self) -> ArrayF64 {
        self.map_elements(|x| crate::bessel_y0(x))
    }
    
    fn bessel_y1(&self) -> ArrayF64 {
        self.map_elements(|x| crate::bessel_y1(x))
    }
    
    fn bessel_i0(&self) -> ArrayF64 {
        self.map_elements(|x| crate::bessel_i0(x))
    }
    
    fn bessel_i1(&self) -> ArrayF64 {
        self.map_elements(|x| crate::bessel_i1(x))
    }
    
    fn bessel_i2(&self) -> ArrayF64 {
        self.map_elements(|x| crate::bessel_i2(x))
    }
    
    fn bessel_k0(&self) -> ArrayF64 {
        self.map_elements(|x| crate::bessel_k0(x))
    }
    
    fn bessel_k1(&self) -> ArrayF64 {
        self.map_elements(|x| crate::bessel_k1(x))
    }
    
    fn erf(&self) -> ArrayF64 {
        self.map_elements(|x| crate::erf(x))
    }
    
    fn erfc(&self) -> ArrayF64 {
        self.map_elements(|x| crate::erfc(x))
    }
    
    fn erfinv(&self) -> ArrayF64 {
        self.map_elements(|x| crate::erfinv(x))
    }
    
    fn erfcinv(&self) -> ArrayF64 {
        self.map_elements(|x| crate::erfcinv(x))
    }
    
    fn gamma(&self) -> ArrayF64 {
        self.map_elements(|x| crate::gamma(x))
    }
    
    fn lgamma(&self) -> ArrayF64 {
        self.map_elements(|x| crate::lgamma(x))
    }
    
    fn digamma(&self) -> ArrayF64 {
        self.map_elements(|x| crate::digamma(x))
    }
    
    fn spherical_bessel_j(&self, n: u32) -> ArrayF64 {
        self.map_elements(|x| crate::spherical_bessel_j(n, x))
    }
    
    fn spherical_bessel_y(&self, n: u32) -> ArrayF64 {
        self.map_elements(|x| crate::spherical_bessel_y(n, x))
    }
}

/// Trait providing special functions for f32 vectors
#[cfg(feature = "integration")]
pub trait SpecialFunctionsVectorF32 {
    /// Apply Bessel function J_0(x) element-wise
    fn bessel_j0(&self) -> VectorF32;
    /// Apply error function erf(x) element-wise  
    fn erf(&self) -> VectorF32;
    /// Apply gamma function Γ(x) element-wise
    fn gamma(&self) -> VectorF32;
}

/// Trait providing special functions for f32 arrays
#[cfg(feature = "integration")]
pub trait SpecialFunctionsArrayF32 {
    /// Apply Bessel function J_0(x) element-wise
    fn bessel_j0(&self) -> ArrayF32;
    /// Apply error function erf(x) element-wise
    fn erf(&self) -> ArrayF32;
    /// Apply gamma function Γ(x) element-wise
    fn gamma(&self) -> ArrayF32;
}

#[cfg(feature = "integration")]
impl SpecialFunctionsVectorF32 for VectorF32 {
    fn bessel_j0(&self) -> VectorF32 {
        self.map(|x| crate::bessel_j0(x as f64) as f32)
    }
    
    fn erf(&self) -> VectorF32 {
        self.map(|x| crate::erf(x as f64) as f32)
    }
    
    fn gamma(&self) -> VectorF32 {
        self.map(|x| crate::gamma(x as f64) as f32)
    }
}

#[cfg(feature = "integration")]
impl SpecialFunctionsArrayF32 for ArrayF32 {
    fn bessel_j0(&self) -> ArrayF32 {
        self.map(|x| crate::bessel_j0(x as f64) as f32)
    }
    
    fn erf(&self) -> ArrayF32 {
        self.map(|x| crate::erf(x as f64) as f32)
    }
    
    fn gamma(&self) -> ArrayF32 {
        self.map(|x| crate::gamma(x as f64) as f32)
    }
}

/// Convenience functions for common operations
#[cfg(feature = "integration")]
pub mod convenience {
    use super::*;
    
    /// Apply beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b) element-wise to two vectors
    pub fn beta_vectors(a: &VectorF64, b: &VectorF64) -> VectorF64 {
        if a.len() != b.len() {
            panic!("Vector lengths must match for beta function");
        }
        
        VectorF64::from_fn(a.len(), |i| {
            let a_val = a.get(i).unwrap();
            let b_val = b.get(i).unwrap();
            crate::beta(a_val, b_val)
        })
    }
    
    /// Apply beta function element-wise to two arrays
    pub fn beta_arrays(a: &ArrayF64, b: &ArrayF64) -> ArrayF64 {
        if a.shape() != b.shape() {
            panic!("Array shapes must match for beta function");
        }
        
        let (rows, cols) = a.shape();
        ArrayF64::from_fn(rows, cols, |i, j| {
            let a_val = a.get(i, j).unwrap();
            let b_val = b.get(i, j).unwrap();
            crate::beta(a_val, b_val)
        })
    }
    
    /// Normal CDF using error function: Φ(x) = (1 + erf(x/√2))/2
    pub fn normal_cdf(x: &VectorF64) -> VectorF64 {
        x.map(|val| 0.5 * (1.0 + crate::erf(val / std::f64::consts::SQRT_2)))
    }
    
    /// Normal PDF using exponential and constants
    pub fn normal_pdf(x: &VectorF64) -> VectorF64 {
        let coeff = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        x.map(|val| coeff * (-0.5 * val * val).exp())
    }
}

#[cfg(all(test, feature = "integration"))]
mod tests {
    use super::*;
    use rustlab_math::{vec64, array64};
    use approx::assert_relative_eq;
    
    #[test]
    fn test_vector_bessel_functions() {
        let x = vec64![1.0, 2.0, 3.0];
        
        let j0_vals = x.bessel_j0();
        let expected_j0_0 = crate::bessel_j0(1.0);
        assert_relative_eq!(j0_vals.get(0).unwrap(), expected_j0_0, epsilon = 1e-14);
        
        let erf_vals = x.erf();
        let expected_erf_0 = crate::erf(1.0);
        assert_relative_eq!(erf_vals.get(0).unwrap(), expected_erf_0, epsilon = 1e-14);
    }
    
    #[test]
    fn test_array_special_functions() {
        let matrix = array64![[1.0, 2.0], [3.0, 4.0]];
        
        let gamma_matrix = matrix.gamma();
        let expected_gamma_00 = crate::gamma(1.0);
        let expected_gamma_11 = crate::gamma(4.0);
        
        assert_relative_eq!(gamma_matrix.get(0, 0).unwrap(), expected_gamma_00, epsilon = 1e-14);
        assert_relative_eq!(gamma_matrix.get(1, 1).unwrap(), expected_gamma_11, epsilon = 1e-14);
    }
    
    #[test]
    fn test_convenience_functions() {
        let a = vec64![1.0, 2.0, 3.0];
        let b = vec64![1.0, 1.0, 1.0];
        
        let beta_vals = convenience::beta_vectors(&a, &b);
        let expected_beta_0 = crate::beta(1.0, 1.0);
        assert_relative_eq!(beta_vals.get(0).unwrap(), expected_beta_0, epsilon = 1e-14);
        
        let x = vec64![0.0, 1.0, -1.0];
        let cdf_vals = convenience::normal_cdf(&x);
        assert_relative_eq!(cdf_vals.get(0).unwrap(), 0.5, epsilon = 1e-14); // Φ(0) = 0.5
    }
    
    #[test]
    fn test_f32_integration() {
        use rustlab_math::{VectorF32, ArrayF32};
        
        let x_f32 = VectorF32::from_slice(&[1.0_f32, 2.0_f32]);
        let j0_vals = x_f32.bessel_j0();
        
        // Should be approximately the same as f64 version, within f32 precision
        let expected = crate::bessel_j0(1.0) as f32;
        assert!((j0_vals.get(0).unwrap() - expected).abs() < 1e-6);
    }
}