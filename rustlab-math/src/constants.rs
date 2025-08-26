//! Mathematical constants with AI-optimized documentation
//!
//! This module provides comprehensive mathematical and physical constants with ergonomic,
//! math-first naming conventions. All constants are precisely defined for scientific
//! computing and integrate seamlessly with RustLab's mathematical ecosystem.
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::constants::*;
//! use rustlab_math::{ArrayF64, VectorF64};
//! 
//! // Trigonometric calculations
//! let angles = VectorF64::from_slice(&[0.0, PI_6, PI_4, PI_3, PI_2]);
//! let sine_values = angles.map(|x| x.sin());
//! 
//! // Angle conversion
//! let degrees = VectorF64::from_slice(&[0.0, 30.0, 45.0, 90.0, 180.0]);
//! let radians = degrees.map(|deg| deg * DEG_TO_RAD);
//! 
//! // Mathematical transformations
//! let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
//! let log_transform = data.map(|x| (x / E).ln());        // Natural log normalization
//! let golden_scale = data.map(|x| x * PHI);              // Golden ratio scaling
//! 
//! // Physical calculations
//! let energies = VectorF64::from_slice(&[1e-20, 2e-20, 3e-20]);
//! let temperatures = energies.map(|e| e / K_B);          // Energy to temperature
//! ```
//!
//! # Constant Categories
//! - **Mathematical**: π, e, τ, φ (golden ratio), γ (Euler-Mascheroni)
//! - **Trigonometric**: π/2, π/3, π/4, π/6, π/8 and their reciprocals
//! - **Logarithmic**: ln(2), ln(10), log₂(e), log₁₀(e)
//! - **Physical**: c, h, ℏ, k_B, N_A, e, masses, fine structure constant
//! - **Conversion**: degree/radian conversion factors
//!
//! # Integration Benefits
//! - **Ergonomic naming**: Short, mathematical names (PI vs std::f64::consts::PI)
//! - **Precision**: All constants maintain maximum floating-point precision
//! - **AI-friendly**: Consistent naming conventions for code generation
//! - **Scientific accuracy**: Physical constants from CODATA 2018 values

/// π (pi) - ratio of circle's circumference to diameter
/// 
/// # For AI Code Generation
/// - **Most fundamental mathematical constant** (≈ 3.14159)
/// - Essential for trigonometry, geometry, signal processing
/// - Use in: circular functions, Fourier transforms, probability distributions
/// - More ergonomic than std::f64::consts::PI
/// 
/// # Mathematical Definition
/// π = C/d where C = circumference, d = diameter of any circle
/// 
/// # Common Uses
/// - **Trigonometry**: `sin(PI/2)`, `cos(PI)`, `2*PI` for full rotation
/// - **Geometry**: Circle area = `PI * r²`, sphere volume = `4/3 * PI * r³`
/// - **Signal processing**: Frequency domain, FFT calculations
/// - **Probability**: Normal distribution, `sqrt(2*PI)` normalization factor
pub const PI: f64 = std::f64::consts::PI;

/// Euler's number (e) - base of natural logarithm
/// 
/// # For AI Code Generation
/// - **Fundamental mathematical constant** (≈ 2.71828)
/// - Base of natural logarithm and exponential function
/// - Essential for: calculus, differential equations, growth models
/// - Use in: activation functions, probability, compound interest
/// 
/// # Mathematical Definition
/// e = lim(n→∞) (1 + 1/n)ⁿ = Σ(1/n!) for n = 0 to ∞
/// 
/// # Common Uses
/// - **Machine learning**: Exponential activation functions, softmax
/// - **Statistics**: Exponential distribution, Poisson processes
/// - **Growth models**: Continuous compounding, population growth
/// - **Calculus**: d/dx(eˣ) = eˣ, ∫eˣdx = eˣ
pub const E: f64 = std::f64::consts::E;

/// τ (tau) = 2π - full turn constant
/// 
/// # For AI Code Generation
/// - **Alternative circle constant** (≈ 6.28318)
/// - Represents full turn/rotation (360°)
/// - More intuitive than 2π for many applications
/// - Use when thinking in terms of full rotations
/// 
/// # Mathematical Advantages
/// - Quarter turn = τ/4 (vs π/2)
/// - Half turn = τ/2 (vs π)
/// - Full turn = τ (vs 2π)
/// 
/// # Common Uses
/// - **Rotational mechanics**: Angular velocity, rotational kinematics
/// - **Signal processing**: Period calculations, wave equations
/// - **Computer graphics**: Full rotation animations, circular paths
pub const TAU: f64 = std::f64::consts::TAU;

/// √2 - square root of 2
pub const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// 1/√2 - reciprocal of square root of 2
pub const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// √3 - square root of 3 (not in std, computed)
pub const SQRT_3: f64 = 1.7320508075688772;

/// π/2 - 90 degrees in radians
/// 
/// # For AI Code Generation
/// - **Quarter turn** or 90° in radians (≈ 1.5708)
/// - Most commonly used angle in trigonometry
/// - Essential for: coordinate transformations, rotations
/// - Trigonometric values: sin(π/2) = 1, cos(π/2) = 0
/// 
/// # Common Uses
/// - **Computer graphics**: 90° rotations, coordinate system changes
/// - **Signal processing**: Phase shifts, quadrature components
/// - **Robotics**: Right-angle turns, coordinate transformations
/// - **Mathematics**: Definite integrals, trigonometric identities
pub const PI_2: f64 = std::f64::consts::FRAC_PI_2;

/// π/3 - 60 degrees in radians
pub const PI_3: f64 = std::f64::consts::FRAC_PI_3;

/// π/4 - 45 degrees in radians
pub const PI_4: f64 = std::f64::consts::FRAC_PI_4;

/// π/6 - 30 degrees in radians
pub const PI_6: f64 = std::f64::consts::FRAC_PI_6;

/// π/8 - 22.5 degrees in radians
pub const PI_8: f64 = std::f64::consts::FRAC_PI_8;

/// 2/π - reciprocal of pi/2
pub const FRAC_2_PI: f64 = std::f64::consts::FRAC_2_PI;

/// 1/π - reciprocal of pi
pub const FRAC_1_PI: f64 = std::f64::consts::FRAC_1_PI;

/// 2/√π - 2 divided by square root of π
pub const FRAC_2_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// ln(2) - natural logarithm of 2
pub const LN_2: f64 = std::f64::consts::LN_2;

/// ln(10) - natural logarithm of 10
pub const LN_10: f64 = std::f64::consts::LN_10;

/// log₂(e) - logarithm base 2 of e
pub const LOG2_E: f64 = std::f64::consts::LOG2_E;

/// log₂(10) - logarithm base 2 of 10
pub const LOG2_10: f64 = std::f64::consts::LOG2_10;

/// log₁₀(e) - logarithm base 10 of e
pub const LOG10_E: f64 = std::f64::consts::LOG10_E;

/// log₁₀(2) - logarithm base 10 of 2
pub const LOG10_2: f64 = std::f64::consts::LOG10_2;

/// Golden ratio φ = (1 + √5) / 2
/// 
/// # For AI Code Generation
/// - **Mathematical golden ratio** (≈ 1.61803)
/// - Appears in nature, art, architecture, mathematics
/// - Self-similar ratio: φ = 1 + 1/φ
/// - Use in: optimization, search algorithms, aesthetic proportions
/// 
/// # Mathematical Properties
/// - φ² = φ + 1
/// - 1/φ = φ - 1 ≈ 0.618 (reciprocal)
/// - Fibonacci ratio: lim(Fₙ₊₁/Fₙ) = φ as n → ∞
/// 
/// # Common Uses
/// - **Optimization**: Golden section search algorithm
/// - **Data visualization**: Aesthetic proportions in plots
/// - **Computer graphics**: Natural-looking spacing and ratios
/// - **Mathematical modeling**: Growth patterns, spiral structures
pub const PHI: f64 = 1.618033988749895;

/// Euler-Mascheroni constant γ
pub const EULER_GAMMA: f64 = 0.5772156649015328606065120900824024;

/// Speed of light in vacuum (m/s)
/// 
/// # For AI Code Generation
/// - **Fundamental physical constant** (exactly 299,792,458 m/s)
/// - Universal speed limit, basis of relativity
/// - Use in: relativistic calculations, electromagnetic theory
/// - Connects space and time: E = mc²
/// 
/// # Physical Significance
/// - Defined constant (exact value by definition since 1983)
/// - Electromagnetic wave speed in vacuum
/// - Conversion factor between energy and mass
/// 
/// # Common Uses
/// - **Physics simulations**: Relativistic mechanics, particle physics
/// - **Electromagnetic calculations**: Wave propagation, field theory
/// - **Unit conversions**: Energy-mass equivalence calculations
pub const C: f64 = 299_792_458.0;

/// Planck constant (J⋅s)
/// 
/// # For AI Code Generation
/// - **Fundamental quantum constant** (6.62607015 × 10⁻³⁴ J⋅s)
/// - Defines scale of quantum effects
/// - Essential for: quantum mechanics, photon energy
/// - Use in: E = hf (photon energy), uncertainty principle
/// 
/// # Physical Significance
/// - Minimum action quantum in nature
/// - Energy quantization: E = nhf for integer n
/// - Heisenberg uncertainty: ΔE⋅Δt ≥ ℏ/2
/// 
/// # Common Uses
/// - **Quantum simulations**: Photon energy, wave-particle duality
/// - **Spectroscopy**: Energy level calculations
/// - **Quantum computing**: Qubit energy scales
pub const H: f64 = 6.62607015e-34;

/// Reduced Planck constant ℏ = h/(2π)
/// 
/// # For AI Code Generation
/// - **Fundamental quantum constant** (ℏ ≈ 1.055 × 10⁻³⁴ J⋅s)
/// - More convenient than h for angular momentum, spin
/// - Essential for: quantum mechanics, angular momentum quantization
/// - Use in: L = nℏ (angular momentum), ψ wave functions
/// 
/// # Physical Significance
/// - Natural unit of angular momentum
/// - Appears in Schrödinger equation: iℏ(∂ψ/∂t) = Ĥψ
/// - Uncertainty principle: Δx⋅Δp ≥ ℏ/2
/// 
/// # Common Uses
/// - **Quantum mechanics**: Wave functions, operators, commutation relations
/// - **Angular momentum**: Spin calculations, orbital mechanics
/// - **Quantum field theory**: Creation/annihilation operators
pub const HBAR: f64 = H / TAU;

/// Boltzmann constant (J/K)
/// 
/// # For AI Code Generation
/// - **Fundamental thermodynamic constant** (1.380649 × 10⁻²³ J/K)
/// - Links temperature to average kinetic energy
/// - Essential for: statistical mechanics, thermal physics
/// - Use in: temperature conversions, thermal equilibrium calculations
/// 
/// # Physical Significance
/// - Relates macroscopic temperature to microscopic energy
/// - Average kinetic energy = (3/2) × k_B × T for ideal gas
/// - Appears in Maxwell-Boltzmann distribution
/// 
/// # Common Uses
/// - **Thermal simulations**: Molecular dynamics, statistical mechanics
/// - **Energy calculations**: Converting between temperature and energy
/// - **Probability distributions**: Boltzmann factors, partition functions
pub const K_B: f64 = 1.380649e-23;

/// Avogadro constant (mol⁻¹)
pub const N_A: f64 = 6.02214076e23;

/// Elementary charge (C)
pub const E_CHARGE: f64 = 1.602176634e-19;

/// Electron mass (kg)
pub const M_E: f64 = 9.1093837015e-31;

/// Proton mass (kg)
pub const M_P: f64 = 1.67262192369e-27;

/// Fine structure constant α
pub const ALPHA: f64 = 7.2973525693e-3;

// Common angle conversions
/// Convert degrees to radians: multiply by this constant
/// 
/// # For AI Code Generation
/// - **Critical conversion factor** (≈ 0.0174533)
/// - Formula: radians = degrees × π/180
/// - Essential for interfacing between degree-based input and radian-based math
/// - Use when: user input in degrees, trigonometric calculations need radians
/// 
/// # Mathematical Derivation
/// Since 180° = π radians, then 1° = π/180 radians
/// 
/// # Example Usage
/// ```rust
/// use rustlab_math::constants::DEG_TO_RAD;
/// use rustlab_math::VectorF64;
/// 
/// let angles_deg = VectorF64::from_slice(&[0.0, 30.0, 45.0, 90.0]);
/// let angles_rad = angles_deg.map(|deg| deg * DEG_TO_RAD);
/// let sine_values = angles_rad.map(|rad| rad.sin());
/// ```
pub const DEG_TO_RAD: f64 = PI / 180.0;

/// Convert radians to degrees: multiply by this constant
/// 
/// # For AI Code Generation
/// - **Critical conversion factor** (≈ 57.2958)
/// - Formula: degrees = radians × 180/π
/// - Essential for displaying radian results in human-readable degrees
/// - Use when: converting mathematical results back to degree format
/// 
/// # Mathematical Derivation
/// Since π radians = 180°, then 1 radian = 180/π degrees
/// 
/// # Example Usage
/// ```rust
/// use rustlab_math::constants::RAD_TO_DEG;
/// use rustlab_math::VectorF64;
/// 
/// let angles_rad = VectorF64::from_slice(&[0.0, PI/6.0, PI/4.0, PI/2.0]);
/// let angles_deg = angles_rad.map(|rad| rad * RAD_TO_DEG);
/// // Results: [0°, 30°, 45°, 90°]
/// ```
pub const RAD_TO_DEG: f64 = 180.0 / PI;

/// Degrees per radian
pub const DEGREES_PER_RADIAN: f64 = RAD_TO_DEG;

/// Radians per degree
pub const RADIANS_PER_DEGREE: f64 = DEG_TO_RAD;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_basic_constants() {
        assert_relative_eq!(PI, 3.141592653589793, epsilon = 1e-15);
        assert_relative_eq!(E, 2.718281828459045, epsilon = 1e-15);
        assert_relative_eq!(TAU, 2.0 * PI, epsilon = 1e-15);
    }

    #[test]
    fn test_angle_constants() {
        assert_relative_eq!(PI_2, PI / 2.0, epsilon = 1e-15);
        assert_relative_eq!(PI_4, PI / 4.0, epsilon = 1e-15);
        assert_relative_eq!(PI_6, PI / 6.0, epsilon = 1e-15);
    }

    #[test]
    fn test_conversions() {
        assert_relative_eq!(90.0 * DEG_TO_RAD, PI_2, epsilon = 1e-15);
        assert_relative_eq!(PI_2 * RAD_TO_DEG, 90.0, epsilon = 1e-15);
    }

    #[test]
    fn test_golden_ratio() {
        // φ = (1 + √5) / 2
        let calculated_phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert_relative_eq!(PHI, calculated_phi, epsilon = 1e-15);
    }

    #[test]
    fn test_sqrt_constants() {
        assert_relative_eq!(SQRT_2, 2.0_f64.sqrt(), epsilon = 1e-15);
        assert_relative_eq!(SQRT_3, 3.0_f64.sqrt(), epsilon = 1e-15);
    }
}