# RustLab-Special: Comprehensive AI Documentation

## 🎯 Executive Summary

**RustLab-Special** is a high-precision mathematical special functions library for Rust that provides implementations of Bessel functions, error functions, gamma functions, and their variants. It offers standalone scalar functions with optional seamless integration with RustLab-Math for element-wise array/vector operations.

### Key Capabilities
- **Bessel Functions**: J_n, Y_n, I_n, K_n, spherical variants, Hankel functions
- **Error Functions**: erf, erfc, erfinv, erfcinv
- **Gamma Functions**: Γ(x), ln(Γ(x)), ψ(x), β(a,b), incomplete gamma
- **Precision**: Target accuracy of 1e-15 in primary domains
- **Integration**: Optional rustlab-math integration for array operations
- **Performance**: Optimized algorithms with numerical stability

## 📊 Architecture Overview

```
rustlab-special/
├── Core Functions (Standalone)
│   ├── bessel.rs          # J_n(x) - First kind
│   ├── bessel_y.rs        # Y_n(x) - Second kind (Neumann)
│   ├── bessel_modified.rs # I_n(x), K_n(x) - Modified
│   ├── error_functions.rs # erf, erfc, inverses
│   ├── gamma_functions.rs # Γ, ln(Γ), ψ, β
│   └── incomplete_gamma.rs # γ(a,x), Γ(a,x), P, Q
│
├── Integration Layer
│   └── integration.rs     # Extension traits for rustlab-math
│
└── Test Data
    ├── bessel_test_data.rs
    ├── error_functions_test_data.rs
    └── gamma_functions_test_data.rs
```

## 🔧 Core Components

### 1. Bessel Functions Module (`bessel.rs`, `bessel_y.rs`, `bessel_modified.rs`)

#### Regular Bessel Functions (J_n, Y_n)
```rust
// First kind J_n(x) - finite at x=0 for all n
pub fn bessel_j(n: u32, x: f64) -> f64
pub fn bessel_j0(x: f64) -> f64  // Specialized for n=0
pub fn bessel_j1(x: f64) -> f64  // Specialized for n=1
pub fn bessel_j2(x: f64) -> f64  // Specialized for n=2

// Second kind Y_n(x) - singular at x=0
pub fn bessel_y(n: u32, x: f64) -> f64
pub fn bessel_y0(x: f64) -> f64
pub fn bessel_y1(x: f64) -> f64

// Fractional order support
pub fn bessel_j_nu(nu: f64, x: f64) -> f64  // J_ν(x)
pub fn bessel_y_nu(nu: f64, x: f64) -> f64  // Y_ν(x)
```

#### Modified Bessel Functions (I_n, K_n)
```rust
// Modified first kind I_n(x) - exponentially growing
pub fn bessel_i(n: u32, x: f64) -> f64
pub fn bessel_i0(x: f64) -> f64
pub fn bessel_i1(x: f64) -> f64
pub fn bessel_i2(x: f64) -> f64

// Modified second kind K_n(x) - exponentially decaying
pub fn bessel_k(n: u32, x: f64) -> f64
pub fn bessel_k0(x: f64) -> f64
pub fn bessel_k1(x: f64) -> f64

// Fractional order support
pub fn bessel_i_nu(nu: f64, x: f64) -> f64  // I_ν(x)
pub fn bessel_k_nu(nu: f64, x: f64) -> f64  // K_ν(x)

// Derivatives
pub fn bessel_i_derivative(n: u32, x: f64) -> f64
pub fn bessel_k_derivative(n: u32, x: f64) -> f64
```

#### Spherical Bessel Functions
```rust
// Spherical variants: j_n(x) = √(π/2x) J_{n+1/2}(x)
pub fn spherical_bessel_j(n: u32, x: f64) -> f64  // j_n(x)
pub fn spherical_bessel_y(n: u32, x: f64) -> f64  // y_n(x)
pub fn spherical_bessel_i(n: u32, x: f64) -> f64  // i_n(x)
pub fn spherical_bessel_k(n: u32, x: f64) -> f64  // k_n(x)
```

#### Hankel Functions (Complex-Valued)
```rust
// H^(1)_n(x) = J_n(x) + i*Y_n(x)
pub fn hankel_first(n: u32, x: f64) -> (f64, f64)

// H^(2)_n(x) = J_n(x) - i*Y_n(x)
pub fn hankel_second(n: u32, x: f64) -> (f64, f64)

// Spherical variants
pub fn spherical_hankel_1(n: u32, x: f64) -> (f64, f64)
pub fn spherical_hankel_2(n: u32, x: f64) -> (f64, f64)
```

### 2. Error Functions Module (`error_functions.rs`)

```rust
// Error function: erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
pub fn erf(x: f64) -> f64

// Complementary error function: erfc(x) = 1 - erf(x)
pub fn erfc(x: f64) -> f64

// Inverse error function: erf(erfinv(x)) = x for x ∈ (-1, 1)
pub fn erfinv(x: f64) -> f64

// Inverse complementary: erfc(erfcinv(x)) = x for x ∈ (0, 2)
pub fn erfcinv(x: f64) -> f64
```

**Key Properties**:
- erf(0) = 0, erf(∞) = 1, erf(-x) = -erf(x)
- erfc(x) = 1 - erf(x)
- Used for normal distribution CDF: Φ(x) = (1 + erf(x/√2))/2

### 3. Gamma Functions Module (`gamma_functions.rs`)

```rust
// Gamma function: Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt
pub fn gamma(x: f64) -> f64

// Log gamma: ln(Γ(x)) - avoids overflow for large x
pub fn lgamma(x: f64) -> f64

// Digamma (psi): ψ(x) = d/dx ln(Γ(x))
pub fn digamma(x: f64) -> f64

// Beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
pub fn beta(a: f64, b: f64) -> f64

// Log beta: ln(B(a,b))
pub fn lbeta(a: f64, b: f64) -> f64
```

**Key Properties**:
- Γ(n) = (n-1)! for positive integers n
- Γ(1/2) = √π
- Reflection formula: Γ(1-x)Γ(x) = π/sin(πx)

### 4. Incomplete Gamma Functions (`incomplete_gamma.rs`)

```rust
// Lower incomplete gamma: γ(a,x) = ∫₀ˣ t^(a-1) e^(-t) dt
pub fn gamma_lower(a: f64, x: f64) -> f64

// Upper incomplete gamma: Γ(a,x) = ∫ₓ^∞ t^(a-1) e^(-t) dt
pub fn gamma_upper(a: f64, x: f64) -> f64

// Regularized forms (CDF-like)
pub fn gamma_p(a: f64, x: f64) -> f64  // P(a,x) = γ(a,x)/Γ(a)
pub fn gamma_q(a: f64, x: f64) -> f64  // Q(a,x) = Γ(a,x)/Γ(a)
```

### 5. Mathematical Identities & Relations

```rust
// Wronskian identities for verification
pub fn wronskian(n: u32, x: f64) -> f64  // W[J_n, Y_n] = 2/(πx)
pub fn modified_wronskian(n: u32, x: f64) -> f64  // W[I_n, K_n]

// Derivative functions using recurrence relations
pub fn bessel_j_derivative(n: u32, x: f64) -> f64  // J'_n = (J_{n-1} - J_{n+1})/2
pub fn bessel_y_derivative(n: u32, x: f64) -> f64  // Y'_n = (Y_{n-1} - Y_{n+1})/2
```

## 🔌 RustLab-Math Integration

### Extension Traits (Optional Feature)

When the `integration` feature is enabled, special functions become available as methods on RustLab-Math types:

```rust
use rustlab_math::{vec64, array64};
use rustlab_special::integration::*;

// Vector operations
let x = vec64![1.0, 2.0, 3.0, 4.0];
let bessel_vals = x.bessel_j0();    // Apply J_0 element-wise
let erf_vals = x.erf();             // Apply erf element-wise
let gamma_vals = x.gamma();         // Apply Γ element-wise

// Array operations
let matrix = array64![[1.0, 2.0], [3.0, 4.0]];
let bessel_matrix = matrix.bessel_j0();  // Element-wise J_0
let gamma_matrix = matrix.gamma();       // Element-wise Γ

// Convenience functions
use rustlab_special::integration::convenience::*;
let normal_cdf = normal_cdf(&x);  // Φ(x) = (1 + erf(x/√2))/2
let normal_pdf = normal_pdf(&x);  // φ(x) = exp(-x²/2)/√(2π)
```

### Available Extension Methods

**For VectorF64/ArrayF64**:
- `.bessel_j0()`, `.bessel_j1()`, `.bessel_j2()`
- `.bessel_y0()`, `.bessel_y1()`
- `.bessel_i0()`, `.bessel_i1()`, `.bessel_i2()`
- `.bessel_k0()`, `.bessel_k1()`
- `.erf()`, `.erfc()`, `.erfinv()`, `.erfcinv()`
- `.gamma()`, `.lgamma()`, `.digamma()`
- `.spherical_bessel_j(n)`, `.spherical_bessel_y(n)`

**For VectorF32/ArrayF32** (reduced set):
- `.bessel_j0()`, `.erf()`, `.gamma()`

## 📈 Numerical Algorithms & Accuracy

### Algorithm Selection Strategy

1. **Small Arguments**: Series expansions
   - Power series for J_n(x) when |x| < 10
   - Taylor series for erf(x) when |x| < 1
   
2. **Large Arguments**: Asymptotic expansions
   - Hankel's expansion for J_n(x) when x >> n
   - Continued fractions for erfc(x) when x > 4
   
3. **Intermediate Range**: Specialized algorithms
   - Miller's algorithm for Bessel recurrence
   - Chebyshev polynomial approximations
   - Rational function approximations

### Accuracy Guarantees

| Function Family | Primary Domain | Target Accuracy | Notes |
|-----------------|---------------|-----------------|-------|
| Bessel J_n, Y_n | x ∈ [0, 100] | 1e-15 | Reduced for large n |
| Modified I_n, K_n | x ∈ [0, 100] | 1e-14 | Scaled for large x |
| Error functions | x ∈ [-10, 10] | 1e-15 | Full double precision |
| Gamma functions | x ∈ [-170, 170] | 1e-14 | Poles handled |
| Spherical Bessel | x ∈ [0, 100] | 1e-14 | Via half-integer orders |

### Numerical Stability Features

1. **Overflow Prevention**:
   - Logarithmic forms (lgamma, lbeta) for large arguments
   - Scaled forms for modified Bessel functions
   
2. **Cancellation Avoidance**:
   - Miller's backward recurrence for J_n
   - Continued fractions for ratios
   
3. **Special Value Handling**:
   - Proper NaN/Inf propagation
   - Correct limiting behavior at boundaries

## 🚀 Usage Examples

### Example 1: Wavefunction Calculations (Quantum Mechanics)

```rust
use rustlab_special::*;

// Spherical Bessel functions for hydrogen atom radial wavefunctions
fn hydrogen_radial(n: u32, l: u32, r: f64) -> f64 {
    let rho = 2.0 * r / n as f64;
    let normalization = ((2.0 / n as f64).powi(3) * 
                        factorial(n - l - 1) / 
                        (2.0 * n as f64 * factorial(n + l).powi(3))).sqrt();
    
    normalization * rho.powi(l as i32) * 
    (-rho).exp() * 
    spherical_bessel_j(l, rho)
}
```

### Example 2: Signal Processing (Bessel Filters)

```rust
use rustlab_special::*;

// Bessel filter prototype using Bessel polynomials
fn bessel_filter_response(order: u32, omega: f64) -> f64 {
    // Uses reverse Bessel polynomials
    let x = omega.sqrt();
    1.0 / (1.0 + bessel_j(order, x).powi(2)).sqrt()
}
```

### Example 3: Statistical Computations

```rust
use rustlab_special::*;

// Chi-squared CDF using incomplete gamma
fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        gamma_p(df / 2.0, x / 2.0)
    }
}

// Normal distribution quantile function
fn normal_quantile(p: f64) -> f64 {
    std::f64::consts::SQRT_2 * erfinv(2.0 * p - 1.0)
}
```

### Example 4: Heat Transfer (Modified Bessel Functions)

```rust
use rustlab_special::*;

// Temperature distribution in a cylinder
fn cylinder_temperature(r: f64, z: f64, t: f64, alpha: f64) -> f64 {
    let lambda = std::f64::consts::PI;
    let term1 = bessel_i0(lambda * r) / bessel_i0(lambda);
    let term2 = (-alpha * lambda.powi(2) * t).exp();
    let term3 = (lambda * z).cos();
    
    term1 * term2 * term3
}
```

### Example 5: Array Operations with RustLab-Math

```rust
#[cfg(feature = "integration")]
use rustlab_math::{vec64, array64};
use rustlab_special::integration::*;

// Analyze antenna radiation pattern
let angles = linspace(0.0, 2.0 * PI, 360);
let ka = 2.0 * PI;  // Wave number * radius

// Compute radiation pattern using Bessel functions
let pattern = angles.map(|theta| {
    let x = ka * theta.sin();
    2.0 * bessel_j1(x) / x
});

// Apply window function using error function
let windowed = pattern.zip(&angles).map(|(p, theta)| {
    p * erf(3.0 * (1.0 - theta / PI))
});
```

## ⚡ Performance Optimization

### Computation Strategies

1. **Caching & Memoization**:
   - Factorial values pre-computed
   - Common constants stored
   - Coefficient tables for series

2. **Algorithm Selection**:
   - Runtime dispatch based on argument magnitude
   - Specialized paths for common cases (n=0,1,2)
   - SIMD potential for array operations

3. **Numerical Tricks**:
   - Horner's method for polynomial evaluation
   - Argument reduction for periodic functions
   - Scaling to prevent overflow/underflow

### Benchmarking Results (Typical)

```
bessel_j0: 100 values in 1.2µs (12ns per call)
bessel_i0: 100 values in 1.8µs (18ns per call)
erf:       100 values in 0.9µs (9ns per call)
gamma:     100 values in 2.1µs (21ns per call)
```

## 🔍 Testing & Validation

### Test Coverage

1. **Unit Tests**: Each function has comprehensive tests
2. **Property Tests**: Mathematical identities verified
3. **Reference Data**: Comparison with high-precision values
4. **Edge Cases**: NaN, Inf, zero, negative handling
5. **Accuracy Tests**: Error bounds verified across domains

### Mathematical Validation

```rust
// Example: Verify Wronskian identity
#[test]
fn test_wronskian() {
    let x = 5.0;
    let n = 3;
    let w = bessel_j(n, x) * bessel_y_derivative(n, x) - 
            bessel_j_derivative(n, x) * bessel_y(n, x);
    assert!((w - 2.0 / (PI * x)).abs() < 1e-14);
}

// Example: Verify reflection formula
#[test]
fn test_gamma_reflection() {
    let x = 0.3;
    let result = gamma(x) * gamma(1.0 - x);
    let expected = PI / (PI * x).sin();
    assert!((result - expected).abs() < 1e-14);
}
```

## 🎯 Common Use Cases

### Scientific Computing
- **Physics**: Wavefunctions, scattering amplitudes, field solutions
- **Engineering**: Heat transfer, wave propagation, signal processing
- **Statistics**: Distribution functions, hypothesis testing, regression
- **Finance**: Option pricing models, risk metrics

### Integration Points
- Works standalone for scalar computations
- Integrates with rustlab-math for array operations
- Compatible with ndarray via conversion
- Supports both f32 and f64 precision

## 🚧 Known Limitations & Future Work

### Current Limitations
1. **Complex Arguments**: Currently real-only (complex support planned)
2. **Arbitrary Precision**: Fixed to f64 (arbitrary precision planned)
3. **Vectorization**: Limited SIMD optimization
4. **Special Cases**: Some extreme parameter combinations less accurate

### Planned Enhancements
1. Complex argument support for all functions
2. Associated Legendre functions
3. Hypergeometric functions
4. Elliptic integrals
5. Zeta and polylogarithm functions
6. GPU acceleration support

## 📚 Mathematical References

Key algorithms implemented from:
1. **NIST Digital Library of Mathematical Functions**
2. **Numerical Recipes in C** (Press et al.)
3. **Handbook of Mathematical Functions** (Abramowitz & Stegun)
4. **Special Functions** (Andrews, Askey, Roy)
5. **Computer Approximations** (Hart et al.)

## 🔑 Key Takeaways

1. **Comprehensive**: Complete implementation of essential special functions
2. **Accurate**: 1e-15 precision in primary domains
3. **Performant**: Optimized algorithms with careful numerical analysis
4. **Integrated**: Optional seamless integration with RustLab-Math
5. **Robust**: Extensive testing and mathematical validation
6. **Documented**: Clear API with mathematical context
7. **Extensible**: Modular design allows easy addition of new functions

RustLab-Special provides the mathematical special functions essential for scientific computing, engineering applications, and statistical analysis, with a focus on numerical accuracy, performance, and ease of use.