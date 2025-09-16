# AI Documentation - RustLab Optimize

## Quick Reference for AI Code Generation

RustLab Optimize provides mathematical optimization and curve fitting with automatic algorithm selection and math-first APIs.

---

## 🎯 Core Functions

### Function Minimization

```rust
// 1D minimization: find x that minimizes f(x)
let x_min = minimize_1d(|x| (x - 2.0).powi(2)).solve()?;

// 2D minimization: find (x,y) that minimizes f(x,y)
let (x, y) = minimize_2d(|x, y| (1.0 - x).powi(2) + 100.0 * (y - x*x).powi(2))
    .from(-1.2, 1.0)
    .solve()?;

// N-dimensional: find x⃗ that minimizes f(x⃗)
let result = minimize(|x| x.iter().map(|&xi| xi * xi).sum())
    .from(&[1.0, 2.0, 3.0])
    .solve()?;
```

### Curve Fitting

```rust
// Linear regression: y = a + bx
let fit = fit_linear(&x_data, &y_data)?;
println!("y = {:.2} + {:.2}*x, R² = {:.3}", fit.intercept, fit.slope, fit.r_squared);

// Exponential decay: y = A*exp(-k*x)
let fit = fit_exponential(&x_data, &y_data)?;
println!("Half-life: {:.2}", fit.half_life);

// Custom models
let result = curve_fit(&x_data, &y_data, |x, params| {
    params[0] * x.powf(params[1]) + params[2]  // y = a*x^b + c
})
.with_initial(&[1.0, 2.0, 0.0])
.solve()?;
```

### Nonlinear Least Squares

```rust
// Direct residual minimization
let result = least_squares(|params| {
    // Return Vec<f64> of residuals
    data.iter().map(|(x, y)| y - model(*x, params)).collect()
})
.from(&[initial_guess])
.solve()?;
```

---

## 📐 Mathematical Specifications

### Problem Types

| Function | Mathematical Form | Best For |
|----------|------------------|----------|
| `minimize_1d(f)` | min f(x), x ∈ ℝ | Root finding, parameter tuning |
| `minimize_2d(f)` | min f(x,y), (x,y) ∈ ℝ² | Small optimization problems |
| `minimize(f)` | min f(x⃗), x⃗ ∈ ℝⁿ | General optimization |
| `fit_linear` | y = a + bx | Trend analysis, calibration |
| `fit_exponential` | y = A·exp(-kx) | Decay processes |
| `curve_fit` | y = model(x, θ⃗) | Custom functional forms |
| `least_squares` | min ‖r(θ⃗)‖² | Parameter estimation |

### Algorithm Selection

| Problem Characteristics | Auto-Selected Algorithm | Complexity |
|------------------------|-------------------------|------------|
| Curve fitting / Least squares | Levenberg-Marquardt | O(mn² + n³) |
| Small scale (n ≤ 10) | BFGS | O(n²k) |
| Large scale (n > 1000) | Gradient Descent | O(nk) |
| Noisy / Non-smooth | Nelder-Mead | O(n²k) |

---

## ⚙️ Builder Patterns

### Constraints and Bounds

```rust
// Parameter bounds
let result = minimize(objective)
    .from(&[1.0, 2.0, 3.0])
    .bounds(&[-5.0, -5.0, -5.0], &[5.0, 5.0, 5.0])
    .solve()?;

// Parameter fixing
let result = minimize(objective)
    .from(&[1.0, 2.0, 3.0])
    .fix_parameter(1, 2.0)  // Hold x[1] = 2.0 constant
    .solve()?;

// Algorithm selection
let result = minimize(objective)
    .from(&initial)
    .using_bfgs()  // Override automatic selection
    .tolerance(1e-10)
    .solve()?;
```

### Advanced Curve Fitting

```rust
// Exponential with constraints
let fit = fit_exponential_advanced(&x_data, &y_data)
    .amplitude_bounds(0.0, 100.0)      // A ∈ [0, 100]
    .decay_rate_bounds(0.01, 2.0)      // k ∈ [0.01, 2]
    .with_initial(10.0, 0.5)           // Starting guess
    .solve()?;

// Fixed parameters (known from theory)
let fit = fit_exponential_advanced(&x_data, &y_data)
    .fix_amplitude(10.0)               // A = 10 exactly
    .solve()?;
```

---

## 🎲 Return Types and Access Patterns

### Function Minimization Results

```rust
// 1D returns scalar
let x_min: f64 = minimize_1d(f).solve()?;

// 2D returns tuple
let (x, y): (f64, f64) = minimize_2d(f).from(0.0, 0.0).solve()?;

// N-D returns OptimizationResult
let result = minimize(f).from(&initial).solve()?;
let solution: &[f64] = &result.solution;
let final_value: f64 = result.final_value;
let iterations: usize = result.iterations;
```

### Curve Fitting Results

```rust
// Linear fit
let fit = fit_linear(&x, &y)?;
let prediction = fit.predict(new_x);
let is_significant = fit.slope_is_significant();

// Exponential fit  
let fit = fit_exponential(&x, &y)?;
let prediction = fit.predict(new_x);
let half_life = fit.half_life;
let time_to_10_percent = fit.time_to_fraction(0.1);

// Custom fit
let result = curve_fit(&x, &y, model).with_initial(&params).solve()?;
let fitted_params = result.solution;
```

---

## ⚠️ Common Patterns and Pitfalls

### ✅ Correct Patterns

```rust
// Always provide initial point for multi-dimensional
let result = minimize(|x| x[0].powi(2) + x[1].powi(2))
    .from(&[1.0, 2.0])  // Required!
    .solve()?;

// Use ? operator for error handling
let fit = fit_exponential(&x_data, &y_data)?;

// Check data dimensions match
assert_eq!(x_data.len(), y_data.len());

// Use builder pattern for constraints
let result = minimize(objective)
    .from(&initial)
    .bounds(&lower, &upper)
    .tolerance(1e-8)
    .solve()?;
```

### ❌ Common Mistakes

```rust
// Missing initial point
let result = minimize(|x| x[0].powi(2)).solve()?;  // ERROR!

// Wrong data types
let x_data = vec![1, 2, 3];  // Should be Vec<f64>

// Ignoring errors
let result = minimize(f).from(&[0.0]).solve().unwrap();  // Dangerous!

// Wrong model signature
curve_fit(&x, &y, |params, x| params[0] * x);  // Should be |x, params|
```

---

## 🔧 Error Handling

### Common Errors and Fixes

| Error Type | Common Cause | Fix |
|------------|--------------|-----|
| `InvalidInput` | Missing initial point | Add `.from(&initial)` |
| `DimensionMismatch` | x.len() ≠ y.len() | Check data array lengths |
| `ConvergenceFailed` | Poor initial guess | Try different starting point |
| `NumericalError` | Function returns NaN | Add bounds, check function |
| `BoundsViolation` | Initial point outside bounds | Choose feasible starting point |

### Error Handling Patterns

```rust
// Handle specific errors
match result {
    Ok(solution) => println!("Found solution: {:?}", solution.solution),
    Err(Error::ConvergenceFailed { iterations, reason }) => {
        println!("Failed after {} iterations: {}", iterations, reason);
        // Try different algorithm or starting point
    },
    Err(Error::DimensionMismatch { expected, actual }) => {
        println!("Data size mismatch: expected {}, got {}", expected, actual);
        // Check input data dimensions
    },
    Err(e) => println!("Other error: {}", e),
}

// Simple error propagation
fn my_optimization() -> Result<Vec<f64>, Error> {
    let result = minimize(|x| x[0].powi(2) + x[1].powi(2))
        .from(&[1.0, 2.0])
        .solve()?;  // Propagate any error
    Ok(result.solution)
}
```

---

## 🚀 Performance Guidelines

### Algorithm Choice by Problem Size

```rust
// Small problems (n ≤ 10): BFGS is excellent
let result = minimize(objective)
    .from(&small_initial)
    .using_bfgs()  // Fast quadratic convergence
    .solve()?;

// Large problems (n > 1000): Gradient descent
let result = minimize(objective)
    .from(&large_initial)
    .using_gradient_descent()  // Memory efficient
    .solve()?;

// Curve fitting: Always use Levenberg-Marquardt
let result = curve_fit(&x, &y, model)
    .with_initial(&params)
    .using_levenberg_marquardt()  // Optimal for least squares
    .solve()?;
```

### Memory Considerations

- **BFGS**: O(n²) memory for Hessian approximation
- **Gradient Descent**: O(n) memory, scales to large problems
- **Levenberg-Marquardt**: O(mn + n²) for m data points, n parameters

---

## 📝 Import Patterns

```rust
// Basic imports
use rustlab_optimize::{minimize, minimize_1d, minimize_2d};
use rustlab_optimize::{fit_linear, fit_exponential, curve_fit};
use rustlab_optimize::{least_squares, Error, Result};

// Model result types
use rustlab_optimize::{LinearFit, ExponentialFit, OptimizationResult};

// Everything (for scripts/notebooks)
use rustlab_optimize::prelude::*;

// With RustLab math
use rustlab_math::{vec64, linspace, VectorF64};
use rustlab_optimize::*;
```

---

## 🎓 Quick Conversion from SciPy

| SciPy | RustLab Optimize |
|-------|------------------|
| `scipy.optimize.minimize_scalar(f)` | `minimize_1d(f).solve()?` |
| `scipy.optimize.minimize(f, x0)` | `minimize(f).from(&x0).solve()?` |
| `scipy.optimize.curve_fit(f, x, y)` | `curve_fit(&x, &y, f).with_initial(&p0).solve()?` |
| `scipy.optimize.least_squares(f, x0)` | `least_squares(f).from(&x0).solve()?` |

---

## 🔥 Most Important Rules for AI

1. **Always provide initial point** for multi-dimensional optimization
2. **Use `?` operator** for error handling - never ignore Results
3. **Check data dimensions** match for curve fitting (x.len() == y.len())
4. **Model function signature** is `|x: f64, params: &[f64]| -> f64`
5. **Returns vary by function**: scalar, tuple, or OptimizationResult
6. **Use builder pattern** for constraints: `.bounds()`, `.fix_parameter()`
7. **Algorithm selection** is automatic but can be overridden
8. **Curve fitting** uses Levenberg-Marquardt by default (optimal)
9. **Linear fitting** is analytical (no iteration needed)
10. **Error messages** contain specific guidance for fixes

---

*This documentation is optimized for AI code generation. All examples are tested and follow best practices.*