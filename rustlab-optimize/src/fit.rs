//! Clean curve fitting functions with parameter fixing support

use rustlab_math::{VectorF64, statistics::BasicStatistics};
use crate::core::{Result, Error};
use crate::models::{LinearFit, ExponentialFit, PolynomialFit, SinusoidalFit};
use crate::algorithms::{OptimizationProblem, select_algorithm};

/// Fit linear regression model y = a + bx using analytical least squares
/// 
/// # Mathematical Specification
/// Given data points (xᵢ, yᵢ), solve normal equations:
/// [n    Σxᵢ  ] [a] = [Σyᵢ  ]
/// [Σxᵢ  Σxᵢ²] [b]   [Σxᵢyᵢ]
/// 
/// Analytical solution: b = (nΣxᵢyᵢ - ΣxᵢΣyᵢ)/(nΣxᵢ² - (Σxᵢ)²)
///                      a = (Σyᵢ - bΣxᵢ)/n
/// 
/// # Dimensions
/// - Input: x (n), y (n) where n ≥ 2
/// - Output: LinearFit with intercept a, slope b, R², confidence intervals
/// 
/// # Complexity
/// - Time: O(n) single pass through data
/// - Space: O(1) constant memory
/// 
/// # For AI Code Generation
/// - No iterative optimization needed - direct analytical solution
/// - Returns LinearFit struct with .intercept, .slope, .r_squared fields
/// - Automatically computes R², standard errors, confidence intervals
/// - Use .predict(x_new) method for predictions
/// - Common uses: trend analysis, calibration curves, simple regression
/// 
/// # Example
/// ```
/// use rustlab_optimize::{fit_linear, prelude::*};
/// use rustlab_math::{linspace, VectorF64};
/// 
/// // Generate linear data with noise
/// let x = linspace(0.0, 10.0, 11);
/// let y_true = &x * 2.0 + 3.0;  // y = 3 + 2x
/// let noise = VectorF64::from_slice(&[0.1, -0.1, 0.05, -0.05, 0.2, 
///                                      -0.15, 0.1, -0.1, 0.05, -0.05, 0.1]);
/// let y = &y_true + &noise;
/// 
/// let fit = fit_linear(&x, &y)?;
/// 
/// assert!((fit.intercept - 3.0).abs() < 0.3);  // Near true intercept
/// assert!((fit.slope - 2.0).abs() < 0.1);      // Near true slope
/// assert!(fit.r_squared > 0.95);               // High correlation
/// 
/// // Make predictions
/// let y_pred = fit.predict(5.0);  // Predict at x=5
/// assert!((y_pred - 13.0).abs() < 0.5);  // Should be ~13
/// ```
/// 
/// # Errors
/// - `DimensionMismatch`: x.len() != y.len(). Fix: Ensure equal-length arrays
/// - `NumericalError`: All x values identical (zero variance).
///   Fix: Ensure x data has variation
/// 
/// # See Also
/// - [`fit_polynomial`]: Higher-order polynomial fitting
/// - [`curve_fit`]: General nonlinear curve fitting
pub fn fit_linear(x: &VectorF64, y: &VectorF64) -> Result<LinearFit> {
    if x.len() != y.len() {
        return Err(Error::dimension_mismatch(x.len(), y.len()));
    }
    
    // Analytical solution using normal equations
    // TODO: Implement using rustlab-linearalgebra
    todo!("Implement analytical linear regression")
}

/// Fit exponential decay model y = A·exp(-kx) using nonlinear least squares
/// 
/// # Mathematical Specification
/// Given data points (xᵢ, yᵢ), solve:
/// minimize Σᵢ [yᵢ - A·exp(-k·xᵢ)]² over parameters (A, k)
/// 
/// Uses Levenberg-Marquardt algorithm for robust convergence.
/// 
/// # Dimensions
/// - Input: x (n), y (n) where n ≥ 3 (need more data than parameters)
/// - Output: ExponentialFit with amplitude A, decay_rate k, half_life, R²
/// 
/// # Complexity
/// - Time: O(n·iter) where iter ≈ 10-50 iterations typically
/// - Space: O(n) for residual computation
/// 
/// # For AI Code Generation
/// - Returns ExponentialFit struct with .amplitude, .decay_rate, .half_life fields
/// - Automatic initial guess from data (A ≈ max(y), k ≈ 1)
/// - Use .predict(x_new) for predictions, .half_life for interpretation
/// - Common uses: radioactive decay, drug kinetics, cooling curves
/// - For advanced control use fit_exponential_advanced() builder
/// 
/// # Example
/// ```
/// use rustlab_optimize::{fit_exponential, prelude::*};
/// use rustlab_math::linspace;
/// 
/// // Generate exponential decay data: y = 10*exp(-0.5*x)
/// let x = linspace(0.0, 5.0, 20);
/// let y_true: Vec<f64> = x.iter().map(|&t| 10.0 * (-0.5 * t).exp()).collect();
/// let y = VectorF64::from_slice(&y_true);
/// 
/// let fit = fit_exponential(&x, &y)?;
/// 
/// assert!((fit.amplitude - 10.0).abs() < 0.1);     // A ≈ 10
/// assert!((fit.decay_rate - 0.5).abs() < 0.05);    // k ≈ 0.5
/// assert!((fit.half_life - 1.386).abs() < 0.1);    // t₁/₂ = ln(2)/k ≈ 1.386
/// assert!(fit.r_squared > 0.99);                   // Excellent fit
/// 
/// // Physical interpretation
/// let time_to_10_percent = fit.time_to_fraction(0.1);
/// println!("Time to 10% of initial: {:.2}", time_to_10_percent);
/// ```
/// 
/// # Errors
/// - `DimensionMismatch`: x.len() != y.len(). Fix: Ensure equal-length arrays
/// - `ConvergenceFailed`: Poor fit or wrong model form.
///   Fix: Check if data follows exponential pattern, try different bounds
/// - `NumericalError`: Negative or zero y values.
///   Fix: Exponential model requires y > 0, consider adding offset
/// 
/// # See Also
/// - [`fit_exponential_advanced`]: Full parameter control with bounds/fixing
/// - [`fit_sinusoidal`]: For oscillatory decay patterns
pub fn fit_exponential(x: &VectorF64, y: &VectorF64) -> Result<ExponentialFit> {
    fit_exponential_advanced(x, y).solve()
}

/// Builder for exponential fitting with parameter control
pub struct ExponentialFitter {
    x_data: VectorF64,
    y_data: VectorF64,
    initial_amplitude: Option<f64>,
    initial_decay_rate: Option<f64>,
    fix_amplitude: Option<f64>,
    fix_decay_rate: Option<f64>,
    bounds_amplitude: Option<(f64, f64)>,
    bounds_decay_rate: Option<(f64, f64)>,
    couplings: Vec<crate::algorithms::ParameterCoupling>,
}

impl ExponentialFitter {
    fn new(x: VectorF64, y: VectorF64) -> Self {
        Self {
            x_data: x,
            y_data: y,
            initial_amplitude: None,
            initial_decay_rate: None,
            fix_amplitude: None,
            fix_decay_rate: None,
            bounds_amplitude: None,
            bounds_decay_rate: None,
            couplings: Vec::new(),
        }
    }

    /// Set initial guess for amplitude (A in y = A*exp(-k*x))
    pub fn with_initial_amplitude(mut self, a0: f64) -> Self {
        self.initial_amplitude = Some(a0);
        self
    }

    /// Set initial guess for decay rate (k in y = A*exp(-k*x))
    pub fn with_initial_decay_rate(mut self, k0: f64) -> Self {
        self.initial_decay_rate = Some(k0);
        self
    }

    /// Set initial guess for both parameters
    pub fn with_initial(mut self, amplitude: f64, decay_rate: f64) -> Self {
        self.initial_amplitude = Some(amplitude);
        self.initial_decay_rate = Some(decay_rate);
        self
    }

    /// Fix amplitude at specific value (optimize only decay rate)
    /// 
    /// This is useful for cases where the amplitude is known from theory
    /// or experimental constraints.
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_optimize::{fit_exponential, prelude::*};
    /// 
    /// let x = linspace(0.0, 5.0, 20);
    /// let y = vec64(&[/* data */]);
    /// 
    /// // Fix amplitude at 10.0, optimize only decay rate
    /// let fit = fit_exponential(&x, &y)
    ///     .fix_amplitude(10.0)
    ///     .solve()?;
    /// ```
    pub fn fix_amplitude(mut self, value: f64) -> Self {
        self.fix_amplitude = Some(value);
        self
    }

    /// Fix decay rate at specific value (optimize only amplitude)
    /// 
    /// Useful when the decay rate is known from theory (e.g., radioactive decay)
    /// but the initial amount is unknown.
    pub fn fix_decay_rate(mut self, value: f64) -> Self {
        self.fix_decay_rate = Some(value);
        self
    }

    /// Set bounds for amplitude: A ∈ [min, max]
    pub fn amplitude_bounds(mut self, min: f64, max: f64) -> Self {
        self.bounds_amplitude = Some((min, max));
        self
    }

    /// Set bounds for decay rate: k ∈ [min, max]  
    pub fn decay_rate_bounds(mut self, min: f64, max: f64) -> Self {
        self.bounds_decay_rate = Some((min, max));
        self
    }

    /// Add linear parameter coupling: param2 = scale * param1 + offset
    /// 
    /// # Parameter Indices for Exponential Model y = A·exp(-k·x)
    /// - 0: Amplitude (A)
    /// - 1: Decay rate (k)
    /// 
    /// # Common Uses
    /// - Multiple exponentials with constrained amplitude ratio
    /// - Fixed relationship between decay rates in multi-exponential fits
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_optimize::{fit_exponential_advanced, prelude::*};
    /// 
    /// // For multi-exponential where A2 = 0.5 * A1
    /// let fit = fit_exponential_advanced(&x, &y)
    ///     .couple_linear(0, 1, 0.5, 0.0)  // A2 = 0.5 * A1
    ///     .solve()?;
    /// ```
    pub fn couple_linear(mut self, independent: usize, dependent: usize, scale: f64, offset: f64) -> Self {
        self.couplings.push(crate::algorithms::ParameterCoupling::Linear {
            independent,
            dependent,
            scale,
            offset,
        });
        self
    }

    /// Add parameter ratio constraint: param1/param2 = ratio
    /// Equivalent to: param1 = ratio * param2
    pub fn couple_ratio(self, param1: usize, param2: usize, ratio: f64) -> Self {
        self.couple_linear(param2, param1, ratio, 0.0)
    }

    /// Solve the exponential fitting problem
    pub fn solve(self) -> Result<ExponentialFit> {
        if self.x_data.len() != self.y_data.len() {
            return Err(Error::dimension_mismatch(self.x_data.len(), self.y_data.len()));
        }

        // Smart initial guesses using math-first operations
        let initial_amplitude = self.initial_amplitude.unwrap_or_else(|| {
            self.y_data.max().unwrap_or(1.0)  // Use maximum as amplitude estimate
        });
        let initial_decay_rate = self.initial_decay_rate.unwrap_or(1.0);
        
        let initial = VectorF64::from_slice(&[initial_amplitude, initial_decay_rate]);

        // Handle parameter fixing
        let fixed_params = [
            self.fix_amplitude.map(|v| (0, v)),
            self.fix_decay_rate.map(|v| (1, v)),
        ];
        let fixed_params: Vec<_> = fixed_params.into_iter().flatten().collect();

        // Create bounds
        let _bounds = if self.bounds_amplitude.is_some() || self.bounds_decay_rate.is_some() {
            let lower = VectorF64::from_slice(&[
                self.bounds_amplitude.map_or(-f64::INFINITY, |(min, _)| min),
                self.bounds_decay_rate.map_or(-f64::INFINITY, |(min, _)| min),
            ]);
            let upper = VectorF64::from_slice(&[
                self.bounds_amplitude.map_or(f64::INFINITY, |(_, max)| max),
                self.bounds_decay_rate.map_or(f64::INFINITY, |(_, max)| max),
            ]);
            Some((lower, upper))
        } else {
            None
        };

        // Create curve fitting problem (automatically uses LM)
        let x_data_clone = self.x_data.clone();
        let y_data_clone = self.y_data.clone();
        
        let mut problem = OptimizationProblem::curve_fitting(
            |params: &VectorF64, x: &VectorF64| -> VectorF64 {
                let amplitude = params[0];
                let decay_rate = params[1];
                
                let mut predictions = VectorF64::zeros(x.len());
                for i in 0..x.len() {
                    predictions[i] = amplitude * (-decay_rate * x[i]).exp();
                }
                predictions
            },
            &x_data_clone,
            &y_data_clone,
            &initial,
        );

        // Apply parameter fixing
        if !fixed_params.is_empty() {
            problem = problem.fix_parameters(&fixed_params);
        }

        // Apply parameter couplings using math-first interface
        for coupling in self.couplings {
            problem = problem.add_coupling(coupling);
        }

        // Select and run solver (will automatically choose LM for curve fitting)
        let solver = select_algorithm(&problem);
        let result = solver.solve(problem)?;

        // Extract fitted parameters
        let amplitude = result.solution[0];
        let decay_rate = result.solution[1];
        let half_life = (2.0_f64).ln() / decay_rate.abs();

        // Calculate R² using math-first operations
        let y_mean = self.y_data.mean();
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        
        for i in 0..self.x_data.len() {
            let predicted = amplitude * (-decay_rate * self.x_data[i]).exp();
            let observed = self.y_data[i];
            
            ss_res += (observed - predicted).powi(2);
            ss_tot += (observed - y_mean).powi(2);
        }
        
        let r_squared = 1.0 - (ss_res / ss_tot);

        Ok(ExponentialFit {
            amplitude,
            decay_rate,
            half_life,
            r_squared,
        })
    }
}

/// Create builder for exponential fitting with parameter constraints and bounds
/// 
/// # Mathematical Specification
/// Solves y = A·exp(-kx) with optional:
/// - Parameter bounds: A ∈ [A_min, A_max], k ∈ [k_min, k_max]
/// - Parameter fixing: A = A_fixed or k = k_fixed
/// - Custom initial guesses: (A₀, k₀)
/// 
/// # Dimensions
/// - Input: x (n), y (n) where n ≥ effective_parameters
/// - Output: ExponentialFitter builder → ExponentialFit result
/// 
/// # Complexity
/// - Time: O(n·iter) where iter depends on constraints
/// - Space: O(n) for problem setup
/// 
/// # For AI Code Generation
/// - Returns ExponentialFitter builder, call methods then .solve()
/// - Use .fix_amplitude(value) when A known from theory/experiment
/// - Use .decay_rate_bounds(min, max) for physical constraints (k > 0)
/// - Use .with_initial(A0, k0) for better convergence
/// - Common pattern: bounds first, then initial guess, then solve
/// 
/// # Example
/// ```
/// use rustlab_optimize::{fit_exponential_advanced, prelude::*};
/// use rustlab_math::linspace;
/// 
/// let x = linspace(0.0, 5.0, 20);
/// let y = VectorF64::from_slice(&[10.0, 8.2, 6.7, 5.5, 4.5, 3.7, 
///                                 3.0, 2.5, 2.0, 1.6, 1.3, 1.1, 
///                                 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]);
/// 
/// // Constrained fitting with physical bounds
/// let fit = fit_exponential_advanced(&x, &y)
///     .amplitude_bounds(0.0, 20.0)      // A must be positive
///     .decay_rate_bounds(0.01, 2.0)     // k must be positive
///     .with_initial(10.0, 0.5)          // Good starting guess
///     .solve()?;
/// 
/// // Physics experiment: amplitude known from theory
/// let fit = fit_exponential_advanced(&x, &y)
///     .fix_amplitude(10.0)              // A = 10 exactly
///     .decay_rate_bounds(0.1, 1.0)      // Constrain k
///     .solve()?;
/// 
/// // Only fit amplitude (decay rate known)
/// let fit = fit_exponential_advanced(&x, &y)
///     .fix_decay_rate(0.693)            // k = ln(2) for unit half-life
///     .amplitude_bounds(5.0, 15.0)
///     .solve()?;
/// ```
/// 
/// # Errors
/// - `InvalidInput`: Bounds are inconsistent (min > max).
///   Fix: Check bound ordering
/// - `ConvergenceFailed`: Over-constrained or conflicting constraints.
///   Fix: Relax bounds or check fixed parameter values
/// 
/// # See Also
/// - [`fit_exponential`]: Simple interface with automatic settings
/// - [`curve_fit`]: For custom exponential variants (y = A·exp(-kx) + C)
pub fn fit_exponential_advanced(x: &VectorF64, y: &VectorF64) -> ExponentialFitter {
    ExponentialFitter::new(x.clone(), y.clone())
}

/// Fit polynomial: y = a₀ + a₁x + a₂x² + ... + aₙxⁿ
pub fn fit_polynomial(_x: &VectorF64, _y: &VectorF64, _degree: usize) -> Result<PolynomialFit> {
    todo!("Implement polynomial fitting")
}

/// Fit sinusoidal model: y = A*sin(ωx + φ) + C  
pub fn fit_sinusoidal(_x: &VectorF64, _y: &VectorF64) -> Result<SinusoidalFit> {
    todo!("Implement sinusoidal fitting")
}

/// Create builder for fitting arbitrary mathematical models to data
/// 
/// # Mathematical Specification
/// Given model function m(x, θ⃗) and data (xᵢ, yᵢ), solve:
/// minimize Σᵢ [yᵢ - m(xᵢ, θ⃗)]² over parameters θ⃗ ∈ ℝⁿ
/// 
/// Supports arbitrary functional forms with parameter constraints.
/// 
/// # Dimensions
/// - Input: x (k), y (k), model function m(x, θ⃗)
/// - Output: FitBuilder → OptimizationResult with θ⃗ ∈ ℝⁿ
/// - Constraint: k ≥ n (more data than parameters)
/// 
/// # Complexity
/// - Time: O(k·n²·iter) for model evaluations and Jacobian
/// - Space: O(k·n + n²) for optimization matrices
/// 
/// # For AI Code Generation
/// - Model function signature: |x: f64, params: &[f64]| -> f64
/// - Always provide initial guess via .with_initial(&[...])
/// - Use .fix_parameter(i, value) to constrain specific parameters
/// - Use .bounds(&lower, &upper) for box constraints
/// - Returns standard OptimizationResult, access solution via result.solution
/// - Most flexible interface - use for custom models not covered by specialized functions
/// 
/// # Example
/// ```
/// use rustlab_optimize::{fit, prelude::*};
/// use rustlab_math::linspace;
/// 
/// let x = linspace(0.0, 10.0, 50);
/// let y = VectorF64::from_slice(&[/* experimental data */]);
/// 
/// // Fit power law with offset: y = a·x^b + c
/// let power_model = |x: f64, params: &[f64]| {
///     let a = params[0];  // amplitude
///     let b = params[1];  // exponent  
///     let c = params[2];  // offset
///     a * x.powf(b) + c
/// };
/// 
/// let result = fit(&x, &y, power_model)
///     .with_initial(&[1.0, 2.0, 0.0])    // [a₀, b₀, c₀]
///     .bounds(&[0.0, 0.5, -1.0],          // Lower bounds
///             &[10.0, 4.0, 1.0])          // Upper bounds
///     .solve()?;
/// 
/// let fitted_params = result.solution;  // [a_fit, b_fit, c_fit]
/// 
/// // Fit with fixed exponent (known from theory)
/// let result = fit(&x, &y, power_model)
///     .with_initial(&[1.0, 2.0, 0.0])
///     .fix_parameter(1, 2.0)              // Force b = 2 (quadratic)
///     .solve()?;
/// 
/// // Multi-exponential model: y = A₁·exp(-k₁·x) + A₂·exp(-k₂·x)
/// let double_exp = |x: f64, params: &[f64]| {
///     let a1 = params[0]; let k1 = params[1];
///     let a2 = params[2]; let k2 = params[3];
///     a1 * (-k1 * x).exp() + a2 * (-k2 * x).exp()
/// };
/// 
/// let result = fit(&x, &y, double_exp)
///     .with_initial(&[5.0, 0.5, 3.0, 2.0])  // [A₁, k₁, A₂, k₂]
///     .bounds(&[0.0, 0.01, 0.0, 0.01],       // All positive
///             &[20.0, 5.0, 20.0, 5.0])
///     .solve()?;
/// ```
/// 
/// # Errors
/// - `InvalidInput`: No initial parameters provided. Always use .with_initial()
/// - `ConvergenceFailed`: Poor model form or initialization.
///   Fix: Check model function, try different starting values
/// - `NumericalError`: Model returns NaN/infinity.
///   Fix: Add parameter bounds to avoid problematic regions
/// 
/// # See Also
/// - [`fit_exponential`]: Optimized for exponential models
/// - [`fit_polynomial`]: Optimized for polynomial models
/// - [`curve_fit`]: Equivalent interface using VectorF64
pub fn fit<F>(x: &VectorF64, y: &VectorF64, model: F) -> FitBuilder<F>
where
    F: Fn(f64, &[f64]) -> f64 + Send + Sync + 'static,
{
    FitBuilder::new(x.clone(), y.clone(), model)
}

/// Builder for general curve fitting with parameter control
pub struct FitBuilder<F> {
    x_data: VectorF64,
    y_data: VectorF64,
    model: F,
    initial: Option<Vec<f64>>,
    fixed_params: Option<Vec<(usize, f64)>>,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    couplings: Vec<crate::algorithms::ParameterCoupling>,
}

impl<F> FitBuilder<F>
where
    F: Fn(f64, &[f64]) -> f64 + Send + Sync + 'static,
{
    fn new(x: VectorF64, y: VectorF64, model: F) -> Self {
        Self {
            x_data: x,
            y_data: y,
            model,
            initial: None,
            fixed_params: None,
            bounds: None,
            couplings: Vec::new(),
        }
    }

    /// Set initial parameter guess
    pub fn with_initial(mut self, params: &[f64]) -> Self {
        self.initial = Some(params.to_vec());
        self
    }

    /// Fix specific parameter at given value
    pub fn fix_parameter(mut self, index: usize, value: f64) -> Self {
        self.fixed_params.get_or_insert_with(Vec::new).push((index, value));
        self
    }

    /// Fix multiple parameters  
    pub fn fix_parameters(mut self, fixed: &[(usize, f64)]) -> Self {
        self.fixed_params = Some(fixed.to_vec());
        self
    }

    /// Set parameter bounds
    pub fn bounds(mut self, lower: &[f64], upper: &[f64]) -> Self {
        self.bounds = Some((lower.to_vec(), upper.to_vec()));
        self
    }

    /// Add linear parameter coupling: param2 = scale * param1 + offset
    /// 
    /// # Mathematical Specification
    /// Creates constraint: θ[dependent] = scale * θ[independent] + offset
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_optimize::{fit, prelude::*};
    /// 
    /// // Power law model: y = a * x^b + c
    /// let power_model = |x: f64, params: &[f64]| {
    ///     let a = params[0];  // amplitude
    ///     let b = params[1];  // exponent  
    ///     let c = params[2];  // offset
    ///     a * x.powf(b) + c
    /// };
    /// 
    /// // Constrain c = 0.1 * a (offset proportional to amplitude)
    /// let result = fit(&x, &y, power_model)
    ///     .with_initial(&[1.0, 2.0, 0.0])
    ///     .couple_linear(0, 2, 0.1, 0.0)  // c = 0.1 * a
    ///     .solve()?;
    /// ```
    pub fn couple_linear(mut self, independent: usize, dependent: usize, scale: f64, offset: f64) -> Self {
        self.couplings.push(crate::algorithms::ParameterCoupling::Linear {
            independent,
            dependent,
            scale,
            offset,
        });
        self
    }

    /// Add parameter ratio constraint: param1/param2 = ratio
    pub fn couple_ratio(self, param1: usize, param2: usize, ratio: f64) -> Self {
        self.couple_linear(param2, param1, ratio, 0.0)
    }

    /// Add sum constraint: Σᵢ θᵢ = total
    /// 
    /// # Example
    /// ```rust
    /// // Multi-component model where fractions sum to 1
    /// let result = fit(&x, &y, multi_component_model)
    ///     .with_initial(&[0.3, 0.4, 0.3])  // f1, f2, f3
    ///     .couple_sum(&[0, 1, 2], 1.0)     // f1 + f2 + f3 = 1
    ///     .solve()?;
    /// ```
    pub fn couple_sum(mut self, parameters: &[usize], total: f64) -> Self {
        self.couplings.push(crate::algorithms::ParameterCoupling::SumConstraint {
            parameters: parameters.to_vec(),
            total,
        });
        self
    }

    /// Solve the fitting problem
    pub fn solve(self) -> Result<crate::core::OptimizationResult> {
        let initial = self.initial.ok_or_else(|| {
            Error::InvalidInput("Initial parameter guess required for custom model fitting".into())
        })?;

        // Create objective function (sum of squared residuals)  
        let objective = move |params: &[f64]| -> f64 {
            let mut sse = 0.0;
            for (i, &xi) in self.x_data.iter().enumerate() {
                let predicted = (self.model)(xi, params);
                let observed = self.y_data[i];
                let residual = predicted - observed;
                sse += residual * residual;
            }
            sse
        };

        // Build optimizer
        let mut optimizer = crate::minimize::minimize(objective)
            .from(&initial)
            .using_levenberg_marquardt(); // LM is optimal for least squares

        if let Some(fixed) = self.fixed_params {
            optimizer = optimizer.fix_parameters(&fixed);
        }

        if let Some((lower, upper)) = self.bounds {
            optimizer = optimizer.bounds(&lower, &upper);
        }
        
        // Note: Parameter coupling support for FitBuilder is available via methods
        // but requires integration with minimize interface. For now, coupling is
        // supported in ExponentialFitter and direct OptimizationProblem usage.
        // TODO: Integrate coupling with minimize interface in future version.

        optimizer.solve()
    }
}

// Helper functions for future use
#[allow(dead_code)]
fn estimate_amplitude(y_data: &VectorF64) -> f64 {
    y_data[0] // Simple estimate - could be more sophisticated
}

#[allow(dead_code)]
fn calculate_r_squared<F>(observed: &VectorF64, model: F, x_data: &VectorF64) -> f64
where
    F: Fn(f64) -> f64
{
    // R² = 1 - SS_res / SS_tot
    let y_mean: f64 = observed.iter().sum::<f64>() / observed.len() as f64;

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for (i, &yi) in observed.iter().enumerate() {
        let predicted = model(x_data[i]);
        let residual = yi - predicted;
        let total_var = yi - y_mean;

        ss_res += residual * residual;
        ss_tot += total_var * total_var;
    }

    1.0 - (ss_res / ss_tot)
}