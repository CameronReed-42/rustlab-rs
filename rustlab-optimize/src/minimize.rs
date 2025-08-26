//! Clean minimization functions with math-first API

use rustlab_math::VectorF64;
use crate::core::{Result, OptimizationResult, Algorithm, Error};
use crate::algorithms::{select_algorithm, OptimizationProblem, ProblemCharacteristics};

/// Find the scalar value x that minimizes a univariate function f(x)
/// 
/// # Mathematical Specification
/// Given function f: ℝ → ℝ, find x* such that:
/// f(x*) = min{f(x) : x ∈ ℝ} or x ∈ [a,b] if bounds specified
/// 
/// # Dimensions
/// - Input: Scalar function f(x) where x ∈ ℝ
/// - Output: Scalar x* ∈ ℝ (optimal point)
/// 
/// # Complexity
/// - Time: O(log(1/ε)) for unimodal functions with tolerance ε
/// - Space: O(1) constant memory
/// 
/// # For AI Code Generation
/// - Returns scalar f64 value, not a vector
/// - Use `.from(x0)` to set starting point (optional for most functions)
/// - Use `.bounds(min, max)` for constrained optimization
/// - Common uses: finding roots via minimizing f(x)², parameter tuning
/// - Automatic algorithm selection (Brent's method or golden section)
/// 
/// # Example
/// ```
/// use rustlab_optimize::minimize_1d;
/// 
/// // Find minimum of parabola (x - 2)²
/// let x_min = minimize_1d(|x| (x - 2.0).powi(2))
///     .solve()?;
/// assert!((x_min - 2.0).abs() < 1e-6);
/// 
/// // With bounds: minimize sin(x) on [0, π]
/// let x_min = minimize_1d(|x| x.sin())
///     .bounds(0.0, std::f64::consts::PI)
///     .solve()?;
/// // Result: x_min ≈ 3π/2 (where sin(x) = -1)
/// ```
/// 
/// # Errors
/// - `ConvergenceFailed`: Function doesn't converge within max iterations.
///   Fix: Check function properties or increase tolerance
/// - `NumericalError`: Function returns NaN or infinite values.
///   Fix: Add bounds to avoid problematic regions
/// 
/// # See Also
/// - [`minimize_2d`]: Two-variable optimization f(x,y)
/// - [`minimize`]: N-dimensional optimization f(x⃗)
pub fn minimize_1d<F>(f: F) -> Minimize1D<F>
where
    F: Fn(f64) -> f64 + Send + Sync + 'static,
{
    Minimize1D::new(f)
}

/// Builder for 1D minimization with method chaining
pub struct Minimize1D<F> {
    objective: F,
    initial: Option<f64>,
    bounds: Option<(f64, f64)>,
    tolerance: f64,
}

impl<F> Minimize1D<F>
where
    F: Fn(f64) -> f64 + Send + Sync + 'static,
{
    fn new(objective: F) -> Self {
        Self {
            objective,
            initial: None,
            bounds: None,
            tolerance: 1e-8,
        }
    }

    /// Set starting point (optional - will use intelligent default if not provided)
    pub fn from(mut self, x0: f64) -> Self {
        self.initial = Some(x0);
        self
    }

    /// Set bounds: x must be in [min, max]
    pub fn bounds(mut self, min: f64, max: f64) -> Self {
        self.bounds = Some((min, max));
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Solve the optimization problem
    pub fn solve(self) -> Result<f64> {
        // Convert 1D function to vector form
        let vector_objective = move |params: &VectorF64| (self.objective)(params[0]);
        
        let x0 = self.initial.unwrap_or(0.0);
        let initial = VectorF64::from_slice(&[x0]);
        
        let bounds = self.bounds.map(|(min, max)| {
            (VectorF64::from_slice(&[min]), VectorF64::from_slice(&[max]))
        });
        
        let problem = OptimizationProblem::new(vector_objective, &initial, bounds)
            .with_characteristics(ProblemCharacteristics::SmallScale);
        
        let algorithm = select_algorithm(&problem);
        let result = algorithm.solve(problem)?;
        
        Ok(result.solution[0])
    }
}

/// Find the point (x,y) that minimizes a bivariate function f(x,y)
/// 
/// # Mathematical Specification
/// Given function f: ℝ² → ℝ, find (x*, y*) such that:
/// f(x*, y*) = min{f(x,y) : (x,y) ∈ ℝ² or within bounds}
/// 
/// # Dimensions
/// - Input: Bivariate function f(x,y) where (x,y) ∈ ℝ²
/// - Output: Tuple (x*, y*) ∈ ℝ² (optimal point)
/// 
/// # Complexity
/// - Time: O(k log(1/ε)) where k depends on algorithm (BFGS: k≈n², Nelder-Mead: k≈n)
/// - Space: O(1) for algorithm state
/// 
/// # For AI Code Generation
/// - Returns tuple (f64, f64), not a vector or array
/// - Use `.from(x0, y0)` to set starting point (required for most functions)
/// - Use `.bounds((x_min, x_max), (y_min, y_max))` for box constraints
/// - Automatic algorithm selection based on smoothness
/// - Common uses: Rosenbrock optimization, parameter estimation, calibration
/// 
/// # Example
/// ```
/// use rustlab_optimize::minimize_2d;
/// 
/// // Minimize Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
/// let (x, y) = minimize_2d(|x, y| {
///     (1.0 - x).powi(2) + 100.0 * (y - x*x).powi(2)
/// })
/// .from(-1.2, 1.0)  // Classic starting point
/// .solve()?;
/// 
/// assert!((x - 1.0).abs() < 1e-3);  // Global minimum at (1,1)
/// assert!((y - 1.0).abs() < 1e-3);
/// 
/// // With bounds: minimize x² + y² subject to x,y ∈ [-1,1]
/// let (x, y) = minimize_2d(|x, y| x*x + y*y)
///     .bounds((-1.0, 1.0), (-1.0, 1.0))
///     .solve()?;
/// // Result: (0,0)
/// ```
/// 
/// # Errors
/// - `InvalidInput`: No starting point provided. Use `.from(x0, y0)`
/// - `ConvergenceFailed`: Poor starting point or pathological function.
///   Fix: Try different starting points or check function properties
/// - `BoundsViolation`: Starting point outside bounds.
///   Fix: Choose starting point within feasible region
/// 
/// # See Also
/// - [`minimize_1d`]: Single-variable optimization f(x)
/// - [`minimize`]: N-dimensional optimization f(x⃗) for more than 2 variables
pub fn minimize_2d<F>(f: F) -> Minimize2D<F>
where
    F: Fn(f64, f64) -> f64 + Send + Sync + 'static,
{
    Minimize2D::new(f)
}

/// Builder for 2D minimization
pub struct Minimize2D<F> {
    objective: F,
    initial: Option<(f64, f64)>,
    bounds: Option<((f64, f64), (f64, f64))>,
    tolerance: f64,
}

impl<F> Minimize2D<F>
where
    F: Fn(f64, f64) -> f64 + Send + Sync + 'static,
{
    fn new(objective: F) -> Self {
        Self {
            objective,
            initial: None,
            bounds: None,
            tolerance: 1e-8,
        }
    }

    /// Set starting point
    pub fn from(mut self, x0: f64, y0: f64) -> Self {
        self.initial = Some((x0, y0));
        self
    }

    /// Set bounds: x ∈ [x_min, x_max], y ∈ [y_min, y_max]
    pub fn bounds(mut self, x_bounds: (f64, f64), y_bounds: (f64, f64)) -> Self {
        self.bounds = Some((x_bounds, y_bounds));
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Solve the optimization problem
    pub fn solve(self) -> Result<(f64, f64)> {
        // Convert 2D function to vector form
        let vector_objective = move |params: &VectorF64| (self.objective)(params[0], params[1]);
        
        let initial = match self.initial {
            Some((x, y)) => VectorF64::from_slice(&[x, y]),
            None => VectorF64::zeros(2), // Math-first default
        };

        let bounds = self.bounds.map(|((x_min, x_max), (y_min, y_max))| {
            (VectorF64::from_slice(&[x_min, y_min]), VectorF64::from_slice(&[x_max, y_max]))
        });

        let problem = OptimizationProblem::new(vector_objective, &initial, bounds)
            .with_characteristics(ProblemCharacteristics::SmallScale);
        
        let algorithm = select_algorithm(&problem);
        let result = algorithm.solve(problem)?;

        Ok((result.solution[0], result.solution[1]))
    }
}

/// Find the vector x⃗ that minimizes a multivariable function f(x⃗)
/// 
/// # Mathematical Specification
/// Given function f: ℝⁿ → ℝ, find x⃗* such that:
/// f(x⃗*) = min{f(x⃗) : x⃗ ∈ ℝⁿ or within bounds/constraints}
/// 
/// # Dimensions
/// - Input: Multivariate function f(x⃗) where x⃗ ∈ ℝⁿ, n ≥ 1
/// - Output: OptimizationResult with solution ∈ ℝⁿ
/// 
/// # Complexity
/// - Time: O(n² k) for BFGS, O(n k) for gradient descent, where k = iterations
/// - Space: O(n²) for BFGS approximation, O(n) for others
/// 
/// # For AI Code Generation
/// - Returns OptimizationResult struct, access solution via result.solution
/// - Always provide starting point via `.from(&[x0, x1, ...])`
/// - Use `.bounds(&lower, &upper)` for box constraints
/// - Use `.fix_parameter(i, value)` to hold parameters constant
/// - Automatic algorithm selection: BFGS (small), L-BFGS/GD (large)
/// - Common uses: maximum likelihood, loss minimization, hyperparameter tuning
/// 
/// # Example
/// ```
/// use rustlab_optimize::minimize;
/// 
/// // Minimize sphere function: f(x) = Σᵢ xᵢ²
/// let result = minimize(|x| x.iter().map(|&xi| xi * xi).sum())
///     .from(&[1.0, 2.0, 3.0])
///     .solve()?;
/// 
/// for &xi in &result.solution {
///     assert!(xi.abs() < 1e-6);  // Should converge to origin
/// }
/// 
/// // With parameter fixing: optimize f(x,y,z) but keep y=2.0 fixed
/// let result = minimize(|x| x[0].powi(2) + x[1].powi(2) + x[2].powi(2))
///     .from(&[1.0, 2.0, 3.0])
///     .fix_parameter(1, 2.0)  // y fixed at 2.0
///     .solve()?;
/// 
/// // With bounds: constrain all variables to [-5, 5]
/// let result = minimize(|x| x.iter().map(|&xi| xi.powi(4)).sum())
///     .from(&[2.0, -3.0, 1.0])
///     .bounds(&[-5.0, -5.0, -5.0], &[5.0, 5.0, 5.0])
///     .solve()?;
/// ```
/// 
/// # Errors
/// - `InvalidInput`: No starting point provided. Always use `.from(&initial)`
/// - `DimensionMismatch`: Bounds arrays don't match parameter dimension.
///   Fix: Ensure lower.len() == upper.len() == initial.len()
/// - `ConvergenceFailed`: Poor conditioning or too strict tolerance.
///   Fix: Scale variables, adjust tolerance, or try different algorithm
/// 
/// # See Also
/// - [`minimize_1d`]: Optimized for single variables
/// - [`minimize_2d`]: Optimized for two variables
/// - [`least_squares`]: Specialized for sum-of-squares objectives
pub fn minimize<F>(f: F) -> MinimizeND<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
{
    MinimizeND::new(f)
}

/// Builder for N-dimensional minimization
pub struct MinimizeND<F> {
    objective: F,
    initial: Option<Vec<f64>>,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    fixed_params: Option<Vec<(usize, f64)>>,  // (parameter_index, fixed_value)
    algorithm: Option<Algorithm>,
    characteristics: Option<ProblemCharacteristics>,
    tolerance: f64,
    max_iterations: Option<usize>,
}

impl<F> MinimizeND<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
{
    fn new(objective: F) -> Self {
        Self {
            objective,
            initial: None,
            bounds: None,
            fixed_params: None,
            algorithm: None,
            characteristics: None,
            tolerance: 1e-8,
            max_iterations: None,
        }
    }

    /// Set initial point
    pub fn from(mut self, x0: &[f64]) -> Self {
        self.initial = Some(x0.to_vec());
        self
    }

    /// Set box bounds: x_i ∈ [lower_i, upper_i]
    pub fn bounds(mut self, lower: &[f64], upper: &[f64]) -> Self {
        self.bounds = Some((lower.to_vec(), upper.to_vec()));
        self
    }

    /// Fix parameter at specific index to a constant value
    /// 
    /// # Arguments
    /// * `index` - Parameter index to fix (0-based)
    /// * `value` - Value to fix the parameter at
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_optimize::minimize;
    /// 
    /// // Optimize f(x,y,z) but keep y fixed at 2.0
    /// let result = minimize(|params| params[0].powi(2) + params[1].powi(2) + params[2].powi(2))
    ///     .from(&[1.0, 2.0, 3.0])
    ///     .fix_parameter(1, 2.0)  // Fix y = 2.0
    ///     .solve()?;
    /// ```
    pub fn fix_parameter(mut self, index: usize, value: f64) -> Self {
        self.fixed_params.get_or_insert_with(Vec::new).push((index, value));
        self
    }

    /// Fix multiple parameters at once
    /// 
    /// # Arguments
    /// * `fixed` - Slice of (index, value) pairs
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_optimize::minimize;
    /// 
    /// // Fix first and third parameters
    /// let result = minimize(|params| /* objective */)
    ///     .from(&[1.0, 2.0, 3.0, 4.0])
    ///     .fix_parameters(&[(0, 1.5), (2, 3.5)])  // Fix x[0]=1.5, x[2]=3.5
    ///     .solve()?;
    /// ```
    pub fn fix_parameters(mut self, fixed: &[(usize, f64)]) -> Self {
        self.fixed_params = Some(fixed.to_vec());
        self
    }

    /// Fix parameters using a mask: true = optimize, false = fix at initial value
    /// 
    /// This is convenient when you want to fix parameters at their initial values.
    /// 
    /// # Arguments
    /// * `mask` - Boolean mask where true means optimize, false means fix
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_optimize::minimize;
    /// 
    /// let result = minimize(|params| /* objective */)
    ///     .from(&[1.0, 2.0, 3.0, 4.0])
    ///     .optimize_mask(&[true, false, true, false])  // Optimize x[0] and x[2] only
    ///     .solve()?;
    /// ```
    pub fn optimize_mask(mut self, mask: &[bool]) -> Self {
        if let Some(initial) = &self.initial {
            let fixed: Vec<(usize, f64)> = mask.iter()
                .enumerate()
                .filter_map(|(i, &optimize)| {
                    if !optimize && i < initial.len() {
                        Some((i, initial[i]))
                    } else {
                        None
                    }
                })
                .collect();
            self.fixed_params = Some(fixed);
        }
        self
    }

    /// Explicitly choose Levenberg-Marquardt (best for sum-of-squares)
    pub fn using_levenberg_marquardt(mut self) -> Self {
        self.algorithm = Some(Algorithm::LevenbergMarquardt);
        self
    }

    /// Explicitly choose gradient descent (good for large problems)
    pub fn using_gradient_descent(mut self) -> Self {
        self.algorithm = Some(Algorithm::GradientDescent);
        self
    }

    /// Explicitly choose Nelder-Mead (derivative-free, robust)
    pub fn using_nelder_mead(mut self) -> Self {
        self.algorithm = Some(Algorithm::NelderMead);
        self
    }

    /// Explicitly choose BFGS (fast for smooth problems)
    pub fn using_bfgs(mut self) -> Self {
        self.algorithm = Some(Algorithm::BFGS);
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = Some(max_iter);
        self
    }

    /// Set problem characteristics to guide algorithm selection
    pub fn with_characteristics(mut self, characteristics: ProblemCharacteristics) -> Self {
        self.characteristics = Some(characteristics);
        self
    }

    /// Build optimization problem for advanced use (algorithm selection, etc.)
    pub fn build_problem(self) -> Result<OptimizationProblem> {
        let initial_vec = self.initial.ok_or_else(|| {
            Error::InvalidInput("Initial point required for N-D optimization".into())
        })?;
        
        let initial = VectorF64::from_slice(&initial_vec);

        let bounds = self.bounds.map(|(lower, upper)| {
            (VectorF64::from_slice(&lower), VectorF64::from_slice(&upper))
        });

        // Create adapter closure that converts VectorF64 to &[f64] for the user's function
        let objective_adapter = move |params: &VectorF64| -> f64 {
            (self.objective)(params.as_slice_unchecked())
        };

        let mut problem = OptimizationProblem::new(objective_adapter, &initial, bounds);
        
        // Apply parameter fixing if specified
        if let Some(fixed) = self.fixed_params {
            problem = problem.fix_parameters(&fixed);
        }
        
        // Set characteristics based on problem size or explicit setting
        let characteristics = if let Some(specified_characteristics) = self.characteristics {
            specified_characteristics
        } else if initial.len() > 1000 {
            ProblemCharacteristics::LargeScale
        } else if initial.len() <= 10 {
            ProblemCharacteristics::SmallScale
        } else {
            ProblemCharacteristics::General
        };
        problem = problem.with_characteristics(characteristics);
        
        Ok(problem)
    }
    
    /// Solve the optimization problem
    pub fn solve(self) -> Result<OptimizationResult> {
        let initial_vec = self.initial.ok_or_else(|| {
            Error::InvalidInput("Initial point required for N-D optimization".into())
        })?;
        
        let initial = VectorF64::from_slice(&initial_vec);

        let bounds = self.bounds.map(|(lower, upper)| {
            (VectorF64::from_slice(&lower), VectorF64::from_slice(&upper))
        });

        // Create adapter closure that converts VectorF64 to &[f64] for the user's function
        let objective_adapter = move |params: &VectorF64| -> f64 {
            (self.objective)(params.as_slice_unchecked())
        };

        let mut problem = OptimizationProblem::new(objective_adapter, &initial, bounds);
        
        // Apply parameter fixing if specified
        if let Some(fixed) = self.fixed_params {
            problem = problem.fix_parameters(&fixed);
        }
        
        // Set characteristics based on problem size
        let characteristics = if initial.len() > 1000 {
            ProblemCharacteristics::LargeScale
        } else if initial.len() <= 10 {
            ProblemCharacteristics::SmallScale
        } else {
            ProblemCharacteristics::General
        };
        problem = problem.with_characteristics(characteristics);
        
        let algorithm = match self.algorithm {
            Some(Algorithm::LevenbergMarquardt) => {
                Box::new(crate::algorithms::levenberg_marquardt::LevenbergMarquardt::new()) as Box<dyn crate::algorithms::Solver>
            },
            Some(Algorithm::BFGS) => {
                Box::new(crate::algorithms::bfgs::BFGS::new()) as Box<dyn crate::algorithms::Solver>
            },
            Some(Algorithm::GradientDescent) => {
                Box::new(crate::algorithms::gradient_descent::GradientDescent::new()) as Box<dyn crate::algorithms::Solver>
            },
            Some(Algorithm::NelderMead) => {
                Box::new(crate::algorithms::nelder_mead::NelderMead::new()) as Box<dyn crate::algorithms::Solver>
            },
            _ => select_algorithm(&problem),
        };

        algorithm.solve(problem)
    }
}

/// Solve nonlinear least squares: minimize ||r(x⃗)||² where r(x⃗) are residuals
/// 
/// # Mathematical Specification
/// Given residual function r: ℝⁿ → ℝᵐ, solve:
/// minimize ½||r(x⃗)||² = ½Σᵢ rᵢ(x⃗)² over x⃗ ∈ ℝⁿ
/// 
/// Uses Levenberg-Marquardt by default (Gauss-Newton + gradient descent)
/// 
/// # Dimensions
/// - Input: Residual function r(x⃗) returning m residuals for n parameters
/// - Output: OptimizationResult with solution ∈ ℝⁿ
/// - Constraint: typically m ≥ n (overdetermined system)
/// 
/// # Complexity
/// - Time: O(mn² + n³) per iteration for Jacobian computation and solve
/// - Space: O(mn + n²) for Jacobian matrix and normal equations
/// 
/// # For AI Code Generation
/// - Residual function must return Vec<f64>, not scalar
/// - Each residual corresponds to one data point or constraint
/// - Levenberg-Marquardt is optimal for this problem type
/// - Use for curve fitting, parameter estimation, data reconciliation
/// - Returns same OptimizationResult as minimize(), access via result.solution
/// 
/// # Example
/// ```
/// use rustlab_optimize::least_squares;
/// 
/// // Fit exponential decay: data = A*exp(-k*t) + noise
/// let t_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y_data = vec![10.0, 6.7, 4.5, 3.0, 2.0];
/// 
/// let residual_fn = move |params: &[f64]| -> Vec<f64> {
///     let amplitude = params[0];
///     let decay_rate = params[1];
///     
///     t_data.iter().zip(y_data.iter())
///         .map(|(&t, &y)| {
///             let predicted = amplitude * (-decay_rate * t).exp();
///             y - predicted  // residual = observed - predicted
///         })
///         .collect()
/// };
/// 
/// let result = least_squares(residual_fn)
///     .from(&[8.0, 0.5])  // Initial guess [A₀, k₀]
///     .solve()?;
/// 
/// let fitted_amplitude = result.solution[0];
/// let fitted_decay_rate = result.solution[1];
/// ```
/// 
/// # Errors
/// - `InvalidInput`: No starting point provided. Always use `.from(&initial)`
/// - `ConvergenceFailed`: Singular Jacobian or poor conditioning.
///   Fix: Better initial guess, regularization, or check data quality
/// - `NumericalError`: Residual function returns NaN or infinite values.
///   Fix: Add bounds or validate input parameters
/// 
/// # See Also
/// - [`curve_fit`]: High-level interface for data fitting with x,y arrays
/// - [`fit_exponential`]: Specialized exponential fitting
/// - [`minimize`]: General optimization for non-sum-of-squares objectives
pub fn least_squares<F>(residual: F) -> LeastSquares<F>
where
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
{
    LeastSquares::new(residual)
}

/// Builder for least squares optimization problems
pub struct LeastSquares<F> {
    residual: F,
    initial: Option<Vec<f64>>,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    fixed_params: Option<Vec<(usize, f64)>>,
    algorithm: Option<Algorithm>,
    tolerance: f64,
    max_iterations: Option<usize>,
}

impl<F> LeastSquares<F>
where
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
{
    fn new(residual: F) -> Self {
        Self {
            residual,
            initial: None,
            bounds: None,
            fixed_params: None,
            algorithm: Some(Algorithm::LevenbergMarquardt), // Default to LM for least squares
            tolerance: 1e-8,
            max_iterations: None,
        }
    }

    /// Set initial point
    pub fn from(mut self, x0: &[f64]) -> Self {
        self.initial = Some(x0.to_vec());
        self
    }

    /// Set box bounds: x_i ∈ [lower_i, upper_i]
    pub fn bounds(mut self, lower: &[f64], upper: &[f64]) -> Self {
        self.bounds = Some((lower.to_vec(), upper.to_vec()));
        self
    }

    /// Fix parameter at specific index to a constant value
    pub fn fix_parameter(mut self, index: usize, value: f64) -> Self {
        self.fixed_params.get_or_insert_with(Vec::new).push((index, value));
        self
    }

    /// Fix multiple parameters at once
    pub fn fix_parameters(mut self, fixed: &[(usize, f64)]) -> Self {
        self.fixed_params = Some(fixed.to_vec());
        self
    }

    /// Use Levenberg-Marquardt (default for least squares)
    pub fn using_levenberg_marquardt(mut self) -> Self {
        self.algorithm = Some(Algorithm::LevenbergMarquardt);
        self
    }

    /// Use BFGS instead of LM (converts to scalar objective)
    pub fn using_bfgs(mut self) -> Self {
        self.algorithm = Some(Algorithm::BFGS);
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = Some(max_iter);
        self
    }

    /// Solve the least squares problem
    pub fn solve(self) -> Result<OptimizationResult> {
        let initial_vec = self.initial.ok_or_else(|| {
            Error::InvalidInput("Initial point required for least squares optimization".into())
        })?;
        
        let initial = VectorF64::from_slice(&initial_vec);
        let bounds = self.bounds.map(|(lower, upper)| {
            (VectorF64::from_slice(&lower), VectorF64::from_slice(&upper))
        });

        // Create residual adapter that converts VectorF64 to &[f64]
        let residual_adapter = move |params: &VectorF64| -> VectorF64 {
            let residuals = (self.residual)(params.as_slice_unchecked());
            VectorF64::from_slice(&residuals)
        };

        let mut problem = OptimizationProblem::least_squares(residual_adapter, &initial, bounds);
        
        // Apply parameter fixing if specified
        if let Some(fixed) = self.fixed_params {
            problem = problem.fix_parameters(&fixed);
        }
        
        let algorithm = match self.algorithm {
            Some(Algorithm::LevenbergMarquardt) => {
                Box::new(crate::algorithms::levenberg_marquardt::LevenbergMarquardt::new()) as Box<dyn crate::algorithms::Solver>
            },
            Some(Algorithm::BFGS) => {
                Box::new(crate::algorithms::bfgs::BFGS::new()) as Box<dyn crate::algorithms::Solver>
            },
            _ => {
                // Default to LM for least squares
                Box::new(crate::algorithms::levenberg_marquardt::LevenbergMarquardt::new()) as Box<dyn crate::algorithms::Solver>
            },
        };

        algorithm.solve(problem)
    }
}

/// Fit arbitrary mathematical models to experimental data
/// 
/// # Mathematical Specification
/// Given data points (xᵢ, yᵢ) and model function m(x, θ⃗), solve:
/// minimize Σᵢ [yᵢ - m(xᵢ, θ⃗)]² over parameters θ⃗ ∈ ℝⁿ
/// 
/// This is curve fitting via nonlinear least squares.
/// 
/// # Dimensions
/// - Input: x_data (k points), y_data (k points), model function m(x, θ⃗)
/// - Output: OptimizationResult with fitted parameters θ⃗ ∈ ℝⁿ
/// - Constraint: x_data.len() == y_data.len(), typically k ≥ n
/// 
/// # Complexity
/// - Time: O(kn² + n³) per iteration where k = data points, n = parameters
/// - Space: O(kn + n²) for Jacobian and normal equations
/// 
/// # For AI Code Generation
/// - Model function signature: |x: f64, params: &[f64]| -> f64
/// - Always provide initial parameter guess via `.with_initial(&[...])`
/// - Use `.fix_parameter(i, value)` to constrain specific parameters
/// - Automatic Levenberg-Marquardt solver (optimal for curve fitting)
/// - Common uses: exponential fitting, polynomial regression, custom models
/// 
/// # Example
/// ```
/// use rustlab_optimize::curve_fit;
/// use rustlab_math::vec64;
/// 
/// // Fit exponential model: y = A*exp(-k*x) + C
/// let x_data = vec64![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y_data = vec64![10.2, 6.1, 3.7, 2.3, 1.4];
/// 
/// let exp_model = |x: f64, params: &[f64]| {
///     let amplitude = params[0];     // A
///     let decay_rate = params[1];    // k  
///     let baseline = params[2];      // C
///     amplitude * (-decay_rate * x).exp() + baseline
/// };
/// 
/// let result = curve_fit(&x_data, &y_data, exp_model)
///     .with_initial(&[10.0, 0.5, 0.0])  // [A₀, k₀, C₀]
///     .bounds(&[0.0, 0.0, -1.0], &[20.0, 2.0, 1.0])  // Physical constraints
///     .solve()?;
/// 
/// let fitted_params = result.solution;  // [A_fit, k_fit, C_fit]
/// 
/// // Fix baseline at zero (pure exponential decay)
/// let result = curve_fit(&x_data, &y_data, exp_model)
///     .with_initial(&[10.0, 0.5, 0.0])
///     .fix_parameter(2, 0.0)  // Force C = 0
///     .solve()?;
/// ```
/// 
/// # Errors
/// - `InvalidInput`: Missing initial guess. Always use `.with_initial(&[...])`
/// - `DimensionMismatch`: x_data.len() != y_data.len().
///   Fix: Ensure equal length data arrays
/// - `ConvergenceFailed`: Poor model fit or bad initial guess.
///   Fix: Check model form, try different starting values, add bounds
/// 
/// # See Also
/// - [`fit_exponential`]: Specialized for exponential models
/// - [`fit_linear`]: Optimized for linear regression
/// - [`least_squares`]: Lower-level residual-based interface
pub fn curve_fit<F>(x_data: &VectorF64, y_data: &VectorF64, model: F) -> CurveFit<F>
where
    F: Fn(f64, &[f64]) -> f64 + Send + Sync + 'static,
{
    CurveFit::new(x_data.as_slice_unchecked(), y_data.as_slice_unchecked(), model)
}

/// Builder for curve fitting optimization with method chaining
pub struct CurveFit<F> {
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    model: F,
    initial: Option<Vec<f64>>,
    bounds: Option<(Vec<f64>, Vec<f64>)>,
    fixed_params: Option<Vec<(usize, f64)>>,
    algorithm: Option<Algorithm>,
    tolerance: f64,
    max_iterations: Option<usize>,
}

impl<F> CurveFit<F>
where
    F: Fn(f64, &[f64]) -> f64 + Send + Sync + 'static,
{
    fn new(x_data: &[f64], y_data: &[f64], model: F) -> Self {
        Self {
            x_data: x_data.to_vec(),
            y_data: y_data.to_vec(),
            model,
            initial: None,
            bounds: None,
            fixed_params: None,
            algorithm: None,
            tolerance: 1e-8,
            max_iterations: None,
        }
    }

    /// Set initial parameter guess
    pub fn with_initial(mut self, params: &[f64]) -> Self {
        self.initial = Some(params.to_vec());
        self
    }

    /// Set parameter bounds: params[i] must be in [lower[i], upper[i]]
    pub fn bounds(mut self, lower: &[f64], upper: &[f64]) -> Self {
        self.bounds = Some((lower.to_vec(), upper.to_vec()));
        self
    }

    /// Fix a specific parameter to a constant value
    pub fn fix_parameter(mut self, param_index: usize, value: f64) -> Self {
        if let Some(ref mut fixed) = self.fixed_params {
            fixed.push((param_index, value));
        } else {
            self.fixed_params = Some(vec![(param_index, value)]);
        }
        self
    }

    /// Force use of Levenberg-Marquardt algorithm (default for curve fitting)
    pub fn using_levenberg_marquardt(mut self) -> Self {
        self.algorithm = Some(Algorithm::LevenbergMarquardt);
        self
    }

    /// Force use of BFGS algorithm
    pub fn using_bfgs(mut self) -> Self {
        self.algorithm = Some(Algorithm::BFGS);
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = Some(max_iter);
        self
    }

    /// Solve the curve fitting problem
    pub fn solve(self) -> Result<OptimizationResult> {
        let initial_vec = self.initial.ok_or_else(|| {
            Error::InvalidInput("Initial parameter guess required for curve fitting".into())
        })?;
        
        if self.x_data.len() != self.y_data.len() {
            return Err(Error::InvalidInput("x_data and y_data must have the same length".into()));
        }

        // Create residual function that converts model to residuals
        let x_data = self.x_data.clone();
        let y_data = self.y_data.clone();
        let model = self.model;
        
        let residual_function = move |params: &[f64]| -> Vec<f64> {
            x_data.iter().zip(y_data.iter())
                .map(|(&x, &y)| y - model(x, params))
                .collect()
        };

        // Use the existing least_squares builder with our residual function
        let mut builder = least_squares(residual_function)
            .from(&initial_vec)
            .tolerance(self.tolerance);

        if let Some((lower, upper)) = self.bounds {
            builder = builder.bounds(&lower, &upper);
        }

        if let Some(fixed) = self.fixed_params {
            for (index, value) in fixed {
                builder = builder.fix_parameter(index, value);
            }
        }

        if let Some(algorithm) = self.algorithm {
            builder = match algorithm {
                Algorithm::LevenbergMarquardt => builder.using_levenberg_marquardt(),
                Algorithm::BFGS => builder.using_bfgs(),
                _ => builder, // Use default
            };
        }

        if let Some(max_iter) = self.max_iterations {
            builder = builder.max_iterations(max_iter);
        }

        builder.solve()
    }
}