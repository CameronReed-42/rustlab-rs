//! Algorithm infrastructure and solver traits
//!
//! This module provides the core abstractions for optimization algorithms,
//! emphasizing math-first expressions using rustlab-math.

use rustlab_math::{VectorF64, ArrayF64};
use crate::core::{Result, OptimizationResult, Algorithm, ConvergenceStatus};
use crate::core::result::{OptimizationInfo};

pub mod levenberg_marquardt;
pub mod bfgs;
pub mod gradient_descent;
pub mod nelder_mead;

/// Core trait defining the interface for optimization algorithm implementations
/// 
/// # Mathematical Specification
/// All solvers implement the interface: solve(problem) → OptimizationResult
/// where problem defines f(x⃗), constraints, and algorithm hints.
/// 
/// # For AI Code Generation
/// - Trait used internally by the library - users typically don't implement this
/// - Use minimize(), least_squares(), or curve_fit() instead of calling solvers directly
/// - All solvers support bounds and parameter fixing via transformation
/// - Access via algorithm selection: .using_bfgs(), .using_levenberg_marquardt(), etc.
pub trait Solver: Send + Sync {
    /// Solve the optimization problem
    fn solve(&self, problem: OptimizationProblem) -> Result<OptimizationResult>;
    
    /// Get the algorithm type
    fn algorithm(&self) -> Algorithm;
    
    /// Check if this solver supports bounds
    /// Note: All solvers now support bounds via parameter transformation
    fn supports_bounds(&self) -> bool { true }
    
    /// Check if this solver supports parameter fixing
    fn supports_parameter_fixing(&self) -> bool { true }
}

/// Mathematical optimization problem specification using RustLab vector operations
/// 
/// # Mathematical Specification
/// Encapsulates:
/// - Objective function f: ℝⁿ → ℝ (scalar) or r: ℝⁿ → ℝᵐ (residuals)
/// - Initial point x⃗₀ ∈ ℝⁿ  
/// - Box constraints: x⃗ ∈ [lower, upper]
/// - Fixed parameters: x[i] = constant for i ∈ fixed_indices
/// 
/// # For AI Code Generation
/// - Created automatically by minimize(), least_squares(), curve_fit()
/// - Users don't typically construct this directly
/// - Contains bounds transformation and parameter fixing logic
/// - Provides math-first operations like .evaluate(), .numerical_gradient()
#[derive(Debug)]
pub struct OptimizationProblem {
    /// Objective function
    pub objective: ObjectiveFunction,
    
    /// Initial parameter values (using rustlab-math VectorF64)
    pub initial: VectorF64,
    
    /// Parameter bounds (optional)
    pub bounds: Option<(VectorF64, VectorF64)>,
    
    /// Fixed parameters: (index, value) pairs
    pub fixed_params: Vec<(usize, f64)>,
    
    /// Problem characteristics for algorithm selection
    pub characteristics: ProblemCharacteristics,
}

/// Objective function types with math-first signatures
pub enum ObjectiveFunction {
    /// Scalar objective: f(x) → ℝ
    Scalar(Box<dyn Fn(&VectorF64) -> f64 + Send + Sync>),
    
    /// Residual function: r(x) → ℝⁿ (for least squares)
    Residual(Box<dyn Fn(&VectorF64) -> VectorF64 + Send + Sync>),
    
    /// Residual with Jacobian: (r(x), J(x)) → (ℝⁿ, ℝⁿˣᵐ)
    ResidualWithJacobian(
        Box<dyn Fn(&VectorF64) -> VectorF64 + Send + Sync>,
        Box<dyn Fn(&VectorF64) -> ArrayF64 + Send + Sync>,
    ),
}

/// Problem characteristics for intelligent algorithm selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProblemCharacteristics {
    /// General unconstrained optimization
    General,
    
    /// Least squares problem (sum of squared residuals)
    LeastSquares,
    
    /// Large scale (>1000 parameters)
    LargeScale,
    
    /// Small scale (<10 parameters)
    SmallScale,
    
    /// Curve fitting with experimental data
    CurveFitting,
    
    /// Noisy or non-smooth objective
    NoisyNonSmooth,
    
    /// Likely multimodal (global optimization needed)
    Multimodal,
}

impl std::fmt::Debug for ObjectiveFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectiveFunction::Scalar(_) => write!(f, "Scalar objective function"),
            ObjectiveFunction::Residual(_) => write!(f, "Residual function"),
            ObjectiveFunction::ResidualWithJacobian(_, _) => write!(f, "Residual with Jacobian"),
        }
    }
}

impl OptimizationProblem {
    /// Create new optimization problem with math-first interface
    pub fn new<F>(objective: F, initial: &VectorF64, bounds: Option<(VectorF64, VectorF64)>) -> Self
    where
        F: Fn(&VectorF64) -> f64 + Send + Sync + 'static,
    {
        Self {
            objective: ObjectiveFunction::Scalar(Box::new(objective)),
            initial: initial.clone(),
            bounds,
            fixed_params: Vec::new(),
            characteristics: ProblemCharacteristics::General,
        }
    }
    
    /// Create least squares problem from residual function
    pub fn least_squares<F>(residual: F, initial: &VectorF64, bounds: Option<(VectorF64, VectorF64)>) -> Self
    where
        F: Fn(&VectorF64) -> VectorF64 + Send + Sync + 'static,
    {
        Self {
            objective: ObjectiveFunction::Residual(Box::new(residual)),
            initial: initial.clone(),
            bounds,
            fixed_params: Vec::new(),
            characteristics: ProblemCharacteristics::LeastSquares,
        }
    }
    
    /// Create curve fitting problem (specialized least squares)
    pub fn curve_fitting<F>(model: F, x_data: &VectorF64, y_data: &VectorF64, initial: &VectorF64) -> Self
    where
        F: Fn(&VectorF64, &VectorF64) -> VectorF64 + Send + Sync + 'static,
    {
        let x_data = x_data.clone();
        let y_data = y_data.clone();
        
        let residual = move |params: &VectorF64| -> VectorF64 {
            let predictions = model(params, &x_data);
            &predictions - &y_data  // Math-first: vector subtraction
        };
        
        Self {
            objective: ObjectiveFunction::Residual(Box::new(residual)),
            initial: initial.clone(),
            bounds: None,
            fixed_params: Vec::new(),
            characteristics: ProblemCharacteristics::CurveFitting,
        }
    }
    
    /// Add parameter fixing constraints
    pub fn fix_parameters(mut self, fixed: &[(usize, f64)]) -> Self {
        self.fixed_params = fixed.to_vec();
        self
    }
    
    /// Set problem characteristics for algorithm selection
    pub fn with_characteristics(mut self, characteristics: ProblemCharacteristics) -> Self {
        self.characteristics = characteristics;
        self
    }
    
    /// Get effective dimension (accounting for fixed parameters)
    pub fn effective_dimension(&self) -> usize {
        self.initial.len() - self.fixed_params.len()
    }
    
    /// Evaluate objective function with math-first operations
    pub fn evaluate(&self, params: &VectorF64) -> f64 {
        match &self.objective {
            ObjectiveFunction::Scalar(f) => f(params),
            ObjectiveFunction::Residual(r) => {
                let residuals = r(params);
                residuals.dot(&residuals)  // Math-first: ||r||² using dot product
            }
            ObjectiveFunction::ResidualWithJacobian(r, _) => {
                let residuals = r(params);
                residuals.dot(&residuals)  // Math-first: ||r||² using dot product
            }
        }
    }
    
    /// Evaluate residual function (for least squares problems)
    pub fn evaluate_residual(&self, params: &VectorF64) -> Option<VectorF64> {
        match &self.objective {
            ObjectiveFunction::Residual(r) => Some(r(params)),
            ObjectiveFunction::ResidualWithJacobian(r, _) => Some(r(params)),
            ObjectiveFunction::Scalar(_) => None,
        }
    }
    
    /// Compute numerical gradient using central differences
    pub fn numerical_gradient(&self, params: &VectorF64, h: f64) -> VectorF64 {
        let n = params.len();
        let mut gradient = VectorF64::zeros(n);
        
        for i in 0..n {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            
            // Math-first: direct indexing and arithmetic
            params_plus[i] += h;
            params_minus[i] -= h;
            
            let f_plus = self.evaluate(&params_plus);
            let f_minus = self.evaluate(&params_minus);
            
            gradient[i] = (f_plus - f_minus) / (2.0 * h);
        }
        
        gradient
    }
    
    /// Apply parameter fixing transformation
    /// Maps reduced parameters to full parameter vector
    pub fn expand_parameters(&self, reduced_params: &VectorF64) -> VectorF64 {
        if self.fixed_params.is_empty() {
            return reduced_params.clone();
        }
        
        let mut full_params = VectorF64::zeros(self.initial.len());
        let mut reduced_idx = 0;
        
        for i in 0..self.initial.len() {
            if let Some(&(_, fixed_value)) = self.fixed_params.iter().find(|(idx, _)| *idx == i) {
                full_params[i] = fixed_value;
            } else {
                full_params[i] = reduced_params[reduced_idx];
                reduced_idx += 1;
            }
        }
        
        full_params
    }
    
    /// Extract free parameters from full parameter vector
    pub fn reduce_parameters(&self, full_params: &VectorF64) -> VectorF64 {
        if self.fixed_params.is_empty() {
            return full_params.clone();
        }
        
        let mut reduced = Vec::new();
        
        for i in 0..full_params.len() {
            if !self.fixed_params.iter().any(|(idx, _)| *idx == i) {
                reduced.push(full_params[i]);
            }
        }
        
        VectorF64::from_slice(&reduced)
    }
    
    /// Check if problem has parameter bounds
    pub fn has_bounds(&self) -> bool {
        self.bounds.is_some()
    }
    
    /// Get bounds as Bounds struct for transformation
    pub fn get_bounds(&self) -> Option<crate::bounds::Bounds> {
        self.bounds.as_ref().map(|(lower, upper)| {
            crate::bounds::Bounds::new(lower.clone(), upper.clone()).unwrap()
        })
    }
    
    /// Create bounds transformer for this problem
    pub fn create_bounds_transformer(&self) -> Option<crate::bounds::BoundsTransformer> {
        self.get_bounds().map(crate::bounds::BoundsTransformer::new)
    }
    
    /// Evaluate objective with bounds checking and transformation
    /// This handles bounds transformation transparently for algorithms
    pub fn evaluate_with_bounds(&self, unbounded_params: &VectorF64) -> f64 {
        if let Some(transformer) = self.create_bounds_transformer() {
            // Transform from unbounded space to bounded space
            match transformer.to_bounded(unbounded_params) {
                Ok(bounded_params) => self.evaluate(&bounded_params),
                Err(_) => f64::INFINITY, // Return high cost for invalid parameters
            }
        } else {
            // No bounds, evaluate directly
            self.evaluate(unbounded_params)
        }
    }
    
    /// Get initial parameters transformed to unbounded space
    pub fn initial_unbounded(&self) -> crate::core::Result<VectorF64> {
        if let Some(transformer) = self.create_bounds_transformer() {
            transformer.to_unbounded(&self.initial)
        } else {
            Ok(self.initial.clone())
        }
    }
    
    /// Transform gradient from bounded to unbounded space
    pub fn transform_gradient_to_unbounded(&self, bounded_gradient: &VectorF64, unbounded_params: &VectorF64) -> crate::core::Result<VectorF64> {
        if let Some(transformer) = self.create_bounds_transformer() {
            transformer.transform_gradient(bounded_gradient, unbounded_params)
        } else {
            Ok(bounded_gradient.clone())
        }
    }
}

/// Automatically select optimal optimization algorithm based on problem properties
/// 
/// # Mathematical Specification
/// Algorithm selection rules:
/// - Least squares/curve fitting → Levenberg-Marquardt (gold standard)
/// - Small scale (n ≤ 10) → BFGS (superlinear convergence)
/// - Large scale (n > 1000) → Gradient descent (memory efficient)
/// - Noisy/non-smooth → Nelder-Mead (derivative-free)
/// 
/// # For AI Code Generation
/// - Called automatically by minimize(), least_squares(), curve_fit()
/// - Override with explicit algorithm choice: .using_bfgs(), .using_levenberg_marquardt()
/// - All selected algorithms support bounds via parameter transformation
/// - Returns Box<dyn Solver> - use solver.solve(problem) to optimize
pub fn select_algorithm(problem: &OptimizationProblem) -> Box<dyn Solver> {
    let has_bounds = problem.has_bounds();
    let has_fixed_params = !problem.fixed_params.is_empty();
    let dimension = problem.initial.len();
    
    match problem.characteristics {
        ProblemCharacteristics::LeastSquares | ProblemCharacteristics::CurveFitting => {
            // Levenberg-Marquardt is the gold standard for least squares
            // It has full bounds and parameter fixing support
            Box::new(levenberg_marquardt::LevenbergMarquardt::new())
        }
        ProblemCharacteristics::SmallScale => {
            // For small problems, BFGS is excellent and supports bounds
            if has_bounds || has_fixed_params {
                Box::new(bfgs::BFGS::new())
            } else {
                Box::new(bfgs::BFGS::new())
            }
        }
        ProblemCharacteristics::LargeScale => {
            // For large problems, prefer algorithms with bounds support
            if has_bounds {
                // BFGS can handle bounds via transformation, better than gradient descent
                Box::new(bfgs::BFGS::new())
            } else {
                Box::new(gradient_descent::GradientDescent::new())
            }
        }
        ProblemCharacteristics::NoisyNonSmooth => {
            // Nelder-Mead doesn't have explicit bounds support yet
            if has_bounds {
                // Fall back to BFGS for bounds-constrained noisy problems
                Box::new(bfgs::BFGS::new())
            } else {
                Box::new(nelder_mead::NelderMead::new())
            }
        }
        ProblemCharacteristics::General => {
            // Smart selection based on problem properties
            if dimension <= 10 {
                // Small to medium problems: BFGS is excellent
                Box::new(bfgs::BFGS::new())
            } else if dimension <= 100 {
                // Medium problems: BFGS still good
                Box::new(bfgs::BFGS::new())
            } else {
                // Large problems: prefer simpler algorithms
                if has_bounds {
                    Box::new(bfgs::BFGS::new())
                } else {
                    Box::new(gradient_descent::GradientDescent::new())
                }
            }
        }
        ProblemCharacteristics::Multimodal => {
            // For multimodal problems, we need global optimization
            // For now, fall back to BFGS (future: add global optimizers)
            Box::new(bfgs::BFGS::new())
        }
    }
}

/// Advanced algorithm selection with bounds awareness and performance considerations
/// 
/// This function provides more sophisticated algorithm selection by testing candidate
/// algorithms for bounds support and choosing based on problem characteristics.
pub fn select_algorithm_advanced(problem: &OptimizationProblem) -> Box<dyn Solver> {
    let has_bounds = problem.has_bounds();
    let has_fixed_params = !problem.fixed_params.is_empty();
    let dimension = problem.initial.len();
    
    // Define candidate algorithms in preference order
    let candidates: Vec<Box<dyn Solver>> = match problem.characteristics {
        ProblemCharacteristics::LeastSquares | ProblemCharacteristics::CurveFitting => {
            vec![
                Box::new(levenberg_marquardt::LevenbergMarquardt::new()),
                Box::new(bfgs::BFGS::new()),
            ]
        }
        ProblemCharacteristics::SmallScale => {
            vec![
                Box::new(bfgs::BFGS::new()),
                Box::new(levenberg_marquardt::LevenbergMarquardt::new()),
            ]
        }
        ProblemCharacteristics::LargeScale => {
            if dimension > 1000 {
                vec![
                    Box::new(gradient_descent::GradientDescent::new()),
                    Box::new(bfgs::BFGS::new()),
                ]
            } else {
                vec![
                    Box::new(bfgs::BFGS::new()),
                    Box::new(gradient_descent::GradientDescent::new()),
                ]
            }
        }
        ProblemCharacteristics::NoisyNonSmooth => {
            vec![
                Box::new(nelder_mead::NelderMead::new()),
                Box::new(bfgs::BFGS::new()),
            ]
        }
        _ => {
            vec![
                Box::new(bfgs::BFGS::new()),
                Box::new(levenberg_marquardt::LevenbergMarquardt::new()),
                Box::new(gradient_descent::GradientDescent::new()),
            ]
        }
    };
    
    // Select first algorithm that supports required features
    for candidate in candidates {
        let supports_bounds = candidate.supports_bounds();
        let supports_fixing = candidate.supports_parameter_fixing();
        
        // Check if algorithm supports all required features
        let bounds_ok = !has_bounds || supports_bounds;
        let fixing_ok = !has_fixed_params || supports_fixing;
        
        if bounds_ok && fixing_ok {
            return candidate;
        }
    }
    
    // Fallback: BFGS (should always work)
    Box::new(bfgs::BFGS::new())
}

/// Get algorithm recommendation with reasoning
/// 
/// Returns the selected algorithm along with human-readable explanation
/// of why this algorithm was chosen for the given problem.
pub fn recommend_algorithm(problem: &OptimizationProblem) -> (Box<dyn Solver>, String) {
    let has_bounds = problem.has_bounds();
    let has_fixed_params = !problem.fixed_params.is_empty();
    let dimension = problem.initial.len();
    
    let (algorithm, reason) = match problem.characteristics {
        ProblemCharacteristics::LeastSquares | ProblemCharacteristics::CurveFitting => {
            (
                Box::new(levenberg_marquardt::LevenbergMarquardt::new()) as Box<dyn Solver>,
                "Levenberg-Marquardt is the gold standard for least squares problems and curve fitting, with excellent convergence properties and full bounds support".to_string()
            )
        }
        ProblemCharacteristics::SmallScale => {
            (
                Box::new(bfgs::BFGS::new()) as Box<dyn Solver>,
                format!("BFGS is excellent for small-scale problems ({}D) with fast quadratic convergence and bounds support", dimension)
            )
        }
        ProblemCharacteristics::LargeScale => {
            if has_bounds {
                (
                    Box::new(bfgs::BFGS::new()) as Box<dyn Solver>,
                    format!("BFGS chosen for large-scale problem ({}D) with bounds - handles bounds via parameter transformation", dimension)
                )
            } else {
                (
                    Box::new(gradient_descent::GradientDescent::new()) as Box<dyn Solver>,
                    format!("Gradient descent for large-scale unconstrained problem ({}D) - memory efficient", dimension)
                )
            }
        }
        ProblemCharacteristics::NoisyNonSmooth => {
            if has_bounds {
                (
                    Box::new(bfgs::BFGS::new()) as Box<dyn Solver>,
                    "BFGS chosen over Nelder-Mead due to bounds constraints - handles noise reasonably well".to_string()
                )
            } else {
                (
                    Box::new(nelder_mead::NelderMead::new()) as Box<dyn Solver>,
                    "Nelder-Mead is derivative-free and robust for noisy, non-smooth objectives".to_string()
                )
            }
        }
        _ => {
            if dimension <= 10 {
                (
                    Box::new(bfgs::BFGS::new()) as Box<dyn Solver>,
                    format!("BFGS for general {}D problem - excellent convergence and bounds support", dimension)
                )
            } else {
                (
                    Box::new(bfgs::BFGS::new()) as Box<dyn Solver>,
                    format!("BFGS for general {}D problem - good balance of speed and robustness", dimension)
                )
            }
        }
    };
    
    let mut full_reason = reason;
    if has_bounds {
        full_reason.push_str(&format!(" (bounds: {})", bounds_description(problem)));
    }
    if has_fixed_params {
        full_reason.push_str(&format!(" (fixed parameters: {})", problem.fixed_params.len()));
    }
    
    (algorithm, full_reason)
}

/// Generate human-readable description of bounds
fn bounds_description(problem: &OptimizationProblem) -> String {
    if let Some((lower, upper)) = &problem.bounds {
        let n = lower.len();
        let mut desc = String::new();
        
        for i in 0..n.min(3) {  // Show first 3 bounds
            if i > 0 { desc.push_str(", "); }
            
            let l = lower[i];
            let u = upper[i];
            
            if l.is_finite() && u.is_finite() {
                desc.push_str(&format!("[{:.1}, {:.1}]", l, u));
            } else if l.is_finite() {
                desc.push_str(&format!(">= {:.1}", l));
            } else if u.is_finite() {
                desc.push_str(&format!("<= {:.1}", u));
            } else {
                desc.push_str("unbounded");
            }
        }
        
        if n > 3 {
            desc.push_str(&format!(", ... ({} total)", n));
        }
        
        desc
    } else {
        "none".to_string()
    }
}

/// Convergence testing utilities using math-first operations
pub struct ConvergenceTest {
    gradient_tolerance: f64,
    parameter_tolerance: f64,
    objective_tolerance: f64,
}

impl ConvergenceTest {
    pub fn new(gradient_tol: f64, parameter_tol: f64, objective_tol: f64) -> Self {
        Self {
            gradient_tolerance: gradient_tol,
            parameter_tolerance: parameter_tol,
            objective_tolerance: objective_tol,
        }
    }
    
    /// Test convergence using math-first operations
    pub fn check_convergence(
        &self,
        gradient: Option<&VectorF64>,
        parameter_change: Option<&VectorF64>,
        objective_change: Option<f64>,
    ) -> bool {
        // Gradient convergence: ||∇f|| < tol
        if let Some(grad) = gradient {
            let grad_norm = grad.dot(grad).sqrt();  // Math-first L2 norm
            if grad_norm < self.gradient_tolerance {
                return true;
            }
        }
        
        // Parameter convergence: ||Δx|| < tol
        if let Some(delta_x) = parameter_change {
            let delta_norm = delta_x.dot(delta_x).sqrt();  // Math-first L2 norm
            if delta_norm < self.parameter_tolerance {
                return true;
            }
        }
        
        // Objective convergence: |Δf| < tol
        if let Some(delta_f) = objective_change {
            if delta_f.abs() < self.objective_tolerance {
                return true;
            }
        }
        
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::vec64;
    
    #[test]
    fn test_optimization_problem_creation() {
        let objective = |x: &VectorF64| x[0].powi(2) + x[1].powi(2);
        let initial = vec64![1.0, 2.0];
        
        let problem = OptimizationProblem::new(objective, &initial, None);
        assert_eq!(problem.initial.len(), 2);
        assert_eq!(problem.effective_dimension(), 2);
    }
    
    #[test]
    fn test_parameter_fixing() {
        let objective = |x: &VectorF64| x[0].powi(2) + x[1].powi(2) + x[2].powi(2);
        let initial = vec64![1.0, 2.0, 3.0];
        
        let problem = OptimizationProblem::new(objective, &initial, None)
            .fix_parameters(&[(1, 2.5)]);  // Fix x[1] = 2.5
        
        assert_eq!(problem.effective_dimension(), 2);
        
        let reduced = vec64![1.0, 3.0];  // Only free parameters
        let expanded = problem.expand_parameters(&reduced);
        
        assert_eq!(expanded[0], 1.0);
        assert_eq!(expanded[1], 2.5);  // Fixed value
        assert_eq!(expanded[2], 3.0);
    }
    
    #[test]
    fn test_math_first_operations() {
        let residual = |x: &VectorF64| -> VectorF64 {
            vec64![x[0] - 1.0, x[1] - 2.0]  // Math-first vector creation
        };
        
        let initial = vec64![0.0, 0.0];
        let problem = OptimizationProblem::least_squares(residual, &initial, None);
        
        let test_point = vec64![1.0, 2.0];
        let objective_value = problem.evaluate(&test_point);
        assert!(objective_value < 1e-10);  // Should be near zero at optimum
    }
}