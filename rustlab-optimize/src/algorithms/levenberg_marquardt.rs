//! Levenberg-Marquardt algorithm implementation using math-first expressions
//!
//! This is the gold standard for nonlinear least squares optimization,
//! designed specifically for curve fitting and parameter estimation.

use rustlab_math::{VectorF64, ArrayF64};
use rustlab_linearalgebra::decompositions::lu_decompose;
use crate::core::{Result, OptimizationResult, Algorithm, ConvergenceStatus, Error};
use crate::core::result::{OptimizationInfo, AlgorithmData};
use super::{Solver, OptimizationProblem, ConvergenceTest};

/// Levenberg-Marquardt solver for nonlinear least squares
#[derive(Debug, Clone)]
pub struct LevenbergMarquardt {
    /// Maximum iterations
    max_iterations: usize,
    
    /// Gradient norm tolerance
    gradient_tolerance: f64,
    
    /// Parameter change tolerance  
    parameter_tolerance: f64,
    
    /// Objective change tolerance
    objective_tolerance: f64,
    
    /// Initial damping parameter (λ)
    lambda_init: f64,
    
    /// Factor to increase λ when step is rejected
    lambda_increase: f64,
    
    /// Factor to decrease λ when step is accepted
    lambda_decrease: f64,
    
    /// Maximum λ before giving up
    lambda_max: f64,
    
    /// Step size for numerical differentiation
    finite_diff_step: f64,
    
    /// Minimum improvement ratio to accept step
    min_improvement_ratio: f64,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self::new()
    }
}

impl LevenbergMarquardt {
    /// Create new Levenberg-Marquardt solver with default parameters
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            gradient_tolerance: 1e-8,
            parameter_tolerance: 1e-8,
            objective_tolerance: 1e-12,
            lambda_init: 1e-3,
            lambda_increase: 10.0,
            lambda_decrease: 0.7,
            lambda_max: 1e10,
            finite_diff_step: 1e-8,
            min_improvement_ratio: 0.25,
        }
    }
    
    /// Create conservative solver (more robust convergence)
    pub fn conservative() -> Self {
        Self {
            max_iterations: 200,
            gradient_tolerance: 1e-10,
            parameter_tolerance: 1e-10,
            objective_tolerance: 1e-15,
            lambda_init: 1e-1,
            lambda_increase: 5.0,
            lambda_decrease: 0.8,
            lambda_max: 1e12,
            finite_diff_step: 1e-10,
            min_improvement_ratio: 0.1,
        }
    }
    
    /// Create aggressive solver (faster convergence, less robust)
    pub fn aggressive() -> Self {
        Self {
            max_iterations: 50,
            gradient_tolerance: 1e-6,
            parameter_tolerance: 1e-6,
            objective_tolerance: 1e-8,
            lambda_init: 1e-5,
            lambda_increase: 20.0,
            lambda_decrease: 0.5,
            lambda_max: 1e8,
            finite_diff_step: 1e-6,
            min_improvement_ratio: 0.5,
        }
    }
    
    /// Compute numerical Jacobian using central differences with math-first operations
    fn numerical_jacobian(&self, problem: &OptimizationProblem, params: &VectorF64) -> Result<ArrayF64> {
        // Only works for residual-type problems
        let residuals = problem.evaluate_residual(params)
            .ok_or_else(|| Error::algorithm_error("LM requires residual function"))?;
        
        let m = residuals.len();  // Number of residuals
        let n = params.len();     // Number of parameters
        let h = self.finite_diff_step;
        
        let mut jacobian = ArrayF64::zeros(m, n);
        
        // Compute each column of Jacobian: ∂r/∂x_j
        for j in 0..n {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            
            // Math-first: direct parameter perturbation
            params_plus[j] += h;
            params_minus[j] -= h;
            
            let r_plus = problem.evaluate_residual(&params_plus)
                .ok_or_else(|| Error::numerical_error("Failed to evaluate residual"))?;
            let r_minus = problem.evaluate_residual(&params_minus)
                .ok_or_else(|| Error::numerical_error("Failed to evaluate residual"))?;
            
            // Central difference: ∂r_i/∂x_j ≈ (r_i(x + h·e_j) - r_i(x - h·e_j)) / (2h)
            let column_j = (&r_plus - &r_minus) * (1.0 / (2.0 * h));  // Math-first operations
            
            // Set column j of Jacobian
            for i in 0..m {
                jacobian.set(i, j, column_j[i]).map_err(|_| {
                    Error::numerical_error("Failed to set Jacobian element")
                })?;
            }
        }
        
        Ok(jacobian)
    }
    
    /// Solve the Levenberg-Marquardt step: (J^T J + λI) δ = -J^T r
    /// Using real rustlab-linearalgebra operations
    fn solve_lm_step(
        &self, 
        jacobian: &ArrayF64, 
        residuals: &VectorF64, 
        lambda: f64
    ) -> Result<VectorF64> {
        let n = jacobian.ncols();
        
        // Compute J^T J using math-first operations
        let jt = jacobian.transpose();
        let jtj = &jt ^ jacobian;  // Math-first matrix multiplication
        
        // Add damping: J^T J + λI
        let mut lhs = jtj;
        for i in 0..n {
            let current = lhs.get(i, i).ok_or_else(|| {
                Error::numerical_error("Failed to access diagonal element")
            })?;
            lhs.set(i, i, current + lambda).map_err(|_| {
                Error::numerical_error("Failed to set diagonal element")
            })?;
        }
        
        // Compute right-hand side: -J^T r
        let jt_r_array = &jt ^ residuals;  // Math-first matrix-vector multiplication (returns Array)
        let jt_r_vec = VectorF64::from_vec(jt_r_array.to_vec());  // Convert to Vector
        let rhs = &jt_r_vec * (-1.0);    // Math-first scalar multiplication (Vector result)
        
        // Convert to column vector for linear system solver
        let mut rhs_matrix = ArrayF64::zeros(n, 1);
        for i in 0..n {
            rhs_matrix.set(i, 0, rhs[i]).map_err(|_| {
                Error::numerical_error("Failed to set RHS element")
            })?;
        }
        
        // Solve linear system using LU decomposition: (J^T J + λI) δ = -J^T r
        let lu_decomp = lu_decompose(&lhs).map_err(|_| {
            Error::numerical_error("Failed to perform LU decomposition")
        })?;
        
        let solution_matrix = lu_decomp.solve(&rhs_matrix).map_err(|_| {
            Error::numerical_error("Failed to solve LM linear system")
        })?;
        
        // Convert back to vector
        let mut solution = VectorF64::zeros(n);
        for i in 0..n {
            solution[i] = solution_matrix.get(i, 0).ok_or_else(|| {
                Error::numerical_error("Failed to extract solution element")
            })?;
        }
        
        Ok(solution)
    }
    
    /// Compute improvement ratio for step acceptance
    fn compute_improvement_ratio(
        &self,
        problem: &OptimizationProblem,
        current_params: &VectorF64,
        new_params: &VectorF64,
        step: &VectorF64,
        jacobian: &ArrayF64,
        residuals: &VectorF64,
        lambda: f64,
    ) -> Result<f64> {
        let current_cost = residuals.dot(residuals);  // Math-first: ||r||²
        let new_cost = {
            let new_residuals = problem.evaluate_residual(new_params)
                .ok_or_else(|| Error::numerical_error("Failed to evaluate new residuals"))?;
            new_residuals.dot(&new_residuals)  // Math-first: ||r_new||²
        };
        
        // Actual reduction
        let actual_reduction = current_cost - new_cost;
        
        // Predicted reduction from linear model: Δ = δ^T (λδ - J^T r)
        let jt_r = &jacobian.transpose() ^ residuals;  // This is an Array
        let jt_r_vec = VectorF64::from_vec(jt_r.to_vec());  // Convert to Vector
        let lambda_step = step * lambda;  // Math-first scalar multiplication
        let predicted_change = &lambda_step - &jt_r_vec;
        let predicted_reduction = step.dot(&predicted_change);  // Math-first dot product
        
        if predicted_reduction <= 0.0 {
            return Ok(0.0);
        }
        
        Ok(actual_reduction / predicted_reduction)
    }
}

impl Solver for LevenbergMarquardt {
    fn solve(&self, problem: OptimizationProblem) -> Result<OptimizationResult> {
        // Step 1: Transform to reduced parameter space if needed (parameter fixing)
        let reduced_initial = problem.reduce_parameters(&problem.initial);
        let n = reduced_initial.len();
        
        if n == 0 {
            // All parameters are fixed
            let final_params = problem.expand_parameters(&reduced_initial);
            let objective_value = problem.evaluate(&final_params);
            return Ok(OptimizationResult {
                solution: final_params.to_vec(),
                objective_value,
                iterations: 0,
                function_evaluations: 1,
                algorithm_used: Algorithm::LevenbergMarquardt,
                convergence: ConvergenceStatus::Success,
                success: true,
                info: OptimizationInfo::default(),
            });
        }
        
        // Step 2: Handle bounds transformation
        let has_bounds = problem.has_bounds() && problem.fixed_params.is_empty();
        
        // Step 3: Transform initial parameters to unbounded space if bounds exist
        let mut current_params = if has_bounds {
            // Transform bounded initial parameters to unbounded space
            let full_initial = problem.expand_parameters(&reduced_initial);
            match problem.initial_unbounded() {
                Ok(unbounded) => problem.reduce_parameters(&unbounded),
                Err(_) => reduced_initial, // Fallback if transformation fails
            }
        } else {
            reduced_initial
        };
        
        let mut lambda = self.lambda_init;
        let mut iterations = 0;
        let mut function_evaluations = 0;
        
        let convergence_test = ConvergenceTest::new(
            self.gradient_tolerance,
            self.parameter_tolerance,
            self.objective_tolerance,
        );
        
        let mut best_cost = f64::INFINITY;
        let mut best_params = current_params.clone();
        
        for iter in 0..self.max_iterations {
            iterations = iter + 1;
            
            // Expand to full parameter space and transform if needed
            let full_params_unbounded = problem.expand_parameters(&current_params);
            let full_params = if has_bounds {
                // Transform from unbounded to bounded space for evaluation
                match problem.create_bounds_transformer() {
                    Some(transformer) => {
                        match transformer.to_bounded(&full_params_unbounded) {
                            Ok(bounded) => bounded,
                            Err(_) => full_params_unbounded, // Fallback
                        }
                    },
                    None => full_params_unbounded,
                }
            } else {
                full_params_unbounded
            };
            
            // Evaluate residuals at current point (in bounded space)
            let residuals = problem.evaluate_residual(&full_params)
                .ok_or_else(|| Error::algorithm_error("LM requires residual function"))?;
            function_evaluations += 1;
            
            let current_cost = residuals.dot(&residuals);  // Math-first: ||r||²
            
            // Track best solution
            if current_cost < best_cost {
                best_cost = current_cost;
                best_params = current_params.clone();
            }
            
            // Compute Jacobian (in unbounded parameter space)
            let jacobian = if problem.fixed_params.is_empty() && !has_bounds {
                // Simple case: no parameter fixing, no bounds
                self.numerical_jacobian(&problem, &full_params)?
            } else {
                // Complex case: parameter fixing and/or bounds
                let mut reduced_jacobian = ArrayF64::zeros(residuals.len(), current_params.len());
                let h = self.finite_diff_step;
                
                for j in 0..current_params.len() {
                    let mut params_plus = current_params.clone();
                    let mut params_minus = current_params.clone();
                    
                    // Perturb in unbounded space
                    params_plus[j] += h;
                    params_minus[j] -= h;
                    
                    // Expand and transform for evaluation
                    let full_plus_unbounded = problem.expand_parameters(&params_plus);
                    let full_minus_unbounded = problem.expand_parameters(&params_minus);
                    
                    let full_plus = if has_bounds {
                        match problem.create_bounds_transformer() {
                            Some(transformer) => {
                                transformer.to_bounded(&full_plus_unbounded).unwrap_or(full_plus_unbounded)
                            },
                            None => full_plus_unbounded,
                        }
                    } else {
                        full_plus_unbounded
                    };
                    
                    let full_minus = if has_bounds {
                        match problem.create_bounds_transformer() {
                            Some(transformer) => {
                                transformer.to_bounded(&full_minus_unbounded).unwrap_or(full_minus_unbounded)
                            },
                            None => full_minus_unbounded,
                        }
                    } else {
                        full_minus_unbounded
                    };
                    
                    let r_plus = problem.evaluate_residual(&full_plus)
                        .ok_or_else(|| Error::numerical_error("Failed to evaluate residual"))?;
                    let r_minus = problem.evaluate_residual(&full_minus)
                        .ok_or_else(|| Error::numerical_error("Failed to evaluate residual"))?;
                    
                    let column_j = (&r_plus - &r_minus) * (1.0 / (2.0 * h));
                    
                    for i in 0..residuals.len() {
                        reduced_jacobian.set(i, j, column_j[i]).map_err(|_| {
                            Error::numerical_error("Failed to set reduced Jacobian")
                        })?;
                    }
                }
                
                reduced_jacobian
            };
            
            // Compute gradient: g = J^T r (math-first)
            let gradient_array = &jacobian.transpose() ^ &residuals;
            let gradient = VectorF64::from_vec(gradient_array.to_vec());
            
            // Check convergence
            if convergence_test.check_convergence(Some(&gradient), None, None) {
                let final_params_unbounded = problem.expand_parameters(&best_params);
                let final_params = if has_bounds {
                    match problem.create_bounds_transformer() {
                        Some(transformer) => {
                            match transformer.to_bounded(&final_params_unbounded) {
                                Ok(bounded) => bounded,
                                Err(_) => final_params_unbounded, // Fallback
                            }
                        },
                        None => final_params_unbounded,
                    }
                } else {
                    final_params_unbounded
                };
                return Ok(OptimizationResult {
                    solution: final_params.to_vec(),
                    objective_value: best_cost,
                    iterations,
                    function_evaluations,
                    algorithm_used: Algorithm::LevenbergMarquardt,
                    convergence: ConvergenceStatus::Success,
                    success: true,
                    info: OptimizationInfo {
                        gradient_norm: Some(gradient.norm()),
                        step_size: None,
                        condition_number: None,
                        algorithm_data: AlgorithmData::LevenbergMarquardt {
                            lambda,
                            residual_norm: residuals.norm(),
                        },
                    },
                });
            }
            
            // Try Levenberg-Marquardt step with current λ
            let mut step_accepted = false;
            let mut trial_lambda = lambda;
            
            while trial_lambda <= self.lambda_max && !step_accepted {
                // Solve LM step: (J^T J + λI) δ = -J^T r
                if let Ok(step) = self.solve_lm_step(&jacobian, &residuals, trial_lambda) {
                    let new_params = &current_params + &step;  // Math-first vector addition
                    let full_new_params = problem.expand_parameters(&new_params);
                    
                    // Compute improvement ratio
                    if let Ok(rho) = self.compute_improvement_ratio(
                        &problem, &full_params, &full_new_params, &step, 
                        &jacobian, &residuals, trial_lambda
                    ) {
                        function_evaluations += 1;
                        
                        if rho > self.min_improvement_ratio {
                            // Accept step
                            current_params = new_params;
                            lambda = (trial_lambda * self.lambda_decrease).max(1e-16);
                            step_accepted = true;
                            
                            // Check parameter change convergence
                            if convergence_test.check_convergence(None, Some(&step), None) {
                                let final_params_unbounded = problem.expand_parameters(&current_params);
                                let final_params = if has_bounds {
                                    match problem.create_bounds_transformer() {
                                        Some(transformer) => {
                                            match transformer.to_bounded(&final_params_unbounded) {
                                                Ok(bounded) => bounded,
                                                Err(_) => final_params_unbounded, // Fallback
                                            }
                                        },
                                        None => final_params_unbounded,
                                    }
                                } else {
                                    final_params_unbounded
                                };
                                return Ok(OptimizationResult {
                                    solution: final_params.to_vec(),
                                    objective_value: current_cost,
                                    iterations,
                                    function_evaluations,
                                    algorithm_used: Algorithm::LevenbergMarquardt,
                                    convergence: ConvergenceStatus::Success,
                                    success: true,
                                    info: OptimizationInfo {
                                        gradient_norm: Some(gradient.norm()),
                                        step_size: Some(step.norm()),
                                        condition_number: None,
                                        algorithm_data: AlgorithmData::LevenbergMarquardt {
                                            lambda: trial_lambda,
                                            residual_norm: residuals.norm(),
                                        },
                                    },
                                });
                            }
                        } else {
                            // Reject step, increase damping
                            trial_lambda *= self.lambda_increase;
                        }
                    } else {
                        trial_lambda *= self.lambda_increase;
                    }
                } else {
                    trial_lambda *= self.lambda_increase;
                }
            }
            
            if !step_accepted {
                lambda = trial_lambda;
                if lambda > self.lambda_max {
                    break;
                }
            }
        }
        
        // Return best solution found (transform back to bounded space if needed)
        let final_params_unbounded = problem.expand_parameters(&best_params);
        let final_params = if has_bounds {
            match problem.create_bounds_transformer() {
                Some(transformer) => {
                    match transformer.to_bounded(&final_params_unbounded) {
                        Ok(bounded) => bounded,
                        Err(_) => final_params_unbounded, // Fallback
                    }
                },
                None => final_params_unbounded,
            }
        } else {
            final_params_unbounded
        };
        Ok(OptimizationResult {
            solution: final_params.to_vec(),
            objective_value: best_cost,
            iterations,
            function_evaluations,
            algorithm_used: Algorithm::LevenbergMarquardt,
            convergence: ConvergenceStatus::MaxIterations,
            success: best_cost < 1e-6,  // Heuristic success criterion
            info: OptimizationInfo {
                gradient_norm: None,
                step_size: None,
                condition_number: None,
                algorithm_data: AlgorithmData::LevenbergMarquardt {
                    lambda,
                    residual_norm: best_cost.sqrt(),
                },
            },
        })
    }
    
    fn algorithm(&self) -> Algorithm {
        Algorithm::LevenbergMarquardt
    }
    
    fn supports_bounds(&self) -> bool {
        true  // Now supports bounds via parameter transformation
    }
    
    fn supports_parameter_fixing(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::vec64;
    
    #[test]
    fn test_simple_least_squares() {
        // Solve simple linear least squares: min ||Ax - b||²
        let residual = |x: &VectorF64| -> VectorF64 {
            // System: [1, 1; 2, 1] * [x0; x1] = [3; 5]
            // Solution should be x0 = 2, x1 = 1
            vec64![
                x[0] + x[1] - 3.0,      // 1*x0 + 1*x1 = 3
                2.0*x[0] + x[1] - 5.0   // 2*x0 + 1*x1 = 5
            ]
        };
        
        let initial = vec64![0.0, 0.0];
        let problem = OptimizationProblem::least_squares(residual, &initial, None);
        
        let solver = LevenbergMarquardt::new();
        let result = solver.solve(problem).unwrap();
        
        assert!(result.success);
        assert!((result.solution[0] - 2.0).abs() < 1e-6);
        assert!((result.solution[1] - 1.0).abs() < 1e-6);
        assert!(result.objective_value < 1e-10);
    }
    
    #[test]
    fn test_parameter_fixing() {
        // Test with one parameter fixed
        let residual = |x: &VectorF64| -> VectorF64 {
            vec64![x[0] + x[1] - 3.0, 2.0*x[0] + x[1] - 5.0]
        };
        
        let initial = vec64![0.0, 1.0];  // x[1] will be fixed at 1.0
        let problem = OptimizationProblem::least_squares(residual, &initial, None)
            .fix_parameters(&[(1, 1.0)]);  // Fix x[1] = 1.0
        
        let solver = LevenbergMarquardt::new();
        let result = solver.solve(problem).unwrap();
        
        assert!(result.success);
        assert!((result.solution[0] - 2.0).abs() < 1e-10);  // Should find x[0] = 2
        assert!((result.solution[1] - 1.0).abs() < 1e-15);  // Fixed at 1.0
    }
}