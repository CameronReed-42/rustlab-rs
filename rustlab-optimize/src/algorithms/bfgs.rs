//! BFGS quasi-Newton algorithm with math-first expressions
//!
//! BFGS (Broyden-Fletcher-Goldfarb-Shanno) is excellent for smooth, 
//! unconstrained optimization problems.

use rustlab_math::{VectorF64, ArrayF64};
use crate::core::{Result, OptimizationResult, Algorithm, ConvergenceStatus, Error};
use crate::core::result::{OptimizationInfo, AlgorithmData};
use super::{Solver, OptimizationProblem, ConvergenceTest};

/// BFGS quasi-Newton solver
#[derive(Debug, Clone)]
pub struct BFGS {
    max_iterations: usize,
    gradient_tolerance: f64,
    parameter_tolerance: f64,
    objective_tolerance: f64,
    finite_diff_step: f64,
    line_search_max_iterations: usize,
    line_search_tolerance: f64,
    c1: f64,  // Armijo condition parameter
    c2: f64,  // Curvature condition parameter
}

impl Default for BFGS {
    fn default() -> Self {
        Self::new()
    }
}

impl BFGS {
    /// Create new BFGS solver with default parameters
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            gradient_tolerance: 1e-6,
            parameter_tolerance: 1e-8,
            objective_tolerance: 1e-8,
            finite_diff_step: 1e-8,
            line_search_max_iterations: 20,
            line_search_tolerance: 1e-4,
            c1: 1e-4,   // Armijo condition
            c2: 0.9,    // Strong Wolfe condition
        }
    }
    
    /// Compute numerical gradient using central differences
    fn numerical_gradient(&self, problem: &OptimizationProblem, params: &VectorF64) -> VectorF64 {
        let n = params.len();
        let h = self.finite_diff_step;
        let mut gradient = VectorF64::zeros(n);
        
        for i in 0..n {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            
            // Math-first parameter perturbation
            params_plus[i] += h;
            params_minus[i] -= h;
            
            let f_plus = problem.evaluate(&params_plus);
            let f_minus = problem.evaluate(&params_minus);
            
            // Central difference
            gradient[i] = (f_plus - f_minus) / (2.0 * h);
        }
        
        gradient
    }
    
    /// Simple backtracking line search with Armijo condition
    fn line_search(
        &self,
        problem: &OptimizationProblem,
        current_params: &VectorF64,
        direction: &VectorF64,
        current_f: f64,
        current_gradient: &VectorF64,
    ) -> f64 {
        let mut alpha = 1.0;
        let directional_derivative = current_gradient.dot(direction);  // Math-first dot product
        
        // Armijo condition: f(x + α*p) ≤ f(x) + c1*α*∇f^T*p
        for _ in 0..self.line_search_max_iterations {
            let new_params = current_params + &(direction * alpha);  // Math-first operations
            let new_f = problem.evaluate(&new_params);
            
            let armijo_condition = current_f + self.c1 * alpha * directional_derivative;
            
            if new_f <= armijo_condition {
                break;
            }
            
            alpha *= 0.5;  // Backtrack
        }
        
        alpha.max(1e-10)  // Prevent too small steps
    }
    
    /// Bounds-aware line search using backtracking with Armijo condition
    fn line_search_bounds_aware<F>(
        &self,
        current_params: &VectorF64,
        direction: &VectorF64,
        current_f: f64,
        current_gradient: &VectorF64,
        evaluate_objective: &F,
    ) -> f64 
    where
        F: Fn(&VectorF64) -> f64,
    {
        let mut alpha = 1.0;
        let directional_derivative = current_gradient.dot(direction);  // Math-first dot product
        
        // Armijo condition: f(x + α*p) ≤ f(x) + c1*α*∇f^T*p
        for _ in 0..self.line_search_max_iterations {
            let new_params = current_params + &(direction * alpha);  // Math-first operations
            let new_f = evaluate_objective(&new_params);
            
            let armijo_condition = current_f + self.c1 * alpha * directional_derivative;
            
            if new_f <= armijo_condition {
                break;
            }
            
            alpha *= 0.5;  // Backtrack
        }
        
        alpha.max(1e-10)  // Prevent too small steps
    }
    
    /// Update BFGS Hessian approximation using Sherman-Morrison-Woodbury formula
    fn update_hessian_inverse(
        &self,
        hessian_inv: ArrayF64,
        step: &VectorF64,
        gradient_change: &VectorF64,
    ) -> Result<ArrayF64> {
        let n = step.len();
        let mut h_inv = hessian_inv;
        
        // Check curvature condition: s^T y > 0
        let sy = step.dot(gradient_change);  // Math-first dot product
        if sy <= 1e-12 {
            // Skip update if curvature condition is not satisfied
            return Ok(h_inv);
        }
        
        // BFGS update: H_{k+1}^{-1} = H_k^{-1} + (s^T y + y^T H_k^{-1} y)(ss^T)/(s^T y)^2 
        //                              - (H_k^{-1} y s^T + s y^T H_k^{-1})/(s^T y)
        
        // Compute H_k^{-1} * y (matrix × vector → vector)
        let hy = &h_inv ^ gradient_change;  // Math-first matrix-vector product
        let yhy = gradient_change.dot(&hy);  // y^T H_k^{-1} y
        
        // First term: (s^T y + y^T H_k^{-1} y)(ss^T)/(s^T y)^2
        let factor1 = (sy + yhy) / (sy * sy);
        for i in 0..n {
            for j in 0..n {
                let current = h_inv.get(i, j).ok_or_else(|| {
                    Error::numerical_error("Failed to access Hessian element")
                })?;
                let update = current + factor1 * step[i] * step[j];
                h_inv.set(i, j, update).map_err(|_| {
                    Error::numerical_error("Failed to update Hessian")
                })?;
            }
        }
        
        // Second term: -(H_k^{-1} y s^T + s y^T H_k^{-1})/(s^T y)
        let factor2 = 1.0 / sy;
        for i in 0..n {
            for j in 0..n {
                let current = h_inv.get(i, j).ok_or_else(|| {
                    Error::numerical_error("Failed to access Hessian element")
                })?;
                let update = current - factor2 * (hy[i] * step[j] + step[i] * hy[j]);
                h_inv.set(i, j, update).map_err(|_| {
                    Error::numerical_error("Failed to update Hessian")
                })?;
            }
        }
        
        Ok(h_inv)
    }
}

impl Solver for BFGS {
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
                algorithm_used: Algorithm::BFGS,
                convergence: ConvergenceStatus::Success,
                success: true,
                info: OptimizationInfo::default(),
            });
        }
        
        // Step 2: Handle bounds transformation
        // Check if we need bounds transformation for the reduced parameter space
        let bounds_needed = if problem.fixed_params.is_empty() {
            problem.has_bounds()
        } else {
            // Need to check if reduced bounds exist - for now, disable bounds with parameter fixing
            false  // TODO: Implement bounds + parameter fixing combination
        };
        
        // Step 3: Transform initial parameters to unbounded space if bounds exist
        let mut current_params = if bounds_needed {
            // Transform bounded initial parameters to unbounded space
            let full_initial = problem.expand_parameters(&reduced_initial);
            match problem.initial_unbounded() {
                Ok(unbounded) => problem.reduce_parameters(&unbounded),
                Err(_) => reduced_initial, // Fallback if transformation fails
            }
        } else {
            reduced_initial
        };
        
        // Initialize inverse Hessian as identity matrix (math-first)
        let mut hessian_inv = ArrayF64::eye(n);
        
        let mut iterations = 0;
        let mut function_evaluations = 0;
        
        let convergence_test = ConvergenceTest::new(
            self.gradient_tolerance,
            self.parameter_tolerance,
            self.objective_tolerance,
        );
        
        // Step 4: Create bounds-aware objective function
        let evaluate_objective = |params: &VectorF64| -> f64 {
            let full_params = problem.expand_parameters(params);
            if bounds_needed {
                problem.evaluate_with_bounds(&full_params)
            } else {
                problem.evaluate(&full_params)
            }
        };
        
        // Step 5: Create bounds-aware gradient function
        let compute_gradient = |params: &VectorF64| -> VectorF64 {
            let full_params = problem.expand_parameters(params);
            let h = self.finite_diff_step;
            
            if problem.fixed_params.is_empty() {
                // No parameter fixing - compute gradient directly
                let mut gradient = VectorF64::zeros(n);
                
                for i in 0..n {
                    let mut params_plus = params.clone();
                    let mut params_minus = params.clone();
                    
                    params_plus[i] += h;
                    params_minus[i] -= h;
                    
                    let f_plus = evaluate_objective(&params_plus);
                    let f_minus = evaluate_objective(&params_minus);
                    
                    gradient[i] = (f_plus - f_minus) / (2.0 * h);
                }
                
                // Transform gradient if bounds are present
                if bounds_needed {
                    match problem.transform_gradient_to_unbounded(&gradient, &full_params) {
                        Ok(transformed) => problem.reduce_parameters(&transformed),
                        Err(_) => gradient, // Fallback
                    }
                } else {
                    gradient
                }
                
            } else {
                // With parameter fixing - compute reduced gradient
                let mut gradient = VectorF64::zeros(n);
                
                for i in 0..n {
                    let mut params_plus = params.clone();
                    let mut params_minus = params.clone();
                    
                    params_plus[i] += h;
                    params_minus[i] -= h;
                    
                    let f_plus = evaluate_objective(&params_plus);
                    let f_minus = evaluate_objective(&params_minus);
                    
                    gradient[i] = (f_plus - f_minus) / (2.0 * h);
                }
                
                gradient
            }
        };
        
        // Evaluate initial point
        let mut current_f = evaluate_objective(&current_params);
        let mut current_gradient = compute_gradient(&current_params);
        
        function_evaluations += 1;
        
        for iter in 0..self.max_iterations {
            iterations = iter + 1;
            
            // Check convergence  
            if convergence_test.check_convergence(Some(&current_gradient), None, None) {
                // Transform final parameters back to bounded space if needed
                let final_unbounded = problem.expand_parameters(&current_params);
                let final_params = if bounds_needed {
                    // Transform from unbounded back to bounded space
                    match problem.create_bounds_transformer() {
                        Some(transformer) => {
                            match transformer.to_bounded(&final_unbounded) {
                                Ok(bounded) => bounded,
                                Err(_) => final_unbounded, // Fallback
                            }
                        },
                        None => final_unbounded,
                    }
                } else {
                    final_unbounded
                };
                
                return Ok(OptimizationResult {
                    solution: final_params.to_vec(),
                    objective_value: current_f,
                    iterations,
                    function_evaluations,
                    algorithm_used: Algorithm::BFGS,
                    convergence: ConvergenceStatus::Success,
                    success: true,
                    info: OptimizationInfo {
                        gradient_norm: Some(current_gradient.norm()),
                        step_size: None,
                        condition_number: None,
                        algorithm_data: AlgorithmData::None,
                    },
                });
            }
            
            // Compute search direction: p = -H^{-1} * ∇f (matrix × vector → vector)
            let mut direction = &hessian_inv ^ &current_gradient;  // Math-first matrix-vector multiplication
            direction *= -1.0;  // Negate for descent direction
            
            // Bounds-aware line search - use existing line search with bounds-aware evaluation
            let alpha = self.line_search_bounds_aware(&current_params, &direction, current_f, &current_gradient, &evaluate_objective);
            
            // Take step
            let mut step = direction.clone();
            step *= alpha;  // Math-first scalar multiplication: step = direction * alpha
            let mut new_params = current_params.clone();
            new_params += &step;  // Math-first in-place vector addition
            
            // Evaluate new point using bounds-aware function
            let new_f = evaluate_objective(&new_params);
            function_evaluations += 1;
            
            // Compute new gradient using bounds-aware function
            let new_gradient = compute_gradient(&new_params);
            function_evaluations += 2 * n; // Gradient evaluation cost
            
            // Check for convergence based on parameter change
            if convergence_test.check_convergence(None, Some(&step), None) {
                let final_params = problem.expand_parameters(&new_params);
                return Ok(OptimizationResult {
                    solution: final_params.to_vec(),
                    objective_value: new_f,
                    iterations,
                    function_evaluations,
                    algorithm_used: Algorithm::BFGS,
                    convergence: ConvergenceStatus::Success,
                    success: true,
                    info: OptimizationInfo {
                        gradient_norm: Some(new_gradient.norm()),
                        step_size: Some(step.norm()),
                        condition_number: None,
                        algorithm_data: AlgorithmData::None,
                    },
                });
            }
            
            // Update BFGS Hessian approximation
            let gradient_change = &new_gradient - &current_gradient;  // Math-first vector subtraction
            hessian_inv = self.update_hessian_inverse(hessian_inv, &step, &gradient_change)?;
            
            // Update for next iteration
            current_params = new_params;
            current_f = new_f;
            current_gradient = new_gradient;
        }
        
        // Max iterations reached
        let final_params = problem.expand_parameters(&current_params);
        // Transform final parameters back to bounded space if needed
        let final_unbounded = problem.expand_parameters(&current_params);
        let final_params = if bounds_needed {
            // Transform from unbounded back to bounded space
            match problem.create_bounds_transformer() {
                Some(transformer) => {
                    match transformer.to_bounded(&final_unbounded) {
                        Ok(bounded) => bounded,
                        Err(_) => final_unbounded, // Fallback
                    }
                },
                None => final_unbounded,
            }
        } else {
            final_unbounded
        };
        
        Ok(OptimizationResult {
            solution: final_params.to_vec(),
            objective_value: current_f,
            iterations,
            function_evaluations,
            algorithm_used: Algorithm::BFGS,
            convergence: ConvergenceStatus::MaxIterations,
            success: current_gradient.norm() < 1e-3,  // Relaxed success criterion
            info: OptimizationInfo {
                gradient_norm: Some(current_gradient.norm()),
                step_size: None,
                condition_number: None,
                algorithm_data: AlgorithmData::None,
            },
        })
    }
    
    fn algorithm(&self) -> Algorithm {
        Algorithm::BFGS
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
    fn test_quadratic_minimization() {
        // Minimize f(x,y) = (x-1)² + (y-2)²
        // Solution should be x=1, y=2
        let objective = |params: &VectorF64| -> f64 {
            (params[0] - 1.0).powi(2) + (params[1] - 2.0).powi(2)
        };
        
        let initial = vec64![0.0, 0.0];
        let problem = OptimizationProblem::new(objective, &initial, None);
        
        let solver = BFGS::new();
        let result = solver.solve(problem).unwrap();
        
        assert!(result.success);
        assert!((result.solution[0] - 1.0).abs() < 1e-6);
        assert!((result.solution[1] - 2.0).abs() < 1e-6);
        assert!(result.objective_value < 1e-10);
    }
    
    #[test]
    fn test_bfgs_with_parameter_fixing() {
        // Minimize f(x,y,z) = x² + y² + z² with y fixed at 3.0
        let objective = |params: &VectorF64| -> f64 {
            params[0].powi(2) + params[1].powi(2) + params[2].powi(2)
        };
        
        let initial = vec64![1.0, 3.0, 2.0];
        let problem = OptimizationProblem::new(objective, &initial, None)
            .fix_parameters(&[(1, 3.0)]);  // Fix y = 3.0
        
        let solver = BFGS::new();
        let result = solver.solve(problem).unwrap();
        
        assert!(result.success);
        assert!(result.solution[0].abs() < 1e-6);  // x → 0
        assert!((result.solution[1] - 3.0).abs() < 1e-15);  // y fixed at 3.0
        assert!(result.solution[2].abs() < 1e-6);  // z → 0
    }
}