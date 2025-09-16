//! Optimization result types

/// Result of an optimization operation
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationResult {
    /// Final parameter values (the solution)
    pub solution: Vec<f64>,
    
    /// Final objective function value
    pub objective_value: f64,
    
    /// Number of iterations performed
    pub iterations: usize,
    
    /// Number of function evaluations
    pub function_evaluations: usize,
    
    /// Algorithm that was used
    pub algorithm_used: Algorithm,
    
    /// Convergence status
    pub convergence: ConvergenceStatus,
    
    /// Whether optimization was successful
    pub success: bool,
    
    /// Additional algorithm-specific information
    pub info: OptimizationInfo,
}

/// Available optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Levenberg-Marquardt (nonlinear least squares)
    LevenbergMarquardt,
    
    /// Gradient Descent with momentum
    GradientDescent,
    
    /// BFGS quasi-Newton method
    BFGS,
    
    /// Nelder-Mead simplex method (derivative-free)
    NelderMead,
    
    /// Differential Evolution (global optimization)
    DifferentialEvolution,
    
    /// Golden section search (1D only)
    GoldenSection,
    
    /// Brent's method (1D only)
    Brent,
}

impl Algorithm {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Algorithm::LevenbergMarquardt => "Levenberg-Marquardt",
            Algorithm::GradientDescent => "Gradient Descent",
            Algorithm::BFGS => "BFGS",
            Algorithm::NelderMead => "Nelder-Mead",
            Algorithm::DifferentialEvolution => "Differential Evolution",
            Algorithm::GoldenSection => "Golden Section Search",
            Algorithm::Brent => "Brent's Method",
        }
    }

    /// Check if algorithm is derivative-free
    pub fn is_derivative_free(&self) -> bool {
        matches!(
            self,
            Algorithm::NelderMead | Algorithm::DifferentialEvolution | Algorithm::GoldenSection | Algorithm::Brent
        )
    }

    /// Check if algorithm is designed for global optimization
    pub fn is_global(&self) -> bool {
        matches!(self, Algorithm::DifferentialEvolution)
    }

    /// Create solver instance for this algorithm
    pub fn create_solver(&self) -> Box<dyn crate::algorithms::Solver> {
        match self {
            Algorithm::LevenbergMarquardt => {
                Box::new(crate::algorithms::levenberg_marquardt::LevenbergMarquardt::new())
            }
            Algorithm::GradientDescent => {
                Box::new(crate::algorithms::gradient_descent::GradientDescent::new())
            }
            Algorithm::BFGS => {
                Box::new(crate::algorithms::bfgs::BFGS::new())
            }
            Algorithm::NelderMead => {
                Box::new(crate::algorithms::nelder_mead::NelderMead::new())
            }
            Algorithm::DifferentialEvolution => {
                // TODO: Implement differential evolution algorithm
                panic!("DifferentialEvolution not yet implemented")
            }
            Algorithm::GoldenSection => {
                // TODO: Implement golden section search
                panic!("GoldenSection not yet implemented")
            }
            Algorithm::Brent => {
                // TODO: Implement Brent's method
                panic!("Brent not yet implemented")
            }
        }
    }
}

/// Convergence status indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Converged successfully
    Success,
    
    /// Maximum iterations reached
    MaxIterations,
    
    /// Maximum function evaluations reached
    MaxFunctionEvaluations,
    
    /// Stalled (no significant progress)
    Stalled,
    
    /// Numerical issues encountered
    NumericalIssues,
    
    /// User-requested termination
    UserTermination,
    
    /// Algorithm-specific failure
    AlgorithmFailure,
}

impl ConvergenceStatus {
    /// Check if convergence was successful
    pub fn is_success(&self) -> bool {
        matches!(self, ConvergenceStatus::Success)
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            ConvergenceStatus::Success => "Successfully converged",
            ConvergenceStatus::MaxIterations => "Maximum iterations reached",
            ConvergenceStatus::MaxFunctionEvaluations => "Maximum function evaluations reached",
            ConvergenceStatus::Stalled => "Optimization stalled",
            ConvergenceStatus::NumericalIssues => "Numerical issues encountered",
            ConvergenceStatus::UserTermination => "User requested termination",
            ConvergenceStatus::AlgorithmFailure => "Algorithm-specific failure",
        }
    }
}

/// Additional optimization information
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationInfo {
    /// Final gradient norm (if available)
    pub gradient_norm: Option<f64>,
    
    /// Final step size (if available)
    pub step_size: Option<f64>,
    
    /// Condition number estimate (if available)
    pub condition_number: Option<f64>,
    
    /// Algorithm-specific data
    pub algorithm_data: AlgorithmData,
}

/// Algorithm-specific optimization data
#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmData {
    /// No specific data
    None,
    
    /// Levenberg-Marquardt specific data
    LevenbergMarquardt {
        /// Final damping parameter (lambda)
        lambda: f64,
        /// Final residual norm
        residual_norm: f64,
    },
    
    /// Gradient descent specific data
    GradientDescent {
        /// Final learning rate
        learning_rate: f64,
        /// Momentum coefficient used
        momentum: f64,
    },
    
    /// Differential evolution specific data  
    DifferentialEvolution {
        /// Population size used
        population_size: usize,
        /// Final diversity measure
        diversity: f64,
    },
    
    /// Generic algorithm data
    Generic {
        /// Key-value pairs of algorithm-specific information
        data: Vec<(String, f64)>,
    },
}

impl Default for OptimizationInfo {
    fn default() -> Self {
        Self {
            gradient_norm: None,
            step_size: None,
            condition_number: None,
            algorithm_data: AlgorithmData::None,
        }
    }
}