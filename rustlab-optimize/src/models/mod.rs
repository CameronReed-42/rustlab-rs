//! Curve fitting model types and results

/// Linear regression fit result containing fitted parameters and statistics
/// 
/// # Mathematical Specification
/// Represents fitted linear model: y = a + bx where:
/// - a = intercept (y-value when x = 0)
/// - b = slope (rate of change dy/dx)
/// - R² = coefficient of determination (fraction of variance explained)
/// 
/// # For AI Code Generation
/// - Access fitted parameters via .intercept and .slope fields
/// - Use .predict(x) method for making predictions
/// - Check .r_squared for goodness of fit (0 = poor, 1 = perfect)
/// - Use .slope_is_significant() to test if slope differs from zero
/// - Standard errors available for uncertainty quantification
#[derive(Debug, Clone, PartialEq)]
pub struct LinearFit {
    /// Intercept (a in y = a + b*x)
    pub intercept: f64,
    
    /// Slope (b in y = a + b*x)  
    pub slope: f64,
    
    /// Coefficient of determination (R²)
    pub r_squared: f64,
    
    /// Standard error of the slope
    pub std_error_slope: f64,
    
    /// Standard error of the intercept
    pub std_error_intercept: f64,
    
    /// 95% confidence interval for slope
    pub slope_confidence_interval: (f64, f64),
    
    /// 95% confidence interval for intercept
    pub intercept_confidence_interval: (f64, f64),
}

impl LinearFit {
    /// Predict y value for given x
    pub fn predict(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }
    
    /// Check if slope is significantly different from zero at α = 0.05
    pub fn slope_is_significant(&self) -> bool {
        let t_stat = self.slope / self.std_error_slope;
        t_stat.abs() > 1.96 // Approximate for large samples
    }
}

/// Exponential decay fit result with physical interpretation methods
/// 
/// # Mathematical Specification
/// Represents fitted exponential model: y = A·exp(-kx) where:
/// - A = amplitude (initial value at x = 0)
/// - k = decay rate (larger k = faster decay)
/// - half_life = ln(2)/k (time for y to reach 50% of initial value)
/// 
/// # For AI Code Generation
/// - Access parameters via .amplitude, .decay_rate fields
/// - Use .predict(x) for predictions and .half_life for interpretation
/// - Physical methods: .time_constant(), .time_to_fraction(f)
/// - Check .r_squared for fit quality
/// - Common uses: radioactive decay, pharmacokinetics, cooling curves
#[derive(Debug, Clone, PartialEq)]
pub struct ExponentialFit {
    /// Amplitude (A in y = A*exp(-k*x))
    pub amplitude: f64,
    
    /// Decay rate (k in y = A*exp(-k*x))
    pub decay_rate: f64,
    
    /// Half-life (ln(2)/k)
    pub half_life: f64,
    
    /// Coefficient of determination (R²)
    pub r_squared: f64,
}

impl ExponentialFit {
    /// Predict y value for given x
    pub fn predict(&self, x: f64) -> f64 {
        self.amplitude * (-self.decay_rate * x).exp()
    }
    
    /// Time constant (1/k)
    pub fn time_constant(&self) -> f64 {
        1.0 / self.decay_rate
    }
    
    /// Time for decay to 1/e of initial value
    pub fn time_to_1_over_e(&self) -> f64 {
        1.0 / self.decay_rate
    }
    
    /// Time for decay to specified fraction (0 < fraction < 1)
    pub fn time_to_fraction(&self, fraction: f64) -> f64 {
        -fraction.ln() / self.decay_rate
    }
}

/// Polynomial fit result with coefficients and analytical methods
/// 
/// # Mathematical Specification
/// Represents fitted polynomial: y = Σᵢ aᵢxᵢ where:
/// - coefficients[0] = a₀ (constant term)
/// - coefficients[1] = a₁ (linear coefficient)
/// - coefficients[i] = aᵢ (coefficient of xᵢ)
/// 
/// # For AI Code Generation
/// - Access coefficients via .coefficients[i] (i-th power of x)
/// - Use .predict(x) for evaluation and .derivative(x) for slopes
/// - Use .roots() for finding zeros (available for degree ≤ 2)
/// - Check .r_squared and .r_squared_adjusted for model quality
/// - Higher degree → better fit but risk of overfitting
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialFit {
    /// Polynomial coefficients [a₀, a₁, a₂, ..., aₙ]
    pub coefficients: Vec<f64>,
    
    /// Degree of the polynomial
    pub degree: usize,
    
    /// Coefficient of determination (R²)
    pub r_squared: f64,
    
    /// Adjusted R² (accounts for number of parameters)
    pub r_squared_adjusted: f64,
}

impl PolynomialFit {
    /// Predict y value for given x
    pub fn predict(&self, x: f64) -> f64 {
        self.coefficients.iter()
            .enumerate()
            .map(|(i, &coef)| coef * x.powi(i as i32))
            .sum()
    }
    
    /// Evaluate derivative at given x
    pub fn derivative(&self, x: f64) -> f64 {
        self.coefficients.iter()
            .enumerate()
            .skip(1) // Skip constant term
            .map(|(i, &coef)| coef * (i as f64) * x.powi(i as i32 - 1))
            .sum()
    }
    
    /// Find roots of the polynomial (for degree ≤ 2)
    pub fn roots(&self) -> Option<Vec<f64>> {
        match self.degree {
            1 => {
                // Linear: ax + b = 0 → x = -b/a
                if self.coefficients.len() >= 2 && self.coefficients[1] != 0.0 {
                    Some(vec![-self.coefficients[0] / self.coefficients[1]])
                } else {
                    None
                }
            },
            2 => {
                // Quadratic formula
                if self.coefficients.len() >= 3 {
                    let a = self.coefficients[2];
                    let b = self.coefficients[1];
                    let c = self.coefficients[0];
                    let discriminant = b*b - 4.0*a*c;
                    
                    if discriminant >= 0.0 {
                        let sqrt_disc = discriminant.sqrt();
                        Some(vec![
                            (-b + sqrt_disc) / (2.0 * a),
                            (-b - sqrt_disc) / (2.0 * a),
                        ])
                    } else {
                        None // Complex roots
                    }
                } else {
                    None
                }
            },
            _ => None, // Higher-order polynomials require numerical methods
        }
    }
}

/// Sinusoidal fit result with frequency domain interpretation
/// 
/// # Mathematical Specification
/// Represents fitted sinusoid: y = A·sin(ωx + φ) + C where:
/// - A = amplitude (peak deviation from offset)
/// - ω = angular frequency (radians per unit x)
/// - φ = phase shift (horizontal shift in radians)
/// - C = vertical offset (DC component)
/// 
/// # For AI Code Generation
/// - Access parameters via .amplitude, .frequency, .phase, .offset
/// - Use .predict(x) for evaluation
/// - Frequency methods: .period(), .frequency_hz() for time-domain interpretation
/// - Signal methods: .peak_to_peak(), .rms() for amplitude analysis
/// - Common uses: oscillations, periodic signals, harmonic analysis
#[derive(Debug, Clone, PartialEq)]
pub struct SinusoidalFit {
    /// Amplitude (A)
    pub amplitude: f64,
    
    /// Angular frequency (ω)
    pub frequency: f64,
    
    /// Phase shift (φ)
    pub phase: f64,
    
    /// Vertical offset (C)
    pub offset: f64,
    
    /// Coefficient of determination (R²)
    pub r_squared: f64,
}

impl SinusoidalFit {
    /// Predict y value for given x
    pub fn predict(&self, x: f64) -> f64 {
        self.amplitude * (self.frequency * x + self.phase).sin() + self.offset
    }
    
    /// Period of the oscillation (2π/ω)
    pub fn period(&self) -> f64 {
        2.0 * std::f64::consts::PI / self.frequency
    }
    
    /// Frequency in Hz (if x is in seconds)
    pub fn frequency_hz(&self) -> f64 {
        self.frequency / (2.0 * std::f64::consts::PI)
    }
    
    /// Peak-to-peak amplitude
    pub fn peak_to_peak(&self) -> f64 {
        2.0 * self.amplitude
    }
    
    /// RMS (root mean square) value
    pub fn rms(&self) -> f64 {
        (self.amplitude / std::f64::consts::SQRT_2 + self.offset.powi(2)).sqrt()
    }
}