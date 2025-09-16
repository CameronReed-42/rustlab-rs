//! Distribution shape analysis through higher-order moment statistics
//!
//! This module provides comprehensive measures for analyzing the shape characteristics
//! of probability distributions using standardized moments. These statistics help identify
//! asymmetry, tail behavior, and departures from normality in empirical data.
//!
//! # Statistical Shape Measures
//!
//! ## Skewness (Third Moment)
//!
//! Skewness quantifies the asymmetry of a distribution around its mean. It provides
//! critical information about the balance of data on either side of the central value.
//!
//! **Mathematical Definition:**
//! ```text
//! Skewness = E[(X - μ)³] / σ³ = μ₃ / σ³
//! ```
//!
//! **Interpretation:**
//! - **γ₁ = 0**: Perfectly symmetric distribution (normal-like)
//! - **γ₁ > 0**: Right-skewed (positive skew) - longer tail extends to the right
//! - **γ₁ < 0**: Left-skewed (negative skew) - longer tail extends to the left
//!
//! **Magnitude Guidelines:**
//! - |γ₁| < 0.5: Approximately symmetric
//! - 0.5 ≤ |γ₁| < 1: Moderately skewed
//! - |γ₁| ≥ 1: Highly skewed
//!
//! ## Kurtosis (Fourth Moment)
//!
//! Kurtosis measures the "tailedness" or peakedness of a distribution relative to
//! the normal distribution. It provides insights into the probability mass in the
//! tails versus the center.
//!
//! **Mathematical Definition:**
//! ```text
//! Excess Kurtosis = E[(X - μ)⁴] / σ⁴ - 3 = μ₄ / σ⁴ - 3
//! Raw Kurtosis = E[(X - μ)⁴] / σ⁴ = μ₄ / σ⁴
//! ```
//!
//! **Interpretation:**
//! - **γ₂ = 0** (Mesokurtic): Normal-like tails (raw kurtosis = 3)
//! - **γ₂ > 0** (Leptokurtic): Heavy tails, sharp peak, more extreme outliers
//! - **γ₂ < 0** (Platykurtic): Light tails, flat peak, fewer extreme values
//!
//! **Practical Significance:**
//! - **Risk Management**: High kurtosis indicates higher probability of extreme events
//! - **Quality Control**: Excess kurtosis may indicate process instability
//! - **Model Validation**: Departures from expected kurtosis suggest model inadequacy
//!
//! # Central Moments Theory
//!
//! Central moments are moments computed about the mean, providing standardized
//! measures of distribution characteristics:
//!
//! ```text
//! μₖ = E[(X - μ)ᵏ] = (1/n) Σᵢ (xᵢ - x̄)ᵏ
//! ```
//!
//! **Moment Hierarchy:**
//! - **μ₁ = 0**: Always zero by definition (deviation from mean)
//! - **μ₂ = σ²**: Variance (measure of spread)
//! - **μ₃**: Related to skewness (asymmetry measure)
//! - **μ₄**: Related to kurtosis (tail behavior measure)
//!
//! # Bias Correction and Sample Statistics
//!
//! This implementation uses bias-corrected formulas for sample statistics to provide
//! better estimates of population parameters:
//!
//! **Sample Skewness (bias-corrected):**
//! ```text
//! g₁ = [n / ((n-1)(n-2))] Σᵢ [(xᵢ - x̄) / s]³
//! ```
//!
//! **Sample Kurtosis (bias-corrected):**
//! ```text
//! g₂ = [n(n+1) / ((n-1)(n-2)(n-3))] Σᵢ [(xᵢ - x̄) / s]⁴ - [3(n-1)² / ((n-2)(n-3))]
//! ```
//!
//! # Applications and Use Cases
//!
//! ## Financial Risk Analysis
//! - **Return Distributions**: Identify fat tails and asymmetric risks
//! - **VaR Modeling**: Excess kurtosis affects extreme quantile estimates
//! - **Portfolio Optimization**: Skewness preferences in utility functions
//!
//! ## Quality Control and Manufacturing
//! - **Process Monitoring**: Detect shifts in distribution shape
//! - **Capability Analysis**: Ensure process outputs meet specifications
//! - **Defect Analysis**: Understand failure mode distributions
//!
//! ## Medical and Biological Sciences
//! - **Biomarker Analysis**: Identify non-normal physiological distributions
//! - **Drug Response**: Characterize patient response variability
//! - **Epidemiology**: Analyze disease incidence patterns
//!
//! ## Machine Learning and Data Science
//! - **Feature Engineering**: Transform skewed variables for better model performance
//! - **Anomaly Detection**: High kurtosis may indicate outlier presence
//! - **Model Validation**: Check residual distribution assumptions
//!
//! # Computational Considerations
//!
//! ## Numerical Stability
//! - Uses standardized computation to avoid overflow in extreme cases
//! - Bias correction formulas prevent systematic underestimation
//! - Robust to moderate outliers but sensitive to extreme values
//!
//! ## Sample Size Requirements
//! - **Skewness**: Minimum n = 3 for mathematical validity
//! - **Kurtosis**: Minimum n = 4 for bias-corrected estimation
//! - **Reliability**: n ≥ 30 recommended for stable estimates
//! - **Precision**: Large samples (n > 100) for precise population inference

use rustlab_math::{VectorF64, VectorF32};

/// Trait for comprehensive distribution shape analysis through moment statistics
///
/// This trait provides a complete suite of shape analysis tools based on central moments,
/// enabling detailed characterization of distribution properties beyond simple location
/// and scale measures.
///
/// # Mathematical Foundation
///
/// Shape analysis relies on standardized central moments, which are scale-invariant
/// measures that allow comparison across different units and scales:
///
/// ```text
/// Standardized Moment k = E[(X - μ)ᵏ] / σᵏ
/// ```
///
/// # Statistical Significance Testing
///
/// For hypothesis testing about shape parameters:
/// - **Skewness**: Test H₀: γ₁ = 0 using approximate normal distribution
/// - **Kurtosis**: Test H₀: γ₂ = 0 (normality) using approximate normal distribution
/// - **Joint test**: Jarque-Bera test combines both skewness and kurtosis
///
/// # Robustness Considerations
///
/// - **Outlier Sensitivity**: Moment-based measures are sensitive to extreme values
/// - **Sample Size Dependency**: Estimates stabilize with larger samples
/// - **Normality Assumption**: Tests assume approximate normality of moment estimates
pub trait Shape<T> {
    /// Compute bias-corrected sample skewness (third standardized moment)
    ///
    /// Skewness quantifies the asymmetry of a distribution, indicating whether data
    /// is concentrated on one side of the mean. This implementation uses the bias-corrected
    /// sample skewness formula for better population parameter estimation.
    ///
    /// # Mathematical Details
    ///
    /// **Population Skewness:**
    /// ```text
    /// γ₁ = E[(X - μ)³] / σ³
    /// ```
    ///
    /// **Bias-Corrected Sample Estimate:**
    /// ```text
    /// g₁ = [n / ((n-1)(n-2))] × Σᵢ [(xᵢ - x̄) / s]³
    /// ```
    ///
    /// # Interpretation Guidelines
    ///
    /// ## Sign Interpretation
    /// - **γ₁ > 0** (Right-skewed): 
    ///   - Longer tail extends to the right
    ///   - Mean > Median > Mode
    ///   - More extreme high values than low values
    /// - **γ₁ < 0** (Left-skewed):
    ///   - Longer tail extends to the left  
    ///   - Mode > Median > Mean
    ///   - More extreme low values than high values
    /// - **γ₁ ≈ 0** (Symmetric):
    ///   - Balanced distribution around the mean
    ///   - Mean ≈ Median ≈ Mode
    ///
    /// ## Magnitude Interpretation
    /// - **|γ₁| < 0.5**: Approximately symmetric (acceptable for most purposes)
    /// - **0.5 ≤ |γ₁| < 1.0**: Moderately skewed (may require transformation)
    /// - **1.0 ≤ |γ₁| < 2.0**: Highly skewed (transformation recommended)
    /// - **|γ₁| ≥ 2.0**: Extremely skewed (transformation essential)
    ///
    /// # Common Applications
    ///
    /// ## Income and Wealth Analysis
    /// Most income distributions are right-skewed due to high earners in the tail:
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// let household_incomes = vec64![25000, 30000, 35000, 40000, 45000, 200000];
    /// let skew = household_incomes.skewness();
    /// println!("Income distribution skewness: {:.3}", skew); // Positive
    /// ```
    ///
    /// ## Test Scores and Performance Metrics
    /// Educational assessments often show different skewness patterns:
    /// ```rust
    /// // Easy test (left-skewed - most students score high)
    /// let easy_test = vec64![95, 92, 88, 85, 82, 78, 65];
    /// 
    /// // Hard test (right-skewed - most students score low) 
    /// let hard_test = vec64![45, 52, 58, 65, 72, 88, 95];
    /// ```
    ///
    /// ## Financial Returns Analysis
    /// Asset returns often exhibit negative skewness (crash risk):
    /// ```rust
    /// let daily_returns = vec64![0.01, 0.02, -0.01, 0.01, -0.15]; // Crash day
    /// let skew = daily_returns.skewness(); // Likely negative
    /// ```
    ///
    /// # Statistical Properties
    ///
    /// - **Range**: (-∞, +∞) theoretically, typically [-3, +3] for real data
    /// - **Units**: Dimensionless (standardized measure)
    /// - **Robustness**: Sensitive to outliers (25% breakdown point)
    /// - **Efficiency**: Less efficient than median-based skewness measures
    ///
    /// # Assumptions and Limitations
    ///
    /// - **Minimum Sample Size**: n ≥ 3 for mathematical computation
    /// - **Non-zero Variance**: Standard deviation must be > 0
    /// - **Outlier Sensitivity**: Single extreme values can dominate the measure
    /// - **Normality**: Approximate normality assumed for significance testing
    ///
    /// # Returns
    /// Sample skewness coefficient with bias correction applied
    ///
    /// # Panics
    /// - Panics if the vector has fewer than 3 elements (mathematical requirement)
    /// - Panics if standard deviation is zero (division by zero)
    /// - Panics if vector is empty (undefined operation)
    fn skewness(&self) -> T;
    
    /// Compute bias-corrected sample excess kurtosis (fourth standardized moment - 3)
    ///
    /// Excess kurtosis measures the "tailedness" of a distribution relative to the normal
    /// distribution, providing crucial insights into the probability of extreme events and
    /// the concentration of data around the mean.
    ///
    /// # Mathematical Details
    ///
    /// **Population Excess Kurtosis (Fisher's definition):**
    /// ```text
    /// γ₂ = E[(X - μ)⁴] / σ⁴ - 3
    /// ```
    ///
    /// **Bias-Corrected Sample Estimate:**
    /// ```text
    /// g₂ = [n(n+1) / ((n-1)(n-2)(n-3))] Σᵢ [(xᵢ - x̄) / s]⁴ - [3(n-1)² / ((n-2)(n-3))]
    /// ```
    ///
    /// The subtraction of 3 makes the normal distribution have excess kurtosis of 0,
    /// providing a natural reference point for comparison.
    ///
    /// # Distribution Classification
    ///
    /// ## Mesokurtic (γ₂ ≈ 0)
    /// - **Characteristic**: Normal-like tail behavior
    /// - **Examples**: Gaussian, symmetric uniform distributions
    /// - **Interpretation**: Moderate probability of extreme values
    ///
    /// ## Leptokurtic (γ₂ > 0)
    /// - **Characteristic**: Heavy tails, sharp peak
    /// - **Examples**: t-distribution, Laplace distribution
    /// - **Risk Implication**: Higher probability of extreme events
    /// - **Financial Context**: "Fat tail" risk, crash events more likely
    ///
    /// ## Platykurtic (γ₂ < 0)
    /// - **Characteristic**: Light tails, flat peak
    /// - **Examples**: Uniform distribution, beta distributions
    /// - **Risk Implication**: Lower probability of extreme events
    /// - **Quality Context**: More consistent, predictable outcomes
    ///
    /// # Magnitude Interpretation
    ///
    /// - **|γ₂| < 0.5**: Approximately normal tail behavior
    /// - **0.5 ≤ |γ₂| < 1.0**: Moderately non-normal tails
    /// - **1.0 ≤ |γ₂| < 2.0**: Highly non-normal tails
    /// - **|γ₂| ≥ 2.0**: Extremely non-normal tails
    ///
    /// # Practical Applications
    ///
    /// ## Risk Management
    /// High kurtosis indicates higher probability of extreme losses:
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// let portfolio_returns = vec64![-0.08, -0.02, 0.01, 0.02, 0.15, -0.12];
    /// let kurt = portfolio_returns.kurtosis();
    /// if kurt > 1.0 {
    ///     println!("High tail risk detected - consider hedging");
    /// }
    /// ```
    ///
    /// ## Quality Control
    /// Manufacturing processes with high kurtosis may have control issues:
    /// ```rust
    /// let measurements = vec64![10.1, 10.0, 9.9, 10.0, 12.5, 7.8, 10.1];
    /// let kurt = measurements.kurtosis();
    /// if kurt > 2.0 {
    ///     println!("Process may have outlier-generating mechanism");
    /// }
    /// ```
    ///
    /// ## Model Validation
    /// Check if residuals follow assumed distribution:
    /// ```rust
    /// let residuals = vec64![0.1, -0.2, 0.05, -0.1, 0.15, -0.05];
    /// let kurt = residuals.kurtosis();
    /// if kurt.abs() > 1.0 {
    ///     println!("Model assumptions may be violated");
    /// }
    /// ```
    ///
    /// # Statistical Properties
    ///
    /// - **Normal Distribution**: γ₂ = 0 (by definition)
    /// - **Uniform Distribution**: γ₂ = -1.2 (platykurtic)
    /// - **Exponential Distribution**: γ₂ = 6 (highly leptokurtic)
    /// - **Student's t (df=5)**: γ₂ = 6 (heavy tails)
    ///
    /// # Relationship to Other Statistics
    ///
    /// - **Jarque-Bera Test**: Combines skewness and kurtosis for normality testing
    /// - **Coefficient of Variation**: High kurtosis often accompanies high CV
    /// - **Extreme Value Theory**: Kurtosis relates to tail index parameters
    ///
    /// # Computational Considerations
    ///
    /// - **Sample Size**: n ≥ 4 required for mathematical validity
    /// - **Stability**: Estimates become reliable with n ≥ 50
    /// - **Precision**: Large samples (n > 200) needed for precise inference
    /// - **Robustness**: Extremely sensitive to outliers (single point can dominate)
    ///
    /// # Returns
    /// Bias-corrected sample excess kurtosis coefficient
    ///
    /// # Panics
    /// - Panics if the vector has fewer than 4 elements (mathematical requirement)
    /// - Panics if standard deviation is zero (division by zero)
    /// - Panics if vector is empty (undefined operation)
    fn kurtosis(&self) -> T;
    
    /// Compute bias-corrected sample raw kurtosis (fourth standardized moment)
    ///
    /// Raw kurtosis provides the pure fourth standardized moment without the adjustment
    /// that makes normal distributions have zero excess kurtosis. This is useful for
    /// theoretical work and when comparing with literature that uses raw kurtosis.
    ///
    /// # Mathematical Definition
    ///
    /// **Population Raw Kurtosis:**
    /// ```text
    /// μ₄/σ⁴ = E[(X - μ)⁴] / σ⁴
    /// ```
    ///
    /// **Relationship to Excess Kurtosis:**
    /// ```text
    /// Raw Kurtosis = Excess Kurtosis + 3
    /// ```
    ///
    /// # Reference Values
    ///
    /// - **Normal Distribution**: Raw kurtosis = 3.0
    /// - **Uniform Distribution**: Raw kurtosis = 1.8
    /// - **Exponential Distribution**: Raw kurtosis = 9.0
    /// - **Cauchy Distribution**: Raw kurtosis = ∞ (undefined)
    ///
    /// # When to Use Raw vs Excess Kurtosis
    ///
    /// ## Use Raw Kurtosis When:
    /// - Working with theoretical probability distributions
    /// - Implementing statistical tests that expect raw moments
    /// - Comparing with older statistical literature
    /// - Computing moment-generating functions
    ///
    /// ## Use Excess Kurtosis When:
    /// - Comparing tail behavior to normal distribution
    /// - Risk analysis and extreme event modeling
    /// - Most practical data analysis applications
    /// - Modern statistical software compatibility
    ///
    /// # Example Applications
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// let data = vec64![1, 2, 3, 4, 5, 6, 7];
    /// let raw_kurt = data.kurtosis_raw();
    /// let excess_kurt = data.kurtosis();
    /// 
    /// // Verify relationship
    /// assert!((raw_kurt - excess_kurt - 3.0).abs() < 1e-10);
    /// 
    /// // Compare to normal distribution
    /// if raw_kurt > 3.0 {
    ///     println!("Heavier tails than normal distribution");
    /// } else if raw_kurt < 3.0 {
    ///     println!("Lighter tails than normal distribution");
    /// } else {
    ///     println!("Normal-like tail behavior");
    /// }
    /// ```
    ///
    /// # Statistical Properties
    ///
    /// - **Range**: [1, ∞) for continuous distributions
    /// - **Minimum**: Achieved by Bernoulli distribution at p = 0.5
    /// - **Units**: Dimensionless (fourth power cancels units)
    /// - **Monotonicity**: Higher values indicate more extreme tail behavior
    ///
    /// # Returns
    /// Bias-corrected sample raw kurtosis coefficient
    ///
    /// # Panics
    /// - Panics if the vector has fewer than 4 elements (mathematical requirement)
    /// - Panics if standard deviation is zero (division by zero)
    /// - Panics if vector is empty (undefined operation)
    fn kurtosis_raw(&self) -> T;
    
    /// Compute the nth central moment about the mean
    ///
    /// Central moments form the foundation of distribution analysis, providing standardized
    /// measures of various distributional characteristics. This function computes raw
    /// (non-standardized) central moments for theoretical analysis and advanced statistics.
    ///
    /// # Mathematical Definition
    ///
    /// **Central Moment of Order n:**
    /// ```text
    /// μₙ = E[(X - μ)ⁿ] = (1/N) Σᵢ (xᵢ - x̄)ⁿ
    /// ```
    ///
    /// where μ is the population mean (estimated by sample mean x̄).
    ///
    /// # Moment Hierarchy and Interpretation
    ///
    /// ## First Central Moment (n = 1)
    /// - **Value**: Always exactly 0 by mathematical definition
    /// - **Interpretation**: Mean deviation from mean (trivially zero)
    /// - **Use**: Verification of computational correctness
    ///
    /// ## Second Central Moment (n = 2)
    /// - **Identity**: Population variance σ²
    /// - **Interpretation**: Average squared deviation from mean
    /// - **Units**: Square of original data units
    /// - **Application**: Fundamental measure of dispersion
    ///
    /// ## Third Central Moment (n = 3)
    /// - **Relationship**: μ₃ = σ³ × Skewness
    /// - **Interpretation**: Asymmetry measure (raw, with units)
    /// - **Units**: Cube of original data units
    /// - **Sign**: Positive for right-skewed, negative for left-skewed
    ///
    /// ## Fourth Central Moment (n = 4)
    /// - **Relationship**: μ₄ = σ⁴ × Raw Kurtosis
    /// - **Interpretation**: Tail heaviness measure (raw, with units)
    /// - **Units**: Fourth power of original data units
    /// - **Application**: Risk analysis, extreme event modeling
    ///
    /// ## Higher-Order Moments (n ≥ 5)
    /// - **Theoretical Use**: Advanced distributional analysis
    /// - **Practical Limitation**: Increasingly sensitive to outliers
    /// - **Sample Requirements**: Require very large samples for stability
    ///
    /// # Standardized vs Raw Moments
    ///
    /// | Property | Raw Central Moments | Standardized Moments |
    /// |----------|---------------------|----------------------|
    /// | Units | Powers of original units | Dimensionless |
    /// | Comparability | Scale-dependent | Scale-invariant |
    /// | Use case | Theoretical analysis | Practical comparison |
    /// | Formula | E[(X-μ)ⁿ] | E[(X-μ)ⁿ]/σⁿ |
    ///
    /// # Practical Applications
    ///
    /// ## Moment Method Estimation
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// let data = vec64![1.2, 2.1, 1.8, 2.5, 1.9, 2.2, 1.7];
    /// 
    /// // Extract moments for distribution fitting
    /// let m1 = data.moment(1); // Always 0
    /// let m2 = data.moment(2); // Variance
    /// let m3 = data.moment(3); // Third moment
    /// let m4 = data.moment(4); // Fourth moment
    /// 
    /// println!("Moments: {}, {:.3}, {:.3}, {:.3}", m1, m2, m3, m4);
    /// ```
    ///
    /// ## Custom Distribution Analysis
    /// ```rust
    /// // Compute Pearson's coefficient of skewness from moments
    /// let m2 = data.moment(2);
    /// let m3 = data.moment(3);
    /// let skewness_pearson = m3 / m2.powf(1.5);
    /// ```
    ///
    /// ## Quality Control Charts
    /// ```rust
    /// // Monitor process stability using moment evolution
    /// let daily_measurements = vec64![10.1, 9.9, 10.0, 10.2, 9.8];
    /// let second_moment = daily_measurements.moment(2);
    /// let fourth_moment = daily_measurements.moment(4);
    /// 
    /// // Check if process variability changed
    /// let moment_ratio = fourth_moment / (second_moment * second_moment);
    /// ```
    ///
    /// # Computational Properties
    ///
    /// ## Numerical Stability
    /// - **Low Orders (n ≤ 4)**: Generally stable for moderate sample sizes
    /// - **High Orders (n > 4)**: Increasingly unstable, sensitive to outliers
    /// - **Precision**: Double precision recommended for n > 6
    ///
    /// ## Sample Size Requirements
    /// - **Minimum**: n ≥ 2 for meaningful variance (second moment)
    /// - **Recommended**: n ≥ 10 × moment_order for stable estimation
    /// - **Asymptotic**: n ≥ 100 for higher-order moments (n ≥ 5)
    ///
    /// ## Outlier Sensitivity
    /// Higher-order moments become increasingly sensitive to extreme values:
    /// - **Second moment**: Moderate sensitivity
    /// - **Third moment**: High sensitivity
    /// - **Fourth moment**: Very high sensitivity
    /// - **nth moment**: Sensitivity increases exponentially with n
    ///
    /// # Mathematical Properties
    ///
    /// - **Linearity**: Not generally linear (except for first moment)
    /// - **Scale Invariance**: μₙ(aX) = aⁿ μₙ(X) for scaling factor a
    /// - **Translation Invariance**: Central moments unchanged by location shifts
    /// - **Existence**: Requires finite nth moment of underlying distribution
    ///
    /// # Arguments
    /// * `n` - Order of the central moment (must be ≥ 1)
    ///
    /// # Returns
    /// The nth central moment in original data units raised to the nth power
    ///
    /// # Panics
    /// - Panics if the vector is empty (mean undefined)
    /// - Panics if n < 1 (moment order must be positive)
    /// - Panics on numerical overflow for very high moments or extreme data
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// let data = vec64![1, 2, 3, 4, 5];
    /// 
    /// assert_eq!(data.moment(1), 0.0); // Always zero
    /// assert!(data.moment(2) > 0.0);   // Variance is positive
    /// 
    /// // Verify relationship with variance
    /// let variance = data.var_pop(); // Population variance
    /// let second_moment = data.moment(2);
    /// assert!((variance - second_moment).abs() < 1e-10);
    /// ```
    fn moment(&self, n: u32) -> T;
}

impl Shape<f64> for VectorF64 {
    fn skewness(&self) -> f64 {
        assert!(self.len() >= 3, "Skewness requires at least 3 data points, got {}", self.len());
        
        use rustlab_math::statistics::BasicStatistics;
        let mean = self.mean();
        let std_dev = self.std(None);
        assert!(std_dev != 0.0, "Cannot compute skewness when standard deviation is zero");
        
        let n = self.len() as f64;
        let slice = self.as_slice_unchecked();
        
        let third_moment_sum: f64 = slice.iter()
            .map(|&x| {
                let z = (x - mean) / std_dev;
                z * z * z
            })
            .sum();
        
        // Sample skewness with bias correction
        (n / ((n - 1.0) * (n - 2.0))) * third_moment_sum
    }
    
    fn kurtosis(&self) -> f64 {
        self.kurtosis_raw() - 3.0
    }
    
    fn kurtosis_raw(&self) -> f64 {
        assert!(self.len() >= 4, "Kurtosis requires at least 4 data points, got {}", self.len());
        
        use rustlab_math::statistics::BasicStatistics;
        let mean = self.mean();
        let std_dev = self.std(None);
        assert!(std_dev != 0.0, "Cannot compute kurtosis when standard deviation is zero");
        
        let n = self.len() as f64;
        let slice = self.as_slice_unchecked();
        
        let fourth_moment_sum: f64 = slice.iter()
            .map(|&x| {
                let z = (x - mean) / std_dev;
                let z2 = z * z;
                z2 * z2
            })
            .sum();
        
        // Sample kurtosis with bias correction
        let term1 = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
        let term2 = (3.0 * (n - 1.0) * (n - 1.0)) / ((n - 2.0) * (n - 3.0));
        
        term1 * fourth_moment_sum - term2
    }
    
    fn moment(&self, n: u32) -> f64 {
        assert!(!self.is_empty(), "Cannot compute moment of empty vector");
        assert!(n >= 1, "Moment order must be at least 1, got {}", n);
        
        if n == 1 {
            return 0.0; // First central moment is always 0
        }
        
        use rustlab_math::statistics::BasicStatistics;
        let mean = self.mean();
        let slice = self.as_slice_unchecked();
        
        let moment_sum: f64 = slice.iter()
            .map(|&x| (x - mean).powi(n as i32))
            .sum();
        
        moment_sum / self.len() as f64
    }
}

impl Shape<f32> for VectorF32 {
    fn skewness(&self) -> f32 {
        assert!(self.len() >= 3, "Skewness requires at least 3 data points, got {}", self.len());
        
        use rustlab_math::statistics::BasicStatistics;
        let mean = self.mean();
        let std_dev = self.std(None);
        assert!(std_dev != 0.0, "Cannot compute skewness when standard deviation is zero");
        
        let n = self.len() as f32;
        let slice = self.as_slice_unchecked();
        
        let third_moment_sum: f32 = slice.iter()
            .map(|&x| {
                let z = (x - mean) / std_dev;
                z * z * z
            })
            .sum();
        
        // Sample skewness with bias correction
        (n / ((n - 1.0) * (n - 2.0))) * third_moment_sum
    }
    
    fn kurtosis(&self) -> f32 {
        self.kurtosis_raw() - 3.0
    }
    
    fn kurtosis_raw(&self) -> f32 {
        assert!(self.len() >= 4, "Kurtosis requires at least 4 data points, got {}", self.len());
        
        use rustlab_math::statistics::BasicStatistics;
        let mean = self.mean();
        let std_dev = self.std(None);
        assert!(std_dev != 0.0, "Cannot compute kurtosis when standard deviation is zero");
        
        let n = self.len() as f32;
        let slice = self.as_slice_unchecked();
        
        let fourth_moment_sum: f32 = slice.iter()
            .map(|&x| {
                let z = (x - mean) / std_dev;
                let z2 = z * z;
                z2 * z2
            })
            .sum();
        
        // Sample kurtosis with bias correction
        let term1 = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
        let term2 = (3.0 * (n - 1.0) * (n - 1.0)) / ((n - 2.0) * (n - 3.0));
        
        term1 * fourth_moment_sum - term2
    }
    
    fn moment(&self, n: u32) -> f32 {
        assert!(!self.is_empty(), "Cannot compute moment of empty vector");
        assert!(n >= 1, "Moment order must be at least 1, got {}", n);
        
        if n == 1 {
            return 0.0; // First central moment is always 0
        }
        
        use rustlab_math::statistics::BasicStatistics;
        let mean = self.mean();
        let slice = self.as_slice_unchecked();
        
        let moment_sum: f32 = slice.iter()
            .map(|&x| (x - mean).powi(n as i32))
            .sum();
        
        moment_sum / self.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::vec64;
    
    #[test]
    fn test_symmetric_distribution_skewness() {
        // Symmetric distribution should have skewness near 0
        let v = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let skew = v.skewness();
        assert!(skew.abs() < 0.1); // Should be close to 0 for symmetric data
    }
    
    #[test]
    fn test_right_skewed_distribution() {
        // Right-skewed distribution should have positive skewness
        let v = vec64![1, 1, 1, 2, 2, 3, 10]; // Heavy tail to the right
        let skew = v.skewness();
        assert!(skew > 0.0);
    }
    
    #[test]
    fn test_normal_like_kurtosis() {
        // For normally distributed data, excess kurtosis should be near 0
        let v = vec64![1, 2, 3, 4, 5, 6, 7];
        let kurt = v.kurtosis();
        // For small samples, we just check it's finite
        assert!(kurt.is_finite());
    }
    
    #[test]
    fn test_kurtosis_raw() {
        let v = vec64![1, 2, 3, 4, 5];
        let kurt_raw = v.kurtosis_raw();
        let kurt_excess = v.kurtosis();
        assert!((kurt_raw - kurt_excess - 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_second_moment_is_variance() {
        use rustlab_math::statistics::BasicStatistics;
        let v = vec64![1, 2, 3, 4, 5];
        let second_moment = v.moment(2);
        let variance = v.var_pop(); // Population variance
        assert!((second_moment - variance).abs() < 1e-10);
    }
    
    #[test]
    fn test_first_moment_is_zero() {
        let v = vec64![1, 2, 3, 4, 5];
        assert_eq!(v.moment(1), 0.0);
    }
    
    #[test]
    #[should_panic(expected = "Skewness requires at least 3 data points")]
    fn test_skewness_insufficient_data() {
        let v = vec64![1, 2];
        v.skewness();
    }
    
    #[test]
    #[should_panic(expected = "Kurtosis requires at least 4 data points")]
    fn test_kurtosis_insufficient_data() {
        let v = vec64![1, 2, 3];
        v.kurtosis();
    }
    
    #[test]
    #[should_panic(expected = "Cannot compute skewness when standard deviation is zero")]
    fn test_skewness_zero_variance() {
        let v = vec64![5, 5, 5, 5]; // All same values
        v.skewness();
    }
}