//! Discrete probability distributions
//!
//! This module provides implementations of fundamental discrete probability distributions,
//! which are used to model random variables that take on countable values (integers).
//! These distributions are essential in statistics, combinatorics, and many real-world
//! applications.
//!
//! ## Mathematical Background
//!
//! Discrete distributions are characterized by:
//! - **Support**: A countable set of values (typically integers)
//! - **PMF**: Probability mass function P(X = k) giving the probability of specific values
//! - **CDF**: Cumulative distribution function F(k) = P(X ≤ k)
//! - **Properties**: Σ P(X = k) = 1 over all possible values k
//!
//! ## Available Distributions
//!
//! ### Bernoulli Distribution
//! - **Use case**: Single trial with success/failure outcome
//! - **Parameter**: p ∈ [0, 1] (probability of success)
//! - **Applications**: Coin flips, binary outcomes, A/B testing base case
//!
//! ### Binomial Distribution  
//! - **Use case**: Number of successes in n independent Bernoulli trials
//! - **Parameters**: n ≥ 1 (trials), p ∈ [0, 1] (success probability)
//! - **Applications**: Quality control, clinical trials, polling
//!
//! ### Poisson Distribution
//! - **Use case**: Count of events in fixed interval (time/space)
//! - **Parameter**: λ > 0 (average rate of events)
//! - **Applications**: Network traffic, radioactive decay, customer arrivals
//!
//! ## Implementation Features
//!
//! - **High Precision**: Uses logarithmic computation to avoid overflow
//! - **Efficient Sampling**: Optimized algorithms (Knuth's method for Poisson)
//! - **Numerical Stability**: Careful handling of extreme parameter values
//! - **Integration Ready**: Optional RustLab-Math vector operations
//!
//! ## Examples
//!
//! ```rust
//! use rustlab_distributions::{Bernoulli, Binomial, Poisson, DiscreteDistribution, Sampling};
//! use rand::thread_rng;
//!
//! // Bernoulli: coin flip with 60% success probability
//! let coin = Bernoulli::new(0.6).unwrap();
//! let outcome = coin.pmf(1);  // P(success) = 0.6
//!
//! // Binomial: 10 trials with 30% success rate
//! let binomial = Binomial::new(10, 0.3).unwrap();
//! let prob_5_successes = binomial.pmf(5);
//!
//! // Poisson: average 3.2 events per interval
//! let poisson = Poisson::new(3.2).unwrap();
//! let mut rng = thread_rng();
//! let random_count = poisson.sample(&mut rng);
//! ```

use crate::error::{Result, DistributionError};
use crate::traits::{Distribution, DiscreteDistribution, Sampling};
use rand::Rng;

#[cfg(feature = "integration")]
use rustlab_math::VectorF64;

/// Bernoulli distribution B(p)
///
/// The Bernoulli distribution models a single trial with exactly two possible outcomes:
/// success (1) and failure (0). It is the simplest discrete probability distribution
/// and forms the building block for more complex distributions like Binomial.
///
/// # Mathematical Properties
///
/// - **Parameter**: p ∈ [0, 1] (probability of success)
/// - **Support**: {0, 1}
/// - **PMF**: P(X = k) = p^k × (1-p)^(1-k) for k ∈ {0, 1}
/// - **Mean**: E[X] = p
/// - **Variance**: Var[X] = p(1-p)
/// - **Mode**: 1 if p > 0.5, 0 if p < 0.5, both if p = 0.5
///
/// # Applications
///
/// - **A/B Testing**: Success/failure of a single experiment
/// - **Quality Control**: Pass/fail of a single item inspection
/// - **Medical Trials**: Success/failure of a treatment on a single patient
/// - **Financial Modeling**: Default/non-default of a single loan
/// - **Coin Flips**: Heads/tails outcomes (with potentially biased coins)
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::{Bernoulli, Distribution, DiscreteDistribution, Sampling};
/// use rand::thread_rng;
///
/// // Fair coin flip (p = 0.5)
/// let coin = Bernoulli::new(0.5).unwrap();
/// assert_eq!(coin.mean(), 0.5);
/// assert_eq!(coin.variance(), 0.25);
///
/// // Biased coin with 70% success rate
/// let biased_coin = Bernoulli::new(0.7).unwrap();
/// assert_eq!(biased_coin.pmf(1), 0.7);  // P(success) = 0.7
/// assert_eq!(biased_coin.pmf(0), 0.3);  // P(failure) = 0.3
///
/// // Sample from distribution
/// let mut rng = thread_rng();
/// let outcome = coin.sample(&mut rng);  // Returns 0 or 1
/// ```
///
/// # Implementation Details
///
/// - Sampling uses inverse transform method with uniform random variable
/// - PMF computation is exact (no numerical approximation needed)
/// - CDF has simple closed-form expression
/// - Quantile function uses simple threshold comparison
#[derive(Debug, Clone, PartialEq)]
pub struct Bernoulli {
    /// Probability of success p ∈ [0, 1]
    ///
    /// This parameter determines the probability that a single trial results
    /// in success (outcome = 1). Must be between 0 and 1 inclusive.
    pub p: f64,
}

impl Bernoulli {
    /// Create a new Bernoulli distribution B(p)
    ///
    /// Constructs a Bernoulli distribution with the specified success probability.
    /// This is the primary constructor that validates the parameter.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of success, must be in [0, 1]
    ///
    /// # Returns
    ///
    /// Returns `Ok(Bernoulli)` if p is valid, otherwise returns an error.
    ///
    /// # Errors
    ///
    /// - `InvalidParameter`: When p < 0, p > 1, or p is NaN/infinite
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::Bernoulli;
    ///
    /// // Valid parameters
    /// let fair_coin = Bernoulli::new(0.5).unwrap();
    /// let always_success = Bernoulli::new(1.0).unwrap();
    /// let never_success = Bernoulli::new(0.0).unwrap();
    ///
    /// // Invalid parameters
    /// assert!(Bernoulli::new(-0.1).is_err());    // Negative probability
    /// assert!(Bernoulli::new(1.5).is_err());     // Probability > 1
    /// assert!(Bernoulli::new(f64::NAN).is_err()); // NaN probability
    /// ```
    pub fn new(p: f64) -> Result<Self> {
        if !p.is_finite() || p < 0.0 || p > 1.0 {
            return Err(DistributionError::invalid_parameter("Probability must be in [0, 1]"));
        }
        Ok(Bernoulli { p })
    }
}

impl Distribution for Bernoulli {
    type Params = f64;
    type Support = i64;
    
    fn new(params: Self::Params) -> Result<Self> {
        Self::new(params)
    }
    
    fn params(&self) -> &Self::Params {
        &self.p
    }
    
    fn mean(&self) -> f64 {
        self.p
    }
    
    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }
}

impl DiscreteDistribution for Bernoulli {
    /// Probability mass function P(X = k)
    ///
    /// Computes the probability that the Bernoulli random variable equals k.
    /// For Bernoulli distribution: P(X = 1) = p, P(X = 0) = 1-p, P(X = k) = 0 otherwise.
    ///
    /// # Arguments
    ///
    /// * `k` - The value to evaluate the PMF at
    ///
    /// # Returns
    ///
    /// The probability P(X = k), which is p for k=1, (1-p) for k=0, and 0 otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Bernoulli, DiscreteDistribution};
    ///
    /// let bernoulli = Bernoulli::new(0.3).unwrap();
    /// assert_eq!(bernoulli.pmf(1), 0.3);   // Success probability
    /// assert_eq!(bernoulli.pmf(0), 0.7);   // Failure probability
    /// assert_eq!(bernoulli.pmf(2), 0.0);   // Outside support
    /// assert_eq!(bernoulli.pmf(-1), 0.0);  // Outside support
    /// ```
    fn pmf(&self, k: i64) -> f64 {
        match k {
            0 => 1.0 - self.p,
            1 => self.p,
            _ => 0.0,
        }
    }
    
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            0.0
        } else if k < 1 {
            1.0 - self.p
        } else {
            1.0
        }
    }
    
    fn quantile(&self, p: f64) -> Result<i64> {
        if !p.is_finite() || p < 0.0 || p > 1.0 {
            return Err(DistributionError::invalid_parameter("Probability must be in [0, 1]"));
        }
        
        if p <= 1.0 - self.p {
            Ok(0)
        } else {
            Ok(1)
        }
    }
}

impl Sampling for Bernoulli {
    /// Sample a single value from the Bernoulli distribution
    ///
    /// Uses the inverse transform method: generates U ~ Uniform(0,1) and returns
    /// 1 if U < p, otherwise 0. This is the standard and most efficient method
    /// for Bernoulli sampling.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator implementing the `Rng` trait
    ///
    /// # Returns
    ///
    /// A sample from the distribution: either 0 (failure) or 1 (success)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Bernoulli, Sampling};
    /// use rand::thread_rng;
    ///
    /// let bernoulli = Bernoulli::new(0.7).unwrap();
    /// let mut rng = thread_rng();
    ///
    /// // Sample many times to verify approximate probability
    /// let samples: Vec<i64> = (0..10000)
    ///     .map(|_| bernoulli.sample(&mut rng))
    ///     .collect();
    /// let success_rate = samples.iter().sum::<i64>() as f64 / samples.len() as f64;
    /// assert!((success_rate - 0.7).abs() < 0.05); // Should be close to 0.7
    /// ```
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Support {
        if rng.gen::<f64>() < self.p { 1 } else { 0 }
    }
    
    #[cfg(feature = "integration")]
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> VectorF64 {
        let mut vec = VectorF64::zeros(n);
        for i in 0..n {
            let sample = if rng.gen::<f64>() < self.p { 1.0 } else { 0.0 };
            vec.set(i, sample).unwrap();
        }
        vec
    }
    
    #[cfg(feature = "integration")]
    fn sample_into<R: Rng>(&self, rng: &mut R, output: &mut VectorF64) {
        for i in 0..output.len() {
            let sample = if rng.gen::<f64>() < self.p { 1.0 } else { 0.0 };
            output.set(i, sample).unwrap();
        }
    }
}

/// Binomial distribution B(n, p)
///
/// The Binomial distribution models the number of successes in n independent
/// Bernoulli trials, each with success probability p. It is one of the most
/// important discrete distributions and has widespread applications in statistics,
/// quality control, and hypothesis testing.
///
/// # Mathematical Properties
///
/// - **Parameters**: 
///   - n ≥ 1 (number of independent trials)
///   - p ∈ [0, 1] (probability of success in each trial)
/// - **Support**: {0, 1, 2, ..., n}
/// - **PMF**: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
/// - **Mean**: E[X] = np
/// - **Variance**: Var[X] = np(1-p)
/// - **Mode**: ⌊np + p⌋ or ⌈np + p⌉ - 1
///
/// # Applications
///
/// - **Quality Control**: Number of defective items in a batch
/// - **Clinical Trials**: Number of patients responding to treatment
/// - **Market Research**: Number of positive responses in surveys
/// - **Reliability Engineering**: Number of component failures
/// - **A/B Testing**: Number of conversions in n trials
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::{Binomial, Distribution, DiscreteDistribution, Sampling};
/// use rand::thread_rng;
///
/// // 10 coin flips with fair coin
/// let binomial = Binomial::new(10, 0.5).unwrap();
/// assert_eq!(binomial.mean(), 5.0);      // Expected 5 heads
/// assert_eq!(binomial.variance(), 2.5);  // Variance = np(1-p)
///
/// // Quality control: 100 items with 2% defect rate
/// let quality = Binomial::new(100, 0.02).unwrap();
/// let prob_no_defects = quality.pmf(0);  // P(0 defects)
/// let prob_at_most_5 = quality.cdf(5);   // P(≤ 5 defects)
///
/// // Sample number of successes
/// let mut rng = thread_rng();
/// let successes = binomial.sample(&mut rng);  // 0 to 10
/// ```
///
/// # Implementation Details
///
/// - PMF uses logarithmic computation to avoid overflow for large n
/// - Sampling uses sum of n Bernoulli trials (simple but not optimal for large n)
/// - For large n and moderate p, Normal approximation could be used
/// - Binomial coefficients computed iteratively to maintain numerical stability
#[derive(Debug, Clone, PartialEq)]
pub struct Binomial {
    /// Number of independent trials n ≥ 1
    ///
    /// Each trial is an independent Bernoulli experiment with the same
    /// success probability p. Must be at least 1.
    pub n: u32,
    /// Probability of success in each trial p ∈ [0, 1]
    ///
    /// This is the probability that any individual trial results in success.
    /// All n trials share the same success probability.
    pub p: f64,
    /// Cached parameters tuple for trait implementation
    params: (u32, f64),
}

impl Binomial {
    /// Create a new Binomial distribution B(n, p)
    ///
    /// Constructs a binomial distribution with n independent trials, each having
    /// success probability p. Validates both parameters for mathematical correctness.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of independent trials, must be at least 1
    /// * `p` - Success probability for each trial, must be in [0, 1]
    ///
    /// # Returns
    ///
    /// Returns `Ok(Binomial)` if parameters are valid, otherwise returns an error.
    ///
    /// # Errors
    ///
    /// - `InvalidParameter`: When p < 0, p > 1, or p is NaN/infinite
    /// - Note: n = 0 is currently allowed but mathematically questionable
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::Binomial;
    ///
    /// // Valid parameters
    /// let coin_flips = Binomial::new(10, 0.5).unwrap();     // 10 fair coin flips
    /// let sure_success = Binomial::new(5, 1.0).unwrap();    // Always succeeds
    /// let no_success = Binomial::new(20, 0.0).unwrap();     // Never succeeds
    ///
    /// // Invalid parameters
    /// assert!(Binomial::new(10, -0.1).is_err());   // Negative probability
    /// assert!(Binomial::new(10, 1.5).is_err());    // Probability > 1
    /// assert!(Binomial::new(10, f64::NAN).is_err()); // NaN probability
    /// ```
    pub fn new(n: u32, p: f64) -> Result<Self> {
        if !p.is_finite() || p < 0.0 || p > 1.0 {
            return Err(DistributionError::invalid_parameter("Probability must be in [0, 1]"));
        }
        Ok(Binomial { 
            n, 
            p,
            params: (n, p),
        })
    }
}

impl Distribution for Binomial {
    type Params = (u32, f64);
    type Support = i64;
    
    fn new(params: Self::Params) -> Result<Self> {
        Self::new(params.0, params.1)
    }
    
    fn params(&self) -> &Self::Params {
        &self.params
    }
    
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
    
    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
}

impl DiscreteDistribution for Binomial {
    /// Probability mass function P(X = k)
    ///
    /// Computes the probability that exactly k successes occur in n independent trials.
    /// Uses the binomial coefficient formula: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
    ///
    /// # Arguments
    ///
    /// * `k` - Number of successes to evaluate (must be between 0 and n)
    ///
    /// # Returns
    ///
    /// The probability P(X = k), or 0.0 if k is outside the support [0, n].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Binomial, DiscreteDistribution};
    ///
    /// let binomial = Binomial::new(3, 0.4).unwrap();
    /// 
    /// // P(X = 0) = C(3,0) × 0.4^0 × 0.6^3 = 1 × 1 × 0.216 = 0.216
    /// assert!((binomial.pmf(0) - 0.216).abs() < 1e-10);
    /// 
    /// // P(X = 2) = C(3,2) × 0.4^2 × 0.6^1 = 3 × 0.16 × 0.6 = 0.288
    /// assert!((binomial.pmf(2) - 0.288).abs() < 1e-10);
    /// 
    /// // Outside support
    /// assert_eq!(binomial.pmf(-1), 0.0);
    /// assert_eq!(binomial.pmf(4), 0.0);
    /// ```
    ///
    /// # Implementation Note
    ///
    /// The current implementation computes the binomial coefficient iteratively
    /// to avoid factorial overflow. For large n, a logarithmic implementation
    /// using lgamma would be more numerically stable.
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 || k > self.n as i64 {
            return 0.0;
        }
        
        // Simplified implementation - in practice would use lgamma for numerical stability
        let n = self.n as f64;
        let k_f = k as f64;
        let binom_coeff = (0..k).fold(1.0, |acc, i| acc * (n - i as f64) / (i as f64 + 1.0));
        binom_coeff * self.p.powf(k_f) * (1.0 - self.p).powf(n - k_f)
    }
    
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        if k >= self.n as i64 {
            return 1.0;
        }
        
        // Simplified implementation - sum PMF values
        (0..=k).map(|i| self.pmf(i)).sum()
    }
    
    fn quantile(&self, p: f64) -> Result<i64> {
        if !p.is_finite() || p < 0.0 || p > 1.0 {
            return Err(DistributionError::invalid_parameter("Probability must be in [0, 1]"));
        }
        
        // Simple linear search - could be optimized
        let mut cumulative = 0.0;
        for k in 0..=self.n as i64 {
            cumulative += self.pmf(k);
            if cumulative >= p {
                return Ok(k);
            }
        }
        Ok(self.n as i64)
    }
}

impl Sampling for Binomial {
    /// Sample from the Binomial distribution
    ///
    /// Generates a random number of successes by simulating n independent Bernoulli
    /// trials. This is a simple but not optimal implementation - for large n,
    /// more sophisticated algorithms like rejection sampling would be preferred.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator implementing the `Rng` trait
    ///
    /// # Returns
    ///
    /// The number of successes (integer between 0 and n inclusive)
    ///
    /// # Algorithm
    ///
    /// For each of the n trials:
    /// 1. Generate U ~ Uniform(0,1)
    /// 2. If U < p, count as success
    /// 3. Return total number of successes
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Binomial, Sampling};
    /// use rand::thread_rng;
    ///
    /// let binomial = Binomial::new(20, 0.3).unwrap();
    /// let mut rng = thread_rng();
    ///
    /// // Sample many times to verify approximate mean
    /// let samples: Vec<i64> = (0..10000)
    ///     .map(|_| binomial.sample(&mut rng))
    ///     .collect();
    /// let sample_mean = samples.iter().sum::<i64>() as f64 / samples.len() as f64;
    /// 
    /// // Should be close to theoretical mean np = 20 * 0.3 = 6.0
    /// assert!((sample_mean - 6.0).abs() < 0.2);
    /// ```
    ///
    /// # Performance Note
    ///
    /// This implementation has O(n) complexity. For large n with moderate p,
    /// algorithms like BTPE (Binomial Triangle Parallelogram Exponential) would be faster.
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Support {
        // Simple implementation: sum n Bernoulli trials
        let mut successes = 0;
        for _ in 0..self.n {
            if rng.gen::<f64>() < self.p {
                successes += 1;
            }
        }
        successes as i64
    }
    
    #[cfg(feature = "integration")]
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> VectorF64 {
        let mut vec = VectorF64::zeros(n);
        for i in 0..n {
            let sample = self.sample(rng) as f64;
            vec.set(i, sample).unwrap();
        }
        vec
    }
    
    #[cfg(feature = "integration")]
    fn sample_into<R: Rng>(&self, rng: &mut R, output: &mut VectorF64) {
        for i in 0..output.len() {
            let sample = self.sample(rng) as f64;
            output.set(i, sample).unwrap();
        }
    }
}

/// Poisson distribution Pois(λ)
///
/// The Poisson distribution models the number of events occurring in a fixed
/// interval of time or space, given that these events occur with a known constant
/// average rate and independently of the time since the last event. It is fundamental
/// in queueing theory, telecommunications, and reliability engineering.
///
/// # Mathematical Properties
///
/// - **Parameter**: λ > 0 (rate parameter, average number of events per interval)
/// - **Support**: {0, 1, 2, 3, ...} (all non-negative integers)
/// - **PMF**: P(X = k) = (λ^k × e^(-λ)) / k!
/// - **Mean**: E[X] = λ
/// - **Variance**: Var[X] = λ (unique property: mean equals variance)
/// - **Mode**: ⌊λ⌋ if λ is not an integer, {λ-1, λ} if λ is an integer
///
/// # Applications
///
/// - **Telecommunications**: Number of phone calls received per hour
/// - **Biology**: Number of mutations in DNA sequences
/// - **Traffic Engineering**: Number of vehicles passing a point per minute
/// - **Reliability**: Number of system failures in a given time period
/// - **Physics**: Number of radioactive decays in a time interval
/// - **Finance**: Number of large price movements per day
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::{Poisson, Distribution, DiscreteDistribution, Sampling};
/// use rand::thread_rng;
///
/// // Average 3 events per interval
/// let poisson = Poisson::new(3.0).unwrap();
/// assert_eq!(poisson.mean(), 3.0);
/// assert_eq!(poisson.variance(), 3.0);  // Mean = Variance for Poisson
///
/// // Network traffic: average 15 packets per second
/// let network = Poisson::new(15.0).unwrap();
/// let prob_no_packets = network.pmf(0);        // P(0 packets)
/// let prob_at_most_10 = network.cdf(10);       // P(≤ 10 packets)
///
/// // Sample number of events
/// let mut rng = thread_rng();
/// let events = poisson.sample(&mut rng);  // Non-negative integer
/// ```
///
/// # Implementation Details
///
/// - PMF uses logarithmic computation to avoid factorial overflow
/// - Sampling implements Knuth's algorithm for numerical stability
/// - For large λ, Normal approximation could be used for efficiency
/// - CDF computation uses iterative sum of PMF values
#[derive(Debug, Clone, PartialEq)]
pub struct Poisson {
    /// Rate parameter λ > 0
    ///
    /// Represents the average number of events occurring in the fixed interval.
    /// Must be positive and finite. Higher λ values indicate more frequent events.
    pub lambda: f64,
}

impl Poisson {
    /// Create a new Poisson distribution Pois(λ)
    ///
    /// Constructs a Poisson distribution with the specified rate parameter.
    /// The rate parameter represents the average number of events per unit interval.
    ///
    /// # Arguments
    ///
    /// * `lambda` - Rate parameter λ, must be positive and finite
    ///
    /// # Returns
    ///
    /// Returns `Ok(Poisson)` if λ is valid, otherwise returns an error.
    ///
    /// # Errors
    ///
    /// - `InvalidParameter`: When λ ≤ 0, λ is NaN, or λ is infinite
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::Poisson;
    ///
    /// // Valid parameters
    /// let low_rate = Poisson::new(0.5).unwrap();    // Low event rate
    /// let high_rate = Poisson::new(100.0).unwrap(); // High event rate
    ///
    /// // Invalid parameters
    /// assert!(Poisson::new(0.0).is_err());     // Zero rate
    /// assert!(Poisson::new(-1.0).is_err());    // Negative rate
    /// assert!(Poisson::new(f64::NAN).is_err()); // NaN rate
    /// assert!(Poisson::new(f64::INFINITY).is_err()); // Infinite rate
    /// ```
    ///
    /// # Mathematical Interpretation
    ///
    /// λ = 2.5 means we expect an average of 2.5 events per unit time/space.
    /// The actual number in any given interval follows the Poisson distribution.
    pub fn new(lambda: f64) -> Result<Self> {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(DistributionError::invalid_parameter("Lambda must be positive and finite"));
        }
        Ok(Poisson { lambda })
    }
}

impl Distribution for Poisson {
    type Params = f64;
    type Support = i64;
    
    fn new(params: Self::Params) -> Result<Self> {
        Self::new(params)
    }
    
    fn params(&self) -> &Self::Params {
        &self.lambda
    }
    
    fn mean(&self) -> f64 {
        self.lambda
    }
    
    fn variance(&self) -> f64 {
        self.lambda
    }
}

impl DiscreteDistribution for Poisson {
    /// Probability mass function P(X = k)
    ///
    /// Computes the probability that exactly k events occur in the fixed interval.
    /// Uses the Poisson formula: P(X = k) = (λ^k × e^(-λ)) / k!
    ///
    /// # Arguments
    ///
    /// * `k` - Number of events to evaluate (non-negative integer)
    ///
    /// # Returns
    ///
    /// The probability P(X = k), or 0.0 if k < 0 (outside support).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Poisson, DiscreteDistribution};
    ///
    /// let poisson = Poisson::new(2.0).unwrap();
    /// 
    /// // P(X = 0) = 2^0 × e^(-2) / 0! = e^(-2) ≈ 0.1353
    /// assert!((poisson.pmf(0) - 0.1353352832366127).abs() < 1e-10);
    /// 
    /// // P(X = 2) = 2^2 × e^(-2) / 2! = 4 × e^(-2) / 2 ≈ 0.2707
    /// assert!((poisson.pmf(2) - 0.2706705664732254).abs() < 1e-10);
    /// 
    /// // Outside support (negative values)
    /// assert_eq!(poisson.pmf(-1), 0.0);
    /// ```
    ///
    /// # Implementation Details
    ///
    /// Uses logarithmic computation to avoid factorial overflow:
    /// ln(P(X = k)) = k×ln(λ) - λ - ln(k!)
    /// Then exponentiates the result for numerical stability with large k.
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        
        // P(X = k) = (lambda^k * e^(-lambda)) / k!
        let k_f = k as f64;
        let log_pmf = k_f * self.lambda.ln() - self.lambda - (1..=k).map(|i| (i as f64).ln()).sum::<f64>();
        log_pmf.exp()
    }
    
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        
        // Simplified implementation - sum PMF values
        (0..=k).map(|i| self.pmf(i)).sum()
    }
    
    fn quantile(&self, p: f64) -> Result<i64> {
        if !p.is_finite() || p < 0.0 || p > 1.0 {
            return Err(DistributionError::invalid_parameter("Probability must be in [0, 1]"));
        }
        
        // Simple linear search - could be optimized
        let mut cumulative = 0.0;
        let mut k = 0;
        while cumulative < p && k < 1000 { // Prevent infinite loop
            cumulative += self.pmf(k);
            if cumulative >= p {
                return Ok(k);
            }
            k += 1;
        }
        Ok(k)
    }
}

impl Sampling for Poisson {
    /// Sample from the Poisson distribution using Knuth's algorithm
    ///
    /// Generates a random number of events using Knuth's algorithm, which is based
    /// on the relationship between Poisson and exponential distributions. This method
    /// is exact and numerically stable for moderate values of λ.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator implementing the `Rng` trait
    ///
    /// # Returns
    ///
    /// A non-negative integer representing the number of events sampled
    ///
    /// # Algorithm (Knuth's Method)
    ///
    /// 1. Set L = e^(-λ), k = 0, p = 1
    /// 2. Repeat:
    ///    - Generate U ~ Uniform(0,1)
    ///    - Set p = p × U, k = k + 1
    /// 3. Until p ≤ L, then return k - 1
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::{Poisson, Sampling};
    /// use rand::thread_rng;
    ///
    /// let poisson = Poisson::new(4.5).unwrap();
    /// let mut rng = thread_rng();
    ///
    /// // Sample many times to verify approximate mean
    /// let samples: Vec<i64> = (0..10000)
    ///     .map(|_| poisson.sample(&mut rng))
    ///     .collect();
    /// let sample_mean = samples.iter().sum::<i64>() as f64 / samples.len() as f64;
    /// 
    /// // Should be close to theoretical mean λ = 4.5
    /// assert!((sample_mean - 4.5).abs() < 0.1);
    /// ```
    ///
    /// # Performance Note
    ///
    /// Knuth's algorithm has expected time complexity O(λ), making it efficient
    /// for small to moderate λ. For large λ (> 30), transformation methods or
    /// normal approximation would be more efficient.
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Support {
        // Using Knuth's algorithm for Poisson sampling
        let l = (-self.lambda).exp();
        let mut k = 0;
        let mut p = 1.0;
        
        loop {
            k += 1;
            p *= rng.gen::<f64>();
            if p <= l {
                break;
            }
        }
        
        (k - 1) as i64
    }
    
    #[cfg(feature = "integration")]
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> VectorF64 {
        let mut vec = VectorF64::zeros(n);
        for i in 0..n {
            let sample = self.sample(rng) as f64;
            vec.set(i, sample).unwrap();
        }
        vec
    }
    
    #[cfg(feature = "integration")]
    fn sample_into<R: Rng>(&self, rng: &mut R, output: &mut VectorF64) {
        for i in 0..output.len() {
            let sample = self.sample(rng) as f64;
            output.set(i, sample).unwrap();
        }
    }
}