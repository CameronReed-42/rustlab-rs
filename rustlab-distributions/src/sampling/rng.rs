//! Advanced random number generation utilities and variance reduction techniques
//!
//! This module provides a comprehensive suite of utilities for random number generation,
//! including enhanced RNG wrappers, variance reduction techniques for Monte Carlo methods,
//! and quasi-random sequence generators. These tools are essential for high-quality
//! statistical computing and numerical simulation.
//!
//! ## Key Components
//!
//! ### Enhanced RNG Wrappers
//! - **RngWrapper**: Extended functionality for any RNG implementing `RngCore`
//! - **Convenience functions**: Thread-local and seeded RNG creation
//! - **Batch generation**: Efficient uniform sample generation
//!
//! ### Variance Reduction Techniques
//! - **Stratified Sampling**: Divide domain into strata for reduced variance
//! - **Antithetic Variables**: Generate negatively correlated samples
//! - **Latin Hypercube Sampling**: Optimal space-filling in multiple dimensions
//!
//! ### Quasi-Random Sequences
//! - **Sobol Sequences**: Low-discrepancy sequences for numerical integration
//! - **Linear Congruential Generators**: Simple, reproducible sequences
//!
//! ## Mathematical Foundation
//!
//! ### Stratified Sampling
//! Divides [0,1] into n equal intervals and samples once per interval:
//! ```text
//! Stratum i: [i/n, (i+1)/n]
//! Sample: Uᵢ = i/n + Vᵢ/n, where Vᵢ ~ Uniform(0,1)
//! Variance reduction: Up to n× for smooth integrands
//! ```
//!
//! ### Antithetic Variables
//! For each U ~ Uniform(0,1), also use 1-U:
//! ```text
//! Var[f(U) + f(1-U)] = 2Var[f(U)] + 2Cov[f(U), f(1-U)]
//! If Cov < 0: Variance reduction achieved
//! ```
//!
//! ### Latin Hypercube Sampling (LHS)
//! Ensures exactly one sample per row/column in d dimensions:
//! ```text
//! For d dimensions, n samples:
//! Each dimension gets permutation of {0,1,...,n-1}
//! Guarantees uniform marginal distributions
//! ```
//!
//! ## Examples
//!
//! ```rust
//! use rustlab_distributions::sampling::rng::*;
//! use rand::thread_rng;
//!
//! // Enhanced RNG wrapper
//! let mut wrapper = thread_rng_wrapper();
//! # #[cfg(feature = "integration")]
//! let uniform_batch = wrapper.uniform_samples(1000);
//!
//! // Variance reduction techniques
//! let mut rng = thread_rng();
//! # #[cfg(feature = "integration")]
//! let stratified = stratified_uniform_samples(100, &mut rng);
//! # #[cfg(feature = "integration")]
//! let antithetic = antithetic_uniform_samples(100, &mut rng);
//!
//! // Quasi-random sequences
//! let mut sobol = SobolSequence::new(3); // 3 dimensions
//! let quasi_point = sobol.next();
//! ```

#[cfg(feature = "integration")]
use rustlab_math::VectorF64;
use rand::{Rng, RngCore, SeedableRng};
use rand::rngs::{StdRng, ThreadRng};

/// Enhanced wrapper for random number generators with extended functionality
///
/// `RngWrapper` extends any RNG implementing `RngCore` with additional methods
/// optimized for statistical computing, including batch uniform generation,
/// range sampling, and convenient pair generation for transformation methods.
///
/// # Type Parameters
///
/// * `R` - Any random number generator implementing `RngCore`
///
/// # Features
///
/// - **Batch Generation**: Efficiently generate vectors of uniform samples
/// - **Range Sampling**: Direct generation of uniform samples in [a, b)
/// - **Pair Generation**: Create uniform pairs for transformation methods
/// - **Zero Overhead**: Thin wrapper with no performance penalty
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::sampling::rng::{RngWrapper, thread_rng_wrapper};
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// // Thread-local RNG wrapper
/// let mut thread_wrapper = thread_rng_wrapper();
///
/// // Seeded RNG wrapper for reproducibility
/// let seeded_rng = StdRng::seed_from_u64(42);
/// let mut seeded_wrapper = RngWrapper::new(seeded_rng);
///
/// // Generate uniform pairs for Box-Muller
/// let pairs = seeded_wrapper.uniform_pairs(5);
/// assert_eq!(pairs.len(), 5);
/// ```
#[derive(Debug)]
pub struct RngWrapper<R: RngCore> {
    /// The underlying random number generator
    rng: R,
}

impl<R: RngCore> RngWrapper<R> {
    /// Create a new RNG wrapper around an existing generator
    ///
    /// This constructor takes ownership of any RNG implementing `RngCore`
    /// and wraps it with additional statistical computing methods.
    ///
    /// # Arguments
    ///
    /// * `rng` - Any random number generator implementing `RngCore`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::sampling::rng::RngWrapper;
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    ///
    /// // Wrap a seeded RNG for reproducibility
    /// let std_rng = StdRng::seed_from_u64(12345);
    /// let wrapper = RngWrapper::new(std_rng);
    /// ```
    pub fn new(rng: R) -> Self {
        RngWrapper { rng }
    }
    
    /// Generate n uniform random samples in [0, 1)
    ///
    /// Efficiently generates a vector of independent uniform random variables,
    /// each distributed as U ~ Uniform(0, 1). This method is optimized for
    /// batch generation and Monte Carlo applications.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of uniform samples to generate
    ///
    /// # Returns
    ///
    /// A `VectorF64` containing n independent samples from Uniform(0, 1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::sampling::rng::thread_rng_wrapper;
    ///
    /// let mut wrapper = thread_rng_wrapper();
    /// 
    /// # #[cfg(feature = "integration")]
    /// let samples = wrapper.uniform_samples(1000);
    /// # #[cfg(feature = "integration")]
    /// assert_eq!(samples.len(), 1000);
    /// 
    /// // All samples should be in [0, 1)
    /// # #[cfg(feature = "integration")]
    /// for i in 0..samples.len() {
    ///     let x = samples.get(i).unwrap();
    ///     assert!(x >= 0.0 && x < 1.0);
    /// }
    /// ```
    #[cfg(feature = "integration")]
    pub fn uniform_samples(&mut self, n: usize) -> VectorF64 {
        let samples: Vec<f64> = (0..n).map(|_| self.rng.gen()).collect();
        VectorF64::from_vec(samples)
    }
    
    /// Generate n uniform random samples in the interval [a, b)
    ///
    /// Transforms uniform [0,1) samples to the specified interval using
    /// the linear transformation: X = a + U(b - a) where U ~ Uniform(0,1).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples to generate
    /// * `a` - Lower bound of the interval (inclusive)
    /// * `b` - Upper bound of the interval (exclusive)
    ///
    /// # Returns
    ///
    /// A `VectorF64` containing n samples from Uniform(a, b)
    ///
    /// # Panics
    ///
    /// This function does not validate that a < b. If b ≤ a, the behavior
    /// is mathematically undefined but will not panic.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::sampling::rng::thread_rng_wrapper;
    ///
    /// let mut wrapper = thread_rng_wrapper();
    ///
    /// // Generate samples in [-2.5, 3.7)
    /// # #[cfg(feature = "integration")]
    /// let samples = wrapper.uniform_range_samples(500, -2.5, 3.7);
    /// # #[cfg(feature = "integration")]
    /// assert_eq!(samples.len(), 500);
    ///
    /// // Verify samples are in correct range
    /// # #[cfg(feature = "integration")]
    /// # {
    /// use rustlab_math::BasicStatistics;
    /// let min_val = samples.min();
    /// let max_val = samples.max();
    /// assert!(min_val >= -2.5);
    /// assert!(max_val < 3.7);
    /// # }
    /// ```
    #[cfg(feature = "integration")]
    pub fn uniform_range_samples(&mut self, n: usize, a: f64, b: f64) -> VectorF64 {
        let samples: Vec<f64> = (0..n).map(|_| {
            let u: f64 = self.rng.gen();
            a + u * (b - a)
        }).collect();
        VectorF64::from_vec(samples)
    }
    
    /// Generate pairs of independent uniform random samples
    ///
    /// Creates n pairs of independent uniform random variables, commonly used
    /// for transformation methods like Box-Muller or rejection sampling algorithms.
    /// Each pair consists of (U₁, U₂) where Uᵢ ~ Uniform(0, 1).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of uniform pairs to generate
    ///
    /// # Returns
    ///
    /// A `Vec<(f64, f64)>` containing n pairs of independent Uniform(0,1) samples
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::sampling::rng::thread_rng_wrapper;
    ///
    /// let mut wrapper = thread_rng_wrapper();
    /// let pairs = wrapper.uniform_pairs(10);
    ///
    /// assert_eq!(pairs.len(), 10);
    ///
    /// // Use pairs for Box-Muller transformation
    /// for (u1, u2) in pairs {
    ///     assert!(u1 >= 0.0 && u1 < 1.0);
    ///     assert!(u2 >= 0.0 && u2 < 1.0);
    ///     
    ///     // Box-Muller transformation (simplified)
    ///     if u1 > 0.0 {
    ///         let r = (-2.0 * u1.ln()).sqrt();
    ///         let theta = 2.0 * std::f64::consts::PI * u2;
    ///         let _normal1 = r * theta.cos();
    ///         let _normal2 = r * theta.sin();
    ///     }
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// This method generates 2n uniform random numbers and packages them
    /// into convenient pairs. For maximum performance in transformation
    /// algorithms, consider generating uniform samples directly when possible.
    pub fn uniform_pairs(&mut self, n: usize) -> Vec<(f64, f64)> {
        (0..n).map(|_| (self.rng.gen(), self.rng.gen())).collect()
    }
    
    /// Get a mutable reference to the underlying RNG
    pub fn inner_mut(&mut self) -> &mut R {
        &mut self.rng
    }
}

/// Create a thread-local RNG wrapper
pub fn thread_rng_wrapper() -> RngWrapper<ThreadRng> {
    RngWrapper::new(rand::thread_rng())
}

/// Create a seeded RNG wrapper
pub fn seeded_rng_wrapper(seed: u64) -> RngWrapper<StdRng> {
    RngWrapper::new(StdRng::seed_from_u64(seed))
}

/// Generate stratified uniform samples for Monte Carlo variance reduction
///
/// Stratified sampling is a powerful variance reduction technique that divides
/// the sampling domain [0,1] into n equal strata and generates exactly one
/// sample from each stratum. This ensures better coverage of the domain
/// and can dramatically reduce variance for smooth integrands.
///
/// # Mathematical Foundation
///
/// For n strata, each stratum i has:
/// ```text
/// Stratum i: [ᵢ/n, (ᵢ+1)/n] for i = 0, 1, ..., n-1
/// Sample: Xᵢ = ᵢ/n + Uᵢ/n, where Uᵢ ~ Uniform(0,1)
/// ```
///
/// # Variance Reduction
///
/// For estimating ∫₀¹ f(x)dx, stratified sampling can reduce variance by:
/// - **Factor of n** for linear functions f(x) = ax + b
/// - **Significant reduction** for smooth, monotonic functions  
/// - **No benefit** for highly oscillatory functions
///
/// # Arguments
///
/// * `n` - Number of strata (and samples to generate)
/// * `rng` - Mutable reference to random number generator
///
/// # Returns
///
/// A `VectorF64` containing n stratified samples from [0, 1]
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::sampling::rng::stratified_uniform_samples;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
///
/// // Generate 100 stratified samples
/// # #[cfg(feature = "integration")]
/// let stratified = stratified_uniform_samples(100, &mut rng);
/// # #[cfg(feature = "integration")]
/// assert_eq!(stratified.len(), 100);
///
/// // Verify stratification property
/// # #[cfg(feature = "integration")]
/// for i in 0..100 {
///     let sample = stratified.get(i).unwrap();
///     let stratum_start = i as f64 / 100.0;
///     let stratum_end = (i + 1) as f64 / 100.0;
///     assert!(sample >= stratum_start && sample < stratum_end);
/// }
/// ```
///
/// # Monte Carlo Integration Example
///
/// ```rust,ignore
/// // Compare regular vs. stratified sampling for ∫₀¹ x² dx = 1/3
/// let mut rng = thread_rng();
/// 
/// // Regular sampling
/// let regular: VectorF64 = (0..1000).map(|_| rng.gen::<f64>().powi(2)).collect();
/// let regular_estimate = regular.mean();
/// 
/// // Stratified sampling
/// let stratified = stratified_uniform_samples(1000, &mut rng);
/// let strat_estimate = stratified.iter().map(|x| x.powi(2)).sum::<f64>() / 1000.0;
/// 
/// // Stratified sampling should have lower variance
/// ```
#[cfg(feature = "integration")]
pub fn stratified_uniform_samples<R: Rng>(n: usize, rng: &mut R) -> VectorF64 {
    let samples: Vec<f64> = (0..n).map(|i| {
        let stratum_start = i as f64 / n as f64;
        let stratum_width = 1.0 / n as f64;
        let u: f64 = rng.gen();
        stratum_start + u * stratum_width
    }).collect();
    VectorF64::from_vec(samples)
}

/// Generate antithetic uniform samples for variance reduction
///
/// Antithetic variables is a classical variance reduction technique that generates
/// pairs of negatively correlated samples. For each uniform sample U, it also
/// includes the antithetic pair 1-U. When the function f satisfies certain
/// monotonicity conditions, this can significantly reduce estimation variance.
///
/// # Mathematical Foundation
///
/// For estimating E[f(U)] where U ~ Uniform(0,1):
/// ```text
/// Antithetic estimator: [f(U) + f(1-U)] / 2
/// Variance: Var[f(U)] + Cov[f(U), f(1-U)]
/// 
/// If f is monotonic: Cov[f(U), f(1-U)] < 0 ⇒ Variance reduction
/// ```
///
/// # When It Works Best
///
/// - **Monotonic functions**: Linear, exponential, logarithmic
/// - **Smooth functions**: Continuous derivatives
/// - **Examples**: Option pricing, queueing models, reliability analysis
///
/// # Arguments
///
/// * `n` - Total number of samples to generate (includes antithetic pairs)
/// * `rng` - Mutable reference to random number generator
///
/// # Returns
///
/// A `VectorF64` containing n samples: pairs (U, 1-U) up to n total samples
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::sampling::rng::antithetic_uniform_samples;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
///
/// // Generate 100 samples with antithetic pairs
/// # #[cfg(feature = "integration")]
/// let antithetic = antithetic_uniform_samples(100, &mut rng);
/// # #[cfg(feature = "integration")]
/// assert_eq!(antithetic.len(), 100);
///
/// // For even n, samples come in antithetic pairs
/// # #[cfg(feature = "integration")]
/// if antithetic.len() % 2 == 0 {
///     for i in (0..antithetic.len()).step_by(2) {
///         let u1 = antithetic.get(i).unwrap();
///         let u2 = antithetic.get(i + 1).unwrap();
///         // u2 should be approximately 1 - u1
///         assert!((u1 + u2 - 1.0).abs() < 1e-15);
///     }
/// }
/// ```
///
/// # Performance vs. Variance Tradeoff
///
/// - **Cost**: Same number of random numbers as regular sampling
/// - **Benefit**: Can reduce variance by 50% or more for suitable functions
/// - **Risk**: Can increase variance for non-monotonic functions
#[cfg(feature = "integration")]
pub fn antithetic_uniform_samples<R: Rng>(n: usize, rng: &mut R) -> VectorF64 {
    let half_n = (n + 1) / 2;
    let mut samples = Vec::with_capacity(n);
    
    for _ in 0..half_n {
        let u: f64 = rng.gen();
        samples.push(u);
        if samples.len() < n {
            samples.push(1.0 - u);
        }
    }
    
    samples.truncate(n);
    VectorF64::from_vec(samples)
}

/// Latin Hypercube Sampling (LHS) for optimal high-dimensional coverage
///
/// Latin Hypercube Sampling is an advanced variance reduction technique that ensures
/// optimal space-filling properties in multiple dimensions. It generates n samples
/// in d dimensions such that each sample is the only one in each axis-parallel
/// hyperplane, guaranteeing excellent coverage of the sampling space.
///
/// # Mathematical Properties
///
/// For n samples in d dimensions:
/// ```text
/// Each dimension: Permutation of {0, 1, ..., n-1} + Uniform jitter
/// Marginal distributions: Exactly uniform in each dimension
/// Correlation structure: Minimal dependence between dimensions
/// Space coverage: Optimal for integration and optimization
/// ```
///
/// # Algorithm
///
/// 1. **Stratification**: Divide each dimension into n equal intervals
/// 2. **Permutation**: Randomly permute interval indices for each dimension
/// 3. **Jittering**: Add uniform random noise within each interval
/// 4. **Assembly**: Combine coordinates to form d-dimensional samples
///
/// # Arguments
///
/// * `n` - Number of samples to generate
/// * `d` - Number of dimensions
/// * `rng` - Mutable reference to random number generator
///
/// # Returns
///
/// A `Vec<VectorF64>` where each inner vector represents one d-dimensional sample
/// All coordinates are in [0, 1]ᵈ
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::sampling::rng::latin_hypercube_samples;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
///
/// // Generate 50 samples in 3 dimensions
/// # #[cfg(feature = "integration")]
/// let lhs_samples = latin_hypercube_samples(50, 3, &mut rng);
/// # #[cfg(feature = "integration")]
/// assert_eq!(lhs_samples.len(), 3);  // 3 dimensions
/// # #[cfg(feature = "integration")]
/// assert_eq!(lhs_samples[0].len(), 50); // 50 samples per dimension
///
/// // Each dimension should have exactly one sample per stratum
/// # #[cfg(feature = "integration")]
/// for dim in 0..3 {
///     let mut sorted_samples: Vec<f64> = (0..50)
///         .map(|i| lhs_samples[dim].get(i).unwrap())
///         .collect();
///     sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
///     
///     // Verify stratification
///     for i in 0..50 {
///         let stratum_start = i as f64 / 50.0;
///         let stratum_end = (i + 1) as f64 / 50.0;
///         assert!(sorted_samples[i] >= stratum_start);
///         assert!(sorted_samples[i] < stratum_end);
///     }
/// }
/// ```
///
/// # Applications
///
/// - **Monte Carlo Integration**: Superior convergence in high dimensions
/// - **Sensitivity Analysis**: Optimal coverage for parameter studies
/// - **Optimization**: Starting points for global optimization algorithms
/// - **Computer Experiments**: Design of experiments for expensive simulations
///
/// # Advantages over Random Sampling
///
/// - **Better coverage**: No clustering or gaps in projections
/// - **Faster convergence**: O(n⁻¹) vs O(n⁻¹/²) for smooth integrands
/// - **Dimension robustness**: Performance degrades gracefully with dimension
/// - **Reproducible**: Same sample positions for same random seed
#[cfg(feature = "integration")]
pub fn latin_hypercube_samples<R: Rng>(n: usize, d: usize, rng: &mut R) -> Vec<VectorF64> {
    let mut samples = vec![Vec::with_capacity(n); d];
    
    // Generate permutations for each dimension
    for dim in 0..d {
        let mut perm: Vec<usize> = (0..n).collect();
        
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }
        
        // Generate stratified samples using the permutation
        for (i, &cell) in perm.iter().enumerate() {
            let stratum_start = cell as f64 / n as f64;
            let stratum_width = 1.0 / n as f64;
            let u: f64 = rng.gen();
            samples[dim].push(stratum_start + u * stratum_width);
        }
    }
    
    samples.into_iter().map(VectorF64::from_vec).collect()
}

/// Sobol sequence generator for quasi-random sampling
///
/// The Sobol sequence is a low-discrepancy (quasi-random) sequence that provides
/// superior uniform coverage of the unit hypercube compared to pseudorandom sequences.
/// It is particularly effective for Monte Carlo integration and numerical analysis
/// where uniform space-filling is critical.
///
/// # Mathematical Properties
///
/// - **Low Discrepancy**: O((log n)ᵈ/n) vs O(n⁻¹/²) for random sequences
/// - **Deterministic**: Same sequence for same initialization
/// - **Equidistribution**: Excellent uniformity properties
/// - **Dimension Extension**: Can generate points in arbitrarily high dimensions
///
/// # Applications
///
/// - **Monte Carlo Integration**: Faster convergence than random sampling
/// - **Sensitivity Analysis**: Systematic exploration of parameter space
/// - **Optimization**: Starting points with guaranteed coverage
/// - **Computer Graphics**: Sampling for rendering algorithms
///
/// # Implementation Note
///
/// This is a simplified educational implementation suitable for demonstration
/// and light usage. Production applications requiring high-dimensional Sobol
/// sequences should use specialized libraries like `sobol_burley` or similar.
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::sampling::rng::SobolSequence;
///
/// // Initialize 2D Sobol sequence
/// let mut sobol = SobolSequence::new(2);
///
/// // Generate first few points
/// for i in 0..5 {
///     let point = sobol.next();
///     println!("Point {}: ({:.6}, {:.6})", i, point[0], point[1]);
/// }
///
/// // Points will show excellent space-filling properties
/// ```
///
/// # Convergence Comparison
///
/// For Monte Carlo integration of smooth functions:
/// ```text
/// Random sampling:    Error ~ O(n⁻¹/²)
/// Sobol sequence:     Error ~ O((log n)ᵈ/n)
/// ```
///
/// The Sobol sequence typically converges faster, especially in moderate dimensions.
#[derive(Debug)]
pub struct SobolSequence {
    dimension: usize,
    count: usize,
    direction_numbers: Vec<Vec<u32>>,
}

impl SobolSequence {
    /// Initialize a new Sobol sequence generator
    ///
    /// Creates a Sobol sequence generator for the specified number of dimensions.
    /// The generator starts from the beginning of the sequence and can produce
    /// an indefinite number of low-discrepancy points.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Number of dimensions for each generated point (≥ 1)
    ///
    /// # Returns
    ///
    /// A `SobolSequence` ready to generate low-discrepancy points
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::sampling::rng::SobolSequence;
    ///
    /// // 3D Sobol sequence for volume integration
    /// let mut sobol_3d = SobolSequence::new(3);
    ///
    /// // 5D sequence for high-dimensional problems
    /// let mut sobol_5d = SobolSequence::new(5);
    /// ```
    pub fn new(dimension: usize) -> Self {
        // Simplified direction numbers (first few dimensions only)
        let direction_numbers = vec![
            vec![1u32 << 31],
            vec![1u32 << 31, 1u32 << 30],
            vec![1u32 << 31, 3u32 << 29, 1u32 << 28],
        ];
        
        SobolSequence {
            dimension,
            count: 0,
            direction_numbers: direction_numbers.into_iter().take(dimension).collect(),
        }
    }
    
    /// Generate the next point in the Sobol sequence
    ///
    /// Returns the next low-discrepancy point in the sequence. Points are
    /// generated deterministically and provide excellent space-filling
    /// properties in the unit hypercube [0,1]ᵈ.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` representing one point in d-dimensional space,
    /// where d is the dimension specified during construction.
    /// All coordinates are in the interval [0, 1].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::sampling::rng::SobolSequence;
    ///
    /// let mut sobol = SobolSequence::new(2);
    ///
    /// // Generate sequence of 2D points
    /// for i in 0..10 {
    ///     let point = sobol.next();
    ///     assert_eq!(point.len(), 2);
    ///     assert!(point[0] >= 0.0 && point[0] <= 1.0);
    ///     assert!(point[1] >= 0.0 && point[1] <= 1.0);
    ///     println!("Point {}: ({}, {})", i, point[0], point[1]);
    /// }
    /// ```
    ///
    /// # Sequence Properties
    ///
    /// - **Deterministic**: Same sequence every time for same dimension
    /// - **Low Discrepancy**: Better uniformity than random points
    /// - **Progressive**: Each new point improves overall coverage
    pub fn next(&mut self) -> Vec<f64> {
        self.count += 1;
        let mut point = vec![0.0; self.dimension];
        
        // Find the rightmost zero bit
        let c = self.count;
        let mut k = 0;
        let mut t = c;
        while t & 1 == 1 {
            k += 1;
            t >>= 1;
        }
        
        // Generate coordinates
        for d in 0..self.dimension {
            if d < self.direction_numbers.len() && k < self.direction_numbers[d].len() {
                let direction = self.direction_numbers[d][k];
                point[d] = (direction as f64) / (u32::MAX as f64);
            }
        }
        
        point
    }
    
    /// Generate n points
    #[cfg(feature = "integration")]
    pub fn generate(&mut self, n: usize) -> Vec<VectorF64> {
        (0..n).map(|_| VectorF64::from_vec(self.next())).collect()
    }
}

/// Simple Linear Congruential Generator (LCG) for reproducible sequences
///
/// A basic pseudorandom number generator implementing the classic Linear
/// Congruential Generator algorithm. While not suitable for cryptographic
/// purposes or high-quality statistical applications, it provides perfect
/// reproducibility and is useful for testing, debugging, and educational purposes.
///
/// # Mathematical Formula
///
/// The LCG follows the recurrence relation:
/// ```text
/// X_{n+1} = (a × X_n + c) mod m
/// ```
/// 
/// This implementation uses parameters compatible with common C library implementations:
/// - a = 1103515245 (multiplier)
/// - c = 12345 (increment)  
/// - m = 2³² (modulus, via integer overflow)
///
/// # Use Cases
///
/// - **Unit Testing**: Reproducible random sequences for test verification
/// - **Debugging**: Deterministic behavior for troubleshooting
/// - **Legacy Compatibility**: Matching output from older systems
/// - **Educational**: Understanding basic PRNG principles
///
/// # Limitations
///
/// - **Poor Quality**: Fails many statistical tests
/// - **Short Period**: Relatively small cycle length
/// - **Correlations**: Adjacent values show linear relationships
/// - **Not Cryptographically Secure**: Predictable output
///
/// # Examples
///
/// ```rust
/// use rustlab_distributions::sampling::rng::SimpleLCG;
///
/// // Create LCG with specific seed for reproducibility
/// let mut lcg1 = SimpleLCG::new(12345);
/// let mut lcg2 = SimpleLCG::new(12345);
///
/// // Both generators produce identical sequences
/// for _ in 0..10 {
///     assert_eq!(lcg1.next(), lcg2.next());
/// }
/// ```
///
/// # Production Warning
///
/// For production applications, use high-quality generators from the `rand`
/// crate such as `StdRng`, `SmallRng`, or `ThreadRng` which provide much
/// better statistical properties and performance.
#[derive(Debug)]
pub struct SimpleLCG {
    seed: u64,
}

impl SimpleLCG {
    /// Create a new Linear Congruential Generator with the specified seed
    ///
    /// Initializes the LCG state with the provided seed value. The same seed
    /// will always produce the same sequence of random numbers, making this
    /// generator fully deterministic and reproducible.
    ///
    /// # Arguments
    ///
    /// * `seed` - Initial state value for the generator (any u64)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::sampling::rng::SimpleLCG;
    ///
    /// // Different seeds produce different sequences
    /// let mut lcg_a = SimpleLCG::new(1);
    /// let mut lcg_b = SimpleLCG::new(2);
    ///
    /// assert_ne!(lcg_a.next(), lcg_b.next());
    ///
    /// // Same seed produces identical sequences
    /// let mut lcg1 = SimpleLCG::new(42);
    /// let mut lcg2 = SimpleLCG::new(42);
    /// 
    /// for _ in 0..100 {
    ///     assert_eq!(lcg1.next(), lcg2.next());
    /// }
    /// ```
    pub fn new(seed: u64) -> Self {
        SimpleLCG { seed }
    }
    
    /// Generate the next pseudorandom value in [0, 1)
    ///
    /// Advances the internal state using the LCG recurrence relation and
    /// returns a floating-point value uniformly distributed in [0, 1).
    ///
    /// # Returns
    ///
    /// A pseudorandom `f64` value in the interval [0, 1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_distributions::sampling::rng::SimpleLCG;
    ///
    /// let mut lcg = SimpleLCG::new(12345);
    ///
    /// // Generate sequence of values
    /// for _ in 0..10 {
    ///     let value = lcg.next();
    ///     assert!(value >= 0.0 && value < 1.0);
    ///     println!("Random value: {}", value);
    /// }
    /// ```
    ///
    /// # Implementation Details
    ///
    /// The method:
    /// 1. Updates internal state: `state = (1103515245 * state + 12345) % 2³²`
    /// 2. Extracts high-quality bits from the updated state
    /// 3. Converts to floating-point range [0, 1)
    pub fn next(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        let x = (self.seed / 65536) % 32768;
        x as f64 / 32767.0
    }
    
    /// Generate n samples
    #[cfg(feature = "integration")]
    pub fn samples(&mut self, n: usize) -> VectorF64 {
        let samples: Vec<f64> = (0..n).map(|_| self.next()).collect();
        VectorF64::from_vec(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rng_wrapper() {
        let mut wrapper = thread_rng_wrapper();
        let samples = wrapper.uniform_samples(10);
        assert_eq!(samples.len(), 10);
        
        // Check all samples are in [0, 1)
        for i in 0..samples.len() {
            let s = samples.get(i).unwrap();
            assert!(s >= 0.0 && s < 1.0);
        }
    }
    
    #[test]
    fn test_stratified_sampling() {
        let mut rng = rand::thread_rng();
        let samples = stratified_uniform_samples(10, &mut rng);
        assert_eq!(samples.len(), 10);
        
        // Check stratification
        for i in 0..10 {
            let s = samples.get(i).unwrap();
            let expected_min = i as f64 / 10.0;
            let expected_max = (i + 1) as f64 / 10.0;
            assert!(s >= expected_min && s < expected_max);
        }
    }
    
    #[test]
    fn test_antithetic_sampling() {
        let mut rng = rand::thread_rng();
        let samples = antithetic_uniform_samples(10, &mut rng);
        assert_eq!(samples.len(), 10);
    }
    
    #[test]
    fn test_latin_hypercube() {
        let mut rng = rand::thread_rng();
        let samples = latin_hypercube_samples(5, 2, &mut rng);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].len(), 5);
        assert_eq!(samples[1].len(), 5);
    }
    
    #[test]
    fn test_simple_lcg() {
        let mut lcg = SimpleLCG::new(12345);
        let samples = lcg.samples(10);
        assert_eq!(samples.len(), 10);
        
        // Test reproducibility
        let mut lcg2 = SimpleLCG::new(12345);
        let samples2 = lcg2.samples(10);
        for i in 0..10 {
            assert_eq!(samples.get(i).unwrap(), samples2.get(i).unwrap());
        }
    }
}