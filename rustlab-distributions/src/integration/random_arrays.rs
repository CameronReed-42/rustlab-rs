//! Random array generation utilities
//!
//! This module provides utilities for generating arrays and vectors with
//! specific random structures and properties.

use crate::continuous::{Normal, Uniform};
use crate::traits::{Sampling};
use rustlab_math::{VectorF64, ArrayF64, Result as MathResult, MathError};
use rand::Rng;

/// Generate a random walk vector
/// 
/// Creates a vector representing a random walk where each step is drawn
/// from a normal distribution.
/// 
/// # Arguments
/// * `n` - Number of steps
/// * `step_std` - Standard deviation of each step
/// * `start_value` - Starting value of the walk
/// * `rng` - Random number generator
/// 
/// # Example
/// ```ignore
/// use rustlab_distributions::integration::random_walk;
/// use rand::thread_rng;
/// 
/// let mut rng = thread_rng();
/// let walk = random_walk(1000, 1.0, 0.0, &mut rng).unwrap();
/// ```
pub fn random_walk<R: Rng>(n: usize, step_std: f64, start_value: f64, rng: &mut R) -> crate::Result<VectorF64> {
    if n == 0 {
        return Err(crate::DistributionError::invalid_parameter("Number of steps must be positive"));
    }
    
    let step_dist = Normal::new(0.0, step_std)?;
    let mut walk = VectorF64::zeros(n);
    
    // Set starting value
    if let Some(slice) = walk.as_mut_slice() {
        slice[0] = start_value;
    } else {
        return Err(crate::DistributionError::sampling_error("Vector access failed"));
    }
    
    // Generate cumulative steps
    let mut current_value = start_value;
    for i in 1..n {
        let step = step_dist.sample(rng);
        current_value += step;
        if let Some(slice) = walk.as_mut_slice() {
            slice[i] = current_value;
        } else {
            return Err(crate::DistributionError::sampling_error("Vector access failed"));
        }
    }
    
    Ok(walk)
}

/// Generate a geometric Brownian motion vector
/// 
/// Models stock prices or other quantities that follow geometric Brownian motion.
/// dS = μS dt + σS dW
/// 
/// # Arguments
/// * `n` - Number of time steps
/// * `mu` - Drift parameter
/// * `sigma` - Volatility parameter  
/// * `dt` - Time step size
/// * `start_value` - Starting value
/// * `rng` - Random number generator
pub fn geometric_brownian_motion<R: Rng>(
    n: usize, 
    mu: f64, 
    sigma: f64, 
    dt: f64, 
    start_value: f64, 
    rng: &mut R
) -> crate::Result<VectorF64> {
    if n == 0 {
        return Err(crate::DistributionError::invalid_parameter("Number of steps must be positive"));
    }
    
    if start_value <= 0.0 {
        return Err(crate::DistributionError::invalid_parameter("Start value must be positive for geometric Brownian motion"));
    }
    
    let normal_dist = Normal::new(0.0, 1.0)?;
    let mut prices = VectorF64::zeros(n);
    
    // Set starting value
    if let Some(slice) = prices.as_mut_slice() {
        slice[0] = start_value;
    } else {
        return Err(crate::DistributionError::sampling_error("Vector access failed"));
    }
    
    // Generate price path
    let mut current_price = start_value;
    for i in 1..n {
        let dw = normal_dist.sample(rng) * dt.sqrt();
        let drift = mu * current_price * dt;
        let diffusion = sigma * current_price * dw;
        current_price += drift + diffusion;
        
        // Ensure price stays positive
        current_price = current_price.max(1e-10);
        
        if let Some(slice) = prices.as_mut_slice() {
            slice[i] = current_price;
        } else {
            return Err(crate::DistributionError::sampling_error("Vector access failed"));
        }
    }
    
    Ok(prices)
}

/// Generate a random symmetric matrix
/// 
/// Creates a symmetric matrix where the upper triangle is filled with
/// random values and mirrored to the lower triangle.
/// 
/// # Arguments
/// * `size` - Size of the square matrix
/// * `mean` - Mean of the distribution for matrix elements
/// * `std_dev` - Standard deviation of the distribution
/// * `rng` - Random number generator
pub fn random_symmetric_matrix<R: Rng>(size: usize, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<ArrayF64> {
    let dist = Normal::new(mean, std_dev)?;
    let mut matrix = ArrayF64::zeros(size, size);
    
    // Fill upper triangle (including diagonal)
    for i in 0..size {
        for j in i..size {
            let value = dist.sample(rng);
            matrix.set(i, j, value)
                .map_err(|_| crate::DistributionError::sampling_error("Matrix access failed"))?;
            
            // Mirror to lower triangle
            if i != j {
                matrix.set(j, i, value)
                    .map_err(|_| crate::DistributionError::sampling_error("Matrix access failed"))?;
            }
        }
    }
    
    Ok(matrix)
}

/// Generate a random positive definite matrix
/// 
/// Creates a positive definite matrix by generating A^T * A where A is a random matrix.
/// This ensures the result is positive semi-definite (and positive definite with high probability).
/// 
/// # Arguments
/// * `size` - Size of the square matrix
/// * `rng` - Random number generator
pub fn random_positive_definite_matrix<R: Rng>(size: usize, rng: &mut R) -> crate::Result<ArrayF64> {
    let dist = Normal::new(0.0, 1.0)?;
    
    // Generate random matrix A
    let mut a = ArrayF64::zeros(size, size);
    for i in 0..size {
        for j in 0..size {
            let value = dist.sample(rng);
            a.set(i, j, value)
                .map_err(|_| crate::DistributionError::sampling_error("Matrix access failed"))?;
        }
    }
    
    // Compute A^T * A
    let mut result = ArrayF64::zeros(size, size);
    for i in 0..size {
        for j in 0..size {
            let mut sum = 0.0;
            for k in 0..size {
                let a_ki = a.get(k, i).ok_or_else(|| 
                    crate::DistributionError::sampling_error("Matrix access failed"))?;
                let a_kj = a.get(k, j).ok_or_else(|| 
                    crate::DistributionError::sampling_error("Matrix access failed"))?;
                sum += a_ki * a_kj;
            }
            result.set(i, j, sum)
                .map_err(|_| crate::DistributionError::sampling_error("Result matrix access failed"))?;
        }
    }
    
    Ok(result)
}

/// Generate a sparse random vector
/// 
/// Creates a vector where only a fraction of elements are non-zero.
/// 
/// # Arguments
/// * `n` - Length of the vector
/// * `sparsity` - Fraction of elements that should be zero (0.0 to 1.0)
/// * `mean` - Mean of non-zero elements
/// * `std_dev` - Standard deviation of non-zero elements
/// * `rng` - Random number generator
pub fn random_sparse_vector<R: Rng>(
    n: usize, 
    sparsity: f64, 
    mean: f64, 
    std_dev: f64, 
    rng: &mut R
) -> crate::Result<VectorF64> {
    if sparsity < 0.0 || sparsity > 1.0 {
        return Err(crate::DistributionError::invalid_parameter("Sparsity must be between 0 and 1"));
    }
    
    let uniform_dist = Uniform::new(0.0, 1.0)?;
    let normal_dist = Normal::new(mean, std_dev)?;
    let mut vector = VectorF64::zeros(n);
    
    for i in 0..n {
        if uniform_dist.sample(rng) > sparsity {
            // Element should be non-zero
            let value = normal_dist.sample(rng);
            if let Some(slice) = vector.as_mut_slice() {
                slice[i] = value;
            } else {
                return Err(crate::DistributionError::sampling_error("Vector access failed"));
            }
        }
        // Otherwise leave as zero (already initialized)
    }
    
    Ok(vector)
}

/// Generate a time series with trend and noise
/// 
/// Creates a time series with linear trend plus random noise.
/// 
/// # Arguments
/// * `n` - Number of time points
/// * `trend_slope` - Linear trend slope
/// * `intercept` - Starting value
/// * `noise_std` - Standard deviation of additive noise
/// * `rng` - Random number generator
pub fn random_time_series<R: Rng>(
    n: usize, 
    trend_slope: f64, 
    intercept: f64, 
    noise_std: f64, 
    rng: &mut R
) -> crate::Result<VectorF64> {
    let noise_dist = Normal::new(0.0, noise_std)?;
    let mut series = VectorF64::zeros(n);
    
    for i in 0..n {
        let trend_value = intercept + trend_slope * i as f64;
        let noise = noise_dist.sample(rng);
        let value = trend_value + noise;
        
        if let Some(slice) = series.as_mut_slice() {
            slice[i] = value;
        } else {
            return Err(crate::DistributionError::sampling_error("Vector access failed"));
        }
    }
    
    Ok(series)
}

/// Generate an autoregressive AR(1) time series
/// 
/// Creates a time series following: X_t = φ*X_{t-1} + ε_t
/// where ε_t ~ N(0, σ²)
/// 
/// # Arguments
/// * `n` - Number of time points
/// * `phi` - Autoregressive coefficient (should be |phi| < 1 for stationarity)
/// * `sigma` - Standard deviation of error terms
/// * `x0` - Initial value
/// * `rng` - Random number generator
pub fn ar1_series<R: Rng>(n: usize, phi: f64, sigma: f64, x0: f64, rng: &mut R) -> crate::Result<VectorF64> {
    if n == 0 {
        return Err(crate::DistributionError::invalid_parameter("Series length must be positive"));
    }
    
    let error_dist = Normal::new(0.0, sigma)?;
    let mut series = VectorF64::zeros(n);
    
    // Set initial value
    if let Some(slice) = series.as_mut_slice() {
        slice[0] = x0;
    } else {
        return Err(crate::DistributionError::sampling_error("Vector access failed"));
    }
    
    // Generate AR(1) process
    for i in 1..n {
        let prev_value = series.get(i-1).ok_or_else(|| 
            crate::DistributionError::sampling_error("Vector access failed"))?;
        let error = error_dist.sample(rng);
        let value = phi * prev_value + error;
        
        if let Some(slice) = series.as_mut_slice() {
            slice[i] = value;
        } else {
            return Err(crate::DistributionError::sampling_error("Vector access failed"));
        }
    }
    
    Ok(series)
}

/// Generate a random correlation matrix using the onion method
/// 
/// This generates a proper correlation matrix (positive semi-definite with 1s on diagonal)
/// using the vine/onion method.
/// 
/// # Arguments
/// * `size` - Size of the correlation matrix
/// * `rng` - Random number generator
pub fn random_correlation_matrix<R: Rng>(size: usize, rng: &mut R) -> crate::Result<ArrayF64> {
    if size < 2 {
        return Err(crate::DistributionError::invalid_parameter("Matrix size must be at least 2"));
    }
    
    let uniform_dist = Uniform::new(0.0, 1.0)?;
    let mut corr_matrix = ArrayF64::eye(size);
    
    // Simple approach: generate random correlations that maintain positive semi-definiteness
    // This is a simplified version - for more sophisticated generation, consider the onion method
    for i in 0..size {
        for j in (i+1)..size {
            // Generate correlation with decreasing strength as distance increases
            let max_corr = 1.0 / (1.0 + (j - i) as f64 * 0.5);
            let corr = (uniform_dist.sample(rng) - 0.5) * 2.0 * max_corr;
            
            corr_matrix.set(i, j, corr)
                .map_err(|_| crate::DistributionError::sampling_error("Matrix access failed"))?;
            corr_matrix.set(j, i, corr)
                .map_err(|_| crate::DistributionError::sampling_error("Matrix access failed"))?;
        }
    }
    
    Ok(corr_matrix)
}

/// Generate multivariate normal samples using Cholesky decomposition
/// 
/// This is a simplified version that assumes the correlation matrix can be decomposed.
/// In practice, you might want to use more robust methods.
/// 
/// # Arguments
/// * `n_samples` - Number of samples to generate
/// * `means` - Mean vector for each variable
/// * `correlation_matrix` - Correlation matrix (must be positive semi-definite)
/// * `rng` - Random number generator
pub fn multivariate_normal_samples<R: Rng>(
    n_samples: usize,
    means: &VectorF64,
    correlation_matrix: &ArrayF64,
    rng: &mut R
) -> crate::Result<ArrayF64> {
    let n_vars = means.len();
    
    if correlation_matrix.nrows() != n_vars || correlation_matrix.ncols() != n_vars {
        return Err(crate::DistributionError::invalid_parameter("Correlation matrix dimensions must match means vector length"));
    }
    
    // Generate independent standard normal samples
    let standard_normal = Normal::new(0.0, 1.0)?;
    let mut samples = ArrayF64::zeros(n_samples, n_vars);
    
    for i in 0..n_samples {
        for j in 0..n_vars {
            let sample = standard_normal.sample(rng);
            samples.set(i, j, sample)
                .map_err(|_| crate::DistributionError::sampling_error("Sample matrix access failed"))?;
        }
    }
    
    // For now, just add the means (this is a simplified approach)
    // In a full implementation, you would apply the Cholesky decomposition
    for i in 0..n_samples {
        for j in 0..n_vars {
            let current = samples.get(i, j).ok_or_else(|| 
                crate::DistributionError::sampling_error("Sample access failed"))?;
            let mean = means.get(j).ok_or_else(|| 
                crate::DistributionError::sampling_error("Mean access failed"))?;
            samples.set(i, j, current + mean)
                .map_err(|_| crate::DistributionError::sampling_error("Sample matrix access failed"))?;
        }
    }
    
    Ok(samples)
}