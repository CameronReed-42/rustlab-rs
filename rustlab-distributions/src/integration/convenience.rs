//! Convenience functions for common distribution operations
//!
//! This module provides high-level convenience functions that combine
//! distributions with rustlab-math arrays and vectors.

use crate::continuous::{Normal, Uniform, Exponential, Gamma};
use crate::discrete::{Bernoulli, Binomial, Poisson};
use crate::traits::{Sampling};
use rustlab_math::{VectorF64, ArrayF64};
use rand::Rng;

/// Generate a vector of samples from a normal distribution
/// 
/// # Arguments
/// * `n` - Number of samples
/// * `mean` - Mean of the distribution
/// * `std_dev` - Standard deviation
/// * `rng` - Random number generator
/// 
/// # Example
/// ```ignore
/// use rustlab_distributions::integration::normal_samples;
/// use rand::thread_rng;
/// 
/// let mut rng = thread_rng();
/// let samples = normal_samples(1000, 0.0, 1.0, &mut rng).unwrap();
/// ```
pub fn normal_samples<R: Rng>(n: usize, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<VectorF64> {
    let dist = Normal::new(mean, std_dev)?;
    let mut vec = VectorF64::zeros(n);
    if let Some(slice) = vec.as_mut_slice() {
        for i in 0..n {
            let sample = dist.sample(rng);
            slice[i] = sample;
        }
    } else {
        return Err(crate::DistributionError::sampling_error("Vector creation failed"));
    }
    Ok(vec)
}

/// Generate a vector of samples from a uniform distribution
/// 
/// # Arguments
/// * `n` - Number of samples
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `rng` - Random number generator
pub fn uniform_samples<R: Rng>(n: usize, a: f64, b: f64, rng: &mut R) -> crate::Result<VectorF64> {
    let dist = Uniform::new(a, b)?;
    let mut vec = VectorF64::zeros(n);
    if let Some(slice) = vec.as_mut_slice() {
        for i in 0..n {
            let sample = dist.sample(rng);
            slice[i] = sample;
        }
    } else {
        return Err(crate::DistributionError::sampling_error("Vector creation failed"));
    }
    Ok(vec)
}

/// Generate a vector of samples from an exponential distribution
/// 
/// # Arguments
/// * `n` - Number of samples
/// * `lambda` - Rate parameter
/// * `rng` - Random number generator
pub fn exponential_samples<R: Rng>(n: usize, lambda: f64, rng: &mut R) -> crate::Result<VectorF64> {
    let dist = Exponential::new(lambda)?;
    let mut vec = VectorF64::zeros(n);
    for i in 0..n {
        let sample = dist.sample(rng);
        vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
    }
    Ok(vec)
}

/// Generate a vector of samples from a gamma distribution
/// 
/// # Arguments
/// * `n` - Number of samples
/// * `alpha` - Shape parameter
/// * `beta` - Rate parameter
/// * `rng` - Random number generator
pub fn gamma_samples<R: Rng>(n: usize, alpha: f64, beta: f64, rng: &mut R) -> crate::Result<VectorF64> {
    let dist = Gamma::new(alpha, beta)?;
    let mut vec = VectorF64::zeros(n);
    for i in 0..n {
        let sample = dist.sample(rng);
        vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
    }
    Ok(vec)
}

/// Generate a vector of samples from a Bernoulli distribution (as 0.0/1.0)
/// 
/// # Arguments
/// * `n` - Number of samples
/// * `p` - Probability of success
/// * `rng` - Random number generator
pub fn bernoulli_samples<R: Rng>(n: usize, p: f64, rng: &mut R) -> crate::Result<VectorF64> {
    let dist = Bernoulli::new(p)?;
    let mut vec = VectorF64::zeros(n);
    for i in 0..n {
        let sample = if dist.sample(rng) == 1 { 1.0 } else { 0.0 };
        vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
    }
    Ok(vec)
}

/// Generate a vector of samples from a Binomial distribution
/// 
/// # Arguments
/// * `n` - Number of samples
/// * `trials` - Number of trials
/// * `p` - Probability of success per trial
/// * `rng` - Random number generator
pub fn binomial_samples<R: Rng>(n: usize, trials: u32, p: f64, rng: &mut R) -> crate::Result<VectorF64> {
    let dist = Binomial::new(trials, p)?;
    let mut vec = VectorF64::zeros(n);
    for i in 0..n {
        let sample = dist.sample(rng) as f64;
        vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
    }
    Ok(vec)
}

/// Generate a vector of samples from a Poisson distribution
/// 
/// # Arguments
/// * `n` - Number of samples
/// * `lambda` - Rate parameter
/// * `rng` - Random number generator
pub fn poisson_samples<R: Rng>(n: usize, lambda: f64, rng: &mut R) -> crate::Result<VectorF64> {
    let dist = Poisson::new(lambda)?;
    let mut vec = VectorF64::zeros(n);
    for i in 0..n {
        let sample = dist.sample(rng) as f64;
        vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
    }
    Ok(vec)
}

/// Generate a 2D array of samples from a normal distribution
/// 
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `mean` - Mean of the distribution
/// * `std_dev` - Standard deviation
/// * `rng` - Random number generator
/// 
/// # Example
/// ```ignore
/// use rustlab_distributions::integration::normal_array;
/// use rand::thread_rng;
/// 
/// let mut rng = thread_rng();
/// let samples = normal_array(10, 10, 0.0, 1.0, &mut rng).unwrap();
/// ```
pub fn normal_array<R: Rng>(rows: usize, cols: usize, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<ArrayF64> {
    let dist = Normal::new(mean, std_dev)?;
    let mut arr = ArrayF64::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let sample = dist.sample(rng);
            arr.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array creation failed"))?;
        }
    }
    Ok(arr)
}

/// Generate a 2D array of samples from a uniform distribution
/// 
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `rng` - Random number generator
pub fn uniform_array<R: Rng>(rows: usize, cols: usize, a: f64, b: f64, rng: &mut R) -> crate::Result<ArrayF64> {
    let dist = Uniform::new(a, b)?;
    let mut arr = ArrayF64::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let sample = dist.sample(rng);
            arr.set(i, j, sample).map_err(|_| crate::DistributionError::sampling_error("Array creation failed"))?;
        }
    }
    Ok(arr)
}

/// Generate standard normal samples (mean=0, std_dev=1)
/// 
/// # Arguments
/// * `n` - Number of samples
/// * `rng` - Random number generator
pub fn standard_normal<R: Rng>(n: usize, rng: &mut R) -> crate::Result<VectorF64> {
    normal_samples(n, 0.0, 1.0, rng)
}

/// Generate standard uniform samples (0, 1)
/// 
/// # Arguments
/// * `n` - Number of samples
/// * `rng` - Random number generator
pub fn standard_uniform<R: Rng>(n: usize, rng: &mut R) -> crate::Result<VectorF64> {
    uniform_samples(n, 0.0, 1.0, rng)
}

/// Generate a correlation matrix with specified correlation structure
/// 
/// This creates a positive semi-definite correlation matrix that can be used
/// to generate multivariate normal samples with specified correlations.
/// 
/// # Arguments
/// * `size` - Size of the correlation matrix (square)
/// * `base_correlation` - Base correlation between variables
/// 
/// # Returns
/// A correlation matrix where off-diagonal elements are `base_correlation`
/// and diagonal elements are 1.0
pub fn correlation_matrix(size: usize, base_correlation: f64) -> crate::Result<ArrayF64> {
    if base_correlation.abs() >= 1.0 {
        return Err(crate::DistributionError::invalid_parameter("Correlation must be between -1 and 1"));
    }
    
    let mut corr_matrix = ArrayF64::eye(size);
    
    // Set off-diagonal elements to base correlation
    for i in 0..size {
        for j in 0..size {
            if i != j {
                corr_matrix.set(i, j, base_correlation)
                    .map_err(|_| crate::DistributionError::sampling_error("Matrix creation failed"))?;
            }
        }
    }
    
    Ok(corr_matrix)
}

/// Generate samples that follow a specified empirical distribution
/// 
/// This uses inverse transform sampling to generate samples that match
/// the empirical distribution of the provided data.
/// 
/// # Arguments
/// * `data` - Original data vector to match the distribution of
/// * `n_samples` - Number of samples to generate
/// * `rng` - Random number generator
pub fn empirical_samples<R: Rng>(data: &VectorF64, n_samples: usize, rng: &mut R) -> crate::Result<VectorF64> {
    if data.len() == 0 {
        return Err(crate::DistributionError::invalid_parameter("Data vector cannot be empty"));
    }
    
    // Create sorted data for inverse transform sampling
    let mut sorted_data = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        sorted_data.push(data.get(i).ok_or_else(|| 
            crate::DistributionError::sampling_error("Data access failed"))?);
    }
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let uniform_dist = Uniform::new(0.0, 1.0)?;
    let mut samples = VectorF64::zeros(n_samples);
    
    for i in 0..n_samples {
        // Generate uniform sample
        let u = uniform_dist.sample(rng);
        
        // Find corresponding value in sorted data (inverse transform)
        let index = ((u * sorted_data.len() as f64).floor() as usize).min(sorted_data.len() - 1);
        let sample_value = sorted_data[index];
        
        samples.set(i, sample_value)
            .map_err(|_| crate::DistributionError::sampling_error("Sample vector creation failed"))?;
    }
    
    Ok(samples)
}

/// Bootstrap resampling from existing data
/// 
/// # Arguments
/// * `data` - Original data to resample from
/// * `n_samples` - Number of bootstrap samples
/// * `rng` - Random number generator
pub fn bootstrap_samples<R: Rng>(data: &VectorF64, n_samples: usize, rng: &mut R) -> crate::Result<VectorF64> {
    if data.len() == 0 {
        return Err(crate::DistributionError::invalid_parameter("Data vector cannot be empty"));
    }
    
    let uniform_dist = Uniform::new(0.0, data.len() as f64)?;
    let mut samples = VectorF64::zeros(n_samples);
    
    for i in 0..n_samples {
        let random_index = uniform_dist.sample(rng).floor() as usize;
        let index = random_index.min(data.len() - 1);
        let value = data.get(index).ok_or_else(|| 
            crate::DistributionError::sampling_error("Data access failed"))?;
        
        samples.set(i, value)
            .map_err(|_| crate::DistributionError::sampling_error("Sample vector creation failed"))?;
    }
    
    Ok(samples)
}