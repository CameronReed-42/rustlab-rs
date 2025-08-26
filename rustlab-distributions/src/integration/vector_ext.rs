//! Vector extensions for probability distributions
//!
//! This module provides extension traits to add distribution functionality
//! to rustlab-math vectors.

use crate::continuous::{Normal, Uniform, Exponential, Gamma};
use crate::discrete::{Bernoulli, Binomial, Poisson};
use crate::traits::{Sampling, ContinuousDistribution};
use rustlab_math::{VectorF64, VectorF32, ArrayF64, Result as MathResult, MathError};
use rand::Rng;

/// Extension trait for VectorF64 to add distribution sampling methods
pub trait DistributionVectorF64 {
    /// Fill vector with samples from a normal distribution
    /// 
    /// # Arguments
    /// * `mean` - Mean of the normal distribution
    /// * `std_dev` - Standard deviation of the normal distribution
    /// * `rng` - Random number generator
    /// 
    /// # Example
    /// ```ignore
    /// use rustlab_math::VectorF64;
    /// use rustlab_distributions::integration::DistributionVectorF64;
    /// use rand::thread_rng;
    /// 
    /// let mut rng = thread_rng();
    /// let mut vec = VectorF64::zeros(100);
    /// vec.fill_normal(0.0, 1.0, &mut rng).unwrap();
    /// ```
    fn fill_normal<R: Rng>(&mut self, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Fill vector with samples from a uniform distribution
    /// 
    /// # Arguments
    /// * `a` - Lower bound
    /// * `b` - Upper bound  
    /// * `rng` - Random number generator
    fn fill_uniform<R: Rng>(&mut self, a: f64, b: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Fill vector with samples from an exponential distribution
    /// 
    /// # Arguments
    /// * `lambda` - Rate parameter
    /// * `rng` - Random number generator
    fn fill_exponential<R: Rng>(&mut self, lambda: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Fill vector with samples from a gamma distribution
    /// 
    /// # Arguments
    /// * `alpha` - Shape parameter
    /// * `beta` - Rate parameter
    /// * `rng` - Random number generator
    fn fill_gamma<R: Rng>(&mut self, alpha: f64, beta: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Create a new vector with normal distribution samples
    /// 
    /// # Arguments
    /// * `n` - Number of samples
    /// * `mean` - Mean of the distribution
    /// * `std_dev` - Standard deviation
    /// * `rng` - Random number generator
    fn normal<R: Rng>(n: usize, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<VectorF64>;
    
    /// Create a new vector with uniform distribution samples
    fn uniform<R: Rng>(n: usize, a: f64, b: f64, rng: &mut R) -> crate::Result<VectorF64>;
    
    /// Create a new vector with exponential distribution samples
    fn exponential<R: Rng>(n: usize, lambda: f64, rng: &mut R) -> crate::Result<VectorF64>;
    
    /// Create a new vector with gamma distribution samples
    fn gamma<R: Rng>(n: usize, alpha: f64, beta: f64, rng: &mut R) -> crate::Result<VectorF64>;
    
    /// Apply normal PDF to all elements in the vector
    /// 
    /// # Arguments
    /// * `mean` - Mean of the normal distribution
    /// * `std_dev` - Standard deviation
    fn normal_pdf(&self, mean: f64, std_dev: f64) -> crate::Result<VectorF64>;
    
    /// Apply normal CDF to all elements in the vector
    fn normal_cdf(&self, mean: f64, std_dev: f64) -> crate::Result<VectorF64>;
    
    /// Apply uniform PDF to all elements in the vector
    fn uniform_pdf(&self, a: f64, b: f64) -> crate::Result<VectorF64>;
    
    /// Apply uniform CDF to all elements in the vector  
    fn uniform_cdf(&self, a: f64, b: f64) -> crate::Result<VectorF64>;
    
    /// Apply exponential PDF to all elements in the vector
    fn exponential_pdf(&self, lambda: f64) -> crate::Result<VectorF64>;
    
    /// Apply exponential CDF to all elements in the vector
    fn exponential_cdf(&self, lambda: f64) -> crate::Result<VectorF64>;
    
    /// Calculate empirical PDF using histogram
    /// 
    /// # Arguments
    /// * `bins` - Number of bins for the histogram
    fn empirical_pdf(&self, bins: usize) -> MathResult<(VectorF64, VectorF64)>;
    
    /// Calculate empirical CDF
    fn empirical_cdf(&self) -> MathResult<(VectorF64, VectorF64)>;
}

impl DistributionVectorF64 for VectorF64 {
    fn fill_normal<R: Rng>(&mut self, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Normal::new(mean, std_dev)?;
        for i in 0..self.len() {
            let sample = dist.sample(rng);
            self.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
        }
        Ok(())
    }
    
    fn fill_uniform<R: Rng>(&mut self, a: f64, b: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Uniform::new(a, b)?;
        for i in 0..self.len() {
            let sample = dist.sample(rng);
            self.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
        }
        Ok(())
    }
    
    fn fill_exponential<R: Rng>(&mut self, lambda: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Exponential::new(lambda)?;
        for i in 0..self.len() {
            let sample = dist.sample(rng);
            self.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
        }
        Ok(())
    }
    
    fn fill_gamma<R: Rng>(&mut self, alpha: f64, beta: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Gamma::new(alpha, beta)?;
        for i in 0..self.len() {
            let sample = dist.sample(rng);
            self.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
        }
        Ok(())
    }
    
    fn normal<R: Rng>(n: usize, mean: f64, std_dev: f64, rng: &mut R) -> crate::Result<VectorF64> {
        let dist = Normal::new(mean, std_dev)?;
        let mut vec = VectorF64::zeros(n);
        for i in 0..n {
            let sample = dist.sample(rng);
            vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
        }
        Ok(vec)
    }
    
    fn uniform<R: Rng>(n: usize, a: f64, b: f64, rng: &mut R) -> crate::Result<VectorF64> {
        let dist = Uniform::new(a, b)?;
        let mut vec = VectorF64::zeros(n);
        for i in 0..n {
            let sample = dist.sample(rng);
            vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
        }
        Ok(vec)
    }
    
    fn exponential<R: Rng>(n: usize, lambda: f64, rng: &mut R) -> crate::Result<VectorF64> {
        let dist = Exponential::new(lambda)?;
        let mut vec = VectorF64::zeros(n);
        for i in 0..n {
            let sample = dist.sample(rng);
            vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
        }
        Ok(vec)
    }
    
    fn gamma<R: Rng>(n: usize, alpha: f64, beta: f64, rng: &mut R) -> crate::Result<VectorF64> {
        let dist = Gamma::new(alpha, beta)?;
        let mut vec = VectorF64::zeros(n);
        for i in 0..n {
            let sample = dist.sample(rng);
            vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
        }
        Ok(vec)
    }
    
    fn normal_pdf(&self, mean: f64, std_dev: f64) -> crate::Result<VectorF64> {
        let dist = Normal::new(mean, std_dev)?;
        let mut result = VectorF64::zeros(self.len());
        for i in 0..self.len() {
            let val = self.get(i).ok_or_else(|| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
            let pdf_val = dist.pdf(val);
            result.set(i, pdf_val).map_err(|_| crate::DistributionError::sampling_error("Result vector access failed"))?;
        }
        Ok(result)
    }
    
    fn normal_cdf(&self, mean: f64, std_dev: f64) -> crate::Result<VectorF64> {
        let dist = Normal::new(mean, std_dev)?;
        let mut result = VectorF64::zeros(self.len());
        for i in 0..self.len() {
            let val = self.get(i).ok_or_else(|| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
            let cdf_val = dist.cdf(val);
            result.set(i, cdf_val).map_err(|_| crate::DistributionError::sampling_error("Result vector access failed"))?;
        }
        Ok(result)
    }
    
    fn uniform_pdf(&self, a: f64, b: f64) -> crate::Result<VectorF64> {
        let dist = Uniform::new(a, b)?;
        let mut result = VectorF64::zeros(self.len());
        for i in 0..self.len() {
            let val = self.get(i).ok_or_else(|| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
            let pdf_val = dist.pdf(val);
            result.set(i, pdf_val).map_err(|_| crate::DistributionError::sampling_error("Result vector access failed"))?;
        }
        Ok(result)
    }
    
    fn uniform_cdf(&self, a: f64, b: f64) -> crate::Result<VectorF64> {
        let dist = Uniform::new(a, b)?;
        let mut result = VectorF64::zeros(self.len());
        for i in 0..self.len() {
            let val = self.get(i).ok_or_else(|| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
            let cdf_val = dist.cdf(val);
            result.set(i, cdf_val).map_err(|_| crate::DistributionError::sampling_error("Result vector access failed"))?;
        }
        Ok(result)
    }
    
    fn exponential_pdf(&self, lambda: f64) -> crate::Result<VectorF64> {
        let dist = Exponential::new(lambda)?;
        let mut result = VectorF64::zeros(self.len());
        for i in 0..self.len() {
            let val = self.get(i).ok_or_else(|| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
            let pdf_val = dist.pdf(val);
            result.set(i, pdf_val).map_err(|_| crate::DistributionError::sampling_error("Result vector access failed"))?;
        }
        Ok(result)
    }
    
    fn exponential_cdf(&self, lambda: f64) -> crate::Result<VectorF64> {
        let dist = Exponential::new(lambda)?;
        let mut result = VectorF64::zeros(self.len());
        for i in 0..self.len() {
            let val = self.get(i).ok_or_else(|| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
            let cdf_val = dist.cdf(val);
            result.set(i, cdf_val).map_err(|_| crate::DistributionError::sampling_error("Result vector access failed"))?;
        }
        Ok(result)
    }
    
    fn empirical_pdf(&self, bins: usize) -> MathResult<(VectorF64, VectorF64)> {
        if bins == 0 {
            return Err(MathError::IndexOutOfBounds { index: 0, size: 0 });
        }
        
        // Find min and max values
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        
        for i in 0..self.len() {
            let val = self.get(i).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.len() })?;
            if val < min_val { min_val = val; }
            if val > max_val { max_val = val; }
        }
        
        if min_val >= max_val {
            return Err(MathError::IndexOutOfBounds { index: 0, size: 0 });
        }
        
        let range = max_val - min_val;
        let bin_width = range / bins as f64;
        
        // Create histogram
        let mut counts = vec![0.0; bins];
        for i in 0..self.len() {
            let val = self.get(i).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.len() })?;
            let bin_idx = ((val - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1); // Handle edge case for max value
            counts[bin_idx] += 1.0;
        }
        
        // Normalize to get PDF
        let total_area = self.len() as f64 * bin_width;
        for count in &mut counts {
            *count /= total_area;
        }
        
        // Create bin centers
        let bin_centers: Vec<f64> = (0..bins)
            .map(|i| min_val + (i as f64 + 0.5) * bin_width)
            .collect();
        
        Ok((VectorF64::from_slice(&bin_centers), VectorF64::from_slice(&counts)))
    }
    
    fn empirical_cdf(&self) -> MathResult<(VectorF64, VectorF64)> {
        // Create sorted values
        let mut sorted_values = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            let val = self.get(i).ok_or_else(|| MathError::IndexOutOfBounds { index: i, size: self.len() })?;
            sorted_values.push(val);
        }
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Create CDF values (empirical probabilities)
        let cdf_values: Vec<f64> = (1..=sorted_values.len())
            .map(|i| i as f64 / sorted_values.len() as f64)
            .collect();
        
        Ok((VectorF64::from_slice(&sorted_values), VectorF64::from_slice(&cdf_values)))
    }
}

/// Extension trait for discrete distribution operations on vectors
pub trait DiscreteDistributionVectorF64 {
    /// Fill vector with Bernoulli distribution samples (0.0 or 1.0)
    fn fill_bernoulli<R: Rng>(&mut self, p: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Fill vector with Binomial distribution samples
    fn fill_binomial<R: Rng>(&mut self, n: u32, p: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Fill vector with Poisson distribution samples
    fn fill_poisson<R: Rng>(&mut self, lambda: f64, rng: &mut R) -> crate::Result<()>;
    
    /// Create vector with Bernoulli samples
    fn bernoulli<R: Rng>(n: usize, p: f64, rng: &mut R) -> crate::Result<VectorF64>;
    
    /// Create vector with Binomial samples  
    fn binomial<R: Rng>(n: usize, trials: u32, p: f64, rng: &mut R) -> crate::Result<VectorF64>;
    
    /// Create vector with Poisson samples
    fn poisson<R: Rng>(n: usize, lambda: f64, rng: &mut R) -> crate::Result<VectorF64>;
}

impl DiscreteDistributionVectorF64 for VectorF64 {
    fn fill_bernoulli<R: Rng>(&mut self, p: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Bernoulli::new(p)?;
        for i in 0..self.len() {
            let sample = if dist.sample(rng) == 1 { 1.0 } else { 0.0 };
            self.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
        }
        Ok(())
    }
    
    fn fill_binomial<R: Rng>(&mut self, n: u32, p: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Binomial::new(n, p)?;
        for i in 0..self.len() {
            let sample = dist.sample(rng) as f64;
            self.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
        }
        Ok(())
    }
    
    fn fill_poisson<R: Rng>(&mut self, lambda: f64, rng: &mut R) -> crate::Result<()> {
        let dist = Poisson::new(lambda)?;
        for i in 0..self.len() {
            let sample = dist.sample(rng) as f64;
            self.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector index out of bounds"))?;
        }
        Ok(())
    }
    
    fn bernoulli<R: Rng>(n: usize, p: f64, rng: &mut R) -> crate::Result<VectorF64> {
        let dist = Bernoulli::new(p)?;
        let mut vec = VectorF64::zeros(n);
        for i in 0..n {
            let sample = if dist.sample(rng) == 1 { 1.0 } else { 0.0 };
            vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
        }
        Ok(vec)
    }
    
    fn binomial<R: Rng>(n: usize, trials: u32, p: f64, rng: &mut R) -> crate::Result<VectorF64> {
        let dist = Binomial::new(trials, p)?;
        let mut vec = VectorF64::zeros(n);
        for i in 0..n {
            let sample = dist.sample(rng) as f64;
            vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
        }
        Ok(vec)
    }
    
    fn poisson<R: Rng>(n: usize, lambda: f64, rng: &mut R) -> crate::Result<VectorF64> {
        let dist = Poisson::new(lambda)?;
        let mut vec = VectorF64::zeros(n);
        for i in 0..n {
            let sample = dist.sample(rng) as f64;
            vec.set(i, sample).map_err(|_| crate::DistributionError::sampling_error("Vector creation failed"))?;
        }
        Ok(vec)
    }
}