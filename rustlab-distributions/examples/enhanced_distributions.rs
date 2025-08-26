//! Example demonstrating the enhanced distributions with rustlab-special integration

use rustlab_distributions::{Normal, Uniform, Gamma, ContinuousDistribution, Distribution, Sampling};
use rustlab_core::rvec;
use rand::thread_rng;

fn main() {
    println!("=== Enhanced Distributions with rustlab-special Integration ===\n");

    // Normal distribution with special functions
    println!("1. Normal Distribution (using rustlab-special's erf and erfinv):");
    let normal = Normal::new(0.0, 1.0).unwrap();
    println!("   Mean: {:.6}", normal.mean());
    println!("   Variance: {:.6}", normal.variance());
    println!("   PDF(0.0): {:.6}", normal.pdf(0.0));
    println!("   CDF(0.0): {:.6}", normal.cdf(0.0));
    println!("   Quantile(0.5): {:.6}", normal.quantile(0.5).unwrap());
    println!("   Quantile(0.975): {:.6}", normal.quantile(0.975).unwrap());
    println!("   Prob within 1 std: {:.6}", normal.prob_within_std(1.0));
    println!("   Prob within 2 std: {:.6}", normal.prob_within_std(2.0));
    println!("   Entropy: {:.6}", normal.entropy());
    
    // Test sampling
    let mut rng = thread_rng();
    let sample = normal.sample(&mut rng);
    println!("   Sample: {:.6}", sample);
    
    let samples = normal.sample_n(&mut rng, 5);
    println!("   5 samples: {:?}", samples.as_slice());
    
    println!();

    // Uniform distribution
    println!("2. Uniform Distribution:");
    let uniform = Uniform::new(0.0, 10.0).unwrap();
    println!("   Mean: {:.6}", uniform.mean());
    println!("   Variance: {:.6}", uniform.variance());
    println!("   PDF(5.0): {:.6}", uniform.pdf(5.0));
    println!("   CDF(5.0): {:.6}", uniform.cdf(5.0));
    println!("   Quantile(0.5): {:.6}", uniform.quantile(0.5).unwrap());
    println!("   Entropy: {:.6}", uniform.entropy());
    println!("   Skewness: {:.6}", uniform.skewness());
    println!("   Kurtosis: {:.6}", uniform.kurtosis());
    
    let uniform_sample = uniform.sample(&mut rng);
    println!("   Sample: {:.6}", uniform_sample);
    
    println!();

    // Gamma distribution with special functions
    println!("3. Gamma Distribution (using rustlab-special's gamma, lgamma, digamma):");
    let gamma = Gamma::new(2.0, 1.0).unwrap();
    println!("   Shape (alpha): {:.6}", gamma.alpha());
    println!("   Rate (beta): {:.6}", gamma.beta());
    println!("   Scale: {:.6}", gamma.scale());
    println!("   Mean: {:.6}", gamma.mean());
    println!("   Variance: {:.6}", gamma.variance());
    println!("   Mode: {:.6}", gamma.mode());
    println!("   PDF(1.0): {:.6}", gamma.pdf(1.0));
    println!("   CDF(1.0): {:.6}", gamma.cdf(1.0));
    println!("   Quantile(0.5): {:.6}", gamma.quantile(0.5).unwrap());
    println!("   Skewness: {:.6}", gamma.skewness());
    println!("   Kurtosis: {:.6}", gamma.kurtosis());
    println!("   Entropy: {:.6}", gamma.entropy());
    
    let gamma_sample = gamma.sample(&mut rng);
    println!("   Sample: {:.6}", gamma_sample);
    
    println!();

    // Test with rustlab-core vectors
    println!("4. Integration with rustlab-core vectors:");
    let data = rvec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("   Input data: {:?}", data.as_slice());
    
    let normal_pdfs = normal.pdf_array(&data);
    println!("   Normal PDFs: {:?}", normal_pdfs.as_slice());
    
    let uniform_cdfs = uniform.cdf_array(&data);
    println!("   Uniform CDFs: {:?}", uniform_cdfs.as_slice());
    
    println!();

    // Test sampling into pre-allocated arrays
    println!("5. Sampling into pre-allocated arrays:");
    let mut output = rvec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    normal.sample_into(&mut rng, &mut output);
    println!("   Normal samples: {:?}", output.as_slice());
    
    gamma.sample_into(&mut rng, &mut output);
    println!("   Gamma samples: {:?}", output.as_slice());
    
    println!();

    // Test special distribution methods
    println!("6. Special distribution methods:");
    println!("   Normal 95% CI: {:?}", normal.confidence_interval(0.95).unwrap());
    println!("   Uniform hazard(5.0): {:.6}", uniform.hazard(5.0));
    println!("   Gamma survival(2.0): {:.6}", gamma.sf(2.0));
    println!("   Gamma cumulative hazard(2.0): {:.6}", gamma.cumulative_hazard(2.0));
    
    println!("\n=== All tests completed successfully! ===");
}