//! Integration demo showing rustlab-distributions with rustlab-math
//! 
//! This example demonstrates the seamless integration between rustlab-distributions
//! and rustlab-math arrays/vectors.

#[cfg(feature = "integration")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustlab_distributions::integration::*;
    use rustlab_distributions::*;
    use rustlab_math::{VectorF64, ArrayF64};
    use rand::thread_rng;
    
    println!("🎯 RustLab Distributions + RustLab Math Integration Demo");
    println!("=".repeat(60));
    
    let mut rng = thread_rng();
    
    // 1. Vector Distribution Operations
    println!("\n📊 Vector Distribution Operations");
    println!("-".repeat(40));
    
    // Generate samples directly into vectors
    let normal_vec = VectorF64::normal(1000, 0.0, 1.0, &mut rng)?;
    let uniform_vec = VectorF64::uniform(1000, -2.0, 2.0, &mut rng)?;
    
    println!("Generated {} normal samples", normal_vec.len());
    println!("Generated {} uniform samples", uniform_vec.len());
    
    // Apply distribution functions to existing data
    let test_data = VectorF64::from_slice(&[0.0, 1.0, 2.0, -1.0, -2.0]);
    let pdf_values = test_data.normal_pdf(0.0, 1.0)?;
    let cdf_values = test_data.normal_cdf(0.0, 1.0)?;
    
    println!("Test data: [0.0, 1.0, 2.0, -1.0, -2.0]");
    println!("Normal PDF values: {:?}", (0..pdf_values.len()).map(|i| pdf_values.get(i).unwrap()).collect::<Vec<_>>());
    println!("Normal CDF values: {:?}", (0..cdf_values.len()).map(|i| cdf_values.get(i).unwrap()).collect::<Vec<_>>());
    
    // 2. Array Distribution Operations  
    println!("\n📋 Array Distribution Operations");
    println!("-".repeat(40));
    
    // Generate random arrays
    let mut normal_array = ArrayF64::normal(5, 5, 0.0, 1.0, &mut rng)?;
    let uniform_array = ArrayF64::uniform(3, 4, 0.0, 1.0, &mut rng)?;
    
    println!("Generated {}x{} normal array", normal_array.nrows(), normal_array.ncols());
    println!("Generated {}x{} uniform array", uniform_array.nrows(), uniform_array.ncols());
    
    // Fill existing array with distribution samples
    normal_array.fill_exponential(1.0, &mut rng)?;
    println!("Filled array with exponential samples");
    
    // Calculate empirical statistics
    let (col_means, col_stds) = normal_array.empirical_stats_cols()?;
    println!("Column means: {:?}", (0..col_means.len()).map(|i| col_means.get(i).unwrap()).collect::<Vec<_>>());
    println!("Column std devs: {:?}", (0..col_stds.len()).map(|i| col_stds.get(i).unwrap()).collect::<Vec<_>>());
    
    // 3. Convenience Functions
    println!("\n🛠️  Convenience Functions");
    println!("-".repeat(40));
    
    // Generate common distributions
    let std_normal = standard_normal(100, &mut rng)?;
    let std_uniform = standard_uniform(100, &mut rng)?;
    
    println!("Generated {} standard normal samples", std_normal.len());
    println!("Generated {} standard uniform samples", std_uniform.len());
    
    // Generate structured data
    let walk = random_walk(50, 0.1, 0.0, &mut rng)?;
    let gbm = geometric_brownian_motion(50, 0.05, 0.2, 1.0/252.0, 100.0, &mut rng)?;
    
    println!("Generated random walk with {} steps", walk.len());
    println!("Generated geometric Brownian motion with {} steps", gbm.len());
    
    let start_price = gbm.get(0).unwrap();
    let end_price = gbm.get(gbm.len()-1).unwrap();
    println!("Stock price: ${:.2} -> ${:.2}", start_price, end_price);
    
    // 4. Random Matrix Generation
    println!("\n🔢 Random Matrix Generation");
    println!("-".repeat(40));
    
    let sym_matrix = random_symmetric_matrix(4, 0.0, 1.0, &mut rng)?;
    let pos_def_matrix = random_positive_definite_matrix(3, &mut rng)?;
    let corr_matrix = random_correlation_matrix(3, &mut rng)?;
    
    println!("Generated {}x{} symmetric matrix", sym_matrix.nrows(), sym_matrix.ncols());
    println!("Generated {}x{} positive definite matrix", pos_def_matrix.nrows(), pos_def_matrix.ncols());
    println!("Generated {}x{} correlation matrix", corr_matrix.nrows(), corr_matrix.ncols());
    
    // Verify correlation matrix properties
    let diag_elem = corr_matrix.get(0, 0).unwrap();
    println!("Correlation matrix diagonal element: {:.3} (should be 1.0)", diag_elem);
    
    // 5. Time Series Generation
    println!("\n📈 Time Series Generation");
    println!("-".repeat(40));
    
    let trend_series = random_time_series(100, 0.1, 10.0, 0.5, &mut rng)?;
    let ar1_series_data = ar1_series(100, 0.8, 0.3, 0.0, &mut rng)?;
    
    let trend_start = trend_series.get(0).unwrap();
    let trend_end = trend_series.get(trend_series.len()-1).unwrap();
    println!("Trend series: {:.2} -> {:.2}", trend_start, trend_end);
    
    let ar1_start = ar1_series_data.get(0).unwrap();
    let ar1_end = ar1_series_data.get(ar1_series_data.len()-1).unwrap();
    println!("AR(1) series: {:.3} -> {:.3}", ar1_start, ar1_end);
    
    // 6. Empirical Distribution Analysis
    println!("\n📊 Empirical Distribution Analysis");
    println!("-".repeat(40));
    
    // Generate some test data
    let test_samples = VectorF64::normal(500, 2.0, 1.5, &mut rng)?;
    
    // Calculate empirical PDF and CDF
    let (pdf_bins, pdf_values) = test_samples.empirical_pdf(20)?;
    let (cdf_x, cdf_y) = test_samples.empirical_cdf()?;
    
    println!("Calculated empirical PDF with {} bins", pdf_bins.len());
    println!("Calculated empirical CDF with {} points", cdf_x.len());
    
    // Sample from empirical distribution
    let empirical_samples = empirical_samples(&test_samples, 100, &mut rng)?;
    println!("Generated {} samples from empirical distribution", empirical_samples.len());
    
    // Bootstrap sampling
    let bootstrap_samples = bootstrap_samples(&test_samples, 200, &mut rng)?;
    println!("Generated {} bootstrap samples", bootstrap_samples.len());
    
    // 7. Discrete Distributions
    println!("\n🎲 Discrete Distributions");
    println!("-".repeat(40));
    
    let mut discrete_vec = VectorF64::zeros(100);
    discrete_vec.fill_bernoulli(0.3, &mut rng)?;
    discrete_vec.fill_binomial(10, 0.4, &mut rng)?;
    discrete_vec.fill_poisson(2.5, &mut rng)?;
    
    println!("Filled vector with discrete distribution samples");
    
    let bernoulli_samples = VectorF64::bernoulli(50, 0.6, &mut rng)?;
    let binomial_samples = VectorF64::binomial(50, 20, 0.3, &mut rng)?;
    let poisson_samples = VectorF64::poisson(50, 3.0, &mut rng)?;
    
    println!("Generated discrete distribution samples:");
    println!("  Bernoulli: {} samples", bernoulli_samples.len());
    println!("  Binomial: {} samples", binomial_samples.len());
    println!("  Poisson: {} samples", poisson_samples.len());
    
    // 8. Statistical Testing
    println!("\n🧪 Statistical Testing");
    println!("-".repeat(40));
    
    let test_array = ArrayF64::normal(10, 10, 0.0, 1.0, &mut rng)?;
    let normality_p_value = test_array.test_normality()?;
    println!("Normality test p-value (approximation): {:.4}", normality_p_value);
    
    let correlation_matrix_result = test_array.correlation_matrix()?;
    println!("Calculated {}x{} correlation matrix", 
             correlation_matrix_result.nrows(), correlation_matrix_result.ncols());
    
    // 9. Sparse and Structured Data
    println!("\n🕸️  Sparse and Structured Data");
    println!("-".repeat(40));
    
    let sparse_vec = random_sparse_vector(100, 0.8, 0.0, 1.0, &mut rng)?; // 80% sparse
    let non_zero_count = (0..sparse_vec.len())
        .map(|i| sparse_vec.get(i).unwrap())
        .filter(|&x| x.abs() > 1e-10)
        .count();
    
    println!("Generated sparse vector: {}/{} non-zero elements ({:.1}% sparse)", 
             non_zero_count, sparse_vec.len(), 
             (sparse_vec.len() - non_zero_count) as f64 / sparse_vec.len() as f64 * 100.0);
    
    println!("\n✅ Integration demo completed successfully!");
    println!("All distribution operations work seamlessly with rustlab-math arrays and vectors.");
    
    Ok(())
}

#[cfg(not(feature = "integration"))]
fn main() {
    println!("This example requires the 'integration' feature to be enabled.");
    println!("Run with: cargo run --example integration_demo --features integration");
}