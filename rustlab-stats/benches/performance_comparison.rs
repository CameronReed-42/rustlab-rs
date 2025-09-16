//! Performance benchmarks comparing different implementations
//!
//! Run with: cargo bench --features parallel

use rustlab_stats::prelude::*;
use rustlab_math::{vec64, ArrayF64};
use std::time::Instant;

fn main() {
    println!("RustLab Stats Performance Benchmarks");
    println!("=====================================");
    
    // Test different data sizes
    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];
    
    for &size in &sizes {
        println!("\n--- Testing with {} elements ---", size);
        run_vector_benchmarks(size);
        run_array_benchmarks(size / 100); // Smaller for 2D arrays
    }
}

fn run_vector_benchmarks(size: usize) {
    let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
    let vector = VectorF64::from_slice(&data);
    
    println!("\nVector Operations:");
    
    // Mean comparison
    let start = Instant::now();
    let mean1 = vector.mean();
    let duration1 = start.elapsed();
    
    let start = Instant::now();
    let mean2 = f64::mean_slice(vector.as_slice_unchecked());
    let duration2 = start.elapsed();
    
    println!("Mean - Standard: {:?}, Zero-copy: {:?}, Speedup: {:.2}x", 
             duration1, duration2, duration1.as_secs_f64() / duration2.as_secs_f64());
    assert!((mean1 - mean2).abs() < 1e-10);
    
    // Variance comparison
    let start = Instant::now();
    let var1 = vector.var(None);
    let duration1 = start.elapsed();
    
    let start = Instant::now();
    let var2 = f64::var_slice(vector.as_slice_unchecked(), 1);
    let duration2 = start.elapsed();
    
    println!("Variance - Standard: {:?}, Zero-copy: {:?}, Speedup: {:.2}x", 
             duration1, duration2, duration1.as_secs_f64() / duration2.as_secs_f64());
    assert!((var1 - var2).abs() < 1e-10);
    
    // Min-max comparison
    let start = Instant::now();
    let data_slice = vector.as_slice_unchecked();
    let min1 = data_slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max1 = data_slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let duration1 = start.elapsed();
    
    let start = Instant::now();
    let (min2, max2) = f64::minmax_slice(vector.as_slice_unchecked());
    let duration2 = start.elapsed();
    
    println!("MinMax - Standard: {:?}, Zero-copy: {:?}, Speedup: {:.2}x", 
             duration1, duration2, duration1.as_secs_f64() / duration2.as_secs_f64());
    assert_eq!(min1, min2);
    assert_eq!(max1, max2);
    
    // Parallel comparison (if available)
    #[cfg(feature = "parallel")]
    {
        let start = Instant::now();
        let mean_serial = vector.mean();
        let duration1 = start.elapsed();
        
        let start = Instant::now();
        let mean_parallel = vector.mean_parallel();
        let duration2 = start.elapsed();
        
        println!("Mean - Serial: {:?}, Parallel: {:?}, Speedup: {:.2}x", 
                 duration1, duration2, duration1.as_secs_f64() / duration2.as_secs_f64());
        assert!((mean_serial - mean_parallel).abs() < 1e-10);
        
        let start = Instant::now();
        let var_serial = vector.var(None);
        let duration1 = start.elapsed();
        
        let start = Instant::now();
        let var_parallel = vector.var_parallel(None);
        let duration2 = start.elapsed();
        
        println!("Variance - Serial: {:?}, Parallel: {:?}, Speedup: {:.2}x", 
                 duration1, duration2, duration1.as_secs_f64() / duration2.as_secs_f64());
        assert!((var_serial - var_parallel).abs() < 1e-10);
    }
}

fn run_array_benchmarks(size: usize) {
    // Create square array
    let total_elements = size * size;
    let data: Vec<f64> = (0..total_elements).map(|i| (i as f64) * 0.1).collect();
    let array = ArrayF64::from_slice(&data, size, size).unwrap();
    
    println!("\nArray Operations ({}x{}):", size, size);
    
    // Array mean comparison
    let start = Instant::now();
    let mean1 = {
        let mut sum = 0.0;
        for i in 0..array.nrows() {
            for j in 0..array.ncols() {
                sum += array.get(i, j).unwrap();
            }
        }
        sum / (array.nrows() * array.ncols()) as f64
    };
    let duration1 = start.elapsed();
    
    #[cfg(feature = "parallel")]
    {
        let start = Instant::now();
        let mean2 = array.mean_parallel();
        let duration2 = start.elapsed();
        
        println!("Array Mean - Serial: {:?}, Parallel: {:?}, Speedup: {:.2}x", 
                 duration1, duration2, duration1.as_secs_f64() / duration2.as_secs_f64());
        assert!((mean1 - mean2).abs() < 1e-10);
    }
    
    #[cfg(not(feature = "parallel"))]
    {
        println!("Array Mean - Serial: {:?} (parallel not enabled)", duration1);
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use rustlab_stats::performance::benchmarks::*;
    
    #[test]
    fn test_benchmark_framework() {
        let results = benchmark_vector_ops(1000);
        assert!(results.len() > 0);
        
        for result in &results {
            assert!(result.duration.as_nanos() > 0);
            assert!(result.operations_per_second > 0.0);
            assert_eq!(result.data_size, 1000);
        }
    }
    
    #[test]
    fn test_streaming_stats_performance() {
        let size = 10_000;
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        
        // Standard approach
        let start = Instant::now();
        let mean1 = data.iter().sum::<f64>() / data.len() as f64;
        let var1 = data.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        let duration1 = start.elapsed();
        
        // Streaming approach
        let start = Instant::now();
        let mut stats = StreamingStats::<f64>::new();
        for &value in &data {
            stats.update(value);
        }
        let mean2 = stats.mean();
        let var2 = stats.variance();
        let duration2 = start.elapsed();
        
        println!("Stats - Standard: {:?}, Streaming: {:?}", duration1, duration2);
        assert!((mean1 - mean2).abs() < 1e-10);
        assert!((var1 - var2).abs() < 1e-10);
    }
    
    #[test]
    fn test_quickselect_performance() {
        let size = 10_000;
        let mut data1: Vec<f64> = (0..size).map(|i| (i as f64) * 0.73).collect(); // Some pattern
        let mut data2 = data1.clone();
        
        // Full sort approach
        let start = Instant::now();
        data1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median1 = data1[data1.len() / 2];
        let duration1 = start.elapsed();
        
        // Quickselect approach
        let start = Instant::now();
        let median2 = f64::quantile_slice_fast(&mut data2, 0.5);
        let duration2 = start.elapsed();
        
        println!("Median - Full sort: {:?}, Quickselect: {:?}, Speedup: {:.2}x", 
                 duration1, duration2, duration1.as_secs_f64() / duration2.as_secs_f64());
        assert!((median1 - median2).abs() < 1e-10);
    }
}