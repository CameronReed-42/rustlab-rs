//! Integration tests for rustlab-stats with the entire ecosystem
//!
//! Tests compatibility with:
//! - rustlab-math types and operations
//! - rustlab-distributions integration
//! - rustlab-linearalgebra matrices (through rustlab-math)
//! - Migration from rustlab-statistics

use rustlab_stats::prelude::*;
use rustlab_math::{vec64, ArrayF64, VectorF64, VectorF32, BasicStatistics, reductions::{Axis, AxisReductions}};

#[cfg(test)]
mod ecosystem_integration {
    use super::*;
    
    #[test]
    fn test_rustlab_math_vector_integration() {
        // Test that rustlab-math vectors work seamlessly with stats
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Basic stats from rustlab-math
        let basic_mean = data.mean();
        let basic_var = data.var(None);
        let basic_std = data.std(None);
        
        // Advanced stats from rustlab-stats
        let median = data.median();
        let iqr = data.iqr();
        let skew = data.skewness();
        
        // Verify integration
        assert_eq!(basic_mean, 3.0);
        assert_eq!(median, 3.0);
        assert!(iqr > 0.0);
        assert_eq!(skew, 0.0); // Perfect symmetry
        
        // Test method chaining
        let normalized = data.zscore();
        assert!((normalized.mean()).abs() < 1e-10);
        assert!((normalized.std(None) - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_rustlab_math_array_integration() {
        // Create array using rustlab-math
        let data = ArrayF64::from_slice(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ], 3, 3).unwrap();
        
        // Basic axis operations from rustlab-math
        let mean_axis0 = data.mean_axis(0);
        let var_axis1 = data.var_axis(1, None);
        
        // Advanced axis operations from rustlab-stats
        let median_axis0 = data.median_axis(0);
        let iqr_axis1 = data.iqr_axis(1);
        
        // Verify shapes are consistent
        assert_eq!(mean_axis0.unwrap().len(), 3);
        assert_eq!(median_axis0.len(), 3);
        assert_eq!(var_axis1.unwrap().len(), 3);
        assert_eq!(iqr_axis1.len(), 3);
        
        // Test values
        assert_eq!(median_axis0.get(0).unwrap(), 4.0);
        assert_eq!(median_axis0.get(1).unwrap(), 5.0);
        assert_eq!(median_axis0.get(2).unwrap(), 6.0);
    }
    
    #[test]
    fn test_matrix_operations_integration() {
        // Test that matrix operations from rustlab-math work with stats
        let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = ArrayF64::from_slice(&[2.0, 0.0, 1.0, 2.0], 2, 2).unwrap();
        
        // Matrix multiplication using rustlab-math
        let product = a ^ b;  // Uses faer's optimized matrix multiply
        
        // Statistical analysis on the result
        let flattened = VectorF64::from_slice(product.as_slice_unchecked());
        let stats_mean = flattened.mean();
        let stats_std = flattened.std(None);
        
        assert!(stats_mean > 0.0);
        assert!(stats_std > 0.0);
    }
    
    #[test]
    fn test_type_conversion_compatibility() {
        // Test conversions between different numeric types
        let data_f64 = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let values_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data_f32 = VectorF32::from_slice(&values_f32);
        
        // Both should support the same operations
        let mean_f64 = data_f64.mean();
        let mean_f32 = data_f32.mean();
        
        let median_f64 = data_f64.median();
        let median_f32 = data_f32.median();
        
        assert!((mean_f64 - mean_f32 as f64).abs() < 1e-6);
        assert!((median_f64 - median_f32 as f64).abs() < 1e-6);
    }
    
    #[test]
    fn test_error_propagation() {
        // Test that errors are handled consistently across the ecosystem
        let empty_vector = VectorF64::zeros(0);
        
        // rustlab-math operations should panic (math-first philosophy)
        let result = std::panic::catch_unwind(|| {
            empty_vector.mean()
        });
        assert!(result.is_err());
        
        // rustlab-stats should follow the same pattern
        let result = std::panic::catch_unwind(|| {
            empty_vector.median()
        });
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod performance_comparison {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_performance_vs_basic_implementation() {
        let size = 10_000;
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let vector = VectorF64::from_slice(&data);
        
        // Time our optimized median
        let start = Instant::now();
        let median1 = vector.median();
        let duration1 = start.elapsed();
        
        // Time naive implementation
        let start = Instant::now();
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median2 = sorted[size / 2];
        let duration2 = start.elapsed();
        
        println!("Optimized median: {:?}, Naive: {:?}", duration1, duration2);
        assert_eq!(median1, median2);
        
        // Our implementation should be faster (uses quickselect)
        assert!(duration1 < duration2);
    }
    
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_performance_scaling() {
        // Test that parallel operations scale properly
        let sizes = vec![1_000, 10_000, 100_000];
        
        for size in sizes {
            let data = VectorF64::ones(size);
            
            // Serial
            let start = Instant::now();
            let mean_serial = data.mean();
            let serial_time = start.elapsed();
            
            // Parallel
            let start = Instant::now();
            let mean_parallel = data.mean_parallel();
            let parallel_time = start.elapsed();
            
            println!("Size {}: Serial {:?}, Parallel {:?}", 
                     size, serial_time, parallel_time);
            
            assert!((mean_serial - mean_parallel).abs() < 1e-10);
            
            // For large sizes, parallel should be faster
            if size >= 10_000 {
                assert!(parallel_time < serial_time);
            }
        }
    }
}

#[cfg(test)]
mod migration_compatibility {
    use super::*;
    
    #[test]
    fn test_api_migration_from_statistics() {
        // Simulate migration from rustlab-statistics
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Old API (rustlab-statistics style with stat_ prefix)
        // let old_mean = data.stat_mean();  // Would have used prefix
        
        // New API (clean, no prefix needed)
        let new_mean = data.mean();  // From rustlab-math
        let new_median = data.median();  // From rustlab-stats
        let new_skewness = data.skewness();  // From rustlab-stats
        
        // Verify all methods work without prefix
        assert_eq!(new_mean, 3.0);
        assert_eq!(new_median, 3.0);
        assert_eq!(new_skewness, 0.0);
    }
    
    #[test]
    fn test_numerical_accuracy_compatibility() {
        // Ensure numerical accuracy matches expectations
        let data = vec64![1e-10, 1e10, 1e-10, 1e10];
        
        // Test numerical stability
        let mean = data.mean();
        let var = data.var(None);
        
        // Should handle extreme values correctly
        assert!(mean > 0.0);
        assert!(var > 0.0);
        assert!(!mean.is_nan());
        assert!(!var.is_nan());
    }
}

#[cfg(test)]
mod comprehensive_workflow {
    use super::*;
    
    #[test]
    fn test_complete_statistical_workflow() {
        // Simulate a complete data analysis workflow
        
        // 1. Create data using rustlab-math
        let x = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec64![2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        
        // 2. Basic descriptive statistics
        println!("X mean: {}, std: {}", x.mean(), x.std(None));
        println!("Y mean: {}, std: {}", y.mean(), y.std(None));
        
        // 3. Advanced statistics
        println!("X median: {}, IQR: {}", x.median(), x.iqr());
        println!("X skewness: {}, kurtosis: {}", x.skewness(), x.kurtosis());
        
        // 4. Correlation analysis
        let corr = x.pearson_correlation(&y);
        let spearman = x.spearman_correlation(&y);
        println!("Pearson: {}, Spearman: {}", corr, spearman);
        
        // 5. Hypothesis testing
        let t_test = x.ttest_2samp(&y, true, Alternative::TwoSided);
        println!("T-test p-value: {}", t_test.p_value);
        
        // 6. Data normalization
        let x_normalized = x.zscore(None);
        let y_normalized = y.zscore(None);
        
        // 7. Performance optimization (if parallel enabled)
        #[cfg(feature = "parallel")]
        {
            let large_x = VectorF64::ones(100_000);
            let large_y = VectorF64::ones(100_000);
            let corr_parallel = large_x.correlation_parallel(&large_y, CorrelationMethod::Pearson);
            println!("Large correlation (parallel): {}", corr_parallel);
        }
        
        // Verify the workflow completes successfully
        assert!(corr > 0.9);  // Strong positive correlation
        assert!(t_test.p_value > 0.05);  // Not significantly different
        assert!((x_normalized.mean()).abs() < 1e-10);  // Normalized to zero mean
    }
    
    #[test]
    fn test_array_workflow() {
        // Create a 2D dataset
        let data = ArrayF64::from_slice(&[
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
            3.0, 6.0, 9.0, 12.0,
            4.0, 8.0, 12.0, 16.0
        ], 4, 4).unwrap();
        
        // Column-wise statistics
        let col_means = data.mean_axis(Axis::Rows).unwrap();
        let col_medians = data.median_axis(Axis::Rows);
        let col_iqrs = data.iqr_axis(Axis::Rows);
        
        // Row-wise normalization
        let normalized = data.zscore_axis(Axis::Cols, None);
        
        // Verify shapes
        assert_eq!(col_means.len(), 4);
        assert_eq!(col_medians.len(), 4);
        assert_eq!(col_iqrs.len(), 4);
        assert_eq!(normalized.shape(), (4, 4));
        
        // Verify normalization
        for i in 0..4 {
            let row_data: Vec<f64> = (0..4).map(|j| normalized.get(i, j).unwrap()).collect();
            let row_vec = VectorF64::from_slice(&row_data);
            assert!((row_vec.mean()).abs() < 1e-10);
            assert!((row_vec.std(None) - 1.0).abs() < 1e-10);
        }
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;
    
    #[test]
    fn test_single_element_operations() {
        let single = vec64![42.0];
        
        assert_eq!(single.mean(), 42.0);
        assert_eq!(single.median(), 42.0);
        assert_eq!(single.var(None), 0.0);
        assert_eq!(single.iqr(), 0.0);
        assert!(single.skewness().is_nan());  // Undefined for n=1
    }
    
    #[test]
    fn test_constant_data() {
        let constant = vec64![5.0, 5.0, 5.0, 5.0, 5.0];
        
        assert_eq!(constant.mean(), 5.0);
        assert_eq!(constant.median(), 5.0);
        assert_eq!(constant.var(None), 0.0);
        assert_eq!(constant.std(None), 0.0);
        assert_eq!(constant.iqr(), 0.0);
        
        // Z-score normalization should handle zero variance
        let result = std::panic::catch_unwind(|| {
            constant.zscore(None)
        });
        assert!(result.is_err());  // Should panic on zero variance
    }
    
    #[test]
    fn test_extreme_values() {
        let extreme = vec64![1e-308, 1e308, 1e-308, 1e308];
        
        // Should handle without overflow/underflow
        let mean = extreme.mean();
        let median = extreme.median();
        
        assert!(!mean.is_nan());
        assert!(!mean.is_infinite());
        assert!(!median.is_nan());
        assert!(!median.is_infinite());
    }
}