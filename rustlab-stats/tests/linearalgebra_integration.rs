//! Integration tests with linear algebra operations
//!
//! Tests that rustlab-stats works seamlessly with matrix operations
//! from rustlab-math (which uses faer internally).

#[cfg(test)]
mod linearalgebra_integration {
    use rustlab_stats::prelude::*;
    use rustlab_math::{ArrayF64, VectorF64, vec64};
    
    #[test]
    fn test_matrix_statistics_workflow() {
        // Create a data matrix (observations x features)
        let data = ArrayF64::from_slice(&[
            1.0, 2.0, 3.0,  // Observation 1
            4.0, 5.0, 6.0,  // Observation 2
            7.0, 8.0, 9.0,  // Observation 3
            10.0, 11.0, 12.0 // Observation 4
        ], 4, 3).unwrap();
        
        // Feature-wise statistics (along observations)
        let feature_means = data.mean_axis(0);  // Mean of each feature
        let feature_stds = data.std_axis(0, None);
        let feature_medians = data.median_axis(0);
        
        assert_eq!(feature_means.len(), 3);
        assert_eq!(feature_means.get(0).unwrap(), 5.5);  // Mean of [1, 4, 7, 10]
        assert_eq!(feature_medians.get(0).unwrap(), 5.5);
        
        // Standardize the data matrix
        let standardized = data.zscore_axis(0);
        
        // Verify standardization
        let new_means = standardized.mean_axis(0);
        let new_stds = standardized.std_axis(0, None);
        
        for i in 0..3 {
            assert!((new_means.get(i).unwrap()).abs() < 1e-10);
            assert!((new_stds.get(i).unwrap() - 1.0).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_covariance_matrix_computation() {
        // Create correlated data
        let data = ArrayF64::from_slice(&[
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0,
            4.0, 8.0,
            5.0, 10.0
        ], 5, 2).unwrap();
        
        // Extract columns as vectors
        let x = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = VectorF64::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
        
        // Compute covariance
        let cov = x.covariance(&y);
        let corr = x.pearson_correlation(&y);
        
        assert_eq!(cov, 5.0);  // Perfect linear relationship
        assert_eq!(corr, 1.0);  // Perfect correlation
        
        // Future: data.covariance_matrix() when implemented
    }
    
    #[test]
    fn test_pca_preparation() {
        // Prepare data for PCA (which would use linear algebra)
        let data = ArrayF64::from_slice(&[
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0,
            4.0, 8.0, 12.0
        ], 4, 3).unwrap();
        
        // 1. Center the data
        let means = data.mean_axis(0);
        let centered = ArrayF64::from_fn(4, 3, |i, j| {
            data.get(i, j).unwrap() - means.get(j).unwrap()
        });
        
        // 2. Standardize (optional)
        let standardized = data.zscore_axis(0);
        
        // Verify centering
        let centered_means = centered.mean_axis(0);
        for i in 0..3 {
            assert!((centered_means.get(i).unwrap()).abs() < 1e-10);
        }
        
        // 3. Would compute covariance matrix C = (1/n) * X^T * X
        // 4. Would compute eigendecomposition of C
        // 5. Project data onto principal components
    }
    
    #[test]
    fn test_matrix_vector_operations() {
        // Test statistics on matrix-vector products
        let matrix = ArrayF64::from_slice(&[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ], 3, 2).unwrap();
        
        let vector = VectorF64::from_slice(&[1.0, 2.0]);
        
        // Matrix-vector multiply (using faer internally)
        let result = ArrayF64::from_fn(3, 1, |i, _| {
            let row = matrix.row(i).unwrap();
            row[0] * vector.get(0).unwrap() + row[1] * vector.get(1).unwrap()
        });
        
        // Convert result to vector for statistics
        let result_vec = VectorF64::from_slice(&[
            result.get(0, 0).unwrap(),
            result.get(1, 0).unwrap(),
            result.get(2, 0).unwrap()
        ]);
        
        // Analyze the result
        let mean = result_vec.mean();
        let std = result_vec.std(None);
        
        assert_eq!(result_vec.get(0).unwrap(), 5.0);   // 1*1 + 2*2
        assert_eq!(result_vec.get(1).unwrap(), 11.0);  // 3*1 + 4*2
        assert_eq!(result_vec.get(2).unwrap(), 17.0);  // 5*1 + 6*2
        assert_eq!(mean, 11.0);
    }
    
    #[test]
    fn test_svd_statistics_preparation() {
        // Prepare for SVD-based analysis
        let data = ArrayF64::from_slice(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        ], 4, 3).unwrap();
        
        // Compute column-wise statistics for scaling
        let col_means = data.mean_axis(0);
        let col_stds = data.std_axis(0, None);
        
        // Scale the matrix
        let scaled = data.zscore_axis(0);
        
        // Verify scaling preserved rank (approximately)
        // All columns should have different patterns after scaling
        let col1 = VectorF64::from_slice(&[
            scaled.get(0, 0).unwrap(),
            scaled.get(1, 0).unwrap(),
            scaled.get(2, 0).unwrap(),
            scaled.get(3, 0).unwrap()
        ]);
        let col2 = VectorF64::from_slice(&[
            scaled.get(0, 1).unwrap(),
            scaled.get(1, 1).unwrap(),
            scaled.get(2, 1).unwrap(),
            scaled.get(3, 1).unwrap()
        ]);
        
        // Columns should be standardized but still perfectly correlated
        let corr = col1.pearson_correlation(&col2);
        assert!((corr - 1.0).abs() < 1e-10);  // Still perfectly correlated
    }
    
    #[test]
    fn test_distance_matrix_statistics() {
        // Create points for distance matrix computation
        let points = ArrayF64::from_slice(&[
            0.0, 0.0,   // Point 1
            1.0, 0.0,   // Point 2
            0.0, 1.0,   // Point 3
            1.0, 1.0    // Point 4
        ], 4, 2).unwrap();
        
        // Compute pairwise distances (simplified)
        let mut distances = Vec::new();
        for i in 0..4 {
            for j in (i+1)..4 {
                let xi = points.row(i).unwrap();
                let xj = points.row(j).unwrap();
                let dist = ((xi[0] - xj[0]).powi(2) + (xi[1] - xj[1]).powi(2)).sqrt();
                distances.push(dist);
            }
        }
        
        let dist_vec = VectorF64::from_slice(&distances);
        
        // Analyze distance distribution
        let mean_dist = dist_vec.mean();
        let median_dist = dist_vec.median();
        let std_dist = dist_vec.std(None);
        
        println!("Distance stats: mean={:.3}, median={:.3}, std={:.3}", 
                 mean_dist, median_dist, std_dist);
        
        // Should have 4 distances of 1.0 and 2 distances of sqrt(2)
        assert_eq!(distances.len(), 6);  // C(4,2) = 6
    }
}