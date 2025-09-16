//! Parallel cross-validation and hyperparameter optimization using Rayon
//! 
//! Math-first design with efficient parallel execution across CPU cores

use rayon::prelude::*;
use rustlab_math::{ArrayF64, VectorF64, BasicStatistics};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashMap;
use crate::error::{LinearRegressionError, Result};
use crate::traits::{LinearModel, FittedModel};

/// K-fold cross-validation results
#[derive(Clone, Debug)]
pub struct CrossValidationResult {
    /// Individual scores for each fold
    pub scores: VectorF64,
    /// Mean score across all folds
    pub mean_score: f64,
    /// Standard deviation of scores
    pub std_score: f64,
    /// Index of the best performing fold
    pub best_fold: usize,
}

/// Generate k-fold indices for cross-validation
pub fn kfold_indices(n_samples: usize, n_folds: usize, shuffle: bool, seed: Option<u64>) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut indices: Vec<usize> = (0..n_samples).collect();
    
    if shuffle {
        let mut rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::from_entropy()
        };
        
        // Fisher-Yates shuffle
        for i in (1..n_samples).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
    }
    
    let fold_size = n_samples / n_folds;
    let remainder = n_samples % n_folds;
    
    let mut folds = Vec::with_capacity(n_folds);
    let mut start = 0;
    
    for i in 0..n_folds {
        let fold_samples = if i < remainder {
            fold_size + 1
        } else {
            fold_size
        };
        
        let val_indices: Vec<usize> = indices[start..start + fold_samples].to_vec();
        let train_indices: Vec<usize> = indices[..start]
            .iter()
            .chain(indices[start + fold_samples..].iter())
            .copied()
            .collect();
        
        folds.push((train_indices, val_indices));
        start += fold_samples;
    }
    
    folds
}

/// Perform parallel k-fold cross-validation
/// 
/// Uses Rayon for parallel fold evaluation
pub fn cross_validate<M: LinearModel + Send + Sync>(
    model: &M,
    X: &ArrayF64,
    y: &VectorF64,
    cv_folds: usize,
    shuffle: bool,
    seed: Option<u64>,
) -> Result<CrossValidationResult> 
where
    M::Fitted: Send,
{
    if cv_folds < 2 {
        return Err(LinearRegressionError::InvalidParameter {
            name: "cv_folds".to_string(),
            value: cv_folds as f64,
            constraint: ">= 2".to_string(),
        });
    }
    
    let n_samples = X.nrows();
    let n_features = X.ncols();
    
    // Generate fold indices
    let folds = kfold_indices(n_samples, cv_folds, shuffle, seed);
    
    // Parallel cross-validation using Rayon
    let cv_scores: Result<Vec<f64>> = folds
        .par_iter()
        .map(|(train_idx, val_idx)| {
            // Create training and validation sets
            let mut X_train = ArrayF64::zeros(train_idx.len(), n_features);
            let mut y_train = VectorF64::zeros(train_idx.len());
            
            for (i, &idx) in train_idx.iter().enumerate() {
                for j in 0..n_features {
                    X_train[(i, j)] = X[(idx, j)];
                }
                y_train[i] = y[idx];
            }
            
            let mut X_val = ArrayF64::zeros(val_idx.len(), n_features);
            let mut y_val = VectorF64::zeros(val_idx.len());
            
            for (i, &idx) in val_idx.iter().enumerate() {
                for j in 0..n_features {
                    X_val[(i, j)] = X[(idx, j)];
                }
                y_val[i] = y[idx];
            }
            
            // Fit model and evaluate
            let fitted = model.fit(&X_train, &y_train)?;
            Ok(fitted.score(&X_val, &y_val))
        })
        .collect();
    
    let scores = VectorF64::from_vec(cv_scores?);
    let mean_score = scores.mean();
    let std_score = scores.std(None);
    
    let best_fold = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    
    Ok(CrossValidationResult {
        scores,
        mean_score,
        std_score,
        best_fold,
    })
}

/// Grid search with parallel cross-validation
pub struct GridSearchCV<M: LinearModel> {
    model_factory: Box<dyn Fn(&HashMap<String, f64>) -> M + Send + Sync>,
    param_grid: Vec<HashMap<String, f64>>,
    cv_folds: usize,
    shuffle: bool,
    seed: Option<u64>,
    verbose: bool,
}

impl<M: LinearModel + Send + Sync + 'static> GridSearchCV<M> 
where
    M::Fitted: Send,
{
    /// Create new grid search
    pub fn new<F>(model_factory: F, param_grid: Vec<HashMap<String, f64>>) -> Self 
    where
        F: Fn(&HashMap<String, f64>) -> M + Send + Sync + 'static,
    {
        Self {
            model_factory: Box::new(model_factory),
            param_grid,
            cv_folds: 5,
            shuffle: true,
            seed: None,
            verbose: false,
        }
    }
    
    /// Set number of CV folds
    pub fn with_cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    
    /// Perform parallel grid search
    pub fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<GridSearchResult<M>> {
        if self.param_grid.is_empty() {
            return Err(LinearRegressionError::InvalidInput(
                "Parameter grid is empty".to_string()
            ));
        }
        
        // Parallel evaluation of all parameter combinations
        let results: Vec<(HashMap<String, f64>, CrossValidationResult)> = self.param_grid
            .par_iter()
            .map(|params| {
                if self.verbose {
                    println!("Evaluating parameters: {:?}", params);
                }
                
                let model = (self.model_factory)(params);
                let cv_result = cross_validate(&model, X, y, self.cv_folds, self.shuffle, self.seed)?;
                
                if self.verbose {
                    println!("  Score: {:.4} ± {:.4}", cv_result.mean_score, cv_result.std_score);
                }
                
                Ok((params.clone(), cv_result))
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Find best parameters
        let (best_params, best_cv_result) = results
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.mean_score.partial_cmp(&b.mean_score).unwrap()
            })
            .ok_or_else(|| LinearRegressionError::InvalidInput("No valid results".to_string()))?;
        
        // Refit on full dataset with best parameters
        let best_model = (self.model_factory)(best_params);
        let best_fitted = best_model.fit(X, y)?;
        
        Ok(GridSearchResult {
            best_params: best_params.clone(),
            best_score: best_cv_result.mean_score,
            best_model: best_fitted,
            cv_results: results,
        })
    }
}

/// Grid search results
#[derive(Clone, Debug)]
pub struct GridSearchResult<M: LinearModel> {
    /// Best hyperparameters found
    pub best_params: HashMap<String, f64>,
    /// Best cross-validation score achieved
    pub best_score: f64,
    /// Model trained with best parameters
    pub best_model: M::Fitted,
    /// Results for all parameter combinations tested
    pub cv_results: Vec<(HashMap<String, f64>, CrossValidationResult)>,
}

/// Random search with parallel cross-validation
pub struct RandomSearchCV<M: LinearModel> {
    model_factory: Box<dyn Fn(&HashMap<String, f64>) -> M + Send + Sync>,
    param_distributions: HashMap<String, (f64, f64)>, // (min, max) for uniform sampling
    n_iter: usize,
    cv_folds: usize,
    shuffle: bool,
    seed: Option<u64>,
    verbose: bool,
}

impl<M: LinearModel + Send + Sync + 'static> RandomSearchCV<M>
where
    M::Fitted: Send,
{
    /// Create new random search
    pub fn new<F>(
        model_factory: F,
        param_distributions: HashMap<String, (f64, f64)>,
        n_iter: usize,
    ) -> Self
    where
        F: Fn(&HashMap<String, f64>) -> M + Send + Sync + 'static,
    {
        Self {
            model_factory: Box::new(model_factory),
            param_distributions,
            n_iter,
            cv_folds: 5,
            shuffle: true,
            seed: None,
            verbose: false,
        }
    }
    
    /// Perform parallel random search
    pub fn fit(&self, X: &ArrayF64, y: &VectorF64) -> Result<GridSearchResult<M>> {
        let mut rng = if let Some(s) = self.seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::from_entropy()
        };
        
        // Generate random parameter combinations
        let param_samples: Vec<HashMap<String, f64>> = (0..self.n_iter)
            .map(|_| {
                let mut params = HashMap::new();
                for (name, (min, max)) in &self.param_distributions {
                    let value = rng.gen_range(*min..*max);
                    params.insert(name.clone(), value);
                }
                params
            })
            .collect();
        
        // Parallel evaluation using Rayon
        let results: Vec<(HashMap<String, f64>, CrossValidationResult)> = param_samples
            .par_iter()
            .enumerate()
            .map(|(i, params)| {
                if self.verbose {
                    println!("Random search iteration {}/{}: {:?}", i + 1, self.n_iter, params);
                }
                
                let model = (self.model_factory)(params);
                let cv_result = cross_validate(&model, X, y, self.cv_folds, self.shuffle, self.seed)?;
                
                if self.verbose {
                    println!("  Score: {:.4} ± {:.4}", cv_result.mean_score, cv_result.std_score);
                }
                
                Ok((params.clone(), cv_result))
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Find best parameters
        let (best_params, best_cv_result) = results
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.mean_score.partial_cmp(&b.mean_score).unwrap()
            })
            .ok_or_else(|| LinearRegressionError::InvalidInput("No valid results".to_string()))?;
        
        // Refit on full dataset
        let best_model = (self.model_factory)(best_params);
        let best_fitted = best_model.fit(X, y)?;
        
        Ok(GridSearchResult {
            best_params: best_params.clone(),
            best_score: best_cv_result.mean_score,
            best_model: best_fitted,
            cv_results: results,
        })
    }
}

/// Helper function to create parameter grid from ranges
pub fn make_param_grid(param_ranges: HashMap<String, Vec<f64>>) -> Vec<HashMap<String, f64>> {
    if param_ranges.is_empty() {
        return vec![HashMap::new()];
    }
    
    let keys: Vec<String> = param_ranges.keys().cloned().collect();
    let values: Vec<Vec<f64>> = keys.iter().map(|k| param_ranges[k].clone()).collect();
    
    // Cartesian product of all parameter values
    let mut grid = Vec::new();
    let mut indices = vec![0; keys.len()];
    
    loop {
        let mut params = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            params.insert(key.clone(), values[i][indices[i]]);
        }
        grid.push(params);
        
        // Increment indices
        let mut carry = true;
        for i in (0..indices.len()).rev() {
            if carry {
                indices[i] += 1;
                if indices[i] >= values[i].len() {
                    indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }
        
        if carry {
            break;
        }
    }
    
    grid
}