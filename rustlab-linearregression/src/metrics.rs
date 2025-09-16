//! Regression metrics with math-first design

use rustlab_math::{VectorF64, BasicStatistics};

/// Coefficient of determination R²
/// 
/// R² = 1 - SS_res / SS_tot
pub fn r2_score(y_true: &VectorF64, y_pred: &VectorF64) -> f64 {
    let residuals = y_true - y_pred;
    let squared_residuals = &residuals * &residuals;
    let ss_res = squared_residuals.sum_elements();
    
    let y_mean = y_true.mean();
    let y_centered = y_true - y_mean;
    let squared_centered = &y_centered * &y_centered;
    let ss_tot = squared_centered.sum_elements();
    
    if ss_tot == 0.0 {
        if ss_res == 0.0 { 1.0 } else { 0.0 }
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Mean Squared Error
/// 
/// MSE = (1/n) ∑(y_true - y_pred)²
pub fn mean_squared_error(y_true: &VectorF64, y_pred: &VectorF64) -> f64 {
    let residuals = y_true - y_pred;
    let squared_residuals = &residuals * &residuals;
    squared_residuals.sum_elements() / y_true.len() as f64
}

/// Root Mean Squared Error
pub fn root_mean_squared_error(y_true: &VectorF64, y_pred: &VectorF64) -> f64 {
    mean_squared_error(y_true, y_pred).sqrt()
}

/// Mean Absolute Error
/// 
/// MAE = (1/n) ∑|y_true - y_pred|
pub fn mean_absolute_error(y_true: &VectorF64, y_pred: &VectorF64) -> f64 {
    let residuals = y_true - y_pred;
    residuals.iter().map(|x| x.abs()).sum::<f64>() / y_true.len() as f64
}

/// Mean Absolute Percentage Error
pub fn mean_absolute_percentage_error(y_true: &VectorF64, y_pred: &VectorF64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    
    for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
        if *true_val != 0.0 {
            sum += ((true_val - pred_val) / true_val).abs();
            count += 1;
        }
    }
    
    if count > 0 {
        sum / count as f64 * 100.0
    } else {
        0.0
    }
}