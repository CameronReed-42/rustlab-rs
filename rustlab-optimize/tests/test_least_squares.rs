use rustlab_optimize::prelude::*;

#[test]
fn test_least_squares_exponential() {
    // Test data: exponential decay y = 10 * exp(-0.5 * x)
    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y_data = vec![10.0, 6.065, 3.679, 2.231, 1.353];
    
    // Define residual function for exponential model
    let residual_function = move |params: &[f64]| -> Vec<f64> {
        let a = params[0];  // amplitude
        let k = params[1];  // decay rate
        
        x_data.iter().zip(y_data.iter())
            .map(|(&x, &y_obs)| {
                let y_pred = a * (-k * x).exp();
                y_obs - y_pred  // residual = observed - predicted
            })
            .collect()
    };
    
    // Fit using least_squares with Levenberg-Marquardt
    let result = least_squares(residual_function)
        .from(&[8.0, 0.3])  // Initial guess
        .using_levenberg_marquardt()
        .solve()
        .unwrap();
    
    // Check accuracy
    let a_error = (result.solution[0] - 10.0).abs();
    let k_error = (result.solution[1] - 0.5).abs();
    
    println!("Fitted: A={:.3}, k={:.3}", result.solution[0], result.solution[1]);
    println!("Errors: A={:.3}, k={:.3}", a_error, k_error);
    
    assert!(a_error < 0.1, "Amplitude error too large: {}", a_error);
    assert!(k_error < 0.1, "Decay rate error too large: {}", k_error);
}

#[test]  
fn test_least_squares_linear() {
    // Test data: linear y = 2x + 1
    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    
    // Define residual function for linear model
    let residual_function = move |params: &[f64]| -> Vec<f64> {
        let m = params[0];  // slope
        let b = params[1];  // intercept
        
        x_data.iter().zip(y_data.iter())
            .map(|(&x, &y_obs)| {
                let y_pred = m * x + b;
                y_obs - y_pred  // residual = observed - predicted
            })
            .collect()
    };
    
    // Fit using least_squares with Levenberg-Marquardt
    let result = least_squares(residual_function)
        .from(&[1.0, 0.0])  // Initial guess
        .using_levenberg_marquardt()
        .solve()
        .unwrap();
    
    // Check accuracy
    let m_error = (result.solution[0] - 2.0).abs();
    let b_error = (result.solution[1] - 1.0).abs();
    
    println!("Fitted: m={:.3}, b={:.3}", result.solution[0], result.solution[1]);
    
    assert!(m_error < 1e-6, "Slope error too large: {}", m_error);
    assert!(b_error < 1e-6, "Intercept error too large: {}", b_error);
}