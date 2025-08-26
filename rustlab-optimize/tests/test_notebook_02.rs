use rustlab_optimize::prelude::*;
use rustlab_math::linspace;

#[test]
fn test_linear_regression() {
    // Simple linear regression test
    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];  // y = 2x + 1
    
    let linear_model = move |params: &[f64]| {
        let m = params[0];
        let b = params[1];
        
        x_data.iter().zip(y_data.iter())
            .map(|(&x, &y)| {
                let y_pred = m * x + b;
                (y - y_pred).powi(2)
            })
            .sum::<f64>()
    };
    
    let result = minimize(linear_model)
        .from(&[1.0, 0.0])
        .solve()
        .unwrap();
    
    assert!((result.solution[0] - 2.0).abs() < 1e-6);  // slope
    assert!((result.solution[1] - 1.0).abs() < 1e-6);  // intercept
}

#[test]
fn test_exponential_with_bounds() {
    let t = linspace(0.0, 3.0, 10);
    let y_data: Vec<f64> = t.to_vec()
        .iter()
        .map(|&ti| 10.0 * (-0.5 * ti).exp())
        .collect();
    
    let exp_model = move |params: &[f64]| {
        let a = params[0];
        let k = params[1];
        
        t.to_vec().iter().zip(y_data.iter())
            .map(|(&ti, &yi)| {
                let y_pred = a * (-k * ti).exp();
                (yi - y_pred).powi(2)
            })
            .sum::<f64>()
    };
    
    let result = minimize(exp_model)
        .from(&[8.0, 0.3])
        .bounds(&[5.0, 0.1], &[15.0, 1.0])
        .solve()
        .unwrap();
    
    // The fit should be reasonable within the bounds
    assert!(result.solution[0] >= 5.0 && result.solution[0] <= 15.0);  // within amplitude bounds
    assert!(result.solution[1] >= 0.1 && result.solution[1] <= 1.0);   // within decay bounds
    assert!((result.solution[0] - 10.0).abs() < 2.0);  // amplitude roughly correct
    assert!((result.solution[1] - 0.5).abs() < 0.5);   // decay rate roughly correct
}

#[test]
fn test_polynomial_fitting() {
    let x = vec![-1.0, 0.0, 1.0, 2.0];
    let y = vec![3.0, 1.0, 3.0, 9.0];  // y = 2x² + 1
    
    let poly_model = move |params: &[f64]| {
        x.iter().zip(y.iter())
            .map(|(&xi, &yi)| {
                let y_pred = params[0] + params[1] * xi + params[2] * xi.powi(2);
                (yi - y_pred).powi(2)
            })
            .sum::<f64>()
    };
    
    let result = minimize(poly_model)
        .from(&[0.0, 0.0, 1.0])
        .solve()
        .unwrap();
    
    assert!((result.solution[0] - 1.0).abs() < 0.1);  // constant term
    assert!(result.solution[1].abs() < 0.1);          // linear term (should be ~0)
    assert!((result.solution[2] - 2.0).abs() < 0.1);  // quadratic term
}