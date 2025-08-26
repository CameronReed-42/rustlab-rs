use rustlab_optimize::prelude::*;
use rustlab_math::linspace;

fn main() -> rustlab_optimize::core::Result<()> {
    // Test the patterns from the notebook
    
    // Cell 4 pattern
    let concentrations = vec![0.1, 0.5, 1.0, 2.0, 5.0];
    let rates = vec![0.05, 0.22, 0.40, 0.71, 1.41];
    
    let concentrations_clone = concentrations.clone();
    let rates_clone = rates.clone();
    let rate_model = move |params: &[f64]| {
        let k_param = params[0];
        let n_param = params[1];
        concentrations_clone.iter().zip(rates_clone.iter())
            .map(|(&c_pt, &r_pt)| {
                let predicted = k_param * (c_pt as f64).powf(n_param);
                (r_pt - predicted).powi(2)
            })
            .sum::<f64>()
    };
    
    // Cell 5 pattern  
    let concentrations_clone2 = concentrations.clone();
    let rates_clone2 = rates.clone();
    let rate_model_unbounded = move |params: &[f64]| {
        let k_param = params[0];
        let n_param = params[1];
        concentrations_clone2.iter().zip(rates_clone2.iter())
            .map(|(&c_pt, &r_pt)| {
                let predicted = k_param * (c_pt as f64).powf(n_param);
                (r_pt - predicted).powi(2)
            })
            .sum::<f64>()
    };
    
    let _result1 = minimize(rate_model).from(&[1.0, 1.0]).solve()?;
    let _result2 = minimize(rate_model_unbounded).from(&[1.0, 1.0]).solve()?;
    
    Ok(())
}