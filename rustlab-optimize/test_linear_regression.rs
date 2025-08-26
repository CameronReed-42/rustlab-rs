use rustlab_optimize::prelude::*;
use rustlab_math::{VectorF64, vec64};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing linear regression cell...");

    // Create synthetic linear data: y = 2.5x + 1.2 + noise
    fn create_linear_data() -> (VectorF64, VectorF64) {
        let x_values = vec64![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y_values = vec64![1.3, 3.6, 6.1, 8.7, 11.2, 13.8]; // ~2.5x + 1.2 with small noise
        (x_values, y_values)
    }

    let (x_data, y_data) = create_linear_data();

    // Define linear model: f(x, params) -> y
    let linear_model = |x: f64, params: &[f64]| {
        let m = params[0];  // slope
        let b = params[1];  // intercept
        m * x + b
    };

    // Use least_squares for proper Levenberg-Marquardt interface
    let result = least_squares(&x_data, &y_data, linear_model)
        .with_initial(&[1.0, 0.0])  // Initial guess
        .solve()?;

    println!("Result fields: {:?}", result);
    
    Ok(())
}
