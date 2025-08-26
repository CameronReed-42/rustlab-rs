//! Contour plot demonstration
//!
//! This example shows how to create contour plots using rustlab-plotting's
//! new contour plot functionality.

use rustlab_plotting::{contour_plot, array2d_from_function, ContourBuilder, Result};
use rustlab_math::{range, VectorF64};

fn main() -> Result<()> {
    println!("Contour Plot Demo");
    
    // Create a 2D grid
    let x = range!(-2.0 => 2.0, 20);
    let y = range!(-2.0 => 2.0, 20);
    
    // Test function 1: Saddle function z = x² - y²
    let z_saddle = array2d_from_function(&x, &y, |x, y| x*x - y*y);
    
    println!("Creating saddle function contour plot...");
    contour_plot(&x, &y, &z_saddle)?;
    
    // Test function 2: Gaussian function
    let z_gaussian = array2d_from_function(&x, &y, |x, y| {
        let r_squared = x*x + y*y;
        (-r_squared).exp()
    });
    
    println!("Creating Gaussian contour plot...");
    ContourBuilder::new(&x, &y, &z_gaussian)?
        .title("Gaussian Function: exp(-(x² + y²))")
        .xlabel("x")
        .ylabel("y")
        .colorbar(true)
        .n_levels(8)
        .show()?;
    
    // Test function 3: Optimization landscape
    let z_optimization = array2d_from_function(&x, &y, |x, y| {
        // Rosenbrock function (shifted)
        let a = 1.0;
        let b = 100.0;
        let x_shift = x + 1.0;
        let y_shift = y + 1.0;
        (a - x_shift).powi(2) + b * (y_shift - x_shift.powi(2)).powi(2)
    });
    
    println!("Creating optimization landscape contour plot...");
    ContourBuilder::new(&x, &y, &z_optimization)?
        .title("Rosenbrock Function")
        .xlabel("x")
        .ylabel("y")
        .filled(true)
        .colorbar(true)
        .show()?;
    
    println!("Contour plot demo completed!");
    
    Ok(())
}