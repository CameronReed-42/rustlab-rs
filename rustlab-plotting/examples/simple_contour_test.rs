//! Simple contour test to debug issues

use rustlab_plotting::{array2d_from_function, ContourBuilder, Result};
use rustlab_math::range;

fn main() -> Result<()> {
    println!("Simple contour test starting...");
    
    // Create a very small grid for testing
    let x = range!(-1.0 => 1.0, 5);
    let y = range!(-1.0 => 1.0, 5);
    
    println!("Created grid: x.len()={}, y.len()={}", x.len(), y.len());
    
    // Simple function: z = x + y
    let z = array2d_from_function(&x, &y, |x, y| x + y);
    
    println!("Created Z data: {}x{}", z.nrows(), z.ncols());
    
    // Try to create ContourBuilder
    match ContourBuilder::new(&x, &y, &z) {
        Ok(builder) => {
            println!("ContourBuilder created successfully");
            
            // Try to build the plot
            match builder.build() {
                Ok(plot) => {
                    println!("Plot built successfully");
                    println!("Test completed - contour infrastructure is working!");
                },
                Err(e) => {
                    println!("Error building plot: {}", e);
                }
            }
        },
        Err(e) => {
            println!("Error creating ContourBuilder: {}", e);
        }
    }
    
    Ok(())
}