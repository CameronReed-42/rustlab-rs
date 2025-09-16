//! Test contour plotting by saving to PNG file
//!
//! This example tests if contour plotting actually works by creating
//! a simple contour plot and saving it to a PNG file.

use rustlab_plotting::{array2d_from_function, ContourBuilder, Result};
use rustlab_math::range;

fn main() -> Result<()> {
    println!("Testing contour plot PNG export...");
    
    // Create a simple grid
    let x = range!(-2.0 => 2.0, 30);
    let y = range!(-2.0 => 2.0, 30);
    
    println!("Created grid: {}×{}", x.len(), y.len());
    
    // Simple test function: z = x² + y² (circular contours)
    let z = array2d_from_function(&x, &y, |x, y| x*x + y*y);
    
    println!("Function range: [{:.2}, {:.2}]", z.min().unwrap(), z.max().unwrap());
    
    // Try to create and save contour plot
    println!("Creating contour plot...");
    
    ContourBuilder::new(&x, &y, &z)?
        .title("Test Function: f(x,y) = x² + y²")
        .xlabel("x")
        .ylabel("y")
        .n_levels(8)
        .save("test_contour.png")?;
    
    println!("✅ Contour plot successfully saved to 'test_contour.png'");
    println!("If this message appears, contour plotting is working!");
    
    Ok(())
}