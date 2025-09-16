use rustlab_core::*;
use crate::Plot;

// Simple test to ensure basic functionality works
pub fn test_basic_plot() -> crate::error::Result<()> {
    let x = Vec64::linspace(0.0, 10.0, 100);
    let y = x.map_f64(|x| x.sin());
    
    let plot = Plot::new()
        .line(&x, &y)
        .title("Test Plot")
        .xlabel("x")
        .ylabel("sin(x)");
    
    // Just test that we can create the plot without errors
    println!("Successfully created basic plot");
    Ok(())
}