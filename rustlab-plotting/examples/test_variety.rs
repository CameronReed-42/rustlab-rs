//! Test that different mathematical functions create visually distinct contour plots

use rustlab_plotting::{ContourBuilder, array2d_from_function};
use rustlab_math::range;
use std::f64::consts::PI;

fn main() -> rustlab_plotting::Result<()> {
    println!("🧪 Testing contour plot variety with different mathematical functions...");
    
    // Test 1: Linear function (should be straight lines)
    println!("\n=== Test 1: Linear Function z = x + y ===");
    let x1 = range!(-2.0 => 2.0, 30);
    let y1 = range!(-2.0 => 2.0, 30);
    let z1 = array2d_from_function(&x1, &y1, |x, y| x + y);
    println!("Linear function range: [{:.2}, {:.2}]", z1.min().unwrap(), z1.max().unwrap());
    println!("z(-2,-2) = {:.2}, z(0,0) = {:.2}, z(2,2) = {:.2}", 
             z1.get(0, 0).unwrap(), z1.get(15, 15).unwrap(), z1.get(29, 29).unwrap());
    
    ContourBuilder::new(&x1, &y1, &z1)?
        .title("Linear: z = x + y")
        .n_levels(5)
        .save("test_linear.png")?;
    println!("✅ Linear contour saved to test_linear.png");
    
    // Test 2: Circular function (should be circles)
    println!("\n=== Test 2: Circular Function z = x² + y² ===");
    let x2 = range!(-3.0 => 3.0, 40);
    let y2 = range!(-3.0 => 3.0, 40);
    let z2 = array2d_from_function(&x2, &y2, |x, y| x*x + y*y);
    println!("Circular function range: [{:.2}, {:.2}]", z2.min().unwrap(), z2.max().unwrap());
    println!("z(-3,-3) = {:.2}, z(0,0) = {:.2}, z(3,3) = {:.2}", 
             z2.get(0, 0).unwrap(), z2.get(20, 20).unwrap(), z2.get(39, 39).unwrap());
    
    ContourBuilder::new(&x2, &y2, &z2)?
        .title("Circular: z = x² + y²")
        .n_levels(6)
        .save("test_circular.png")?;
    println!("✅ Circular contour saved to test_circular.png");
    
    // Test 3: Saddle function (should be hyperbolic)
    println!("\n=== Test 3: Saddle Function z = x² - y² ===");
    let x3 = range!(-2.0 => 2.0, 35);
    let y3 = range!(-2.0 => 2.0, 35);
    let z3 = array2d_from_function(&x3, &y3, |x, y| x*x - y*y);
    println!("Saddle function range: [{:.2}, {:.2}]", z3.min().unwrap(), z3.max().unwrap());
    println!("z(-2,-2) = {:.2}, z(0,0) = {:.2}, z(2,-2) = {:.2}", 
             z3.get(0, 0).unwrap(), z3.get(17, 17).unwrap(), z3.get(34, 0).unwrap());
    
    ContourBuilder::new(&x3, &y3, &z3)?
        .title("Saddle: z = x² - y²")
        .n_levels(7)
        .save("test_saddle.png")?;
    println!("✅ Saddle contour saved to test_saddle.png");
    
    // Test 4: Sinusoidal function (should be wavy)
    println!("\n=== Test 4: Sinusoidal Function z = sin(x) * cos(y) ===");
    let x4 = range!(-PI => PI, 50);
    let y4 = range!(-PI => PI, 50);
    let z4 = array2d_from_function(&x4, &y4, |x, y| x.sin() * y.cos());
    println!("Sinusoidal function range: [{:.3}, {:.3}]", z4.min().unwrap(), z4.max().unwrap());
    println!("z(-π,-π) = {:.3}, z(0,0) = {:.3}, z(π,π) = {:.3}", 
             z4.get(0, 0).unwrap(), z4.get(25, 25).unwrap(), z4.get(49, 49).unwrap());
    
    ContourBuilder::new(&x4, &y4, &z4)?
        .title("Sinusoidal: z = sin(x) * cos(y)")
        .n_levels(8)
        .save("test_sinusoidal.png")?;
    println!("✅ Sinusoidal contour saved to test_sinusoidal.png");
    
    // Test 5: Exponential decay with oscillation
    println!("\n=== Test 5: Exponential Decay z = e^(-r²) * cos(3θ) ===");
    let x5 = range!(-2.0 => 2.0, 45);
    let y5 = range!(-2.0 => 2.0, 45);
    let z5 = array2d_from_function(&x5, &y5, |x, y| {
        let r = (x*x + y*y).sqrt();
        let theta = y.atan2(x);
        (-r*r).exp() * (3.0 * theta).cos()
    });
    println!("Exponential function range: [{:.4}, {:.4}]", z5.min().unwrap(), z5.max().unwrap());
    println!("z(-2,0) = {:.4}, z(0,0) = {:.4}, z(2,0) = {:.4}", 
             z5.get(22, 0).unwrap(), z5.get(22, 22).unwrap(), z5.get(22, 44).unwrap());
    
    ContourBuilder::new(&x5, &y5, &z5)?
        .title("Exponential: z = e^(-r²) * cos(3θ)")
        .n_levels(10)
        .save("test_exponential.png")?;
    println!("✅ Exponential contour saved to test_exponential.png");
    
    println!("\n🎯 All test plots saved! Each should show distinctly different contour patterns:");
    println!("   📏 Linear: Straight parallel diagonal lines");
    println!("   ⭕ Circular: Concentric circles centered at origin");
    println!("   ⚡ Saddle: Hyperbolic curves crossing at origin");
    println!("   🌊 Sinusoidal: Wave-like grid patterns");
    println!("   🌸 Exponential: Flower-like petals with exponential decay");
    
    // List the files created
    println!("\n📁 Generated files:");
    println!("   - test_linear.png");
    println!("   - test_circular.png");
    println!("   - test_saddle.png");
    println!("   - test_sinusoidal.png");
    println!("   - test_exponential.png");
    
    Ok(())
}