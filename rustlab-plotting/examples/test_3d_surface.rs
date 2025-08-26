//! Test example for 3D surface plotting functionality

use rustlab_plotting::*;
use rustlab_math::*;

fn main() -> rustlab_plotting::Result<()> {
    println!("🎯 Testing 3D Surface Plotting...\n");
    
    // Create grid using math-first ergonomics
    let x = range!(0.0 => 10.0, 30);
    let y = range!(-5.0 => 5.0, 30);
    
    // Test 1: Simple saddle surface z = x² - y²
    println!("📊 Test 1: Saddle Surface (z = x² - y²)");
    let z_saddle = array2d_from_function(&x, &y, |x, y| x.powi(2) - y.powi(2));
    
    Surface3DBuilder::from_vectors(&x, &y, &z_saddle)?
        .title("Saddle Surface: z = x² - y²")
        .xlabel("X")
        .ylabel("Y")
        .zlabel("Z")
        .colormap(ColorMap::Viridis)
        .elevation(30.0)
        .azimuth(45.0)
        .save("test_3d_saddle.png")?;
    
    println!("✅ Saddle surface saved to test_3d_saddle.png");
    
    // Test 2: Sinusoidal surface z = sin(x) * cos(y)
    println!("\n📊 Test 2: Sinusoidal Surface (z = sin(x) * cos(y))");
    let z_sin = array2d_from_function(&x, &y, |x, y| x.sin() * y.cos());
    
    Surface3DBuilder::from_vectors(&x, &y, &z_sin)?
        .title("Sinusoidal Surface: z = sin(x) * cos(y)")
        .xlabel("X")
        .ylabel("Y")
        .zlabel("Z")
        .colormap(ColorMap::Plasma)
        .elevation(45.0)
        .azimuth(60.0)
        .save("test_3d_sinusoidal.png")?;
    
    println!("✅ Sinusoidal surface saved to test_3d_sinusoidal.png");
    
    // Test 3: Gaussian peak z = exp(-(x² + y²))
    println!("\n📊 Test 3: Gaussian Peak");
    let x_gauss = range!(-3.0 => 3.0, 40);
    let y_gauss = range!(-3.0 => 3.0, 40);
    let z_gauss = array2d_from_function(&x_gauss, &y_gauss, |x, y| {
        (-0.5 * (x.powi(2) + y.powi(2))).exp()
    });
    
    Surface3DBuilder::from_vectors(&x_gauss, &y_gauss, &z_gauss)?
        .title("Gaussian Peak: z = exp(-0.5(x² + y²))")
        .xlabel("X")
        .ylabel("Y")
        .zlabel("Z")
        .colormap(ColorMap::Inferno)
        .elevation(20.0)
        .azimuth(30.0)
        .save("test_3d_gaussian.png")?;
    
    println!("✅ Gaussian peak saved to test_3d_gaussian.png");
    
    // Test 4: Wireframe only
    println!("\n📊 Test 4: Wireframe Only (Paraboloid)");
    let z_paraboloid = array2d_from_function(&x, &y, |x, y| x.powi(2) + y.powi(2));
    
    Surface3DBuilder::from_vectors(&x, &y, &z_paraboloid)?
        .title("Wireframe Paraboloid: z = x² + y²")
        .xlabel("X") 
        .ylabel("Y")
        .zlabel("Z")
        .wireframe_only()
        .save("test_3d_wireframe.png")?;
    
    println!("✅ Wireframe paraboloid saved to test_3d_wireframe.png");
    
    // Test 5: Surface only (no wireframe)
    println!("\n📊 Test 5: Surface Only (Ripple)");
    let z_ripple = array2d_from_function(&x, &y, |x, y| {
        let r = (x.powi(2) + y.powi(2)).sqrt();
        if r == 0.0 { 1.0 } else { r.sin() / r }
    });
    
    Surface3DBuilder::from_vectors(&x, &y, &z_ripple)?
        .title("Ripple Surface: z = sin(r) / r")
        .xlabel("X")
        .ylabel("Y")
        .zlabel("Z")
        .colormap(ColorMap::Coolwarm)
        .surface_only()
        .save("test_3d_surface_only.png")?;
    
    println!("✅ Surface-only ripple saved to test_3d_surface_only.png");
    
    println!("\n🎉 All 3D surface tests completed successfully!");
    println!("📁 Generated files:");
    println!("   - test_3d_saddle.png");
    println!("   - test_3d_sinusoidal.png");
    println!("   - test_3d_gaussian.png");
    println!("   - test_3d_wireframe.png");
    println!("   - test_3d_surface_only.png");
    
    Ok(())
}