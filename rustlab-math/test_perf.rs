use rustlab_math::{vectorize, Complexity};
use std::time::Instant;

fn main() {
    let test_data: Vec<f64> = (0..10000).map(|x| x as f64 / 1000.0).collect();
    
    // Check what complexity level triggers parallelization
    println!("Testing with {} elements", test_data.len());
    println!("Simple complexity should_parallelize: {}", 
             Complexity::Simple.should_parallelize(test_data.len()));
    
    // Warm up
    for _ in 0..10 {
        let _: Vec<f64> = test_data.iter().map(|x| x.sin() * x.cos()).collect();
    }
    
    // Test 1: Pure serial
    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _: Vec<f64> = test_data.iter().map(|x| x.sin() * x.cos()).collect();
        times.push(start.elapsed().as_nanos());
    }
    let avg_serial = times.iter().sum::<u128>() / times.len() as u128;
    
    // Test 2: Serial macro
    times.clear();
    for _ in 0..100 {
        let start = Instant::now();
        let _: Vec<f64> = vectorize![serial: x.sin() * x.cos(), for x in &test_data];
        times.push(start.elapsed().as_nanos());
    }
    let avg_serial_macro = times.iter().sum::<u128>() / times.len() as u128;
    
    // Test 3: Auto
    times.clear();
    for _ in 0..100 {
        let start = Instant::now();
        let _: Vec<f64> = vectorize![x.sin() * x.cos(), for x in &test_data];
        times.push(start.elapsed().as_nanos());
    }
    let avg_auto = times.iter().sum::<u128>() / times.len() as u128;
    
    println!("\nAverage times (100 iterations):");
    println!("Pure serial:       {} ns", avg_serial);
    println!("Serial macro:      {} ns ({:+.1}%)", avg_serial_macro, 
             (avg_serial_macro as f64 - avg_serial as f64) / avg_serial as f64 * 100.0);
    println!("Auto:              {} ns ({:+.1}%)", avg_auto,
             (avg_auto as f64 - avg_serial as f64) / avg_serial as f64 * 100.0);
}