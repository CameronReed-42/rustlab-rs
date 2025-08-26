use rustlab_math::*;

fn main() {
    println!("Testing fill function availability...");
    
    // Test 1: Try to use fill directly
    let result: ArrayF64 = fill(2, 2, 3.14);
    println!("fill(2, 2, 3.14) worked!");
    println!("Result shape: {}x{}", result.nrows(), result.ncols());
    println!("Value at (0,0): {:?}", result.get(0, 0));
}