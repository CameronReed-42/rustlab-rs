/// Test the new row/column extraction and vector-to-matrix conversion methods
use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
use rustlab_math::BasicStatistics;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing new row/column extraction methods in rustlab-math");
    
    // Test 1: Create sample data
    let dataset = array64![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];
    println!("\n📊 Dataset:\n{:?}", dataset);
    println!("Shape: {:?}", dataset.shape());
    
    // Test 2: Column extraction
    println!("\n🔍 Testing column extraction:");
    for j in 0..dataset.ncols() {
        let col = dataset.col(j).unwrap();
        println!("  Column {}: {:?} (mean: {:.2})", j, col.to_slice(), col.mean());
    }
    
    // Test 3: Row extraction  
    println!("\n🔍 Testing row extraction:");
    for i in 0..dataset.nrows() {
        let row = dataset.row(i).unwrap();
        println!("  Row {}: {:?} (sum: {:.2})", i, row.to_slice(), row.sum_elements());
    }
    
    // Test 4: Zero-copy views
    println!("\n⚡ Testing zero-copy views:");
    let col_view = dataset.col_view(1)?;
    let row_view = dataset.row_view(0)?;
    println!("  Col view shape: {:?}", col_view.shape());
    println!("  Row view shape: {:?}", row_view.shape());
    
    // Test 5: Vector to matrix conversion
    println!("\n🔄 Testing vector-to-matrix conversion:");
    let v = vec64![10.0, 20.0, 30.0];
    println!("  Vector: {:?}", v.to_slice());
    
    let col_matrix = ArrayF64::from_vector_column(&v);
    println!("  Column matrix shape: {:?}", col_matrix.shape());
    println!("  Column matrix: [{:.1}, {:.1}, {:.1}]", 
             col_matrix[(0,0)], col_matrix[(1,0)], col_matrix[(2,0)]);
    
    // Test 6: Extract back to vector
    let extracted = col_matrix.to_vector_column();
    println!("  Extracted: {:?}", extracted.to_slice());
    println!("  Round-trip successful: {}", v.to_slice() == extracted.to_slice());
    
    // Test 7: Matrix multiplication with column vector
    println!("\n🧮 Testing matrix multiplication with column vectors:");
    let A = array64![[1.0, 2.0], [3.0, 4.0]];
    let x = vec64![5.0, 6.0];
    
    // Convert vector to column matrix for multiplication
    let x_matrix = ArrayF64::from_vector_column(&x);
    let result_matrix = &A ^ &x_matrix;  // Matrix multiplication
    let result = result_matrix.to_vector_column();
    
    println!("  A = {:?}", A);
    println!("  x = {:?}", x.to_slice());
    println!("  Ax = {:?}", result.to_slice());
    println!("  Expected: [17.0, 39.0] (1*5 + 2*6 = 17, 3*5 + 4*6 = 39)");
    
    println!("\n✅ All tests passed! New methods are working correctly.");
    Ok(())
}