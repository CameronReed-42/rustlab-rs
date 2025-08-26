# RustLab Math - Complete AI Documentation

## Critical Information for AI Code Generation

This comprehensive documentation covers ALL functionality in rustlab-math, including the newly added row/column extraction and vector-to-matrix conversion methods.

### 🔴 CRITICAL: Most Important Rules

1. **Operator Distinction**: 
   - `^` = Mathematical multiplication (matrix/dot product)
   - `*` = Element-wise multiplication (Hadamard product)

2. **Type System**:
   - Default to `ArrayF64` / `VectorF64` for scientific computing
   - Use type aliases, not generic types: `ArrayF64` not `Array<f64>`

3. **New Column/Row Methods** (Added to rustlab-linearregression):
   - `array.col(index)` - Extract column as owned Vector
   - `array.row(index)` - Extract row as owned Vector  
   - `array.col_view(index)` - Zero-copy column view
   - `array.row_view(index)` - Zero-copy row view
   - `ArrayF64::from_vector_column(&vector)` - Convert vector to n×1 matrix
   - `array.to_vector_column()` - Extract first column as vector

---

## Core Data Structures

### Array<T> - 2D Matrices

```rust
use rustlab_math::{ArrayF64, VectorF64, array64};

// Creation
let A = ArrayF64::zeros(3, 4);           // 3×4 zero matrix
let B = ArrayF64::ones(2, 2);            // 2×2 ones matrix  
let I = ArrayF64::eye(5);                // 5×5 identity matrix
let M = array64![[1.0, 2.0], [3.0, 4.0]]; // From literal

// Element access
let value = A.get(0, 0);                 // Safe access (returns Option)
let value = A[(0, 0)];                   // Direct access (can panic)
A.set(0, 0, 5.0)?;                       // Safe mutation

// Shape information
let (rows, cols) = A.shape();            // Get dimensions
let nrows = A.nrows();                   // Number of rows
let ncols = A.ncols();                   // Number of columns
```

### Vector<T> - 1D Vectors

```rust
use rustlab_math::{VectorF64, vec64};

// Creation
let v = VectorF64::zeros(100);           // 100-element zero vector
let u = VectorF64::ones(50);             // 50-element ones vector
let w = vec64![1.0, 2.0, 3.0, 4.0];      // From literal

// Element access
let value = v.get(0);                    // Safe access (returns Option)
let value = v[0];                        // Direct access (can panic)
v.set(0, 5.0)?;                          // Safe mutation

// Properties
let len = v.len();                       // Vector length
let is_empty = v.is_empty();             // Check if empty
```

---

## 🆕 Row and Column Operations (NEW!)

### Extracting Rows and Columns

These methods were added specifically for rustlab-linearregression to enable efficient feature extraction:

```rust
use rustlab_math::{ArrayF64, VectorF64, array64};

let matrix = array64![[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]];

// OWNED extraction (creates new Vector with copied data)
let col1 = matrix.col(1).unwrap();       // Extract column 1: [2.0, 5.0, 8.0]
let row0 = matrix.row(0).unwrap();       // Extract row 0: [1.0, 2.0, 3.0]

// ZERO-COPY views (efficient, no allocation)
let col_view = matrix.col_view(1)?;      // View of column 1
let row_view = matrix.row_view(0)?;      // View of row 0

// Use in computations
let col_mean = col1.mean();              // Statistics on extracted column
let row_sum = row0.sum_elements();       // Sum of row elements
```

### Vector to Matrix Conversion

Critical for linear algebra operations in rustlab-linearregression:

```rust
use rustlab_math::{ArrayF64, VectorF64, vec64};

let vector = vec64![1.0, 2.0, 3.0];

// Convert vector to column matrix (n×1)
let col_matrix = ArrayF64::from_vector_column(&vector);
assert_eq!(col_matrix.shape(), (3, 1));

// Use in matrix multiplication
let A = ArrayF64::ones(2, 3);
let result = &A ^ &col_matrix;           // (2×3) × (3×1) → (2×1)

// Extract column back to vector
let extracted = col_matrix.to_vector_column();
assert_eq!(extracted.len(), 3);
```

### Why These Methods Were Added

These methods solve specific problems in linear regression:

1. **Column extraction** for feature analysis in datasets
2. **Row extraction** for sample-wise operations
3. **Vector-to-matrix conversion** for solving normal equations
4. **Zero-copy views** for memory efficiency with large datasets

---

## Mathematical Operations

### Matrix Multiplication (^ operator)

```rust
use rustlab_math::{ArrayF64, VectorF64, array64, vec64};

let A = array64![[1.0, 2.0], [3.0, 4.0]];
let B = array64![[2.0, 1.0], [1.0, 2.0]];
let v = vec64![1.0, 2.0];

// Matrix operations using ^
let C = &A ^ &B;                         // Matrix × Matrix
let u = &A ^ &v;                         // Matrix × Vector  
let dot = &v ^ &v;                       // Vector dot product
let w = &v ^ &A;                         // Vector × Matrix

// Using new column methods in linear algebra
let X = array64![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];  // 3×2 design matrix
let y = vec64![1.0, 2.0, 3.0];                         // 3×1 target vector

// Convert to column matrix for normal equations
let y_matrix = ArrayF64::from_vector_column(&y);       // 3×1 matrix
let Xt = X.transpose();                                // 2×3
let XtX = &Xt ^ &X;                                    // 2×2
let Xty = &Xt ^ &y_matrix;                             // 2×1
let beta = Xty.to_vector_column();                     // Extract solution as vector
```

### Element-wise Operations (* operator)

```rust
let A = array64![[1.0, 2.0], [3.0, 4.0]];
let B = array64![[2.0, 1.0], [1.0, 2.0]];

// Element-wise operations using *
let hadamard = &A * &B;                  // Element-wise multiplication
let sum = &A + &B;                       // Element-wise addition
let diff = &A - &B;                      // Element-wise subtraction
let scaled = &A * 2.0;                   // Scalar multiplication
```

---

## Statistical Operations with Row/Column Methods

### Using Column Extraction for Feature Statistics

```rust
use rustlab_math::{ArrayF64, array64};

// Dataset: samples × features
let data = array64![[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]];

// Extract and analyze individual features
for j in 0..data.ncols() {
    let feature = data.col(j).unwrap();
    let mean = feature.mean();
    let std = feature.std(None);
    println!("Feature {}: mean={:.2}, std={:.2}", j, mean, std);
}

// Extract specific feature for processing
let feature_1 = data.col(1).unwrap();
let normalized = (&feature_1 - feature_1.mean()) / feature_1.std(None);
```

### Using Row Extraction for Sample Operations

```rust
// Extract and process individual samples
for i in 0..data.nrows() {
    let sample = data.row(i).unwrap();
    let sample_sum = sample.sum_elements();
    let sample_norm = (&sample ^ &sample).sqrt();  // L2 norm
    println!("Sample {}: sum={:.2}, norm={:.2}", i, sample_sum, sample_norm);
}
```

---

## Slicing Operations

### Vector Slicing

```rust
let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];

// Natural slicing (zero-copy)
let slice = &v[1..4];                    // [2.0, 3.0, 4.0]

// Ergonomic slicing (owned)
let owned = v.slice_at(1..4)?;           // Owned VectorF64
let last_two = v.slice_at(-2..)?;        // Negative indexing
```

### Array Slicing with Row/Column Views

```rust
let A = array64![[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]];

// 2D slicing
let sub = A.slice_2d_at((1..3, 0..2))?;  // Submatrix

// Row/column slicing using new methods
let row_slice = A.row_view(1)?;          // Zero-copy view of row 1
let col_slice = A.col_view(2)?;          // Zero-copy view of column 2

// Convert views to owned when needed
let owned_row = A.row(1).unwrap();       // Owned copy of row 1
let owned_col = A.col(2).unwrap();       // Owned copy of column 2
```

---

## File I/O Operations

### Simple I/O API (Only 4 Functions)

```rust
use rustlab_math::{ArrayF64, VectorF64};
use rustlab_math::io::MathIO;

let A = array64![[1.0, 2.0], [3.0, 4.0]];
let v = vec64![1.0, 2.0, 3.0];

// SAVE (2 functions)
A.save("matrix.csv")?;                   // Basic save
A.save_with_precision("data.csv", 10)?;  // High precision

// LOAD (2 functions)
let B = ArrayF64::load("matrix.csv")?;   // Basic load
let C = ArrayF64::load_skip("data.csv", 3)?; // Skip headers

// Same for vectors
v.save("vector.txt")?;
let w = VectorF64::load("vector.txt")?;
```

---

## Complete Example: Linear Regression with New Methods

This example shows how the new row/col methods are used in practice:

```rust
use rustlab_math::{ArrayF64, VectorF64, array64, vec64};

// Design matrix X and target vector y
let X = array64![[1.0, 2.0, 3.0],
                 [1.0, 4.0, 5.0],
                 [1.0, 6.0, 7.0],
                 [1.0, 8.0, 9.0]];  // 4×3 with intercept column
let y = vec64![2.0, 4.0, 6.0, 8.0];

// Compute normal equations: β = (X'X)^(-1)X'y
let Xt = X.transpose();

// Method 1: Using column extraction (original approach)
let mut xty = VectorF64::zeros(Xt.nrows());
for i in 0..Xt.nrows() {
    let row_i = Xt.row(i).unwrap();
    xty[i] = &row_i ^ &y;  // Dot product
}

// Method 2: Using vector-to-matrix conversion (NEW approach)
let y_matrix = ArrayF64::from_vector_column(&y);  // Convert to 4×1 matrix
let xty_matrix = &Xt ^ &y_matrix;                 // Matrix multiplication: 3×1
let xty_vec = xty_matrix.to_vector_column();      // Extract as vector

// Both methods produce the same result!

// Solve for coefficients
let xtx = &Xt ^ &X;                               // 3×3 matrix
let xtx_inv = xtx.inv()?;                         // Using rustlab-linearalgebra
let beta_matrix = &xtx_inv ^ &xty_matrix;         // 3×1 matrix
let beta = beta_matrix.to_vector_column();        // Extract coefficients

// Extract individual coefficients
let intercept = beta[0];
let coef1 = beta[1];
let coef2 = beta[2];

// Make predictions
let X_test = array64![[1.0, 10.0, 11.0]];         // New sample
let y_pred_matrix = &X_test ^ &ArrayF64::from_vector_column(&beta);
let y_pred = y_pred_matrix.to_vector_column()[0]; // Extract prediction
```

---

## Performance Considerations

### When to Use Each Method

| Operation | Use Case | Performance |
|-----------|----------|-------------|
| `array.col(i)` | Need owned Vector for further processing | O(n) copy |
| `array.col_view(i)` | Temporary access, no modification | O(1) zero-copy |
| `array.row(i)` | Need owned Vector for storage | O(m) copy |
| `array.row_view(i)` | Iteration without ownership | O(1) zero-copy |
| `from_vector_column` | Convert vector for matrix operations | O(n) copy |
| `to_vector_column` | Extract result from matrix operation | O(n) copy |

### Memory Efficiency Tips

```rust
// ✅ GOOD: Use views for temporary operations
let data = ArrayF64::ones(10000, 100);
for i in 0..data.ncols() {
    let col_view = data.col_view(i)?;     // Zero-copy
    // Process without allocation
}

// ❌ BAD: Creating unnecessary copies
for i in 0..data.ncols() {
    let col = data.col(i).unwrap();       // Allocates every iteration!
    // Process with allocation overhead
}

// ✅ GOOD: Convert once when needed for multiple operations
let y_matrix = ArrayF64::from_vector_column(&y);  // Single conversion
let result1 = &A ^ &y_matrix;
let result2 = &B ^ &y_matrix;
let result3 = &C ^ &y_matrix;
```

---

## Common Patterns

### Feature Extraction and Analysis

```rust
// Extract features from dataset
let dataset = ArrayF64::load("data.csv")?;
let features = (0..5).map(|i| dataset.col(i).unwrap()).collect::<Vec<_>>();

// Compute feature statistics
let feature_stats: Vec<(f64, f64)> = features.iter()
    .map(|f| (f.mean(), f.std(None)))
    .collect();
```

### Sample Processing

```rust
// Process samples individually
let predictions = (0..dataset.nrows())
    .map(|i| {
        let sample = dataset.row(i).unwrap();
        model.predict(&sample)  // Some prediction function
    })
    .collect::<Vec<_>>();
```

### Matrix Assembly from Vectors

```rust
// Build design matrix from feature vectors
let features = vec![vec64![1.0, 2.0], vec64![3.0, 4.0], vec64![5.0, 6.0]];
let n_samples = features[0].len();
let n_features = features.len();

let mut X = ArrayF64::zeros(n_samples, n_features);
for (j, feature) in features.iter().enumerate() {
    for (i, &value) in feature.iter().enumerate() {
        X[(i, j)] = value;
    }
}
```

---

## Error Handling

All operations that can fail return `Result<T>`:

```rust
use rustlab_math::Result;

fn process_data(data: &ArrayF64) -> Result<VectorF64> {
    let col = data.col(0)
        .ok_or("Column 0 not found")?;
    
    let normalized = (&col - col.mean()) / col.std(None);
    
    Ok(normalized)
}

// Use with ? operator or match
match process_data(&matrix) {
    Ok(result) => println!("Success: {:?}", result),
    Err(e) => eprintln!("Error: {}", e),
}
```

---

## Summary of New Additions

The following methods were added to support rustlab-linearregression:

1. **`Array::row(index)`** - Extract row as owned Vector
2. **`Array::col(index)`** - Extract column as owned Vector  
3. **`Array::row_view(index)`** - Zero-copy row view
4. **`Array::col_view(index)`** - Zero-copy column view
5. **`Array::from_vector_column(&vector)`** - Convert Vector to n×1 matrix
6. **`Array::to_vector_column()`** - Extract first column as Vector

These additions enable:
- Efficient feature extraction from data matrices
- Sample-wise operations in machine learning
- Seamless vector-matrix conversions for linear algebra
- Memory-efficient views for large datasets
- Natural mathematical operations in linear regression

All methods follow RustLab's math-first design philosophy and integrate seamlessly with the existing API.