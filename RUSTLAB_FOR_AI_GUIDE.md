# RustLab for AI Code Generation Guide

## Quick Reference for AI Code Generators

This guide provides essential RustLab conventions and patterns for AI code generation tools (GitHub Copilot, Claude, Cursor, etc.) to generate correct, idiomatic RustLab code.

---

## 🎯 Core Principles

1. **Math-First Design**: Operations use natural mathematical syntax
2. **Zero-Cost Abstractions**: High-level operations compile to optimal code
3. **Type Safety**: Dimensions checked at compile-time where possible
4. **Predictable Naming**: Consistent patterns across all modules

---

## 📐 Essential Conventions

### Matrix Multiplication vs Element-wise

**CRITICAL**: RustLab uses different operators for different operations:

```rust
// Matrix multiplication (like NumPy @)
let C = A ^ B;        // Matrix × Matrix → Matrix
let v = A ^ x;        // Matrix × Vector → Vector
let s = x ^ y;        // Vector × Vector → Scalar (dot product)

// Element-wise operations (like NumPy *)
let C = &A * &B;      // Element-by-element multiplication
let v = &x * &y;      // Element-by-element multiplication
let v = &x + &y;      // Element-wise addition
let v = &x - &y;      // Element-wise subtraction
```

**Remember**: 
- `^` = Matrix/dot product (mathematical multiplication)
- `*` = Element-wise multiplication
- Always use references (`&`) for element-wise operations to avoid ownership issues

### Type Aliases

RustLab provides convenient type aliases for common numeric types:

```rust
// Vectors
VectorF64  // Vector<f64> - Most common
VectorF32  // Vector<f32> - Single precision
VectorC64  // Vector<Complex<f64>> - Complex numbers
VectorC32  // Vector<Complex<f32>>

// Arrays (2D matrices)
ArrayF64   // Array<f64> - Most common
ArrayF32   // Array<f32>
ArrayC64   // Array<Complex<f64>>
ArrayC32   // Array<Complex<f32>>
```

### Creation Patterns

```rust
// Vectors
let v = VectorF64::zeros(100);           // Zero vector
let v = VectorF64::ones(50);             // Ones vector
let v = VectorF64::fill(20, 3.14);       // Constant fill
let v = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
let v = vec64![1.0, 2.0, 3.0];           // Macro for convenience

// Arrays
let A = ArrayF64::zeros(3, 4);           // 3×4 zero matrix
let A = ArrayF64::ones(5, 5);            // 5×5 ones matrix
let A = ArrayF64::eye(4);                // 4×4 identity matrix
let A = array64![[1.0, 2.0], [3.0, 4.0]]; // Macro for 2D arrays
```

---

## 🔢 Dimension Rules

### Vector Operations

| Operation | Dimensions | Result | Example |
|-----------|-----------|---------|---------|
| `v1 + v2` | (n) + (n) | (n) | Element-wise addition |
| `v1 - v2` | (n) - (n) | (n) | Element-wise subtraction |
| `v1 * v2` | (n) × (n) | (n) | Element-wise multiplication |
| `v1 ^ v2` | (n) × (n) | scalar | Dot product |
| `v * scalar` | (n) × 1 | (n) | Scalar multiplication |

### Matrix Operations

| Operation | Dimensions | Result | Example |
|-----------|-----------|---------|---------|
| `A + B` | (m×n) + (m×n) | (m×n) | Element-wise addition |
| `A * B` | (m×n) × (m×n) | (m×n) | Element-wise multiplication |
| `A ^ B` | (m×n) × (n×p) | (m×p) | Matrix multiplication |
| `A ^ v` | (m×n) × (n) | (m) | Matrix-vector multiplication |
| `v ^ A` | (m) × (m×n) | (n) | Vector-matrix multiplication |

**Key Rule**: For `A ^ B`, inner dimensions must match: (m×**n**) × (**n**×p)

---

## 🎲 Element Access

### Safe vs Direct Access

```rust
// Safe access (returns Option)
let val = vector.get(index);           // Some(value) or None
let val = array.get(row, col);         // Some(value) or None

// Direct access (panics if out of bounds)
let val = vector[index];               // Panics if index >= length
let val = array[(row, col)];           // Panics if out of bounds

// Safe patterns
let val = vector.get(i).unwrap_or(0.0);  // Default value
if let Some(val) = vector.get(i) {
    // Process value
}
```

### Indexing Rules

- **Zero-based**: First element is index 0
- **Vectors**: Use single index `vector[i]`
- **Arrays**: Use tuple `array[(row, col)]` not `array[row][col]`
- **Prefer `.get()`** when index might be invalid

---

## ⚠️ Common Pitfalls & Solutions

### Pitfall 1: Wrong Multiplication Operator

```rust
// WRONG - This is element-wise!
let result = matrix1 * matrix2;  

// CORRECT - Matrix multiplication
let result = matrix1 ^ matrix2;
```

### Pitfall 2: Dimension Mismatch

```rust
// WRONG - Dimensions don't match
let A = ArrayF64::zeros(3, 4);  // 3×4
let B = ArrayF64::zeros(2, 3);  // 2×3
let C = A ^ B;  // ERROR: 4 ≠ 2

// CORRECT - Inner dimensions match
let A = ArrayF64::zeros(3, 4);  // 3×4
let B = ArrayF64::zeros(4, 2);  // 4×2
let C = A ^ B;  // OK: 3×2 result
```

### Pitfall 3: Ownership Issues

```rust
// WRONG - Consumes vectors
let sum = v1 + v2;  // v1 and v2 are moved

// CORRECT - Use references
let sum = &v1 + &v2;  // v1 and v2 still usable
```

### Pitfall 4: Complex Number Conjugation

```rust
// For complex vectors, dot product includes conjugation
let z1 = VectorC64::from_slice(&[Complex::new(1.0, 2.0)]);
let z2 = VectorC64::from_slice(&[Complex::new(3.0, 4.0)]);
let dot = z1 ^ z2;  // Conjugates z2 automatically
```

---

## 📊 Statistical Operations

```rust
// Vector statistics
let mean = vector.mean();
let std = vector.std();
let var = vector.variance();
let sum = vector.sum_elements();
let min = vector.min().unwrap();
let max = vector.max().unwrap();

// Array statistics
let mean = array.mean();
let sum = array.sum_elements();
let trace = array.trace();  // Sum of diagonal
```

---

## 🔄 Broadcasting

RustLab supports NumPy-style broadcasting:

```rust
// Scalar broadcasting
let result = &array + 5.0;        // Add 5 to all elements
let result = &vector * 2.0;       // Multiply all by 2

// Vector-Matrix broadcasting
let matrix = array64![[1.0, 2.0], [3.0, 4.0]];
let row_vec = vec64![10.0, 20.0];
let result = &matrix + &row_vec;  // Broadcasts to each row
```

---

## 🔗 Concatenation

```rust
// Vector concatenation
let combined = v1.append(&v2).unwrap();
let combined = vconcat![v1, v2, v3].unwrap();

// Array concatenation
let stacked_h = hstack![A, B].unwrap();  // Horizontal
let stacked_v = vstack![A, B].unwrap();  // Vertical
```

---

## 🎨 Linear Algebra

```rust
// Solving linear systems: Ax = b
let x = A.solve(&b)?;

// Matrix decompositions
let (L, U) = A.lu()?;
let (Q, R) = A.qr()?;
let chol = A.cholesky()?;  // A must be positive definite

// Eigenvalues (symmetric matrices)
let eigenvalues = A.eigenvalues()?;

// Matrix properties
let det = A.determinant();
let inv = A.inverse()?;  // Prefer solve() over inverse
let rank = A.rank();
```

---

## 🚀 Performance Tips for AI

1. **Use views for slicing** - Zero-copy operations
   ```rust
   let view = vector.slice(10..20);  // No allocation
   ```

2. **Chain operations** - Compiler optimizes into single loop
   ```rust
   let result = (&v1 + &v2) * 3.0 - &v3;  // Single pass
   ```

3. **Preallocate with zeros** - Then fill
   ```rust
   let mut result = VectorF64::zeros(1000);
   // Fill result...
   ```

4. **Use type aliases** - Clearer and helps type inference
   ```rust
   let v: VectorF64 = zeros_vec(100);  // Clear type
   ```

---

## 🔴 Error Handling

### Operations That Panic
- Index out of bounds: `vector[bad_index]`
- Dimension mismatch in operators: `incompatible_matrix1 ^ incompatible_matrix2`
- Invalid matrix operations: `singular_matrix.inverse()`

### Operations That Return Result
- File I/O: `read_csv("data.csv")?`
- Decompositions: `matrix.cholesky()?`
- Solving systems: `A.solve(&b)?`

### Safe Patterns
```rust
// Check dimensions before operations
if A.ncols() == B.nrows() {
    let C = A ^ B;
}

// Use ? operator for Result types
let solution = A.solve(&b)?;

// Provide defaults for Option types
let value = vector.get(index).unwrap_or(0.0);
```

---

## 📝 Import Patterns

```rust
// Basic imports
use rustlab::{VectorF64, ArrayF64};
use rustlab::{vec64, array64};

// Math functions
use rustlab::math::*;

// Linear algebra
use rustlab::linalg::*;

// Statistics
use rustlab::stats::*;

// Everything (for scripts/notebooks)
use rustlab::prelude::*;
```

---

## 🎓 Quick Conversion from NumPy

| NumPy | RustLab |
|-------|---------|
| `np.zeros(100)` | `VectorF64::zeros(100)` |
| `np.ones((3, 4))` | `ArrayF64::ones(3, 4)` |
| `np.eye(5)` | `ArrayF64::eye(5)` |
| `A @ B` | `A ^ B` |
| `A * B` | `&A * &B` |
| `np.dot(a, b)` | `a ^ b` or `a.dot(&b)` |
| `np.linalg.solve(A, b)` | `A.solve(&b)?` |
| `np.linalg.inv(A)` | `A.inverse()?` |
| `np.mean(x)` | `x.mean()` |
| `np.std(x)` | `x.std()` |

---

## 🔥 Most Important Rules for AI

1. **Use `^` for matrix multiplication, not `*`**
2. **Use references `&` for element-wise operations**
3. **Check dimensions match for matrix operations**
4. **Use `.get()` for safe access, `[]` when bounds are known**
5. **Handle `Result` with `?` operator**
6. **Zero-based indexing everywhere**
7. **Arrays use `(row, col)` not `[row][col]`**
8. **Prefer `solve()` over `inverse()`**
9. **Use type aliases (VectorF64, ArrayF64)**
10. **Import macros for convenience (vec64!, array64!)**

---

## 💡 Example: Complete Linear Regression

```rust
use rustlab::prelude::*;

fn linear_regression(X: &ArrayF64, y: &VectorF64) -> Result<VectorF64> {
    // Add intercept column
    let n = X.nrows();
    let ones = VectorF64::ones(n);
    let X_with_intercept = hconcat![ones.to_column(), X.clone()]?;
    
    // Normal equations: β = (X'X)^(-1)X'y
    let Xt = X_with_intercept.transpose();
    let XtX = &Xt ^ &X_with_intercept;
    let Xty = &Xt ^ y;
    
    // Solve for coefficients
    let beta = XtX.solve(&Xty)?;
    
    Ok(beta)
}
```

---

*This guide is optimized for AI code generation. When in doubt, refer to the full documentation for detailed specifications.*