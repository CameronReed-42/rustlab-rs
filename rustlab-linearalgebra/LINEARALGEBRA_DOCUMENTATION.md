# AI Documentation - RustLab Linear Algebra

## Quick Reference for AI Code Generation

RustLab Linear Algebra provides high-performance matrix decompositions, eigenvalue computation, and linear system solvers built on the faer backend.

---

## 🎯 Core Operations

### Basic Matrix Operations

```rust
use rustlab_linearalgebra::{BasicLinearAlgebra, ArrayF64};

// Matrix transpose: A.T()
let matrix = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2)?;
let transposed = matrix.T();

// Determinant: A.det()
let det = matrix.det()?;

// Matrix inverse: A.inv()
let inverse = matrix.inv()?;
```

### Linear System Solving

```rust
use rustlab_linearalgebra::{LinearSystemOps, ArrayF64, VectorF64};

// Solve Ax = b
let A = array64![[2.0, 1.0], [1.0, 1.0]];
let b = vec64![5.0, 3.0];
let x = A.solve_system(&b)?;

// Multiple systems: AX = B
let B = array64![[5.0, 7.0], [3.0, 4.0]];
let X = A.solve_systems(&B)?;

// Triangular systems
let x_lower = L.solve_lower_triangular(&b)?;
let x_upper = U.solve_upper_triangular(&b)?;
```

### Matrix Decompositions

```rust
use rustlab_linearalgebra::{DecompositionMethods, ArrayF64};

// LU decomposition with partial pivoting
let lu = matrix.lu()?;
let solution = lu.solve(&b)?;

// QR decomposition
let qr = matrix.qr()?;
let solution = qr.solve(&b)?;

// Cholesky decomposition (positive definite matrices)
let chol = matrix.cholesky()?;
let solution = chol.solve(&b)?;

// Singular Value Decomposition
let svd = matrix.svd()?;
let singular_values = svd.singular_values();
```

### Eigenvalue Computation

```rust
use rustlab_linearalgebra::{EigenvalueOps, eigenvalues, eigenvectors};

// Eigenvalues only
let eigenvals = matrix.eigenvalues()?;

// Eigenvalues and eigenvectors
let eig_decomp = matrix.eigenvectors()?;
let eigenvals = eig_decomp.eigenvalues;
let eigenvecs = eig_decomp.eigenvectors;
```

---

## 📐 Mathematical Specifications

### Decomposition Types

| Decomposition | Mathematical Form | Use Cases | Requirements |
|---------------|-------------------|-----------|--------------|
| **LU** | A = PLU | General linear systems | Square matrix |
| **QR** | A = QR | Least squares, rank | Any m×n matrix |
| **Cholesky** | A = L·L^T | Symmetric positive definite | SPD matrix |
| **SVD** | A = U·Σ·V^T | Rank, pseudoinverse, PCA | Any m×n matrix |

### Linear System Types

| Method | Mathematical Problem | Algorithm | Complexity |
|--------|---------------------|-----------|------------|
| `solve_system(b)` | Ax = b | LU with partial pivoting | O(n³) |
| `solve_lower_triangular(b)` | Lx = b | Forward substitution | O(n²) |
| `solve_upper_triangular(b)` | Ux = b | Back substitution | O(n²) |
| `solve_systems(B)` | AX = B | Batch LU solve | O(n³ + n²k) |

### Eigenvalue Problems

| Function | Mathematical Problem | Algorithm | Output Type |
|----------|---------------------|-----------|-------------|
| `eigenvalues()` | Av = λv | QR algorithm | Vec<Complex<f64>> |
| `eigenvectors()` | Av = λv | Schur decomposition | EigenDecomposition |

---

## ⚙️ API Patterns

### Extension Traits

RustLab Linear Algebra uses extension traits to add methods to ArrayF64:

```rust
// Basic operations trait
impl BasicLinearAlgebra<f64> for ArrayF64 {
    fn T(&self) -> ArrayF64;        // Transpose
    fn det(&self) -> Result<f64>;   // Determinant
    fn inv(&self) -> Result<ArrayF64>; // Inverse
}

// Linear system operations trait
impl LinearSystemOps<f64> for ArrayF64 {
    fn solve_system(&self, b: &VectorF64) -> Result<VectorF64>;
    fn solve_systems(&self, b: &ArrayF64) -> Result<ArrayF64>;
    fn solve_lower_triangular(&self, b: &VectorF64) -> Result<VectorF64>;
    fn solve_upper_triangular(&self, b: &VectorF64) -> Result<VectorF64>;
}

// Decomposition methods trait
impl DecompositionMethods for ArrayF64 {
    fn lu(&self) -> Result<LuDecomposition>;
    fn qr(&self) -> Result<QrDecomposition>;
    fn cholesky(&self) -> Result<CholeskyDecomposition>;
    fn svd(&self) -> Result<SvdDecomposition>;
}

// Eigenvalue operations trait
impl EigenvalueOps for ArrayF64 {
    fn eigenvalues(&self) -> Result<Vec<Complex<f64>>>;
    fn eigenvectors(&self) -> Result<EigenDecomposition>;
}
```

### Decomposition Result Access

```rust
// LU decomposition
let lu = A.lu()?;
let x = lu.solve(&b)?;  // Solve using decomposition

// QR decomposition
let qr = A.qr()?;
let x = qr.solve(&b)?;  // Least squares solution

// Cholesky decomposition
let chol = A.cholesky()?;
let x = chol.solve(&b)?;  // Efficient for SPD matrices

// SVD decomposition
let svd = A.svd()?;
let u = svd.u();                    // Left singular vectors
let s = svd.singular_values();      // Singular values
let vt = svd.vt();                  // Right singular vectors (transposed)
```

---

## 🎲 Return Types and Access Patterns

### Basic Operations

```rust
// Scalar results
let det: f64 = matrix.det()?;

// Matrix results
let transposed: ArrayF64 = matrix.T();
let inverse: ArrayF64 = matrix.inv()?;

// Vector results
let solution: VectorF64 = A.solve_system(&b)?;
```

### Decomposition Results

```rust
// Decomposition structs (opaque handles)
let lu: LuDecomposition = A.lu()?;
let qr: QrDecomposition = A.qr()?;
let chol: CholeskyDecomposition = A.cholesky()?;
let svd: SvdDecomposition = A.svd()?;

// Use decompositions to solve systems
let x = lu.solve(&b)?;
let x = qr.solve(&b)?;
let x = chol.solve(&b)?;

// SVD provides component access
let singular_vals: &[f64] = svd.singular_values();
let u_matrix: Mat<f64> = svd.u();
let vt_matrix: Mat<f64> = svd.vt();
```

### Eigenvalue Results

```rust
// Eigenvalues only
let eigenvals: Vec<Complex<f64>> = matrix.eigenvalues()?;

// Full eigendecomposition
let eig: EigenDecomposition = matrix.eigenvectors()?;
let eigenvals: Vec<Complex<f64>> = eig.eigenvalues;
let eigenvecs: Mat<Complex<f64>> = eig.eigenvectors;

// Access individual eigenvalues
for (i, eigenval) in eigenvals.iter().enumerate() {
    println!("λ_{} = {}", i, eigenval);
}
```

---

## ⚠️ Common Patterns and Pitfalls

### ✅ Correct Patterns

```rust
// Always check matrix is square for operations requiring it
let det = if matrix.nrows() == matrix.ncols() {
    matrix.det()?
} else {
    return Err("Matrix must be square".into());
};

// Use decompositions for multiple solves with same A
let lu = A.lu()?;  // Factor once
let x1 = lu.solve(&b1)?;  // Solve multiple times
let x2 = lu.solve(&b2)?;
let x3 = lu.solve(&b3)?;

// Check for positive definiteness before Cholesky
let chol = match A.cholesky() {
    Ok(decomp) => decomp,
    Err(_) => {
        // Fall back to LU if not positive definite
        return A.lu()?.solve(&b);
    }
};

// Handle complex eigenvalues properly
let eigenvals = matrix.eigenvalues()?;
for eigenval in eigenvals {
    if eigenval.im.abs() < 1e-10 {
        println!("Real eigenvalue: {}", eigenval.re);
    } else {
        println!("Complex eigenvalue: {} + {}i", eigenval.re, eigenval.im);
    }
}
```

### ❌ Common Mistakes

```rust
// Don't ignore dimension requirements
let det = non_square_matrix.det()?;  // ERROR: Will fail

// Don't recompute decompositions unnecessarily
for _ in 0..1000 {
    let x = A.lu()?.solve(&b)?;  // INEFFICIENT: Factor each time
}

// Don't assume eigenvalues are real
let eigenvals = matrix.eigenvalues()?;
let real_parts: Vec<f64> = eigenvals.iter()
    .map(|z| z.re)  // WRONG: Ignores imaginary parts
    .collect();

// Don't use wrong decomposition for matrix type
let chol = arbitrary_matrix.cholesky()?;  // ERROR: May not be SPD
```

---

## 🔧 Error Handling

### Common Errors and Fixes

| Error Type | Common Cause | Fix |
|------------|--------------|-----|
| `NotSquare` | Operation requires square matrix | Check dimensions before calling |
| `DimensionMismatch` | b.len() ≠ A.nrows() | Ensure compatible dimensions |
| `Singular` | Matrix is not invertible | Check conditioning, use pseudoinverse |
| `NotPositiveDefinite` | Cholesky on non-SPD matrix | Use LU decomposition instead |
| `DecompositionFailed` | Numerical issues | Check matrix conditioning |

### Error Handling Patterns

```rust
// Handle specific linear algebra errors
match result {
    Ok(solution) => println!("Solution: {:?}", solution),
    Err(LinearAlgebraError::NotSquare { rows, cols }) => {
        println!("Matrix must be square: got {}×{}", rows, cols);
    },
    Err(LinearAlgebraError::Singular) => {
        println!("Matrix is singular, trying regularization...");
        // Add small diagonal regularization
    },
    Err(LinearAlgebraError::DimensionMismatch { expected, actual }) => {
        println!("Dimension error: expected {}, got {}", expected, actual);
    },
    Err(e) => println!("Other error: {}", e),
}

// Graceful fallback for decompositions
fn robust_solve(A: &ArrayF64, b: &VectorF64) -> Result<VectorF64> {
    // Try Cholesky first (fastest for SPD)
    if let Ok(chol) = A.cholesky() {
        return chol.solve(&array64![[b[0]], [b[1]]]).map(|x| vec64![x[(0,0)], x[(1,0)]]);
    }
    
    // Fall back to LU
    if let Ok(lu) = A.lu() {
        return lu.solve(&array64![[b[0]], [b[1]]]).map(|x| vec64![x[(0,0)], x[(1,0)]]);
    }
    
    // Last resort: use general solver
    A.solve_system(b)
}
```

---

## 🚀 Performance Guidelines

### Algorithm Choice by Problem Type

```rust
// Small dense systems (n < 1000): Direct methods
let x = A.lu()?.solve(&b)?;  // O(n³) but very fast constant

// Large sparse systems: Use specialized sparse solvers
// (Not yet available in this crate - use faer directly)

// Multiple solves with same A: Factor once
let lu = A.lu()?;  // O(n³) factorization
for b in many_right_hand_sides {
    let x = lu.solve(&b)?;  // O(n²) per solve
}

// Symmetric positive definite: Cholesky is 2x faster
let chol = A.cholesky()?;  // O(n³/3) vs O(n³) for LU
let x = chol.solve(&b)?;   // More stable too

// Overdetermined least squares: QR decomposition
let qr = A.qr()?;          // Handles m×n with m > n
let x = qr.solve(&b)?;     // Minimum norm solution
```

### Memory Considerations

- **LU decomposition**: O(n²) storage for factors
- **QR decomposition**: O(mn) storage for Q, R factors  
- **Cholesky**: O(n²) storage, but can overwrite input
- **SVD**: O(mn + n²) storage for U, Σ, V^T
- **Eigenvalues**: O(n²) working space

---

## 📝 Import Patterns

```rust
// Basic linear algebra operations
use rustlab_linearalgebra::{BasicLinearAlgebra, ArrayF64, VectorF64};

// Linear system solving
use rustlab_linearalgebra::{LinearSystemOps, solve, solve_triangular};

// Matrix decompositions
use rustlab_linearalgebra::{
    DecompositionMethods, LuDecomposition, QrDecomposition, 
    CholeskyDecomposition, SvdDecomposition
};

// Eigenvalue computation
use rustlab_linearalgebra::{EigenvalueOps, EigenDecomposition, eigenvalues, eigenvectors};

// Error handling
use rustlab_linearalgebra::{LinearAlgebraError, Result};

// Everything (for scripts/notebooks)
use rustlab_linearalgebra::*;

// With RustLab math types
use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
use rustlab_linearalgebra::*;
```

---

## 🎓 Quick Conversion from NumPy/SciPy

| NumPy/SciPy | RustLab Linear Algebra |
|-------------|------------------------|
| `A.T` | `A.T()` |
| `np.linalg.det(A)` | `A.det()?` |
| `np.linalg.inv(A)` | `A.inv()?` |
| `np.linalg.solve(A, b)` | `A.solve_system(&b)?` |
| `scipy.linalg.lu(A)` | `A.lu()?` |
| `scipy.linalg.qr(A)` | `A.qr()?` |
| `scipy.linalg.cholesky(A)` | `A.cholesky()?` |
| `np.linalg.svd(A)` | `A.svd()?` |
| `np.linalg.eig(A)` | `A.eigenvectors()?` |
| `np.linalg.eigvals(A)` | `A.eigenvalues()?` |

---

## 🔥 Most Important Rules for AI

1. **Check matrix dimensions** before operations (square for det/inv/eigen)
2. **Use `?` operator** for error handling - all operations return Result
3. **Factor once, solve many** - decompositions are reusable
4. **Choose right decomposition**: Cholesky (SPD) > LU (general) > QR (overdetermined)
5. **Handle complex eigenvalues** - use Complex<f64> type properly
6. **Extension traits** - methods added to ArrayF64 via traits
7. **Decomposition results** are opaque structs with solve() methods
8. **Dimension compatibility** - A.nrows() must equal b.len() for Ax=b
9. **Error specificity** - match on LinearAlgebraError variants for good UX
10. **Performance order**: Cholesky < LU < QR < SVD (increasing cost)

---

*This documentation is optimized for AI code generation. All examples use the latest faer 0.22 backend for maximum performance.*