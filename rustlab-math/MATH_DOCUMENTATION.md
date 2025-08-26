# RustLab Math - Comprehensive Documentation

A high-performance numerical computing library designed for intuitive mathematical syntax with comprehensive AI code generation support. Built on faer for optimal performance and scientific computing workflows.

## 🎯 CRITICAL for AI Code Generation

**Operator Distinction (Most Important Rule)**:
- **`^` operator**: Mathematical multiplication (matrix/dot product)  
- **`*` operator**: Element-wise multiplication (Hadamard product)

This prevents the most common AI hallucination in numerical computing.

---

## 📚 Table of Contents

1. [Quick Start for AI](#quick-start-for-ai)
2. [MAT*/NumPy-Style Macros Quick Reference](#matnumpy-style-macros-quick-reference)
3. [Core Data Structures](#core-data-structures)
4. [Mathematical Operations](#mathematical-operations)
5. [Advanced Slicing System](#advanced-slicing-system)
6. [Data Creation and Manipulation](#data-creation-and-manipulation)
   - [Array-Vector Interoperability](#array-vector-interoperability)
   - [List Comprehension with Automatic Parallelism](#list-comprehension-with-automatic-parallelism)
7. [Broadcasting and Element-wise Operations](#broadcasting-and-element-wise-operations)
8. [Statistical Operations](#statistical-operations)
9. [File I/O Operations](#file-io-operations)
10. [Functional Programming](#functional-programming)
11. [Performance Features](#performance-features)
12. [Migration from NumPy/MAT*](#migration-from-numpymat*)
13. [Common Patterns and Best Practices](#common-patterns-and-best-practices)

---

## 🚀 Quick Start for AI

```rust
use rustlab_math::{ArrayF64, VectorF64, array64, vec64};

// Create data structures
let A = array64![[1.0, 2.0], [3.0, 4.0]];  // 2×2 matrix
let v = vec64![1.0, 2.0];                   // 2D vector

// Mathematical operations (use ^ for matrix math)
let result = &A ^ &v;          // Matrix-vector multiplication
let dot_prod = &v ^ &v;        // Vector dot product (scalar)
let mat_mult = &A ^ &A;        // Matrix multiplication

// Element-wise operations (use * for element-wise)
let elem_mult = &A * &A;       // Element-wise (Hadamard) product
let scaled = &A * 2.0;         // Scalar multiplication
let sum = &A + &A;             // Element-wise addition

// Natural slicing (zero-copy views)
let slice = &v[1..2];          // Slice reference
let owned = v.slice_at(1..2)?; // Owned slice

// Advanced slicing with numpy-style indexing
let indices = vec![0, 1]; 
let selected = v.slice_at(indices)?;    // Fancy indexing

// Array-vector interoperability
let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let reshaped = v.to_array(2, 3)?;      // Reshape vector to 2×3 matrix
let col_vec = A.col(1)?;               // Extract column as vector
let row_vec = A.row(0)?;               // Extract row as vector

// List comprehension with automatic parallelism
let squared = vectorize![x * x, for x in &v];              // Auto-parallel
let serial_only = vectorize![serial: x * x, for x in &v]; // Zero-overhead
let results = vectorize![complex: simulation(x), for x in &data]; // Force parallel
```

---

## 🎨 MAT*/NumPy-Style Macros Quick Reference

For users coming from MAT* or NumPy, use these macros for familiar syntax:

```rust
// Matrix creation
let A = array64![[1.0, 2.0], [3.0, 4.0]];       // Create 2×2 matrix
let Z = carray64![[(1.0, 2.0), (3.0, -1.0)]];   // Complex matrix from tuples

// Concatenation (math-first syntax)
let horizontal = hcat![A, B].unwrap();           // [A, B] in MAT*
let vertical = vcat![A, B].unwrap();             // [A; B] in MAT*
let block_mat = block![[A, B], [C, D]].unwrap(); // [A, B; C, D] in MAT*

// Statistical operations (math-first syntax)
let col_means = mean![A, axis=0].unwrap();       // Column means
let row_sums = sum![A, axis=1].unwrap();         // Row sums  
let col_maxs = max![A, axis=0].unwrap();         // Column maximums
let row_mins = min![A, axis=1].unwrap();         // Row minimums
let col_stds = std![A, axis=0].unwrap();         // Column std dev
let row_vars = var![A, axis=1].unwrap();         // Row variance

// Special matrices
let zeros = matrix!(zeros: 3, 4);                // 3×4 zero matrix
let identity = matrix!(eye: 5);                  // 5×5 identity
let diagonal = matrix!(diag: [1, 2, 3]);         // Diagonal matrix

// List comprehension with automatic parallelism
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let squared = vectorize![x * x, for x in &data]; // NumPy/Julia-style vectorization
let (X, Y) = meshgrid!(x: x_coords, y: y_coords); // Coordinate grid generation
```

**Key Equivalences:**
- MAT* `[A, B]` → RustLab `hcat![A, B]`
- MAT* `[A; B]` → RustLab `vcat![A, B]`
- NumPy `np.hstack([A, B])` → RustLab `hcat![A, B]`
- NumPy `np.vstack([A, B])` → RustLab `vcat![A, B]`
- MAT* `mean(A, 1)` → RustLab `mean![A, axis=0]`
- MAT* `sum(A, 2)` → RustLab `sum![A, axis=1]`
- NumPy `np.mean(A, axis=0)` → RustLab `mean![A, axis=0]`
- NumPy `np.std(A, axis=1)` → RustLab `std![A, axis=1]`
- NumPy `A[:, 1]` → RustLab `A.col(1)?`
- NumPy `A[0, :]` → RustLab `A.row(0)?`
- NumPy `v.reshape(2, 3)` → RustLab `v.to_array(2, 3)?`
- NumPy `[x**2 for x in data]` → RustLab `vectorize![x * x, for x in &data]`
- NumPy `np.meshgrid(x, y)` → RustLab `meshgrid!(x: x, y: y)`

---

## 🧬 Type System Overview

### Generic Foundation

RustLab is built on two generic base types that work with any numeric type:

```rust
Array<T>   // 2D matrices: rows × columns  
Vector<T>  // 1D vectors: single dimension
```

### Type Hierarchy

```
Generic Types                 Common Aliases              Usage
─────────────────            ─────────────────            ─────────────
Array<f64>          →        ArrayF64                    Double precision (most common)
Array<f32>          →        ArrayF32                    Single precision (memory/speed)
Array<Complex<f64>> →        ArrayC64                    Complex double precision  
Array<Complex<f32>> →        ArrayC32                    Complex single precision

Vector<f64>         →        VectorF64                   Double precision (most common)
Vector<f32>         →        VectorF32                   Single precision (memory/speed)
Vector<Complex<f64>>→        VectorC64                   Complex double precision
Vector<Complex<f32>>→        VectorC32                   Complex single precision
```

### **CRITICAL for AI**: Type Selection Rules

| Type | When to Use | Memory/Element | Performance |
|------|-------------|----------------|-------------|
| `ArrayF64` / `VectorF64` | **Default choice** for scientific computing | 8 bytes | Optimal precision |
| `ArrayF32` / `VectorF32` | Large datasets, GPU compatibility | 4 bytes | 2× memory savings |
| `ArrayC64` / `VectorC64` | Complex analysis, signal processing | 16 bytes | Complex arithmetic |
| `ArrayC32` / `VectorC32` | Memory-constrained complex operations | 8 bytes | Fastest complex |

### Type Conversion Patterns

```rust
use rustlab_math::{ArrayF64, ArrayF32, ArrayC64};

// Precision conversion (explicit)
let f64_matrix = ArrayF64::ones(100, 50);
let f32_matrix = f64_matrix.cast_to_f32();      // Precision loss possible

// Real to complex conversion  
let real_matrix = ArrayF64::ones(50, 50);
let complex_matrix = real_matrix.to_complex();   // Imaginary part = 0

// Complex to real (extract components)
let complex_data = ArrayC64::ones(100, 100);
let real_part = complex_data.real();             // Extract real components
let imag_part = complex_data.imag();             // Extract imaginary components
```

### Interoperability Rules

**✅ Compatible Operations:**
```rust
let a = ArrayF64::ones(3, 3);
let b = ArrayF64::zeros(3, 3);
let result = &a + &b;                    // Same type: works

let v1 = VectorF64::ones(100);
let v2 = VectorF64::zeros(100);  
let dot = &v1 ^ &v2;                     // Same type: works
```

**❌ Incompatible Operations (Compile-time errors):**
```rust
let a_f64 = ArrayF64::ones(3, 3);        // f64 type
let a_f32 = ArrayF32::ones(3, 3);        // f32 type
// let result = &a_f64 + &a_f32;         // ERROR: Type mismatch

let real_vec = VectorF64::ones(100);      // Real type
let complex_vec = VectorC64::ones(100);   // Complex type
// let dot = &real_vec ^ &complex_vec;    // ERROR: Type mismatch
```

### Memory Layout Implications

| Type | Alignment | Cache Performance | SIMD Support |
|------|-----------|-------------------|--------------|
| `ArrayF64` | 64-byte | Optimal for AVX-512 | Full support |
| `ArrayF32` | 64-byte | Better cache density | Full support |
| `ArrayC64` | 64-byte | Complex SIMD operations | Limited |
| `ArrayC32` | 64-byte | Highest complex density | Limited |

### Creation Convenience Macros

```rust
// Real types (most common)
let A = array64![[1.0, 2.0], [3.0, 4.0]];       // Creates ArrayF64
let v = vec64![1.0, 2.0, 3.0];                   // Creates VectorF64

// Single precision
let B = array32![[1.0f32, 2.0f32]];              // Creates ArrayF32
let u = vec32![1.0f32, 2.0f32];                  // Creates VectorF32

// Complex types
let C = carray64![[(1.0, 2.0), (3.0, 4.0)]];     // Creates ArrayC64 
let z = cvec64![(1.0, 2.0), (3.0, 4.0)];         // Creates VectorC64
```

### **AI Guidelines**: Type Usage Patterns

**✅ ALWAYS Use These Patterns:**
```rust
// Default to f64 for scientific computing
use rustlab_math::{ArrayF64, VectorF64, array64, vec64};

// Explicit type when needed
let data: ArrayF64 = ArrayF64::zeros(1000, 500);
let weights: VectorF64 = VectorF64::ones(500);

// Type conversion when necessary
let high_precision = low_precision_data.cast_to_f64();
```

**❌ NEVER Use These Patterns:**
```rust
// Don't mix types without conversion
let result = &array_f64 + &array_f32;           // Compile error

// Don't assume automatic conversions
let complex_result = &real_matrix ^ &complex_vector;  // Error

// Don't use generic types in user code
use rustlab_math::Array;  // Too generic - use ArrayF64 instead
```

---

## 🏗️ Core Data Structures

### Array<T> - Generic 2D Matrices

The foundational 2D matrix type with comprehensive mathematical operations.

**Key Features:**
- Generic over numeric types: `f64`, `f32`, `Complex<f64>`, `Complex<f32>`
- Cache-aligned memory layout (64-byte alignment)
- Direct faer integration for optimal performance
- Zero-cost abstractions with compile-time optimizations
- **NEW**: Row and column extraction methods for efficient data access
- **NEW**: Vector-to-matrix conversion for linear algebra operations

**Type Aliases:**
```rust
ArrayF64  // Array<f64> - Most common, double precision
ArrayF32  // Array<f32> - Single precision 
ArrayC64  // Array<Complex<f64>> - Complex double precision
ArrayC32  // Array<Complex<f32>> - Complex single precision
```

**Creation Examples:**
```rust
use rustlab_math::{ArrayF64, ArrayC64, array64, carray64, cmatrix};

// Real matrices
let A = ArrayF64::zeros(3, 4);           // 3×4 zero matrix
let B = ArrayF64::ones(2, 2);            // 2×2 ones matrix
let I = ArrayF64::eye(5);                // 5×5 identity matrix
let M = array64![[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]];       // From nested arrays

// Complex matrices (math-first syntax)
let Z1 = carray64![[(1.0, 2.0), (3.0, -1.0)],
                   [(0.0, 1.0), (2.0, 0.0)]];   // Complex from tuples
let Z2 = carray64![[1.0, 2.0], [3.0, 4.0]];    // Real-only complex matrix

// Special complex matrices
let Z_zeros = cmatrix!(zeros: 3, 3);            // 3×3 complex zero matrix
let Z_eye = cmatrix!(eye: 4);                   // 4×4 complex identity
let Z_diag = cmatrix!(cdiag: [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]); // Complex diagonal

// From slice data
let data = vec![1.0, 2.0, 3.0, 4.0];
let A = ArrayF64::from_slice(&data, 2, 2)?; // 2×2 matrix

// NEW: From vector (column matrix)
let v = vec64![1.0, 2.0, 3.0];
let col_matrix = ArrayF64::from_vector_column(&v); // 3×1 matrix
```

**Row and Column Extraction (NEW):**
```rust
use rustlab_math::{ArrayF64, array64};

let matrix = array64![[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]];

// Extract columns and rows as owned vectors
let col1 = matrix.col(1).unwrap();       // Column 1: [2.0, 5.0, 8.0]
let row0 = matrix.row(0).unwrap();       // Row 0: [1.0, 2.0, 3.0]

// Zero-copy views for efficiency
let col_view = matrix.col_view(1)?;      // View of column 1 (no allocation)
let row_view = matrix.row_view(0)?;      // View of row 0 (no allocation)

// Convert back to vector from column matrix
let y_matrix = ArrayF64::from_vector_column(&y);  // Vector → n×1 matrix
let y_vector = y_matrix.to_vector_column();       // n×1 matrix → Vector

// Use in linear algebra
let X = array64![[1.0, 2.0], [3.0, 4.0]];
let y = vec64![5.0, 6.0];
let y_mat = ArrayF64::from_vector_column(&y);
let result = &X ^ &y_mat;  // Matrix multiplication with column vector
```

### Vector<T> - Generic 1D Vectors

The foundational 1D vector type optimized for mathematical operations.

**Key Features:**
- Seamless integration with Array<T> for matrix-vector operations
- Direct faer column backend for memory efficiency
- SIMD optimization for large vectors (> 64 elements)
- Natural mathematical syntax with other vectors and scalars

**Type Aliases:**
```rust
VectorF64  // Vector<f64> - Most common
VectorF32  // Vector<f32> - Single precision
VectorC64  // Vector<Complex<f64>> - Complex double precision  
VectorC32  // Vector<Complex<f32>> - Complex single precision
```

**Creation Examples:**
```rust
use rustlab_math::{VectorF64, VectorC64, vec64, cvec64};

// Real vectors
let v = VectorF64::zeros(100);           // 100-element zero vector
let u = VectorF64::ones(50);             // 50-element ones vector
let w = vec64![1.0, 2.0, 3.0, 4.0, 5.0]; // From array literal

// Complex vectors (math-first syntax)
let z1 = cvec64![(1.0, 2.0), (3.0, -1.0), (0.0, 1.0)];  // Complex from tuples
let z2 = cvec64![1.0, 2.0, 3.0];                         // Real-only complex
let z3 = VectorC64::zeros(50);                           // 50-element complex zeros

// From slice
let data = vec![1.0, 2.0, 3.0];
let x = VectorF64::from_slice(&data);    // From Vec<f64>
```

---

## 🧮 Mathematical Operations

### The Critical ^ Operator (Matrix Multiplication)

The `^` operator provides natural mathematical syntax for true mathematical multiplication:

```rust
use rustlab_math::{ArrayF64, VectorF64, ArrayC64, VectorC64, array64, vec64, carray64, cvec64};

// Real matrices and vectors
let A = array64![[1.0, 2.0], [3.0, 4.0]];  // 2×2 matrix
let B = array64![[2.0, 1.0], [1.0, 2.0]];  // 2×2 matrix  
let v = vec64![1.0, 2.0];                   // 2D vector
let u = vec64![3.0, 4.0];                   // 2D vector

// Real matrix operations
let C = &A ^ &B;           // [[4.0, 5.0], [10.0, 11.0]]
let result = &A ^ &v;      // [5.0, 11.0]
let dot = &v ^ &u;         // 11.0
let row_result = &v ^ &A;  // [7.0, 10.0]

// Complex matrices and vectors (math-first syntax)
let Z = carray64![[(1.0, 1.0), (2.0, -1.0)],
                  [(0.0, 1.0), (1.0, 0.0)]];   // 2×2 complex matrix
let w = cvec64![(1.0, 0.0), (0.0, 1.0)];      // Complex vector

// Complex mathematical operations
let Z_mult = &Z ^ &Z;      // Complex matrix multiplication
let z_result = &Z ^ &w;    // Complex matrix-vector multiplication
let z_dot = &w ^ &w;       // Complex dot product (includes conjugation)

// Mixed real-complex operations
let mixed = &A ^ &w;       // Real matrix × complex vector → complex vector
```

### Dimension Rules for AI

| Operation | Input Dimensions | Output | Description |
|-----------|------------------|---------|-------------|
| `A ^ B` | (m×n) × (n×p) | (m×p) | Matrix multiplication |
| `A ^ v` | (m×n) × (n) | (m) | Matrix-vector multiplication |  
| `u ^ v` | (n) × (n) | scalar | Vector dot product |
| `v ^ A` | (n) × (n×p) | (p) | Vector-matrix multiplication |

**Key Rule**: For `A ^ B`, inner dimensions must match: (m×**n**) × (**n**×p)

### Element-wise Operations (* operator)

The `*` operator provides element-wise (Hadamard) operations:

```rust
// Real matrices
let A = array64![[1.0, 2.0], [3.0, 4.0]];
let B = array64![[2.0, 1.0], [1.0, 2.0]];

// Real element-wise operations
let hadamard = &A * &B;    // [[2.0, 2.0], [3.0, 8.0]]
let sum = &A + &B;         // [[3.0, 3.0], [4.0, 6.0]]
let diff = &A - &B;        // [[-1.0, 1.0], [2.0, 2.0]]
let scaled = &A * 2.0;     // [[2.0, 4.0], [6.0, 8.0]]
let shifted = &A + 1.0;    // [[2.0, 3.0], [4.0, 5.0]]

// Complex element-wise operations (math-first syntax)
let Z1 = carray64![[(1.0, 1.0), (2.0, 0.0)],
                   [(0.0, 1.0), (1.0, -1.0)]];
let Z2 = carray64![[(1.0, 0.0), (1.0, 1.0)],
                   [(2.0, 0.0), (0.0, 1.0)]];

// Complex element-wise operations
let z_hadamard = &Z1 * &Z2;    // Complex element-wise multiplication
let z_sum = &Z1 + &Z2;         // Complex addition
let z_diff = &Z1 - &Z2;        // Complex subtraction

// Complex scalar operations
let z_scaled = &Z1 * (2.0, 1.0);    // Scale by complex scalar (2 + i)
let z_real_scaled = &Z1 * 2.0;      // Scale by real scalar

// Mixed real-complex element-wise
let mixed_add = &A + &Z1;      // Real + complex → complex (broadcasting)
```

---

## ✂️ Advanced Slicing System

RustLab provides three complementary slicing approaches for maximum flexibility:

### 1. Natural Slicing (Index Trait)

Zero-copy slicing using Rust's Index trait for the most natural syntax:

```rust
use rustlab_math::{VectorF64, vec64};

let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];

// Range slicing (returns &[f64])
let slice = &v[1..4];      // [2.0, 3.0, 4.0] - zero copy
let tail = &v[2..];        // [3.0, 4.0, 5.0] - zero copy
let head = &v[..3];        // [1.0, 2.0, 3.0] - zero copy
let full = &v[..];         // [1.0, 2.0, 3.0, 4.0, 5.0] - zero copy

// Use with mathematical operations
let sum = (&v[1..4]).iter().sum::<f64>();  // Sum middle elements
```

### 2. Ergonomic Slicing (Method-based)

Comprehensive NumPy/MAT*-style slicing with owned results:

```rust
use rustlab_math::{VectorF64, BooleanVector, vec64};

let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];

// Range slicing (returns owned VectorF64)
let owned = v.slice_at(1..4)?;           // [2.0, 3.0, 4.0]

// Negative indexing (Python-style)
let last = v.slice_at(-1)?;              // [5.0] (last element)
let last_two = v.slice_at(-2..)?;        // [4.0, 5.0]

// Fancy indexing (arbitrary indices)
let indices = vec![0, 2, 4];
let selected = v.slice_at(indices)?;     // [1.0, 3.0, 5.0]

// Boolean mask filtering
let mask = BooleanVector::from_slice(&[true, false, true, false, true]);
let filtered = v.slice_at(mask)?;        // [1.0, 3.0, 5.0]

// Partial ranges
let from_index = v.slice_at(2..)?;       // [3.0, 4.0, 5.0]
let to_index = v.slice_at(..3)?;         // [1.0, 2.0, 3.0]
```

### 3. Extension Trait (Natural owned operations)

Bridge Index trait limitations with owned results:

```rust
use rustlab_math::{VectorF64, NaturalSlicing, BooleanVector, vec64};

let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];

// Owned slicing operations  
let owned_slice = v.slice_owned(1..4);           // [2.0, 3.0, 4.0]
let from_owned = v.slice_from_owned(2);          // [3.0, 4.0, 5.0]
let to_owned = v.slice_to_owned(3);              // [1.0, 2.0, 3.0]

// Fancy indexing
let selected = v.select(vec![0, 2, 4]);          // [1.0, 3.0, 5.0]

// Boolean filtering
let mask = BooleanVector::from_slice(&[true, false, true, false, true]);
let filtered = v.select_where(mask);             // [1.0, 3.0, 5.0]

// Method chaining
let processed = v.slice_owned(1..4)              // [2.0, 3.0, 4.0]
                 .slice_owned(0..2);             // [2.0, 3.0]
```

### 2D Array Slicing

Matrices support tuple-based 2D slicing:

```rust
use rustlab_math::{ArrayF64, array64};

let A = array64![[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]];

// Submatrix extraction
let sub = A.slice_2d_at((1..3, 0..2))?;     // [[4.0, 5.0], [7.0, 8.0]]

// Single row (as 1×n matrix)
let row = A.slice_2d_at((1, ..))?;          // [[4.0, 5.0, 6.0]]

// Single column (as n×1 matrix)
let col = A.slice_2d_at((.., 0))?;          // [[1.0], [4.0], [7.0]]

// Corner extraction with negative indexing
let corner = A.slice_2d_at((-2.., -2..))?; // [[5.0, 6.0], [8.0, 9.0]]
```

---

## 🔍 Zero-Copy Views (.view() Method)

The `.view()` method is **critical for high-performance numerical computing** in RustLab. It creates lightweight, zero-copy references to matrix/vector data, enabling efficient operations without memory allocation.

### Why Views are Essential

Views solve three fundamental problems in numerical computing:

1. **Memory Efficiency**: Avoid copying large matrices/vectors
2. **Clean APIs**: Functions can accept data without taking ownership
3. **Performance**: Enable zero-copy operation chains

### Basic View Usage

```rust
use rustlab_math::{ArrayF64, VectorF64, ArrayView, VectorView};

let matrix = ArrayF64::ones(10000, 5000);  // ~400MB of data
let vector = VectorF64::ones(5000);        // ~40KB of data

// Create zero-copy views (instantaneous, no allocation)
let matrix_view = matrix.view();           // ArrayView<f64>
let vector_view = vector.view();           // VectorView<f64>

// All operators work with views
let result = matrix_view ^ vector_view;    // Matrix multiplication
let scaled = matrix_view * 2.0;            // Scalar multiplication
let sum = matrix_view + matrix_view;       // Element-wise addition
```

### Performance Impact

The performance difference is dramatic for large data:

```rust
let A = ArrayF64::ones(1000, 1000);  // ~8MB
let B = ArrayF64::ones(1000, 1000);  // ~8MB

// ❌ WITHOUT views - creates copies (slow)
let result1 = A.clone() ^ B.clone();  // ~16MB copied + computation

// ✅ WITH views - zero-copy (fast)
let result2 = A.view() ^ B.view();    // 0 bytes copied + computation
// Typically 3-5x faster for large matrices!
```

### Function Design with Views

Views enable clean, efficient function signatures:

```rust
use rustlab_math::{ArrayView, VectorView, VectorF64};

// ❌ BAD: Takes ownership, forces clones
fn bad_multiply(A: ArrayF64, v: VectorF64) -> VectorF64 {
    A ^ v  // Caller loses A and v!
}

// 🔶 OKAY: Takes references, but requires & everywhere  
fn okay_multiply(A: &ArrayF64, v: &VectorF64) -> VectorF64 {
    A ^ v  // Works but less ergonomic
}

// ✅ BEST: Takes views - most ergonomic and efficient
fn best_multiply(A: ArrayView<f64>, v: VectorView<f64>) -> VectorF64 {
    A ^ v  // Clean syntax, zero-copy, caller keeps data
}

// Usage comparison
let A = ArrayF64::ones(1000, 500);
let v = VectorF64::ones(500);

// Clean caller syntax with views
let result = best_multiply(A.view(), v.view());
// A and v are still usable here!
```

### Chaining Operations Without Copies

Views enable efficient operation chains:

```rust
let A = ArrayF64::ones(100, 100);
let B = ArrayF64::ones(100, 100);
let C = ArrayF64::ones(100, 100);

// ❌ Creates intermediate temporary matrices
let result1 = (&A ^ &B) ^ &C;  // Temporary 100×100 matrix allocated

// ✅ Zero-copy throughout the chain
let result2 = A.view() ^ B.view() ^ C.view();  // No temporaries!
```

### Mixed Operations with Views

Views seamlessly mix with owned data and references:

```rust
let matrix = ArrayF64::ones(100, 50);
let vector = VectorF64::ones(50);

// All these work
let r1 = matrix.view() ^ vector.view();    // view × view
let r2 = &matrix ^ vector.view();          // reference × view
let r3 = matrix.view() ^ &vector;          // view × reference  
let r4 = matrix.clone() + matrix.view();   // owned + view
```

### Library API Best Practices

When designing libraries, use views for maximum flexibility:

```rust
use rustlab_math::{ArrayView, VectorView, ArrayF64, VectorF64};

pub struct LinearModel {
    weights: ArrayF64,
    bias: VectorF64,
}

impl LinearModel {
    // Accept views in public APIs
    pub fn predict(&self, X: ArrayView<f64>) -> VectorF64 {
        X ^ self.weights.view() + &self.bias
    }
    
    // Support batch operations efficiently
    pub fn predict_batch(&self, samples: &[ArrayView<f64>]) -> Vec<VectorF64> {
        samples.iter()
            .map(|&x| self.predict(x))
            .collect()
    }
}

// Users can pass any form of data
let model = LinearModel { 
    weights: ArrayF64::ones(10, 5),
    bias: VectorF64::zeros(5),
};

let data = ArrayF64::ones(100, 10);
let predictions = model.predict(data.view());  // Clean API
```

### View Lifetime Management

Views borrow data, so Rust's lifetime system ensures safety:

```rust
let result = {
    let matrix = ArrayF64::ones(100, 100);
    matrix.view()  // ❌ Error: matrix doesn't live long enough
};

// Correct approach
let matrix = ArrayF64::ones(100, 100);
let result = {
    let view = matrix.view();  // ✅ matrix outlives view
    view ^ view  // Operations produce owned results
};  // view dropped here, matrix still available
```

### When to Use Views vs References

| Scenario | Use `.view()` | Use `&` Reference |
|----------|--------------|-------------------|
| Function parameters | ✅ Best choice | Okay but less clear |
| Temporary computations | ✅ Zero-copy | Requires careful lifetimes |
| Library APIs | ✅ Most flexible | Too restrictive |
| Simple operations | Optional | ✅ Simpler syntax |
| Storing in structs | ❌ Lifetime complexity | ❌ Same issue |

### Advanced View Patterns

```rust
// View-based algorithm implementation
pub fn power_iteration(A: ArrayView<f64>, iterations: usize) -> VectorF64 {
    let n = A.ncols();
    let mut v = VectorF64::ones(n);
    
    for _ in 0..iterations {
        v = A ^ v.view();  // Use view to avoid move
        let norm = (&v ^ &v).sqrt();
        v = &v / norm;
    }
    
    v
}

// Sliding window with views (zero-copy)
pub fn sliding_window_mean(data: &VectorF64, window_size: usize) -> VectorF64 {
    let n = data.len() - window_size + 1;
    let mut means = VectorF64::zeros(n);
    
    for i in 0..n {
        let window = &data[i..i + window_size];  // Zero-copy slice
        means[i] = window.iter().sum::<f64>() / window_size as f64;
    }
    
    means
}
```

### Performance Guidelines

1. **Always use views for large matrices** (> 1000 elements)
2. **Use views in hot loops** to avoid allocation overhead
3. **Design APIs to accept views** for maximum flexibility
4. **Chain operations with views** to eliminate temporaries
5. **Profile to verify** - views typically provide 2-5x speedup

### Summary

The `.view()` method is not just a convenience - it's **essential for production-grade numerical computing**. It enables:

- **Zero-copy operations** on large datasets
- **Clean API design** without ownership complications
- **Significant performance gains** (often 3-5x faster)
- **Memory efficiency** for large-scale computations
- **Functional programming patterns** in Rust

**Rule of thumb**: When in doubt, use `.view()` - it's almost always the right choice for numerical operations!

---

## 🏭 Data Creation and Manipulation

### Creation Functions

Comprehensive NumPy-style creation utilities:

```rust
use rustlab_math::creation::*;

// Real matrices and vectors
let zeros_mat = zeros(3, 4);              // 3×4 zero matrix
let ones_mat = ones(2, 3);                // 2×3 ones matrix
let identity = eye(5);                    // 5×5 identity matrix
let filled = fill(3, 3, 2.5);             // 3×3 matrix filled with 2.5

let zeros_vec = zeros_vec(100);           // 100-element zero vector
let ones_vec = ones_vec(50);              // 50-element ones vector
let filled_vec = fill_vec(10, 3.14);      // 10-element vector filled with π

// Complex creation (math-first syntax)
let c_zeros = cmatrix!(zeros: 3, 3);      // 3×3 complex zero matrix
let c_eye = cmatrix!(eye: 4);             // 4×4 complex identity matrix
let c_diag = cmatrix!(cdiag: [(1.0, 0.0), (0.0, 1.0)]); // Complex diagonal

// Sequential data generation
let points = linspace(0.0, 1.0, 101);     // 101 points from 0 to 1
let indices = arange(100);                // [0.0, 1.0, 2.0, ..., 99.0]
let custom = arange_step(0.0, 2.0, 0.1);  // [0.0, 0.1, 0.2, ..., 1.9]

// Template-based creation (shape copying)
let A = array64![[1.0, 2.0], [3.0, 4.0]];
let Z = carray64![[(1.0, 0.0), (0.0, 1.0)], [(1.0, 1.0), (2.0, -1.0)]];

let B = zeros_like(&A);                   // 2×2 real zeros with same shape as A
let C = ones_like(&A);                    // 2×2 real ones with same shape as A
let Z_zeros = zeros_like(&Z);             // 2×2 complex zeros with same shape as Z
let Z_ones = ones_like(&Z);               // 2×2 complex ones with same shape as Z

let v = vec64![1.0, 2.0, 3.0];
let w = cvec64![(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
let u = zeros_like_vec(&v);               // 3-element real zeros 
let z = zeros_like_vec(&w);               // 3-element complex zeros
```

### Convenient Macros

Natural syntax for creating arrays and vectors:

```rust
use rustlab_math::{array64, vec64};

// Array creation with natural nested syntax
let A = array64![
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
];

// Vector creation with array-like syntax
let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];

// Complex numbers with math-first syntax
let complex_mat = carray64![
    [(1.0, 2.0), (3.0, 4.0)],
    [(5.0, 6.0), (7.0, 8.0)]
];

// Alternative: real matrices with complex operations
let real_mat = array64![
    [1.0, 3.0],
    [5.0, 7.0]
];
let imag_mat = array64![
    [2.0, 4.0],
    [6.0, 8.0]
];
// Complex arithmetic with real matrices: real_mat + i*imag_mat
```

### Array-Vector Interoperability

Convert between arrays and vectors, and extract matrix components:

```rust
use rustlab_math::{ArrayF64, VectorF64, array64, vec64};

let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let A = array64![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

// Vector to array (reshape)
let reshaped = v.to_array(2, 3)?;        // 2×3 array from 6-element vector
let as_col = v.to_column_array();         // 5×1 column array  
let as_row = v.to_row_array();            // 1×5 row array

// Array to vector
let flattened = A.to_vector();            // Flatten 2×3 array to 6-element vector

// Extract matrix rows/columns as vectors
let row_0 = A.row(0)?;                    // First row as vector
let col_1 = A.col(1)?;                    // Second column as vector

// Matrix decomposition
let rows: Vec<VectorF64> = A.rows();      // All rows as vector collection
let cols: Vec<VectorF64> = A.cols();      // All columns as vector collection
```

### List Comprehension with Automatic Parallelism

Powerful NumPy/Julia-style vectorization with intelligent parallelism:

```rust
use rustlab_math::{vectorize, meshgrid, linspace};

// Basic list comprehensions
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let squared = vectorize![x * x, for x in &data];              // Auto-parallel
let serial_only = vectorize![serial: x * x, for x in &data]; // Zero-overhead serial

// Complex operations (automatically parallel for small datasets)
let simulation_results = vectorize![
    complex: expensive_simulation(x),
    for x in &small_dataset  // Parallels even with 50+ elements
];

// Coordinate grid operations
let x = linspace(-2.0, 2.0, 100);
let y = linspace(-1.0, 1.0, 50);
let (X, Y) = meshgrid!(x: x, y: y);

// Mathematical surface evaluation
let surface = vectorize![
    x_val.powi(2) + y_val.powi(2),  // f(x,y) = x² + y²
    for (x_val, y_val) in X.iter().zip(Y.iter())
];
```

**Parallelization Strategy**: Uses cost-based decisions (`complexity_factor × elements ≥ 500,000`)
- Simple math functions: 50,000+ elements needed
- Complex simulations: Only 50+ elements needed
- See [LIST_COMPREHENSION_AI_DOCUMENTATION.md](LIST_COMPREHENSION_AI_DOCUMENTATION.md) for complete details

### Concatenation Operations

Joining matrices and vectors efficiently with multiple syntax options:

#### Method-Based Concatenation
```rust
use rustlab_math::{ArrayF64, VectorF64, concatenation::*};

let A = ArrayF64::ones(2, 3);
let B = ArrayF64::zeros(2, 2);

// Horizontal concatenation (side-by-side)
let horizontal = A.hstack(&B)?;           // 2×5 matrix

// Vertical concatenation (stacked)
let C = ArrayF64::ones(1, 3);
let vertical = A.vstack(&C)?;             // 3×3 matrix

// Vector concatenation
let v1 = vec64![1.0, 2.0];
let v2 = vec64![3.0, 4.0];
let combined = v1.append(&v2)?;           // [1.0, 2.0, 3.0, 4.0]
```

#### MAT*/NumPy-Style Concatenation Macros (Math-First)

**For AI Code Generation**: Use these macros for natural mathematical syntax:

```rust
use rustlab_math::{array64, hcat, vcat, hstack, vstack, block};

// Create test matrices
let A = array64![[1.0, 2.0], [3.0, 4.0]];     // 2×2
let B = array64![[5.0], [6.0]];               // 2×1
let C = array64![[7.0, 8.0]];                 // 1×2

// MAT*-style horizontal concatenation: [A, B]
let result1 = hcat![A, B].unwrap();            // 2×3 matrix
// Equivalent to: result1 = [A, B] in MAT*
// Equivalent to: result1 = np.hstack([A, B]) in NumPy

// MAT*-style vertical concatenation: [A; C]  
let result2 = vcat![A, C].unwrap();            // 3×2 matrix
// Equivalent to: result2 = [A; C] in MAT*
// Equivalent to: result2 = np.vstack([A, C]) in NumPy

// Alternative syntax (identical functionality)
let result3 = hstack![A, B].unwrap();          // Same as hcat!
let result4 = vstack![A, C].unwrap();          // Same as vcat!

// 2D block matrix construction
let D = array64![[9.0]];                       // 1×1
let block_matrix = block![
    [A, B],    // Top row: [2×2, 2×1] 
    [C, D]     // Bottom row: [1×2, 1×1]
].unwrap();    // Result: 3×3 matrix
// Equivalent to: [A, B; C, D] in MAT*
```

#### Multiple Matrix Concatenation

```rust
use rustlab_math::{array64, hcat, vcat};

let A = array64![[1.0, 2.0], [3.0, 4.0]];
let B = array64![[5.0, 6.0], [7.0, 8.0]];  
let C = array64![[9.0, 10.0], [11.0, 12.0]];

// Chain multiple matrices horizontally
let wide_matrix = hcat![A, B, C].unwrap();     // 2×6 matrix

// Chain multiple matrices vertically  
let tall_matrix = vcat![A, B, C].unwrap();     // 6×2 matrix
```

#### Concatenation Syntax Comparison

| Operation | MAT* | NumPy | RustLab (macro) | RustLab (method) |
|-----------|------|-------|-----------------|------------------|
| Horizontal | `[A, B]` | `np.hstack([A, B])` | `hcat![A, B]` | `A.hstack(&B)` |
| Vertical | `[A; B]` | `np.vstack([A, B])` | `vcat![A, B]` | `A.vstack(&B)` |
| Block 2D | `[A, B; C, D]` | Complex | `block![[A, B], [C, D]]` | Method chain |

---

## 📡 Broadcasting and Element-wise Operations

### Automatic Broadcasting

NumPy-style broadcasting with natural syntax:

```rust
use rustlab_math::{ArrayF64, VectorF64, array64, vec64};

let data = array64![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];  // 2×3 matrix

// Row-wise broadcasting (vector length = columns)
let col_means = vec64![1.5, 2.5, 3.5];                   // Length 3
let centered = &data - &col_means;        // Subtract from each column

// Column-wise broadcasting (vector length = rows)  
let row_bias = vec64![10.0, 20.0];                       // Length 2
let biased = &data + &row_bias;           // Add to each row

// Scalar broadcasting
let scaled = &data * 2.0;                 // Scale all elements
let shifted = &data + 1.0;                // Add to all elements
```

### Broadcasting Rules

RustLab follows NumPy broadcasting rules:

1. **Scalar-Matrix**: Scalar applied to every matrix element
2. **Vector-Matrix (row)**: Vector length must match matrix columns
3. **Vector-Matrix (col)**: Vector length must match matrix rows
4. **Matrix-Matrix**: Dimensions must be identical

```rust
let A = ArrayF64::ones(100, 5);           // 100 samples, 5 features
let means = VectorF64::ones(5);           // Feature means (length 5)
let stds = VectorF64::ones(5);            // Feature stds (length 5)

// Feature normalization (z-score)
let normalized = (&A - &means) / &stds;   // Automatic broadcasting
```

---

## 📊 Statistical Operations

### Basic Statistics

Essential statistical functions with numerical stability:

```rust
use rustlab_math::{VectorF64, statistics::BasicStatistics, vec64};

let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

// Central tendency
let mean = data.mean();                   // Arithmetic mean: 5.5
let median = data.median()?;              // Median value: 5.5

// Variability measures
let variance = data.var(None);            // Sample variance (ddof=1)
let std_dev = data.std(None);             // Sample standard deviation
let pop_var = data.var(Some(0));          // Population variance
let pop_std = data.std(Some(0));          // Population standard deviation

// Range statistics
let min_val = data.min().unwrap();        // Minimum: 1.0
let max_val = data.max().unwrap();        // Maximum: 10.0
let range = max_val - min_val;            // Range: 9.0

// Aggregation
let total = data.sum_elements();          // Sum: 55.0
let product = data.product();             // Product of all elements
```

### Axis-specific Reductions

NumPy-style reductions along matrix dimensions:

```rust
use rustlab_math::{ArrayF64, reductions::{AxisReductions, Axis}};

let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)?; // 2×3 matrix

// Column statistics (reduce along rows, Axis::Rows)
let col_means = data.mean_axis(Axis::Rows)?;      // [2.5, 3.5, 4.5] - per feature
let col_sums = data.sum_axis(Axis::Rows)?;        // [5.0, 7.0, 9.0] - column sums
let col_stds = data.std_axis(Axis::Rows)?;        // Standard deviation per feature

// Row statistics (reduce along columns, Axis::Cols)  
let row_means = data.mean_axis(Axis::Cols)?;      // [2.0, 5.0] - per sample
let row_sums = data.sum_axis(Axis::Cols)?;        // [6.0, 15.0] - sample totals

// Common ML pattern: feature normalization
let feature_means = data.mean_axis(Axis::Rows)?;
let feature_stds = data.std_axis(Axis::Rows)?;
// let normalized = (&data - &feature_means) / &feature_stds;  // Broadcasting
```

### MAT*/NumPy-Style Statistical Macros (Math-First)

**For AI Code Generation**: Use these macros for natural mathematical syntax:

```rust
use rustlab_math::{ArrayF64, array64, sum, mean, min, max, std, var};

// Create sample data matrix
let grades = array64![
    [85.0, 92.0, 78.0],  // Student 1: Math, Science, English  
    [90.0, 88.0, 85.0],  // Student 2
    [75.0, 95.0, 82.0],  // Student 3
    [88.0, 91.0, 89.0]   // Student 4
];  // 4×3 matrix: 4 students, 3 subjects

// Column statistics (axis=0) - Per subject analysis
let subject_totals = sum![grades, axis=0].unwrap();     // Total per subject: [338, 366, 334]
let subject_averages = mean![grades, axis=0].unwrap(); // Average per subject: [84.5, 91.5, 83.5]
let subject_highs = max![grades, axis=0].unwrap();      // Highest per subject: [90, 95, 89] 
let subject_lows = min![grades, axis=0].unwrap();       // Lowest per subject: [75, 88, 78]
let subject_stds = std![grades, axis=0].unwrap();       // Std dev per subject
let subject_vars = var![grades, axis=0].unwrap();       // Variance per subject

// Row statistics (axis=1) - Per student analysis
let student_totals = sum![grades, axis=1].unwrap();     // Total per student: [255, 263, 252, 268]
let student_averages = mean![grades, axis=1].unwrap(); // Average per student: [85.0, 87.7, 84.0, 89.3]
let student_highs = max![grades, axis=1].unwrap();      // Highest per student: [92, 90, 95, 91]
let student_lows = min![grades, axis=1].unwrap();       // Lowest per student: [78, 85, 75, 88]

// With keepdims for broadcasting compatibility
let col_means_keep = mean![grades, axis=0, keep=true].unwrap();  // Shape: (1, 3)
let row_means_keep = mean![grades, axis=1, keep=true].unwrap();  // Shape: (4, 1)
```

#### Statistical Macros Reference

| Macro | MAT* Equivalent | NumPy Equivalent | Description |
|-------|----------------|------------------|-------------|
| `sum![A, axis=0]` | `sum(A, 1)` | `np.sum(A, axis=0)` | Column sums |
| `sum![A, axis=1]` | `sum(A, 2)` | `np.sum(A, axis=1)` | Row sums |
| `mean![A, axis=0]` | `mean(A, 1)` | `np.mean(A, axis=0)` | Column means |
| `mean![A, axis=1]` | `mean(A, 2)` | `np.mean(A, axis=1)` | Row means |
| `min![A, axis=0]` | `min(A, [], 1)` | `np.min(A, axis=0)` | Column minimums |
| `max![A, axis=0]` | `max(A, [], 1)` | `np.max(A, axis=0)` | Column maximums |
| `std![A, axis=0]` | `std(A, 0, 1)` | `np.std(A, axis=0)` | Column std dev |
| `var![A, axis=0]` | `var(A, 0, 1)` | `np.var(A, axis=0)` | Column variance |

#### Keepdims Support

All statistical macros support `keep=true` for preserving dimensions:

```rust
use rustlab_math::{array64, mean, sum};

let data = array64![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

// Without keepdims (default) - returns vector
let col_means = mean![data, axis=0].unwrap();        // VectorF64: [2.5, 3.5, 4.5]

// With keepdims - returns matrix for broadcasting
let col_means_keep = mean![data, axis=0, keep=true].unwrap();  // ArrayF64: [[2.5, 3.5, 4.5]]
let row_means_keep = mean![data, axis=1, keep=true].unwrap();  // ArrayF64: [[2.0], [5.0]]

// Enable broadcasting operations
let normalized = &data - &col_means_keep;            // Subtract column means from all rows
let centered = &data - &row_means_keep;              // Subtract row means from all columns
```

#### Performance Comparison

```rust
use rustlab_math::{ArrayF64, reductions::{AxisReductions, Axis}, mean};

let large_data = ArrayF64::ones(1000, 500);  // 1000×500 matrix

// Traditional verbose syntax
let old_way = large_data.mean_axis_keepdims(Axis::Rows).unwrap();

// Math-first macro syntax (identical performance)
let new_way = mean![large_data, axis=0, keep=true].unwrap();

// Both approaches:
// - Zero-cost abstractions
// - Identical performance  
// - Same memory layout
// - SIMD optimizations
```

---

## 💾 File I/O Operations

### Math-First I/O with Ultimate Simplicity

**CRITICAL for AI Code Generation**: RustLab provides the cleanest possible I/O API with just 2 core functions for all file operations. This prevents the most common AI hallucination of using complex, multi-function I/O APIs.

#### The Complete I/O API (Only 4 Functions Total)

```rust
use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
use rustlab_math::io::MathIO;

// Create data
let A = array64![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];

// SAVE (2 functions total) - NO OTHER SAVE FUNCTIONS EXIST
A.save("matrix.csv")?;                     // Basic save (6 decimal places)
A.save_with_precision("data.csv", 10)?;    // High precision (10 decimal places)

// LOAD (2 functions total) - NO OTHER LOAD FUNCTIONS EXIST  
let B = ArrayF64::load("matrix.csv")?;          // Basic load
let C = ArrayF64::load_skip("data.csv", 3)?;    // Skip 3 header/metadata lines

// Same functions work for vectors
v.save("vector.txt")?;                     // Auto-detects text format
let w = VectorF64::load("vector.txt")?;    // Loads vector back
```

#### Key I/O Features (All Automatic)

- **Format Auto-Detection**: `.csv` → comma-delimited, other → space-delimited
- **Header Detection**: Non-numeric first lines automatically skipped
- **Scientific Notation**: Numbers > 1e6 or < 1e-4 use scientific notation
- **Vector Layout**: CSV files save as rows, text files save as columns
- **Precision Control**: 1-15+ decimal places for scientific data
- **Metadata Support**: Skip comment/header lines with `load_skip()`

#### Real-World I/O Examples

**Loading Scientific Data with Metadata:**
```rust
use rustlab_math::{ArrayF64, VectorF64};
use rustlab_math::io::MathIO;

// File: experiment.csv
// # Experimental Data - 2024-01-15
// # Instrument: Spectrometer  
// # Units: wavelength (nm), intensity (counts)
// wavelength,intensity
// 400.0,1250.0
// 500.0,1800.0  
// 600.0,2100.0

// Load skipping 3 comment lines (header auto-detected and skipped)
let data = ArrayF64::load_skip("experiment.csv", 3)?;
let wavelengths = (0..data.nrows()).map(|i| data.get(i, 0).unwrap()).collect::<Vec<_>>();
let intensities = (0..data.nrows()).map(|i| data.get(i, 1).unwrap()).collect::<Vec<_>>();
```

**Saving High-Precision Computational Results:**
```rust
use rustlab_math::{ArrayF64, array64};
use rustlab_math::io::MathIO;

// High-precision scientific constants
let constants = array64![
    [std::f64::consts::PI, std::f64::consts::E],
    [std::f64::consts::TAU, std::f64::consts::SQRT_2]
];

// Save with maximum precision for scientific accuracy
constants.save_with_precision("constants.csv", 15)?;

// Save with reduced precision for reports
constants.save_with_precision("report_data.csv", 3)?;
```

**Working with Large Datasets:**
```rust
// Load large computational results (buffered automatically)
let simulation_data = ArrayF64::load("large_simulation.csv")?;

// Process and save subset
let important_region = simulation_data.slice_2d_at((0..1000, 0..50))?;
important_region.save("processed_subset.csv")?;
```

**Vector I/O with Different Formats:**
```rust
let measurements = vec64![12.345, 67.890, 23.456, 78.123];

// Text format: column vector (default for .txt)
measurements.save("measurements.txt")?;     
// File contents:
// 12.345000
// 67.890000  
// 23.456000
// 78.123000

// CSV format: row vector (default for .csv)
measurements.save("measurements.csv")?;
// File contents: 12.345000,67.890000,23.456000,78.123000

// Load back (format auto-detected)
let loaded_txt = VectorF64::load("measurements.txt")?;
let loaded_csv = VectorF64::load("measurements.csv")?;
// Both produce identical vectors
```

#### NumPy/MAT* I/O Migration

| NumPy/MAT* | RustLab | Description |
|------------|---------|-------------|
| `np.savetxt("data.txt", A)` | `A.save("data.txt")?` | Save matrix to text |
| `np.savetxt("data.csv", A, delimiter=",")` | `A.save("data.csv")?` | Save matrix to CSV |
| `np.savetxt("data.txt", A, fmt="%.10f")` | `A.save_with_precision("data.txt", 10)?` | High precision save |
| `np.loadtxt("data.txt")` | `ArrayF64::load("data.txt")?` | Load matrix |
| `np.loadtxt("data.csv", delimiter=",")` | `ArrayF64::load("data.csv")?` | Load CSV |
| `np.loadtxt("data.txt", skiprows=3)` | `ArrayF64::load_skip("data.txt", 3)?` | Skip metadata |
| `np.genfromtxt("data.csv", skip_header=2)` | `ArrayF64::load_skip("data.csv", 2)?` | Skip headers |
| MAT* `save("data.txt", A, "-ascii")` | `A.save("data.txt")?` | Save as text |
| MAT* `load("data.txt")` | `ArrayF64::load("data.txt")?` | Load from text |

#### I/O Error Handling

```rust
use rustlab_math::{ArrayF64, Result};
use rustlab_math::io::MathIO;

fn safe_data_processing(file_path: &str) -> Result<ArrayF64> {
    // All I/O operations return Result<T> for safe error handling
    let raw_data = ArrayF64::load_skip(file_path, 2)?;  // Skip metadata
    
    // Validate data dimensions
    if raw_data.ncols() < 3 {
        return Err("Insufficient columns in data file".into());
    }
    
    // Process and save results
    let processed = raw_data.slice_2d_at((0..100, 0..3))?;
    processed.save("processed_data.csv")?;
    
    Ok(processed)
}

// Usage with proper error handling
match safe_data_processing("experiment.csv") {
    Ok(data) => println!("Successfully processed {} samples", data.nrows()),
    Err(e) => eprintln!("Data processing failed: {}", e),
}
```

#### AI Guidelines for I/O Code Generation

**✅ ALWAYS Use These Patterns:**
```rust
// Correct I/O imports
use rustlab_math::io::MathIO;

// Correct save operations
data.save("file.csv")?;                    // Basic save
data.save_with_precision("file.csv", 12)?; // High precision

// Correct load operations  
let matrix = ArrayF64::load("file.csv")?;        // Basic load
let clean = ArrayF64::load_skip("file.csv", 3)?; // Skip metadata

// Correct error handling
let result = ArrayF64::load("data.csv")?;  // Use ? operator or .unwrap()
```

**❌ NEVER Use These (Common AI Hallucinations):**
```rust
// ❌ WRONG - These functions DO NOT exist
ArrayF64::load_csv("file.csv")?;          // No separate CSV function
data.save_txt("file.txt", config)?;       // No config objects  
ArrayF64::from_csv_file("data.csv")?;     // No alternative constructors
load_matrix_from_file("data.csv")?;       // No standalone functions
data.save("file", FileFormat::Csv)?;      // No format specification
ArrayF64::read_file("data.csv")?;         // Wrong function names
```

#### I/O Performance Characteristics

- **Memory Efficient**: Buffered I/O for large files (> 1MB)
- **Format Detection**: Zero-overhead file extension checking
- **Scientific Notation**: Automatic for extreme values (> 1e6, < 1e-4)
- **Precision Control**: Configurable decimal places (1-15+)
- **Error Recovery**: Clear error messages for dimension mismatches
- **Cross-Platform**: Works on Windows, macOS, Linux

#### Integration with Other Operations

```rust
use rustlab_math::{ArrayF64, reductions::{AxisReductions, Axis}};
use rustlab_math::io::MathIO;

// Complete data analysis workflow
let raw_data = ArrayF64::load_skip("experiment.csv", 3)?;   // Load with metadata

// Statistical analysis
let feature_means = raw_data.mean_axis(Axis::Rows)?;
let feature_stds = raw_data.std_axis(Axis::Rows)?;

// Data normalization  
let normalized = (&raw_data - &feature_means) / &feature_stds;

// Save results with different precisions
normalized.save("normalized_data.csv")?;                    // Standard precision
feature_means.save_with_precision("statistics.csv", 8)?;   // High precision stats

// Create summary report
let summary = array64![
    [feature_means.get(0).unwrap(), feature_stds.get(0).unwrap()],
    [feature_means.get(1).unwrap(), feature_stds.get(1).unwrap()]
];
summary.save("summary_report.csv")?;
```

---

## 🧬 Functional Programming  

### Map, Filter, and Reduce Operations

Functional programming paradigms with mathematical operations:

```rust
use rustlab_math::{VectorF64, functional::*, vec64};

let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];

// Transformations (map)
let squared = data.map(|x| x * x);            // [1.0, 4.0, 9.0, 16.0, 25.0]
let scaled = data.map(|x| x * 2.0 + 1.0);     // [3.0, 5.0, 7.0, 9.0, 11.0]
let logs = data.map(|x| x.ln());              // Natural logarithm

// Filtering (conditional selection)
let large = data.filter(|x| x > 3.0);         // [4.0, 5.0]
let even_indices = data.filter_enumerate(|i, _| i % 2 == 0); // Elements at even indices

// Reductions (fold/reduce)  
let sum = data.fold(0.0, |acc, x| acc + x);   // Sum: 15.0
let product = data.fold(1.0, |acc, x| acc * x); // Product: 120.0
let max = data.fold(f64::NEG_INFINITY, |acc, x| acc.max(x)); // Maximum

// Scanning (cumulative operations)
let cumsum = data.scan(0.0, |acc, x| *acc += x); // [1.0, 3.0, 6.0, 10.0, 15.0]
let cumprod = data.scan(1.0, |acc, x| *acc *= x); // [1.0, 2.0, 6.0, 24.0, 120.0]

// Combining operations
let v2 = vec64![2.0, 1.0, 3.0, 2.0, 1.0];
let combined = data.zip_with(&v2, |a, b| a * b + 1.0)?; // Element-wise combination
```

### Chaining Operations

Efficient operation chaining with lazy evaluation:

```rust
let data = vec64![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];

let result = data
    .map(|x| x.abs())                         // Absolute values
    .filter(|x| x > 2.0)                      // Keep values > 2
    .map(|x| x.sqrt())                        // Square root
    .fold(0.0, |acc, x| acc + x);             // Sum the results
```

---

## ⚡ Performance Features

### Automatic SIMD Optimization

Transparent vectorization for large arrays:

```rust
use rustlab_math::{ArrayF64, VectorF64};

// For arrays/vectors > 64 elements, SIMD is automatically applied
let large_vec = VectorF64::zeros(1000);      // SIMD operations enabled
let large_mat = ArrayF64::zeros(100, 100);   // SIMD operations enabled

// Element-wise operations use SIMD when beneficial
let result = &large_vec + &large_vec;        // Vectorized addition
let scaled = &large_vec * 2.0;               // Vectorized scaling
```

### Memory Layout Optimization

Cache-friendly operations via faer backend:

```rust
// Arrays are cache-aligned (64-byte alignment)
let A = ArrayF64::zeros(1000, 1000);         // Optimal memory layout

// Operations preserve cache locality
let B = ArrayF64::ones(1000, 1000);
let C = &A + &B;                             // Cache-efficient addition
```

### Zero-Copy Views

Efficient slicing without allocation:

```rust
let large_data = VectorF64::zeros(1_000_000);

// Zero-copy slicing (no allocation)
let window = &large_data[1000..2000];        // O(1) operation
let sum = window.iter().sum::<f64>();        // Process efficiently

// Views can be reused
for i in 0..100 {
    let chunk = &large_data[i*1000..(i+1)*1000];
    // Process each chunk without allocation
}
```

### Reference Operations

Use references to avoid unnecessary moves:

```rust
let A = array64![[1.0, 2.0], [3.0, 4.0]];
let B = array64![[2.0, 1.0], [1.0, 2.0]];

// Correct: Use references to avoid moves
let sum = &A + &B;                           // A and B remain usable
let product = &A ^ &B;                       // Matrix multiplication

// Avoid: Operations that consume operands
// let sum = A + B;                          // A and B are moved (consumed)
```

---

## 🔄 Migration from NumPy/MAT*

### Common Operations Translation

| NumPy/MAT* | RustLab | Description |
|--------------|---------|-------------|
| `np.zeros(100)` | `VectorF64::zeros(100)` | Zero vector |
| `np.zeros((3, 4))` | `ArrayF64::zeros(3, 4)` | Zero matrix |
| `np.ones((3, 4))` | `ArrayF64::ones(3, 4)` | Ones matrix |
| `np.eye(5)` | `ArrayF64::eye(5)` | Identity matrix |
| `np.linspace(0, 1, 100)` | `linspace(0.0, 1.0, 100)` | Linear spacing |
| `np.arange(100)` | `arange(100)` | Integer sequence |
| `A @ B` | `&A ^ &B` | Matrix multiplication |
| `A * B` | `&A * &B` | Element-wise multiplication |
| `np.dot(a, b)` | `&a ^ &b` | Dot product |
| `A[1:4, 0:2]` | `A.slice_2d_at((1..4, 0..2))?` | 2D slicing |
| `arr[mask]` | `arr.slice_at(mask)?` | Boolean indexing |
| `np.mean(A, axis=0)` | `mean![A, axis=0].unwrap()` or `A.mean_axis(Axis::Rows)?` | Column means |
| `np.sum(A, axis=1)` | `sum![A, axis=1].unwrap()` or `A.sum_axis(Axis::Cols)?` | Row sums |
| `np.min(A, axis=0)` | `min![A, axis=0].unwrap()` | Column minimums |
| `np.max(A, axis=1)` | `max![A, axis=1].unwrap()` | Row maximums |
| `np.std(A, axis=0)` | `std![A, axis=0].unwrap()` | Column std deviation |
| `np.var(A, axis=1)` | `var![A, axis=1].unwrap()` | Row variance |
| `np.hstack([A, B])` | `hcat![A, B].unwrap()` or `A.hstack(&B)?` | Horizontal concatenation |
| `np.vstack([A, B])` | `vcat![A, B].unwrap()` or `A.vstack(&B)?` | Vertical concatenation |
| `[A, B]` (MAT*) | `hcat![A, B].unwrap()` | MAT* horizontal concatenation |
| `[A; B]` (MAT*) | `vcat![A, B].unwrap()` | MAT* vertical concatenation |
| `[A, B; C, D]` (MAT*) | `block![[A, B], [C, D]].unwrap()` | MAT* block matrix |
| `np.array([[1+2j, 3-1j]])` | `carray64![[(1.0, 2.0), (3.0, -1.0)]]` | Complex matrix creation |
| `np.array([1+2j, 3-1j])` | `cvec64![(1.0, 2.0), (3.0, -1.0)]` | Complex vector creation |
| `np.zeros((3,3), dtype=complex)` | `cmatrix!(zeros: 3, 3)` | Complex zero matrix |
| `np.eye(3, dtype=complex)` | `cmatrix!(eye: 3)` | Complex identity matrix |
| `np.savetxt("data.txt", A)` | `A.save("data.txt")?` | Save matrix to text |
| `np.savetxt("data.csv", A, delimiter=",")` | `A.save("data.csv")?` | Save matrix to CSV |
| `np.savetxt("data.txt", A, fmt="%.10f")` | `A.save_with_precision("data.txt", 10)?` | High precision save |
| `np.loadtxt("data.txt")` | `ArrayF64::load("data.txt")?` | Load matrix from text |
| `np.loadtxt("data.csv", delimiter=",")` | `ArrayF64::load("data.csv")?` | Load matrix from CSV |
| `np.loadtxt("data.txt", skiprows=3)` | `ArrayF64::load_skip("data.txt", 3)?` | Skip metadata rows |
| `np.genfromtxt("data.csv", skip_header=2)` | `ArrayF64::load_skip("data.csv", 2)?` | Skip header lines |
| MAT* `save("data.txt", A, "-ascii")` | `A.save("data.txt")?` | Save matrix as ASCII |
| MAT* `load("data.txt")` | `ArrayF64::load("data.txt")?` | Load matrix from file |

### Key Differences

1. **Error Handling**: RustLab uses `Result` types instead of exceptions
2. **Memory Management**: Rust's ownership system prevents memory leaks
3. **Type Safety**: Compile-time dimension checking where possible
4. **Operator Distinction**: `^` for mathematical multiplication, `*` for element-wise
5. **I/O Simplicity**: Just 4 total I/O functions vs NumPy's many specialized functions
6. **Automatic Format Detection**: File extension determines format, no manual specification needed

### Migration Example

**NumPy Code:**
```python
import numpy as np

# Create data
A = np.random.rand(100, 5)
means = np.mean(A, axis=0)
stds = np.std(A, axis=0)

# Normalize
normalized = (A - means) / stds

# Linear algebra
x = np.random.rand(5)
y = A @ x  # Matrix-vector multiplication
dot_result = np.dot(x, x)  # Dot product
```

**RustLab Code:**
```rust
use rustlab_math::{ArrayF64, VectorF64, reductions::{AxisReductions, Axis}};

// Create data (would use random generation in practice)
let A = ArrayF64::ones(100, 5);  // Placeholder for random data
let means = A.mean_axis(Axis::Rows)?;
let stds = A.std_axis(Axis::Rows)?;

// Normalize (broadcasting automatically applies)
let normalized = (&A - &means) / &stds;

// Linear algebra
let x = VectorF64::ones(5);      // Placeholder for random data
let y = &A ^ &x;                 // Matrix-vector multiplication
let dot_result = &x ^ &x;        // Dot product
```

### MAT*-Style Concatenation Example

**MAT* Code:**
```matlab
% Create matrices
A = [1, 2; 3, 4];
B = [5; 6];
C = [7, 8];
D = [9];

% Concatenation operations
horizontal = [A, B];              % 2×3 matrix
vertical = [A; C];                % 3×2 matrix
block_matrix = [A, B; C, D];      % 3×3 block matrix

% Multiple concatenations
E = [10, 11; 12, 13];
F = [14, 15; 16, 17];
chain_h = [A, E, F];              % 2×6 matrix
chain_v = [A; E; F];              % 6×2 matrix
```

**RustLab Code:**
```rust
use rustlab_math::{array64, hcat, vcat, block};

// Create matrices (identical to MAT*)
let A = array64![[1.0, 2.0], [3.0, 4.0]];
let B = array64![[5.0], [6.0]];
let C = array64![[7.0, 8.0]];
let D = array64![[9.0]];

// Concatenation operations (natural MAT* syntax)
let horizontal = hcat![A, B].unwrap();           // 2×3 matrix
let vertical = vcat![A, C].unwrap();             // 3×2 matrix
let block_matrix = block![
    [A, B],
    [C, D]
].unwrap();                                      // 3×3 block matrix

// Multiple concatenations
let E = array64![[10.0, 11.0], [12.0, 13.0]];
let F = array64![[14.0, 15.0], [16.0, 17.0]];
let chain_h = hcat![A, E, F].unwrap();           // 2×6 matrix
let chain_v = vcat![A, E, F].unwrap();           // 6×2 matrix
```

**Key Benefits of RustLab Macros:**
- **Familiar Syntax**: Direct translation from MAT*/NumPy
- **Type Safety**: Compile-time dimension checking
- **Performance**: Zero-cost abstractions with optimal memory layout
- **Error Handling**: Explicit `Result` types prevent runtime crashes

---

## 💡 Common Patterns and Best Practices

### 1. Always Use References for Operations

```rust
// ✅ Correct: Use references to avoid moves
let A = array64![[1.0, 2.0], [3.0, 4.0]];
let B = array64![[2.0, 1.0], [1.0, 2.0]];

let sum = &A + &B;               // A and B remain usable
let product = &A ^ &B;           // Matrix multiplication

// ❌ Avoid: Operations that consume operands
// let sum = A + B;              // A and B are moved and unusable
```

### 2. Operator Distinction is Critical

```rust
let A = array64![[1.0, 2.0], [3.0, 4.0]];
let B = array64![[2.0, 1.0], [1.0, 2.0]];

// ✅ Correct: Matrix multiplication
let matrix_mult = &A ^ &B;       // Mathematical multiplication

// ✅ Correct: Element-wise multiplication  
let hadamard = &A * &B;          // Element-wise (Hadamard) product

// ❌ Common mistake: Using * for matrix multiplication
// let wrong = &A * &B;          // This is element-wise, not matrix multiplication!
```

### 3. Dimension Checking

```rust
let A = ArrayF64::zeros(3, 4);   // 3×4 matrix
let B = ArrayF64::zeros(4, 2);   // 4×2 matrix
let v = VectorF64::zeros(4);     // 4-element vector

// ✅ Correct: Inner dimensions match
let C = &A ^ &B;                 // (3×4) × (4×2) → (3×2) ✓
let u = &A ^ &v;                 // (3×4) × (4×1) → (3×1) ✓

// ❌ Would fail: Dimension mismatch
// let D = &A ^ &A;              // (3×4) × (3×4) - inner dimensions don't match!
```

### 4. Efficient Slicing Patterns

```rust
let large_data = VectorF64::zeros(1_000_000);

// ✅ Efficient: Use views for temporary access
for i in 0..1000 {
    let window = &large_data[i*1000..(i+1)*1000];  // Zero-copy view
    let mean = window.iter().sum::<f64>() / window.len() as f64;
    // Process without allocation
}

// ✅ Efficient: Convert to owned only when needed
let important_slice = large_data.slice_at(1000..2000)?;  // Owned copy for storage
```

### 5. Feature Normalization Pattern

```rust
use rustlab_math::{ArrayF64, reductions::{AxisReductions, Axis}};

// Common machine learning preprocessing
let data = ArrayF64::ones(1000, 10);        // 1000 samples, 10 features

// Compute statistics per feature (column-wise)
let feature_means = data.mean_axis(Axis::Rows)?;
let feature_stds = data.std_axis(Axis::Rows)?;

// Z-score normalization (broadcasting applies automatically)
let normalized_data = (&data - &feature_means) / &feature_stds;
```

### 6. Safe Indexing Practices

```rust
let v = vec64![1.0, 2.0, 3.0, 4.0, 5.0];

// ✅ Safe: Use .get() for bounds checking
if let Some(value) = v.get(10) {
    println!("Value: {}", value);
} else {
    println!("Index out of bounds");
}

// ✅ Safe: Use .at() for negative indexing
if let Some(last) = v.at(-1) {
    println!("Last element: {}", last);
}

// ✅ Safe: Use slice_at() with error handling
match v.slice_at(1..10) {
    Ok(slice) => {
        // Process slice
    },
    Err(e) => {
        println!("Slicing error: {}", e);
    }
}
```

### 7. Performance Optimization

```rust
// ✅ Chain operations for compiler optimization
let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
let result = (&data + &data) * 2.0 - &data;  // Single optimized loop

// ✅ Use SIMD-friendly sizes when possible
let simd_friendly = VectorF64::zeros(1024);  // Power of 2, > SIMD threshold

// ✅ Preallocate workspace arrays
let workspace = ArrayF64::zeros(1000, 1000); // Reuse for multiple operations
```

### 8. Complex Number Best Practices (Math-First)

```rust
use rustlab_math::{ArrayC64, VectorC64, carray64, cvec64, cmatrix};

// ✅ Correct: Use math-first macros for complex numbers
let Z = carray64![[(1.0, 2.0), (3.0, -1.0)],    // 1+2i, 3-i
                  [(0.0, 1.0), (2.0, 0.0)]];     // i, 2

let w = cvec64![(1.0, 0.0), (0.0, 1.0)];        // 1, i

// ✅ Special complex matrices with ergonomic syntax
let C_zeros = cmatrix!(zeros: 3, 3);             // 3×3 complex zeros
let C_eye = cmatrix!(eye: 4);                    // 4×4 complex identity
let C_diag = cmatrix!(cdiag: [(1.0, 0.0), (0.0, 1.0)]); // Complex diagonal

// ✅ Mixed real-complex operations work naturally
let A_real = array64![[1.0, 2.0], [3.0, 4.0]];
let mixed_result = &A_real ^ &w;                 // Real matrix × complex vector

// ❌ Avoid: Verbose Complex::new() syntax
// let bad_Z = array64![[Complex::new(1.0, 2.0), Complex::new(3.0, -1.0)]];

// ✅ Mathematical operations respect complex arithmetic
let Z_mult = &Z ^ &Z;                            // Complex matrix multiplication
let z_dot = &w ^ &w;                             // Complex dot product (with conjugation)
let z_norm = (&w ^ &w).norm();                   // Vector norm in complex space
```

### 9. Error Handling Best Practices  

```rust
use rustlab_math::Result;

fn process_data(data: &ArrayF64) -> Result<VectorF64> {
    // Chain operations with ? operator
    let means = data.mean_axis(Axis::Rows)?;
    let result = data.slice_2d_at((0..10, 0..5))?;
    let final_result = result.sum_axis(Axis::Cols)?;
    
    Ok(final_result)
}

// Use the function
let data = ArrayF64::ones(100, 10);
match process_data(&data) {
    Ok(result) => {
        // Process successful result
    },
    Err(e) => {
        eprintln!("Processing failed: {}", e);
    }
}
```

---

## 🎓 Summary

RustLab Math provides a comprehensive, high-performance numerical computing library with:

- **Natural mathematical syntax** with the critical `^` vs `*` operator distinction
- **Comprehensive slicing system** with three complementary approaches
- **NumPy-style operations** for familiar scientific computing workflows
- **Zero-cost abstractions** with automatic SIMD optimization
- **Type safety** with compile-time dimension checking
- **Memory efficiency** through zero-copy views and careful memory management

The library is designed specifically to prevent common AI code generation mistakes while providing maximum mathematical expressiveness and performance.

For more detailed information, refer to the inline documentation in each module, which follows the [AI Documentation Template](../AI_DOCUMENTATION_TEMPLATE.md) for optimal AI code generation support.