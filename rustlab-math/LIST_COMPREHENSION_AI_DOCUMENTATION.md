# List Comprehension with Automatic Parallelism - AI Documentation

## Function Summary - Quick Reference

### Core Functions

#### 1. `vectorize_with_complexity<T, F, R>(data, complexity, f) -> Vec<R>`
**Purpose**: Core vectorization with explicit complexity specification (takes ownership)
```rust
let results = vectorize_with_complexity(data, Complexity::Complex, |x| expensive_fn(x));
```
- **When to use**: When you know the operation complexity and want to consume data
- **Parallelism**: Based on total cost (complexity_factor × size ≥ 500,000)

#### 2. `vectorize_with_complexity_ref<T, F, R>(data: &[T], complexity, f) -> Vec<R>`
**Purpose**: Reference-based vectorization without cloning overhead
```rust
let results = vectorize_with_complexity_ref(&data, Complexity::Simple, |x| x.sin());
```
- **When to use**: When you need to keep the original data
- **Parallelism**: Based on total cost (complexity_factor × size ≥ 500,000)

#### 3. `vectorize_serial<T, F, R>(data: &[T], f) -> Vec<R>`
**Purpose**: Zero-overhead serial vectorization
```rust
let results = vectorize_serial(&data, |x| x * 2.0);
```
- **When to use**: When you know parallelism won't help
- **Parallelism**: Never (guaranteed serial execution)

#### 4. `vectorize_adaptive<T, F, R>(data, f) -> Vec<R>`
**Purpose**: Automatic complexity detection and optimization
```rust
let results = vectorize_adaptive(data, |x| unknown_function(x));
```
- **When to use**: Unknown function complexity
- **Parallelism**: Measures first 10 elements, then decides

#### 5. `vectorize_chunked<T, F, R>(data, chunk_size, f) -> Vec<R>`
**Purpose**: Memory-efficient batch processing for huge datasets
```rust
let results = vectorize_chunked(huge_data, 1000, |chunk| process_batch(chunk));
```
- **When to use**: Very large datasets, memory constraints
- **Parallelism**: Always parallel at chunk level

#### 6. `meshgrid(x: &VectorF64, y: &VectorF64) -> (ArrayF64, ArrayF64)`
**Purpose**: Generate coordinate grids for mathematical surfaces
```rust
let (X, Y) = meshgrid(&x_points, &y_points);
```
- **When to use**: Evaluating functions over 2D grids
- **Output**: Two matrices for coordinate evaluation

### Macros

#### 7. `vectorize!` Macro
**Purpose**: List comprehension syntax with automatic parallelism
```rust
// Zero-overhead serial mode
let serial: Vec<f64> = vectorize![serial: x * 2.0, for x in &data];

// Auto-decide based on Simple complexity (default)
let doubled: Vec<f64> = vectorize![x * 2.0, for x in &data];

// Complex operations (low threshold)
let results: Vec<f64> = vectorize![complex: simulation(x), for x in &data];

// Adaptive complexity
let results: Vec<f64> = vectorize![adaptive: unknown_fn(x), for x in data];

// Explicit complexity
let results: Vec<f64> = vectorize![
    expensive_operation(x), 
    for x in data, 
    complexity = Complexity::Moderate
];
```

#### 6. `meshgrid!` Macro
**Purpose**: Convenient coordinate grid creation
```rust
let (X, Y) = meshgrid!(x: x_vector, y: y_vector);
```

### Vector Extensions

#### 7. `VectorF64::apply_with_complexity<F>(f, complexity) -> VectorF64`
**Purpose**: Apply function to vector elements with specified complexity
```rust
let transformed = vector.apply_with_complexity(|x| x.sin(), Complexity::Simple);
```

#### 8. `VectorF64::apply_adaptive<F>(f) -> VectorF64`
**Purpose**: Apply function with automatic complexity detection
```rust
let processed = vector.apply_adaptive(|x| unknown_function(x));
```

#### 9. `VectorF64::zip_with_complexity<F>(other, f, complexity) -> Result<VectorF64>`
**Purpose**: Binary operation on two vectors with specified complexity
```rust
let combined = v1.zip_with_complexity(&v2, |a, b| complex_op(a, b), Complexity::Complex)?;
```

### Supporting Types and Traits

#### 10. `Complexity` Enum
**Purpose**: Define operation complexity levels with cost-based parallelization
- `Trivial`: Basic arithmetic (factor 1, needs ≥500,000 elements)
- `Simple`: Math functions like sin/cos (factor 10, needs ≥50,000 elements)
- `Moderate`: Matrix ops, FFT (factor 100, needs ≥5,000 elements)
- `Complex`: Simulations, ML (factor 10,000, needs ≥50 elements)

**Parallelization Formula**: `complexity_factor × number_of_elements ≥ 500,000`

#### 11. `CostModel::measure_complexity<F, T, R>(f, samples) -> Complexity`
**Purpose**: Automatically profile function complexity
```rust
let complexity = CostModel::measure_complexity(&expensive_fn, &sample_data);
```

#### 12. `Computable` Trait
**Purpose**: Trait for types with known computational complexity
```rust
impl Computable for MyExpensiveOp {
    const COMPLEXITY: Complexity = Complexity::Complex;
    type Output = f64;
    fn compute(self) -> f64 { /* ... */ }
}
```

### Quick Reference by Use Case

| **Use Case** | **Function** | **Example** |
|--------------|--------------|-------------|
| **Guaranteed serial** | `vectorize![serial: expr, for x in &data]` | `vectorize![serial: x * 2.0, for x in &data]` |
| **Simple math on vectors** | `vectorize![expr, for x in &data]` | `vectorize![x * 2.0, for x in &data]` |
| **Complex operations** | `vectorize![complex: expr, for x in &data]` | `vectorize![complex: simulation(x), for x in &data]` |
| **Unknown complexity** | `vectorize![adaptive: expr, for x in data]` | `vectorize![adaptive: user_fn(*x), for x in data]` |
| **Huge datasets** | `vectorize_chunked(data, size, f)` | `vectorize_chunked(big_data, 1000, process)` |
| **2D coordinate grids** | `meshgrid!(x: xvec, y: yvec)` | `meshgrid!(x: linspace(0,1,100), y: linspace(0,2,200))` |
| **Vector transformations** | `vec.apply_adaptive(f)` | `vec.apply_adaptive(\|x\| expensive(x))` |
| **Vector combinations** | `v1.zip_with_complexity(&v2, f, c)` | `v1.zip_with_complexity(&v2, \|a,b\| a+b, Complexity::Trivial)` |

### Performance Optimization Tips

1. **Use `&data` to avoid cloning**: `vectorize![expr, for x in &data]` is more efficient than `for x in data.clone()`
2. **Use `serial:` for known small/fast operations**: Guarantees zero overhead
3. **Let the system decide for medium workloads**: Default Simple complexity works well for most cases
4. **Force `complex:` for expensive operations**: Even 50 elements will parallelize for truly expensive functions
5. **Reference syntax avoids overhead**: Using `&data` eliminates clone costs

### Key Innovation: Cost-Based Parallelism

The main innovation is **automatic parallelism decisions based on total computational cost**:

- **Traditional**: Parallelize based only on data size
- **rustlab-math**: Parallelize based on `complexity_factor × data_size ≥ 500,000`

This means:
- Complex operations (ML, simulations) parallelize with just 50+ elements
- Simple operations (sin/cos) need 50,000+ elements to benefit from parallelism
- Trivial operations (arithmetic) need 500,000+ elements
- Memory and CPU resources are used optimally

### When Parallelism Kicks In - Examples

| **Operation Type** | **Complexity** | **Factor** | **Parallel Threshold** | **Example** |
|-------------------|---------------|------------|----------------------|-------------|
| Basic arithmetic | Trivial | 1 | 500,000+ elements | `x + 1`, `x * 2` |
| Math functions | Simple | 10 | 50,000+ elements | `x.sin()`, `x.sqrt()` |
| Matrix operations | Moderate | 100 | 5,000+ elements | Matrix multiply, FFT |
| ML/Simulations | Complex | 10,000 | 50+ elements | Neural nets, Monte Carlo |

### Example Performance Scenarios

```rust
// 10,000 elements with sin/cos - NO parallelism (cost = 10 × 10,000 = 100,000 < 500,000)
let result = vectorize![x.sin() * x.cos(), for x in &data_10k];

// 50,000 elements with sin/cos - YES parallelism (cost = 10 × 50,000 = 500,000)
let result = vectorize![x.sin() * x.cos(), for x in &data_50k];

// 100 elements with complex simulation - YES parallelism (cost = 10,000 × 100 = 1,000,000)
let result = vectorize![complex: run_simulation(x), for x in &data_100];

// Force serial for any size - zero overhead
let result = vectorize![serial: x * 2.0, for x in &any_data];
```

---

## Overview for AI Code Generation

The `rustlab_math::comprehension` module provides advanced list comprehension capabilities with **complexity-aware automatic parallelism**. This feature enables NumPy/Julia-style vectorized operations in Rust with optimal performance characteristics.

### Key Innovation: Complexity-Based Parallelization

Unlike traditional approaches that parallelize based solely on data size, this system automatically decides whether to use parallel processing based on **operation complexity**:

- **Complex operations** (neural networks, simulations): Parallelize with as few as 10 elements
- **Simple operations** (arithmetic): Only parallelize with 10,000+ elements
- **Adaptive mode**: Automatically profiles unknown functions to determine complexity

## Critical API Patterns for AI

### 1. Basic Vectorization Macro

```rust
use rustlab_math::vectorize;

// Simple operations - automatically uses appropriate parallelism
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let doubled: Vec<f64> = vectorize![x * 2.0, for x in data];
// Result: [2.0, 4.0, 6.0, 8.0, 10.0]
```

### 2. Complex Operations - Low Parallelism Threshold

```rust
// Monte Carlo simulation - parallel even for small datasets
let seeds = vec![1, 2, 3, 4, 5];  // Only 5 elements!
let prices: Vec<f64> = vectorize![
    complex: monte_carlo_price_simulation(seed, 100000),
    for seed in seeds
];
// Automatically parallelizes because marked as 'complex'
```

### 3. Adaptive Complexity Detection

```rust
// Unknown function complexity - automatically measured and optimized
let data = vec![1.0, 2.0, 3.0]; // Small dataset
let results: Vec<f64> = vectorize![
    adaptive: expensive_unknown_function(x),
    for x in data
];
// First few iterations measure execution time, then decides on parallelism
```

### 4. Explicit Complexity Control

```rust
use rustlab_math::Complexity;

let batch = vec![/* neural network inputs */];
let predictions: Vec<f64> = vectorize![
    model.forward_pass(input),
    for input in batch,
    complexity = Complexity::Complex  // Force low parallelism threshold
];
```

### 5. Vector Extensions for Comprehensions

```rust
use rustlab_math::{VectorF64, Complexity};

let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);

// Apply function with specified complexity
let transformed = vec.apply_with_complexity(
    |x| expensive_transform(x), 
    Complexity::Moderate
);

// Adaptive complexity detection
let processed = vec.apply_adaptive(|x| unknown_function(x));

// Zip two vectors with binary function
let vec2 = VectorF64::from_slice(&[5.0, 6.0, 7.0, 8.0]);
let combined = vec.zip_with_complexity(
    &vec2, 
    |a, b| complex_combination(a, b),
    Complexity::Complex
).unwrap();
```

### 6. Coordinate Grid Generation (meshgrid)

```rust
use rustlab_math::{meshgrid, VectorF64, linspace};

// Create coordinate grids for mathematical functions
let x = linspace(0.0, 1.0, 100);  // 100 points from 0 to 1
let y = linspace(0.0, 2.0, 200);  // 200 points from 0 to 2

let (X, Y) = meshgrid!(x: x, y: y);
// X: 200×100 matrix where X[i,j] = x[j]
// Y: 200×100 matrix where Y[i,j] = y[i]

// Now evaluate function over the entire grid
let Z = X.zip_with_complexity(&Y, |x, y| x*x + y*y, Complexity::Simple)?;
```

## Complexity Classifications for AI

### Complexity::Trivial (Threshold: 10,000 elements)
- Basic arithmetic: `x + 1`, `x * 2`, `x - constant`
- Simple comparisons: `x > threshold`
- Array indexing operations

### Complexity::Simple (Threshold: 1,000 elements)  
- Mathematical functions: `sin(x)`, `cos(x)`, `sqrt(x)`, `exp(x)`
- String operations: basic formatting, concatenation
- Simple statistical operations: `mean()`, basic reductions

### Complexity::Moderate (Threshold: 100 elements)
- Matrix operations: matrix multiplication, eigenvalues
- FFT operations, signal processing
- Image processing kernels (convolution, filtering)
- Sorting algorithms, tree operations

### Complexity::Complex (Threshold: 10 elements)
- Monte Carlo simulations
- Neural network forward/backward passes
- Optimization algorithms (gradient descent, etc.)
- Financial modeling (option pricing, risk calculations)
- Physics simulations, differential equation solving

## Performance Characteristics

### Automatic Parallelism Decision Tree

```rust
// Internal logic (automatically applied):
fn should_parallelize(size: usize, complexity: Complexity) -> bool {
    let threshold = match complexity {
        Complexity::Trivial  => 10_000,  // Only large datasets
        Complexity::Simple   => 1_000,   // Medium datasets  
        Complexity::Moderate => 100,     // Small datasets
        Complexity::Complex  => 10,      // Even tiny datasets
    };
    size >= threshold
}
```

### Memory Efficiency Features

```rust
// Chunked processing for huge datasets
use rustlab_math::vectorize_chunked;

let huge_data = vec![/* millions of elements */];
let results = vectorize_chunked(
    huge_data,
    1000,  // Process in chunks of 1000
    |chunk| {
        chunk.iter().map(|x| expensive_function(x)).collect()
    }
);
```

## Integration with Existing rustlab-math Features

### Works with All Vector Types

```rust
use rustlab_math::{VectorF64, VectorF32, VectorC64};

// All vector types support comprehension extensions
let vec_f64 = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
let vec_f32 = VectorF32::from_slice(&[1.0, 2.0, 3.0]);
let vec_complex = VectorC64::from_slice(&[/* complex numbers */]);

// Apply transformations with automatic parallelism
let result_f64 = vec_f64.apply_adaptive(|x| x.sin());
let result_f32 = vec_f32.apply_with_complexity(|x| x * x, Complexity::Trivial);
```

### Chainable with Existing Operations

```rust
use rustlab_math::{linspace, vectorize};

// Chain list comprehension with other rustlab-math operations  
let x = linspace(0.0, 10.0, 1000);
let processed = x
    .apply_adaptive(|x| x.sin())  // Apply function with auto-parallelism
    .slice_at(100..900).unwrap()  // Slice result
    * 2.0;  // Scale with broadcasting

// Or use in mathematical expressions
let data = vec![1.0, 2.0, 3.0];
let transformed: Vec<f64> = vectorize![x.powi(3) + x.sin(), for x in data];
let vector_result = VectorF64::from_slice(&transformed);
```

## Error Handling Patterns

### Dimension Mismatches

```rust
use rustlab_math::{VectorF64, MathError};

let v1 = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
let v2 = VectorF64::from_slice(&[1.0, 2.0]);  // Different size!

match v1.zip_with_complexity(&v2, |a, b| a + b, Complexity::Trivial) {
    Ok(result) => println!("Combined: {:?}", result),
    Err(MathError::InvalidSliceLength { expected, actual }) => {
        println!("Size mismatch: expected {}, got {}", expected, actual);
    }
    Err(e) => println!("Other error: {}", e),
}
```

## Advanced Usage Examples

### 1. Monte Carlo Portfolio Optimization

```rust
use rustlab_math::{vectorize, VectorF64, Complexity};

fn optimize_portfolio(returns: &VectorF64, n_simulations: usize) -> Vec<f64> {
    let seeds: Vec<u64> = (0..n_simulations).collect();
    
    // Even with just 100 simulations, this parallelizes due to complexity
    vectorize![
        complex: monte_carlo_portfolio_simulation(returns, seed),
        for seed in seeds
    ]
}
```

### 2. Neural Network Batch Processing

```rust
struct NeuralNetwork { /* ... */ }

impl NeuralNetwork {
    fn process_batch(&self, inputs: Vec<VectorF64>) -> Vec<VectorF64> {
        // Automatically parallel even for small batches
        vectorize![
            complex: self.forward_pass(input),
            for input in inputs
        ]
    }
}
```

### 3. Signal Processing Pipeline

```rust
use rustlab_math::{vectorize, VectorF64, Complexity};

fn process_audio_segments(audio: &VectorF64, segment_size: usize) -> Vec<VectorF64> {
    let segments: Vec<VectorF64> = audio.chunks(segment_size)
        .map(|chunk| VectorF64::from_slice(chunk))
        .collect();
    
    // FFT is moderate complexity - parallelizes at 100+ elements
    vectorize![
        fft_transform_and_filter(segment),
        for segment in segments,
        complexity = Complexity::Moderate
    ]
}
```

### 4. Mathematical Surface Evaluation

```rust
use rustlab_math::{meshgrid, linspace, vectorize};

fn evaluate_mathematical_surface(f: fn(f64, f64) -> f64) -> ArrayF64 {
    let x = linspace(-5.0, 5.0, 100);
    let y = linspace(-5.0, 5.0, 100);
    let (X, Y) = meshgrid!(x: x, y: y);
    
    // Element-wise function evaluation over the grid
    // Automatically determines parallelism based on function complexity
    let flat_coords: Vec<(f64, f64)> = X.iter()
        .zip(Y.iter())
        .map(|(x, y)| (*x, *y))
        .collect();
        
    let flat_results: Vec<f64> = vectorize![
        adaptive: f(coord.0, coord.1),
        for coord in flat_coords
    ];
    
    ArrayF64::from_slice(&flat_results, 100, 100).unwrap()
}
```

## Best Practices for AI Code Generation

### 1. Always Specify Complexity for Known Functions

```rust
// Good - explicit complexity for known expensive operations
let results: Vec<f64> = vectorize![
    optimize_parameters(data),
    for data in datasets,
    complexity = Complexity::Complex
];

// Avoid - letting simple operations auto-parallelize inefficiently
let simple_results: Vec<f64> = vectorize![
    x + 1.0,
    for x in huge_array  // This will parallelize, which may be overkill
];
```

### 2. Use Adaptive Mode for Unknown Functions

```rust
// Good - let the system learn the complexity
let results: Vec<f64> = vectorize![
    adaptive: user_defined_function(x),
    for x in data
];
```

### 3. Combine with rustlab-math Ecosystem

```rust
// Good - integrate with existing mathematical operations
let processed = linspace(0.0, 10.0, 1000)
    .apply_adaptive(|x| complex_transform(x))  // Auto-parallel comprehension
    .reduce_axis(0, |acc, x| acc + x)          // Use existing reductions
    .unwrap();
```

### 4. Handle Errors Appropriately

```rust
// Good - proper error propagation
fn process_vectors(v1: &VectorF64, v2: &VectorF64) -> Result<VectorF64, MathError> {
    v1.zip_with_complexity(v2, |a, b| expensive_combine(a, b), Complexity::Complex)
}
```

## Summary for AI Systems

The `rustlab_math::comprehension` module provides:

1. **`vectorize!` macro** - List comprehension with automatic parallelism
2. **Complexity-aware thresholds** - Smart parallelization based on operation cost
3. **Adaptive profiling** - Automatic complexity detection for unknown functions  
4. **Vector extensions** - Integration with existing rustlab-math types
5. **Meshgrid generation** - NumPy-style coordinate grid creation
6. **Memory efficiency** - Chunked processing for large datasets

This enables writing high-performance mathematical code that automatically optimizes for both simple arithmetic and complex computational tasks, making it ideal for scientific computing, machine learning, and financial modeling applications.