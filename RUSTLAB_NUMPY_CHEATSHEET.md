# RustLab vs NumPy Comprehensive Cheatsheet

A complete comparison of RustLab and NumPy operations for scientific computing, data analysis, and machine learning.

## 🎯 Critical Distinctions

### Operator Philosophy
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Matrix multiplication | `@` or `np.dot()` | `^` | RustLab's signature operator |
| Element-wise multiplication | `*` | `*` | Same behavior |
| Matrix power | `np.linalg.matrix_power()` | Not `^` | `^` is for multiplication only |

### Memory Management
| Concept | NumPy | RustLab | Benefits |
|---------|-------|---------|----------|
| Default behavior | Often copies | References (`&`) | Zero-copy by default |
| Views | `arr.view()` | `arr.view()` or `&arr[..]` | Extensive zero-copy support |
| Memory ownership | Garbage collected | Rust ownership | No GC pauses |

## 📦 Imports and Setup

```python
# NumPy
import numpy as np
```

```rust
// RustLab - Core types
use rustlab_math::{ArrayF64, VectorF64, array64, vec64};

// RustLab - Operations
use rustlab_math::functional::*;
use rustlab_math::statistics::BasicStatistics;
use rustlab_math::reductions::{AxisReductions, Axis};
use rustlab_math::io::MathIO;
use rustlab_math::creation::*;
```

## 🔧 Array/Vector Creation

### Basic Creation
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| 2D array from literal | `np.array([[1, 2], [3, 4]])` | `array64![[1.0, 2.0], [3.0, 4.0]]` |
| 1D vector from literal | `np.array([1, 2, 3])` | `vec64![1.0, 2.0, 3.0]` |
| Zeros matrix | `np.zeros((3, 4))` | `ArrayF64::zeros(3, 4)` or `zeros(3, 4)` |
| Ones matrix | `np.ones((3, 4))` | `ArrayF64::ones(3, 4)` or `ones(3, 4)` |
| Identity matrix | `np.eye(5)` | `ArrayF64::eye(5)` or `eye(5)` |
| Zeros vector | `np.zeros(100)` | `VectorF64::zeros(100)` or `zeros_vec(100)` |
| Ones vector | `np.ones(100)` | `VectorF64::ones(100)` or `ones_vec(100)` |
| From list/slice | `np.array(data)` | `ArrayF64::from_slice(&data, rows, cols)?` |
| Complex matrix | `np.array([[1+2j, 3-1j]])` | `carray64![[(1.0, 2.0), (3.0, -1.0)]]` |
| Complex vector | `np.array([1+2j, 3-1j])` | `cvec64![(1.0, 2.0), (3.0, -1.0)]` |

### Sequential Generation
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Linear spacing | `np.linspace(0, 1, 100)` | `linspace(0.0, 1.0, 100)` |
| Range | `np.arange(100)` | `arange(100)` |
| Range with step | `np.arange(0, 2, 0.1)` | `arange_step(0.0, 2.0, 0.1)` |
| Like another | `np.zeros_like(arr)` | `zeros_like(&arr)` |
| Like another (ones) | `np.ones_like(arr)` | `ones_like(&arr)` |
| Filled with value | `np.full((3, 3), 2.5)` | `fill(3, 3, 2.5)` |

### Special Matrices
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Diagonal matrix | `np.diag([1, 2, 3])` | `matrix!(diag: [1, 2, 3])` |
| Random uniform | `np.random.rand(3, 4)` | Use `rand` crate with RustLab |
| Random normal | `np.random.randn(3, 4)` | Use `rand_distr` crate with RustLab |

## ➕ Mathematical Operations

### Element-wise Operations
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Addition | `A + B` | `&A + &B` | Use `&` to avoid move |
| Subtraction | `A - B` | `&A - &B` | |
| Multiplication (element-wise) | `A * B` | `&A * &B` | Hadamard product |
| Division | `A / B` | `&A / &B` | Element-wise |
| Scalar multiplication | `A * 2` | `&A * 2.0` | |
| Scalar addition | `A + 1` | `&A + 1.0` | |
| Power | `A ** 2` | `A.map(\|x\| x.powi(2))` | |
| Negation | `-A` | `-&A` | |

### Matrix Operations
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Matrix multiplication | `A @ B` or `np.dot(A, B)` | `&A ^ &B` | RustLab's ^ operator |
| Matrix-vector product | `A @ v` | `&A ^ &v` | |
| Vector-matrix product | `v @ A` | `&v ^ &A` | |
| Dot product | `np.dot(u, v)` | `&u ^ &v` | Returns scalar |
| Transpose | `A.T` | `A.T()` or `A.transpose()` | Math notation supported |
| Inverse | `np.linalg.inv(A)` | `A.inv()?` | With rustlab-linearalgebra |
| Determinant | `np.linalg.det(A)` | `A.det()?` | With rustlab-linearalgebra |
| Trace | `np.trace(A)` | `A.trace()` | Sum of diagonal |
| Matrix rank | `np.linalg.matrix_rank(A)` | `A.rank()?` | Via SVD |
| Condition number | `np.linalg.cond(A)` | `A.condition_number()?` | |

### Linear Algebra (Advanced)
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Eigenvalues | `np.linalg.eig(A)[0]` | `A.eigenvalues()?` | May return complex |
| Eigenvalues (symmetric) | `np.linalg.eigh(A)[0]` | `A.eigenvalues_self_adjoint()?` | Real values only |
| Eigenvectors | `np.linalg.eig(A)` | `A.eigenvectors()?` | Returns (values, vectors) |
| Eigenvectors (symmetric) | `np.linalg.eigh(A)` | `A.eigenvectors_self_adjoint()?` | Orthogonal vectors |
| LU decomposition | `scipy.linalg.lu(A)` | `A.lu()?` | Returns LU struct |
| QR decomposition | `np.linalg.qr(A)` | `A.qr()?` | Returns QR struct |
| Cholesky decomposition | `np.linalg.cholesky(A)` | `A.cholesky()?` | For SPD matrices |
| SVD | `np.linalg.svd(A)` | `A.svd()?` | Returns SVD struct |
| Solve linear system | `np.linalg.solve(A, b)` | `A.solve_system(&b)?` | General solver |
| Solve (with decomposition) | - | `A.lu()?.solve(&b)?` | Factor once, solve many |
| Least squares | `np.linalg.lstsq(A, b)` | `A.qr()?.solve(&b)?` | Overdetermined systems |
| Pseudoinverse | `np.linalg.pinv(A)` | `A.svd()?.pseudoinverse()?` | Moore-Penrose |

## 🔪 Slicing and Indexing

### Basic Indexing
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Single element | `A[i, j]` | `A.get(i, j)?` |
| Single element (unsafe) | `A[i, j]` | `A[(i, j)]` |
| Set element | `A[i, j] = value` | `A.set(i, j, value)?` |
| Vector element | `v[i]` | `v.get(i)?` or `v[i]` |
| Negative indexing | `v[-1]` | `v.at(-1)?` |

### Slicing Operations
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|
| Vector slice | `v[1:4]` | `&v[1..4]` or `v.slice_at(1..4)?` | Zero-copy vs owned |
| Vector slice (from) | `v[2:]` | `&v[2..]` or `v.slice_at(2..)?` | |
| Vector slice (to) | `v[:3]` | `&v[..3]` or `v.slice_at(..3)?` | |
| Matrix submatrix | `A[1:3, 0:2]` | `A.slice_2d((1..3, 0..2))?` | |
| Matrix row | `A[i, :]` | `A.row(i)?` | Returns vector |
| Matrix column | `A[:, j]` | `A.col(j)?` | Returns vector |
| Fancy indexing | `v[[0, 2, 4]]` | `v.slice_at(vec![0, 2, 4])?` | |
| Boolean indexing | `arr[mask]` | `arr.slice_at(mask)?` or `arr.where_mask(&mask)?` | |

## 📊 Statistical Operations

### Basic Statistics
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Mean | `np.mean(arr)` | `arr.mean()` |
| Standard deviation | `np.std(arr)` | `arr.std(None)` |
| Variance | `np.var(arr)` | `arr.var(None)` |
| Population std | `np.std(arr, ddof=0)` | `arr.std(Some(0))` |
| Sum | `np.sum(arr)` | `arr.sum_elements()` |
| Product | `np.prod(arr)` | `arr.product()` |
| Min | `np.min(arr)` | `arr.min()?` |
| Max | `np.max(arr)` | `arr.max()?` |
| Median | `np.median(arr)` | `arr.median()?` |

### Axis Reductions
| Operation | NumPy | RustLab (Method) | RustLab (Macro) |
|-----------|-------|------------------|-----------------|
| Column mean | `np.mean(A, axis=0)` | `A.mean_axis(Axis::Rows)?` | `mean![A, axis=0]` |
| Row mean | `np.mean(A, axis=1)` | `A.mean_axis(Axis::Cols)?` | `mean![A, axis=1]` |
| Column sum | `np.sum(A, axis=0)` | `A.sum_axis(Axis::Rows)?` | `sum![A, axis=0]` |
| Row sum | `np.sum(A, axis=1)` | `A.sum_axis(Axis::Cols)?` | `sum![A, axis=1]` |
| Column std | `np.std(A, axis=0)` | `A.std_axis(Axis::Rows)?` | `std![A, axis=0]` |
| Row std | `np.std(A, axis=1)` | `A.std_axis(Axis::Cols)?` | `std![A, axis=1]` |
| Keep dimensions | `np.mean(A, axis=0, keepdims=True)` | `A.mean_axis_keepdims(Axis::Rows)?` | `mean![A, axis=0, keep=true]` |

## 🔗 Concatenation and Reshaping

### Concatenation
| Operation | NumPy | RustLab (Method) | RustLab (Macro) |
|-----------|-------|------------------|-----------------|
| Horizontal stack | `np.hstack([A, B])` | `A.hstack(&B)?` | `hcat![A, B]` |
| Vertical stack | `np.vstack([A, B])` | `A.vstack(&B)?` | `vcat![A, B]` |
| Column stack | `np.column_stack([v1, v2])` | - | `hcat![v1.to_column_array(), v2.to_column_array()]` |
| Block matrix | `np.block([[A, B], [C, D]])` | - | `block![[A, B], [C, D]]` |
| Vector append | `np.append(v1, v2)` | `v1.append(&v2)?` | - |

### Reshaping
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Reshape | `v.reshape(2, 3)` | `v.to_array(2, 3)?` |
| Flatten | `A.flatten()` | `A.to_vector()` |
| Ravel | `A.ravel()` | `A.to_vector()` |
| To column vector | `v.reshape(-1, 1)` | `v.to_column_array()` |
| To row vector | `v.reshape(1, -1)` | `v.to_row_array()` |

## 🎭 Comparison Operations

### Basic Comparisons
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Greater than | `arr > threshold` | `arr.gt(threshold)` |
| Greater or equal | `arr >= threshold` | `arr.ge(threshold)` |
| Less than | `arr < threshold` | `arr.lt(threshold)` |
| Less or equal | `arr <= threshold` | `arr.le(threshold)` |
| Equal | `arr == value` | `arr.eq(value)` |
| Not equal | `arr != value` | `arr.ne(value)` |
| Element-wise comparison | `A > B` | `A.gt_array(&B)` or `A.gt_vec(&B)` |
| Close (tolerance) | `np.isclose(A, B)` | `A.is_close(&B, 1e-10, 1e-12)` |

### Boolean Operations
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Logical AND | `mask1 & mask2` | `mask1 & mask2` |
| Logical OR | `mask1 \| mask2` | `mask1 \| mask2` |
| Logical NOT | `~mask` | `!mask` |
| Logical XOR | `mask1 ^ mask2` | `mask1 ^ mask2` (for BooleanVector) |

### Boolean Reductions
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Any true | `np.any(mask)` | `mask.any()` |
| All true | `np.all(mask)` | `mask.all()` |
| Count nonzero | `np.count_nonzero(mask)` | `mask.count_true()` |
| Where (indices) | `np.where(condition)` | `mask.where_true()` |

### Find Operations
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Find first > value | `arr[arr > val][0]` if exists | `arr.find_gt(val)` |
| Find first < value | `arr[arr < val][0]` if exists | `arr.find_lt(val)` |
| Find first == value | `arr[arr == val][0]` if exists | `arr.find_eq(val)` |
| Find index of first > value | `np.where(arr > val)[0][0]` | `arr.find_index_gt(val)` |
| Find index of first < value | `np.where(arr < val)[0][0]` | `arr.find_index_lt(val)` |
| Find index of first == value | `np.where(arr == val)[0][0]` | `arr.find_index_eq(val)` |
| Find with predicate | List comprehension | `arr.find(\|x\| predicate(x))` |
| Find index with predicate | - | `arr.find_index(\|x\| predicate(x))` |
| Find all where mask | `arr[mask]` | `arr.find_where(&mask)?` or `arr.where_mask(&mask)?` |
| Find all indices where mask | `np.where(mask)[0]` | `arr.find_indices(&mask)?` |

### Any/All Shortcuts
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Any > value | `np.any(arr > val)` | `arr.any_gt(val)` |
| Any < value | `np.any(arr < val)` | `arr.any_lt(val)` |
| Any == value | `np.any(arr == val)` | `arr.any_eq(val)` |
| Any >= value | `np.any(arr >= val)` | `arr.any_ge(val)` |
| Any <= value | `np.any(arr <= val)` | `arr.any_le(val)` |
| All > value | `np.all(arr > val)` | `arr.all_gt(val)` |
| All < value | `np.all(arr < val)` | `arr.all_lt(val)` |
| All == value | `np.all(arr == val)` | `arr.all_eq(val)` |
| Any with predicate | - | `arr.any(\|x\| predicate(x))` |
| All with predicate | - | `arr.all(\|x\| predicate(x))` |

## 🔢 Complex Number Operations

### Complex Creation
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Complex from real/imag | `1 + 2j` | `Complex::new(1.0, 2.0)` | Direct constructor |
| Complex vector | `np.array([1+2j, 3-1j])` | `cvec64![(1.0, 2.0), (3.0, -1.0)]` | Macro syntax |
| Complex matrix | `np.array([[1+2j, 3-1j]])` | `carray64![[(1.0, 2.0), (3.0, -1.0)]]` | 2D macro |
| From polar | `r * np.exp(1j * theta)` | `Complex::from_polar(r, theta)` | Polar coordinates |
| Zeros complex | `np.zeros(5, dtype=complex)` | `VectorC64::zeros(5)` | Complex zeros |
| Ones complex | `np.ones(5, dtype=complex)` | `VectorC64::ones(5)` | Complex ones |
| Real part | `z.real` | `z.re` | Extract real |
| Imaginary part | `z.imag` | `z.im` | Extract imaginary |
| Complex conjugate | `np.conj(z)` | `z.conj()` | Conjugation |

### Complex Properties
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Magnitude/Modulus | `np.abs(z)` or `np.absolute(z)` | `z.norm()` | \|z\| |
| Phase/Argument | `np.angle(z)` | `z.arg()` | θ in polar form |
| Squared magnitude | `np.abs(z)**2` | `z.norm_sqr()` | \|z\|² |
| Is real | `z.imag == 0` | `z.im == 0.0` | Check if purely real |
| Is imaginary | `z.real == 0` | `z.re == 0.0` | Check if purely imaginary |

### Complex Arithmetic
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Addition | `z1 + z2` | `z1 + z2` | Complex addition |
| Subtraction | `z1 - z2` | `z1 - z2` | Complex subtraction |
| Multiplication | `z1 * z2` | `z1 * z2` | Complex multiplication |
| Division | `z1 / z2` | `z1 / z2` | Complex division |
| Power | `z ** n` | `z.powf(n)` or `z.powi(n)` | Complex power |
| Square root | `np.sqrt(z)` | `z.sqrt()` | Principal square root |
| Exponential | `np.exp(z)` | `z.exp()` | e^z |
| Natural log | `np.log(z)` | `z.ln()` | Principal branch |
| Reciprocal | `1 / z` | `z.inv()` | 1/z |

### Complex Vector Operations  
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Element-wise conjugate | `np.conj(vec)` | `vec.map(\|z\| z.conj())` | Vector conjugation |
| Vector magnitude | `np.abs(vec)` | `vec.map(\|z\| z.norm())` | \|z_i\| for each element |
| Vector phase | `np.angle(vec)` | `vec.map(\|z\| z.arg())` | arg(z_i) for each |
| Real parts | `vec.real` | `vec.map(\|z\| z.re)` | Extract all real parts |
| Imaginary parts | `vec.imag` | `vec.map(\|z\| z.im)` | Extract all imag parts |
| Dot product (complex) | `np.vdot(u, v)` | `&u ^ &v.map(\|z\| z.conj())` | u·v̄ |
| Inner product | `np.inner(u, v)` | `&u ^ &v` | Standard complex inner |

### Complex Matrix Operations
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Matrix multiplication | `A @ B` | `&A ^ &B` | Complex matrix multiplication |
| Hermitian transpose | `A.T.conj()` | `A.hermitian()` | A† (conjugate transpose) |
| Transpose only | `A.T` | `A.transpose()` | Just transpose, no conjugate |
| Determinant (complex) | `np.linalg.det(A)` | `A.det()?` | Complex determinant |
| Eigenvalues | `np.linalg.eig(A)[0]` | `A.eigenvalues()?` | May be complex |
| Complex trace | `np.trace(A)` | `A.trace()` | Sum of diagonal (complex) |
| Matrix inverse | `np.linalg.inv(A)` | `A.inv()?` | Complex matrix inverse |

### Complex Trigonometric Functions
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Complex sine | `np.sin(z)` | `z.sin()` | sin(z) |
| Complex cosine | `np.cos(z)` | `z.cos()` | cos(z) |
| Complex tangent | `np.tan(z)` | `z.tan()` | tan(z) |
| Complex hyperbolic sine | `np.sinh(z)` | `z.sinh()` | sinh(z) |
| Complex hyperbolic cosine | `np.cosh(z)` | `z.cosh()` | cosh(z) |
| Complex hyperbolic tangent | `np.tanh(z)` | `z.tanh()` | tanh(z) |

### Complex Array Functions
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Complex array from real | `real_arr + 0j` | `real_arr.map(\|x\| Complex::new(x, 0.0))` | Real → Complex |
| Split into real/imag | `arr.real, arr.imag` | `arr.map(\|z\| z.re), arr.map(\|z\| z.im)` | Separate components |
| Combine real/imag | `real + 1j * imag` | `real.zip(&imag).map(\|(r, i)\| Complex::new(r, i))` | Combine arrays |
| Complex broadcasting | `real_matrix + 1j * imag_vector` | Use broadcasting with `Complex::new` | Mixed operations |

### Fourier Transform Preparation
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Complex exponential | `np.exp(1j * theta)` | `Complex::from_polar(1.0, theta)` | e^{iθ} = cos(θ) + i·sin(θ) |
| Twiddle factors | `np.exp(-2j * np.pi * k / N)` | `(0..N).map(\|k\| Complex::from_polar(1.0, -2.0 * PI * k as f64 / N as f64))` | FFT basis |
| Complex unit circle | `np.exp(1j * np.linspace(0, 2*pi, N))` | `linspace(0.0, 2.0*PI, N).map(\|theta\| Complex::from_polar(1.0, theta))` | Unit circle points |

### Advanced Complex Operations
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Complex sorting (by magnitude) | `arr[np.argsort(np.abs(arr))]` | `vec.sort_by(\|a, b\| a.norm().partial_cmp(&b.norm()).unwrap())` | Sort by |z| |
| Complex sorting (by phase) | `arr[np.argsort(np.angle(arr))]` | `vec.sort_by(\|a, b\| a.arg().partial_cmp(&b.arg()).unwrap())` | Sort by arg(z) |
| Real part filtering | `arr[arr.real > 0]` | `vec.filter(\|z\| z.re > 0.0)` | Filter by real part |
| Quadrant analysis | Custom logic | `vec.filter(\|z\| z.re > 0.0 && z.im > 0.0)` | First quadrant |
| Complex mean | `np.mean(arr)` | `vec.fold(Complex::zero(), \|sum, z\| sum + z) / vec.len() as f64` | Average complex |
| Complex variance | Custom calculation | Custom with complex arithmetic | Complex statistics |

### Signal Processing with Complex Numbers
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Analytic signal | `scipy.signal.hilbert(real_signal)` | Use Hilbert transform crate | Complex envelope |
| Frequency shift | `signal * np.exp(1j * 2 * pi * f * t)` | `signal.zip(&time).map(\|(s, t)\| s * Complex::from_polar(1.0, 2.0*PI*f*t))` | Modulation |
| Phase unwrapping | `np.unwrap(np.angle(signal))` | Custom phase unwrapping logic | Continuous phase |
| Complex demodulation | `signal * np.conj(carrier)` | `signal.zip(&carrier).map(\|(s, c)\| s * c.conj())` | Remove carrier |

### Performance Tips for Complex Numbers
```rust
// Efficient complex vector operations
let z1 = cvec64![(1.0, 2.0), (3.0, -1.0), (0.5, 1.5)];
let z2 = cvec64![(2.0, 1.0), (-1.0, 2.0), (1.0, -0.5)];

// Vectorized complex operations (preferred)
let product = z1.zip(&z2).map(|(a, b)| a * b);
let magnitude = z1.map(|z| z.norm());
let phase = z1.map(|z| z.arg());

// Complex matrix operations with views (zero-copy)
let complex_matrix = carray64![[(1.0, 0.0), (0.0, 1.0)], [(0.0, -1.0), (1.0, 0.0)]];
let hermitian = complex_matrix.map(|z| z.conj()).transpose();

// FFT-ready complex vectors
let n = 1024;
let signal: VectorC64 = (0..n)
    .map(|k| Complex::from_polar(1.0, 2.0 * PI * k as f64 / n as f64))
    .collect::<Vec<_>>()
    .into();
```

## 🧮 Mathematical Functions

### Trigonometric
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Sine | `np.sin(A)` | `A.sin()` | Auto-SIMD |
| Cosine | `np.cos(A)` | `A.cos()` | Auto-SIMD |
| Tangent | `np.tan(A)` | `A.tan()` | |
| Arcsine | `np.arcsin(A)` | `A.asin()` | |
| Arccosine | `np.arccos(A)` | `A.acos()` | |
| Arctangent | `np.arctan(A)` | `A.atan()` | |
| Arctangent2 | `np.arctan2(y, x)` | Use `y.map2(&x, \|y, x\| y.atan2(x))` | Two-argument arctangent |

### Hyperbolic Functions
| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Hyperbolic sine | `np.sinh(A)` | `A.sinh()` | |
| Hyperbolic cosine | `np.cosh(A)` | `A.cosh()` | |
| Hyperbolic tangent | `np.tanh(A)` | `A.tanh()` | |
| Inverse hyperbolic sine | `np.arcsinh(A)` | `A.map(\|x\| x.asinh())` | Via scalar function |
| Inverse hyperbolic cosine | `np.arccosh(A)` | `A.map(\|x\| x.acosh())` | Via scalar function |
| Inverse hyperbolic tangent | `np.arctanh(A)` | `A.map(\|x\| x.atanh())` | Via scalar function |

### Exponential and Logarithmic
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Exponential | `np.exp(A)` | `A.exp()` |
| Natural log | `np.log(A)` | `A.ln()` |
| Log base 10 | `np.log10(A)` | `A.log10()` |
| Log base 2 | `np.log2(A)` | `A.log2()` |
| Power | `np.power(A, n)` | `A.map(\|x\| x.powf(n))` |

### Other Functions
| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Square root | `np.sqrt(A)` | `A.sqrt()` |
| Absolute value | `np.abs(A)` | `A.abs()` |
| Sign | `np.sign(A)` | `A.signum()` |
| Ceiling | `np.ceil(A)` | `A.ceil()` |
| Floor | `np.floor(A)` | `A.floor()` |
| Round | `np.round(A)` | `A.round()` |

## 💾 File I/O

| Operation | NumPy | RustLab | Notes |
|-----------|-------|---------|-------|
| Save to text | `np.savetxt("file.txt", A)` | `A.save("file.txt")?` | Auto format detection |
| Save to CSV | `np.savetxt("file.csv", A, delimiter=",")` | `A.save("file.csv")?` | Extension determines format |
| Save with precision | `np.savetxt("file.txt", A, fmt="%.10f")` | `A.save_with_precision("file.txt", 10)?` | |
| Load from text | `np.loadtxt("file.txt")` | `ArrayF64::load("file.txt")?` | |
| Load from CSV | `np.loadtxt("file.csv", delimiter=",")` | `ArrayF64::load("file.csv")?` | |
| Skip headers | `np.loadtxt("file.txt", skiprows=3)` | `ArrayF64::load_skip("file.txt", 3)?` | |
| Load with comments | `np.genfromtxt("file.csv", skip_header=2)` | `ArrayF64::load_skip("file.csv", 2)?` | |

## 🔄 Functional Operations

| Operation | NumPy | RustLab |
|-----------|-------|---------|
| Map function | `np.vectorize(func)(arr)` | `arr.map(func)` |
| Filter | List comprehension | `arr.filter(predicate)` |
| Reduce | `np.reduce()` | `arr.fold(init, func)` |
| Cumulative sum | `np.cumsum(arr)` | `arr.scan(0.0, \|acc, x\| acc + x)` |
| Apply along axis | `np.apply_along_axis()` | Use `map()` with axis operations |
| Meshgrid | `np.meshgrid(x, y)` | `meshgrid!(x: x, y: y)` |

## 🚀 RustLab-Unique Features (Not in NumPy)

### List Comprehension with Automatic Parallelism
```rust
// Automatic complexity-based parallelization
let results = vectorize![x.sin() * x.cos(), for x in &data];

// Force serial execution (zero overhead)
let serial = vectorize![serial: x * x, for x in &data];

// Force parallel for complex operations
let parallel = vectorize![complex: expensive_simulation(x), for x in &data];

// Adaptive profiling for unknown functions
let adaptive = vectorize![adaptive: mystery_function(x), for x in data];
```

### Zero-Copy Views
```rust
// Extensive view support for efficient operations
let view = matrix.view();           // Zero-copy view
let result = view ^ view;           // Operations on views

// Function design with views
fn process(data: ArrayView<f64>) -> VectorF64 {
    // No ownership taken, zero-copy operations
}
```

### Math-First Statistical Macros
```rust
// Natural mathematical syntax for reductions
let col_means = mean![data, axis=0].unwrap();
let row_stds = std![data, axis=1, keep=true].unwrap();
```

### The ^ Operator Advantage
```rust
// Clear distinction between mathematical and element-wise operations
let matrix_mult = &A ^ &B;     // Matrix multiplication
let hadamard = &A * &B;        // Element-wise multiplication
let dot = &u ^ &v;             // Dot product (returns scalar)
```

### Natural Slicing with Index Trait
```rust
// Direct indexing without method calls
let slice = &vector[1..4];     // Zero-copy slice
let tail = &vector[2..];       // From index 2
let head = &vector[..3];       // Up to index 3
```

## 🔀 Broadcasting Rules

| Rule | NumPy | RustLab | Example |
|------|-------|---------|---------|
| Scalar-Array | Automatic | Automatic | `&A + 1.0` |
| Vector-Matrix (row) | Shape (1, n) with (m, n) | Length n with (m, n) | `&matrix + &row_vector` |
| Vector-Matrix (col) | Shape (m, 1) with (m, n) | Length m with (m, n) | `&matrix + &col_vector` |
| Matrix-Matrix | Same shape or 1 | Same shape required | `&A + &B` |

## ⚡ Performance Considerations

| Aspect | NumPy | RustLab | Advantage |
|--------|-------|---------|-----------|
| SIMD | Automatic for some ops | Automatic for arrays > 64 elements | Both optimized |
| Parallelism | Limited (some functions) | Extensive (vectorize!, rayon) | RustLab |
| Memory safety | Runtime checks | Compile-time guarantees | RustLab |
| GC pauses | Yes (Python GC) | No (Rust ownership) | RustLab |
| Zero-copy ops | Views available | Extensive view support | RustLab |
| Type safety | Dynamic typing | Static typing | RustLab |

## 🎯 Common Patterns

### Data Normalization
```python
# NumPy
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
normalized = (data - mean) / std
```

```rust
// RustLab
let mean = data.mean_axis(Axis::Rows)?;
let std = data.std_axis(Axis::Rows)?;
let normalized = (&data - &mean) / &std;
```

### Matrix Operations
```python
# NumPy
C = A @ B.T + np.eye(n) * 0.1
```

```rust
// RustLab
let C = &A ^ &B.transpose() + &eye(n) * 0.1;
```

### Filtering Data
```python
# NumPy
mask = data > threshold
filtered = data[mask]
```

```rust
// RustLab
let mask = data.gt(threshold);
let filtered = data.where_mask(&mask)?;
```

## 📝 Key Migration Notes

1. **Always use references (`&`) in RustLab** to avoid moving values
2. **Use `^` for matrix multiplication**, not `*`
3. **File I/O is simpler in RustLab** - just `save()` and `load()`
4. **Error handling is explicit** - use `?` or `.unwrap()`
5. **Axis convention**: `Axis::Rows` (0) reduces along rows, `Axis::Cols` (1) along columns
6. **Type annotations often needed** in RustLab for clarity
7. **Views are more prominent** in RustLab for performance
8. **Parallelism is more accessible** via `vectorize!` macro

## 🚨 Common Pitfalls to Avoid

1. **Don't use `*` for matrix multiplication** - use `^`
2. **Don't forget `&` references** - prevents moves
3. **Don't mix types** - no automatic f32/f64 conversion
4. **Don't ignore Results** - handle with `?` or `.unwrap()`
5. **Don't use wrong axis** - remember Rows=0, Cols=1
6. **Don't create unnecessary copies** - use views when possible
