# RustLab Math Jupyter Notebooks

A comprehensive collection of Jupyter notebooks demonstrating RustLab Math features with ergonomic, math-first syntax.

## 📚 Notebook Organization

### Core Fundamentals
1. **[01_vectors_and_arrays_basics.ipynb](01_vectors_and_arrays_basics.ipynb)** - Vector and Array creation, basic operations
2. **[02_element_wise_operations.ipynb](02_element_wise_operations.ipynb)** - Element-wise arithmetic and mathematical functions
3. **[03_comparisons_and_filtering.ipynb](03_comparisons_and_filtering.ipynb)** - NEW VectorOps API for comparisons and data filtering

### Advanced Operations
4. **[04_linear_algebra.ipynb](04_linear_algebra.ipynb)** - Matrix operations, dot products, matrix multiplication
5. **[05_broadcasting.ipynb](05_broadcasting.ipynb)** - Broadcasting operations for vectors and arrays
6. **[06_reductions_and_statistics.ipynb](06_reductions_and_statistics.ipynb)** - Statistical operations, reductions, aggregations

### Data Manipulation
7. **[07_slicing_and_views.ipynb](07_slicing_and_views.ipynb)** - Slicing, views, and zero-copy operations
7b. **[07b_ergonomic_math_first_slicing.ipynb](07b_ergonomic_math_first_slicing.ipynb)** - 🆕 **Ergonomic NumPy-style slicing for ML workflows**
8. **[08_concatenation_and_reshaping.ipynb](08_concatenation_and_reshaping.ipynb)** - Joining, splitting, and reshaping data
9. **[09_functional_programming.ipynb](09_functional_programming.ipynb)** - Map, filter, reduce, and functional patterns

### Specialized Topics
10. **[10_complex_numbers.ipynb](10_complex_numbers.ipynb)** - Complex number operations
10b. **[10b_ergonomic_math_first_complex_numbers.ipynb](10b_ergonomic_math_first_complex_numbers.ipynb)** - 🆕 **Ergonomic complex arrays, vectorized operations, signal processing & quantum computing**
11. **[11_creation_utilities.ipynb](11_creation_utilities.ipynb)** - linspace, arange, eye, and other creation functions
12. **[12_real_world_examples.ipynb](12_real_world_examples.ipynb)** - Practical applications and complete examples

### Advanced Math-First Features
13. **[13_math_first_find_and_any.ipynb](13_math_first_find_and_any.ipynb)** - 🆕 Math-first find and any operations with boolean vector integration

## 🚀 Getting Started

### Prerequisites
```bash
# Install Jupyter and Evcxr Rust kernel
cargo install evcxr_jupyter
evcxr_jupyter --install

# Verify installation
jupyter kernelspec list
```

### Running the Notebooks
```bash
cd notebooks
jupyter notebook
```

Or use JupyterLab:
```bash
jupyter lab
```

## 📖 Learning Path

### For Beginners
Start with notebooks 1-3 to learn the basics:
1. Vectors and Arrays Basics
2. Element-wise Operations  
3. Comparisons and Filtering

### For Data Scientists
Focus on data manipulation and analysis:
- Notebook 3: Comparisons and Filtering (VectorOps API)
- Notebook 6: Reductions and Statistics
- Notebook 9: Functional Programming
- Notebook 12: Real-World Examples
- Notebook 13: Math-First Find and Any Operations (🆕)

### For Scientific Computing
Emphasize numerical operations:
- Notebook 4: Linear Algebra
- Notebook 5: Broadcasting
- Notebook 10: Complex Numbers
- Notebook 10b: Ergonomic Complex Numbers (🆕)

## 🎯 Key Features Demonstrated

- ✅ **Math-first ergonomic syntax** - Natural mathematical notation
- ✅ **NEW VectorOps API** - Clean comparison operations like `v.gt(3.0)`
- ✅ **🆕 Math-first find & any** - Boolean vector integration with `data.find_where(&mask)`, `data.any_gt(5.0)`
- ✅ **🆕 Ergonomic slicing** - NumPy-style slicing: `data.slice(1..5)`, `matrix.slice_rows(0..100)`, ML-ready patterns
- ✅ **🆕 Ergonomic complex numbers** - `cvec64![(1,2), (3,-1)]`, vectorized complex functions, signal processing & quantum computing
- ✅ **Creation macros** - `vec64!`, `array64!`, `matrix!`
- ✅ **Implicit typing** - Let Rust infer types where possible
- ✅ **Zero-copy views** - Efficient memory usage with slicing and views
- ✅ **SIMD operations** - Automatic vectorization
- ✅ **Broadcasting** - NumPy-style automatic broadcasting
- ✅ **Functional patterns** - Map, reduce, filter operations

## 💡 Tips for Using the Notebooks

1. **Memory Management**: Each notebook is self-contained. Restart kernel if needed.
2. **Type Inference**: We use implicit typing where possible for cleaner code.
3. **Ergonomic Macros**: Prefer `vec64!` over `VectorF64::from_slice()`.
4. **Math-First Syntax**: Use `&a + &b` for element-wise operations.
5. **Comparison Operations**: Use new `v.gt(threshold)` instead of old `v.gt_scalar(threshold)`.
6. **🆕 Find & Any Operations**: Use `data.any_gt(5.0)` and `data.find_where(&mask)` for math-first data filtering.

## 📝 Notebook Conventions

- **Markdown cells** explain concepts before code
- **Code cells** are kept small and focused
- **Output** is shown for all operations
- **Comments** explain non-obvious operations
- **Best practices** are highlighted throughout

## 🔧 Troubleshooting

If you encounter issues:
1. Restart the kernel: Kernel → Restart
2. Clear outputs: Cell → All Output → Clear
3. Check RustLab is in dependencies
4. Ensure you're using the latest notebook

## 📚 Additional Resources

- [RustLab Documentation](https://github.com/JulesArcher/rustlab-rs)
- [Comprehensive Cheat Sheet](../rustlab_math_comprehensive_cheatsheet.rs)
- [API Documentation](https://docs.rs/rustlab-math)