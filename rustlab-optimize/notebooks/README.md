# RustLab-Optimize Jupyter Notebooks

This folder contains a comprehensive set of Jupyter notebooks demonstrating the capabilities of the `rustlab-optimize` crate. Each notebook is accompanied by standalone Rust files (`.rs`) that can be tested independently before being integrated into the notebooks.

## 📚 Notebook Overview

### **01_getting_started.ipynb** - Introduction to Optimization
- Basic optimization concepts and terminology
- Overview of the rustlab-optimize API
- Simple 1D, 2D, and N-D optimization examples
- Understanding optimization results and convergence
- **Supporting file**: `01_getting_started.rs`

### **02_curve_fitting_fundamentals.ipynb** - Data Fitting and Regression
- Linear regression with least squares
- Exponential decay fitting (radioactive decay, pharmacokinetics)
- Polynomial fitting for calibration curves
- Sinusoidal fitting for periodic data
- Model evaluation metrics (R², residuals, confidence intervals)
- **Supporting file**: `02_curve_fitting.rs`

### **03_parameter_constraints.ipynb** - Bounds and Fixed Parameters
- Setting parameter bounds (box constraints)
- Fixing parameters during optimization
- Parameter transformation techniques
- Mixed constraints (bounds + fixed parameters)
- Real-world examples: enzyme kinetics with known constants
- **Supporting file**: `03_constraints.rs`

### **04_algorithm_selection.ipynb** - Choosing the Right Algorithm
- Algorithm comparison: BFGS vs Levenberg-Marquardt
- When to use each algorithm
- Performance benchmarks
- Convergence behavior analysis
- Direct algorithm specification vs intelligent selection
- **Supporting file**: `04_algorithms.rs`

### **05_scientific_applications.ipynb** - Real-World Use Cases
- **Pharmacokinetics**: Drug concentration modeling
- **Enzyme Kinetics**: Michaelis-Menten parameter estimation
- **Signal Processing**: Peak fitting in spectroscopy
- **Chemical Kinetics**: Reaction rate determination
- **Growth Models**: Bacterial growth curve analysis
- **Supporting file**: `05_applications.rs`

### **06_advanced_optimization.ipynb** - Advanced Techniques
- Multi-objective optimization
- Global vs local optimization strategies
- Handling noisy data
- Robust fitting with outlier detection
- Custom objective functions and models
- **Supporting file**: `06_advanced.rs`

### **07_performance_optimization.ipynb** - Speed and Efficiency
- Profiling optimization algorithms
- Memory usage analysis
- Parallelization strategies
- Large-scale optimization problems
- Comparison with other optimization libraries
- **Supporting file**: `07_performance.rs`

### **08_visualization.ipynb** - Visualizing Optimization
- Plotting objective function landscapes
- Convergence trajectories
- Parameter uncertainty visualization
- Residual analysis plots
- Interactive optimization exploration
- **Supporting file**: `08_visualization.rs`

## 🔧 Supporting Rust Files

Each notebook has a corresponding `.rs` file that contains:
- Standalone examples that can be compiled and run
- Helper functions used in the notebooks
- Test cases to verify functionality
- Performance benchmarks

### File Structure:
```
notebooks/
├── README.md                          # This file
├── 01_getting_started.ipynb         # Notebook 1
├── 01_getting_started.rs            # Supporting code for Notebook 1
├── 02_curve_fitting_fundamentals.ipynb
├── 02_curve_fitting.rs
├── 03_parameter_constraints.ipynb
├── 03_constraints.rs
├── 04_algorithm_selection.ipynb
├── 04_algorithms.rs
├── 05_scientific_applications.ipynb
├── 05_applications.rs
├── 06_advanced_optimization.ipynb
├── 06_advanced.rs
├── 07_performance_optimization.ipynb
├── 07_performance.rs
├── 08_visualization.ipynb
├── 08_visualization.rs
└── utils.rs                          # Shared utilities for all notebooks
```

## 🚀 Getting Started

### Prerequisites
1. **Jupyter with Rust kernel**: Install using evcxr_jupyter
   ```bash
   cargo install evcxr_jupyter
   evcxr_jupyter --install
   ```

2. **Required dependencies**: All notebooks use:
   - `rustlab-optimize` (main optimization crate)
   - `rustlab-math` (vectors and matrices)
   - `rustlab-plotting` (visualization)
   - `rustlab-stats` (statistics functions)

### Running the Examples

#### In Jupyter:
```bash
jupyter notebook 01_getting_started.ipynb
```

#### Standalone Rust files:
```bash
# Test individual example files
rustc --edition 2021 01_getting_started.rs -L ../target/debug/deps
./01_getting_started

# Or use cargo to run
cargo build --examples
cargo run --example notebook_01_getting_started
```

## 📊 Example Code Snippets

### Simple Optimization
```rust
use rustlab_optimize::prelude::*;

// 1D optimization
let result = minimize_1d(|x| (x - 2.0).powi(2))
    .solve()?;

// 2D optimization with bounds
let result = minimize_2d(|x, y| x*x + y*y)
    .from(1.0, 1.0)
    .bounds((-10.0, 10.0), (-10.0, 10.0))
    .solve()?;
```

### Curve Fitting
```rust
use rustlab_optimize::fit::*;
use rustlab_math::prelude::*;

// Exponential decay fitting
let t = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let y = vec![10.0, 6.1, 3.7, 2.2, 1.4];

let fit = fit_exponential(&t.into(), &y.into())?;
println!("A = {:.2}, k = {:.3}", fit.amplitude, fit.decay_rate);
```

### Parameter Constraints
```rust
// Fix amplitude, optimize decay rate
let fit = fit_exponential_advanced(&x, &y)
    .fix_amplitude(10.0)
    .bounds(0.0, 2.0)  // Decay rate between 0 and 2
    .solve()?;
```

## 📈 Learning Path

1. **Start with Notebook 01** for basic concepts
2. **Move to Notebook 02** if you need curve fitting
3. **Explore Notebook 03** for constrained problems
4. **Use Notebook 04** to understand algorithm selection
5. **Apply to real problems** with Notebook 05
6. **Dive deeper** with Notebooks 06-08 for advanced topics

## 🔬 Scientific Computing Focus

These notebooks emphasize practical scientific computing applications:
- **Data Analysis**: Fitting experimental data
- **Parameter Estimation**: Extracting physical parameters from measurements
- **Model Validation**: Comparing theoretical models with observations
- **Uncertainty Quantification**: Understanding parameter confidence
- **Reproducible Research**: Clear, documented optimization workflows

## 📝 Contributing

To add a new notebook:
1. Create the notebook file: `XX_topic_name.ipynb`
2. Create supporting Rust code: `XX_topic_name.rs`
3. Test the Rust code independently
4. Add documentation to this README
5. Include example outputs and visualizations

## 🔗 Additional Resources

- [RustLab-Optimize Documentation](../src/lib.rs)
- [API Reference](https://docs.rs/rustlab-optimize)
- [Examples Directory](../examples/)
- [Implementation Checklist](../../RUSTLAB_OPTIMIZE_IMPLEMENTATION_CHECKLIST.md)

## 📊 Notebook Status

| Notebook | Status | Last Updated | Key Features |
|----------|--------|--------------|--------------|
| 01_getting_started | 🔴 Not Started | - | Basic optimization |
| 02_curve_fitting | 🔴 Not Started | - | Regression, fitting |
| 03_constraints | 🔴 Not Started | - | Bounds, fixed params |
| 04_algorithms | 🔴 Not Started | - | Algorithm comparison |
| 05_applications | 🔴 Not Started | - | Scientific use cases |
| 06_advanced | 🔴 Not Started | - | Advanced techniques |
| 07_performance | 🔴 Not Started | - | Speed optimization |
| 08_visualization | 🔴 Not Started | - | Plotting, analysis |

## 🎯 Goals

These notebooks aim to:
- **Demonstrate** the clean, math-first API of rustlab-optimize
- **Educate** users on optimization concepts and best practices
- **Showcase** real-world scientific applications
- **Compare** different algorithms and approaches
- **Provide** ready-to-use code templates
- **Enable** reproducible scientific computing workflows

---

*These notebooks are part of the RustLab scientific computing ecosystem, providing production-ready optimization tools with a focus on ergonomics and performance.*