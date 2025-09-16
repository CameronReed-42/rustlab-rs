# RustLab-Numerical Notebooks

This folder contains Jupyter notebooks showcasing the numerical methods available in the rustlab-numerical crate. Each notebook demonstrates specific functionality with practical examples and visualizations.

## Prerequisites

Before running these notebooks, ensure you have:
- Jupyter with evcxr kernel installed
- RustLab ecosystem crates available
- rustlab-plotting for visualizations

## Notebook Overview

### 1. `01_interpolation_methods.ipynb`
**Interpolation Methods Showcase**
- Linear interpolation for quick approximations
- Polynomial interpolation (Lagrange and Newton forms)
- Cubic splines with different boundary conditions
- 2D interpolation (bilinear and bicubic)
- Performance comparisons and use case recommendations
- Error analysis and convergence studies

### 2. `02_numerical_integration.ipynb`
**Numerical Integration Techniques**
- Trapezoidal rule for basic integration
- Simpson's rule for higher accuracy
- Romberg integration with adaptive refinement
- Error estimation and convergence analysis
- Integration of oscillatory and singular functions
- Performance benchmarks for different methods

### 3. `03_differentiation_methods.ipynb`
**Numerical Differentiation**
- Forward, backward, and central differences
- Higher-order finite difference schemes
- Richardson extrapolation for improved accuracy
- Complex-step differentiation for machine precision
- Gradient computation for vector functions
- Automatic step size selection

### 4. `04_root_finding.ipynb`
**Root Finding Algorithms**
- Bisection method for guaranteed convergence
- Newton-Raphson for fast quadratic convergence
- Secant method when derivatives are unavailable
- Brent's method combining reliability and speed
- Ridders' method with automatic bracketing
- Multiple root detection strategies

### 5. `05_advanced_applications.ipynb`
**Advanced Applications and Integration**
- Solving differential equations with numerical methods
- Parameter estimation using root finding
- Numerical optimization with gradient methods
- Signal processing with interpolation
- Data smoothing with splines
- Integration with other RustLab crates

### 6. `06_performance_optimization.ipynb`
**Performance and Best Practices**
- Choosing the right algorithm for your problem
- Memory-efficient implementations
- Parallel computation with Rayon
- Error handling and numerical stability
- Benchmarking different approaches
- Common pitfalls and how to avoid them

## Running the Notebooks

1. Start Jupyter with the evcxr kernel:
   ```bash
   jupyter notebook
   ```

2. Open any notebook and run cells sequentially

3. Each notebook is self-contained with all necessary imports and setup

## Key Learning Objectives

- **Understand** the mathematical foundations of numerical methods
- **Apply** appropriate algorithms for specific problems
- **Analyze** error characteristics and convergence behavior
- **Optimize** performance for large-scale computations
- **Integrate** numerical methods with the broader RustLab ecosystem

## Additional Resources

- [RustLab-Numerical Documentation](../AI_DOCUMENTATION.md)
- [Examples Directory](../examples/)
- [RustLab-Math Integration Guide](../../rustlab-math/AI_DOCUMENTATION.md)

## Contributing

If you'd like to add more notebooks or improve existing ones:
1. Follow the established notebook structure
2. Include clear explanations and visualizations
3. Add performance comparisons where relevant
4. Ensure all code follows Rust best practices
5. Test notebooks thoroughly before submitting

## License

These notebooks are part of the RustLab project and follow the same licensing terms.