# RustLab 🧬

## The Future of Scientific Computing is AI-Human Partnership

The future of coding is an AI/human partnership with Agentic coding. RustLab is a 100% Rust numerical computing ecosystem designed from the ground up for this new paradigm.

### A Viable Path for the Big Data Era

RustLab provides a production-ready numerical computing platform for the big data era where:
- **Speed** is non-negotiable for processing massive datasets
- **Deterministic Performance** without garbage collection pauses ensures consistent computation
- **Memory Safety** becomes critical as data volumes and system complexity grow
- **Zero-Cost Abstractions** enable high-level code without runtime overhead

## 🎯 Vision

RustLab is **not intended to be a clone** of NumPy, or any other existing numerical library. Instead, it embraces Rust's unique strengths to create a distinctly Rust-native scientific computing ecosystem that:

- **Respects Rust's Idiomatic Syntax**: Leverages Rust's pervasive use of traits and method chaining for elegant, composable mathematical operations
- **Embraces Zero-Cost Abstractions**: High-level mathematical expressions compile to optimal machine code without runtime overhead
- **Prioritizes Memory Efficiency**: Extensive use of views and borrowing patterns for zero-copy operations wherever possible

### Core Capabilities
- **Blazing Performance**: Automatic SIMD and parallel computation via the faer library
- **Math-First Syntax**: Natural, script-like ergonomics with zero-cost abstractions
- **Simplified Type System**: Streamlined types focused on efficient numerical computing (`ArrayF64`, `VectorF64`, etc.)
- **Integrated Visualization**: Built-in 2D and 3D native rust plotters library for basic plotting capabilities to assist with visualization for code development.
- **Memory Safety**: Rust's unique approach to safe, concurrent programming
- **AI-Optimized Design**: Documentation and APIs that prevent common AI code generation errors
- **Accessible to Non-Experts**: Human-readable code that scientists can understand without Rust expertise
- **Python Integration**: High-level, ergonomic Rust libraries that complement Python via PyO3/Maturin framework - not replacing Python but enhancing it
- **Production Ready**: Deterministic performance without GC for mission-critical big data applications

## 🚀 Quick Start

```rust
use rustlab_math::{ArrayF64, VectorF64, array64, vec64, vectorize};

// Natural mathematical syntax - readable by any scientist
let A = array64![[1.0, 2.0], [3.0, 4.0]];
let v = vec64![1.0, 2.0];

// CRITICAL: Use ^ for matrix operations, * for element-wise
let result = &A ^ &v;  // Matrix-vector multiplication

// Script-like clarity for prototyping
let normalized = (&A - &A.mean()) / &A.std();  // Z-score normalization
let correlation = &A.transpose() ^ &A;         // Correlation matrix

// Intelligent auto-parallel list comprehensions
let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
let transformed = vectorize![x.sin() * x.cos(), for x in &data];  // Auto-parallel
let simulation = vectorize![complex: monte_carlo(x), for x in &seeds];  // Force parallel
```

## 📦 Ecosystem Components

| Crate | Purpose | Key Features |
|-------|---------|--------------|
| **rustlab-math** | Core mathematics | Matrices, vectors, I/O, slicing, broadcasting, auto-parallel list comprehensions |
| **rustlab-stats** | Statistical analysis | Descriptive stats, hypothesis testing, correlation |
| **rustlab-plotting** | Data visualization | 2D/3D plots, Jupyter integration, multiple backends |
| **rustlab-numerical** | Numerical methods | Integration, differentiation, interpolation |
| **rustlab-optimize** | Optimization | Curve fitting, minimization algorithms |
| **rustlab-special** | Special functions | Gamma, Bessel, error functions |
| **rustlab-linearalgebra** | Advanced linear algebra | Decompositions, eigenvalues, solvers |
| **rustlab-distributions** | Probability | Distributions, sampling, random number generation |
| **rustlab-linearregression** | Machine Learning | OLS, Ridge regression|

## 🔗 Essential Resources

- **Getting Started**: See `RUSTLAB_GETTING_STARTED_GUIDE.md` for complete setup instructions
- **AI Development**: Read `RUSTLAB_FOR_AI_GUIDE.md` for AI code generation guidelines
- **Jupyter Workflows**: Consult `RUST_NOTEBOOK_BEST_PRACTICES.md` for notebook development
- **Comprehensive Docs**: Each crate has its own `*_DOCUMENTATION.md` file
- **Numpy Migration**: Consult `RUSTLAB_NUMPY_CHEATSHEET.md`for comprehensive comparisons
- **Example Notebooks**: Every crate includes a `notebooks/` directory with examples

## 💡 Key Design Principles

### 1. Performance First (Big Data Ready)
- Automatic SIMD optimization for arrays > 64 elements
- **Zero-copy views** for maximum memory efficiency - operations on slices avoid unnecessary allocations
- **Intelligent auto-parallel list comprehensions** with cost-based decisions
- Parallel computation where beneficial - no overhead for small/fast operations
- No GC pauses - consistent performance at scale
- Predictable memory usage for production systems

### 2. Math-First Syntax
```rust
// Natural mathematical operations
let matrix_mult = &A ^ &B;      // Matrix multiplication
let element_wise = &A * &B;      // Element-wise multiplication
let dot_product = &u ^ &v;       // Dot product
```

### 3. Intelligent Parallel List Comprehensions
The fundamental parallel design includes NumPy/Julia-style list comprehensions with automatic parallelization:
```rust
// Complexity-aware auto-parallelism
let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
let squared = vectorize![x * x, for x in &data];                    // Auto-decides
let results = vectorize![complex: expensive_simulation(x), for x in &data];  // Force parallel
let coords = vectorize![serial: x + y, for x in &simple_data];      // Force serial

// Coordinate grids for mathematical surfaces
let (X, Y) = meshgrid!(x: x_range, y: y_range);
```
**Key Innovation**: Uses total computational cost (`complexity_factor × elements`) to make parallelization decisions:
- Simple math functions need 50,000+ elements to benefit from parallelism  
- Complex simulations parallelize with just 50+ elements
- Zero overhead when parallelism doesn't help

### 4. Safety Without Compromise
- Compile-time dimension checking where possible
- Memory safety guaranteed by Rust
- Clear error messages for debugging

### 5. AI-Human Partnership
- Documentation optimized to prevent AI hallucinations
- Consistent API patterns across all crates
- Extensive examples for learning

### 6. Human-Readable Code for Rapid Prototyping
- **Script-like syntax** that non-Rust experts can understand and modify
- **Natural mathematical notation** familiar to scientists and engineers
- **Minimal boilerplate** - focus on algorithms, not language complexity
- **Interactive development** via Jupyter notebooks for quick experimentation and prototyping
- Enables researchers to prototype ideas quickly without deep Rust expertise

## 📊 Installation

 git clone https://github.com/CameronReed-42/rustlab-rs.git

 cd to rustlab-rs
 cargo build --release

For your own new RustLab projects:

Add to your `Cargo.toml`:
```toml
[dependencies]
rustlab-math = { path = "path/to/rustlab-rs/rustlab-math" }
rustlab-stats = { path = "path/to/rustlab-rs/rustlab-stats" }
# Add other crates as needed
```

For Jupyter notebooks:
```bash
cargo install evcxr_jupyter
evcxr_jupyter --install
```

## 🎓 Learning Path

1. **Start**: Read the Getting Started Guide
2. **Learn**: Explore notebooks in `rustlab-math/notebooks/`
3. **Reference**: Study crate-specific documentation
4. **Build**: Create your own scientific computing workflows

## 🍴 Forking and Development

**Forking is encouraged!** RustLab is designed to be a foundation for further scientific computing development:

- **Fork freely** to explore new directions in numerical computing
- **Experiment** with different approaches to scientific algorithms
- **Build domain-specific** extensions for your field of research
- **Create specialized** versions optimized for your use cases
- **Share improvements** back to the community when beneficial

Whether you're building specialized tools for physics simulations, bioinformatics pipelines, financial modeling, or any other scientific domain - RustLab provides the foundation, and we encourage you to build upon it.


## 🤝 Contributing

RustLab embraces the AI-human partnership model. When contributing:
- Follow AI-friendly documentation patterns
- Maintain consistent API design
- Include comprehensive examples
- Test with both traditional and AI-assisted development

## 📄 License

RustLab is dual-licensed under either:

* **MIT License** - see [LICENSE.md](LICENSE.md) for details
* **Apache License, Version 2.0** - see [LICENSE.md](LICENSE.md) for details

This dual licensing provides maximum flexibility for both open-source and commercial usage.

---

**Welcome to the future of scientific computing in rust with RustLab!** 🚀

For detailed information, see `RUSTLAB_GETTING_STARTED_GUIDE.md`
