//! RustLab Plotting - Scientific visualization with mathematical notation for AI code generation
//!
//! Provides high-performance scientific plotting capabilities integrated with rustlab-math
//! for seamless data visualization. Designed for data analysis, machine learning, and
//! scientific computing with MATLAB/Python-like plotting interface.
//!
//! # For AI Code Generation
//!
//! This crate provides:
//! - **Concise plotting**: `plot(x, y)`, `scatter(x, y)`, `bar(x, y)`, `histogram(data, bins)`
//! - **Mathematical integration**: Direct plotting from rustlab-math vectors and arrays
//! - **Builder pattern**: `Plot::new().line(x, y).scatter(x2, y2).show()`
//! - **Subplot support**: Multi-panel layouts with `subplots(rows, cols)`
//! - **Error handling**: All operations return `Result<()>` for robustness
//!
//! # Quick Start Guide
//!
//! ```rust
//! use rustlab_plotting::*;
//! use rustlab_math::{vec64, range};
//!
//! // Simple plotting
//! let x = range!(0.0 => 10.0, 100);
//! let y = x.sin();
//! plot(&x, &y)?;  // Creates and displays line plot
//!
//! // Multiple series
//! let y1 = x.sin();
//! let y2 = x.cos();
//! Plot::new()
//!     .line(&x, &y1)
//!     .line(&x, &y2)
//!     .title("Trigonometric Functions")
//!     .xlabel("x")
//!     .ylabel("y")
//!     .show()?;
//!
//! // Statistical plots
//! let data = vec64![1.0, 2.0, 1.5, 3.0, 2.5, 2.0, 1.8];
//! histogram(&data, 10)?;  // 10 bins
//!
//! // Subplots
//! Plot::new()
//!     .subplots(2, 1)
//!     .subplot(0, 0)
//!         .line(&x, &y1)
//!         .title("sin(x)")
//!         .build()
//!     .subplot(1, 0)
//!         .line(&x, &y2)
//!         .title("cos(x)")
//!         .build()
//!     .show()?;
//! ```
//!
//! # Chart Types
//!
//! | Function | Chart Type | Use Case | Data Requirements |
//! |----------|------------|----------|-------------------|
//! | `plot(x, y)` | Line plot | Time series, continuous data | Equal length vectors |
//! | `scatter(x, y)` | Scatter plot | Correlation analysis | Equal length vectors |
//! | `bar(x, y)` | Bar chart | Categorical data | Equal length vectors |
//! | `histogram(data, bins)` | Histogram | Distribution analysis | Single vector + bin count |
//!
//! # Builder Pattern API
//!
//! ```rust
//! // Fluent interface for complex plots
//! Plot::new()
//!     .size(800, 600)                    // Figure size
//!     .title("My Plot")
//!     .xlabel("X Axis")
//!     .ylabel("Y Axis")
//!     .line(&x1, &y1)                    // Add line series
//!     .scatter(&x2, &y2)                 // Add scatter series
//!     .color("red")                       // Set color for next series
//!     .line(&x3, &y3)
//!     .legend(true)                       // Show legend
//!     .grid(true)                         // Show grid
//!     .show()?;                           // Display plot
//! ```
//!
//! # Backend Support
//!
//! - **Native**: Desktop applications using native GUI
//! - **Jupyter**: Inline plotting for Jupyter notebooks
//! - **PNG/SVG**: File export for reports and publications
//!
//! # Mathematical Integration
//!
//! ```rust
//! use rustlab_math::*;
//! use rustlab_plotting::*;
//!
//! // Direct plotting from math operations
//! let x = linspace(0.0, 2.0 * PI, 1000);
//! let y = x.sin().add_scalar(0.1 * &x.cos());
//!
//! // Plot mathematical expressions
//! x.plot_expression(|x| x.sin() + 0.1 * x.cos())?;
//!
//! // Array plotting
//! let matrix = zeros(10, 10);
//! matrix.plot_heatmap()?;
//! ```
//!
//! # Error Handling
//!
//! ```rust
//! match plot(&x, &y) {
//!     Ok(()) => println!("Plot displayed successfully"),
//!     Err(PlottingError::DimensionMismatch { .. }) => {
//!         eprintln!("Data vectors must have same length");
//!     },
//!     Err(PlottingError::BackendError { .. }) => {
//!         eprintln!("Display backend not available");
//!     },
//!     Err(e) => eprintln!("Plotting error: {}", e),
//! }
//! ```

pub mod error;
pub mod plot;
pub mod backend;
pub mod charts;
pub mod style;
pub mod prelude;
pub mod jupyter;
pub mod math_integration;
pub mod colormap;
pub mod heatmap;
pub mod contour;
pub mod surface;

// Re-export core types and functions for convenient usage
pub use error::{PlottingError, Result};
pub use plot::*;  // This includes Plot, PlotBuilder, SubplotBuilder from builder.rs and all types from types.rs
pub use style::theme::Theme;
pub use colormap::{ColorMap, ColorMapper};
pub use heatmap::{HeatmapBuilder, heatmap, correlation_heatmap};
pub use contour::{ContourBuilder, contour, contour_levels, contourf};
pub use surface::{Surface3DBuilder, surface3d, wireframe3d, surface3d_colormap, meshgrid, array2d_from_function};

// Convenience functions - these should be at the end of the file
pub use math_integration::{
    PlotExpression, PlotMath, VectorPlotExt, ArrayPlotExt, MathPlotBuilder,
    VectorDataExt, patterns
};

use rustlab_math::{VectorF64, ArrayF64};

// ================================================================================================
// Convenience Functions - MATLAB/Python-style plotting for AI code generation
// ================================================================================================

/// Create and display a line plot with mathematical data vectors
/// 
/// # Mathematical Specification
/// Plots y = f(x) as connected line segments between points (xᵢ, yᵢ).
/// 
/// # Parameters
/// - `x`: Independent variable vector (typically time, space, frequency)
/// - `y`: Dependent variable vector (measurements, function values)
/// 
/// # Requirements
/// - `x.len() == y.len()` (equal length vectors)
/// - Both vectors must contain finite values (no NaN/Inf)
/// 
/// # For AI Code Generation
/// - Use for continuous data visualization (time series, function plots)
/// - Equivalent to MATLAB's `plot(x, y)` or Python's `plt.plot(x, y)`
/// - Automatically handles axis scaling and display
/// - Returns Result<()> for error handling
/// 
/// # Example
/// ```rust
/// use rustlab_plotting::plot;
/// use rustlab_math::{range, VectorMathOps};
/// 
/// // Plot sine function
/// let x = range!(0.0 => 2.0 * PI, 100);
/// let y = x.sin();
/// plot(&x, &y)?;  // Displays line plot
/// 
/// // Plot experimental data
/// let time = vec64![0.0, 1.0, 2.0, 3.0, 4.0];
/// let temperature = vec64![20.0, 22.5, 21.8, 23.1, 22.9];
/// plot(&time, &temperature)?;
/// ```
/// 
/// # Common Use Cases
/// - Function visualization: y = f(x)
/// - Time series data: sensor readings over time
/// - Experimental results: measurement vs parameter
/// - Signal processing: amplitude vs time/frequency
pub fn plot(x: &VectorF64, y: &VectorF64) -> Result<()> {
    Plot::new().line(x, y).show()
}

/// Create and display a scatter plot for correlation analysis
/// 
/// # Mathematical Specification
/// Plots individual points (xᵢ, yᵢ) without connecting lines.
/// Useful for visualizing relationships between two variables.
/// 
/// # Parameters
/// - `x`: Independent variable vector
/// - `y`: Dependent variable vector
/// 
/// # Requirements
/// - `x.len() == y.len()` (equal length vectors)
/// - Both vectors must contain finite values
/// 
/// # For AI Code Generation
/// - Use for discrete data points and correlation analysis
/// - Equivalent to MATLAB's `scatter(x, y)` or Python's `plt.scatter(x, y)`
/// - Shows individual data points without trend lines
/// - Ideal for outlier detection and pattern recognition
/// 
/// # Example
/// ```rust
/// use rustlab_plotting::scatter;
/// use rustlab_math::vec64;
/// 
/// // Scatter plot of correlated data
/// let x = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec64![2.1, 3.9, 6.2, 8.1, 9.8];  // y ≈ 2x with noise
/// scatter(&x, &y)?;
/// 
/// // Multi-dimensional data projection
/// let feature1 = vec64![1.2, 2.3, 1.8, 3.1, 2.7];
/// let feature2 = vec64![0.8, 1.9, 1.2, 2.5, 2.1];
/// scatter(&feature1, &feature2)?;
/// ```
/// 
/// # Common Use Cases
/// - Correlation analysis between variables
/// - Outlier detection in datasets
/// - Feature visualization in machine learning
/// - Experimental data with discrete measurements
pub fn scatter(x: &VectorF64, y: &VectorF64) -> Result<()> {
    Plot::new().scatter(x, y).show()
}

/// Create and display a bar chart for categorical data visualization
/// 
/// # Mathematical Specification
/// Creates vertical bars with heights yᵢ at positions xᵢ.
/// Bar width automatically calculated based on x-spacing.
/// 
/// # Parameters
/// - `x`: Category positions (can be indices or category values)
/// - `y`: Bar heights (typically counts, frequencies, or measurements)
/// 
/// # Requirements
/// - `x.len() == y.len()` (equal length vectors)
/// - `y` values should be non-negative for standard bar charts
/// 
/// # For AI Code Generation
/// - Use for categorical data and frequency distributions
/// - Equivalent to MATLAB's `bar(x, y)` or Python's `plt.bar(x, y)`
/// - Automatically handles bar positioning and spacing
/// - Good for comparing discrete categories
/// 
/// # Example
/// ```rust
/// use rustlab_plotting::bar;
/// use rustlab_math::vec64;
/// 
/// // Category comparison
/// let categories = vec64![1.0, 2.0, 3.0, 4.0];  // Category indices
/// let values = vec64![23.0, 45.0, 56.0, 78.0];   // Category values
/// bar(&categories, &values)?;
/// 
/// // Frequency distribution
/// let bins = vec64![0.0, 1.0, 2.0, 3.0, 4.0];
/// let counts = vec64![12.0, 23.0, 34.0, 19.0, 8.0];
/// bar(&bins, &counts)?;
/// ```
/// 
/// # Common Use Cases
/// - Frequency distributions and histograms
/// - Category comparisons (sales by region, scores by group)
/// - Survey results and polling data
/// - Performance metrics across different conditions
pub fn bar(x: &VectorF64, y: &VectorF64) -> Result<()> {
    Plot::new().bar(x, y).show()
}

/// Create and display a histogram for data distribution analysis
/// 
/// # Mathematical Specification
/// Divides data range into `bins` equal-width intervals and counts
/// frequency of data points in each bin: fᵢ = |{x ∈ data : bᵢ ≤ x < bᵢ₊₁}|
/// 
/// # Parameters
/// - `data`: Vector of numerical observations
/// - `bins`: Number of equal-width bins to create
/// 
/// # Algorithm
/// 1. Compute range: [min(data), max(data)]
/// 2. Create bins with width: Δb = (max - min) / bins
/// 3. Count data points in each bin
/// 4. Display as bar chart with bin centers vs counts
/// 
/// # For AI Code Generation
/// - Use for understanding data distributions and statistical properties
/// - Equivalent to MATLAB's `histogram(data, bins)` or Python's `plt.hist(data, bins)`
/// - Automatically computes bin edges and frequencies
/// - Essential for exploratory data analysis
/// 
/// # Example
/// ```rust
/// use rustlab_plotting::histogram;
/// use rustlab_math::{vec64, VectorMathOps};
/// 
/// // Normal distribution visualization
/// let data = vec64![1.2, 1.8, 2.1, 1.9, 2.3, 1.7, 2.0, 1.6, 2.2, 1.9];
/// histogram(&data, 10)?;  // 10 bins
/// 
/// // Large dataset analysis
/// let measurements = range!(0.0 => 100.0, 1000).map(|x| x.sin() + random_noise());
/// histogram(&measurements, 50)?;  // 50 bins for detailed view
/// ```
/// 
/// # Statistical Applications
/// - Distribution shape analysis (normal, skewed, bimodal)
/// - Outlier detection (isolated bins)
/// - Data quality assessment
/// - Feature engineering for machine learning
/// 
/// # Bin Selection Guidelines
/// - Small datasets (n < 100): bins = 5-15
/// - Medium datasets (100 ≤ n < 1000): bins = 15-50
/// - Large datasets (n ≥ 1000): bins = 50-100
/// - Rule of thumb: bins ≈ √n or bins ≈ 2n^(1/3)
pub fn histogram(data: &VectorF64, bins: usize) -> Result<()> {
    Plot::new().histogram(data, bins).show()
}

// ================================================================================================
// Multiple Series Functions - Comparative visualization
// ================================================================================================

/// Plot two data series on the same axes for comparison
/// 
/// # Mathematical Specification
/// Displays y₁ = f₁(x) and y₂ = f₂(x) on the same coordinate system
/// with automatic color differentiation.
/// 
/// # Parameters
/// - `x`: Shared independent variable vector
/// - `y1`: First dependent variable series
/// - `y2`: Second dependent variable series
/// 
/// # Requirements
/// - `x.len() == y1.len() == y2.len()`
/// - All vectors must contain finite values
/// 
/// # For AI Code Generation
/// - Use for comparing two related functions or datasets
/// - Automatically assigns different colors to each series
/// - More convenient than creating Plot with multiple line() calls
/// - Legend automatically generated
/// 
/// # Example
/// ```rust
/// use rustlab_plotting::plot2;
/// use rustlab_math::{range, VectorMathOps};
/// 
/// // Compare trigonometric functions
/// let x = range!(0.0 => 2.0 * PI, 100);
/// let sin_x = x.sin();
/// let cos_x = x.cos();
/// plot2(&x, &sin_x, &cos_x)?;
/// 
/// // Compare experimental vs theoretical
/// let time = range!(0.0 => 10.0, 100);
/// let experimental = time.map(|t| t.sin() + 0.1 * random_noise());
/// let theoretical = time.sin();
/// plot2(&time, &experimental, &theoretical)?;
/// ```
/// 
/// # Common Use Cases
/// - Before/after comparisons
/// - Experimental vs theoretical data
/// - Different model predictions
/// - Signal and its filtered version
pub fn plot2(x: &VectorF64, y1: &VectorF64, y2: &VectorF64) -> Result<()> {
    Plot::new()
        .line(x, y1)
        .line(x, y2)
        .show()
}

/// Plot multiple data series with shared x-axis for multi-variable analysis
/// 
/// # Mathematical Specification
/// Displays y₁ = f₁(x), y₂ = f₂(x), ..., yₙ = fₙ(x) on the same axes
/// with automatic color cycling and legend generation.
/// 
/// # Parameters
/// - `x`: Shared independent variable vector
/// - `ys`: Slice of dependent variable vectors
/// 
/// # Requirements
/// - `x.len() == ys[i].len()` for all i
/// - All vectors must contain finite values
/// - `ys.len() > 0` (at least one series)
/// 
/// # For AI Code Generation
/// - Use for multi-variable analysis and comparison
/// - Automatically cycles through colors and line styles
/// - More efficient than multiple plot() calls
/// - Handles variable number of series dynamically
/// 
/// # Example
/// ```rust
/// use rustlab_plotting::plot_many;
/// use rustlab_math::{range, VectorMathOps};
/// 
/// // Plot multiple harmonics
/// let x = range!(0.0 => 2.0 * PI, 100);
/// let fundamental = x.sin();
/// let second_harmonic = (2.0 * &x).sin();
/// let third_harmonic = (3.0 * &x).sin();
/// 
/// plot_many(&x, &[&fundamental, &second_harmonic, &third_harmonic])?;
/// 
/// // Compare multiple models
/// let models = vec![model1(&x), model2(&x), model3(&x)];
/// let model_refs: Vec<&VectorF64> = models.iter().collect();
/// plot_many(&x, &model_refs)?;
/// ```
/// 
/// # Performance Considerations
/// - More efficient than multiple individual plots
/// - Single legend and axis computation
/// - Shared memory layout for rendering
/// 
/// # Common Use Cases
/// - Multi-model comparison in machine learning
/// - Parameter sensitivity analysis
/// - Time series ensemble visualization
/// - Harmonic analysis and Fourier components
pub fn plot_many(x: &VectorF64, ys: &[&VectorF64]) -> Result<()> {
    let mut plot = Plot::new();
    for y in ys {
        plot = plot.line(x, y);
    }
    plot.show()
}

/// Create and display a contour plot for 2D scalar field visualization
/// 
/// # Mathematical Specification
/// Displays level curves (contours) of a 2D scalar function z = f(x, y).
/// Contour lines connect points of equal function value: f(x, y) = constant.
/// 
/// # Parameters
/// * `x_grid` - X-axis grid points (1D vector, must be sorted)
/// * `y_grid` - Y-axis grid points (1D vector, must be sorted)
/// * `z_data` - Z values on the 2D grid (2D array: rows=y, cols=x)
/// 
/// # Grid Requirements
/// * `z_data.nrows() == y_grid.len()`
/// * `z_data.ncols() == x_grid.len()`
/// * Grid points must be in ascending order
/// * Z values must be finite (no NaN or infinity)
/// 
/// # For AI Code Generation
/// * Use for optimization landscapes and objective functions
/// * Essential for visualizing 2D functions and their level sets
/// * Equivalent to MATLAB's `contour(X, Y, Z)` or Python's `plt.contour(X, Y, Z)`
/// * Automatically generates contour levels based on data range
/// * Perfect for numerical analysis and mathematical visualization
/// 
/// # Example
/// ```rust
/// use rustlab_plotting::{contour_plot, array2d_from_function};
/// use rustlab_math::range;
/// 
/// // Create grid for plotting
/// let x = range!(-2.0 => 2.0, 50);
/// let y = range!(-2.0 => 2.0, 50);
/// 
/// // Evaluate function z = x² - y² (saddle function)
/// let z = array2d_from_function(&x, &y, |x, y| x*x - y*y);
/// 
/// // Create contour plot
/// contour_plot(&x, &y, &z)?;
/// 
/// // Optimization landscape example
/// let objective = array2d_from_function(&x, &y, |x, y| {
///     (x - 1.0).powi(2) + (y + 0.5).powi(2)  // Minimum at (1, -0.5)
/// });
/// contour_plot(&x, &y, &objective)?;
/// ```
/// 
/// # Mathematical Applications
/// * **Optimization**: Visualizing objective function landscapes
/// * **Root finding**: Showing zero-level contours
/// * **Gradient analysis**: Understanding function behavior
/// * **Level set methods**: Tracking curve evolution
/// * **Potential fields**: Electric, gravitational, magnetic fields
/// * **Topography**: Elevation maps and terrain analysis
/// 
/// # Contour Interpretation
/// * **Closely spaced contours**: Steep gradient (rapid change)
/// * **Widely spaced contours**: Gentle gradient (slow change)
/// * **Circular contours**: Local minimum/maximum
/// * **Saddle patterns**: Optimization saddle points
/// * **Parallel lines**: Constant gradient direction
pub fn contour_plot(x_grid: &VectorF64, y_grid: &VectorF64, z_data: &ArrayF64) -> Result<()> {
    crate::contour::contour(x_grid, y_grid, z_data)
}

// ================================================================================================
// Subplot Functions - Multi-panel layouts for comprehensive analysis
// ================================================================================================

/// Demonstration of subplot functionality with mathematical functions
/// 
/// # Mathematical Specification
/// Creates 2×2 subplot grid displaying:
/// - (0,0): sin(x) for x ∈ [0, 10]
/// - (0,1): cos(x) for x ∈ [0, 10]
/// - (1,0): sin(x/2) for x ∈ [0, 5]
/// - (1,1): cos(x/2) for x ∈ [0, 5]
/// 
/// # For AI Code Generation
/// - Demonstrates subplot creation and layout patterns
/// - Shows integration with rustlab-math function evaluation
/// - Template for multi-panel scientific plots
/// - Use as reference for subplot API usage
/// 
/// # Subplot Layout
/// ```
/// +----------+----------+
/// | sin(x)   | cos(x)   |
/// +----------+----------+
/// | sin(x/2) | cos(x/2) |
/// +----------+----------+
/// ```
/// 
/// # Example Usage
/// ```rust
/// use rustlab_plotting::subplot_demo;
/// 
/// // Display demo with trigonometric functions
/// subplot_demo()?;
/// 
/// // Create custom subplot layout
/// Plot::new()
///     .subplots(2, 3)  // 2 rows, 3 columns
///     .subplot(0, 0).line(&x1, &y1).build()
///     .subplot(0, 1).scatter(&x2, &y2).build()
///     .subplot(1, 0).bar(&x3, &y3).build()
///     .show()?;
/// ```
/// 
/// # Applications
/// - Multi-variable data exploration
/// - Before/after analysis panels
/// - Different chart types for same dataset
/// - Parameter sweep visualization
pub fn subplot_demo() -> Result<()> {
    use rustlab_math::{vec64, range};
    
    // Create x-axis vectors using rustlab-math's range macro
    let x = range!(0.0 => 10.0, 100);          // Full range: [0, 10]
    let x_half = range!(0.0 => 5.0, 100);      // Half range: [0, 5]
    
    // Compute mathematical functions using built-in vector operations
    let y1 = x.sin();           // sin(x)
    let y2 = x.cos();           // cos(x)
    let y3 = x_half.sin();      // sin(x) for x ∈ [0, 5]
    let y4 = x_half.cos();      // cos(x) for x ∈ [0, 5]

    // Create 2×2 subplot layout with mathematical function visualization
    Plot::new()
        .subplots(2, 2)                         // 2 rows × 2 columns grid
        .size(1200, 800)                        // Figure size: 1200×800 pixels
        .title("Subplot Demo: Trigonometric Functions")
        
        // Top-left: sin(x) full range
        .subplot(0, 0)
            .title("sin(x), x ∈ [0, 10]")
            .xlabel("x")
            .ylabel("sin(x)")
            .line(&x, &y1)
            .grid(true)
            .build()
            
        // Top-right: cos(x) full range
        .subplot(0, 1)
            .title("cos(x), x ∈ [0, 10]")
            .xlabel("x")
            .ylabel("cos(x)")
            .line(&x, &y2)
            .grid(true)
            .build()
            
        // Bottom-left: sin(x) half range
        .subplot(1, 0)
            .title("sin(x), x ∈ [0, 5]")
            .xlabel("x")
            .ylabel("sin(x)")
            .line(&x_half, &y3)
            .grid(true)
            .build()
            
        // Bottom-right: cos(x) half range
        .subplot(1, 1)
            .title("cos(x), x ∈ [0, 5]")
            .xlabel("x")
            .ylabel("cos(x)")
            .line(&x_half, &y4)
            .grid(true)
            .build()
            
        .show()  // Display the complete subplot layout
}

/// Demonstration of logarithmic scale plotting for multi-order magnitude data
/// 
/// # Mathematical Specification
/// Creates 2×2 subplot grid demonstrating logarithmic scaling effects:
/// - **Log-log plots**: Reveal power law relationships y ∝ xᵅ as linear
/// - **Semi-log plots**: Show exponential relationships y ∝ eᵃˣ as linear
/// - **Scale transformations**: f(x) = log₁₀(x), enhancing pattern visibility
/// 
/// # For AI Code Generation
/// - Essential for scientific data spanning orders of magnitude
/// - Use when data ranges from 10⁻ⁿ to 10⁺ᵐ (earthquake magnitudes, frequencies)
/// - Power law detection: linear relationship in log-log indicates y = Axᵅ
/// - Exponential growth analysis: linear in semi-log indicates y = Aeᵇˣ
/// 
/// # Scale Selection Guidelines
/// | Data Pattern | X-Scale | Y-Scale | Interpretation |
/// |--------------|---------|---------|----------------|
/// | Power law y=xᵅ | Log₁₀ | Log₁₀ | Linear with slope α |
/// | Exponential y=eᵃˣ | Linear | Log₁₀ | Linear with slope a |
/// | Time series | Linear | Log₁₀ | Growth rate analysis |
/// | Frequency response | Log₁₀ | Linear | Bode plots |
/// 
/// # Example Usage
/// ```rust
/// use rustlab_plotting::logplot_demo;
/// 
/// // Display logarithmic scale demonstration
/// logplot_demo()?;
/// 
/// // Create custom log-scale plot
/// Plot::new()
///     .line(&frequency, &magnitude)
///     .xscale(Scale::Log10)
///     .yscale(Scale::Log10)
///     .title("Frequency Response")
///     .xlabel("Frequency (Hz)")
///     .ylabel("Magnitude (dB)")
///     .show()?;
/// ```
/// 
/// # Applications
/// - **Seismology**: Gutenberg-Richter law (earthquake frequency vs magnitude)
/// - **Network analysis**: Degree distribution in scale-free networks
/// - **Finance**: Stock price analysis and compound returns
/// - **Physics**: Radioactive decay, population dynamics
/// - **Engineering**: Bode plots, frequency response analysis
/// - **Biology**: Allometric scaling laws
pub fn logplot_demo() -> Result<()> {
    use rustlab_math::range;
    use crate::plot::types::Scale;
    
    // Generate data spanning multiple orders of magnitude
    let x = range!(1.0 => 1000.0, 100);                    // x ∈ [1, 1000]
    
    // Create sample data vectors manually for demonstration
    // Power law: y = x^2 (appears linear in log-log)
    let x_vals = vec![1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0];
    let y_power_vals = vec![1.0, 25.0, 100.0, 2500.0, 10000.0, 250000.0, 1000000.0];
    let y_exp_vals = vec![1.0005, 1.025, 1.05, 1.28, 1.649, 12.18, 148.4];
    let y_inverse_vals = vec![1000.0, 200.0, 100.0, 20.0, 10.0, 2.0, 1.0];
    let y_log_vals = vec![0.0, 0.699, 1.0, 1.699, 2.0, 2.699, 3.0];
    
    let x = VectorF64::from_slice(&x_vals);
    let y_power = VectorF64::from_slice(&y_power_vals);
    let y_exp = VectorF64::from_slice(&y_exp_vals);
    let y_inverse = VectorF64::from_slice(&y_inverse_vals);
    let y_log = VectorF64::from_slice(&y_log_vals);
    
    // Create 2×2 subplot layout demonstrating different log scales
    Plot::new()
        .subplots(2, 2)
        .size(1200, 900)
        .title("Logarithmic Scale Demonstration")
        
        // Top-left: Power law on log-log scale (linear relationship)
        .subplot(0, 0)
            .title("Power Law: y = x² (log-log scale)")
            .xlabel("log₁₀(x)")
            .ylabel("log₁₀(y)")
            .line(&x, &y_power)
            .xscale(Scale::Log10)
            .yscale(Scale::Log10)
            .grid(true)
            .build()
            
        // Top-right: Exponential on semi-log scale (linear relationship)
        .subplot(0, 1)
            .title("Exponential: y = e^(x/200) (semi-log scale)")
            .xlabel("x")
            .ylabel("log₁₀(y)")
            .line(&x, &y_exp)
            .yscale(Scale::Log10)
            .grid(true)
            .build()
            
        // Bottom-left: Multiple relationships on log-log
        .subplot(1, 0)
            .title("Multiple Power Laws (log-log scale)")
            .xlabel("log₁₀(x)")
            .ylabel("log₁₀(y)")
            .line_with(&x, &y_power, "y = x²")
            .line_with(&x, &y_inverse, "y = 1000/x")
            .xscale(Scale::Log10)
            .yscale(Scale::Log10)
            .legend(true)
            .grid(true)
            .build()
            
        // Bottom-right: Comparison on linear scale (all curves)
        .subplot(1, 1)
            .title("Linear Scale Comparison")
            .xlabel("x")
            .ylabel("y")
            .line_with(&x, &y_power, "Power: x²")
            .line_with(&x, &y_exp, "Exponential: e^(x/200)")
            .line_with(&x, &y_log, "Logarithmic: log₁₀(x)")
            .legend(true)
            .grid(true)
            .build()
            
        .show()
}