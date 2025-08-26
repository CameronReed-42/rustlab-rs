# RustLab Plotting - AI Documentation

## Overview

RustLab Plotting provides a comprehensive scientific visualization framework optimized for mathematical computing and data analysis. This crate integrates seamlessly with the broader RustLab ecosystem to deliver publication-quality plots with minimal code.

## Key Features for AI Code Generation

### 1. **Fluent Builder Pattern API**
- Chainable method calls for intuitive plot construction
- Type-safe configuration with compile-time validation
- Extensible design for custom chart types and styling

### 2. **Multi-Backend Architecture**
- **Auto**: Intelligent environment detection (Jupyter vs Native)
- **Native**: Desktop windows with interactive controls and file export
- **Jupyter**: Inline notebook display with widget integration

### 3. **Scientific Theming System**
- **Publication themes**: Professional academic styling
- **Accessibility support**: Colorblind-friendly palettes
- **Custom themes**: Corporate branding and specialized use cases

### 4. **Mathematical Integration**
- Direct compatibility with `rustlab-math` data structures
- Statistical visualization with `rustlab-stats` integration
- Numerical analysis support for function plotting

## Quick Start Guide

### Basic Plotting
```rust
use rustlab_plotting::*;
use rustlab_math::*;

// Function visualization
let x = range!(-PI => PI, 1000);
let y = x.sin();
plot(&x, &y)?;

// Data correlation
scatter(&experimental_x, &experimental_y)?;

// Statistical distribution
histogram(&sample_data, 50)?;

// Category comparison
bar(&categories, &values)?;
```

### Advanced Multi-Series Plots
```rust
Plot::new()
    .line(&x, &experiment_1)
    .label("Experiment 1")
    .line(&x, &experiment_2) 
    .label("Experiment 2")
    .line(&x, &theoretical_model)
    .label("Theory")
    .legend(true)
    .scientific_theme()
    .title("Model Validation Study")
    .xlabel("Time (s)")
    .ylabel("Response")
    .show()?;
```

### Logarithmic Scale Plotting
```rust
use rustlab_plotting::plot::types::Scale;

// Power law data spanning orders of magnitude
let x = range!(1.0 => 1000.0, 100);
let y = x.map(|&xi| xi.powi(2));  // y = x²

// Log-log plot reveals linear relationship
Plot::new()
    .line(&x, &y)
    .xscale(Scale::Log10)          // Logarithmic x-axis
    .yscale(Scale::Log10)          // Logarithmic y-axis
    .title("Power Law Analysis")
    .xlabel("log₁₀(x)")
    .ylabel("log₁₀(y)")
    .show()?;

// Semi-log for exponential growth
let time = range!(0.0 => 10.0, 100);
let population = time.map(|&t| (0.2 * t).exp());

Plot::new()
    .line(&time, &population)
    .yscale(Scale::Log10)          // Only y-axis logarithmic
    .title("Exponential Growth")
    .xlabel("Time")
    .ylabel("log₁₀(Population)")
    .show()?;
```

### Multi-Panel Subplots
```rust
// Comparative analysis in 2×2 grid
Plot::new()
    .subplots(2, 2)
    .size(1200, 800)
    
    .subplot(0, 0).line(&x, &sin_data).title("sin(x)").build()
    .subplot(0, 1).line(&x, &cos_data).title("cos(x)").build()
    .subplot(1, 0).scatter(&x, &exp_data).title("Experimental").build()
    .subplot(1, 1).histogram(&residuals, 30).title("Residuals").build()
    
    .show()?;

// Scientific publication layout
Plot::new()
    .subplots(1, 3)
    .scientific_theme()
    .size(1800, 600)
    
    .subplot(0, 0).line(&time, &signal).title("(a) Time Domain").build()
    .subplot(0, 1).line(&freq, &magnitude).xscale(Scale::Log10).title("(b) Frequency Response").build()
    .subplot(0, 2).scatter(&predicted, &observed).title("(c) Model Validation").build()
    
    .show()?;
```

### Publication-Ready Styling
```rust
Plot::new()
    .scatter(&pca_component_1, &pca_component_2)
    .scientific_theme()           // Academic publication style
    .colorblind_friendly_theme()  // Accessibility compliance
    .title("Principal Component Analysis")
    .xlabel("PC1 (42.3% variance)")
    .ylabel("PC2 (28.7% variance)")
    .grid(true)
    .show()?;
```

## Chart Type Selection Guide

| Chart Type | Data Characteristics | Mathematical Applications | Function |
|------------|---------------------|---------------------------|----------|
| **Line** | Continuous, ordered | Functions y=f(x), time series, parametric curves | `plot()` |
| **Scatter** | Discrete points | Correlation analysis, clustering, feature space | `scatter()` |
| **Bar** | Categorical data | Frequencies, ANOVA visualization, comparisons | `bar()` |
| **Histogram** | Statistical distributions | Probability density, normality testing | `histogram()` |
| **Log Plot** | Multi-order magnitude | Power laws, exponential growth, scaling analysis | `logplot_demo()` |
| **Subplots** | Multi-panel analysis | Comparative studies, parameter sweeps | `subplot_demo()` |

## Backend Selection Strategy

### Automatic Detection (Recommended)
```rust
// Uses Auto backend by default - detects Jupyter vs Native
Plot::new()
    .line(&x, &y)
    .show()?;  // Automatic backend selection
```

### Explicit Backend Control
```rust
use rustlab_plotting::plot::types::Backend;

// Force native window (development/analysis)
Plot::new()
    .line(&x, &y)
    .backend(Backend::Native)
    .show()?;

// Force Jupyter inline (notebook exploration)
Plot::new()
    .line(&x, &y)
    .backend(Backend::Jupyter)
    .show()?;
```

## Error Handling Patterns

### Graceful Fallbacks
```rust
use rustlab_plotting::error::PlottingError;

match Plot::new().line(&x, &y).show() {
    Ok(_) => println!("Plot displayed successfully"),
    Err(PlottingError::BackendNotAvailable(_)) => {
        // Fallback to alternative backend
        Plot::new()
            .line(&x, &y)
            .backend(Backend::Native)
            .show()?;
    },
    Err(e) => return Err(e),
}
```

### Data Validation
```rust
// Input validation with meaningful errors
if x.len() != y.len() {
    return Err(PlottingError::DataMismatch {
        x_len: x.len(),
        y_len: y.len(),
    });
}

if x.is_empty() {
    return Err(PlottingError::EmptyData);
}
```

## Mathematical Context Integration

### Function Visualization
```rust
use rustlab_math::*;

// Mathematical function plotting
let x = range!(-2.0 => 2.0, 1000);
let gaussian = x.map(|&xi| (-xi.powi(2) / 2.0).exp() / (2.0 * PI).sqrt());

plot(&x, &gaussian)?;
```

### Statistical Analysis Workflow
```rust
use rustlab_stats::*;

// Complete analysis with visualization
let samples = random_normal_vec(1000, 100.0, 15.0);

// Statistical summary
let mean = samples.mean();
let std = samples.std();
let confidence_interval = samples.confidence_interval(0.95);

// Distribution visualization with theoretical overlay
let x_theory = range!(50.0 => 150.0, 200);
let y_theory = x_theory.map(|&x| normal_pdf(x, mean, std));

Plot::new()
    .histogram(&samples, 50)
    .line(&x_theory, &y_theory)
    .title(&format!("Normal Distribution (μ={:.1}, σ={:.1})", mean, std))
    .scientific_theme()
    .show()?;
```

### Time Series Analysis
```rust
// Signal processing visualization
let t = range!(0.0 => 10.0, 1000);
let signal = t.map(|&ti| (2.0 * PI * ti).sin() + 0.5 * (6.0 * PI * ti).cos());
let noise = random_normal_vec(1000, 0.0, 0.2);
let noisy_signal = signal.add(&noise);

Plot::new()
    .line(&t, &signal)
    .label("Clean Signal")
    .line(&t, &noisy_signal)
    .label("Noisy Signal") 
    .xlabel("Time (s)")
    .ylabel("Amplitude")
    .legend(true)
    .show()?;
```

## Performance Optimization

### Data Size Guidelines
- **Line plots**: Optimal for 100-10,000 points
- **Scatter plots**: Good up to 1,000-5,000 points
- **Bar charts**: Best for <100 categories
- **Histograms**: Efficient for any data size

### Memory Management
```rust
// Efficient subsampling for large datasets
let large_data = generate_large_dataset(1_000_000);

// Subsample for scatter plots
let subsampled = subsample(&large_data, 5000);
scatter(&subsampled.x, &subsampled.y)?;

// Use histograms for massive datasets
histogram(&large_data, 100)?;  // Binning handles size efficiently
```

### Async Rendering
```rust
// Non-blocking rendering for large datasets
Plot::new()
    .scatter(&massive_data.x, &massive_data.y)
    .backend(Backend::Native)
    .async_render(true)
    .show()?;
```

## Theming and Accessibility

### Predefined Themes
```rust
// Academic publication style
Plot::new()
    .line(&x, &y)
    .scientific_theme()
    .show()?;

// Presentation style
Plot::new()
    .line(&x, &y)
    .dark_theme()
    .show()?;

// Accessibility optimized
Plot::new()
    .line(&x, &y)
    .colorblind_friendly_theme()
    .show()?;
```

### Custom Corporate Themes
```rust
use rustlab_plotting::style::theme::Theme;
use rustlab_plotting::plot::types::Color;

let corporate_theme = Theme {
    background_color: Color::from_hex("#FFFFFF")?,
    text_color: Color::from_hex("#333333")?,
    palette: vec![
        Color::from_hex("#2E86AB")?,  // Corporate blue
        Color::from_hex("#A23B72")?,  // Corporate magenta
        Color::from_hex("#F18F01")?,  // Corporate orange
        Color::from_hex("#C73E1D")?,  // Corporate red
    ],
    name: "Corporate".to_string(),
    ..Theme::default()
};

Plot::new()
    .theme(corporate_theme)
    .line(&quarterly_data.x, &quarterly_data.revenue)
    .show()?;
```

## Integration Examples

### With RustLab Math
```rust
use rustlab_math::*;
use rustlab_plotting::*;

// Fourier analysis visualization
let t = range!(0.0 => 1.0, 1000);
let signal = t.map(|&ti| (50.0 * PI * ti).sin() + 0.5 * (120.0 * PI * ti).cos());
let fft_result = fft(&signal);

Plot::new()
    .line(&t, &signal)
    .title("Time Domain Signal")
    .show()?;

Plot::new()
    .line(&fft_result.frequencies, &fft_result.magnitudes)
    .title("Frequency Domain")
    .xlabel("Frequency (Hz)")
    .ylabel("Magnitude")
    .show()?;
```

### With RustLab Stats
```rust
use rustlab_stats::*;
use rustlab_plotting::*;

// Regression analysis with confidence bands
let x = range!(0.0 => 10.0, 100);
let y_true = x.map(|&xi| 2.0 * xi + 1.0);
let noise = random_normal_vec(100, 0.0, 1.0);
let y_observed = y_true.add(&noise);

let regression = linear_regression(&x, &y_observed)?;
let y_predicted = x.map(|&xi| regression.predict(xi));
let confidence_band = regression.confidence_band(&x, 0.95);

Plot::new()
    .scatter(&x, &y_observed)
    .label("Data")
    .line(&x, &y_predicted)
    .label("Regression")
    .fill_between(&x, &confidence_band.lower, &confidence_band.upper)
    .label("95% CI")
    .legend(true)
    .title(&format!("Linear Regression (R² = {:.3})", regression.r_squared))
    .show()?;
```

## Common Patterns

### Logarithmic Scale Plotting

#### Power Law Analysis
```rust
use rustlab_plotting::plot::types::Scale;

// Data spanning multiple orders of magnitude
let x = range!(1.0 => 1000.0, 100);
let y = x.map(|&xi| xi.powi(2));  // y = x²

// Log-log plot reveals linear relationship for power laws
Plot::new()
    .line(&x, &y)
    .xscale(Scale::Log10)
    .yscale(Scale::Log10)
    .title("Power Law: y = x² (appears linear in log-log)")
    .xlabel("log₁₀(x)")
    .ylabel("log₁₀(y)")
    .show()?;
```

#### Exponential Growth Analysis
```rust
// Semi-log plot for exponential data
let t = range!(0.0 => 10.0, 100);
let population = t.map(|&ti| (0.1 * ti).exp());

Plot::new()
    .line(&t, &population)
    .yscale(Scale::Log10)
    .title("Exponential Growth (linear in semi-log)")
    .xlabel("Time")
    .ylabel("log₁₀(Population)")
    .show()?;
```

#### Scale Selection Guidelines
| Data Pattern | X-Scale | Y-Scale | Interpretation |
|--------------|---------|---------|----------------|
| Power law y=xᵅ | Log₁₀ | Log₁₀ | Linear with slope α |
| Exponential y=eᵃˣ | Linear | Log₁₀ | Linear with slope a |
| Time series growth | Linear | Log₁₀ | Growth rate analysis |
| Frequency response | Log₁₀ | Linear | Bode plots |

### Multi-Panel Layouts

#### Basic Subplot Grid
```rust
// 2×2 subplot configuration
Plot::new()
    .subplots(2, 2)
    .size(1200, 800)
    
    // Top row
    .subplot(0, 0).line(&x1, &y1).title("Dataset 1").build()
    .subplot(0, 1).scatter(&x2, &y2).title("Dataset 2").build()
    
    // Bottom row  
    .subplot(1, 0).bar(&categories, &values).title("Categories").build()
    .subplot(1, 1).histogram(&distribution, 30).title("Distribution").build()
    
    .show()?;
```

#### Comparative Analysis Layout
```rust
// Multi-model comparison in subplots
let models = vec![linear_model, quadratic_model, exponential_model];

Plot::new()
    .subplots(2, 2)
    .title("Model Comparison Study")
    
    // Individual model fits
    .subplot(0, 0).scatter(&data.x, &data.y).line(&data.x, &models[0]).title("Linear").build()
    .subplot(0, 1).scatter(&data.x, &data.y).line(&data.x, &models[1]).title("Quadratic").build() 
    .subplot(1, 0).scatter(&data.x, &data.y).line(&data.x, &models[2]).title("Exponential").build()
    
    // Combined comparison
    .subplot(1, 1)
        .scatter(&data.x, &data.y).label("Data")
        .line(&data.x, &models[0]).label("Linear")
        .line(&data.x, &models[1]).label("Quadratic")
        .line(&data.x, &models[2]).label("Exponential")
        .title("All Models")
        .legend(true)
        .build()
    
    .show()?;
```

#### Scientific Publication Layout
```rust
// Professional multi-panel figure
Plot::new()
    .subplots(2, 3)
    .size(1800, 1200)
    .scientific_theme()
    
    // Time series analysis
    .subplot(0, 0).line(&time, &raw_data).title("(a) Raw Data").build()
    .subplot(0, 1).line(&time, &filtered_data).title("(b) Filtered").build()
    .subplot(0, 2).line(&frequency, &spectrum).xscale(Scale::Log10).title("(c) Spectrum").build()
    
    // Statistical analysis
    .subplot(1, 0).histogram(&residuals, 30).title("(d) Residuals").build()
    .subplot(1, 1).scatter(&predicted, &observed).title("(e) Predictions").build()
    .subplot(1, 2).line(&x_theory, &y_theory).line(&x_data, &y_data).title("(f) Theory vs Data").build()
    
    .show()?;
```

### Export Workflows
```rust
// High-resolution export for publications
Plot::new()
    .line(&x, &y)
    .scientific_theme()
    .size(300, 200)  // DPI scaling
    .save("figure_1.pdf")?;  // Vector format

// Multiple format export
let plot = Plot::new().scatter(&x, &y);
plot.save("analysis.png")?;  // Raster for web
plot.save("analysis.svg")?;  // Vector for presentations
plot.save("analysis.pdf")?;  // Publication quality
```

## Module Organization

- **`lib`**: Main plotting interface and convenience functions
- **`plot`**: Core plotting types and builder patterns
- **`charts`**: Specific chart type implementations
- **`style`**: Theming system and color management
- **`backend`**: Multi-target rendering abstraction
- **`error`**: Comprehensive error handling system

## Advanced Plotting Techniques

### Logarithmic Scale Applications

#### Power Law Detection
```rust
// Earthquake magnitude-frequency relationship (Gutenberg-Richter law)
let magnitudes = range!(2.0 => 8.0, 100);
let frequencies = magnitudes.map(|&m| 10_f64.powf(4.0 - m));  // N = 10^(4-M)

Plot::new()
    .line(&magnitudes, &frequencies)
    .yscale(Scale::Log10)
    .title("Earthquake Frequency vs Magnitude")
    .xlabel("Magnitude")
    .ylabel("log₁₀(Frequency)")
    .show()?;
```

#### Network Degree Distribution
```rust
// Scale-free network analysis
let degrees = range!(1.0 => 1000.0, 100);
let probabilities = degrees.map(|&k| k.powf(-2.1));  // P(k) ∝ k^(-γ)

Plot::new()
    .scatter(&degrees, &probabilities)
    .xscale(Scale::Log10)
    .yscale(Scale::Log10)
    .title("Scale-Free Network Degree Distribution")
    .xlabel("log₁₀(Degree)")
    .ylabel("log₁₀(Probability)")
    .show()?;
```

#### Bode Plot Analysis
```rust
// Frequency response analysis
let frequencies = range!(0.1 => 1000.0, 1000);
let magnitudes = frequencies.map(|&f| 20.0 * (1.0 / (1.0 + f.powi(2))).log10());
let phases = frequencies.map(|&f| -f.atan().to_degrees());

Plot::new()
    .subplots(2, 1)
    .size(1000, 800)
    
    .subplot(0, 0)
        .line(&frequencies, &magnitudes)
        .xscale(Scale::Log10)
        .title("Magnitude Response")
        .xlabel("Frequency (Hz)")
        .ylabel("Magnitude (dB)")
        .grid(true)
        .build()
        
    .subplot(1, 0)
        .line(&frequencies, &phases)
        .xscale(Scale::Log10)
        .title("Phase Response")
        .xlabel("Frequency (Hz)")
        .ylabel("Phase (degrees)")
        .grid(true)
        .build()
        
    .show()?;
```

### Advanced Subplot Layouts

#### Time Series Panel Analysis
```rust
// Multi-component time series analysis
Plot::new()
    .subplots(4, 1)
    .size(1200, 1600)
    .title("Time Series Decomposition")
    
    .subplot(0, 0).line(&time, &observed).title("(a) Observed Data").build()
    .subplot(1, 0).line(&time, &trend).title("(b) Trend Component").build()
    .subplot(2, 0).line(&time, &seasonal).title("(c) Seasonal Component").build()
    .subplot(3, 0).line(&time, &residuals).title("(d) Residuals").build()
    
    .show()?;
```

#### Statistical Diagnostic Panel
```rust
// Model validation diagnostics
Plot::new()
    .subplots(2, 3)
    .scientific_theme()
    .size(1800, 1200)
    
    // Fit quality
    .subplot(0, 0).scatter(&observed, &predicted).title("Observed vs Predicted").build()
    .subplot(0, 1).scatter(&predicted, &residuals).title("Residuals vs Fitted").build()
    .subplot(0, 2).histogram(&residuals, 30).title("Residual Distribution").build()
    
    // Model assumptions
    .subplot(1, 0).line(&time, &residuals).title("Residuals vs Time").build()
    .subplot(1, 1).scatter(&leverage, &residuals).title("Leverage vs Residuals").build()
    .subplot(1, 2).line(&quantiles_theory, &quantiles_sample).title("Q-Q Plot").build()
    
    .show()?;
```

#### Parameter Space Exploration
```rust
// Parameter sensitivity analysis
let parameter_grid = create_parameter_grid(param1_range, param2_range);
let results = parameter_grid.iter().map(|params| run_simulation(params)).collect();

Plot::new()
    .subplots(2, 2)
    .title("Parameter Sensitivity Analysis")
    
    .subplot(0, 0).heatmap(&param1_range, &param2_range, &results).title("Response Surface").build()
    .subplot(0, 1).line(&param1_range, &sensitivity_1).title("Sensitivity to Parameter 1").build()
    .subplot(1, 0).line(&param2_range, &sensitivity_2).title("Sensitivity to Parameter 2").build()
    .subplot(1, 1).contour(&param1_range, &param2_range, &results).title("Contour Plot").build()
    
    .show()?;
```

## AI Code Generation Tips

1. **Start with convenience functions** (`plot`, `scatter`, `bar`, `histogram`) for simple cases
2. **Use builder pattern** for complex multi-series or customized plots
3. **Apply themes early** in the builder chain for consistent styling
4. **Handle errors gracefully** with backend fallbacks and data validation
5. **Leverage automatic backend detection** unless specific output target required
6. **Integrate with rustlab-math/stats** for complete scientific workflows
7. **Consider performance** when plotting large datasets (subsampling, histograms)
8. **Use appropriate chart types** based on data characteristics and analysis goals
9. **Apply logarithmic scales** for data spanning multiple orders of magnitude
10. **Use subplots** for comparative analysis and comprehensive presentations
11. **Choose log-log plots** for power law detection and scaling analysis
12. **Use semi-log plots** for exponential growth and decay analysis

This crate enables rapid development of publication-quality scientific visualizations with minimal boilerplate while maintaining full customization capabilities for specialized requirements including advanced logarithmic scaling and sophisticated multi-panel layouts.