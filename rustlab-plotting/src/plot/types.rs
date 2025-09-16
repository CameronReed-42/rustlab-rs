//! Core data types and structures for scientific plotting
//!
//! Defines the fundamental types used throughout the plotting system including
//! backends, scales, configurations, and data series. All types are designed
//! for mathematical computing with rustlab-math integration.

use crate::error::Result;
use crate::style::theme::Theme;
use rustlab_math::VectorF64;
use std::path::Path;

/// Plotting backend specification for different rendering environments
/// 
/// # For AI Code Generation
/// - **Native**: Desktop GUI applications with interactive windows
/// - **Jupyter**: Inline plots for Jupyter notebooks and lab environments
/// - **Auto**: Automatically detect best backend for current environment
/// 
/// # Backend Selection
/// ```rust
/// // Explicit backend selection
/// Plot::new()
///     .backend(Backend::Jupyter)  // Force Jupyter mode
///     .line(&x, &y)
///     .show()?;
/// 
/// // Auto-detection (recommended)
/// Plot::new()  // Uses Backend::Auto by default
///     .line(&x, &y)
///     .show()?;  // Chooses appropriate backend
/// ```
/// 
/// # Environment Detection
/// - **Jupyter**: Detected via environment variables and kernel context
/// - **Native**: Used when GUI display system available
/// - **Headless**: Falls back to file export when no display available
#[derive(Debug, Clone, Copy)]
pub enum Backend {
    /// Native desktop GUI backend for interactive plotting
    /// 
    /// Requires desktop environment with display system.
    /// Supports real-time interaction, zooming, and panning.
    Native,
    
    /// Jupyter notebook/lab inline plotting backend
    /// 
    /// Renders plots directly in notebook cells.
    /// Optimized for data analysis workflows.
    Jupyter,
    
    /// Automatic backend detection based on environment
    /// 
    /// Chooses best available backend automatically.
    /// Recommended for most use cases.
    Auto,
}

/// Axis scaling types for mathematical data visualization
/// 
/// # Mathematical Specifications
/// - **Linear**: f(x) = x (identity mapping)
/// - **Log10**: f(x) = log₁₀(x) (common logarithm)
/// - **Log2**: f(x) = log₂(x) (binary logarithm) 
/// - **Ln**: f(x) = ln(x) (natural logarithm)
/// 
/// # For AI Code Generation
/// Choose scale based on data characteristics:
/// - **Linear**: Default for most data
/// - **Log10**: Scientific data spanning orders of magnitude
/// - **Log2**: Computer science (powers of 2)
/// - **Ln**: Natural processes and exponential data
/// 
/// # Requirements
/// - Logarithmic scales require **positive data only**
/// - Zero or negative values will cause errors
/// - Data should span multiple orders of magnitude for log scales
/// 
/// # Example
/// ```rust
/// // Exponential data visualization
/// let x = range!(1.0 => 1000.0, 100);
/// let y = x.map(|&xi| xi.powf(2.0));  // y = x²
/// 
/// Plot::new()
///     .line(&x, &y)
///     .xscale(Scale::Log10)  // Log scale for x-axis
///     .yscale(Scale::Log10)  // Log scale for y-axis
///     .show()?;
/// ```
#[derive(Debug, Clone, Copy)]
pub enum Scale {
    /// Linear scaling: f(x) = x
    /// 
    /// Standard arithmetic progression.
    /// Use for most general-purpose plotting.
    Linear,
    
    /// Base-10 logarithmic scaling: f(x) = log₁₀(x)
    /// 
    /// Common in scientific applications.
    /// Requires x > 0.
    Log10,
    
    /// Base-2 logarithmic scaling: f(x) = log₂(x)
    /// 
    /// Useful for computer science applications.
    /// Powers of 2 appear as equal intervals.
    Log2,
    
    /// Natural logarithmic scaling: f(x) = ln(x)
    /// 
    /// Natural processes and exponential growth.
    /// Base e ≈ 2.718.
    Ln,
}

impl Default for Backend {
    fn default() -> Self {
        Backend::Auto
    }
}

/// Comprehensive plot configuration with styling and layout parameters
/// 
/// # For AI Code Generation
/// Central configuration structure controlling:
/// - **Dimensions**: Figure size and layout
/// - **Labels**: Titles and axis labels
/// - **Visual**: Grid, legend, theme settings
/// - **Scaling**: Axis limits and scale types
/// - **Backend**: Rendering engine selection
/// 
/// # Default Values
/// - Size: 800×600 pixels
/// - Grid: Enabled
/// - Legend: Disabled (enable when needed)
/// - Scales: Linear for both axes
/// - Backend: Auto-detection
/// 
/// # Configuration Patterns
/// ```rust
/// // Basic configuration
/// Plot::new()
///     .size(1200, 800)
///     .title("My Analysis")
///     .xlabel("Time (s)")
///     .ylabel("Amplitude")
///     .grid(true)
///     .legend(true);
/// 
/// // Scientific publication style
/// Plot::new()
///     .size(800, 600)
///     .scientific_theme()
///     .xlabel("Frequency (Hz)")
///     .ylabel("Power (dB)");
/// ```
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Figure width in pixels
    pub width: u32,
    
    /// Figure height in pixels  
    pub height: u32,
    
    /// Main plot title (optional)
    pub title: Option<String>,
    
    /// X-axis label (optional)
    pub xlabel: Option<String>,
    
    /// Y-axis label (optional)
    pub ylabel: Option<String>,
    
    /// Enable grid lines for easier reading
    pub grid: bool,
    
    /// Show legend for multiple series
    pub legend: bool,
    
    /// Rendering backend selection
    pub backend: Backend,
    
    /// X-axis limits: (min, max) or None for auto-scaling
    pub x_limits: Option<(f64, f64)>,
    
    /// Y-axis limits: (min, max) or None for auto-scaling
    pub y_limits: Option<(f64, f64)>,
    
    /// X-axis scaling type
    pub x_scale: Scale,
    
    /// Y-axis scaling type
    pub y_scale: Scale,
    
    /// Visual theme and color scheme
    pub theme: Theme,
    
    /// Subplot layout configuration (if using subplots)
    pub subplot_layout: Option<SubplotLayout>,
}

/// Subplot grid layout configuration for multi-panel plots
/// 
/// # For AI Code Generation
/// Defines grid structure for subplot arrangements:
/// - **rows**: Number of subplot rows
/// - **cols**: Number of subplot columns  
/// - **current_subplot**: Active subplot index for building
/// 
/// # Layout Indexing
/// Subplots are indexed as (row, col) starting from (0, 0):
/// ```
/// // 2×3 layout:
/// // (0,0) (0,1) (0,2)
/// // (1,0) (1,1) (1,2)
/// ```
/// 
/// # Example
/// ```rust
/// Plot::new()
///     .subplots(2, 3)  // 2 rows, 3 columns
///     .subplot(0, 0)   // Top-left panel
///         .line(&x1, &y1)
///         .build()
///     .subplot(1, 2)   // Bottom-right panel
///         .scatter(&x2, &y2)
///         .build()
///     .show()?;
/// ```
#[derive(Debug, Clone)]
pub struct SubplotLayout {
    /// Number of subplot rows in the grid
    pub rows: usize,
    
    /// Number of subplot columns in the grid
    pub cols: usize,
    
    /// Current subplot index being configured
    pub current_subplot: usize,
}

/// Main plotting structure for creating and configuring plots
/// 
/// # For AI Code Generation
/// Central plotting interface supporting:
/// - **Series management**: Multiple data series with styling
/// - **Configuration**: Titles, labels, scales, themes
/// - **Layouts**: Single plots and multi-panel subplots
/// - **Backends**: Automatic or explicit rendering target selection
/// 
/// # Builder Pattern Usage
/// ```rust
/// Plot::new()
///     .line(&x, &y)
///     .title("My Plot")
///     .xlabel("X Axis")
///     .ylabel("Y Axis")
///     .scientific_theme()
///     .show()?;
/// ```
#[derive(Debug, Clone)]
pub struct Plot {
    /// Plot configuration settings
    pub config: PlotConfig,
    
    /// Data series to be plotted
    pub series: Vec<Series>,
    
    /// Subplot configurations for multi-panel layouts
    pub subplots: Vec<Subplot>,
    
    /// Optional heatmap data (for heatmap plots)
    pub heatmap_data: Option<HeatmapData>,
    
    /// Optional contour data (for contour plots)
    pub contour_data: Option<crate::contour::ContourData>,
    
    /// Optional 3D surface data (for 3D surface plots)
    pub surface3d_data: Option<crate::surface::Surface3DData>,
}

/// Individual subplot configuration within multi-panel layout
/// 
/// Contains plot configuration and series specific to one panel
/// in a subplot grid arrangement.
#[derive(Debug, Clone)]
pub struct Subplot {
    /// Position in subplot grid (row, col)
    pub position: SubplotPosition,
    
    /// Plot configuration for this subplot
    pub config: PlotConfig,
    
    /// Data series for this subplot
    pub series: Vec<Series>,
}

/// Position specification for subplot in grid layout
/// 
/// Defines which cell (row, col) in the subplot grid
/// this subplot occupies.
#[derive(Debug, Clone, Copy)]
pub struct SubplotPosition {
    /// Row index (0-based)
    pub row: usize,
    
    /// Column index (0-based)
    pub col: usize,
}

/// Plot type enumeration for different chart types
/// 
/// Specifies the visual representation style for data series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlotType {
    /// Line plot connecting data points
    Line,
    
    /// Scatter plot with individual point markers
    Scatter,
    
    /// Bar chart for categorical data
    Bar,
    
    /// Histogram for distribution analysis
    Histogram,
    
    /// Heatmap for matrix visualization
    Heatmap,
    
    /// Contour plot for 2D scalar fields
    Contour,
    
    /// 3D surface plot
    Surface3D,
}

/// Data structure for heatmap visualization
#[derive(Debug, Clone)]
pub struct HeatmapData {
    /// The matrix data to visualize (rows × cols)
    pub matrix: rustlab_math::ArrayF64,
    
    /// Colormap to use for visualization
    pub colormap: crate::colormap::ColorMap,
    
    /// Optional minimum value for color scaling
    pub vmin: Option<f64>,
    
    /// Optional maximum value for color scaling
    pub vmax: Option<f64>,
    
    /// Whether to show a colorbar
    pub show_colorbar: bool,
    
    /// Label for the colorbar
    pub colorbar_label: Option<String>,
    
    /// Whether to show values in cells
    pub show_values: bool,
    
    /// Custom x-axis tick labels
    pub x_labels: Option<Vec<String>>,
    
    /// Custom y-axis tick labels
    pub y_labels: Option<Vec<String>>,
}

/// Backend trait for different rendering implementations
/// 
/// Defines the interface that all plotting backends must implement
/// for rendering plots to different targets (native windows, Jupyter, etc.).
pub trait PlotBackend {
    /// Render a plot with the given configuration and data
    fn render(&mut self, plot: &Plot) -> Result<()>;
    
    /// Save plot to file with specified path
    fn save(&mut self, plot: &Plot, path: &Path) -> Result<()>;
    
    /// Check if backend is available in current environment
    fn is_available(&self) -> bool;
    
    /// Display the rendered plot
    fn show(&mut self) -> Result<()>;
}

impl Default for Scale {
    fn default() -> Self {
        Scale::Linear
    }
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: None,
            xlabel: None,
            ylabel: None,
            grid: true,
            legend: false,
            backend: Backend::default(),
            x_limits: None,
            y_limits: None,
            x_scale: Scale::default(),
            y_scale: Scale::default(),
            theme: Theme::default(),
            subplot_layout: None,
        }
    }
}


/// Individual data series with styling and metadata
/// 
/// # For AI Code Generation
/// Represents a single data series within a plot:
/// - **Data**: x and y coordinate vectors
/// - **Styling**: Color, line style, markers
/// - **Metadata**: Labels for legends
/// - **Type**: Chart type (line, scatter, bar, histogram)
/// 
/// # Data Requirements
/// - `x` and `y` must have equal length
/// - Values must be finite (no NaN or infinity)
/// - For log scales: all values must be positive
/// 
/// # Example
/// ```rust
/// // Series are typically created via Plot methods:
/// Plot::new()
///     .line(&x, &y)           // Creates Line series
///     .scatter(&x2, &y2)      // Creates Scatter series
///     .bar(&categories, &counts);  // Creates Bar series
/// ```
#[derive(Debug, Clone)]
pub struct Series {
    /// X-coordinate data vector
    pub x: VectorF64,
    
    /// Y-coordinate data vector
    pub y: VectorF64,
    
    /// Series label for legend (optional)
    pub label: Option<String>,
    
    /// Chart type for this series
    pub plot_type: PlotType,
    
    /// Line/marker color (auto-assigned if None)
    pub color: Option<Color>,
    
    /// Line style (solid, dashed, dotted)
    pub style: Option<LineStyle>,
    
    /// Marker type for data points
    pub marker: Option<Marker>,
    
    /// Marker size in points
    pub marker_size: f32,
}

/// RGB color specification for plot elements
/// 
/// # For AI Code Generation
/// - Use predefined constants for common colors
/// - Create custom colors with RGB values (0-255)
/// - Colors are automatically assigned if not specified
/// 
/// # Color Constants
/// ```rust
/// Plot::new()
///     .line(&x, &y)
///     .color(Color::BLUE)    // Predefined color
///     .line(&x2, &y2)
///     .color(Color { r: 255, g: 100, b: 0 });  // Custom orange
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Color {
    /// Red component (0-255)
    pub r: u8,
    
    /// Green component (0-255)
    pub g: u8,
    
    /// Blue component (0-255)
    pub b: u8,
}

impl Color {
    /// Standard blue color (RGB: 0, 0, 255)
    pub const BLUE: Color = Color { r: 0, g: 0, b: 255 };
    
    /// Standard red color (RGB: 255, 0, 0)
    pub const RED: Color = Color { r: 255, g: 0, b: 0 };
    
    /// Standard green color (RGB: 0, 255, 0)
    pub const GREEN: Color = Color { r: 0, g: 255, b: 0 };
    
    /// Black color for text and borders (RGB: 0, 0, 0)
    pub const BLACK: Color = Color { r: 0, g: 0, b: 0 };
    
    /// Orange color for highlights (RGB: 255, 165, 0)
    pub const ORANGE: Color = Color { r: 255, g: 165, b: 0 };
    
    /// Purple color for contrast (RGB: 128, 0, 128)
    pub const PURPLE: Color = Color { r: 128, g: 0, b: 128 };
    
    /// Create a custom color from RGB values
    /// 
    /// # Example
    /// ```rust
    /// let custom_color = Color::from_rgb(100, 150, 200);
    /// ```
    pub const fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Color { r, g, b }
    }
    
    /// Create color from hex string (e.g., "#FF0000" for red)
    /// 
    /// # Example
    /// ```rust
    /// let red = Color::from_hex("#FF0000").unwrap();
    /// ```
    pub fn from_hex(hex: &str) -> std::result::Result<Self, std::num::ParseIntError> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return Err("Invalid hex color length".parse::<i32>().unwrap_err());
        }
        
        let r = u8::from_str_radix(&hex[0..2], 16)?;
        let g = u8::from_str_radix(&hex[2..4], 16)?;
        let b = u8::from_str_radix(&hex[4..6], 16)?;
        
        Ok(Color { r, g, b })
    }
}

/// Line style specification for plot series
/// 
/// # For AI Code Generation
/// Visual differentiation for multiple series:
/// - **Solid**: Default continuous line
/// - **Dashed**: Intermittent line pattern
/// - **Dotted**: Dotted line pattern
/// 
/// # Usage
/// ```rust
/// PlotBuilder::new()
///     .line(&x, &y1)
///     .style(LineStyle::Solid)
///     .line(&x, &y2)
///     .style(LineStyle::Dashed)
///     .line(&x, &y3)
///     .style(LineStyle::Dotted)
///     .build();
/// ```
#[derive(Debug, Clone, Copy)]
pub enum LineStyle {
    /// Continuous solid line (default)
    Solid,
    
    /// Dashed line pattern
    Dashed,
    
    /// Dotted line pattern
    Dotted,
}

/// Marker style for data points in plots
/// 
/// # For AI Code Generation
/// Point markers for scatter plots and line endpoints:
/// - **None**: No markers (line plots)
/// - **Geometric**: Circle, Square, Triangle, Diamond
/// - **Symbols**: Cross, Plus
/// 
/// # Usage
/// ```rust
/// Plot::new()
///     .scatter_marker(&x, &y, Marker::Circle)
///     .scatter_marker(&x2, &y2, Marker::Square);
/// ```
/// 
/// # Marker Selection Guidelines
/// - **Circle**: Most common, good visibility
/// - **Square**: High contrast, technical plots
/// - **Triangle**: Directional data
/// - **Diamond**: Special emphasis
/// - **Cross/Plus**: Overlapping data points
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Marker {
    /// No marker (line plots)
    None,
    
    /// Circular marker (most common)
    Circle,
    
    /// Square marker
    Square,
    
    /// Triangular marker
    Triangle,
    
    /// Diamond-shaped marker
    Diamond,
    
    /// Cross marker (×)
    Cross,
    
    /// Plus marker (+)
    Plus,
}

