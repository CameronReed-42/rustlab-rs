//! Comprehensive error handling for plotting operations with AI guidance
//!
//! Provides specialized error types for plotting operations, enabling robust
//! error handling and user-friendly diagnostics. All errors include actionable
//! guidance for resolution.
//!
//! # For AI Code Generation
//! - Use Result<T> for all fallible plotting operations
//! - Match on specific error variants for intelligent error handling
//! - Provide fallback strategies for common plotting issues
//! - Include contextual information for debugging
//!
//! # Error Categories
//! - **Backend**: Display system and rendering engine issues
//! - **Data**: Invalid input data or dimension mismatches
//! - **Configuration**: Plot settings and parameter issues
//! - **IO**: File system operations for plot export
//! - **Feature**: Unavailable functionality or backend limitations
//!
//! # Example Error Handling
//! ```
//! use rustlab_plotting::{PlottingError, Result, plot};
//!
//! match plot(&x, &y) {
//!     Ok(()) => println!("Plot displayed successfully"),
//!     Err(PlottingError::InvalidData(msg)) => {
//!         eprintln!("Data issue: {}", msg);
//!         // Fix data dimensions or clean invalid values
//!     },
//!     Err(PlottingError::Backend(msg)) => {
//!         eprintln!("Display issue: {}", msg);
//!         // Try different backend or export to file
//!     },
//!     Err(e) => eprintln!("Plotting error: {}", e),
//! }
//! ```

use thiserror::Error;

/// Comprehensive error types for plotting operations with actionable guidance
/// 
/// # For AI Code Generation
/// Each error variant provides:
/// - Clear context about the failure
/// - Specific conditions that caused the error
/// - Suggested resolution strategies
/// - Integration guidance for error recovery
/// 
/// # Error Handling Strategy
/// ```
/// match plotting_result {
///     Ok(plot) => plot,
///     Err(PlottingError::InvalidData(msg)) => {
///         // Check data dimensions, remove NaN/Inf values
///         eprintln!("Data validation failed: {}", msg);
///         return validate_and_retry(data);
///     },
///     Err(PlottingError::Backend(msg)) => {
///         // Try alternative backend or file export
///         eprintln!("Backend unavailable: {}", msg);
///         return export_to_file(plot_data, "plot.png");
///     },
///     // ... handle other cases
/// }
/// ```
#[derive(Error, Debug)]
pub enum PlottingError {
    /// Backend rendering or display system error
    /// 
    /// # When This Occurs
    /// - Display backend (GUI) not available in headless environment
    /// - Graphics drivers not properly installed
    /// - Jupyter notebook kernel connection issues
    /// - Native window system unavailable
    /// - Rendering engine initialization failure
    /// 
    /// # Resolution Strategies
    /// 1. **Use file export**: Save plots to PNG/SVG instead of displaying
    /// 2. **Switch backends**: Try different rendering backend
    /// 3. **Check environment**: Ensure GUI/display system available
    /// 4. **Jupyter mode**: Use inline plotting for notebooks
    /// 
    /// # Example Recovery
    /// ```
    /// match plot(&x, &y) {
    ///     Ok(()) => {},
    ///     Err(PlottingError::Backend(_)) => {
    ///         // Fallback to file export
    ///         Plot::new()
    ///             .line(&x, &y)
    ///             .save("plot.png")?;
    ///         println!("Plot saved to plot.png");
    ///     }
    /// }
    /// ```
    /// 
    /// # Common Causes
    /// - SSH session without X11 forwarding
    /// - Docker container without display
    /// - CI/CD environment without GUI
    /// - Missing graphics libraries
    #[error("Backend error: {0}")]
    Backend(String),
    
    /// Invalid input data that cannot be plotted
    /// 
    /// # When This Occurs
    /// - Vector length mismatch: x.len() ≠ y.len()
    /// - Empty data vectors
    /// - NaN or infinite values in data
    /// - Non-numeric data in numeric contexts
    /// - Negative values for log scales
    /// - Zero or negative bin counts for histograms
    /// 
    /// # Data Validation Requirements
    /// - **Length consistency**: All data vectors must have equal length
    /// - **Finite values**: No NaN, +∞, or -∞ in plotting data
    /// - **Appropriate ranges**: Data must be within valid ranges for scale type
    /// - **Non-empty**: At least one data point required
    /// 
    /// # Resolution Strategies
    /// 1. **Validate dimensions**: Check x.len() == y.len()
    /// 2. **Clean data**: Remove or interpolate NaN/Inf values
    /// 3. **Filter ranges**: Remove invalid values for log scales
    /// 4. **Provide defaults**: Use reasonable defaults for empty data
    /// 
    /// # Example Validation
    /// ```
    /// fn validate_plot_data(x: &VectorF64, y: &VectorF64) -> Result<()> {
    ///     if x.len() != y.len() {
    ///         return Err(PlottingError::InvalidData(
    ///             format!("Dimension mismatch: x.len()={}, y.len()={}", x.len(), y.len())
    ///         ));
    ///     }
    ///     
    ///     if x.is_empty() {
    ///         return Err(PlottingError::InvalidData(
    ///             "Cannot plot empty data".to_string()
    ///         ));
    ///     }
    ///     
    ///     for &val in x.iter().chain(y.iter()) {
    ///         if !val.is_finite() {
    ///             return Err(PlottingError::InvalidData(
    ///                 "Data contains NaN or infinite values".to_string()
    ///             ));
    ///         }
    ///     }
    ///     
    ///     Ok(())
    /// }
    /// ```
    #[error("Invalid data: {0}")]
    InvalidData(String),
    
    /// Plot configuration or settings error
    /// 
    /// # When This Occurs
    /// - Invalid color specifications (unknown color names)
    /// - Inappropriate axis scale settings (log scale with negative data)
    /// - Invalid subplot layout parameters (0 rows/columns)
    /// - Conflicting style settings
    /// - Unsupported font or style options
    /// - Invalid figure size parameters
    /// 
    /// # Configuration Requirements
    /// - **Positive dimensions**: Figure width, height > 0
    /// - **Valid colors**: Use recognized color names or hex codes
    /// - **Logical scales**: Linear for negative data, log for positive only
    /// - **Reasonable limits**: Axis ranges within plottable bounds
    /// 
    /// # Resolution Strategies
    /// 1. **Use defaults**: Fall back to default settings for invalid options
    /// 2. **Validate ranges**: Check axis limits and scale compatibility
    /// 3. **Sanitize inputs**: Clean and validate configuration parameters
    /// 4. **Progressive fallback**: Try simpler options if complex ones fail
    /// 
    /// # Example Configuration Validation
    /// ```
    /// fn create_safe_plot() -> Result<()> {
    ///     Plot::new()
    ///         .size(800, 600)        // Reasonable figure size
    ///         .color("blue")          // Standard color name
    ///         .title("My Plot")       // Simple title
    ///         .line(&x, &y)
    ///         .show()
    /// }
    /// 
    /// // Handle configuration errors gracefully
    /// match create_plot_with_settings() {
    ///     Ok(plot) => plot,
    ///     Err(PlottingError::Configuration(_)) => {
    ///         // Fall back to default settings
    ///         create_safe_plot()?
    ///     }
    /// }
    /// ```
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// File system I/O error during plot export or template loading
    /// 
    /// # When This Occurs
    /// - Insufficient permissions to write plot files
    /// - Disk space exhausted during export
    /// - Invalid file paths or names
    /// - Network file system issues
    /// - Template or configuration file loading failures
    /// 
    /// # Common I/O Operations
    /// - **Plot export**: Saving PNG, SVG, PDF files
    /// - **Template loading**: Reading style/theme files
    /// - **Cache management**: Temporary file operations
    /// - **Log file writing**: Error and debug logging
    /// 
    /// # Resolution Strategies
    /// 1. **Check permissions**: Ensure write access to target directory
    /// 2. **Verify disk space**: Confirm sufficient storage available
    /// 3. **Use absolute paths**: Avoid relative path ambiguity
    /// 4. **Fallback locations**: Try alternative directories
    /// 
    /// # Example I/O Error Handling
    /// ```
    /// fn safe_plot_export(plot: &Plot, filename: &str) -> Result<()> {
    ///     match plot.save(filename) {
    ///         Ok(()) => {
    ///             println!("Plot saved to {}", filename);
    ///             Ok(())
    ///         },
    ///         Err(PlottingError::Io(io_err)) => {
    ///             eprintln!("Failed to save {}: {}", filename, io_err);
    ///             
    ///             // Try alternative location
    ///             let backup_path = format!("/tmp/{}", filename);
    ///             plot.save(&backup_path)
    ///         },
    ///         Err(e) => Err(e),
    ///     }
    /// }
    /// ```
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Image processing error during plot rendering or export
    /// 
    /// # When This Occurs
    /// - Invalid image format specifications
    /// - Unsupported image dimensions or color depth
    /// - Memory allocation failure for large plots
    /// - Corrupted image data during processing
    /// - Codec or format library issues
    /// 
    /// # Image Format Support
    /// - **Raster formats**: PNG (recommended), JPEG, BMP
    /// - **Vector formats**: SVG (preferred for scalability)
    /// - **Print formats**: PDF (for publications)
    /// - **Web formats**: PNG for web, SVG for interactive
    /// 
    /// # Resolution Strategies
    /// 1. **Reduce resolution**: Lower DPI for large plots
    /// 2. **Change format**: Try different image format
    /// 3. **Simplify plot**: Reduce data points or complexity
    /// 4. **Increase memory**: Check available system memory
    /// 
    /// # Example Image Error Handling
    /// ```
    /// fn export_with_fallback(plot: &Plot, base_name: &str) -> Result<String> {
    ///     // Try high-quality PNG first
    ///     let png_name = format!("{}.png", base_name);
    ///     match plot.save_png(&png_name, 300) { // 300 DPI
    ///         Ok(()) => Ok(png_name),
    ///         Err(PlottingError::Image(_)) => {
    ///             // Fall back to lower resolution
    ///             let low_res_name = format!("{}_lowres.png", base_name);
    ///             plot.save_png(&low_res_name, 150)?; // 150 DPI
    ///             Ok(low_res_name)
    ///         },
    ///         Err(e) => Err(e),
    ///     }
    /// }
    /// ```
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
    
    /// Internal plotting engine error from underlying plotters library
    /// 
    /// # When This Occurs
    /// - Low-level rendering engine failures
    /// - Graphics API incompatibilities
    /// - Memory management issues in plotting backend
    /// - Coordinate transformation errors
    /// - Font rendering failures
    /// 
    /// # Underlying Causes
    /// - **Coordinate overflow**: Very large or small coordinate values
    /// - **Resource exhaustion**: Too many plot elements
    /// - **Backend limitations**: Feature not supported by current backend
    /// - **Threading issues**: Concurrent plotting operations
    /// 
    /// # Resolution Strategies
    /// 1. **Simplify plot**: Reduce number of data points or series
    /// 2. **Check data ranges**: Ensure coordinates within reasonable bounds
    /// 3. **Retry operation**: Some issues may be transient
    /// 4. **Switch backends**: Try different rendering backend
    /// 
    /// # Example Recovery
    /// ```
    /// fn robust_plot(data: &[(f64, f64)]) -> Result<()> {
    ///     // Try plotting with current backend
    ///     match create_complex_plot(data) {
    ///         Ok(plot) => plot.show(),
    ///         Err(PlottingError::Plotters(_)) => {
    ///             // Simplify and retry
    ///             eprintln!("Complex plot failed, trying simplified version");
    ///             
    ///             // Reduce data points if too many
    ///             let simplified_data = if data.len() > 1000 {
    ///                 subsample(data, 1000)
    ///             } else {
    ///                 data.to_vec()
    ///             };
    ///             
    ///             create_simple_plot(&simplified_data)?.show()
    ///         },
    ///         Err(e) => Err(e),
    ///     }
    /// }
    /// ```
    #[error("Plotters error: {0}")]
    Plotters(String),
    
    /// Requested plotting feature not available in current configuration
    /// 
    /// # When This Occurs
    /// - Backend doesn't support requested feature (3D plots, animations)
    /// - Optional dependencies not compiled in
    /// - Platform-specific features unavailable
    /// - Experimental features disabled
    /// - Version compatibility issues
    /// 
    /// # Feature Dependencies
    /// - **3D plotting**: Requires 3D rendering backend
    /// - **Animation**: Needs video encoding libraries
    /// - **Interactive plots**: Requires GUI backend
    /// - **LaTeX rendering**: Needs LaTeX installation
    /// - **Advanced statistics**: Optional statistical libraries
    /// 
    /// # Resolution Strategies
    /// 1. **Check features**: Verify compile-time features enabled
    /// 2. **Install dependencies**: Add required system libraries
    /// 3. **Use alternatives**: Find equivalent functionality
    /// 4. **Upgrade version**: Update to version with feature support
    /// 
    /// # Example Feature Checking
    /// ```
    /// fn create_advanced_plot() -> Result<()> {
    ///     // Try advanced feature first
    ///     match Plot::new().plot_3d(&x, &y, &z) {
    ///         Ok(plot) => plot.show(),
    ///         Err(PlottingError::FeatureNotAvailable(msg)) => {
    ///             eprintln!("3D plotting not available: {}", msg);
    ///             
    ///             // Fall back to 2D projection
    ///             Plot::new()
    ///                 .scatter(&x, &y)
    ///                 .title("2D Projection (Z-axis not shown)")
    ///                 .show()
    ///         },
    ///         Err(e) => Err(e),
    ///     }
    /// }
    /// 
    /// // Check feature availability before using
    /// if plotting::features::has_3d_support() {
    ///     create_3d_plot()?;
    /// } else {
    ///     create_2d_fallback()?;
    /// }
    /// ```
    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),
    
    /// Numerical computation error during algorithm execution
    /// 
    /// # When This Occurs
    /// - Contour generation algorithm failures
    /// - Mathematical optimization convergence issues
    /// - Numerical instability in computations
    /// - Invalid mathematical parameters
    /// - Algorithm-specific numerical errors
    /// 
    /// # Common Causes
    /// - **Ill-conditioned problems**: Poor numerical conditioning
    /// - **Invalid parameters**: Parameters outside valid algorithm ranges
    /// - **Convergence failure**: Iterative algorithms failing to converge
    /// - **Overflow/underflow**: Numerical values outside representable range
    /// - **Algorithmic limitations**: Edge cases not handled by algorithm
    /// 
    /// # Resolution Strategies
    /// 1. **Check parameters**: Validate algorithm parameters are within bounds
    /// 2. **Preprocess data**: Clean or normalize input data
    /// 3. **Adjust tolerances**: Relax convergence criteria if appropriate
    /// 4. **Use alternative**: Try different algorithm or approach
    /// 
    /// # Example
    /// ```
    /// match contour_plot(&x, &y, &z) {
    ///     Ok(()) => {},
    ///     Err(PlottingError::ComputationError(msg)) => {
    ///         eprintln!("Contour computation failed: {}", msg);
    ///         // Try with different parameters or simpler function
    ///     }
    /// }
    /// ```
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    /// Data dimension mismatch error with specific details
    /// 
    /// # When This Occurs
    /// - Vector length mismatch in plotting functions
    /// - Matrix dimension misalignment for 2D plots
    /// - Grid size incompatibility in contour plots
    /// - Inconsistent data structure dimensions
    /// 
    /// # Resolution Strategies
    /// 1. **Check dimensions**: Verify all input arrays have compatible sizes
    /// 2. **Reshape data**: Adjust array dimensions to match requirements
    /// 3. **Subset data**: Use consistent ranges for all input arrays
    /// 4. **Validate early**: Check dimensions before expensive computations
    /// 
    /// # Example
    /// ```
    /// match plot(&x, &y) {
    ///     Err(PlottingError::DimensionMismatch { expected, found, context }) => {
    ///         eprintln!("Dimension error: expected {}, found {} ({})", expected, found, context);
    ///         // Fix data dimensions before retrying
    ///     }
    /// }
    /// ```
    #[error("Dimension mismatch: expected {expected}, found {found} ({context})")]
    DimensionMismatch {
        expected: usize,
        found: usize,
        context: String,
    },
}

/// Standard Result type for plotting operations with comprehensive error information
/// 
/// # For AI Code Generation
/// - Use this type for all fallible plotting operations
/// - Enables robust error handling with ? operator
/// - Provides detailed error diagnostics for debugging
/// - Supports graceful degradation and fallback strategies
/// 
/// # Error Propagation Patterns
/// ```
/// fn create_dashboard() -> Result<()> {
///     let plot1 = Plot::new().line(&x1, &y1)?;  // Propagates errors
///     let plot2 = Plot::new().scatter(&x2, &y2)?;
///     
///     // Combine plots with error handling
///     Plot::new()
///         .subplots(2, 1)
///         .add_subplot(plot1)?
///         .add_subplot(plot2)?
///         .show()  // Returns Result<()>
/// }
/// 
/// // Handle errors at appropriate level
/// match create_dashboard() {
///     Ok(()) => println!("Dashboard created successfully"),
///     Err(e) => {
///         eprintln!("Dashboard creation failed: {}", e);
///         create_fallback_dashboard()?
///     }
/// }
/// ```
/// 
/// # Integration with Other RustLab Crates
/// ```
/// use rustlab_math::Result as MathResult;
/// use rustlab_plotting::Result as PlotResult;
/// 
/// fn analyze_and_plot(data: &[f64]) -> PlotResult<()> {
///     // Math operations return MathResult
///     let processed: MathResult<VectorF64> = process_data(data);
///     
///     // Convert MathResult to PlotResult if needed
///     let vector = processed.map_err(|e| 
///         PlottingError::InvalidData(format!("Math error: {}", e))
///     )?;
///     
///     // Plotting operations return PlotResult
///     plot(&vector.indices(), &vector)
/// }
/// ```
pub type Result<T> = std::result::Result<T, PlottingError>;

// ================================================================================================
// Error Construction Helpers
// ================================================================================================

impl PlottingError {
    /// Create a backend error with detailed context
    /// 
    /// # For AI Code Generation
    /// Use this constructor for backend-specific error reporting
    pub fn backend_error<S: Into<String>>(message: S) -> Self {
        Self::Backend(message.into())
    }
    
    /// Create an invalid data error with validation context
    /// 
    /// # For AI Code Generation
    /// Use for data validation failures with specific details
    pub fn invalid_data<S: Into<String>>(message: S) -> Self {
        Self::InvalidData(message.into())
    }
    
    /// Create a configuration error with setting details
    /// 
    /// # For AI Code Generation
    /// Use for plot configuration and parameter validation
    pub fn configuration_error<S: Into<String>>(message: S) -> Self {
        Self::Configuration(message.into())
    }
    
    /// Create a feature unavailable error with alternative suggestions
    /// 
    /// # For AI Code Generation
    /// Use when advanced features are not available
    pub fn feature_not_available<S: Into<String>>(message: S) -> Self {
        Self::FeatureNotAvailable(message.into())
    }
    
    /// Create a computation error for numerical algorithm failures
    /// 
    /// # For AI Code Generation
    /// Use for mathematical computation failures (contour generation, optimization, etc.)
    pub fn computation_error<S: Into<String>>(message: S) -> Self {
        Self::ComputationError(message.into())
    }
    
    /// Create a plotters engine error with technical details
    /// 
    /// # For AI Code Generation
    /// Use for low-level rendering engine failures
    pub fn plotters_error<S: Into<String>>(message: S) -> Self {
        Self::Plotters(message.into())
    }
}