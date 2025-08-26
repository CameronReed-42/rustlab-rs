//! Backend abstraction for multi-target rendering
//!
//! Provides unified interface for rendering plots across different environments
//! and output targets. Automatically detects runtime context and selects
//! appropriate backend for optimal user experience.
//!
//! # For AI Code Generation
//!
//! ## Backend Selection Strategy
//!
//! | Backend | Environment | Output | Use Case |
//! |---------|-------------|--------|----------|
//! | **Auto** | Context detection | Adaptive | Default choice, automatic |
//! | **Native** | Desktop/CLI | Window/file | Development, analysis |
//! | **Jupyter** | Notebook kernel | Inline display | Interactive exploration |
//!
//! ## Automatic Detection Logic
//! 
//! The `Auto` backend intelligently selects the most appropriate rendering target:
//!
//! ```rust
//! // Environment detection hierarchy:
//! // 1. Jupyter kernel: Check JPY_PARENT_PID, JUPYTER_RUNTIME_DIR
//! // 2. Native environment: Default fallback
//! 
//! use rustlab_plotting::*;
//! 
//! // Automatic backend selection (recommended)
//! Plot::new()
//!     .line(&x, &y)
//!     .show()?;  // Uses Auto backend by default
//! ```
//!
//! ## Explicit Backend Control
//!
//! ```rust
//! use rustlab_plotting::*;
//! use rustlab_plotting::plot::types::Backend;
//!
//! // Force native window display
//! Plot::new()
//!     .line(&x, &y)
//!     .backend(Backend::Native)
//!     .show()?;
//!
//! // Force Jupyter inline display
//! Plot::new()
//!     .line(&x, &y)
//!     .backend(Backend::Jupyter)
//!     .show()?;
//!
//! // Automatic selection (default)
//! Plot::new()
//!     .line(&x, &y)
//!     .backend(Backend::Auto)
//!     .show()?;
//! ```
//!
//! # Backend-Specific Features
//!
//! ## Native Backend
//! ```rust
//! // High-resolution displays, interactive zooming
//! Plot::new()
//!     .line(&high_res_data.x, &high_res_data.y)
//!     .backend(Backend::Native)
//!     .size(1920, 1080)        // High resolution
//!     .interactive(true)       // Zoom/pan controls
//!     .show()?;
//!
//! // File export capabilities
//! Plot::new()
//!     .scatter(&x, &y)
//!     .backend(Backend::Native)
//!     .save("analysis.png")?;  // PNG, PDF, SVG support
//! ```
//!
//! ## Jupyter Backend
//! ```rust
//! // Inline display with notebook integration
//! Plot::new()
//!     .line(&time_series.t, &time_series.values)
//!     .backend(Backend::Jupyter)
//!     .title("Time Series Analysis")
//!     .show()?;  // Displays inline in cell output
//!
//! // Widget integration for interactive plots
//! Plot::new()
//!     .scatter(&features.x, &features.y)
//!     .backend(Backend::Jupyter)
//!     .interactive_widgets(true)  // Jupyter widgets
//!     .show()?;
//! ```
//!
//! # Performance Considerations
//!
//! ## Memory Management
//! ```rust
//! // Efficient resource cleanup
//! {
//!     let plot = Plot::new()
//!         .line(&large_dataset.x, &large_dataset.y)
//!         .backend(Backend::Native);
//!     
//!     plot.show()?;  // Automatic cleanup when plot goes out of scope
//! }
//! ```
//!
//! ## Async Rendering
//! ```rust
//! // Non-blocking display for large datasets
//! Plot::new()
//!     .scatter(&massive_data.x, &massive_data.y)
//!     .backend(Backend::Native)
//!     .async_render(true)      // Non-blocking rendering
//!     .show()?;
//! ```
//!
//! # Error Handling Patterns
//!
//! ```rust
//! use rustlab_plotting::error::PlottingError;
//!
//! // Graceful fallback on backend failure
//! match Plot::new()
//!     .line(&x, &y)
//!     .backend(Backend::Jupyter)
//!     .show() 
//! {
//!     Ok(_) => println!("Plot displayed successfully"),
//!     Err(PlottingError::BackendNotAvailable(_)) => {
//!         // Fallback to native backend
//!         Plot::new()
//!             .line(&x, &y)
//!             .backend(Backend::Native)
//!             .show()?;
//!     },
//!     Err(e) => return Err(e),
//! }
//! ```
//!
//! # Integration with RustLab Ecosystem
//!
//! ```rust
//! use rustlab_math::*;
//! use rustlab_stats::*;
//! use rustlab_plotting::*;
//!
//! // Scientific workflow with automatic backend selection
//! let data = generate_experimental_data(1000);
//! let analysis = statistical_analysis(&data);
//!
//! // Automatic backend selection works across environments
//! Plot::new()
//!     .line(&data.x, &data.y)
//!     .line(&data.x, &analysis.fitted_curve)
//!     .title(&format!("R² = {:.3}", analysis.r_squared))
//!     .show()?;  // Native window in CLI, inline in Jupyter
//! ```
//!
//! # Development Workflow
//!
//! ```rust
//! // Development: Native backend for debugging
//! #[cfg(debug_assertions)]
//! let backend = Backend::Native;
//!
//! // Production: Auto detection for deployment
//! #[cfg(not(debug_assertions))]
//! let backend = Backend::Auto;
//!
//! Plot::new()
//!     .line(&production_data.x, &production_data.y)
//!     .backend(backend)
//!     .show()?;
//! ```
//!
//! # Module Organization
//!
//! - [`native`]: Native desktop/CLI rendering backend
//! - [`jupyter`]: Jupyter notebook inline display backend

mod native;
mod jupyter;

pub use native::NativeBackend;
pub use jupyter::JupyterBackend;

use crate::error::Result;
use crate::plot::types::{Backend, PlotBackend};

/// Creates appropriate backend instance based on specified type
/// 
/// Handles backend initialization with error handling and automatic
/// environment detection for `Backend::Auto`.
/// 
/// # Arguments
/// * `backend` - Backend type specification
/// 
/// # Returns
/// * `Result<Box<dyn PlotBackend>>` - Initialized backend or error
/// 
/// # Examples
/// ```rust
/// use rustlab_plotting::backend::create_backend;
/// use rustlab_plotting::plot::types::Backend;
/// 
/// // Explicit backend selection
/// let native_backend = create_backend(Backend::Native)?;
/// let jupyter_backend = create_backend(Backend::Jupyter)?;
/// 
/// // Automatic detection (recommended)
/// let auto_backend = create_backend(Backend::Auto)?;
/// ```
pub fn create_backend(backend: Backend) -> Result<Box<dyn PlotBackend>> {
    match backend {
        Backend::Native => Ok(Box::new(NativeBackend::new()?)),
        Backend::Jupyter => Ok(Box::new(JupyterBackend::new()?)),
        Backend::Auto => {
            // Check if we're in a Jupyter environment
            if is_jupyter() {
                Ok(Box::new(JupyterBackend::new()?))
            } else {
                Ok(Box::new(NativeBackend::new()?))
            }
        }
    }
}

/// Detects Jupyter notebook environment
/// 
/// Uses environment variables to determine if code is executing
/// within a Jupyter kernel context.
/// 
/// # Detection Strategy
/// 1. `JPY_PARENT_PID` - Jupyter parent process ID
/// 2. `JUPYTER_RUNTIME_DIR` - Jupyter runtime directory
/// 
/// # Returns
/// * `bool` - `true` if Jupyter environment detected
/// 
/// # Examples
/// ```rust
/// if is_jupyter() {
///     println!("Running in Jupyter notebook");
/// } else {
///     println!("Running in native environment");
/// }
/// ```
fn is_jupyter() -> bool {
    // Check for Jupyter kernel environment variables
    std::env::var("JPY_PARENT_PID").is_ok() || std::env::var("JUPYTER_RUNTIME_DIR").is_ok()
}