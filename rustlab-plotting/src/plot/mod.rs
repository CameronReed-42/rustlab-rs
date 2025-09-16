//! Core plotting functionality with builder pattern and mathematical integration
//!
//! This module provides the foundation for scientific plotting with rustlab-math
//! integration. Uses builder pattern for fluent API design and supports multiple
//! chart types, subplot layouts, and styling options.
//!
//! # For AI Code Generation
//!
//! ## Builder Pattern Structure
//! - **Plot**: Main plotting interface with fluent API
//! - **PlotBuilder**: Advanced customization with method chaining
//! - **SubplotBuilder**: Individual subplot configuration
//! - **PlotConfig**: Configuration and styling parameters
//!
//! ## Core Types
//! - **Series**: Individual data series (line, scatter, bar, histogram)
//! - **Backend**: Rendering engines (Native, Jupyter, file export)
//! - **Scale**: Axis scaling (Linear, Log10, Log2, Ln)
//! - **Theme**: Visual styling and color schemes
//!
//! # Example Usage
//!
//! ```rust
//! use rustlab_plotting::plot::*;
//! use rustlab_math::{range, VectorMathOps};
//!
//! // Simple plotting with fluent API
//! let x = range!(0.0 => 10.0, 100);
//! let y = x.sin();
//!
//! Plot::new()
//!     .line(&x, &y)
//!     .title("Sine Function")
//!     .xlabel("x")
//!     .ylabel("sin(x)")
//!     .grid(true)
//!     .show()?;
//!
//! // Advanced customization with builder
//! PlotBuilder::new()
//!     .size(1200, 800)
//!     .line(&x, &y)
//!     .color(Color::BLUE)
//!     .style(LineStyle::Dashed)
//!     .label("sin(x)")
//!     .build()
//!     .show()?;
//!
//! // Multi-panel subplot layout
//! Plot::new()
//!     .subplots(2, 2)
//!     .subplot(0, 0)
//!         .line(&x, &y)
//!         .title("Subplot 1")
//!         .build()
//!     .subplot(0, 1)
//!         .scatter(&x, &y)
//!         .title("Subplot 2")
//!         .build()
//!     .show()?;
//! ```
//!
//! # Module Organization
//!
//! - [`types`]: Core data structures and enums
//! - [`builder`]: Builder pattern implementations
//! - [`validation`]: Input validation and error checking

pub mod types;
pub mod builder;
pub mod validation;

pub use types::*;
pub use builder::*;
pub use validation::*;