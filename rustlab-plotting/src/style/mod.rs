//! Visual styling and theming system for scientific plots
//!
//! Provides comprehensive theming capabilities for professional scientific
//! visualization. Includes predefined themes for different contexts (publication,
//! presentation, accessibility) and tools for custom styling.
//!
//! # For AI Code Generation
//!
//! ## Theme Selection Guide
//!
//! | Theme | Use Case | Color Profile | Accessibility |
//! |-------|----------|---------------|---------------|
//! | **Light** | General purpose, web display | High contrast on white | Good |
//! | **Dark** | Presentations, low-light viewing | Bright on dark background | Good |
//! | **Scientific** | Publications, academic papers | Conservative, professional | Excellent |
//! | **Colorblind Friendly** | Accessible visualization | Deuteranopia/Protanopia safe | Excellent |
//!
//! ## Color Theory Integration
//!
//! ### Perceptual Considerations
//! - **Luminance**: Brightness perception for readability
//! - **Contrast**: Text/background distinction
//! - **Saturation**: Color intensity for emphasis
//! - **Hue**: Color category for differentiation
//!
//! ### Accessibility Standards
//! - **WCAG AA**: Minimum 4.5:1 contrast ratio
//! - **WCAG AAA**: Enhanced 7:1 contrast ratio
//! - **Colorblind**: 8% of males, 0.5% of females affected
//! - **Monochrome**: Grayscale printing compatibility
//!
//! # Usage Examples
//!
//! ```rust
//! use rustlab_plotting::*;
//! use rustlab_plotting::style::theme::Theme;
//!
//! // Apply predefined themes
//! Plot::new()
//!     .line(&x, &y)
//!     .scientific_theme()        // Professional academic style
//!     .title("Research Results")
//!     .show()?;
//!
//! Plot::new()
//!     .scatter(&x, &y)
//!     .dark_theme()              // Dark background
//!     .title("Presentation Data")
//!     .show()?;
//!
//! Plot::new()
//!     .bar(&categories, &values)
//!     .colorblind_friendly_theme()  // Accessibility optimized
//!     .title("Survey Results")
//!     .show()?;
//!
//! // Custom theme configuration
//! let custom_theme = Theme {
//!     background_color: Color::from_hex("#F8F8FF")?,
//!     text_color: Color::from_hex("#191970")?,
//!     palette: vec![
//!         Color::from_hex("#4169E1")?,  // Royal blue
//!         Color::from_hex("#DC143C")?,  // Crimson
//!         Color::from_hex("#228B22")?,  // Forest green
//!     ],
//!     ..Theme::default()
//! };
//!
//! Plot::new()
//!     .line(&x, &y)
//!     .theme(custom_theme)
//!     .show()?;
//! ```
//!
//! # Color Palette Design
//!
//! ## Scientific Publications
//! ```rust
//! // High contrast, printer-friendly
//! let scientific_colors = vec![
//!     Color::from_hex("#000080")?,  // Navy blue
//!     Color::from_hex("#8B0000")?,  // Dark red  
//!     Color::from_hex("#006400")?,  // Dark green
//!     Color::from_hex("#FF8C00")?,  // Dark orange
//! ];
//! ```
//!
//! ## Presentations
//! ```rust
//! // Vibrant, high visibility
//! let presentation_colors = vec![
//!     Color::from_hex("#1E90FF")?,  // Dodger blue
//!     Color::from_hex("#FF4500")?,  // Orange red
//!     Color::from_hex("#32CD32")?,  // Lime green
//!     Color::from_hex("#FFD700")?,  // Gold
//! ];
//! ```
//!
//! ## Accessibility Focus
//! ```rust
//! // Colorblind-safe palette (Okabe-Ito)
//! let accessible_colors = vec![
//!     Color::from_hex("#0072B2")?,  // Blue
//!     Color::from_hex("#D55E00")?,  // Vermillion  
//!     Color::from_hex("#009E73")?,  // Bluish green
//!     Color::from_hex("#CC79A7")?,  // Reddish purple
//! ];
//! ```
//!
//! # Advanced Styling
//!
//! ```rust
//! // Multi-series with manual color control
//! PlotBuilder::new()
//!     .line(&x, &data1)
//!     .color(Color::BLUE)
//!     .style(LineStyle::Solid)
//!     .label("Experiment 1")
//!     
//!     .line(&x, &data2)
//!     .color(Color::RED)
//!     .style(LineStyle::Dashed)
//!     .label("Experiment 2")
//!     
//!     .line(&x, &theory)
//!     .color(Color::BLACK)
//!     .style(LineStyle::Dotted)
//!     .label("Theory")
//!     
//!     .build()
//!     .legend(true)
//!     .show()?;
//! ```
//!
//! # Theme Customization
//!
//! ```rust
//! use rustlab_plotting::style::theme::Theme;
//! use rustlab_plotting::plot::types::Color;
//!
//! // Create organization-specific theme
//! let corporate_theme = Theme {
//!     background_color: Color::from_hex("#FFFFFF")?,
//!     grid_color: Color::from_hex("#E5E5E5")?,
//!     text_color: Color::from_hex("#333333")?,
//!     palette: vec![
//!         Color::from_hex("#2E86AB")?,  // Corporate blue
//!         Color::from_hex("#A23B72")?,  // Corporate magenta
//!         Color::from_hex("#F18F01")?,  // Corporate orange
//!         Color::from_hex("#C73E1D")?,  // Corporate red
//!     ],
//!     name: "Corporate".to_string(),
//! };
//!
//! // Apply to plots
//! Plot::new()
//!     .theme(corporate_theme)
//!     .line(&quarterly_data.x, &quarterly_data.revenue)
//!     .title("Q4 Revenue Analysis")
//!     .show()?;
//! ```
//!
//! # Color Science Integration
//!
//! ## Perceptual Uniformity
//! ```rust
//! // Colors with equal perceptual difference
//! let perceptual_palette = vec![
//!     Color::from_hex("#1f77b4")?,  // Tab blue
//!     Color::from_hex("#ff7f0e")?,  // Tab orange  
//!     Color::from_hex("#2ca02c")?,  // Tab green
//!     Color::from_hex("#d62728")?,  // Tab red
//! ];
//! ```
//!
//! ## Print Compatibility
//! ```rust
//! // CMYK-safe colors for printing
//! let print_safe_palette = vec![
//!     Color::from_hex("#000080")?,  // Navy (C100 M100 Y0 K0)
//!     Color::from_hex("#800000")?,  // Maroon (C0 M100 Y100 K20)
//!     Color::from_hex("#008000")?,  // Green (C100 M0 Y100 K0)
//! ];
//! ```
//!
//! # Module Organization
//!
//! - [`color`]: Color manipulation and conversion utilities
//! - [`theme`]: Predefined themes and custom theme creation

pub mod color;
pub mod theme;