use crate::backend;
use crate::error::Result;
use crate::plot::types::*;
use crate::style::theme::Theme;
use rustlab_math::{VectorF64, linspace};
use std::path::Path;

impl Plot {
    pub fn new() -> Self {
        Self {
            config: PlotConfig::default(),
            series: Vec::new(),
            subplots: Vec::new(),
            heatmap_data: None,
            contour_data: None,
            surface3d_data: None,
        }
    }

    // Method chaining for plot configuration
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.config.title = Some(title.into());
        self
    }

    pub fn xlabel(mut self, label: impl Into<String>) -> Self {
        self.config.xlabel = Some(label.into());
        self
    }

    pub fn ylabel(mut self, label: impl Into<String>) -> Self {
        self.config.ylabel = Some(label.into());
        self
    }

    pub fn grid(mut self, enabled: bool) -> Self {
        self.config.grid = enabled;
        self
    }

    pub fn legend(mut self, enabled: bool) -> Self {
        self.config.legend = enabled;
        self
    }

    pub fn backend(mut self, backend: Backend) -> Self {
        self.config.backend = backend;
        self
    }

    pub fn xlim(mut self, min: f64, max: f64) -> Self {
        self.config.x_limits = Some((min, max));
        self
    }

    pub fn ylim(mut self, min: f64, max: f64) -> Self {
        self.config.y_limits = Some((min, max));
        self
    }

    pub fn xscale(mut self, scale: Scale) -> Self {
        self.config.x_scale = scale;
        self
    }

    pub fn yscale(mut self, scale: Scale) -> Self {
        self.config.y_scale = scale;
        self
    }

    pub fn theme(mut self, theme: Theme) -> Self {
        self.config.theme = theme;
        self
    }

    pub fn dark_theme(mut self) -> Self {
        self.config.theme = Theme::dark();
        self
    }

    pub fn scientific_theme(mut self) -> Self {
        self.config.theme = Theme::scientific();
        self
    }

    pub fn colorblind_friendly_theme(mut self) -> Self {
        self.config.theme = Theme::colorblind_friendly();
        self
    }

    // Subplot layout methods
    pub fn subplots(mut self, rows: usize, cols: usize) -> Self {
        self.config.subplot_layout = Some(SubplotLayout {
            rows,
            cols,
            current_subplot: 0,
        });
        self
    }

    pub fn subplot(mut self, row: usize, col: usize) -> SubplotBuilder {
        if self.config.subplot_layout.is_none() {
            // Default to a 1x1 layout if none specified
            self.config.subplot_layout = Some(SubplotLayout {
                rows: row.max(1),
                cols: col.max(1),
                current_subplot: 0,
            });
        }

        SubplotBuilder::new(self, row, col)
    }

    // Add series with rustlab-style API
    pub fn line(mut self, x: &VectorF64, y: &VectorF64) -> Self {
        self.add_series(x, y, PlotType::Line, None)
    }

    pub fn line_with(mut self, x: &VectorF64, y: &VectorF64, label: impl Into<String>) -> Self {
        self.add_series(x, y, PlotType::Line, Some(label.into()))
    }

    pub fn scatter(mut self, x: &VectorF64, y: &VectorF64) -> Self {
        self.add_series(x, y, PlotType::Scatter, None)
    }

    pub fn scatter_with(mut self, x: &VectorF64, y: &VectorF64, label: impl Into<String>) -> Self {
        self.add_series(x, y, PlotType::Scatter, Some(label.into()))
    }

    pub fn scatter_marker(mut self, x: &VectorF64, y: &VectorF64, marker: Marker) -> Self {
        let mut plot = self.add_series(x, y, PlotType::Scatter, None);
        if let Some(last_series) = plot.series.last_mut() {
            last_series.marker = Some(marker);
        }
        plot
    }

    pub fn scatter_marker_with(mut self, x: &VectorF64, y: &VectorF64, marker: Marker, label: impl Into<String>) -> Self {
        let mut plot = self.add_series(x, y, PlotType::Scatter, Some(label.into()));
        if let Some(last_series) = plot.series.last_mut() {
            last_series.marker = Some(marker);
        }
        plot
    }

    pub fn bar(mut self, x: &VectorF64, y: &VectorF64) -> Self {
        self.add_series(x, y, PlotType::Bar, None)
    }

    pub fn bar_with(mut self, x: &VectorF64, y: &VectorF64, label: impl Into<String>) -> Self {
        self.add_series(x, y, PlotType::Bar, Some(label.into()))
    }

    pub fn histogram(mut self, data: &VectorF64, bins: usize) -> Self {
        let (x, y) = compute_histogram(data, bins);
        self.add_series(&x, &y, PlotType::Histogram, None)
    }

    pub fn histogram_with(mut self, data: &VectorF64, bins: usize, label: impl Into<String>) -> Self {
        let (x, y) = compute_histogram(data, bins);
        self.add_series(&x, &y, PlotType::Histogram, Some(label.into()))
    }

    /// Add a contour plot using automatic level generation
    pub fn contour(mut self, x_grid: &VectorF64, y_grid: &VectorF64, z_data: &rustlab_math::ArrayF64) -> Result<Self> {
        let contour_builder = crate::contour::ContourBuilder::new(x_grid, y_grid, z_data)?;
        let contour_plot = contour_builder.build()?;
        self.contour_data = contour_plot.contour_data;
        Ok(self)
    }

    /// Add a contour plot with specific levels
    pub fn contour_levels(mut self, x_grid: &VectorF64, y_grid: &VectorF64, z_data: &rustlab_math::ArrayF64, levels: &[f64]) -> Result<Self> {
        let contour_builder = crate::contour::ContourBuilder::new(x_grid, y_grid, z_data)?
            .levels(levels);
        let contour_plot = contour_builder.build()?;
        self.contour_data = contour_plot.contour_data;
        Ok(self)
    }

    /// Add a filled contour plot
    pub fn contourf(mut self, x_grid: &VectorF64, y_grid: &VectorF64, z_data: &rustlab_math::ArrayF64) -> Result<Self> {
        let contour_builder = crate::contour::ContourBuilder::new(x_grid, y_grid, z_data)?
            .filled(true)
            .colorbar(true);
        let contour_plot = contour_builder.build()?;
        self.contour_data = contour_plot.contour_data;
        Ok(self)
    }

    // Display methods
    pub fn show(self) -> Result<()> {
        let mut backend = backend::create_backend(self.config.backend)?;
        backend.render(&self)?;
        backend.show()
    }

    pub fn save(self, path: impl AsRef<Path>) -> Result<()> {
        let mut backend = backend::create_backend(self.config.backend)?;
        backend.save(&self, path.as_ref())
    }

    // Internal helper methods
    fn add_series(mut self, x: &VectorF64, y: &VectorF64, plot_type: PlotType, label: Option<String>) -> Self {
        let color = self.get_next_color();
        let marker = match plot_type {
            PlotType::Scatter => Some(Marker::Circle),
            _ => Some(Marker::None),
        };
        self.series.push(Series {
            x: x.clone(),
            y: y.clone(),
            label,
            plot_type,
            color: Some(color),
            style: Some(LineStyle::Solid),
            marker,
            marker_size: 3.0,
        });
        self
    }

    fn get_next_color(&self) -> Color {
        self.config.theme.get_color(self.series.len())
    }
}

impl Default for Plot {
    fn default() -> Self {
        Self::new()
    }
}

// Helper function for histogram computation
fn compute_histogram(data: &VectorF64, bins: usize) -> (VectorF64, VectorF64) {
    // Get data as slice for efficient access
    let slice = data.as_slice().unwrap_or_else(|| {
        // If not contiguous, fall back to element-by-element access
        panic!("Non-contiguous vectors not supported for histogram computation");
    });
    
    let min = slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let bin_width = (max - min) / bins as f64;
    
    let mut counts = vec![0.0; bins];
    for &value in slice.iter() {
        let bin = ((value - min) / bin_width).floor() as usize;
        let bin = bin.min(bins - 1);
        counts[bin] += 1.0;
    }
    
    let x = linspace(min + bin_width / 2.0, max - bin_width / 2.0, bins);
    let y = VectorF64::from_vec(counts);
    
    (x, y)
}

// Builder pattern for advanced customization
pub struct PlotBuilder {
    plot: Plot,
    current_series: Option<Series>,
}

impl PlotBuilder {
    pub fn new() -> Self {
        Self {
            plot: Plot::new(),
            current_series: None,
        }
    }

    // Configuration methods delegate to Plot
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.plot = self.plot.size(width, height);
        self
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.plot = self.plot.title(title);
        self
    }

    pub fn xlabel(mut self, label: impl Into<String>) -> Self {
        self.plot = self.plot.xlabel(label);
        self
    }

    pub fn ylabel(mut self, label: impl Into<String>) -> Self {
        self.plot = self.plot.ylabel(label);
        self
    }

    // Advanced series customization
    pub fn line(mut self, x: &VectorF64, y: &VectorF64) -> Self {
        self.finalize_current_series();
        self.current_series = Some(Series {
            x: x.clone(),
            y: y.clone(),
            label: None,
            plot_type: PlotType::Line,
            color: None,
            style: None,
            marker: Some(Marker::None),
            marker_size: 3.0,
        });
        self
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        if let Some(ref mut series) = self.current_series {
            series.label = Some(label.into());
        }
        self
    }

    pub fn color(mut self, color: Color) -> Self {
        if let Some(ref mut series) = self.current_series {
            series.color = Some(color);
        }
        self
    }

    pub fn style(mut self, style: LineStyle) -> Self {
        if let Some(ref mut series) = self.current_series {
            series.style = Some(style);
        }
        self
    }

    pub fn marker(mut self, marker: Marker) -> Self {
        if let Some(ref mut series) = self.current_series {
            series.marker = Some(marker);
        }
        self
    }

    pub fn marker_size(mut self, size: f32) -> Self {
        if let Some(ref mut series) = self.current_series {
            series.marker_size = size;
        }
        self
    }

    pub fn build(mut self) -> Plot {
        self.finalize_current_series();
        self.plot
    }

    fn finalize_current_series(&mut self) {
        if let Some(mut series) = self.current_series.take() {
            if series.color.is_none() {
                series.color = Some(self.plot.get_next_color());
            }
            if series.style.is_none() {
                series.style = Some(LineStyle::Solid);
            }
            self.plot.series.push(series);
        }
    }
}

// Subplot builder for individual subplot configuration
pub struct SubplotBuilder {
    main_plot: Plot,
    subplot_row: usize,
    subplot_col: usize,
    subplot_config: PlotConfig,
    subplot_series: Vec<Series>,
}

impl SubplotBuilder {
    fn new(plot: Plot, row: usize, col: usize) -> Self {
        let mut subplot_config = plot.config.clone();
        // Reset subplot-specific settings
        subplot_config.title = None;
        subplot_config.xlabel = None;
        subplot_config.ylabel = None;
        subplot_config.legend = false;
        
        Self {
            main_plot: plot,
            subplot_row: row,
            subplot_col: col,
            subplot_config,
            subplot_series: Vec::new(),
        }
    }

    // Configuration methods for the current subplot
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.subplot_config.title = Some(title.into());
        self
    }

    pub fn xlabel(mut self, label: impl Into<String>) -> Self {
        self.subplot_config.xlabel = Some(label.into());
        self
    }

    pub fn ylabel(mut self, label: impl Into<String>) -> Self {
        self.subplot_config.ylabel = Some(label.into());
        self
    }

    pub fn grid(mut self, enabled: bool) -> Self {
        self.subplot_config.grid = enabled;
        self
    }

    pub fn legend(mut self, enabled: bool) -> Self {
        self.subplot_config.legend = enabled;
        self
    }

    pub fn xlim(mut self, min: f64, max: f64) -> Self {
        self.subplot_config.x_limits = Some((min, max));
        self
    }

    pub fn ylim(mut self, min: f64, max: f64) -> Self {
        self.subplot_config.y_limits = Some((min, max));
        self
    }

    pub fn xscale(mut self, scale: Scale) -> Self {
        self.subplot_config.x_scale = scale;
        self
    }

    pub fn yscale(mut self, scale: Scale) -> Self {
        self.subplot_config.y_scale = scale;
        self
    }

    // Series methods for the current subplot
    pub fn line(mut self, x: &VectorF64, y: &VectorF64) -> Self {
        self.add_series(x, y, PlotType::Line, None)
    }

    pub fn line_with(mut self, x: &VectorF64, y: &VectorF64, label: impl Into<String>) -> Self {
        self.add_series(x, y, PlotType::Line, Some(label.into()))
    }

    pub fn scatter(mut self, x: &VectorF64, y: &VectorF64) -> Self {
        self.add_series(x, y, PlotType::Scatter, None)
    }

    pub fn scatter_with(mut self, x: &VectorF64, y: &VectorF64, label: impl Into<String>) -> Self {
        self.add_series(x, y, PlotType::Scatter, Some(label.into()))
    }

    pub fn scatter_marker(mut self, x: &VectorF64, y: &VectorF64, marker: Marker) -> Self {
        let mut subplot_builder = self.add_series(x, y, PlotType::Scatter, None);
        if let Some(last_series) = subplot_builder.subplot_series.last_mut() {
            last_series.marker = Some(marker);
        }
        subplot_builder
    }

    pub fn scatter_marker_with(mut self, x: &VectorF64, y: &VectorF64, marker: Marker, label: impl Into<String>) -> Self {
        let mut subplot_builder = self.add_series(x, y, PlotType::Scatter, Some(label.into()));
        if let Some(last_series) = subplot_builder.subplot_series.last_mut() {
            last_series.marker = Some(marker);
        }
        subplot_builder
    }

    pub fn bar(mut self, x: &VectorF64, y: &VectorF64) -> Self {
        self.add_series(x, y, PlotType::Bar, None)
    }

    pub fn bar_with(mut self, x: &VectorF64, y: &VectorF64, label: impl Into<String>) -> Self {
        self.add_series(x, y, PlotType::Bar, Some(label.into()))
    }

    pub fn histogram(mut self, data: &VectorF64, bins: usize) -> Self {
        let (x, y) = compute_histogram(data, bins);
        self.add_series(&x, &y, PlotType::Histogram, None)
    }

    pub fn histogram_with(mut self, data: &VectorF64, bins: usize, label: impl Into<String>) -> Self {
        let (x, y) = compute_histogram(data, bins);
        self.add_series(&x, &y, PlotType::Histogram, Some(label.into()))
    }

    // Return to the main plot
    pub fn build(mut self) -> Plot {
        // Create the subplot and add it to the main plot
        let subplot = Subplot {
            config: self.subplot_config,
            series: self.subplot_series,
            position: SubplotPosition {
                row: self.subplot_row,
                col: self.subplot_col,
            },
        };
        
        self.main_plot.subplots.push(subplot);
        self.main_plot
    }

    // Internal helper methods
    fn add_series(mut self, x: &VectorF64, y: &VectorF64, plot_type: PlotType, label: Option<String>) -> Self {
        let color = self.get_next_color();
        let marker = match plot_type {
            PlotType::Scatter => Some(Marker::Circle),
            _ => Some(Marker::None),
        };
        self.subplot_series.push(Series {
            x: x.clone(),
            y: y.clone(),
            label,
            plot_type,
            color: Some(color),
            style: Some(LineStyle::Solid),
            marker,
            marker_size: 3.0,
        });
        self
    }

    fn get_next_color(&self) -> Color {
        self.subplot_config.theme.get_color(self.subplot_series.len())
    }
}