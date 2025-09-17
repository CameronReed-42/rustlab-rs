use crate::error::{PlottingError, Result};
use crate::plot::types::{Plot, PlotBackend, PlotType, LineStyle, Color as PlotColor, Scale, Marker};
use crate::colormap::ColorMapper;
use crate::plot::validation::validate_plot_data;
use crate::math_integration::VectorDataExt;
use plotters::prelude::*;
use std::path::Path;

// Macro to reduce code duplication for chart drawing
macro_rules! draw_chart_content {
    ($chart:expr, $plot:expr, $text_color:expr, $x_min:expr, $x_max:expr, $y_min:expr, $y_max:expr) => {{
        // Configure mesh (grid) with theme colors
        $chart.configure_mesh()
            .axis_style(&$text_color)
            .label_style(("sans-serif", 12, &$text_color))
            .x_desc($plot.config.xlabel.as_deref().unwrap_or(""))
            .y_desc($plot.config.ylabel.as_deref().unwrap_or(""))
            .draw()
            .map_err(|e| PlottingError::Plotters(e.to_string()))?;
        
        // Draw each series
        for (series_index, series) in $plot.series.iter().enumerate() {
            let color = series.color.as_ref()
                .map(|c| RGBColor(c.r, c.g, c.b))
                .unwrap_or_else(|| {
                    let theme_color = $plot.config.theme.get_color(series_index);
                    RGBColor(theme_color.r, theme_color.g, theme_color.b)
                });
            
            match series.plot_type {
                PlotType::Line => {
                    let data: Vec<(f64, f64)> = series.x.data().iter()
                        .zip(series.y.data().iter())
                        .map(|(&x, &y)| (x, y))
                        .collect();
                    
                    $chart.draw_series(LineSeries::new(
                        data,
                        color.stroke_width(2),
                    ))
                    .map_err(|e| PlottingError::Plotters(e.to_string()))?
                    .label(series.label.as_deref().unwrap_or(""))
                    .legend(move |(x, y)| Rectangle::new([(x, y), (x + 10, y)], color.filled()));
                }
                PlotType::Scatter => {
                    let data: Vec<(f64, f64)> = series.x.data().iter()
                        .zip(series.y.data().iter())
                        .map(|(&x, &y)| (x, y))
                        .collect();
                    
                    let marker_size = series.marker_size as i32;
                    let marker = series.marker.unwrap_or(Marker::Circle);
                    
                    match marker {
                        Marker::None => {
                            // Don't draw anything for no marker
                        }
                        Marker::Square => {
                            // Use a simple fixed size relative to data range
                            let data_range_x = $x_max - $x_min;
                            let data_range_y = $y_max - $y_min;
                            let square_size = (marker_size as f64 / 5.0) * (data_range_x.min(data_range_y) / 50.0);
                            
                            $chart.draw_series(data.iter().map(|&(x, y)| {
                                Rectangle::new([(x - square_size/2.0, y - square_size/2.0), 
                                               (x + square_size/2.0, y + square_size/2.0)], color.filled())
                            }))
                            .map_err(|e| PlottingError::Plotters(e.to_string()))?
                            .label(series.label.as_deref().unwrap_or(""))
                            .legend(move |(x, y)| Rectangle::new([(x, y), (x + 10, y + 10)], color.filled()));
                        }
                        Marker::Circle => {
                            $chart.draw_series(data.iter().map(|&(x, y)| {
                                Circle::new((x, y), marker_size, color.filled())
                            }))
                            .map_err(|e| PlottingError::Plotters(e.to_string()))?
                            .label(series.label.as_deref().unwrap_or(""))
                            .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
                        }
                        Marker::Triangle => {
                            use plotters::element::Polygon;
                            let data_range_x = $x_max - $x_min;
                            let data_range_y = $y_max - $y_min;
                            let tri_size = (marker_size as f64 / 6.0) * (data_range_x.min(data_range_y) / 50.0);
                            
                            $chart.draw_series(data.iter().map(|&(x, y)| {
                                Polygon::new(vec![
                                    (x, y + tri_size * 0.866),           // Top (height = size * sqrt(3)/2)
                                    (x - tri_size * 0.5, y - tri_size * 0.433), // Bottom left
                                    (x + tri_size * 0.5, y - tri_size * 0.433), // Bottom right
                                ], color.filled())
                            }))
                            .map_err(|e| PlottingError::Plotters(e.to_string()))?
                            .label(series.label.as_deref().unwrap_or(""))
                            .legend(move |(x, y)| Polygon::new(vec![(x+5, y-3), (x, y+3), (x+10, y+3)], color.filled()));
                        }
                        Marker::Diamond => {
                            use plotters::element::Polygon;
                            let data_range_x = $x_max - $x_min;
                            let data_range_y = $y_max - $y_min;
                            let diamond_size = (marker_size as f64 / 6.0) * (data_range_x.min(data_range_y) / 50.0);
                            
                            $chart.draw_series(data.iter().map(|&(x, y)| {
                                Polygon::new(vec![
                                    (x, y + diamond_size * 0.7),        // Top (narrower)
                                    (x - diamond_size * 0.5, y),        // Left
                                    (x, y - diamond_size * 0.7),        // Bottom (narrower)
                                    (x + diamond_size * 0.5, y),        // Right
                                ], color.filled())
                            }))
                            .map_err(|e| PlottingError::Plotters(e.to_string()))?
                            .label(series.label.as_deref().unwrap_or(""))
                            .legend(move |(x, y)| Polygon::new(vec![(x+5, y-3), (x, y), (x+5, y+3), (x+10, y)], color.filled()));
                        }
                        Marker::Cross => {
                            use plotters::element::PathElement;
                            let data_range_x = $x_max - $x_min;
                            let data_range_y = $y_max - $y_min;
                            let cross_size = (marker_size as f64 / 7.0) * (data_range_x.min(data_range_y) / 50.0);
                            
                            // Draw X marks using two lines
                            $chart.draw_series(data.iter().flat_map(|&(x, y)| {
                                vec![
                                    PathElement::new(vec![
                                        (x - cross_size * 0.7, y - cross_size * 0.7),
                                        (x + cross_size * 0.7, y + cross_size * 0.7),
                                    ], color.stroke_width(2)),
                                    PathElement::new(vec![
                                        (x - cross_size * 0.7, y + cross_size * 0.7),
                                        (x + cross_size * 0.7, y - cross_size * 0.7),
                                    ], color.stroke_width(2)),
                                ]
                            }))
                            .map_err(|e| PlottingError::Plotters(e.to_string()))?
                            .label(series.label.as_deref().unwrap_or(""))
                            .legend(move |(x, y)| PathElement::new(vec![(x, y-3), (x+10, y+3)], color.stroke_width(2)));
                        }
                        Marker::Plus => {
                            use plotters::element::PathElement;
                            let data_range_x = $x_max - $x_min;
                            let data_range_y = $y_max - $y_min;
                            let plus_size = (marker_size as f64 / 7.0) * (data_range_x.min(data_range_y) / 50.0);
                            
                            // Draw + marks using two lines
                            $chart.draw_series(data.iter().flat_map(|&(x, y)| {
                                vec![
                                    PathElement::new(vec![
                                        (x - plus_size * 0.7, y),
                                        (x + plus_size * 0.7, y),
                                    ], color.stroke_width(2)),
                                    PathElement::new(vec![
                                        (x, y - plus_size * 0.7),
                                        (x, y + plus_size * 0.7),
                                    ], color.stroke_width(2)),
                                ]
                            }))
                            .map_err(|e| PlottingError::Plotters(e.to_string()))?
                            .label(series.label.as_deref().unwrap_or(""))
                            .legend(move |(x, y)| PathElement::new(vec![(x+5, y-3), (x+5, y+3)], color.stroke_width(2)));
                        }
                    }
                }
                PlotType::Bar => {
                    let data: Vec<(f64, f64)> = series.x.data().iter()
                        .zip(series.y.data().iter())
                        .map(|(&x, &y)| (x, y))
                        .collect();
                    
                    let bar_width = if data.len() > 1 {
                        (data[1].0 - data[0].0) * 0.8
                    } else {
                        ($x_max - $x_min) * 0.1
                    };
                    
                    $chart.draw_series(data.iter().map(|&(x, y)| {
                        Rectangle::new([(x - bar_width/2.0, 0.0), (x + bar_width/2.0, y)], color.filled())
                    }))
                    .map_err(|e| PlottingError::Plotters(e.to_string()))?
                    .label(series.label.as_deref().unwrap_or(""))
                    .legend(move |(x, y)| Rectangle::new([(x, y), (x + 10, y + 10)], color.filled()));
                }
                PlotType::Histogram => {
                    let data: Vec<(f64, f64)> = series.x.data().iter()
                        .zip(series.y.data().iter())
                        .map(|(&x, &y)| (x, y))
                        .collect();
                    
                    let bar_width = if data.len() > 1 {
                        data[1].0 - data[0].0
                    } else {
                        ($x_max - $x_min) * 0.1
                    };
                    
                    $chart.draw_series(data.iter().map(|&(x, y)| {
                        Rectangle::new([(x - bar_width/2.0, 0.0), (x + bar_width/2.0, y)], color.filled())
                    }))
                    .map_err(|e| PlottingError::Plotters(e.to_string()))?
                    .label(series.label.as_deref().unwrap_or(""))
                    .legend(move |(x, y)| Rectangle::new([(x, y), (x + 10, y + 10)], color.filled()));
                }
                PlotType::Heatmap | PlotType::Contour | PlotType::Surface3D => {
                    // These plot types are handled separately, not through series
                    // This match arm exists only to satisfy the compiler
                }
            }
        }
        
        // Draw legend if enabled and there are labeled series
        if $plot.config.legend && $plot.series.iter().any(|s| s.label.is_some()) {
            let bg_color = RGBColor(
                $plot.config.theme.background_color.r,
                $plot.config.theme.background_color.g,
                $plot.config.theme.background_color.b
            );
            $chart.configure_series_labels()
                .background_style(&bg_color.mix(0.8))
                .border_style(&$text_color)
                .label_font(("sans-serif", 12, &$text_color))
                .draw()
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
        }
    }};
}

// Subplot chart renderer (without present() call)
fn render_subplot_chart<DB: DrawingBackend>(root: &DrawingArea<DB, plotters::coord::Shift>, plot: &Plot) -> Result<()>
where
    DB::ErrorType: 'static,
{
    // Validate data compatibility with scales
    validate_plot_data(plot)?;
    // Calculate data ranges
    let (x_min, x_max) = if let Some((min, max)) = plot.config.x_limits {
        (min, max)
    } else {
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        
        for series in &plot.series {
            for &x in series.x.data().iter() {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
            }
        }
        
        let x_range = x_max - x_min;
        (x_min - x_range * 0.05, x_max + x_range * 0.05)
    };

    let (y_min, y_max) = if let Some((min, max)) = plot.config.y_limits {
        (min, max)
    } else {
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        
        for series in &plot.series {
            for &y in series.y.data().iter() {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
        
        let y_range = y_max - y_min;
        (y_min - y_range * 0.05, y_max + y_range * 0.05)
    };
    
    // Apply theme text and background colors
    let text_color = RGBColor(
        plot.config.theme.text_color.r,
        plot.config.theme.text_color.g,
        plot.config.theme.text_color.b
    );
    
    // Create chart with appropriate coordinate system based on scales
    let mut chart_builder = ChartBuilder::on(root);
    chart_builder
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50);
    
    if let Some(title) = &plot.config.title {
        chart_builder.caption(title, ("sans-serif", 20, &text_color)); // Smaller font for subplots
    }
    
    // Build chart with appropriate scale combination
    match (plot.config.x_scale, plot.config.y_scale) {
        (Scale::Linear, Scale::Linear) => {
            let mut chart = chart_builder
                .build_cartesian_2d(x_min..x_max, y_min..y_max)
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            draw_chart_content!(chart, plot, text_color, x_min, x_max, y_min, y_max);
        }
        (Scale::Log10, Scale::Linear) => {
            // Ensure positive x values for log scale
            let x_min_log = if x_min <= 0.0 { 1e-10 } else { x_min };
            let x_max_log = if x_max <= 0.0 { 10.0 } else { x_max };
            
            let mut chart = chart_builder
                .build_cartesian_2d((x_min_log..x_max_log).log_scale(), y_min..y_max)
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            draw_chart_content!(chart, plot, text_color, x_min, x_max, y_min, y_max);
        }
        (Scale::Linear, Scale::Log10) => {
            // Ensure positive y values for log scale
            let y_min_log = if y_min <= 0.0 { 1e-10 } else { y_min };
            let y_max_log = if y_max <= 0.0 { 10.0 } else { y_max };
            
            let mut chart = chart_builder
                .build_cartesian_2d(x_min..x_max, (y_min_log..y_max_log).log_scale())
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            draw_chart_content!(chart, plot, text_color, x_min, x_max, y_min, y_max);
        }
        (Scale::Log10, Scale::Log10) => {
            // Ensure positive values for both log scales
            let x_min_log = if x_min <= 0.0 { 1e-10 } else { x_min };
            let x_max_log = if x_max <= 0.0 { 10.0 } else { x_max };
            let y_min_log = if y_min <= 0.0 { 1e-10 } else { y_min };
            let y_max_log = if y_max <= 0.0 { 10.0 } else { y_max };
            
            let mut chart = chart_builder
                .build_cartesian_2d((x_min_log..x_max_log).log_scale(), (y_min_log..y_max_log).log_scale())
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            draw_chart_content!(chart, plot, text_color, x_min, x_max, y_min, y_max);
        }
        _ => {
            // For now, treat Log2 and Ln as Linear until we implement them
            let mut chart = chart_builder
                .build_cartesian_2d(x_min..x_max, y_min..y_max)
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            draw_chart_content!(chart, plot, text_color, x_min, x_max, y_min, y_max);
        }
    }
    
    // NOTE: No present() call for subplots - this will be called once on the main root
    Ok(())
}

// Standalone legacy chart renderer (shared with native backend)
fn render_legacy_chart<DB: DrawingBackend>(root: &DrawingArea<DB, plotters::coord::Shift>, plot: &Plot) -> Result<()>
where
    DB::ErrorType: 'static,
{
    // Validate data compatibility with scales
    validate_plot_data(plot)?;
    // Calculate data ranges
    let (x_min, x_max) = if let Some((min, max)) = plot.config.x_limits {
        (min, max)
    } else {
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        
        for series in &plot.series {
            for &x in series.x.data().iter() {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
            }
        }
        
        let x_range = x_max - x_min;
        (x_min - x_range * 0.05, x_max + x_range * 0.05)
    };

    let (y_min, y_max) = if let Some((min, max)) = plot.config.y_limits {
        (min, max)
    } else {
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        
        for series in &plot.series {
            for &y in series.y.data().iter() {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
        
        let y_range = y_max - y_min;
        (y_min - y_range * 0.05, y_max + y_range * 0.05)
    };
    
    // Apply theme text and background colors
    let text_color = RGBColor(
        plot.config.theme.text_color.r,
        plot.config.theme.text_color.g,
        plot.config.theme.text_color.b
    );
    
    // Create chart with appropriate coordinate system based on scales
    let mut chart_builder = ChartBuilder::on(root);
    chart_builder
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50);
    
    if let Some(title) = &plot.config.title {
        chart_builder.caption(title, ("sans-serif", 40, &text_color));
    }
    
    // Build chart with appropriate scale combination
    match (plot.config.x_scale, plot.config.y_scale) {
        (Scale::Linear, Scale::Linear) => {
            let mut chart = chart_builder
                .build_cartesian_2d(x_min..x_max, y_min..y_max)
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            draw_chart_content!(chart, plot, text_color, x_min, x_max, y_min, y_max);
        }
        (Scale::Log10, Scale::Linear) => {
            // Ensure positive x values for log scale
            let x_min_log = if x_min <= 0.0 { 1e-10 } else { x_min };
            let x_max_log = if x_max <= 0.0 { 10.0 } else { x_max };
            
            let mut chart = chart_builder
                .build_cartesian_2d((x_min_log..x_max_log).log_scale(), y_min..y_max)
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            draw_chart_content!(chart, plot, text_color, x_min, x_max, y_min, y_max);
        }
        (Scale::Linear, Scale::Log10) => {
            // Ensure positive y values for log scale
            let y_min_log = if y_min <= 0.0 { 1e-10 } else { y_min };
            let y_max_log = if y_max <= 0.0 { 10.0 } else { y_max };
            
            let mut chart = chart_builder
                .build_cartesian_2d(x_min..x_max, (y_min_log..y_max_log).log_scale())
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            draw_chart_content!(chart, plot, text_color, x_min, x_max, y_min, y_max);
        }
        (Scale::Log10, Scale::Log10) => {
            // Ensure positive values for both log scales
            let x_min_log = if x_min <= 0.0 { 1e-10 } else { x_min };
            let x_max_log = if x_max <= 0.0 { 10.0 } else { x_max };
            let y_min_log = if y_min <= 0.0 { 1e-10 } else { y_min };
            let y_max_log = if y_max <= 0.0 { 10.0 } else { y_max };
            
            let mut chart = chart_builder
                .build_cartesian_2d((x_min_log..x_max_log).log_scale(), (y_min_log..y_max_log).log_scale())
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            draw_chart_content!(chart, plot, text_color, x_min, x_max, y_min, y_max);
        }
        _ => {
            // For now, treat Log2 and Ln as Linear until we implement them
            let mut chart = chart_builder
                .build_cartesian_2d(x_min..x_max, y_min..y_max)
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            draw_chart_content!(chart, plot, text_color, x_min, x_max, y_min, y_max);
        }
    }
    
    root.present().map_err(|e| PlottingError::Plotters(e.to_string()))?;
    Ok(())
}

// Standalone subplot renderer (shared with native backend)
fn render_subplots<DB: DrawingBackend>(root: &DrawingArea<DB, plotters::coord::Shift>, plot: &Plot) -> Result<()>
where
    DB::ErrorType: 'static,
{
    if let Some(layout) = &plot.config.subplot_layout {
        let rows = layout.rows;
        let cols = layout.cols;
        
        // Split the main area into grid
        let subplot_areas = root.margin(5, 5, 5, 5).split_evenly((rows, cols));
        
        // Render each subplot
        for subplot in &plot.subplots {
            let row = subplot.position.row;
            let col = subplot.position.col;
            
            // Validate subplot position
            if row >= rows || col >= cols {
                return Err(PlottingError::Backend(format!(
                    "Invalid subplot position ({}, {}) for {}x{} grid", row, col, rows, cols
                )));
            }
            
            // Get the correct subplot area (row-major order)
            let linear_index = row * cols + col;
            let subplot_area = &subplot_areas[linear_index];
            
            // Create a temporary plot with the subplot data
            let temp_plot = Plot {
                config: subplot.config.clone(),
                series: subplot.series.clone(),
                subplots: Vec::new(), // No nested subplots
                heatmap_data: None,  // Subplots don't contain heatmaps
                contour_data: None,  // Subplots don't contain contours
                surface3d_data: None,  // Subplots don't contain 3D surfaces
            };
            
            // Render the subplot (without calling present() on individual areas)
            render_subplot_chart(subplot_area, &temp_plot)?;
        }
    }
    Ok(())
}

pub struct JupyterBackend {
    svg_buffer: String,
    width: u32,
    height: u32,
}

impl JupyterBackend {
    pub fn new() -> Result<Self> {
        Ok(Self {
            svg_buffer: String::new(),
            width: 800,
            height: 600,
        })
    }

    #[allow(dead_code)]
    fn get_color(&self, color: &PlotColor) -> RGBColor {
        RGBColor(color.r, color.g, color.b)
    }

    #[allow(dead_code)]
    fn get_line_style(&self, style: &LineStyle) -> ShapeStyle {
        match style {
            LineStyle::Solid => ShapeStyle::from(&BLACK).stroke_width(2),
            LineStyle::Dashed => ShapeStyle::from(&BLACK).stroke_width(2), // TODO: Add dashed support
            LineStyle::Dotted => ShapeStyle::from(&BLACK).stroke_width(2), // TODO: Add dotted support
        }
    }
}

// Heatmap renderer for Jupyter backend
// Contour plot renderer for Jupyter backend (shared with Native)
fn render_contour<DB: DrawingBackend>(root: &DrawingArea<DB, plotters::coord::Shift>, plot: &Plot) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let contour_data = plot.contour_data.as_ref()
        .ok_or_else(|| PlottingError::InvalidData("No contour data found".to_string()))?;
    
    // Get data bounds
    let x_min = contour_data.x_grid.min().unwrap();
    let x_max = contour_data.x_grid.max().unwrap();
    let y_min = contour_data.y_grid.min().unwrap();
    let y_max = contour_data.y_grid.max().unwrap();
    
    // Create chart context
    let mut chart = ChartBuilder::on(root)
        .caption(plot.config.title.as_deref().unwrap_or("Contour Plot"), ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|e| PlottingError::Plotters(e.to_string()))?;
    
    // Configure mesh
    chart.configure_mesh()
        .x_desc(plot.config.xlabel.as_deref().unwrap_or("x"))
        .y_desc(plot.config.ylabel.as_deref().unwrap_or("y"))
        .draw()
        .map_err(|e| PlottingError::Plotters(e.to_string()))?;
    
    // Draw contour lines
    for contour_line in &contour_data.contour_lines {
        let color = RGBColor(contour_line.color.r, contour_line.color.g, contour_line.color.b);
        let line_data: Vec<(f64, f64)> = contour_line.x.iter()
            .zip(contour_line.y.iter())
            .map(|(&x, &y)| (x, y))
            .collect();
            
        if !line_data.is_empty() {
            chart.draw_series(LineSeries::new(line_data, color.stroke_width(2)))
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
        }
    }
    
    Ok(())
}

fn render_heatmap<DB: DrawingBackend>(root: &DrawingArea<DB, plotters::coord::Shift>, plot: &Plot) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let heatmap_data = plot.heatmap_data.as_ref()
        .ok_or_else(|| PlottingError::InvalidData("No heatmap data found".to_string()))?;
    
    let matrix = &heatmap_data.matrix;
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    
    // Calculate data range for color mapping
    let (min_val, max_val) = if let (Some(vmin), Some(vmax)) = (heatmap_data.vmin, heatmap_data.vmax) {
        (vmin, vmax)
    } else {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        
        for i in 0..rows {
            for j in 0..cols {
                if let Some(val) = matrix.get(i, j) {
                    if val.is_finite() {
                        min = min.min(val);
                        max = max.max(val);
                    }
                }
            }
        }
        
        (
            heatmap_data.vmin.unwrap_or(min),
            heatmap_data.vmax.unwrap_or(max)
        )
    };
    
    // Create color mapper
    let color_mapper = crate::colormap::ColorMapper::new(heatmap_data.colormap, min_val, max_val);
    
    // Calculate cell size based on available space
    let margin = 50;
    let available_width = (plot.config.width as i32 - 2 * margin).max(100);
    let available_height = (plot.config.height as i32 - 2 * margin).max(100);
    
    let cell_width = (available_width / cols as i32).max(1);
    let cell_height = (available_height / rows as i32).max(1);
    
    // Draw the heatmap cells
    for i in 0..rows {
        for j in 0..cols {
            let value = match matrix.get(i, j) {
                Some(val) if val.is_finite() => val,
                _ => continue, // Skip NaN/infinite/missing values
            };
            
            let color = color_mapper.map(value);
            let plot_color = plotters::prelude::RGBColor(color.r, color.g, color.b);
            
            // Calculate cell coordinates (flip i because plotters uses bottom-up coordinates)
            let x1 = margin + j as i32 * cell_width;
            let y1 = margin + (rows - 1 - i) as i32 * cell_height;
            let x2 = x1 + cell_width;
            let y2 = y1 + cell_height;
            
            // Draw the cell as a filled rectangle
            root.draw(&plotters::prelude::Rectangle::new([(x1, y1), (x2, y2)], plot_color.filled()))
                .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            
            // Draw cell value if requested
            if heatmap_data.show_values && cell_width > 20 && cell_height > 20 {
                let _text_color = if value > (min_val + max_val) / 2.0 {
                    plotters::prelude::BLACK
                } else {
                    plotters::prelude::WHITE
                };
                
                let text = if value.abs() < 10.0 {
                    format!("{:.1}", value)
                } else {
                    format!("{:.0}", value)
                };
                
                root.draw(&plotters::prelude::Text::new(
                    text,
                    (x1 + cell_width / 2, y1 + cell_height / 2),
                    ("sans-serif", (cell_height.min(cell_width) / 4).max(8) as u32)
                )).map_err(|e| PlottingError::Plotters(e.to_string()))?;
            }
        }
    }
    
    // Draw title if present
    if let Some(ref title) = plot.config.title {
        let _text_color = plotters::prelude::RGBColor(
            plot.config.theme.text_color.r,
            plot.config.theme.text_color.g,
            plot.config.theme.text_color.b
        );
        
        root.draw(&plotters::prelude::Text::new(
            title.clone(),
            (plot.config.width as i32 / 2, 20),
            ("sans-serif", 16)
        )).map_err(|e| PlottingError::Plotters(e.to_string()))?;
    }
    
    Ok(())
}

// 3D Surface renderer for Jupyter backend  
fn render_surface3d<DB: DrawingBackend>(root: &DrawingArea<DB, plotters::coord::Shift>, plot: &Plot) -> Result<()>
where
    DB::ErrorType: 'static,
{
    use plotters::prelude::*;
    
    let surface_data = plot.surface3d_data.as_ref()
        .ok_or_else(|| PlottingError::InvalidData("No 3D surface data found".to_string()))?;
    
    // Get data bounds
    let x_min = surface_data.x_grid.min().unwrap_or(0.0);
    let x_max = surface_data.x_grid.max().unwrap_or(1.0);
    let y_min = surface_data.y_grid.min().unwrap_or(0.0);
    let y_max = surface_data.y_grid.max().unwrap_or(1.0);
    let z_min = surface_data.z_data.min().unwrap_or(0.0) * surface_data.z_scale;
    let z_max = surface_data.z_data.max().unwrap_or(1.0) * surface_data.z_scale;
    
    // Create enhanced title with axis labels
    let xlabel = plot.config.xlabel.as_deref().unwrap_or("X");
    let ylabel = plot.config.ylabel.as_deref().unwrap_or("Y");
    let zlabel = surface_data.zlabel.as_deref().unwrap_or("Z");
    
    let title = match &plot.config.title {
        Some(t) => format!("{}\n{} vs {} vs {}", t, xlabel, ylabel, zlabel),
        None => format!("3D Surface Plot\n{} vs {} vs {}", xlabel, ylabel, zlabel),
    };
    
    // Create 3D chart context
    let mut chart = ChartBuilder::on(root)
        .caption(&title, ("sans-serif", 30))
        .margin(20)
        .build_cartesian_3d(x_min..x_max, z_min..z_max, y_min..y_max)
        .map_err(|e| PlottingError::Plotters(e.to_string()))?;
    
    // Configure 3D axes with rotation
    chart.with_projection(|mut pb| {
        pb.yaw = surface_data.azimuth.to_radians();
        pb.pitch = surface_data.elevation.to_radians();
        pb.scale = 0.8;
        pb.into_matrix()
    });
    
    chart.configure_axes()
        .x_labels(5)
        .y_labels(5)
        .z_labels(5)
        .x_formatter(&|v| format!("{:.1}", v))
        .y_formatter(&|v| format!("{:.1}", v))
        .z_formatter(&|v| format!("{:.1}", v))
        .draw()
        .map_err(|e| PlottingError::Plotters(e.to_string()))?;
    
    // Create color mapper for surface coloring
    let color_mapper = ColorMapper::new(surface_data.colormap, z_min, z_max);
    
    // Get grid dimensions
    let rows = surface_data.z_data.nrows();
    let cols = surface_data.z_data.ncols();
    
    // Draw surface as polygons
    if surface_data.surface {
        use plotters::element::Polygon;
        
        for i in 0..rows-1 {
            for j in 0..cols-1 {
                // Get the four corners of the current cell
                let x0 = surface_data.x_grid.get(i, j).unwrap_or(0.0);
                let x1 = surface_data.x_grid.get(i, j+1).unwrap_or(0.0);
                let x2 = surface_data.x_grid.get(i+1, j+1).unwrap_or(0.0);
                let x3 = surface_data.x_grid.get(i+1, j).unwrap_or(0.0);
                
                let y0 = surface_data.y_grid.get(i, j).unwrap_or(0.0);
                let y1 = surface_data.y_grid.get(i, j+1).unwrap_or(0.0);
                let y2 = surface_data.y_grid.get(i+1, j+1).unwrap_or(0.0);
                let y3 = surface_data.y_grid.get(i+1, j).unwrap_or(0.0);
                
                let z0 = surface_data.z_data.get(i, j).unwrap() * surface_data.z_scale;
                let z1 = surface_data.z_data.get(i, j+1).unwrap() * surface_data.z_scale;
                let z2 = surface_data.z_data.get(i+1, j+1).unwrap() * surface_data.z_scale;
                let z3 = surface_data.z_data.get(i+1, j).unwrap() * surface_data.z_scale;
                
                // Calculate average Z for color
                let z_avg = (z0 + z1 + z2 + z3) / 4.0;
                let color = color_mapper.map(z_avg);
                let plot_color = RGBColor(color.r, color.g, color.b);
                
                // Create polygon for this cell
                let polygon = Polygon::new(
                    vec![
                        (x0, z0, y0),
                        (x1, z1, y1),
                        (x2, z2, y2),
                        (x3, z3, y3),
                    ],
                    plot_color.mix(0.8).filled(),
                );
                
                chart.draw_series(std::iter::once(polygon))
                    .map_err(|e| PlottingError::Plotters(e.to_string()))?;
            }
        }
    }
    
    // Draw wireframe if enabled
    if surface_data.wireframe {
        // Draw lines along rows
        for i in 0..rows {
            let mut line_points = Vec::new();
            for j in 0..cols {
                let x = surface_data.x_grid.get(i, j).unwrap_or(0.0);
                let y = surface_data.y_grid.get(i, j).unwrap_or(0.0);
                let z = surface_data.z_data.get(i, j).unwrap_or(0.0) * surface_data.z_scale;
                line_points.push((x, z, y));
            }
            
            chart.draw_series(LineSeries::new(
                line_points,
                BLACK.stroke_width(1),
            )).map_err(|e| PlottingError::Plotters(e.to_string()))?;
        }
        
        // Draw lines along columns
        for j in 0..cols {
            let mut line_points = Vec::new();
            for i in 0..rows {
                let x = surface_data.x_grid.get(i, j).unwrap_or(0.0);
                let y = surface_data.y_grid.get(i, j).unwrap_or(0.0);
                let z = surface_data.z_data.get(i, j).unwrap_or(0.0) * surface_data.z_scale;
                line_points.push((x, z, y));
            }
            
            chart.draw_series(LineSeries::new(
                line_points,
                BLACK.stroke_width(1),
            )).map_err(|e| PlottingError::Plotters(e.to_string()))?;
        }
    }
    
    Ok(())
}

impl PlotBackend for JupyterBackend {
    fn render(&mut self, plot: &Plot) -> Result<()> {
        self.width = plot.config.width;
        self.height = plot.config.height;
        
        // Clear the buffer
        self.svg_buffer.clear();
        
        // Create SVG backend
        let root = SVGBackend::with_string(&mut self.svg_buffer, (self.width, self.height))
            .into_drawing_area();
        
        // Apply theme background color
        let bg_color = RGBColor(
            plot.config.theme.background_color.r,
            plot.config.theme.background_color.g,
            plot.config.theme.background_color.b
        );
        root.fill(&bg_color)
            .map_err(|e| PlottingError::Plotters(e.to_string()))?;

        // Check plot type and render appropriately
        let result = if !plot.subplots.is_empty() {
            render_subplots(&root, plot)
        } else if plot.heatmap_data.is_some() {
            // Render heatmap
            render_heatmap(&root, plot)
        } else if plot.contour_data.is_some() {
            // Render contour plot
            render_contour(&root, plot)
        } else if plot.surface3d_data.is_some() {
            // Render 3D surface plot  
            render_surface3d(&root, plot)
        } else {
            // Render standard chart using legacy renderer
            render_legacy_chart(&root, plot)
        };
        
        // Always finalize the drawing area for special plot types
        if !plot.subplots.is_empty() || plot.heatmap_data.is_some() || 
           plot.contour_data.is_some() || plot.surface3d_data.is_some() {
            root.present().map_err(|e| PlottingError::Plotters(e.to_string()))?;
        }
        
        result
    }

    fn show(&mut self) -> Result<()> {
        // For Jupyter notebooks, always try to display inline
        use crate::jupyter::display::evcxr_display_svg;
        evcxr_display_svg(&self.svg_buffer);
        Ok(())
    }

    fn save(&mut self, plot: &Plot, path: &Path) -> Result<()> {
        // First render the plot if not already rendered
        self.render(plot)?;
        
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        file.write_all(self.svg_buffer.as_bytes())?;
        Ok(())
    }
    
    fn is_available(&self) -> bool {
        // Check if we're in a Jupyter environment
        std::env::var("JPY_PARENT_PID").is_ok() || 
        std::env::var("JUPYTER_RUNTIME_DIR").is_ok()
    }
}