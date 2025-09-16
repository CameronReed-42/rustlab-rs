use crate::error::{PlottingError, Result};
use crate::plot::types::{Plot, Scale, Series};
use rustlab_math::VectorF64;

/// Validates that the plot data is compatible with the specified scales
pub fn validate_plot_data(plot: &Plot) -> Result<()> {
    // Validate each series against the plot's scale settings
    for (i, series) in plot.series.iter().enumerate() {
        validate_series_for_scale(series, plot.config.x_scale, plot.config.y_scale, i)?;
    }
    
    // Validate subplots if any
    for subplot in &plot.subplots {
        for (i, series) in subplot.series.iter().enumerate() {
            validate_series_for_scale(series, subplot.config.x_scale, subplot.config.y_scale, i)?;
        }
    }
    
    Ok(())
}

/// Validates a single series against the specified x and y scales
fn validate_series_for_scale(
    series: &Series, 
    x_scale: Scale, 
    y_scale: Scale, 
    series_index: usize
) -> Result<()> {
    // Check X data for log scale compatibility
    if needs_positive_values(x_scale) {
        validate_positive_data(&series.x, "X", series_index)?;
    }
    
    // Check Y data for log scale compatibility
    if needs_positive_values(y_scale) {
        validate_positive_data(&series.y, "Y", series_index)?;
    }
    
    Ok(())
}

/// Returns true if the scale requires positive values
fn needs_positive_values(scale: Scale) -> bool {
    matches!(scale, Scale::Log10 | Scale::Log2 | Scale::Ln)
}

/// Validates that all values in the data are positive (required for log scales)
fn validate_positive_data(data: &VectorF64, axis_name: &str, series_index: usize) -> Result<()> {
    let slice = data.as_slice().expect("Vector must be contiguous for validation");
    let negative_count = slice.iter().filter(|&&x| x <= 0.0).count();
    
    if negative_count > 0 {
        let total_count = slice.len();
        return Err(PlottingError::InvalidData(format!(
            "Series {} has {} non-positive values in {} data (out of {} total). \
            Logarithmic scales require all positive values. \
            Consider filtering the data or using a different scale.",
            series_index, negative_count, axis_name, total_count
        )));
    }
    
    Ok(())
}

/// Filters out non-positive values for log scale compatibility
/// Returns a new VectorF64 with only positive values and their corresponding indices
pub fn filter_positive_data(x_data: &VectorF64, y_data: &VectorF64) -> Result<(VectorF64, VectorF64)> {
    let x_slice = x_data.as_slice().expect("X vector must be contiguous for filtering");
    let y_slice = y_data.as_slice().expect("Y vector must be contiguous for filtering");
    
    if x_slice.len() != y_slice.len() {
        return Err(PlottingError::InvalidData(
            "X and Y data must have the same length".to_string()
        ));
    }
    
    let mut filtered_x = Vec::new();
    let mut filtered_y = Vec::new();
    
    for (&x, &y) in x_slice.iter().zip(y_slice.iter()) {
        if x > 0.0 && y > 0.0 {
            filtered_x.push(x);
            filtered_y.push(y);
        }
    }
    
    if filtered_x.is_empty() {
        return Err(PlottingError::InvalidData(
            "No positive data points found after filtering for log scale".to_string()
        ));
    }
    
    Ok((VectorF64::from_vec(filtered_x), VectorF64::from_vec(filtered_y)))
}

/// Adjusts data range to be compatible with log scales
/// Ensures minimum value is positive and adds appropriate padding
pub fn adjust_log_range(min: f64, max: f64) -> (f64, f64) {
    let adjusted_min = if min <= 0.0 {
        // If min is non-positive, use a small positive value
        1e-10
    } else {
        // Add some padding on the low end (divide by factor for log scale)
        min / 10.0
    };
    
    let adjusted_max = if max <= 0.0 {
        // If max is also non-positive, use a reasonable default
        10.0
    } else {
        // Add some padding on the high end (multiply by factor for log scale)
        max * 10.0
    };
    
    (adjusted_min, adjusted_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::types::{PlotConfig, PlotType};
    use crate::style::theme::Theme;
    
    #[test]
    fn test_validate_positive_data_success() {
        let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert!(validate_positive_data(&data, "X", 0).is_ok());
    }
    
    #[test]
    fn test_validate_positive_data_failure() {
        let data = VectorF64::from_slice(&[1.0, -2.0, 3.0, 0.0]);
        let result = validate_positive_data(&data, "X", 0);
        assert!(result.is_err());
        if let Err(PlottingError::InvalidData(msg)) = result {
            assert!(msg.contains("2 non-positive values"));
        }
    }
    
    #[test]
    fn test_filter_positive_data() {
        let x = VectorF64::from_slice(&[1.0, -2.0, 3.0, 0.0, 5.0]);
        let y = VectorF64::from_slice(&[2.0, 4.0, -1.0, 8.0, 10.0]);
        
        let (filtered_x, filtered_y) = filter_positive_data(&x, &y).unwrap();
        
        // Only (1.0, 2.0) and (5.0, 10.0) should remain
        assert_eq!(filtered_x.as_slice().unwrap(), &[1.0, 5.0]);
        assert_eq!(filtered_y.as_slice().unwrap(), &[2.0, 10.0]);
    }
    
    #[test]
    fn test_adjust_log_range() {
        // Test normal positive range
        let (min, max) = adjust_log_range(1.0, 100.0);
        assert_eq!(min, 0.1);
        assert_eq!(max, 1000.0);
        
        // Test range with negative minimum
        let (min, max) = adjust_log_range(-5.0, 100.0);
        assert_eq!(min, 1e-10);
        assert_eq!(max, 1000.0);
        
        // Test range with all negative values
        let (min, max) = adjust_log_range(-10.0, -1.0);
        assert_eq!(min, 1e-10);
        assert_eq!(max, 10.0);
    }
    
    #[test]
    fn test_needs_positive_values() {
        assert!(needs_positive_values(Scale::Log10));
        assert!(needs_positive_values(Scale::Log2));
        assert!(needs_positive_values(Scale::Ln));
        assert!(!needs_positive_values(Scale::Linear));
    }
}