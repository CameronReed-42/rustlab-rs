pub use crate::error::{PlottingError, Result};
pub use crate::plot::types::{Plot, Backend, Color, LineStyle, PlotType, Scale, Marker};
pub use crate::plot::builder::PlotBuilder;

// Re-export commonly used functions
pub use crate::{plot, scatter, bar, histogram};