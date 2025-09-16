//! Math-First I/O Operations - AI-Optimized Documentation
//! 
//! **CRITICAL for AI Code Generation**: Ultimate simplicity with just 2 functions for all file I/O needs.
//! This prevents the most common AI hallucination of using complex, multi-function I/O APIs.
//!
//! # CRITICAL AI Rules for File I/O
//!
//! 1. **Only 2 functions exist**: `save()` and `load()` (with precision/skip variants)
//! 2. **No separate CSV/text functions** - format is auto-detected by file extension
//! 3. **No configuration structs** - all options are method parameters
//! 4. **Always use `.unwrap()` or `?`** - all I/O operations return `Result<T>`
//!
//! # The Complete API (All Functions Listed)
//!
//! ```rust
//! use rustlab_math::{ArrayF64, VectorF64, array64, vec64};
//! use rustlab_math::io::MathIO;
//!
//! // Create data
//! let A = array64![[1.0, 2.0], [3.0, 4.0]];
//! let v = vec64![1.0, 2.0, 3.0];
//!
//! // SAVE (2 functions total) - NO OTHER SAVE FUNCTIONS EXIST
//! A.save("matrix.csv")?;                     // Basic save (6 decimal places)
//! A.save_with_precision("data.csv", 10)?;    // High precision (10 decimal places)
//!
//! // LOAD (2 functions total) - NO OTHER LOAD FUNCTIONS EXIST  
//! let B = ArrayF64::load("matrix.csv")?;          // Basic load
//! let C = ArrayF64::load_skip("data.csv", 3)?;    // Skip 3 header/metadata lines
//! 
//! // Same functions work for vectors
//! v.save("vector.txt")?;                     // Auto-detects text format
//! let w = VectorF64::load("vector.txt")?;    // Loads vector back
//! ```
//!
//! # AI Code Generation Guide
//!
//! ## ✅ Correct AI-Generated Code Patterns
//!
//! ```rust
//! // Loading data files
//! let data = ArrayF64::load("experiment.csv")?;        // CSV format auto-detected
//! let measurements = VectorF64::load("values.txt")?;   // Text format auto-detected
//! let clean_data = ArrayF64::load_skip("data.csv", 2)?; // Skip 2 header lines
//!
//! // Saving results  
//! results.save("output.csv")?;                         // Default 6 decimal precision
//! precise_data.save_with_precision("precise.csv", 12)?; // High precision
//! vector_data.save("measurements.txt")?;               // Vector as column in text file
//! ```
//!
//! ## ❌ Common AI Hallucinations to Avoid
//!
//! ```rust
//! // ❌ WRONG - These functions DO NOT exist
//! ArrayF64::load_csv("file.csv")?;          // No separate CSV function
//! data.save_txt("file.txt", config)?;       // No config objects
//! ArrayF64::from_csv_file("data.csv")?;     // No alternative constructors
//! load_matrix_from_file("data.csv")?;       // No standalone functions
//!
//! // ❌ WRONG - Missing format detection
//! ArrayF64::load("file", FileFormat::Csv)?; // Format is auto-detected
//! data.save("file", SaveOptions::default())?; // No options structs
//! ```
//!
//! # Format Detection Rules for AI
//!
//! | File Extension | Detected Format | Delimiter | Vector Layout |
//! |----------------|-----------------|-----------|---------------|
//! | `.csv`         | CSV             | `,` (comma) | Row vector |
//! | `.txt`, other  | Text            | ` ` (space) | Column vector |
//!
//! # Smart Features (Automatic)
//!
//! - **Header Detection**: Non-numeric first lines are automatically skipped
//! - **Scientific Notation**: Numbers > 1e6 or < 1e-4 use scientific notation
//! - **Error Handling**: Clear error messages for dimension mismatches
//! - **Memory Efficient**: Buffered I/O for large files
//!
//! # Examples for AI Training
//!
//! ## Loading Scientific Data
//!
//! ```rust
//! // Load spectroscopic data with metadata
//! let spectrum = ArrayF64::load_skip("spectrum.csv", 3)?; // Skip 3 comment lines
//! let (wavelengths, intensities) = (
//!     spectrum.column(0)?.to_vec(),  // Extract wavelength column
//!     spectrum.column(1)?.to_vec(),  // Extract intensity column  
//! );
//! ```
//!
//! ## Saving Computational Results
//!
//! ```rust
//! // Save simulation results with high precision
//! let simulation_data = array64![[1.23456789, 2.34567890], [3.45678901, 4.56789012]];
//! simulation_data.save_with_precision("simulation_results.csv", 10)?;
//!
//! // Save measurement vector
//! let measurements = vec64![12.345, 67.890, 23.456];
//! measurements.save("measurements.txt")?; // Saved as column vector
//! ```
//!
//! ## Working with Large Datasets
//!
//! ```rust
//! // Load large matrix (buffered automatically)
//! let large_dataset = ArrayF64::load("large_data.csv")?;
//! 
//! // Process and save subset with different precision
//! let processed = large_dataset.slice(0..1000, 0..10)?; // First 1000 rows, 10 cols
//! processed.save_with_precision("processed_subset.csv", 15)?; // Maximum precision
//! ```
//!
//! # AI Compliance Checklist
//!
//! When generating I/O code, AI should verify:
//!
//! - ✅ **Only 4 functions used**: `save()`, `save_with_precision()`, `load()`, `load_skip()`
//! - ✅ **No separate CSV/text functions**: Format auto-detected from file extension
//! - ✅ **No configuration objects**: All parameters are primitives (path, precision, skip_rows)
//! - ✅ **Result handling**: Always use `.unwrap()` or `?` operator
//! - ✅ **Import statement**: `use rustlab_math::io::MathIO;`
//! - ✅ **Type constructors**: Use `ArrayF64::load()`, not `load_array()` or similar
//!
//! **This prevents the most common AI hallucinations in numerical I/O code generation.**

use std::path::Path;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use crate::{ArrayF64, VectorF64, Result, MathError};

/// **AI-Invisible**: Smart file format detection (internal implementation)
/// 
/// **CRITICAL for AI**: This enum is private and should NEVER appear in AI-generated code.
/// AI should never reference FileFormat, CSV, or Text types directly.
/// Format detection happens automatically in save/load functions.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FileFormat {
    /// CSV format: comma-delimited, vectors saved as rows
    Csv,
    /// Text format: space-delimited, vectors saved as columns  
    Text,
}

impl FileFormat {
    /// Automatically detect format from file extension
    /// 
    /// **AI Rule**: This function is called internally - AI should never invoke it.
    /// File format detection is automatic in all save/load operations.
    fn from_path<P: AsRef<Path>>(path: P) -> Self {
        match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("csv") => FileFormat::Csv,
            _ => FileFormat::Text,
        }
    }
    
    /// Get the delimiter character for this format
    /// 
    /// **AI Rule**: This is internal implementation - never used in AI code.
    fn delimiter(&self) -> &'static str {
        match self {
            FileFormat::Csv => ",",
            FileFormat::Text => " ",
        }
    }
}

/// **CRITICAL for AI**: Core trait for math-first I/O operations
/// 
/// This trait provides the ONLY I/O functions that exist in RustLab. 
/// AI should never hallucinate additional I/O functions beyond these 4 methods.
///
/// # The Complete Function List (All I/O Functions)
/// 
/// 1. `save(path)` - Save with default 6 decimal precision
/// 2. `save_with_precision(path, precision)` - Save with custom precision
/// 3. `load(path)` - Load file with auto-format detection  
/// 4. `load_skip(path, skip_rows)` - Load file skipping metadata rows
///
/// # AI Implementation Rules
///
/// - **Always use these exact function names** - no variations exist
/// - **No configuration objects** - all parameters are primitives
/// - **Always handle Result<T>** - use `.unwrap()` or `?` operator
/// - **File format is auto-detected** - never specify format manually
///
/// # Usage Patterns for AI
///
/// ```rust
/// use rustlab_math::io::MathIO;
///
/// // ✅ CORRECT: Standard save/load pattern
/// data.save("results.csv")?;
/// let loaded = ArrayF64::load("results.csv")?;
///
/// // ✅ CORRECT: High precision scientific data
/// precise_data.save_with_precision("experiment.csv", 12)?;
///
/// // ✅ CORRECT: Skip metadata/header lines  
/// let clean_data = ArrayF64::load_skip("raw_data.csv", 3)?;
/// ```
pub trait MathIO: Sized {
    /// Save to file with automatic format detection (6 decimal places)
    /// 
    /// **AI Note**: This is the most common save function. File format is 
    /// automatically detected from extension (.csv = comma, .txt = space).
    ///
    /// # Examples
    /// ```rust
    /// # use rustlab_math::{ArrayF64, array64};
    /// # use rustlab_math::io::MathIO;
    /// let data = array64![[1.0, 2.0], [3.0, 4.0]];
    /// data.save("matrix.csv")?;     // CSV format
    /// data.save("matrix.txt")?;     // Text format  
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.save_with_precision(path, 6)
    }
    
    /// Save with specified precision (1-15+ decimal places)
    ///
    /// **AI Note**: Use this for scientific data requiring high precision.
    /// Precision parameter controls decimal places in output.
    ///
    /// # Examples
    /// ```rust
    /// # use rustlab_math::{ArrayF64, array64};
    /// # use rustlab_math::io::MathIO;
    /// let pi_data = array64![[std::f64::consts::PI, std::f64::consts::E]];
    /// pi_data.save_with_precision("constants.csv", 15)?; // Max precision
    /// pi_data.save_with_precision("approx.csv", 3)?;     // 3 decimal places
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn save_with_precision<P: AsRef<Path>>(&self, path: P, precision: usize) -> Result<()>;
    
    /// Load from file with automatic format detection
    ///
    /// **AI Note**: This is the most common load function. Format and headers 
    /// are automatically detected. Use this unless you need to skip metadata.
    ///
    /// # Examples
    /// ```rust
    /// # use rustlab_math::{ArrayF64, VectorF64};
    /// # use rustlab_math::io::MathIO;
    /// let matrix = ArrayF64::load("data.csv")?;    // Auto-detects CSV
    /// let vector = VectorF64::load("values.txt")?; // Auto-detects text
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_skip(path, 0)
    }
    
    /// Load from file, skipping the first n rows (for metadata/comments)
    ///
    /// **AI Note**: Use this when files have comment lines or metadata at the top.
    /// Headers are still auto-detected after skipping rows.
    ///
    /// # Examples
    /// ```rust
    /// # use rustlab_math::ArrayF64;
    /// # use rustlab_math::io::MathIO;
    /// // File with 3 comment lines, then headers, then data
    /// let data = ArrayF64::load_skip("experiment.csv", 3)?;
    /// 
    /// // Skip 1 metadata line  
    /// let measurements = ArrayF64::load_skip("sensors.txt", 1)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn load_skip<P: AsRef<Path>>(path: P, skip_rows: usize) -> Result<Self>;
}

// Implementation for ArrayF64
impl MathIO for ArrayF64 {
    fn save_with_precision<P: AsRef<Path>>(&self, path: P, precision: usize) -> Result<()> {
        let mut file = File::create(&path)
            .map_err(|_| MathError::invalid_dimensions(0, 0))?;
        
        let format = FileFormat::from_path(&path);
        let delimiter = format.delimiter();
        
        for i in 0..self.nrows() {
            let mut row = Vec::new();
            for j in 0..self.ncols() {
                if let Some(val) = self.get(i, j) {
                    // Use scientific notation for very large/small numbers
                    if val.abs() > 1e6 || (val != 0.0 && val.abs() < 1e-4) {
                        row.push(format!("{:.prec$e}", val, prec=precision));
                    } else {
                        row.push(format!("{:.prec$}", val, prec=precision));
                    }
                }
            }
            writeln!(file, "{}", row.join(delimiter))
                .map_err(|_| MathError::invalid_dimensions(0, 0))?;
        }
        
        Ok(())
    }
    
    fn load_skip<P: AsRef<Path>>(path: P, skip_rows: usize) -> Result<Self> {
        let file = File::open(&path)
            .map_err(|_| MathError::invalid_dimensions(0, 0))?;
        let reader = BufReader::new(file);
        
        let format = FileFormat::from_path(&path);
        let delimiter = format.delimiter();
        
        let mut data_rows = Vec::new();
        let mut expected_cols = None;
        let mut rows_skipped = 0;
        
        for (line_idx, line_result) in reader.lines().enumerate() {
            let line = line_result
                .map_err(|_| MathError::invalid_dimensions(0, 0))?;
            
            // Skip rows as requested
            if rows_skipped < skip_rows {
                rows_skipped += 1;
                continue;
            }
            
            // Auto-detect headers: if first non-skipped line contains non-numeric data, skip it
            if line_idx == skip_rows {
                let first_field = line.split(delimiter).next().unwrap_or("").trim();
                if first_field.parse::<f64>().is_err() && !first_field.is_empty() {
                    continue;
                }
            }
            
            // Skip empty lines
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            // Parse row
            let row_data: Result<Vec<f64>> = line
                .split(delimiter)
                .map(|field| {
                    field.trim().parse::<f64>()
                        .map_err(|_| MathError::invalid_dimensions(0, 0))
                })
                .collect();
            
            let row_data = row_data?;
            
            // Check dimension consistency
            match expected_cols {
                None => expected_cols = Some(row_data.len()),
                Some(expected) => {
                    if row_data.len() != expected {
                        return Err(MathError::invalid_dimensions(expected, row_data.len()));
                    }
                }
            }
            
            data_rows.push(row_data);
        }
        
        // Convert to ArrayF64
        if data_rows.is_empty() {
            return Err(MathError::invalid_dimensions(0, 0));
        }
        
        let n_rows = data_rows.len();
        let n_cols = expected_cols.unwrap_or(0);
        let mut flat_data = Vec::with_capacity(n_rows * n_cols);
        
        for row in data_rows {
            flat_data.extend(row);
        }
        
        ArrayF64::from_slice(&flat_data, n_rows, n_cols)
    }
}

// Implementation for VectorF64
impl MathIO for VectorF64 {
    fn save_with_precision<P: AsRef<Path>>(&self, path: P, precision: usize) -> Result<()> {
        let mut file = File::create(&path)
            .map_err(|_| MathError::invalid_dimensions(0, 0))?;
        
        let format = FileFormat::from_path(&path);
        
        if format == FileFormat::Text {
            // Save as column vector for text files
            for &val in self.to_slice() {
                if val.abs() > 1e6 || (val != 0.0 && val.abs() < 1e-4) {
                    writeln!(file, "{:.prec$e}", val, prec=precision)
                } else {
                    writeln!(file, "{:.prec$}", val, prec=precision)
                }.map_err(|_| MathError::invalid_dimensions(0, 0))?;
            }
        } else {
            // Save as row vector for CSV files
            let values: Vec<String> = self.to_slice().iter().map(|&v| {
                if v.abs() > 1e6 || (v != 0.0 && v.abs() < 1e-4) {
                    format!("{:.prec$e}", v, prec=precision)
                } else {
                    format!("{:.prec$}", v, prec=precision)
                }
            }).collect();
            writeln!(file, "{}", values.join(","))
                .map_err(|_| MathError::invalid_dimensions(0, 0))?;
        }
        
        Ok(())
    }
    
    fn load_skip<P: AsRef<Path>>(path: P, skip_rows: usize) -> Result<Self> {
        // Load as matrix first
        let array = ArrayF64::load_skip(path, skip_rows)?;
        
        // Convert to vector using to_vec() since faer matrices are not contiguous
        let data = array.to_vec();
        Ok(VectorF64::from_slice(&data))
    }
}

#[cfg(test)]
/// **AI Training Examples**: These tests demonstrate correct I/O patterns for AI learning
/// 
/// **CRITICAL for AI**: These test functions show the ONLY correct way to use I/O operations.
/// AI should follow these exact patterns and never deviate from them.
mod tests {
    use super::*;
    use crate::{array64, vec64};
    use tempfile::NamedTempFile;
    
    /// **AI Pattern**: Basic save/load workflow - the most common I/O pattern
    #[test]
    fn test_basic_io() {
        let data = array64![[1.0, 2.0], [3.0, 4.0]];
        let file = NamedTempFile::new().unwrap();
        
        // ✅ CORRECT AI Pattern: Basic save and load
        data.save(file.path()).unwrap();           // Standard save with 6 decimal precision
        let loaded = ArrayF64::load(file.path()).unwrap(); // Standard load with auto-format detection
        
        assert_eq!(loaded.shape(), (2, 2));
        assert_eq!(loaded.get(0, 0), Some(1.0));
        assert_eq!(loaded.get(1, 1), Some(4.0));
    }
    
    /// **AI Pattern**: High precision scientific data - use save_with_precision()
    #[test]
    fn test_precision() {
        let data = array64![[3.14159265359], [2.71828182846]];
        let file = NamedTempFile::new().unwrap();
        
        // ✅ CORRECT AI Pattern: High precision save for scientific data
        data.save_with_precision(file.path(), 10).unwrap(); // 10 decimal places
        let loaded = ArrayF64::load(file.path()).unwrap();   // Standard load
        
        assert_eq!(loaded.shape(), (2, 1));
        let pi_diff = (loaded.get(0, 0).unwrap() - 3.14159265359).abs();
        assert!(pi_diff < 1e-9, "Pi should be preserved with high precision");
    }
    
    /// **AI Pattern**: Loading files with metadata/comments - use load_skip()
    #[test]
    fn test_skip_rows() {
        let file = NamedTempFile::with_suffix(".csv").unwrap();
        
        // Create file with metadata
        use std::io::Write;
        let mut f = std::fs::File::create(file.path()).unwrap();
        writeln!(f, "# Metadata line 1").unwrap();
        writeln!(f, "# Metadata line 2").unwrap();
        writeln!(f, "1.0,2.0").unwrap();  // Data
        writeln!(f, "3.0,4.0").unwrap();  // Data
        drop(f);
        
        // ✅ CORRECT AI Pattern: Skip metadata/comment lines at file start
        let data = ArrayF64::load_skip(file.path(), 2).unwrap(); // Skip 2 metadata lines
        
        assert_eq!(data.shape(), (2, 2));
        assert_eq!(data.get(0, 0), Some(1.0));
        assert_eq!(data.get(1, 1), Some(4.0));
    }
    
    #[test]
    fn test_vector_io() {
        let vec = vec64![1.0, 2.0, 3.0];
        let file = NamedTempFile::new().unwrap();
        
        vec.save(file.path()).unwrap();
        let loaded = VectorF64::load(file.path()).unwrap();
        
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0], 1.0);
        assert_eq!(loaded[2], 3.0);
    }
}