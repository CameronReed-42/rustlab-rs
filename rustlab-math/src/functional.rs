//! Functional programming operations with AI-optimized documentation
//!
//! This module provides comprehensive functional programming capabilities with zero-cost
//! abstractions and seamless integration with RustLab's mathematical operators. All
//! operations follow functional programming principles and maintain mathematical correctness.
//!
//! # Common AI Patterns
//! ```rust
//! use rustlab_math::{ArrayF64, VectorF64, functional::*};
//! 
//! let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
//! 
//! // Data transformation and preprocessing
//! let normalized = data.map(|x| (x - 3.0) / 2.0);     // Z-score normalization
//! let squared = data.map(|x| x * x);                  // Element-wise squaring
//! let filtered = data.filter(|x| x > 2.5);            // Conditional filtering
//! 
//! // Statistical aggregation
//! let total = data.sum_elements();                     // Sum all elements
//! let product = data.product();                       // Product of elements
//! let maximum = data.max().unwrap();                  // Find maximum
//! 
//! // Advanced functional operations
//! let cumulative = data.scan(0.0, |acc, x| acc + x);  // Cumulative sum
//! let first_half = data.take(data.len() / 2);         // Take first N
//! let outliers = data.filter(|x| x > mean + 2.0 * std); // Outlier detection
//! 
//! // Combining multiple datasets
//! let other = VectorF64::ones(5);
//! let combined = data.zip_with(&other, |a, b| a * b).unwrap(); // Element-wise combination
//! ```
//!
//! # Functional Programming Principles
//! - **Immutability**: Most operations return new containers
//! - **Composability**: Operations can be chained together
//! - **Lazy evaluation**: Operations are computed when needed
//! - **Higher-order functions**: Functions that take functions as parameters
//!
//! # Cross-Module Integration
//! - Compatible with [`Array`] and [`Vector`] mathematical operators
//! - Works with [`statistics`] for functional statistical analysis
//! - Integrates with [`comparison`] for predicate-based operations
//! - Maintains zero-cost abstractions throughout the functional pipeline

use crate::{Array, Vector};
use faer::{Mat, Col};
use faer_entity::Entity;
use faer_traits::ComplexField;
use num_traits::{Zero, One};

// ========== FUNCTIONAL PROGRAMMING TRAITS ==========

/// Trait for functional map operations with AI-optimized documentation
/// 
/// This trait provides the fundamental map operation that applies a function to each
/// element of a container, creating a new container with the transformed values.
/// 
/// # For AI Code Generation
/// - **Core transformation operation** for data preprocessing
/// - Immutable: creates new container, leaves original unchanged
/// - Type-safe: can transform T to U with compile-time checking
/// - Essential for: normalization, scaling, feature engineering, activation functions
/// - Equivalent to NumPy's vectorized operations or pandas.map()
/// 
/// # Mathematical Specification
/// For container C = [c₁, c₂, ..., cₙ] and function f: T → U:
/// map(C, f) = [f(c₁), f(c₂), ..., f(cₙ)]
/// 
/// # Common Use Cases
/// - **Data normalization**: `data.map(|x| (x - mean) / std)`
/// - **Feature engineering**: `features.map(|x| x.ln())`
/// - **Activation functions**: `layer.map(|x| x.tanh())`
/// - **Unit conversion**: `temps.map(|c| c * 9.0 / 5.0 + 32.0)`
pub trait FunctionalMap<T: Entity> {
    /// The output container type
    type Container<U: Entity>;
    
    /// Apply a function to each element, creating a new container
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, functional::FunctionalMap};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// let squared = data.map(|x| x * x);  // [1.0, 4.0, 9.0]
    /// ```
    fn map<U, F>(&self, f: F) -> Self::Container<U>
    where
        U: Entity + ComplexField,
        F: Fn(T) -> U,
        T: Clone;
    
    /// Apply a function to each element in-place (for mutable operations)
    fn map_mut<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
        T: Clone;
}

/// Trait for functional fold/reduce operations with AI-optimized documentation
/// 
/// This trait provides aggregation operations that collapse a container into a single
/// value using accumulator functions. Essential for statistical computations.
/// 
/// # For AI Code Generation
/// - **Core aggregation operations** for data analysis
/// - fold: Uses initial value, always returns result
/// - reduce: Uses first element as initial, returns Option
/// - Essential for: statistics, loss functions, metric calculations
/// - Equivalent to NumPy's reduction functions or pandas.aggregate()
/// 
/// # Mathematical Specification
/// For container C = [c₁, c₂, ..., cₙ], initial value i, and function f:
/// fold(C, i, f) = f(f(f(i, c₁), c₂), ..., cₙ)
/// reduce(C, f) = f(f(c₁, c₂), c₃), ..., cₙ) if n > 0, None if n = 0
/// 
/// # Common Use Cases
/// - **Sum**: `data.fold(0.0, |acc, x| acc + x)`
/// - **Product**: `data.fold(1.0, |acc, x| acc * x)`
/// - **Mean**: `data.fold(0.0, |acc, x| acc + x) / n as f64`
/// - **Maximum**: `data.reduce(|a, b| if a > b { a } else { b })`
pub trait FunctionalReduce<T: Entity> {
    /// Fold all elements with an initial value and accumulator function
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::{VectorF64, functional::FunctionalReduce};
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let sum = data.fold(0.0, |acc, x| acc + x);  // 10.0
    /// ```
    fn fold<U, F>(&self, init: U, f: F) -> U
    where
        F: Fn(U, T) -> U,
        T: Clone;
    
    /// Reduce elements without initial value (uses first element as initial)
    fn reduce<F>(&self, f: F) -> Option<T>
    where
        F: Fn(T, T) -> T,
        T: Clone;
}

/// Trait for functional zip operations  
pub trait FunctionalZip<T: Entity> {
    /// The output container type
    type Container<U: Entity>;
    
    /// Zip two containers with a binary function
    fn zip_with<V, F>(&self, other: &Self, f: F) -> Result<Self::Container<V>, String>
    where
        V: Entity + ComplexField,
        F: Fn(T, T) -> V,
        T: Clone;
}

// ========== ARRAY FUNCTIONAL OPERATIONS ==========

impl<T: Entity> FunctionalMap<T> for Array<T> {
    type Container<U: Entity> = Array<U>;
    
    fn map<U, F>(&self, f: F) -> Array<U>
    where
        U: Entity + ComplexField,
        F: Fn(T) -> U,
        T: Clone,
    {
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            f(self.get(i, j).unwrap())
        });
        Array::from_faer(result)
    }
    
    fn map_mut<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
        T: Clone,
    {
        let (rows, cols) = self.shape();
        for i in 0..rows {
            for j in 0..cols {
                let old_val = self.get(i, j).unwrap();
                let new_val = f(old_val);
                self.set(i, j, new_val).unwrap();
            }
        }
    }
}

impl<T: Entity> FunctionalReduce<T> for Array<T> {
    fn fold<U, F>(&self, init: U, f: F) -> U
    where
        F: Fn(U, T) -> U,
        T: Clone,
    {
        let (rows, cols) = self.shape();
        let mut acc = init;
        for i in 0..rows {
            for j in 0..cols {
                acc = f(acc, self.get(i, j).unwrap());
            }
        }
        acc
    }
    
    fn reduce<F>(&self, f: F) -> Option<T>
    where
        F: Fn(T, T) -> T,
        T: Clone,
    {
        let (rows, cols) = self.shape();
        if rows == 0 || cols == 0 {
            return None;
        }
        
        let mut acc = self.get(0, 0).unwrap();
        for i in 0..rows {
            for j in 0..cols {
                if i == 0 && j == 0 {
                    continue; // Skip first element as it's our initial value
                }
                acc = f(acc, self.get(i, j).unwrap());
            }
        }
        Some(acc)
    }
}

impl<T: Entity> FunctionalZip<T> for Array<T> {
    type Container<U: Entity> = Array<U>;
    
    fn zip_with<V, F>(&self, other: &Self, f: F) -> Result<Array<V>, String>
    where
        V: Entity + ComplexField,  
        F: Fn(T, T) -> V,
        T: Clone,
    {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        
        let (rows, cols) = self.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            f(self.get(i, j).unwrap(), other.get(i, j).unwrap())
        });
        Ok(Array::from_faer(result))
    }
}

// Array-specific convenience methods
impl<T: Entity> Array<T> {
    /// Apply a function to each element and return a new array
    /// 
    /// # For AI Code Generation
    /// - Alias for map() method for intuitive naming
    /// - Common in data science: "apply transformation to dataset"
    /// - Same functionality as map but more descriptive name
    /// - Use for complex transformations, feature engineering
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let features = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let normalized = features.apply(|x| (x - 2.5) / 1.5);  // Z-score normalization
    /// let activated = features.apply(|x| x.tanh());           // Activation function
    /// ```
    pub fn apply<U, F>(&self, f: F) -> Array<U>
    where
        U: Entity + ComplexField,
        F: Fn(T) -> U,  
        T: Clone + ComplexField,
    {
        self.map(f)
    }
    
    /// Sum all elements in the array
    /// 
    /// # For AI Code Generation
    /// - Aggregates all matrix elements into single sum value
    /// - Essential for loss functions, total calculations, statistics
    /// - Uses fold with zero initial value and addition accumulator
    /// - Equivalent to NumPy's np.sum() with no axis specified
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::ArrayF64;
    /// 
    /// let data = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    /// let total = data.sum();  // 10.0
    /// let mean = total / (data.nrows() * data.ncols()) as f64;  // 2.5
    /// ```
    pub fn sum(&self) -> T
    where
        T: Clone + ComplexField + Zero,
    {
        self.fold(T::zero(), |acc, x| acc + x)
    }
    
    /// Find the product of all elements
    pub fn product(&self) -> T
    where
        T: Clone + ComplexField + One,
    {
        self.fold(T::one(), |acc, x| acc * x)
    }
    
    /// Find the minimum element
    pub fn min(&self) -> Option<T>
    where
        T: Clone + ComplexField + PartialOrd,
    {
        self.reduce(|a, b| if a <= b { a } else { b })
    }
    
    /// Find the maximum element
    pub fn max(&self) -> Option<T>
    where
        T: Clone + ComplexField + PartialOrd,
    {
        self.reduce(|a, b| if a >= b { a } else { b })
    }
}

// ========== VECTOR FUNCTIONAL OPERATIONS ==========

impl<T: Entity + ComplexField> FunctionalMap<T> for Vector<T> {
    type Container<U: Entity> = Vector<U>;
    
    fn map<U, F>(&self, f: F) -> Vector<U>
    where
        U: Entity + ComplexField,
        F: Fn(T) -> U,
        T: Clone,
    {
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            f(self.get(i).unwrap())
        });
        Vector::from_faer(result)
    }
    
    fn map_mut<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
        T: Clone,
    {
        if let Some(slice) = self.as_mut_slice() {
            for elem in slice.iter_mut() {
                *elem = f(elem.clone());
            }
        } else {
            // Fallback for non-contiguous vectors
            let len = self.len();
            // Fallback for non-contiguous vectors - reconstruct the vector
            let new_data: Vec<T> = (0..len)
                .map(|i| f(self.get(i).unwrap()))
                .collect();
            *self = Vector::from_slice(&new_data);
        }
    }
}

impl<T: Entity> FunctionalReduce<T> for Vector<T> {
    fn fold<U, F>(&self, init: U, f: F) -> U
    where
        F: Fn(U, T) -> U,
        T: Clone,
    {
        let len = self.len();
        let mut acc = init;
        for i in 0..len {
            acc = f(acc, self.get(i).unwrap());
        }
        acc
    }
    
    fn reduce<F>(&self, f: F) -> Option<T>
    where
        F: Fn(T, T) -> T,
        T: Clone,
    {
        let len = self.len();
        if len == 0 {
            return None;
        }
        
        let mut acc = self.get(0).unwrap();
        for i in 1..len {
            acc = f(acc, self.get(i).unwrap());
        }
        Some(acc)
    }
}

impl<T: Entity> FunctionalZip<T> for Vector<T> {
    type Container<U: Entity> = Vector<U>;
    
    fn zip_with<V, F>(&self, other: &Self, f: F) -> Result<Vector<V>, String>
    where
        V: Entity + ComplexField,
        F: Fn(T, T) -> V,
        T: Clone,
    {
        if self.len() != other.len() {
            return Err(format!("Length mismatch: {} vs {}", self.len(), other.len()));
        }
        
        let len = self.len();
        let result = Col::from_fn(len, |i| {
            f(self.get(i).unwrap(), other.get(i).unwrap())
        });
        Ok(Vector::from_faer(result))
    }
}

// Vector-specific convenience methods  
impl<T: Entity> Vector<T> {
    /// Apply a function to each element and return a new vector
    /// 
    /// # Examples
    /// ```rust
    /// use rustlab_math::*;
    /// 
    /// let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
    /// let squared = vec.apply(|x| x * x);
    /// assert_eq!(squared.get(0), Some(1.0));
    /// assert_eq!(squared.get(1), Some(4.0));
    /// ```
    pub fn apply<U, F>(&self, f: F) -> Vector<U>
    where
        U: Entity + ComplexField,
        F: Fn(T) -> U,
        T: Clone + ComplexField,
    {
        self.map(f)
    }
    
    /// Sum all elements in the vector
    pub fn sum_elements(&self) -> T
    where
        T: Clone + ComplexField + Zero,
    {
        self.fold(T::zero(), |acc, x| acc + x)
    }
    
    /// Find the product of all elements
    pub fn product(&self) -> T
    where
        T: Clone + ComplexField + One,
    {
        self.fold(T::one(), |acc, x| acc * x)
    }
    
    /// Find the minimum element
    pub fn min(&self) -> Option<T>
    where
        T: Clone + ComplexField + PartialOrd,
    {
        self.reduce(|a, b| if a <= b { a } else { b })
    }
    
    /// Find the maximum element
    pub fn max(&self) -> Option<T>
    where
        T: Clone + ComplexField + PartialOrd,
    {
        self.reduce(|a, b| if a >= b { a } else { b })
    }
    
    /// Filter elements based on a predicate, returning a new vector with matching elements
    /// 
    /// # For AI Code Generation
    /// - **Critical data filtering operation** for preprocessing
    /// - Returns new vector containing only elements that satisfy condition
    /// - Variable output size: length depends on how many elements match
    /// - Essential for: outlier removal, data cleaning, conditional selection
    /// - Equivalent to NumPy's boolean indexing or pandas.filter()
    /// 
    /// # Example
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let data = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// 
    /// // Remove outliers (values > 3.0)
    /// let clean = data.filter(|x| x <= 3.0);         // [1.0, 2.0, 3.0]
    /// 
    /// // Select positive values only
    /// let positive = data.filter(|x| x > 0.0);       // All values (all positive)
    /// 
    /// // Feature selection based on criteria
    /// let significant = data.filter(|x| x.abs() > 2.0); // [3.0, 4.0, 5.0]
    /// ```
    pub fn filter<F>(&self, predicate: F) -> Vector<T>
    where
        F: Fn(T) -> bool,
        T: Clone + ComplexField,
    {
        let filtered_data: Vec<T> = (0..self.len())
            .filter_map(|i| {
                let value = self.get(i).unwrap();
                if predicate(value) {
                    Some(value)
                } else {
                    None
                }
            })
            .collect();
        
        Vector::from_slice(&filtered_data)
    }
    
    /// Take the first n elements, returning a new vector
    pub fn take(&self, n: usize) -> Vector<T>
    where
        T: Clone + ComplexField,
    {
        let len = self.len().min(n);
        let data: Vec<T> = (0..len)
            .map(|i| self.get(i).unwrap())
            .collect();
        
        Vector::from_slice(&data)
    }
    
    /// Skip the first n elements, returning a new vector with the rest
    pub fn skip(&self, n: usize) -> Vector<T>
    where
        T: Clone + ComplexField,
    {
        if n >= self.len() {
            return Vector::zeros(0);
        }
        
        let data: Vec<T> = (n..self.len())
            .map(|i| self.get(i).unwrap())
            .collect();
        
        Vector::from_slice(&data)
    }
    
    /// Create an iterator of (index, value) pairs
    pub fn enumerate(&self) -> Vec<(usize, T)>
    where
        T: Clone + ComplexField,
    {
        (0..self.len())
            .map(|i| (i, self.get(i).unwrap()))
            .collect()
    }
    
    /// Check if all elements satisfy a predicate
    pub fn all<F>(&self, predicate: F) -> bool
    where
        F: Fn(T) -> bool,
        T: Clone + ComplexField,
    {
        (0..self.len()).all(|i| predicate(self.get(i).unwrap()))
    }
    
    /// Check if any element satisfies a predicate
    pub fn any<F>(&self, predicate: F) -> bool
    where
        F: Fn(T) -> bool,
        T: Clone + ComplexField,
    {
        (0..self.len()).any(|i| predicate(self.get(i).unwrap()))
    }
    
    /// Find the first element that satisfies a predicate
    pub fn find<F>(&self, predicate: F) -> Option<T>
    where
        F: Fn(T) -> bool,
        T: Clone + ComplexField,
    {
        (0..self.len())
            .find_map(|i| {
                let value = self.get(i).unwrap();
                if predicate(value) {
                    Some(value)
                } else {
                    None
                }
            })
    }
    
    /// Count elements that satisfy a predicate
    pub fn count<F>(&self, predicate: F) -> usize
    where
        F: Fn(T) -> bool,
        T: Clone + ComplexField,
    {
        (0..self.len())
            .filter(|&i| predicate(self.get(i).unwrap()))
            .count()
    }
    
    /// Cumulative scan with an accumulator function (like fold but returns intermediate results)
    /// 
    /// # For AI Code Generation
    /// - **Essential for time series analysis** and sequential processing
    /// - Like fold but returns vector of intermediate accumulator values
    /// - Output same length as input, each element shows cumulative result
    /// - Common uses: cumulative sums, running averages, progressive computations
    /// - Equivalent to NumPy's cumsum, cumproduct, or pandas.cumsum()
    /// 
    /// # Mathematical Specification
    /// For vector v = [v₁, v₂, ..., vₙ], initial value i, and function f:
    /// scan(v, i, f) = [f(i,v₁), f(f(i,v₁),v₂), f(f(f(i,v₁),v₂),v₃), ...]
    /// 
    /// # Example  
    /// ```rust
    /// use rustlab_math::VectorF64;
    /// 
    /// let prices = VectorF64::from_slice(&[100.0, 105.0, 98.0, 103.0]);
    /// 
    /// // Cumulative sum (portfolio value)
    /// let cumsum = prices.scan(0.0, |acc, x| acc + x);  // [100, 205, 303, 406]
    /// 
    /// // Running maximum (highest price seen)
    /// let running_max = prices.scan(0.0, |acc, x| acc.max(x)); // [100, 105, 105, 105]
    /// 
    /// // Cumulative product (compound returns)
    /// let returns = VectorF64::from_slice(&[1.05, 0.98, 1.03, 1.02]);
    /// let cumulative = returns.scan(1.0, |acc, x| acc * x); // [1.05, 1.029, 1.06, 1.08]
    /// ```
    pub fn scan<U, F>(&self, init: U, f: F) -> Vector<U>
    where
        U: Entity + ComplexField + Clone,
        F: Fn(U, T) -> U,
        T: Clone + ComplexField,
    {
        let mut acc = init;
        let results: Vec<U> = (0..self.len())
            .map(|i| {
                acc = f(acc.clone(), self.get(i).unwrap());
                acc.clone()
            })
            .collect();
        
        Vector::from_slice(&results)
    }
    
    /// Find the index of the first element that satisfies a predicate
    pub fn find_index<F>(&self, predicate: F) -> Option<usize>
    where
        F: Fn(T) -> bool,
        T: Clone + ComplexField,
    {
        (0..self.len())
            .find(|&i| predicate(self.get(i).unwrap()))
    }
}

// ========== ZIPPED STRUCTURES FOR ADVANCED OPERATIONS ==========

/// A zipped view of two arrays for functional operations
pub struct ZippedArrays<'a, T: Entity, U: Entity> {
    left: &'a Array<T>,
    right: &'a Array<U>,
}

impl<'a, T: Entity, U: Entity> ZippedArrays<'a, T, U> {
    /// Create a new zipped arrays view
    pub fn new(left: &'a Array<T>, right: &'a Array<U>) -> Result<Self, String> {
        if left.shape() != right.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", left.shape(), right.shape()));
        }
        Ok(Self { left, right })
    }
    
    /// Apply a function to each pair of elements
    pub fn map<V, F>(&self, f: F) -> Array<V>
    where
        V: Entity + ComplexField,
        F: Fn(T, U) -> V,
        T: Clone,
        U: Clone,
    {
        let (rows, cols) = self.left.shape();
        let result = Mat::from_fn(rows, cols, |i, j| {
            f(self.left.get(i, j).unwrap(), self.right.get(i, j).unwrap())
        });
        Array::from_faer(result)
    }
    
    /// Fold over pairs of elements
    pub fn fold<V, F>(&self, init: V, f: F) -> V
    where
        F: Fn(V, (T, U)) -> V,
        T: Clone,
        U: Clone,
    {
        let (rows, cols) = self.left.shape();
        let mut acc = init;
        for i in 0..rows {
            for j in 0..cols {
                let pair = (self.left.get(i, j).unwrap(), self.right.get(i, j).unwrap());
                acc = f(acc, pair);
            }
        }
        acc
    }
    
    /// Reduce pairs of elements
    pub fn reduce<F>(&self, f: F) -> Option<(T, U)>
    where
        F: Fn((T, U), (T, U)) -> (T, U),
        T: Clone,
        U: Clone,
    {
        let (rows, cols) = self.left.shape();
        if rows == 0 || cols == 0 {
            return None;
        }
        
        let mut acc = (self.left.get(0, 0).unwrap(), self.right.get(0, 0).unwrap());
        for i in 0..rows {
            for j in 0..cols {
                if i == 0 && j == 0 {
                    continue;
                }
                let pair = (self.left.get(i, j).unwrap(), self.right.get(i, j).unwrap());
                acc = f(acc, pair);
            }
        }
        Some(acc)
    }
}

/// A zipped view of two vectors for functional operations
pub struct ZippedVectors<'a, T: Entity, U: Entity> {
    left: &'a Vector<T>,
    right: &'a Vector<U>,
}

impl<'a, T: Entity, U: Entity> ZippedVectors<'a, T, U> {
    /// Create a new zipped vectors view
    pub fn new(left: &'a Vector<T>, right: &'a Vector<U>) -> Result<Self, String> {
        if left.len() != right.len() {
            return Err(format!("Length mismatch: {} vs {}", left.len(), right.len()));
        }
        Ok(Self { left, right })
    }
    
    /// Apply a function to each pair of elements
    pub fn map<V, F>(&self, f: F) -> Vector<V>
    where
        V: Entity + ComplexField,
        F: Fn(T, U) -> V,
        T: Clone,
        U: Clone,
    {
        let len = self.left.len();
        let result = Col::from_fn(len, |i| {
            f(self.left.get(i).unwrap(), self.right.get(i).unwrap())
        });
        Vector::from_faer(result)
    }
    
    /// Fold over pairs of elements
    pub fn fold<V, F>(&self, init: V, f: F) -> V
    where
        F: Fn(V, (T, U)) -> V,
        T: Clone,
        U: Clone,
    {
        let len = self.left.len();
        let mut acc = init;
        for i in 0..len {
            let pair = (self.left.get(i).unwrap(), self.right.get(i).unwrap());
            acc = f(acc, pair);
        }
        acc
    }
    
    /// Reduce pairs of elements
    pub fn reduce<F>(&self, f: F) -> Option<(T, U)>
    where
        F: Fn((T, U), (T, U)) -> (T, U),
        T: Clone,
        U: Clone,
    {
        let len = self.left.len();
        if len == 0 {
            return None;
        }
        
        let mut acc = (self.left.get(0).unwrap(), self.right.get(0).unwrap());
        for i in 1..len {
            let pair = (self.left.get(i).unwrap(), self.right.get(i).unwrap());
            acc = f(acc, pair);
        }
        Some(acc)
    }
}

// Extension methods for creating zipped views
impl<T: Entity> Array<T> {
    /// Create a zipped view with another array
    pub fn zip<'a, U: Entity>(&'a self, other: &'a Array<U>) -> Result<ZippedArrays<'a, T, U>, String> {
        ZippedArrays::new(self, other)
    }
}

impl<T: Entity> Vector<T> {
    /// Create a zipped view with another vector
    pub fn zip<'a, U: Entity>(&'a self, other: &'a Vector<U>) -> Result<ZippedVectors<'a, T, U>, String> {
        ZippedVectors::new(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArrayF64, VectorF64};
    
    #[test]
    fn test_array_map() {
        let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let doubled = arr.map(|x| x * 2.0);
        
        assert_eq!(doubled.get(0, 0), Some(2.0));
        assert_eq!(doubled.get(0, 1), Some(4.0));
        assert_eq!(doubled.get(1, 0), Some(6.0));
        assert_eq!(doubled.get(1, 1), Some(8.0));
    }
    
    #[test]
    fn test_array_fold() {
        let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let sum = arr.fold(0.0, |acc, x| acc + x);
        assert_eq!(sum, 10.0);
        
        let product = arr.fold(1.0, |acc, x| acc * x);
        assert_eq!(product, 24.0);
    }
    
    #[test]
    fn test_array_reduce() {
        let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let sum = arr.reduce(|a, b| a + b);
        assert_eq!(sum, Some(10.0));
        
        let max = arr.reduce(|a, b| if a > b { a } else { b });
        assert_eq!(max, Some(4.0));
    }
    
    #[test]
    fn test_array_zip_with() {
        let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = ArrayF64::from_slice(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
        
        let sum = a.zip_with(&b, |x, y| x + y).unwrap();
        assert_eq!(sum.get(0, 0), Some(3.0));
        assert_eq!(sum.get(0, 1), Some(5.0));
        assert_eq!(sum.get(1, 0), Some(7.0));
        assert_eq!(sum.get(1, 1), Some(9.0));
    }
    
    #[test]
    fn test_vector_functional_ops() {
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        
        // Test map
        let doubled = vec.map(|x| x * 2.0);
        assert_eq!(doubled.get(0), Some(2.0));
        assert_eq!(doubled.get(3), Some(8.0));
        
        // Test fold
        let sum = vec.fold(0.0, |acc, x| acc + x);
        assert_eq!(sum, 10.0);
        
        // Test reduce
        let max = vec.reduce(|a, b| if a > b { a } else { b });
        assert_eq!(max, Some(4.0));
    }
    
    #[test]
    fn test_vector_zip() {
        let v1 = VectorF64::from_slice(&[1.0, 2.0, 3.0]);
        let v2 = VectorF64::from_slice(&[4.0, 5.0, 6.0]);
        
        let sum = v1.zip_with(&v2, |x, y| x + y).unwrap();
        assert_eq!(sum.get(0), Some(5.0));
        assert_eq!(sum.get(1), Some(7.0));
        assert_eq!(sum.get(2), Some(9.0));
    }
    
    #[test]
    fn test_zipped_arrays() {
        let a = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = ArrayF64::from_slice(&[2.0, 3.0, 4.0, 5.0], 2, 2).unwrap();
        
        let zipped = a.zip(&b).unwrap();
        
        // Test map on zipped arrays
        let product = zipped.map(|x, y| x * y);
        assert_eq!(product.get(0, 0), Some(2.0));
        assert_eq!(product.get(1, 1), Some(20.0));
        
        // Test fold on zipped arrays
        let sum_of_products = zipped.fold(0.0, |acc, (x, y)| acc + x * y);
        assert_eq!(sum_of_products, 40.0); // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    }
    
    #[test]
    fn test_convenience_methods() {
        let arr = ArrayF64::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        
        assert_eq!(arr.sum(), 10.0);
        assert_eq!(arr.product(), 24.0);
        assert_eq!(arr.min(), Some(1.0));
        assert_eq!(arr.max(), Some(4.0));
        
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(vec.sum_elements(), 10.0);
        assert_eq!(vec.product(), 24.0);
        assert_eq!(vec.min(), Some(1.0));
        assert_eq!(vec.max(), Some(4.0));
    }
    
    #[test]
    fn test_vector_iterator_adaptors() {
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Test filter
        let evens = vec.filter(|x| x % 2.0 == 0.0);
        assert_eq!(evens.len(), 2);
        assert_eq!(evens.get(0), Some(2.0));
        assert_eq!(evens.get(1), Some(4.0));
        
        // Test take
        let first_three = vec.take(3);
        assert_eq!(first_three.len(), 3);
        assert_eq!(first_three.get(0), Some(1.0));
        assert_eq!(first_three.get(2), Some(3.0));
        
        // Test skip
        let last_three = vec.skip(2);
        assert_eq!(last_three.len(), 3);
        assert_eq!(last_three.get(0), Some(3.0));
        assert_eq!(last_three.get(2), Some(5.0));
        
        // Test enumerate
        let enumerated = vec.enumerate();
        assert_eq!(enumerated.len(), 5);
        assert_eq!(enumerated[0], (0, 1.0));
        assert_eq!(enumerated[4], (4, 5.0));
    }
    
    #[test]
    fn test_vector_higher_order_functions() {
        let vec = VectorF64::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Test all
        assert!(vec.all(|x| x > 0.0));
        assert!(!vec.all(|x| x > 3.0));
        
        // Test any
        assert!(vec.any(|x| x > 4.0));
        assert!(!vec.any(|x| x > 6.0));
        
        // Test find
        assert_eq!(vec.find(|x| x > 3.0), Some(4.0));
        assert_eq!(vec.find(|x| x > 6.0), None);
        
        // Test count
        assert_eq!(vec.count(|x| x > 3.0), 2);
        assert_eq!(vec.count(|x| x > 6.0), 0);
        
        // Test find_index
        assert_eq!(vec.find_index(|x| x > 3.0), Some(3));
        assert_eq!(vec.find_index(|x| x > 6.0), None);
        
        // Test scan (cumulative sum)
        let cumsum = vec.scan(0.0, |acc, x| acc + x);
        assert_eq!(cumsum.len(), 5);
        assert_eq!(cumsum.get(0), Some(1.0));  // 0 + 1
        assert_eq!(cumsum.get(1), Some(3.0));  // 1 + 2
        assert_eq!(cumsum.get(2), Some(6.0));  // 3 + 3
        assert_eq!(cumsum.get(3), Some(10.0)); // 6 + 4
        assert_eq!(cumsum.get(4), Some(15.0)); // 10 + 5
    }
}