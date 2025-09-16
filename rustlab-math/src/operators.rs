//! Ergonomic ^ operator for matrix multiplication (AI-optimized)
//! 
//! This module implements RustLab's signature `^` operator for natural mathematical syntax.
//! This is one of the most important AI-friendly features in RustLab.
//! 
//! # Core Pattern for AI Code Generation
//! ```rust
//! // Matrix multiplication (like NumPy @)
//! let C = A ^ B;        // Matrix × Matrix → Matrix
//! let y = A ^ x;        // Matrix × Vector → Vector  
//! let z = x ^ A;        // Vector × Matrix → Vector
//! let s = u ^ v;        // Vector × Vector → Scalar (dot product)
//! ```
//! 
//! # Critical Distinction for AI
//! - `^` = Mathematical multiplication (matrix/dot product)
//! - `*` = Element-wise multiplication (Hadamard product)
//! 
//! This prevents the most common AI hallucination in numerical computing.
//! 
//! # Dimension Rules
//! | Operation | Input Dimensions | Output | Description |
//! |-----------|------------------|--------|-------------|
//! | `A ^ B` | (m×n) × (n×p) | (m×p) | Matrix multiplication |
//! | `A ^ v` | (m×n) × (n×1) | (m×1) | Matrix-vector product |
//! | `v ^ A` | (1×m) × (m×n) | (1×n) | Vector-matrix product |
//! | `u ^ v` | (n×1) × (n×1) | scalar | Dot product |

use crate::{Array, Vector};
use faer_entity::Entity;
use faer_traits::ComplexField;
use std::ops::BitXor;

// Matrix ^ Matrix multiplication (same type)
impl<T: Entity> BitXor<Array<T>> for Array<T>
where
    T: ComplexField,
{
    type Output = Array<T>;
    
    fn bitxor(self, rhs: Array<T>) -> Array<T> {
        self.matmul(&rhs)
    }
}

impl<T: Entity> BitXor<&Array<T>> for Array<T>
where
    T: ComplexField,
{
    type Output = Array<T>;
    
    fn bitxor(self, rhs: &Array<T>) -> Array<T> {
        self.matmul(rhs)
    }
}

impl<T: Entity> BitXor<Array<T>> for &Array<T>
where
    T: ComplexField,
{
    type Output = Array<T>;
    
    fn bitxor(self, rhs: Array<T>) -> Array<T> {
        self.matmul(&rhs)
    }
}

impl<T: Entity> BitXor<&Array<T>> for &Array<T>
where
    T: ComplexField,
{
    type Output = Array<T>;
    
    fn bitxor(self, rhs: &Array<T>) -> Array<T> {
        self.matmul(rhs)
    }
}

// Matrix ^ Vector multiplication
impl<T: Entity> BitXor<Vector<T>> for Array<T>
where
    T: ComplexField,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: Vector<T>) -> Vector<T> {
        // Array * Vector = matrix-vector multiplication
        Vector {
            inner: &self.inner * &rhs.inner,
        }
    }
}

impl<T: Entity> BitXor<&Vector<T>> for Array<T>
where
    T: ComplexField,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: &Vector<T>) -> Vector<T> {
        Vector {
            inner: &self.inner * &rhs.inner,
        }
    }
}

impl<T: Entity> BitXor<Vector<T>> for &Array<T>
where
    T: ComplexField,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: Vector<T>) -> Vector<T> {
        Vector {
            inner: &self.inner * &rhs.inner,
        }
    }
}

impl<T: Entity> BitXor<&Vector<T>> for &Array<T>
where
    T: ComplexField,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: &Vector<T>) -> Vector<T> {
        Vector {
            inner: &self.inner * &rhs.inner,
        }
    }
}

// Vector ^ Matrix multiplication  
impl<T: Entity> BitXor<Array<T>> for Vector<T>
where
    T: ComplexField,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: Array<T>) -> Vector<T> {
        // Vector * Matrix = vector-matrix multiplication (row vector)
        // Convert row result back to column vector
        let result_row = self.inner.transpose() * &rhs.inner;
        Vector {
            inner: result_row.transpose().to_owned(),
        }
    }
}

impl<T: Entity> BitXor<&Array<T>> for Vector<T>
where
    T: ComplexField,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: &Array<T>) -> Vector<T> {
        let result_row = self.inner.transpose() * &rhs.inner;
        Vector {
            inner: result_row.transpose().to_owned(),
        }
    }
}

impl<T: Entity> BitXor<Array<T>> for &Vector<T>
where
    T: ComplexField,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: Array<T>) -> Vector<T> {
        let result_row = self.inner.transpose() * &rhs.inner;
        Vector {
            inner: result_row.transpose().to_owned(),
        }
    }
}

impl<T: Entity> BitXor<&Array<T>> for &Vector<T>
where
    T: ComplexField,
{
    type Output = Vector<T>;
    
    fn bitxor(self, rhs: &Array<T>) -> Vector<T> {
        let result_row = self.inner.transpose() * &rhs.inner;
        Vector {
            inner: result_row.transpose().to_owned(),
        }
    }
}

// Vector ^ Vector dot product (returns scalar)
impl<T: Entity> BitXor<Vector<T>> for Vector<T>
where
    T: ComplexField,
{
    type Output = T;
    
    fn bitxor(self, rhs: Vector<T>) -> T {
        self.dot(&rhs)
    }
}

impl<T: Entity> BitXor<&Vector<T>> for Vector<T>
where
    T: ComplexField,
{
    type Output = T;
    
    fn bitxor(self, rhs: &Vector<T>) -> T {
        self.dot(rhs)
    }
}

impl<T: Entity> BitXor<Vector<T>> for &Vector<T>
where
    T: ComplexField,
{
    type Output = T;
    
    fn bitxor(self, rhs: Vector<T>) -> T {
        self.dot(&rhs)
    }
}

impl<T: Entity> BitXor<&Vector<T>> for &Vector<T>
where
    T: ComplexField,
{
    type Output = T;
    
    fn bitxor(self, rhs: &Vector<T>) -> T {
        self.dot(rhs)
    }
}