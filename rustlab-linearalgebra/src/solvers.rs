//! Linear system solvers using faer's optimized algorithms

use crate::{Array, Vector, LinearAlgebraError, Result};
use faer::{linalg, Entity};
use rustlab_math::{ArrayF64, ArrayF32, ArrayC64, ArrayC32, VectorF64, VectorF32, VectorC64, VectorC32, array64, vec64};

/// Linear system solver trait
pub trait LinearSolver<T: Entity> {
    /// Solve Ax = b for x
    fn solve(&self, b: &Vector<T>) -> Result<Vector<T>>;
    
    /// Solve AX = B for X (multiple right-hand sides)
    fn solve_multiple(&self, b: &Array<T>) -> Result<Array<T>>;
}

/// Solve linear system Ax = b using LU decomposition
pub fn solve<T>(a: &Array<T>, b: &Vector<T>) -> Result<Vector<T>>
where
    T: Entity + Copy,
{
    // Check dimensions
    let (rows, cols) = (a.nrows(), a.ncols());
    if rows != cols {
        return Err(LinearAlgebraError::NotSquare { rows, cols });
    }
    
    if b.len() != rows {
        return Err(LinearAlgebraError::DimensionMismatch {
            expected: format!("vector length {}", rows),
            actual: format!("vector length {}", b.len()),
        });
    }
    
    // Use faer's direct solve method
    let solution = linalg::solve::solve(
        a.as_faer(),
        b.as_faer(),
        faer::Parallelism::None,
        faer::dyn_stack::PodStack::new(&mut []),
        Default::default(),
    ).map_err(|_| LinearAlgebraError::Singular)?;
    
    Ok(Vector::from_faer(solution))
}

/// Solve triangular system using forward/backward substitution
pub fn solve_triangular<T>(
    a: &Array<T>, 
    b: &Vector<T>, 
    lower: bool
) -> Result<Vector<T>>
where
    T: Entity + Copy,
{
    // Check dimensions
    let (rows, cols) = (a.nrows(), a.ncols());
    if rows != cols {
        return Err(LinearAlgebraError::NotSquare { rows, cols });
    }
    
    if b.len() != rows {
        return Err(LinearAlgebraError::DimensionMismatch {
            expected: format!("vector length {}", rows),
            actual: format!("vector length {}", b.len()),
        });
    }
    
    // Use faer's triangular solve
    let solution = if lower {
        linalg::solve::solve_lower_triangular(
            a.as_faer(),
            b.as_faer(),
            faer::Parallelism::None,
            faer::dyn_stack::PodStack::new(&mut []),
        ).map_err(|_| LinearAlgebraError::Singular)?
    } else {
        linalg::solve::solve_upper_triangular(
            a.as_faer(),
            b.as_faer(),
            faer::Parallelism::None,
            faer::dyn_stack::PodStack::new(&mut []),
        ).map_err(|_| LinearAlgebraError::Singular)?
    };
    
    Ok(Vector::from_faer(solution))
}

// Implement LinearSolver for Array types

impl<T> LinearSolver<T> for Array<T>
where
    T: Entity + Copy,
{
    fn solve(&self, b: &Vector<T>) -> Result<Vector<T>> {
        solve(self, b)
    }
    
    fn solve_multiple(&self, b: &Array<T>) -> Result<Array<T>> {
        // Check dimensions
        let (rows, cols) = (self.nrows(), self.ncols());
        if rows != cols {
            return Err(LinearAlgebraError::NotSquare { rows, cols });
        }
        
        if b.nrows() != rows {
            return Err(LinearAlgebraError::DimensionMismatch {
                expected: format!("matrix with {} rows", rows),
                actual: format!("matrix with {} rows", b.nrows()),
            });
        }
        
        // Use faer's batch solve method
        let solution = linalg::solve::solve(
            self.as_faer(),
            b.as_faer(),
            faer::Parallelism::None,
            faer::dyn_stack::PodStack::new(&mut []),
            Default::default(),
        ).map_err(|_| LinearAlgebraError::Singular)?;
        
        Ok(Array::from_faer(solution))
    }
}

// Extension trait for convenience solver operations
pub trait LinearSystemOps<T: Entity> {
    /// Solve linear system Ax = b
    fn solve_system(&self, b: &Vector<T>) -> Result<Vector<T>>;
    
    /// Solve multiple systems AX = B
    fn solve_systems(&self, b: &Array<T>) -> Result<Array<T>>;
    
    /// Solve lower triangular system
    fn solve_lower_triangular(&self, b: &Vector<T>) -> Result<Vector<T>>;
    
    /// Solve upper triangular system
    fn solve_upper_triangular(&self, b: &Vector<T>) -> Result<Vector<T>>;
}

impl LinearSystemOps<f64> for ArrayF64 {
    fn solve_system(&self, b: &VectorF64) -> Result<VectorF64> {
        self.solve(b)
    }
    
    fn solve_systems(&self, b: &ArrayF64) -> Result<ArrayF64> {
        self.solve_multiple(b)
    }
    
    fn solve_lower_triangular(&self, b: &VectorF64) -> Result<VectorF64> {
        solve_triangular(self, b, true)
    }
    
    fn solve_upper_triangular(&self, b: &VectorF64) -> Result<VectorF64> {
        solve_triangular(self, b, false)
    }
}

impl LinearSystemOps<f32> for ArrayF32 {
    fn solve_system(&self, b: &VectorF32) -> Result<VectorF32> {
        self.solve(b)
    }
    
    fn solve_systems(&self, b: &ArrayF32) -> Result<ArrayF32> {
        self.solve_multiple(b)
    }
    
    fn solve_lower_triangular(&self, b: &VectorF32) -> Result<VectorF32> {
        solve_triangular(self, b, true)
    }
    
    fn solve_upper_triangular(&self, b: &VectorF32) -> Result<VectorF32> {
        solve_triangular(self, b, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_solve_2x2_system() {
        // System: 2x + y = 5, x + y = 3
        // Solution: x = 2, y = 1
        let a = array64![
            [2.0, 1.0],
            [1.0, 1.0]
        ];
        
        let b = vec64![5.0, 3.0];
        
        let x = a.solve_system(&b).unwrap();
        
        assert_relative_eq!(x.get(0).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(x.get(1).unwrap(), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_solve_triangular_system() {
        // Lower triangular system
        let l = array64![
            [2.0, 0.0],
            [1.0, 3.0]
        ];
        
        let b = vec64![4.0, 7.0];
        
        let x = l.solve_lower_triangular(&b).unwrap();
        
        // Verify solution: L * x = b
        assert_relative_eq!(x.get(0).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(x.get(1).unwrap(), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_solve_multiple_systems() {
        let a = array64![
            [2.0, 1.0],
            [1.0, 1.0]
        ];
        
        let b = array64![
            [5.0, 7.0],  // Two right-hand sides
            [3.0, 4.0]
        ];
        
        let x = a.solve_systems(&b).unwrap();
        
        // First solution: x = 2, y = 1
        assert_relative_eq!(x.get(0, 0).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(x.get(1, 0).unwrap(), 1.0, epsilon = 1e-10);
        
        // Second solution: x = 3, y = 1  
        assert_relative_eq!(x.get(0, 1).unwrap(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(x.get(1, 1).unwrap(), 1.0, epsilon = 1e-10);
    }
}