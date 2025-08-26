# Rust Jupyter Notebook Best Practices

## Overview

This document summarizes the best approaches for creating Jupyter notebooks with Rust code that work seamlessly with rust-analyzer in VS Code and other development environments. These practices were developed through extensive testing with the rustlab-optimize notebook series.

## Key Challenges with Rust in Jupyter

### 1. **rust-analyzer Cell Independence**
- rust-analyzer treats each Jupyter cell as an independent Rust file
- Variables defined in one cell are not visible to rust-analyzer in subsequent cells
- This causes "undefined variable" warnings even when the notebook runs correctly

### 2. **evcxr Type Persistence Issues**
- evcxr cannot persist `impl Trait` types between cells
- Opaque types like `impl Fn(&[f64]) -> f64` cause "cannot be persisted" errors
- Some complex generic types also cannot be persisted across cell boundaries

### 3. **Lint Directive Issues**
- `#![allow(...)]` directives can trigger "lint level is defined here" warnings
- Some lint configurations conflict with Jupyter's execution model
- Overuse of allow directives can mask legitimate issues

### 4. **Variable Scope and Ownership**
- Cross-cell variable dependencies create ownership complexity
- Closure capture across cells can lead to move/borrow checker issues
- Variable shadowing warnings when reusing names across cells

## Proven Solutions

### ✅ **1. Self-Contained Cells**

**Problem**: Variables from previous cells not recognized by rust-analyzer

**Solution**: Make each code cell completely self-contained

```rust
// ❌ Bad: Depends on previous cell
let result = minimize(my_model)  // my_model undefined to rust-analyzer
    .from(&[1.0])
    .solve()?;

// ✅ Good: Self-contained cell
fn create_quadratic_model() -> impl Fn(&[f64]) -> f64 {
    move |params: &[f64]| {
        let x = params[0];
        (x - 2.0).powi(2)
    }
}

let model = create_quadratic_model();
let result = minimize(model)
    .from(&[1.0])
    .solve()?;
```

### ✅ **2. Function-Based Approach**

**Problem**: Complex inline closures with ownership issues

**Solution**: Extract logic into functions

```rust
// ❌ Bad: Complex inline closure
let data = vec64![1.0, 2.0, 3.0];
let data_clone = data.clone();
let model = move |params: &[f64]| {
    data_clone.iter().map(|&x| (x - params[0]).powi(2)).sum::<f64>()
};

// ✅ Good: Function-based approach
fn create_data() -> VectorF64 {
    vec64![1.0, 2.0, 3.0]
}

fn create_model(data: VectorF64) -> impl Fn(&[f64]) -> f64 {
    move |params: &[f64]| {
        data.iter().map(|&x| (x - params[0]).powi(2)).sum::<f64>()
    }
}

let data = create_data();
let model = create_model(data);
```

### ✅ **3. Avoid Problematic Lint Directives**

**Problem**: `#![allow(...)]` causing "lint level defined here" warnings

**Solution**: Use targeted approaches instead

```rust
// ❌ Bad: Global allow directive
#![allow(unused_variables, dead_code)]
let x = 5;
println!("Hello");

// ✅ Good: Ensure all variables are used
let x = 5;
let msg = format!("Value is {}", x);
println!("{}", msg);

// ✅ Alternative: Use underscore prefix for intentionally unused
let _temporary_data = vec64![1.0, 2.0, 3.0];
let result = compute_something();
```

### ✅ **4. Explicit String Formatting**

**Problem**: Direct `println!` with format strings can trigger lint warnings

**Solution**: Use explicit string formatting

```rust
// ❌ Potential lint issues
println!("Result: {:.3}", value);

// ✅ Good: Explicit formatting
let message = format!("Result: {:.3}", value);
println!("{}", message);

// ✅ Alternative: Store format string
let fmt_str = "Result: {:.3}";
println!("{}", format!(fmt_str, value));
```

### ✅ **5. Unique Variable Names Across Cells**

**Problem**: Variable shadowing warnings when reusing names

**Solution**: Use descriptive, unique names

```rust
// ❌ Bad: Reusing generic names
// Cell 1
let data = vec![1.0, 2.0];
let model = create_model(data);

// Cell 2  
let data = vec64![3.0, 4.0];  // Shadowing warning
let model = create_model(data);  // Shadowing warning

// ✅ Good: Unique descriptive names
// Cell 1
let reaction_data = vec64![1.0, 2.0];
let reaction_model = create_model(reaction_data);

// Cell 2
let calibration_data = vec64![3.0, 4.0];
let calibration_model = create_model(calibration_data);
```

### ✅ **6. Data Re-definition Pattern**

**Problem**: Need same data in multiple cells but avoid cross-cell dependencies

**Solution**: Re-define data in each cell that needs it

```rust
// ✅ Pattern: Data re-definition with helper functions
fn get_experimental_data() -> (Vec<f64>, Vec<f64>) {
    let x = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
    let y = vec64![0.05, 0.22, 0.40, 0.71, 1.41];
    (x, y)
}

// Cell 1: Unbounded optimization
let (concentrations, rates) = get_experimental_data();
let unbounded_model = create_rate_model(concentrations, rates);
let unbounded_result = minimize(unbounded_model).solve()?;

// Cell 2: Bounded optimization  
let (concentrations, rates) = get_experimental_data();
let bounded_model = create_rate_model(concentrations, rates);
let bounded_result = minimize(bounded_model)
    .bounds(&[0.01, 0.1], &[10.0, 3.0])
    .solve()?;
```

### ✅ **7. Handle Non-Persistable Types with Brace Wrapping**

**Problem**: `impl Fn` and other opaque types cannot be persisted by evcxr between cells

**Solution**: Wrap code in braces to prevent persistence attempts

```rust
// ❌ Bad: evcxr tries to persist the impl Fn type
fn create_model() -> impl Fn(&[f64]) -> f64 {
    |params| params[0].powi(2)
}

let model = create_model();  // Error: cannot persist impl Fn
let result = minimize(model).solve()?;

// ✅ Good: Brace wrapping prevents persistence
{
    fn create_model() -> impl Fn(&[f64]) -> f64 {
        |params| params[0].powi(2)
    }

    let model = create_model();  // Local scope, not persisted
    let result = minimize(model).solve()?;
    
    println!("Result: {:?}", result.solution);
}

// ✅ Alternative: Use Box<dyn Fn> for persistable closures
fn create_model_boxed() -> Box<dyn Fn(&[f64]) -> f64> {
    Box::new(|params| params[0].powi(2))
}

let model: Box<dyn Fn(&[f64]) -> f64> = create_model_boxed();
let result = minimize(model).solve()?;
```

### **When to Use Each Approach**

| Scenario | Recommended Solution | Reason |
|----------|---------------------|---------|
| Self-contained demos | Brace wrapping `{}` | Simplest, no performance overhead |
| Need to persist functions | `Box<dyn Fn>` | Allows cross-cell usage |
| Complex closures | Brace wrapping `{}` | Avoids type complexity |
| Performance critical | `impl Fn` in braces | Zero-cost abstraction preserved |
| Educational notebooks | Brace wrapping `{}` | Keeps focus on algorithm, not Rust internals |

## Complete Example: Before and After

### ❌ **Before: Problematic Approach**

```rust
// Cell 1
#![allow(unused_variables, dead_code)]
let data = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
let rates = vec64![0.05, 0.22, 0.40, 0.71, 1.41];
println!("📊 Data loaded: {} points", data.len());

// Cell 2  
#![allow(unused_variables)]
let data_clone = data.clone();  // rust-analyzer: data not found
let rates_clone = rates.clone();  // rust-analyzer: rates not found

fn create_model() -> impl Fn(&[f64]) -> f64 {
    move |params: &[f64]| {
        data_clone.iter().zip(rates_clone.iter())  // Multiple issues here
            .map(|(&x, &y)| (y - params[0] * x.powf(params[1])).powi(2))
            .sum::<f64>()
    }
}

let model = create_model();  // evcxr: cannot persist impl Fn
println!("Model created");  // Lint warning about format string

// Cell 3
let result = minimize(model)  // rust-analyzer: model not found
    .from(&[1.0, 1.0])
    .solve()?;
```

### ✅ **After: Best Practices Applied**

```rust
// Cell 1: Setup with helper functions (brace-wrapped for persistence safety)
{
    fn get_reaction_data() -> (VectorF64, VectorF64) {
        let concentrations = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let rates = vec64![0.05, 0.22, 0.40, 0.71, 1.41];
        (concentrations, rates)
    }

    let setup_msg = "Helper functions defined successfully";
    println!("{}", setup_msg);
}

// Cell 2: Unbounded optimization (self-contained with brace wrapping)
{
    fn get_reaction_data() -> (VectorF64, VectorF64) {
        let concentrations = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let rates = vec64![0.05, 0.22, 0.40, 0.71, 1.41];
        (concentrations, rates)
    }

    fn create_kinetic_model(conc: VectorF64, rates: VectorF64) -> impl Fn(&[f64]) -> f64 {
        move |params: &[f64]| {
            conc.iter().zip(rates.iter())
                .map(|(&x, &y)| (y - params[0] * x.powf(params[1])).powi(2))
                .sum::<f64>()
        }
    }

    let (concentrations, rates) = get_reaction_data();
    let unbounded_model = create_kinetic_model(concentrations, rates);

    let unbounded_result = minimize(unbounded_model)
        .from(&[1.0, 1.0])
        .solve()?;

    let unbounded_k = unbounded_result.solution[0];
    let unbounded_n = unbounded_result.solution[1];
    let unbounded_msg = format!("Unbounded: k = {:.3}, n = {:.3}", unbounded_k, unbounded_n);
    println!("{}", unbounded_msg);
}

// Cell 3: Bounded optimization (self-contained with brace wrapping)
{
    fn get_reaction_data() -> (VectorF64, VectorF64) {
        let concentrations = vec64![0.1, 0.5, 1.0, 2.0, 5.0];
        let rates = vec64![0.05, 0.22, 0.40, 0.71, 1.41];
        (concentrations, rates)
    }

    fn create_kinetic_model(conc: VectorF64, rates: VectorF64) -> impl Fn(&[f64]) -> f64 {
        move |params: &[f64]| {
            conc.iter().zip(rates.iter())
                .map(|(&x, &y)| (y - params[0] * x.powf(params[1])).powi(2))
                .sum::<f64>()
        }
    }

    let (concentrations_bounded, rates_bounded) = get_reaction_data();
    let bounded_model = create_kinetic_model(concentrations_bounded, rates_bounded);

    let bounded_result = minimize(bounded_model)
        .from(&[1.0, 1.0])
        .bounds(&[0.01, 0.1], &[10.0, 3.0])
        .solve()?;

    let bounded_k = bounded_result.solution[0];
    let bounded_n = bounded_result.solution[1];
    let bounded_msg = format!("Bounded: k = {:.3}, n = {:.3}", bounded_k, bounded_n);
    println!("{}", bounded_msg);
}
```

## Additional Tips

### **Dependencies and Imports**

**Key Discovery**: Import dependencies and types at the top level (outside braces) in the setup cell - they persist across all cells!

```rust
// ✅ Good: Setup cell with persistent imports
:dep rustlab-math = { path = "../../rustlab-math" }
:dep rustlab-optimize = { path = ".." }

// These imports persist across all cells
use rustlab_optimize::prelude::*;
use rustlab_math::{VectorF64, vec64};

// Optional: Test code in braces (doesn't persist)
{
    let test = vec64![1.0, 2.0, 3.0];
    println!("Setup complete! Vector length: {}", test.len());
}
```

```rust
// ❌ Bad: Wrapping imports in braces makes them local
{
    use rustlab_optimize::prelude::*;  // Won't persist to other cells!
    use rustlab_math::vec64;            // Won't persist to other cells!
}
```

### ✅ **8. Import Strategy Summary**

**The optimal setup cell structure:**

```rust
// Setup Cell - imports persist across all cells
:dep rustlab-math = { path = "../../rustlab-math" }
:dep rustlab-optimize = { path = ".." }

// Top-level imports - these persist!
use rustlab_optimize::prelude::*;  // All optimization functions
use rustlab_math::{VectorF64, vec64, array64};  // Math types and macros

// Optional: Brace-wrapped test code
{
    // This code runs but variables don't persist
    let test = vec64![1.0, 2.0, 3.0];
    println!("Setup complete!");
}
```

**Key Insights:**
- **Top-level imports persist** across all notebook cells
- **Brace-wrapped code is isolated** - variables don't persist
- **Best of both worlds**: Import once, use everywhere, but avoid persistence errors
- **Prelude imports**: Most functions available via `prelude::*`
- **Math-first macros**: `vec64!`, `array64!` available after import

### **Error Handling**

```rust
// ✅ Good: Explicit error handling in notebooks
match minimize(model).solve() {
    Ok(result) => {
        let success_msg = format!("Optimization succeeded: {:?}", result.solution);
        println!("{}", success_msg);
    }
    Err(e) => {
        let error_msg = format!("Optimization failed: {}", e);
        println!("{}", error_msg);
    }
}
```

### **Documentation Cells**

```markdown
## Clear Section Headers

Use markdown cells to:
- Explain the mathematical concepts
- Provide context for code examples  
- Include equations using LaTeX: $f(x) = ax^2 + bx + c$
- Link to related notebooks and documentation
```

## Performance Considerations

1. **Avoid Large Data in Closures**: Use references or smaller data structures when possible
2. **Function Reuse**: Define helper functions once and reuse across cells
3. **Lazy Evaluation**: Consider using iterators instead of collecting into vectors immediately
4. **Memory Management**: Be mindful of cloning large datasets across cells

## Testing Your Notebooks

### **rust-analyzer Compatibility Check**
1. Open notebook in VS Code with rust-analyzer extension
2. Verify no red underlines or warnings in code cells
3. Check that autocomplete works within each cell
4. Ensure no "undefined variable" or "unused import" warnings

### **Execution Verification**
1. Run all cells from top to bottom
2. Verify expected output in each cell
3. Test with "Restart and Run All" to ensure cell independence
4. Check that mathematical results are correct

### **Cross-Platform Testing**
1. Test on different operating systems if needed
2. Verify dependency paths are correct
3. Check that relative imports work from notebook directory

## Conclusion

Following these best practices will create Rust Jupyter notebooks that:
- Work seamlessly with rust-analyzer and IDE tooling
- Execute reliably in evcxr/Jupyter environments  
- Are maintainable and easy to understand
- Execute reliably across different environments
- Provide clear educational value with minimal friction

### **Key Principles Summary**

1. **Cell Independence**: Each cell should be analyzable by rust-analyzer as a standalone piece of Rust code
2. **Type Persistence Awareness**: Use brace wrapping for `impl Fn` types that cannot be persisted by evcxr
3. **Self-Contained Design**: Re-define functions and data in each cell to avoid cross-cell dependencies
4. **Explicit over Implicit**: Use explicit string formatting and variable usage to avoid lint warnings

The most important discovery from extensive testing is that **brace wrapping `{}`** solves both rust-analyzer compatibility and evcxr persistence issues simultaneously, making it the preferred approach for most Rust notebook scenarios.