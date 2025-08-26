use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustlab_math::{Array, Vector};

fn array_addition(c: &mut Criterion) {
    let size = 1000;
    
    c.bench_function("array_addition_1000x1000", |bencher| {
        bencher.iter(|| {
            let a = Array::ones(size, size);
            let b = Array::ones(size, size);
            let result = a + b;
            black_box(result)
        })
    });
}

fn array_scalar_mult(c: &mut Criterion) {
    let size = 1000;
    
    c.bench_function("array_scalar_mult_1000x1000", |bencher| {
        bencher.iter(|| {
            let a = Array::ones(size, size);
            let result = a * 2.0;
            black_box(result)
        })
    });
}

fn array_chained_ops(c: &mut Criterion) {
    let size = 1000;
    
    c.bench_function("array_chained_ops_1000x1000", |bencher| {
        bencher.iter(|| {
            let a = Array::zeros(size, size);
            let b = Array::ones(size, size);
            let result = a + b * 2.0;
            black_box(result)
        })
    });
}

fn vector_dot_product(c: &mut Criterion) {
    let size = 100000;
    
    c.bench_function("vector_dot_product_100k", |bencher| {
        bencher.iter(|| {
            let a = Vector::ones(size);
            let b = Vector::ones(size);
            let result = a.dot(&b);
            black_box(result)
        })
    });
}

fn vector_addition(c: &mut Criterion) {
    let size = 100000;
    
    c.bench_function("vector_addition_100k", |bencher| {
        bencher.iter(|| {
            let a = Vector::ones(size);
            let b = Vector::ones(size);
            let result = a + b;
            black_box(result)
        })
    });
}

criterion_group!(
    benches,
    array_addition,
    array_scalar_mult,
    array_chained_ops,
    vector_dot_product,
    vector_addition
);
criterion_main!(benches);