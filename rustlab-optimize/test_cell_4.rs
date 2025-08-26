fn main() {
    // Exact copy of cell 4 content
    let concentrations = vec![0.1, 0.5, 1.0, 2.0, 5.0];
    let rates = vec![0.05, 0.22, 0.40, 0.71, 1.41];

    println!("📊 Reaction Rate Data:");
    println!("[A] (M)\tRate (M/s)");
    for (c_val, r_val) in concentrations.iter().zip(rates.iter()) {
        println!("{:.1}\t{:.2}", c_val, r_val);
    }

    // Define the model: rate = k * [A]^n
    let concentrations_clone = concentrations.clone();
    let rates_clone = rates.clone();
    let rate_model = move |params: &[f64]| {
        let k_param = params[0];
        let n_param = params[1];
        concentrations_clone.iter().zip(rates_clone.iter())
            .map(|(&c_pt, &r_pt)| {
                let predicted = k_param * (c_pt as f64).powf(n_param);
                (r_pt - predicted).powi(2)
            })
            .sum::<f64>()
    };

    println!("\n🔬 Model: rate = k * [A]^n");
    
    // Use the closure to avoid unused warning
    let _test = rate_model(&[1.0, 1.0]);
}