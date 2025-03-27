use ndarray::{Array, Array1, Array2, ArrayView1, Axis, s};
use ndarray_linalg::{c64, Trace};
use rayon::prelude::*;
use std::f64;
use std::sync::{Arc, Mutex};
use itertools::Itertools;

// Complex number type alias
type Complex = c64;

// A 2x2 complex matrix using ndarray
#[derive(Clone, Debug)]
pub struct Matrix2x2 {
    data: Array2<Complex>,
}

impl Matrix2x2 {
    pub fn new(a: Complex, b: Complex, c: Complex, d: Complex) -> Self {
        Matrix2x2 { 
            data: Array2::from_shape_vec((2, 2), vec![a, b, c, d]).unwrap() 
        }
    }

    pub fn zero() -> Self {
        Matrix2x2 { 
            data: Array2::zeros((2, 2))
        }
    }

    pub fn identity() -> Self {
        Matrix2x2 { 
            data: Array2::eye(2)
        }
    }

    pub fn from_params(x: f64, y: f64, z: f64) -> Self {
        let y_complex = Complex::new(y, 0.0);
        Matrix2x2::new(
            Complex::new(x, 0.0),
            y_complex,
            y_complex.conj(),
            Complex::new(z, 0.0),
        )
    }

    pub fn to_params(&self) -> (f64, f64, f64) {
        (self.data[[0, 0]].re, self.data[[0, 1]].re, self.data[[1, 1]].re)
    }

    pub fn add(&self, other: &Matrix2x2) -> Matrix2x2 {
        Matrix2x2 {
            data: &self.data + &other.data
        }
    }

    pub fn sub(&self, other: &Matrix2x2) -> Matrix2x2 {
        Matrix2x2 {
            data: &self.data - &other.data
        }
    }

    pub fn mul(&self, scalar: f64) -> Matrix2x2 {
        Matrix2x2 {
            data: &self.data * Complex::new(scalar, 0.0)
        }
    }

    pub fn is_positive_semidefinite(&self, tol: f64) -> bool {
        // For 2x2 matrices: PSD iff det ≥ 0 and trace > 0
        let det = self.data[[0, 0]] * self.data[[1, 1]] - self.data[[0, 1]] * self.data[[1, 0]];
        let trace = self.data[[0, 0]] + self.data[[1, 1]];

        det.re >= -tol && trace.re >= -tol
    }

    // Sum a slice of matrices
    pub fn sum(matrices: &[Matrix2x2]) -> Matrix2x2 {
        if matrices.is_empty() {
            return Matrix2x2::zero();
        }

        let mut sum = matrices[0].clone();
        for mat in &matrices[1..] {
            sum.data = &sum.data + &mat.data;
        }
        sum
    }

    pub fn trace_product(&self, other: &Matrix2x2) -> Complex {
        // Compute trace(self * other)
        (self.data.dot(&other.data)).trace().unwrap()
    }

    pub fn distance(&self, other: &Matrix2x2) -> f64 {
        // Frobenius norm of the difference
        let diff = &self.data - &other.data;
        diff.iter()
            .map(|&x| (x * x.conj()).re)
            .sum::<f64>()
            .sqrt()
    }
}

// Represents a POVM search candidate
#[derive(Clone, Debug)]
pub struct PovmCandidate {
    pub elements: Vec<Matrix2x2>,
    pub max_objective_value: f64,
    pub min_objective_value: f64,
}

pub fn truncate_povm_candidates(candidates: Vec<PovmCandidate>) -> Vec<PovmCandidate> {
    let max_of_mins = candidates.iter().map(|c| c.min_objective_value).fold(f64::NEG_INFINITY, f64::max);
    let mut sorted_candidates = candidates;
    sorted_candidates.sort_by(|a, b| {
        a.max_objective_value.partial_cmp(&b.max_objective_value).unwrap()
    });
    // Remove all candidates with max value less than the max of the mins
    sorted_candidates.retain(|c| c.max_objective_value > max_of_mins);
    sorted_candidates
}

// Generate a grid with specified granularity
fn generate_grid(start: f64, step_size: f64, steps: usize) -> Vec<f64> {
    (0..steps).map(|i| start + i as f64 * step_size).collect()
}

pub fn initial_grid(
    current_povms: &[Matrix2x2],
    eps: f64,
    steps: usize
) -> Vec<Matrix2x2> {
    // Get bounds for next element based on current set
    let identity = Matrix2x2::identity();
    let remainder = identity.sub(&Matrix2x2::sum(current_povms));

    // Extract bounds for parameters
    let x_max = remainder.data[[0, 0]].re;
    let z_max = remainder.data[[1, 1]].re;

    // Create parameter grids
    let x_grid = generate_grid(0.0, eps, steps);
    let y_grid = vec![-1.0, 1.0];
    let z_grid = generate_grid(0.0, eps, steps);

    // Generate all combinations and filter valid ones

    // Use Rayon for parallel processing
    let all_combinations: Vec<(f64, f64, f64)> = x_grid.iter()
        .flat_map(|&x| y_grid.iter().map(move |&y| (x, y)))
        .flat_map(|(x, y)| z_grid.iter().map(move |&z| (x, x.sqrt() * z.sqrt() * y, z)))
        .collect();

    all_combinations.par_iter()
        .filter_map(|&(x, y, z)| {
            let element = Matrix2x2::from_params(x, y, z);

            // Check if element is positive semidefinite
            if !element.is_positive_semidefinite(1e-6) {
                return None;
            }
            Some(element)
        })
    .collect()
}

// Find valid POVM elements for the current step
pub fn discretize_povms(
    current_povms: &[Matrix2x2],
    eps: f64,
    prior_eps: f64
) -> Vec<Vec<Matrix2x2>> {
    let n_steps = (prior_eps / eps).ceil();
    // For each POVM in current POVMs, discretize
    let discretizations: Vec<Vec<(f64, f64, f64)>> = current_povms.iter().map(|m| {
        let (x, y , z) = m.to_params();
        // Generate a grid around the current element of width prior_eps and step size eps
        let x_grid = generate_grid(x, eps, n_steps as usize);
        let y_grid = vec![-1.0, 1.0]; 
        let z_grid = generate_grid(z, eps, n_steps as usize);
        let v = x_grid.iter()
            .flat_map(|&x| y_grid.iter().map(move |&y| (x, y)))
            .flat_map(|(x, y)| z_grid.iter().map(move |&z| (x, x.sqrt() * z.sqrt() * y, z)))
            .collect::<Vec<(f64, f64, f64)>>();
        v
    }).collect();
    let identity = Matrix2x2::identity();

    let discretizations_filtered: Vec<Vec<Matrix2x2>> = discretizations.par_iter().map(|combinations| {
        combinations.iter().filter_map(|&(x, y, z)| {
            let element = Matrix2x2::from_params(x, y, z);

            // Check if element is positive semidefinite
            if !element.is_positive_semidefinite(1e-6) {
                return None;
            }
            return Some(element);

            // Check if adding this element still allows a valid POVM
            let sum_with_element = Matrix2x2::sum(current_povms).add(&element);
            let diff = identity.sub(&sum_with_element);

            if diff.is_positive_semidefinite(1e-6) {
                Some(element)
            } else {
                None
            }
        }).collect()
    }).collect();
    discretizations_filtered
}

// Create quantum state from angle
pub fn state_from_angle(theta: f64) -> Matrix2x2 {
    let cos_theta = f64::cos(theta);
    let sin_theta = f64::sin(theta);

    let v = Array1::from_vec(vec![
        Complex::new(cos_theta, 0.0),
        Complex::new(sin_theta, 0.0),
    ]);

    // Outer product |v><v|
    let mut data = Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            data[[i, j]] = v[i] * v[j].conj();
        }
    }

    Matrix2x2 { data }
}

// Calculate expectation values for POVMs against states
pub fn calculate_expectation_values(povms: &[Matrix2x2], states: &[Matrix2x2]) -> Vec<f64> {
    let mut results = Vec::with_capacity(states.len() * povms.len());

    for state in states {
        for povm in povms {
            // Tr(POVM * state) - more efficient with ndarray
            let expectation = povm.trace_product(state);
            results.push(expectation.re);
        }
    }

    results
}

// Generate all 2x2 matrices with +- eps in the diagonal entries and +- eps +- i eps in the
// off-diagonal entries
fn gen_eps_extremal(eps: f64) -> Vec<Matrix2x2> {
    let z = 0.0;
    vec![
        Matrix2x2::from_params(eps, eps, z),
        Matrix2x2::from_params(eps, eps, eps),
        Matrix2x2::from_params(eps, z, z),
        Matrix2x2::from_params(eps, z, eps),
        Matrix2x2::from_params(z, eps, z),
        Matrix2x2::from_params(z, eps, eps),
        Matrix2x2::from_params(z, z, z),
        Matrix2x2::from_params(z, z, eps),
    ]
}

// Calculate collision mutual information
fn calculate_collision_mutual_info_fuzzy(joint_prob: &Array2<f64>, 
    fuzzy_traces: &[f64],
    fuzzy_table: &Array2<f64>,
    eps: f64
) -> (f64, f64) {
    // Marginal probabilities
    let px = joint_prob.sum_axis(Axis(1)); // P(X)
    let py = joint_prob.sum_axis(Axis(0)); // P(Y)

    // Denominator: E[Pr[X]] = sum_x P(X=x)^2
    let denom = px.iter().map(|&x| x * x).sum::<f64>();

    // Numerator: E[Pr[X|Y]] = sum_{x,y} P(X=x,Y=y)^2 / P(Y=y)
    let mut numer = 0.0;
    let mut numer_min = 0.0;
    for x in 0..joint_prob.shape()[0] { // states here
        for y in 0..joint_prob.shape()[1] { // POVM index here
            let mut max_val = 0.0;
            let p_xy = joint_prob[[x, y]];
            let p_y = py[y];
            let min_val = if p_y > 1e-10 { (p_xy * p_xy) / (p_y) } else { 0.0 };
            for (i, deno_f) in fuzzy_traces.iter().enumerate() {
                let num_f = fuzzy_table[[x, i]];

                let p_xy_adj = (p_xy + num_f).min(px[x]);

                assert!(*deno_f >= 0.0, "Fuzzy trace should be non-negative");
                let v = if p_y + deno_f > 1e-10 { (p_xy_adj * p_xy_adj) / (p_y + deno_f)} else { 0.0 };
                let cmp = v;
                if cmp >= max_val && p_xy_adj <= 1.0 {
                    max_val = cmp;
                }
            }
            numer += max_val;
            numer_min += min_val;
        }
    }
    // Adjust for epsilon
    let numer_adj : f64 =  if numer >= 1.0 { 1.0 } else { numer };
    let numer_min_adj : f64 =  if numer_min >= 1.0 { 1.0 } else { numer_min };


    // Prevent NaN or inf values
    let upper = if numer_adj <= 0.0 || denom <= 0.0 {
        0.0
    } else {
        numer_adj.log2() - denom.log2()
    };
    let lower = if numer_min_adj <= 0.0 || denom <= 0.0 {
        0.0
    } else {
        numer_min_adj.log2() - denom.log2()
    };

    (lower, upper)
}


// Calculate collision mutual information
pub fn calculate_collision_mutual_info_eps(joint_prob: &Array2<f64>, eps: f64) -> f64 {
    // Marginal probabilities
    let px = joint_prob.sum_axis(Axis(1)); // P(X)
    let py = joint_prob.sum_axis(Axis(0)); // P(Y)

    // Denominator: E[Pr[X]] = sum_x P(X=x)^2
    let denom = px.iter().map(|&x| x * x).sum::<f64>();

    // Numerator: E[Pr[X|Y]] = sum_{x,y} P(X=x,Y=y)^2 / P(Y=y)
    let mut numer = 0.0;
    for x in 0..joint_prob.shape()[0] {
        for y in 0..joint_prob.shape()[1] {
            let p_xy = joint_prob[[x, y]];
            let p_y = py[y];
            let p_xy_adj = p_xy;//+ 4.0 * (1.0 / 16.0) * eps;
            if p_y > 1e-10 {
                numer += (p_xy_adj * p_xy_adj) / (p_y);// - 2.0 * eps);
            }
        }
    }
    // Adjust for epsilon
    let numer_adj : f64 =  if numer >= 1.0 { 1.0 } else { numer };


    // Prevent NaN or inf values
    if numer_adj <= 0.0 || denom <= 0.0 {
        return 0.0;
    }

    numer_adj.log2() - denom.log2()
}



// Calculate collision mutual information
pub fn calculate_collision_mutual_info(joint_prob: &Array2<f64>) -> f64 {
    // Marginal probabilities
    let px = joint_prob.sum_axis(Axis(1)); // P(X)
    let py = joint_prob.sum_axis(Axis(0)); // P(Y)

    // Numerator: E[Pr[X]] = sum_x P(X=x)^2
    let denom = px.iter().map(|&x| x * x).sum::<f64>();

    // Denominator: E[Pr[X|Y]] = sum_{x,y} P(X=x,Y=y)^2 / P(Y=y)
    let mut numer = 0.0;
    for x in 0..joint_prob.shape()[0] {
        for y in 0..joint_prob.shape()[1] {
            let p_xy = joint_prob[[x, y]];
            let p_y = py[y];
            if p_y > 1e-10 {
                numer += (p_xy * p_xy) / p_y;
            }
        }
    }

    // Prevent NaN or inf values
    if numer <= 0.0 || denom <= 0.0 {
        return 0.0;
    }

    numer.log2() - denom.log2()
}


// Objective function for collision mutual information
fn obj_function_collision_fuzzy(
    expec_vals: &[f64], 
    n_states: usize, 
    n_povms: usize, 
    calc_type: &str,
    fuzzy_traces: &[f64],
    fuzzy_state_traces: &[f64],
    eps: f64
) -> (f64, f64) {
    // Reshape to a 2D array for easier indexing
    let fuzzy_state_traces = Array2::from_shape_vec((n_states, fuzzy_traces.len()), fuzzy_state_traces.to_vec())
        .expect("Expectation values should reshape correctly");

    // Reshape to a 2D array for easier indexing
    let expec_2d = Array2::from_shape_vec((n_states, n_povms), expec_vals.to_vec())
        .expect("Expectation values should reshape correctly");

    // Create probability tables using more efficient ndarray operations
    let mut prob_tables_b0 = Array2::zeros((2, n_povms));
    let mut prob_tables_b1 = Array2::zeros((2, n_povms));

    // First basis (b0)
    for j in 0..n_povms {
        prob_tables_b0[[0, j]] = expec_2d[[0, j]] + expec_2d[[1, j]];
        prob_tables_b0[[1, j]] = expec_2d[[2, j]] + expec_2d[[3, j]];

        // Second basis (b1)
        prob_tables_b1[[0, j]] = expec_2d[[0, j]] + expec_2d[[2, j]];
        prob_tables_b1[[1, j]] = expec_2d[[1, j]] + expec_2d[[3, j]];
    }

    let mut fuzzy_table_b0 = Array2::zeros((2, fuzzy_traces.len()));
    let mut fuzzy_table_b1 = Array2::zeros((2, fuzzy_traces.len()));
    for j in 0..fuzzy_traces.len() {
        fuzzy_table_b0[[0, j]] = fuzzy_state_traces[[0, j]] + fuzzy_state_traces[[1, j]];
        fuzzy_table_b0[[1, j]] = fuzzy_state_traces[[2, j]] + fuzzy_state_traces[[3, j]];

        // Second basis (b1)
        fuzzy_table_b1[[0, j]] = fuzzy_state_traces[[0, j]] + fuzzy_state_traces[[2, j]];
        fuzzy_table_b1[[1, j]] = fuzzy_state_traces[[1, j]] + fuzzy_state_traces[[3, j]];
    }

    let (li_b0_min, li_b0) = calculate_collision_mutual_info_fuzzy(&prob_tables_b0, fuzzy_traces, &fuzzy_table_b0, eps);
    let (li_b1_min, li_b1) = calculate_collision_mutual_info_fuzzy(&prob_tables_b1, fuzzy_traces, &fuzzy_table_b1, eps);

    match calc_type {
        "max" => (f64::max(li_b0_min, li_b1_min), f64::max(li_b0, li_b1)),
        "total" => (li_b0_min + li_b1_min, li_b0 + li_b1),
        "0" => (li_b0_min, li_b0),
        "1" => (li_b1_min, li_b1),
        _ => panic!("Unknown calculation type: {}", calc_type),
    }
}



// Objective function for collision mutual information
fn obj_function_collision_i(
    expec_vals: &[f64], 
    n_states: usize, 
    n_povms: usize, 
    calc_type: &str,
    eps: f64
) -> f64 {
    // Reshape to a 2D array for easier indexing
    let expec_2d = Array2::from_shape_vec((n_states, n_povms), expec_vals.to_vec())
        .expect("Expectation values should reshape correctly");

    // Create probability tables using more efficient ndarray operations
    let mut prob_tables_b0 = Array2::zeros((2, n_povms));
    let mut prob_tables_b1 = Array2::zeros((2, n_povms));

    // First basis (b0)
    for j in 0..n_povms {
        prob_tables_b0[[0, j]] = expec_2d[[0, j]] + expec_2d[[1, j]];
        prob_tables_b0[[1, j]] = expec_2d[[2, j]] + expec_2d[[3, j]];

        // Second basis (b1)
        prob_tables_b1[[0, j]] = expec_2d[[0, j]] + expec_2d[[2, j]];
        prob_tables_b1[[1, j]] = expec_2d[[1, j]] + expec_2d[[3, j]];
    }

    let li_b0 = calculate_collision_mutual_info_eps(&prob_tables_b0, eps);
    let li_b1 = calculate_collision_mutual_info_eps(&prob_tables_b1, eps);

    match calc_type {
        "max" => f64::max(li_b0, li_b1),
        "total" => li_b0 + li_b1,
        "0" => li_b0,
        "1" => li_b1,
        _ => panic!("Unknown calculation type: {}", calc_type),
    }
}


// Evaluate a POVM candidate
pub fn evaluate_povm_candidate_fuzzy(povms: &[Matrix2x2], states: &[Matrix2x2], calc_type: &str, eps: f64) -> (f64, f64) {
    let extremals = gen_eps_extremal(eps);
    let expec_vals_orig = calculate_expectation_values(povms, states);

    // Traces with the state matrices
    let traces_states_fuzzy = calculate_expectation_values(&extremals, states);
    let traces_fuzzy: Vec<f64> = extremals.iter().map(|x| x.trace_product(&Matrix2x2::identity()).re).collect::<Vec<_>>();
    obj_function_collision_fuzzy(&expec_vals_orig, states.len(), povms.len(), calc_type, &traces_fuzzy, &traces_states_fuzzy, eps)
}



// Evaluate a POVM candidate
pub fn evaluate_povm_candidate(povms: &[Matrix2x2], states: &[Matrix2x2], calc_type: &str, eps: f64) -> f64 {
    let expec_vals = calculate_expectation_values(povms, states);
    obj_function_collision_i(&expec_vals, states.len(), povms.len(), calc_type, eps)
}

// Complete a partial POVM set with its final element
pub fn complete_povm(povms: &[Matrix2x2]) -> Option<Vec<Matrix2x2>> {
    let identity = Matrix2x2::identity();
    let current_sum = Matrix2x2::sum(povms);
    let last_element = identity.sub(&current_sum);

    if last_element.is_positive_semidefinite(1e-6) {
        let mut complete_set = povms.to_vec();
        complete_set.push(last_element);
        Some(complete_set)
    } else {
        None
    }
}


