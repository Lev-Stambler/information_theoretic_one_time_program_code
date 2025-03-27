/// Author: Lev Stambler
/// Licence: MIT
/// 
/// Notes about the code:
///
/// 1) We only optimize over the real values in POVM elements as our states are real and thus, for
/// POVM M, Tr[M rho] = Tr[M' rho] where the real values of M and M' are the same but M' has some
/// imaginary values in its off-diagonal elements.
///
/// 2) This code is primarily for optimizing POVMs for dimension 2. The code can be extended to
///    higher dimensions. If you want to do that though, the code needs to be extensively
///    refactored
///
///    Big thanks to Anthropic's Claude for helping me out
use rayon::prelude::*;
use std::cmp::Ordering;
use std::f64;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use utils::*;

mod utils;

fn generate_first_level_candidates(
    n: usize,
    eps: f64,
    states: &[Matrix2x2],
    calc_type: &str,
) -> Vec<PovmCandidate> {
    let mut candidates = vec![PovmCandidate {
        elements: Vec::new(),
        max_objective_value: 0.0,
        min_objective_value: 0.0,
    }];
    // Set number of grid points per dimension for each epsilon level
    let steps = (1.0 / eps).ceil() as usize;
    let best_solution = Arc::new(Mutex::new((Vec::new(), f64::NEG_INFINITY)));
    let best_solution_min = Arc::new(Mutex::new((Vec::new(), f64::NEG_INFINITY)));
    for depth in 0..(n - 1) {
        println!("\nDepth for first level {}/{}: Processing {} candidates with eps = {}", 
            depth+1, n-1, candidates.len(), eps);
        // Share best_solution across threads
        let best_solution_ref = Arc::clone(&best_solution);
        let best_solution_min_ref = Arc::clone(&best_solution_min);

        // Process candidates in parallel with proper synchronization
        let all_new_candidates: Vec<PovmCandidate> = candidates.par_iter()
            .flat_map(|candidate| {
                let valid_elements =  initial_grid(&candidate.elements, eps, steps);
                // For each valid element, create a new candidate
                valid_elements.into_iter().filter_map(|element| {
                    let mut new_elements = candidate.elements.clone();
                    new_elements.push(element);
                    // For the last depth, check if we can complete the POVM
                    if depth == n-2 {
                        complete_povm(&new_elements).map(|complete_povm| {
                            let (obj_val_min, obj_val) = evaluate_povm_candidate_fuzzy(&complete_povm, states, calc_type, eps);//, 0.001);

                            // Thread-safe update of best solution
                            let mut best_sol = best_solution_ref.lock().unwrap();
                            let mut best_sol_min = best_solution_min_ref.lock().unwrap();
                            if obj_val > best_sol.1 {
                                *best_sol = (complete_povm.clone(), obj_val);
                                println!("New best: {}", obj_val);
                            }
                            if obj_val_min > best_sol_min.1 {
                                *best_sol_min = (complete_povm.clone(), obj_val_min);
                                println!("New Best Min: {}", obj_val_min);
                            }

                            PovmCandidate {
                                elements: new_elements,
                                max_objective_value: obj_val,
                                min_objective_value: obj_val_min,
                            }
                        })
                    } else {
                        // For earlier depths, just add to candidates
                        Some(PovmCandidate {
                            elements: new_elements,
                            max_objective_value: 0.0, // Will be evaluated later
                            min_objective_value: 0.0,
                        })
                    }
                }).collect::<Vec<_>>()
            })
        .collect();

        candidates = all_new_candidates;

        if candidates.is_empty() {
            println!("No valid candidates found, stopping search");
            break;
        }
    }
    truncate_povm_candidates(candidates)
    //candidates.sort_by(|a, b| {
    //    b.max_objective_value.partial_cmp(&a.max_objective_value)
    //    .unwrap_or(Ordering::Equal)});

    //// TODO: make top_k variable!
    //candidates.truncate(top_k);
    //candidates
}

// Simplified progressive POVM optimization with proper synchronization
fn optimize_povm_progressive(
    n: usize,
    epsilons: &[f64],
    states: &[Matrix2x2],
    calc_type: &str,
) -> (Vec<Matrix2x2>, f64) {
    println!("Starting progressive POVM optimization");

    // Use extremality theorem for d=2: optimal POVMs have at most d²=4 elements
    let n = std::cmp::min(n, 4);
    println!("Using N={} POVM elements", n);

    // Initialize with empty candidates
    let mut candidates = generate_first_level_candidates(n, epsilons[0], states, calc_type);

    let final_result = Arc::new(Mutex::new((Vec::new(), f64::NEG_INFINITY)));
    // Process each epsilon level
    for (_level, &eps) in (epsilons[1..]).iter().enumerate() {
        let level = _level + 1;
        println!("\n--- Refinement level {}/{}: eps = {}", 
            level+1, epsilons.len(), eps);
        // Use Arc<Mutex<>> for thread-safe access to best solution
        //best_solution.lock().unwrap().1 = f64::NEG_INFINITY;  // Reset best solution
        //best_solution.lock().unwrap().0.clear();  // Clear best solution
        // Use Arc<Mutex<>> for thread-safe access to best solution
        let best_solution = Arc::new(Mutex::new((Vec::new(), f64::NEG_INFINITY)));
        let best_solution_min = Arc::new(Mutex::new((Vec::new(), f64::NEG_INFINITY)));

        // For each candidate, generate new candidates by discretizing the grid of the existing
        // elements
        println!("\nLevel {}/{}: Processing {} candidates", 
            level+1, epsilons.len(), candidates.len());
        // Share best_solution across threads
        let best_solution_ref = Arc::clone(&best_solution);
        let best_solution_min_ref = Arc::clone(&best_solution_min);

        // Process candidates in parallel with proper synchronization
        let all_new_candidates: Vec<PovmCandidate> = candidates.par_iter().enumerate()
            .flat_map(|(i, candidate)| {
                if i % 10_000 == 0 {
                    println!("Processing candidate {}/{}", i, candidates.len());
                }
                let valid_elements = discretize_povms(&candidate.elements, eps, epsilons[level - 1]);
                
                let r = valid_elements.par_iter().filter_map(|elem_set| {
                    let l = elem_set.len();
                    //assert!(l == 3, "Something unexpected! Got {} size elem set", l);
                    if let Some(final_povms) = complete_povm(elem_set) {
                        let (obj_val_min, obj_val) = evaluate_povm_candidate_fuzzy(&final_povms, states, calc_type, eps);

                        // Thread-safe update of best solution
                        let mut best_sol = best_solution_ref.lock().unwrap();
                        let mut best_sol_min = best_solution_min_ref.lock().unwrap();
                        if obj_val > best_sol.1 {
                            *best_sol = (final_povms.clone(), obj_val);
                            println!("New best: {}", obj_val);
                        }
                        if obj_val_min > best_sol_min.1 {
                            *best_sol_min = (final_povms.clone(), obj_val_min);
                            println!("New Best Min: {}", obj_val_min);
                        }
                        Some(PovmCandidate {
                            elements: elem_set.to_vec(),
                            max_objective_value: obj_val,
                            min_objective_value: obj_val_min,
                        })
                    } else {
                        None
                    }
                }).collect::<Vec<_>>();
                r
            }).collect();


        // Update candidates for next iteration
        candidates = all_new_candidates;
        if candidates.is_empty() {
            println!("No valid candidates found, stopping search");
            break;
        }

        // If we reached the last epsilon level, we're done
        if level == epsilons.len() - 1 {
            break;
        }

        // Otherwise, prepare candidates for the next epsilon level
        // Only keep complete POVM sets with highest objective values
        println!("Number of elements before truncation: {}", candidates.len());
        candidates = truncate_povm_candidates(candidates);
        println!("Number of elements after truncation: {}", candidates.len());
    }

    // Extract final result from the thread-safe container
    //let final_result = best_solution.lock().unwrap().clone();
    let r = final_result.lock().unwrap().clone();
    r
}

fn main() {
    // Default parameters
    let n = 4;
    let epsilons = vec![0.1, 0.05, 0.025, 0.0125, 0.0125 / 2.0];
    let calc_types = vec!["0"];//, "max", "total", "1"]; "0/ 1" represents the case for conditional
                               //bit accessible mutual information
    for calc_type in calc_types {

        println!("Running POVM optimization with parameters:");
        println!("  - Calculation type: {}", calc_type);
        println!("  - Number of POVM elements: {}", n);
        println!("  - Epsilon sequence: {:?}", epsilons);

        // Create BB84 states
        let a00 = state_from_angle(std::f64::consts::PI / 8.0);
        let a01 = state_from_angle(7.0 * std::f64::consts::PI / 8.0);
        let a10 = state_from_angle(3.0 * std::f64::consts::PI / 8.0);
        let a11 = state_from_angle(5.0 * std::f64::consts::PI / 8.0);

        // Create states tensor based on calculation type
        let states = match calc_type {
            "max" | "total" => {
                vec![
                    a00.mul(0.25),
                    a01.mul(0.25),
                    a10.mul(0.25),
                    a11.mul(0.25),
                ]
            },
            "0" => {
                vec![
                    a00.mul(0.5),
                    a01.mul(0.0),
                    a10.mul(0.5),
                    a11.mul(0.0),
                ]
            },
            "1" => {
                vec![
                    a00.mul(0.5),
                    a01.mul(0.5),
                    a10.mul(0.0),
                    a11.mul(0.0),
                ]
            },
            _ => panic!("Unknown calculation type: {}", calc_type),
        };

        let start_time = Instant::now();
        let (optimal_povm, val) = optimize_povm_progressive(n, &epsilons, &states, calc_type);
        let duration = start_time.elapsed();

        println!("Optimization completed in {:.2?}", duration);
        println!("Optimal objective value: {}", val);

        // Print optimal POVM elements
        println!("Optimal POVM elements:");
        for (i, povm) in optimal_povm.iter().enumerate() {
            println!("POVM {}: [{:?}]", 
                i,
                povm)
        }

    }
}
