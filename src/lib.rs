extern crate bitvec;
extern crate rand;

use bitvec::prelude::*;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::cmp::min;

#[derive(Debug, Default)]
pub struct TsetlinMachine {
    input_states: BitVec,
    output_states: BitVec,
    outputs: Vec<Output>,
}

impl TsetlinMachine {
    pub fn new(
        number_of_inputs: usize,
        number_of_outputs: usize,
        clauses_per_output: usize,
    ) -> TsetlinMachine {
        //TODO: consider changing BitVec to BitArray[N], and placing N in the type sig, so this can get some more type safety for input and output
        let input_states = BitVec::repeat(false, number_of_inputs);
        let output_states = BitVec::repeat(false, number_of_outputs);
        let mut outputs = vec![Output::default(); number_of_outputs];
        for output in outputs.iter_mut() {
            //TODO: avoid indexing via iter_mut
            output.clauses.resize(clauses_per_output, Clause::default());
            for clause in output.clauses.iter_mut() {
                clause.automata_states.resize(number_of_inputs * 2_usize, 0);
            }
        }
        TsetlinMachine {
            input_states,
            output_states,
            outputs,
        }
    }

    fn inclusion_update(&mut self, oi: usize, ci: usize, ai: usize) {
        let inclusion = self.outputs[oi].clauses[ci].automata_states[ai] > 0;
        let it = self.outputs[oi].clauses[ci]
            .inclusions
            .iter()
            .position(|&s| s == ai);
        if inclusion {
            if it.is_none() {
                self.outputs[oi].clauses[ci].inclusions.push(ai);
            }
        } else if let Some(it) = it {
            self.outputs[oi].clauses[ci].inclusions.remove(it);
        }
    }

    ///This goes through the clause states and ??
    fn modify_phase_one(
        &mut self,
        oi: usize,
        ci: usize,
        s_inverse: f32,
        s_inverse_conjugate: f32,
        rng: &mut ThreadRng,
    ) {
        let clause_state = self.outputs[oi].clauses[ci].state;
        for ai in 0..self.outputs[oi].clauses[ci].automata_states.len() {
            let input = if ai >= self.input_states.len() {
                !self.input_states[ai - self.input_states.len()]
            } else {
                self.input_states[ai]
            };
            let inclusion = self.outputs[oi].clauses[ci].automata_states[ai] > 0;
            let s: f32 = rng.gen();
            if clause_state {
                if input {
                    if s < s_inverse_conjugate {
                        self.outputs[oi].clauses[ci].automata_states[ai] += 1;
                        self.inclusion_update(oi, ci, ai);
                    }
                } else if !inclusion && s < s_inverse {
                    self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                    self.inclusion_update(oi, ci, ai);
                }
            } else if s < s_inverse {
                self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                self.inclusion_update(oi, ci, ai);
            }
        }
    }

    /// This does goes through the clause states and ?(updates the forgotten things up) and ??
    fn modify_phase_two(&mut self, oi: usize, ci: usize) {
        let clause_state = self.outputs[oi].clauses[ci].state;
        for ai in 0..self.outputs[oi].clauses[ci].automata_states.len() {
            //If the index is in the second half of the section, we ?? because it's actually the negation term
            let input = if ai >= self.input_states.len() {
                !self.input_states[ai - self.input_states.len()]
            } else {
                self.input_states[ai]
            };
            let inclusion = self.outputs[oi].clauses[ci].automata_states[ai] > 0;
            if clause_state && !input && !inclusion {
                self.outputs[oi].clauses[ci].automata_states[ai] += 1;
                self.inclusion_update(oi, ci, ai);
            }
        }
    }

    pub fn learn(&mut self, target_output_states: &BitVec, s: f32, t: f32, rng: &mut ThreadRng) {
        //TODO: what does the s and t here even mean??
        let s_inv = 1.0 / s;
        let s_inv_conj = 1.0 - s_inv;
        assert_eq!(self.outputs.len(), target_output_states.len());
        for oi in 0..self.outputs.len() {
            let clamped_sum = t.min((-t).max(self.outputs[oi].sum as f32));
            let rescale = 1.0 / (2.0 * t);
            let probability_feedback_alpha = (t - clamped_sum) * rescale;
            let probability_feedback_beta = (t + clamped_sum) * rescale;

            for ci in 0..self.outputs[oi].clauses.len() {
                let s: f32 = rng.gen();
                //TODO: check if this is the best/most predictable order
                if target_output_states[oi] {
                    //TODO: flipping the == to != also fixes the learning
                    if s < probability_feedback_alpha {
                        if ci % 2 == 0 {
                            self.modify_phase_one(oi, ci, s_inv, s_inv_conj, rng);
                        } else {
                            self.modify_phase_two(oi, ci);
                        }
                    }
                } else if s < probability_feedback_beta {
                    if ci % 2 == 0 {
                        self.modify_phase_two(oi, ci);
                    } else {
                        self.modify_phase_one(oi, ci, s_inv, s_inv_conj, rng);
                    }
                }
            }
        }
    }

    pub fn activate(&mut self, input_states: &BitVec) -> &BitVec {
        self.input_states = input_states.clone();
        for (outputs_index, mut outputs_element) in self.outputs.iter_mut().enumerate() {
            let mut sum = 0;
            for (clauses_index, clauses_element) in outputs_element.clauses.iter_mut().enumerate() {
                let mut state = true;
                for cit in clauses_element.inclusions.iter() {
                    let ai = *cit;
                    if ai >= self.input_states.len() {
                        state = state
                            && !self.input_states
                                [min(self.input_states.len() - 1, ai - self.input_states.len())];
                    } else {
                        state = state && self.input_states[ai];
                    }
                }
                clauses_element.state = state;
                {
                    let _state = if state { 1 } else { 0 };
                    //TODO: if this is flipped, everything seems to work
                    sum += if clauses_index % 2 == 0 {
                        _state
                    } else {
                        -_state
                    };
                }
            }
            outputs_element.sum = sum;
            self.output_states.set(outputs_index, sum > 0);
        }
        &self.output_states
    }
}

#[derive(Debug, Default, Clone)]
struct Clause {
    //TODO: is it sane to make this a specific size as well?
    automata_states: Vec<i32>,
    //TODO: consider adding the size(s?) to the type signature for this?
    inclusions: Vec<usize>,
    state: bool,
}

#[derive(Debug, Default, Clone)]
struct Output {
    //TODO: consider adding the size(s?) to the type signature for this?
    clauses: Vec<Clause>,
    sum: i32,
}
