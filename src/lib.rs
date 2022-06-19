extern crate bitvec;
extern crate rand;

use bitvec::prelude::*;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::cmp::min;

//TODO: consider splitting negation/statements into two separate BitVecs, which clears up the usage at the cost of slightly more space usage
#[derive(Debug, Default)]
pub struct TsetlinMachine {
    //TODO: think about what input and output states actually do, and if they are actually needed
    // output_states seems like it is only used as an output of activate
    input_states: BitVec,
    output_states: BitVec,
    outputs: Vec<Output>,
}

impl TsetlinMachine {
    /// Creates a new `TsetlinMachine` with the given parameters
    #[must_use]
    pub fn new(
        number_of_inputs: usize,
        number_of_outputs: usize,
        clauses_per_output: usize,
    ) -> TsetlinMachine {
        //TODO: consider changing BitVec to BitArray[N], and placing N in the type sig, so this can get some more type safety for input and output
        // This will make changing the input/output much harder, so should probably have a different version for that
        let mut outputs = vec![Output::default(); number_of_outputs];
        for output in &mut outputs {
            output.clauses.resize(clauses_per_output, Clause::default());
            for clause in &mut output.clauses {
                clause.automata_states.resize(number_of_inputs * 2_usize, 0);
            }
        }
        TsetlinMachine {
            input_states: BitVec::repeat(false, number_of_inputs),
            output_states: BitVec::repeat(false, number_of_outputs),
            outputs,
        }
    }

    fn inclusion_update(&mut self, oi: usize, ci: usize, ai: usize) {
        let inclusion = self.outputs[oi].clauses[ci].automata_states[ai] > 0;
        let index = self.outputs[oi].clauses[ci]
            .inclusions
            .iter()
            .position(|&s| s == ai);
        if inclusion {
            if index.is_none() {
                self.outputs[oi].clauses[ci].inclusions.push(ai);
            }
        } else if let Some(index) = index {
            self.outputs[oi].clauses[ci].inclusions.remove(index);
        }
    }

    ///This goes through the clause states and ??
    fn modify_phase_one(
        &mut self,
        oi: usize,
        ci: usize,
        s_inverse: f64,
        s_inverse_conjugate: f64,
        rng: &mut ThreadRng,
    ) {
        let clause_state = self.outputs[oi].clauses[ci].state;

        for ai in 0..self.outputs[oi].clauses[ci].automata_states.len() {
            let input = if ai >= self.input_states.len() {
                !self.input_states[ai - self.input_states.len()]
            } else {
                self.input_states[ai]
            };

            let remembered = self.outputs[oi].clauses[ci].automata_states[ai] > 0;
            let s: f64 = rng.gen();
            if clause_state {
                if input {
                    if s < s_inverse_conjugate {
                        self.outputs[oi].clauses[ci].automata_states[ai] += 1;
                        self.inclusion_update(oi, ci, ai);
                    }
                } else if !remembered && s < s_inverse {
                    //TODO: combine this section with the bottom one, until there are only 2 sections
                    self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                    self.inclusion_update(oi, ci, ai);
                }
            } else if s < s_inverse {
                self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                self.inclusion_update(oi, ci, ai);
            }
        }
    }

    /// This does goes through the clause states and ?(bubbles the forgotten things up) and ??
    fn modify_phase_two(&mut self, oi: usize, ci: usize) {
        let clause_state = self.outputs[oi].clauses[ci].state;
        for ai in 0..self.outputs[oi].clauses[ci].automata_states.len() {
            //If the index is in the second half of the section, we ?? because it's actually the negation term
            let input = if ai >= self.input_states.len() {
                !self.input_states[ai - self.input_states.len()]
            } else {
                self.input_states[ai]
            };
            let remembered = self.outputs[oi].clauses[ci].automata_states[ai] > 0;
            //TODO: consider inlining the definitions to avoid work
            if clause_state && !input && !remembered {
                self.outputs[oi].clauses[ci].automata_states[ai] += 1;
                self.inclusion_update(oi, ci, ai);
            }
        }
    }

    /// Learns some bit of data for a targeted output state
    /// # Panics
    /// Will panic if target output states is a different length than outputs length
    pub fn learn(&mut self, target_output_states: &BitVec, s: f64, t: f64, rng: &mut ThreadRng) {
        //TODO: what does the s and t here even mean?? consider getting better names
        let s_inv: f64 = 1.0 / s;
        let s_inv_conj: f64 = 1.0 - s_inv;
        assert_eq!(self.outputs.len(), target_output_states.len());

        for oi in 0..self.outputs.len() {
            let clamped_sum = t.min((-t).max(self.outputs[oi].sum.into()));
            let rescale = 1.0 / (2.0 * t);
            let probability_feedback_alpha: f64 = (t - clamped_sum) * rescale;
            let probability_feedback_beta: f64 = (t + clamped_sum) * rescale;

            for ci in 0..self.outputs[oi].clauses.len() {
                let s: f64 = rng.gen();
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

    /// Gives the predicted output for the given input
    pub fn activate(&mut self, input_states: &BitVec) -> &BitVec {
        //TODO: does this really need to lend out the output states? does the structure of this need to be redone?
        self.input_states = input_states.clone();
        for (outputs_index, mut outputs_element) in self.outputs.iter_mut().enumerate() {
            let mut sum = 0;
            for (clauses_index, clauses_element) in outputs_element.clauses.iter_mut().enumerate() {
                let mut state = true;
                for &ai in &clauses_element.inclusions {
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
                    //TODO: if this is flipped, everything seems to work
                    sum += if clauses_index % 2 == 0 {
                        i32::from(state)
                    } else {
                        -i32::from(state)
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
//TODO: implement a new for output which simplifies the creation in the original new function
