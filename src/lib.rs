extern crate bitvec;
extern crate rand;

use bitvec::prelude::*;
use rand::rngs::ThreadRng;
use rand::Rng;

#[derive(Debug, Default)]
pub struct TsetlinMachine {
    //TODO: think about what input and output states actually do, and if they are actually needed
    // output_states seems like it is only used as an output of activate
    // however, it stops learning when it is removed
    input_states: BitVec,
    output_states: BitVec,
    // TODO: split up this into positive and negative assertions
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
        let outputs = vec![Output::new(clauses_per_output, number_of_outputs); number_of_outputs];
        /*
        for output in &mut outputs {
            output.clauses.resize(clauses_per_output, Clause::default());
            for clause in &mut output.clauses {
                clause.automata_states.resize(number_of_inputs * 2_usize, 0);
            }
        }
        */
        TsetlinMachine {
            input_states: BitVec::repeat(false, number_of_inputs),
            output_states: BitVec::repeat(false, number_of_outputs),
            outputs,
        }
    }

    //TODO: somehow describe what this does?
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
        learning_rate: f64,
        forgetting_rate: f64,
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
            /*
            if clause_state {
                if input {
                    //TODO: is the learing/forgetting rate correct?
                    if s < forgetting_rate {
                        self.outputs[oi].clauses[ci].automata_states[ai] += 1;
                        self.inclusion_update(oi, ci, ai);
                    }
                } else if !remembered && s < learning_rate {
                    //TODO: combine this section with the bottom one, until there are only 2 sections
                    self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                    self.inclusion_update(oi, ci, ai);
                }
            } else if s < learning_rate {
                self.outputs[oi].clauses[ci].automata_states[ai] -= 1;
                self.inclusion_update(oi, ci, ai);
            }
            */
            if clause_state && input && s < forgetting_rate {
                self.outputs[oi].clauses[ci].automata_states[ai] += 1;
                self.inclusion_update(oi, ci, ai);
            } else if s < learning_rate || (clause_state && !remembered) {
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

    /// Updates the model to try and learn some features for a given target
    /// # Panics
    /// Will panic if target output states is a different length than outputs length
    pub fn learn(
        &mut self,
        target_output_states: &BitVec,
        learning_volatility: f64,
        t: f64,
        rng: &mut ThreadRng,
    ) {
        //TODO: what does the t here even mean?? consider getting better names
        fn clamp(min: f64, value: f64, max: f64) -> f64 {
            min.max(max.min(value))
        }
        let learn_rate: f64 = 1.0 / learning_volatility;
        let forget_rate: f64 = 1.0 - learn_rate;
        assert_eq!(self.outputs.len(), target_output_states.len());

        for oi in 0..self.outputs.len() {
            let clamped_sum = clamp(-t, self.outputs[oi].sum.into(), t);
            let rescale = t * 0.5;
            let alpha_chance: f64 = (t - clamped_sum) * rescale;
            let beta_chance: f64 = (t + clamped_sum) * rescale;

            for ci in 0..self.outputs[oi].clauses.len() {
                let s: f64 = rng.gen();
                //TODO: check if this is the best/most predictable order
                if target_output_states[oi] {
                    if s < alpha_chance {
                        if ci % 2 == 0 {
                            self.modify_phase_one(oi, ci, learn_rate, forget_rate, rng);
                        } else {
                            self.modify_phase_two(oi, ci);
                        }
                    }
                } else if s < beta_chance {
                    if ci % 2 == 0 {
                        self.modify_phase_two(oi, ci);
                    } else {
                        self.modify_phase_one(oi, ci, learn_rate, forget_rate, rng);
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
                        state = state && !self.input_states[ai - self.input_states.len()];
                    } else {
                        state = state && self.input_states[ai];
                    }
                }
                clauses_element.state = state;
                {
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

///A Clause in a `TsetlinMachine`, this contains the actual state of the Clause
#[derive(Debug, Default, Clone)]
struct Clause {
    //TODO: is it sane to make this a specific size as well?
    automata_states: Vec<i32>,
    //TODO: consider adding the size(s?) to the type signature for this?
    inclusions: Vec<usize>,
    state: bool,
}

impl Clause {
    fn new(number_of_inputs: usize) -> Self {
        //TODO: replace this with a simple construction
        let mut clause = Clause::default();
        clause.automata_states.resize(number_of_inputs * 2, 0);
        clause
    }
}

#[derive(Debug, Default, Clone)]
struct Output {
    //TODO: consider adding the size(s?) to the type signature for this?
    clauses: Vec<Clause>,
    sum: i32,
}

impl Output {
    fn new(clauses_per_output: usize, number_of_inputs: usize) -> Self {
        //TODO: replace this with a simple construction
        let mut output = Output::default();
        output
            .clauses
            .resize(clauses_per_output, Clause::new(number_of_inputs));
        output
    }
}
//TODO: implement a new for output which simplifies the creation in the original new function
