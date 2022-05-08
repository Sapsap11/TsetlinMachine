extern crate tsetlin_machine;
use tsetlin_machine::TsetlinMachine;

extern crate rand;
use rand::thread_rng;

extern crate bitvec;
use bitvec::prelude::*;

#[test]
fn xor_convergence() {
    let inputs: Vec<BitVec> = vec![bitvec![0, 0], bitvec![0, 1], bitvec![1, 0], bitvec![1, 1]];
    let outputs: Vec<BitVec> = vec![bitvec![0, 1], bitvec![1, 0], bitvec![1, 0], bitvec![0, 1]];
    let mut tm = TsetlinMachine::new(2, 2, 10);

    let mut rng = thread_rng();
    let mut average_error: f32 = 1.0;

    for e in 0..5000 {
        let input_vector = &inputs[e % 4];
        {
            let output_vector = tm.activate(input_vector);
            let correct = (input_vector[0] == input_vector[1])
                && (!output_vector[0] && output_vector[1])
                || (output_vector[0] && !output_vector[1]);

            average_error = 0.99 * average_error + 0.01 * (if !correct { 1.0 } else { 0.0 });
        }
        tm.learn(&outputs[e % 4], 4.0, 4.0, &mut rng);
        if average_error < 0.01 {
            break;
        }
    }

    assert!(average_error < 0.01);
}
