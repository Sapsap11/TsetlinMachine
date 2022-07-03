extern crate tsetlin_machine;
use tsetlin_machine::TsetlinMachine;

extern crate rand;
use rand::thread_rng;

extern crate bitvec;
use bitvec::prelude::*;

fn avg(nums: &[f32]) -> f32 {
    nums.iter().sum::<f32>() / (nums.len() as f32)
}

#[test]
fn xor_convergence() {
    let inputs: Vec<BitVec> = vec![bitvec![0, 0], bitvec![0, 1], bitvec![1, 0], bitvec![1, 1]];
    let outputs: Vec<BitVec> = vec![bitvec![0, 1], bitvec![1, 0], bitvec![1, 0], bitvec![0, 1]];
    let mut tm = TsetlinMachine::new(2, 2, 10);

    let mut rng = thread_rng();
    let mut errors: Vec<f32> = Vec::with_capacity(5000);

    for e in 0..5000 {
        let input_vector = &inputs[e % 4];
        let output_vector = &outputs[e % 4];
        {
            let predicted_vector = tm.activate(input_vector);
            let correct = output_vector == predicted_vector;

            errors.push(!correct as u8 as f32);
        }
        tm.learn(output_vector, 4.0, 4.0, &mut rng);
        if avg(&errors) < 0.01 {
            break;
        }
    }

    assert!(avg(&errors) < 0.01, "avg error was {}", avg(&errors));
}
