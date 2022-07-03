extern crate tsetlin_machine;
use tsetlin_machine::TsetlinMachine;

extern crate rand;
use rand::thread_rng;

extern crate bitvec;
use bitvec::prelude::*;

extern crate mnist;
use mnist::*;
extern crate ndarray;
use ndarray::prelude::*;

const TRAINING_AMOUNT: usize = 500;

const TRAINING_SET_LENGTH: usize = 50_000;
const VALIDATION_SET_LENGTH: usize = 10_000;
const TEST_SET_LENGTH: usize = 10_000;

fn avg(nums: &[f32]) -> f32 {
    nums.iter().sum::<f32>() / (nums.len() as f32)
}

#[test]
fn mnist_accuracy() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(TRAINING_SET_LENGTH as u32)
        .validation_set_length(VALIDATION_SET_LENGTH as u32)
        .test_set_length(TEST_SET_LENGTH as u32)
        .download_and_extract()
        .finalize();
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((TRAINING_SET_LENGTH, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Vec<BitVec> = Array2::from_shape_vec((TRAINING_SET_LENGTH, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .into_iter()
        .map(|x| (x as usize).view_bits::<Lsb0>().to_bitvec())
        .collect();

    let _test_data = Array3::from_shape_vec((TEST_SET_LENGTH, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let _test_labels: Vec<BitVec> = Array2::from_shape_vec((TEST_SET_LENGTH, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .into_iter()
        .map(|x| (x as usize).view_bits::<Lsb0>().to_bitvec())
        .collect();

    let mut tm = TsetlinMachine::new(100, 64, 100);

    let mut rng = thread_rng();
    let mut errors: Vec<f32> = Vec::with_capacity(TRAINING_AMOUNT);

    for e in 0..TRAINING_AMOUNT {
        let input_vector = BitVec::from_vec(
            train_data
                .slice(s![e % TRAINING_SET_LENGTH as usize, .., ..])
                .as_slice()
                .unwrap()
                .iter()
                .map(|f| (f > &0.0) as usize)
                .collect::<Vec<_>>(),
        );
        let output_vector = tm.activate(&input_vector);
        let correct = output_vector == &train_labels[e % TRAINING_SET_LENGTH as usize];
        errors.push(!correct as u8 as f32);
        tm.learn(
            &train_labels[e % TRAINING_SET_LENGTH as usize],
            4.0,
            4.0,
            &mut rng,
        );
        if avg(&errors) < 0.01 {
            break;
        }
    }

    assert!(avg(&errors) < 0.01, "avg error was: {}", avg(&errors));
}
