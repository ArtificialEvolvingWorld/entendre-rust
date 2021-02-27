use itertools::Itertools;

use entendre::*;

fn setup_neural_net(net: &mut impl NeuralNet) {
    net.add_node(NodeType::Input, ActivationFunction::Identity);
    net.add_node(NodeType::Input, ActivationFunction::Identity);
    net.add_node(NodeType::Input, ActivationFunction::Identity);
    net.add_node(NodeType::Output, ActivationFunction::Sigmoid);
    net.add_node(NodeType::Output, ActivationFunction::Sigmoid);

    (0..3)
        .cartesian_product(0..2)
        .for_each(|(i, j)| net.add_connection(i, 3 + j, (i + j) as f32));
}

fn main() {
    let mut net = ConsecutiveNeuralNet::new();
    setup_neural_net(&mut net);
    let _res = net.evaluate(&[1.0, 2.0, 3.0]);
}
