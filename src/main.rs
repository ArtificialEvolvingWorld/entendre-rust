use entendre::*;

fn main() -> Result<(), Error> {
    let mut net = NeuralNetBuilder::new()
        .set_default_activation(ActivationFunction::Sigmoid)
        .add_nodes(NodeType::Bias, 1)
        .add_nodes(NodeType::Input, 1)
        .add_nodes(NodeType::Output, 2)
        .add_normal_connection(0, 2, 1.0)
        .add_normal_connection(1, 2, 1.0)
        .add_normal_connection(0, 3, 1.0)
        .add_normal_connection(1, 3, 1.0)
        .build::<ConsecutiveNeuralNet>()?;

    let res = net.evaluate(&[1.0, 2.0, 3.0]);
    println!("Net output = {:?}", res);

    Ok(())
}
