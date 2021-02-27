#[derive(Debug, PartialEq, Clone, Copy)]
pub enum NodeType {
    Bias,
    Input,
    Output,
    Hidden,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ConnectionType {
    Normal,
    Recurrent,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Identity,
    Tanh,
    Relu,
    Gaussian,
    Sin,
    Cos,
    Abs,
    Square,
}

impl ActivationFunction {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Identity => x,
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Relu => x.max(0.0),
            ActivationFunction::Gaussian => (-x * x / 2.0).exp(),
            ActivationFunction::Sin => x.sin(),
            ActivationFunction::Cos => x.cos(),
            ActivationFunction::Abs => x.abs(),
            ActivationFunction::Square => x * x,
        }
    }
}

pub trait NeuralNet {
    fn add_node(&mut self, node_type: NodeType, func: ActivationFunction);
    fn add_connection(&mut self, origin: u32, dest: u32, weight: f32);

    fn evaluate(&mut self, inputs: &[f32]) -> Vec<f32>;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_activation_function() {
        let func = ActivationFunction::Sigmoid;
        assert!((func.apply(-1.0) - 0.26894).abs() < 1e-4);
        assert!((func.apply(1.0) - 0.73105).abs() < 1e-4);
    }
}
