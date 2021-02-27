#[derive(Debug, PartialEq, Clone, Copy)]
pub enum NodeType {
    Bias,
    Input,
    Output,
    Hidden,
}

impl NodeType {
    fn _is_sensor(&self) -> bool {
        use NodeType::*;
        match self {
            Bias | Input => true,
            Output | Hidden => false,
        }
    }
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
        use ActivationFunction::*;
        match self {
            Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Identity => x,
            Tanh => x.tanh(),
            Relu => x.max(0.0),
            Gaussian => (-x * x / 2.0).exp(),
            Sin => x.sin(),
            Cos => x.cos(),
            Abs => x.abs(),
            Square => x * x,
        }
    }
}

pub struct NodeTemplate {
    pub node_type: NodeType,
    pub func: ActivationFunction,
}

pub struct ConnectionTemplate {
    pub origin: u32,
    pub dest: u32,
    pub weight: f32,
    pub connection_type: ConnectionType,
}

pub struct NeuralNetBuilder {
    pub nodes: Vec<NodeTemplate>,
    pub connections: Vec<ConnectionTemplate>,
    default_func: ActivationFunction,
}

impl NeuralNetBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            default_func: ActivationFunction::Sigmoid,
        }
    }

    pub fn set_default_activation(
        &mut self,
        func: ActivationFunction,
    ) -> &mut Self {
        self.default_func = func;
        self
    }

    pub fn add_input(&mut self) -> &mut Self {
        self.nodes.push(NodeTemplate {
            node_type: NodeType::Input,
            func: ActivationFunction::Identity,
        });
        self
    }

    pub fn add_inputs(&mut self, n: u32) -> &mut Self {
        (0..n).for_each(|_| {
            self.add_input();
        });
        self
    }

    pub fn add_node(
        &mut self,
        node_type: NodeType,
        func: ActivationFunction,
    ) -> &mut Self {
        self.nodes.push(NodeTemplate { node_type, func });
        self
    }

    pub fn add_nodes(&mut self, node_type: NodeType, n: u32) -> &mut Self {
        (0..n).for_each(|_| {
            self.add_node(node_type, self.default_func);
        });
        self
    }

    pub fn add_connection(
        &mut self,
        origin: u32,
        dest: u32,
        weight: f32,
        connection_type: ConnectionType,
    ) -> &mut Self {
        self.connections.push(ConnectionTemplate {
            origin,
            dest,
            weight,
            connection_type,
        });
        self
    }

    pub fn add_normal_connection(
        &mut self,
        origin: u32,
        dest: u32,
        weight: f32,
    ) -> &mut Self {
        self.add_connection(origin, dest, weight, ConnectionType::Normal);
        self
    }

    pub fn add_recurrent_connection(
        &mut self,
        origin: u32,
        dest: u32,
        weight: f32,
    ) -> &mut Self {
        self.add_connection(origin, dest, weight, ConnectionType::Recurrent);
        self
    }

    pub fn build<N>(&mut self) -> Result<N, Error>
    where
        N: NeuralNet,
    {
        N::build_from(self)
    }
}

#[derive(Debug)]
pub enum Error {
    ConnectionLoop,
    InvalidConnectionIndex,
}

pub trait NeuralNet: Sized {
    fn build_from(builder: &NeuralNetBuilder) -> Result<Self, Error>;
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
