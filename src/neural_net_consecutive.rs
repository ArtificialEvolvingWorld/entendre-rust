use std::collections::HashMap;

use crate::neural_net::*;

#[derive(Debug)]
enum NodeValue {
    Accumulator(f32),
    Activated(f32),
}

#[derive(Debug)]
struct Node {
    value: NodeValue,
    node_type: NodeType,
    func: ActivationFunction,
}

impl Node {
    fn get_val(&mut self) -> f32 {
        match self.value {
            NodeValue::Activated(x) => x,
            NodeValue::Accumulator(x) => {
                let output = self.func.apply(x);
                self.value = NodeValue::Activated(output);
                output
            }
        }
    }

    fn add_to_val(&mut self, x: f32) {
        self.value = match self.value {
            NodeValue::Activated(_) => NodeValue::Accumulator(x),
            NodeValue::Accumulator(y) => NodeValue::Accumulator(x + y),
        };
    }
}

#[derive(Debug, Clone, Copy)]
struct Connection {
    origin: u32,
    dest: u32,
    weight: f32,
    connection_type: ConnectionType,
}

#[derive(Debug)]
pub struct ConsecutiveNeuralNet {
    nodes: Vec<Node>,
    connections: Vec<Connection>,
}

fn connection_order(
    connections: &[ConnectionTemplate],
) -> Result<Vec<usize>, Error> {
    let mut must_be_after = (0..connections.len())
        .map(|j| {
            let required_before = (0..connections.len())
                .filter(|i| {
                    let i = *i;
                    let conn_i = &connections[i];
                    let conn_j = &connections[j];

                    // Avoid nonsensical dependencies
                    let different_connection = i != j;

                    // The origin of a normal connection has no unused
                    // input connections.
                    let after_input_conn = (conn_i.dest == conn_j.origin)
                        && (conn_j.connection_type == ConnectionType::Normal);

                    // The destination has no unused recurrent output
                    // connections.
                    let before_output_conn = (conn_i.origin == conn_j.dest)
                        && (conn_i.connection_type
                            == ConnectionType::Recurrent);

                    different_connection
                        && (after_input_conn || before_output_conn)
                })
                .collect::<Vec<_>>();

            (j, required_before)
        })
        .collect::<HashMap<usize, Vec<usize>>>();

    let mut output = Vec::new();

    while must_be_after.len() > 0 {
        let next_connection = must_be_after
            .iter()
            .filter(|(_k, v)| {
                v.iter().all(|before| !must_be_after.contains_key(before))
            })
            .map(|(k, _v)| *k)
            .next()
            // If no connections can occur next, the network contains
            // a loop of normal connections or a loop of recurrent
            // connections..  A loop of normal connections is
            // ill-defined.  A loop of recurrent connections is
            // semantically valid, but isn't possible to represent
            // with this representation.
            .ok_or(Error::ConnectionLoop)?;

        output.push(next_connection);
        must_be_after.remove(&next_connection);
    }

    Ok(output)
}

impl ConsecutiveNeuralNet {
    pub fn new() -> ConsecutiveNeuralNet {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
        }
    }

    fn load_input_values(&mut self, inputs: &[f32]) {
        self.nodes
            .iter_mut()
            .filter(|n| n.node_type == NodeType::Input)
            .zip(inputs.iter())
            .for_each(|(n, x)| {
                n.value = NodeValue::Activated(*x);
            });
    }
}

impl NeuralNet for ConsecutiveNeuralNet {
    //fn build_from(&mut self, builder: NeuralNetBuilder) -> Result<(), Error> {
    fn build_from(builder: &NeuralNetBuilder) -> Result<Self, Error> {
        let nodes = builder
            .nodes
            .iter()
            .map(|t| Node {
                value: NodeValue::Accumulator(0.0),
                node_type: t.node_type,
                func: t.func,
            })
            .collect::<Vec<_>>();

        let connections = connection_order(&builder.connections)?
            .iter()
            .map(|i| {
                let template = builder
                    .connections
                    .get(*i)
                    .ok_or(Error::InvalidConnectionIndex)?;
                Ok(Connection {
                    origin: template.origin,
                    dest: template.dest,
                    weight: template.weight,
                    connection_type: template.connection_type,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { nodes, connections })
    }

    fn evaluate(&mut self, inputs: &[f32]) -> Vec<f32> {
        self.load_input_values(&inputs);

        {
            let connections = &mut self.connections;
            let nodes = &mut self.nodes;

            connections.iter().for_each(|conn| {
                let val = nodes[conn.origin as usize].get_val();
                nodes[conn.dest as usize].add_to_val(val * conn.weight);
            });
        }

        self.nodes
            .iter_mut()
            .filter(|n| n.node_type == NodeType::Output)
            .map(|n| n.get_val())
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_simple_net() -> Result<(), Error> {
        let mut net = NeuralNetBuilder::new()
            .set_default_activation(ActivationFunction::Identity)
            .add_nodes(NodeType::Input, 2)
            .add_nodes(NodeType::Output, 1)
            .add_normal_connection(0, 2, 1.0)
            .add_normal_connection(1, 2, -1.0)
            .build::<ConsecutiveNeuralNet>()?;

        let res = net.evaluate(&[0.5, 1.5]);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], -1.0);
        Ok(())
    }

    #[test]
    fn test_multilayered_net() -> Result<(), Error> {
        let func = ActivationFunction::Sigmoid;
        let mut net = NeuralNetBuilder::new()
            .set_default_activation(func)
            .add_nodes(NodeType::Input, 1)
            .add_nodes(NodeType::Hidden, 1)
            .add_nodes(NodeType::Output, 1)
            // First connection added first, correct order of
            // evaluation
            .add_normal_connection(0, 1, 1.0)
            .add_normal_connection(1, 2, 1.0)
            .build::<ConsecutiveNeuralNet>()?;

        let res = net.evaluate(&[0.0]);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], func.apply(func.apply(0.0)));

        let mut net = NeuralNetBuilder::new()
            .set_default_activation(func)
            .add_nodes(NodeType::Input, 1)
            .add_nodes(NodeType::Hidden, 1)
            .add_nodes(NodeType::Output, 1)
            // Second connection added first, needs to be sorted.
            .add_normal_connection(1, 2, 1.0)
            .add_normal_connection(0, 1, 1.0)
            .build::<ConsecutiveNeuralNet>()?;

        let res = net.evaluate(&[0.0]);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], func.apply(func.apply(0.0)));

        Ok(())
    }
}
