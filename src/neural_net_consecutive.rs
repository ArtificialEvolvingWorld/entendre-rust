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
    sorted: bool,
}

impl ConsecutiveNeuralNet {
    pub fn new() -> ConsecutiveNeuralNet {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            sorted: false,
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

    fn connection_order(&self) -> Vec<usize> {
        let mut must_be_after = (0..self.connections.len())
            .map(|j| {
                let required_before = (0..self.connections.len())
                    .filter(|i| {
                        let i = *i;
                        let ref conn_i = self.connections[i];
                        let ref conn_j = self.connections[j];

                        // Avoid nonsensical dependencies
                        let different_connection = i != j;

                        // The origin of a normal connection has no unused
                        // input connections.
                        let after_input_conn = (conn_i.dest == conn_j.origin)
                            && (conn_j.connection_type
                                == ConnectionType::Normal);

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
                // If this panics, we have a loop, which shouldn't be
                // possible.
                .unwrap();

            output.push(next_connection);
            must_be_after.remove(&next_connection);
        }

        output
    }

    fn sort_connections(&mut self) {
        if self.sorted {
            return;
        }

        self.connections = self
            .connection_order()
            .iter()
            .map(|i| self.connections[*i])
            .collect();

        self.sorted = true;
    }
}

impl NeuralNet for ConsecutiveNeuralNet {
    fn add_node(&mut self, node_type: NodeType, func: ActivationFunction) {
        self.nodes.push(Node {
            value: NodeValue::Accumulator(0.0),
            node_type,
            func,
        });
    }

    fn add_connection(&mut self, origin: u32, dest: u32, weight: f32) {
        assert!((origin as usize) < self.nodes.len());
        assert!((dest as usize) < self.nodes.len());

        // TODO: Verify that the connection added won't cause a loop.

        self.connections.push(Connection {
            origin,
            dest,
            weight,
            connection_type: ConnectionType::Normal,
        });
        self.sorted = false;
    }

    fn evaluate(&mut self, inputs: &[f32]) -> Vec<f32> {
        self.sort_connections();
        self.load_input_values(&inputs);

        {
            let ref mut connections = self.connections;
            let ref mut nodes = self.nodes;

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
    fn test_simple_net() {
        let mut net = ConsecutiveNeuralNet::new();
        net.add_node(NodeType::Input, ActivationFunction::Identity);
        net.add_node(NodeType::Input, ActivationFunction::Identity);
        net.add_node(NodeType::Output, ActivationFunction::Identity);
        net.add_connection(0, 2, 1.0);
        net.add_connection(1, 2, -1.0);

        let res = net.evaluate(&[0.5, 1.5]);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], -1.0);
    }

    #[test]
    fn test_multilayered_net() {
        let func = ActivationFunction::Sigmoid;
        let mut net = ConsecutiveNeuralNet::new();
        net.add_node(NodeType::Input, ActivationFunction::Identity);
        net.add_node(NodeType::Hidden, func);
        net.add_node(NodeType::Output, func);
        // First connection added first
        net.add_connection(0, 1, 1.0);
        net.add_connection(1, 2, 1.0);

        let res = net.evaluate(&[0.0]);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], func.apply(func.apply(0.0)));

        let mut net = ConsecutiveNeuralNet::new();
        net.add_node(NodeType::Input, ActivationFunction::Identity);
        net.add_node(NodeType::Hidden, func);
        net.add_node(NodeType::Output, func);
        // Second connection added first, needs to be sorted.
        net.add_connection(1, 2, 1.0);
        net.add_connection(0, 1, 1.0);

        let res = net.evaluate(&[0.0]);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], func.apply(func.apply(0.0)));
    }
}
