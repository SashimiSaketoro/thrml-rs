use crate::node::{Node, NodeType};
use std::ops::Add;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    nodes: Vec<Node>,
    node_type: NodeType,
}

impl Block {
    pub fn new(nodes: Vec<Node>) -> Result<Self, String> {
        if nodes.is_empty() {
            return Err("Block cannot be empty".to_string());
        }
        let first_type = nodes[0].node_type().clone();
        for node in &nodes {
            if node.node_type() != &first_type {
                return Err("All nodes in a block must be of the same type".to_string());
            }
        }
        Ok(Block {
            nodes,
            node_type: first_type,
        })
    }

    pub fn node_type(&self) -> &NodeType {
        &self.node_type
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Node> {
        self.nodes.iter()
    }
}

impl Add for Block {
    type Output = Result<Block, String>;
    fn add(self, other: Block) -> Result<Block, String> {
        if self.node_type != other.node_type {
            return Err("Cannot add blocks of different node types".to_string());
        }
        let mut nodes = self.nodes;
        nodes.extend(other.nodes);
        Block::new(nodes)
    }
}

impl std::ops::Index<usize> for Block {
    type Output = Node;
    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}
