use crate::block::Block;
use crate::node::{Node, NodeType, TensorSpec};
use indexmap::IndexMap;

pub struct BlockSpec {
    pub blocks: Vec<Block>,
    pub global_sd_order: Vec<TensorSpec>,
    pub sd_index_map: IndexMap<TensorSpec, usize>,
    pub node_global_location_map: IndexMap<Node, (usize, usize)>, // (sd_index, position)
    pub block_to_global_slice_spec: Vec<Vec<usize>>,
    pub node_shape_dtypes: IndexMap<NodeType, TensorSpec>,
}

impl BlockSpec {
    pub fn new(
        blocks: Vec<Block>,
        node_shape_dtypes: IndexMap<NodeType, TensorSpec>,
    ) -> Result<Self, String> {
        // Build global_sd_order (unique TensorSpecs)
        let mut global_sd_order = Vec::new();
        let mut sd_index_map = IndexMap::new();
        for spec in node_shape_dtypes.values() {
            if !sd_index_map.contains_key(spec) {
                let idx = global_sd_order.len();
                global_sd_order.push(spec.clone());
                sd_index_map.insert(spec.clone(), idx);
            }
        }

        // Build node_global_location_map and block_to_global_slice_spec
        let mut node_global_location_map = IndexMap::new();
        let mut block_to_global_slice_spec = vec![Vec::new(); global_sd_order.len()];
        let mut arr_ind_tracker = vec![0; global_sd_order.len()];

        for (block_idx, block) in blocks.iter().enumerate() {
            if block.is_empty() {
                return Err("Encountered an empty block in BlockSpec".to_string());
            }

            let spec = node_shape_dtypes.get(block.node_type()).ok_or_else(|| {
                format!(
                    "Node type {:?} not found in node_shape_dtypes",
                    block.node_type()
                )
            })?;
            let sd_ind = *sd_index_map.get(spec).unwrap();
            let start_ind = arr_ind_tracker[sd_ind];
            arr_ind_tracker[sd_ind] += block.len();
            block_to_global_slice_spec[sd_ind].push(block_idx);

            for (k, node) in block.nodes().iter().enumerate() {
                if node_global_location_map.contains_key(node) {
                    return Err("Node appears twice in blocks".to_string());
                }
                node_global_location_map.insert(node.clone(), (sd_ind, start_ind + k));
            }
        }

        Ok(Self {
            blocks,
            global_sd_order,
            sd_index_map,
            node_global_location_map,
            block_to_global_slice_spec,
            node_shape_dtypes,
        })
    }

    pub fn get_node_locations(&self, block: &Block) -> Result<(usize, Vec<usize>), String> {
        let node_sds = self
            .node_shape_dtypes
            .get(block.node_type())
            .ok_or_else(|| format!("Node type {:?} not found", block.node_type()))?;
        let sd_inds = *self
            .sd_index_map
            .get(node_sds)
            .ok_or_else(|| "SD not found in index map".to_string())?;
        let global_locs: Vec<usize> = block
            .nodes()
            .iter()
            .map(|node| {
                self.node_global_location_map
                    .get(node)
                    .map(|(_, pos)| *pos)
                    .unwrap_or_else(|| panic!("Node not found in global location map"))
            })
            .collect();
        Ok((sd_inds, global_locs))
    }
}
