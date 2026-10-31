//! Pure capability-partition algorithms over the owned graph IR.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use onnx_runtime_ir::{Graph, NodeId, ValueId};

use crate::registry::CompilePartitionClass;

pub(crate) fn build_contiguous_clusters(
    order: &[NodeId],
    supported: &[bool],
    float64: &[bool],
    compile_class: &[CompilePartitionClass],
) -> Vec<Vec<NodeId>> {
    debug_assert_eq!(order.len(), supported.len());
    let mut clusters = Vec::new();
    let mut start = 0usize;
    while start < supported.len() {
        while start < supported.len() && !supported[start] {
            start += 1;
        }
        if start == supported.len() {
            break;
        }
        let mut end = start + 1;
        while end < supported.len()
            && supported[end]
            && float64[end] == float64[start]
            && compile_class[end] == compile_class[start]
        {
            end += 1;
        }
        clusters.push(order[start..end].to_vec());
        start = end;
    }
    clusters
}

/// Prefer convex edge groups; use topological runs to reduce fragmentation or break cycles.
/// A set S is convex iff no outside node lies on a path between two members of S.
pub(crate) fn build_convex_clusters(
    graph: &Graph,
    order: &[NodeId],
    supported: &[bool],
    float64: &[bool],
    compile_class: &[CompilePartitionClass],
) -> Vec<Vec<NodeId>> {
    let positions: HashMap<NodeId, usize> = order
        .iter()
        .enumerate()
        .map(|(index, &node)| (node, index))
        .collect();
    let succ = order
        .iter()
        .map(|&node| {
            graph
                .successors(node)
                .into_iter()
                .filter_map(|successor| positions.get(&successor).copied())
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let pred = order
        .iter()
        .map(|&node| {
            graph
                .predecessors(node)
                .into_iter()
                .filter_map(|predecessor| positions.get(&predecessor).copied())
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    cluster_edges(supported, float64, compile_class, &succ, &pred)
        .into_iter()
        .map(|cluster| cluster.into_iter().map(|index| order[index]).collect())
        .collect()
}

/// Keep proven operator regions intact while partitioning the remaining graph.
/// Regions must be supported, homogeneous, disjoint and convex. A cyclic region
/// quotient falls back to ordinary partitioning without changing graph semantics.
pub(crate) fn build_convex_clusters_with_regions(
    graph: &Graph,
    order: &[NodeId],
    supported: &[bool],
    float64: &[bool],
    compile_class: &[CompilePartitionClass],
    regions: &[Vec<NodeId>],
) -> Vec<Vec<NodeId>> {
    let positions: HashMap<_, _> = order
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();
    let successors: Vec<Vec<usize>> = order
        .iter()
        .map(|&node| {
            graph
                .successors(node)
                .into_iter()
                .filter_map(|next| positions.get(&next).copied())
                .collect()
        })
        .collect();
    let ordered = successors
        .iter()
        .enumerate()
        .all(|(i, next)| next.iter().all(|&j| i < j));
    let mut covered = vec![false; order.len()];
    let mut units = Vec::new();
    for region in regions {
        let Some(mut nodes) = region
            .iter()
            .map(|n| positions.get(n).copied())
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        nodes.sort_unstable();
        nodes.dedup();
        let Some(&first) = nodes.first() else {
            continue;
        };
        if nodes.iter().any(|&i| {
            covered[i]
                || !supported[i]
                || float64[i] != float64[first]
                || compile_class[i] != compile_class[first]
        }) {
            continue;
        }
        let last = *nodes.last().unwrap();
        let members: HashSet<_> = nodes.iter().copied().collect();
        let mut pending: Vec<_> = nodes
            .iter()
            .flat_map(|&i| successors[i].iter().copied())
            .filter(|i| !members.contains(i))
            .collect();
        let mut visited = vec![false; order.len()];
        let mut convex = true;
        while let Some(node) = pending.pop() {
            if ordered && node > last {
                continue;
            }
            if members.contains(&node) {
                convex = false;
                break;
            }
            if visited[node] {
                continue;
            }
            visited[node] = true;
            pending.extend(successors[node].iter().copied());
        }
        if !convex {
            continue;
        }
        for &i in &nodes {
            covered[i] = true;
        }
        units.push(nodes);
    }
    if units.is_empty() {
        return build_convex_clusters(graph, order, supported, float64, compile_class);
    }
    units.extend((0..order.len()).filter(|&i| !covered[i]).map(|i| vec![i]));
    let mut owner = vec![0; order.len()];
    for (unit, nodes) in units.iter().enumerate() {
        for &node in nodes {
            owner[node] = unit;
        }
    }
    let mut edges = vec![HashSet::new(); units.len()];
    let mut indegree = vec![0usize; units.len()];
    for (node, next) in successors.iter().enumerate() {
        for &next in next {
            let (a, b) = (owner[node], owner[next]);
            if a != b && edges[a].insert(b) {
                indegree[b] += 1;
            }
        }
    }
    let mut ready = BinaryHeap::new();
    for (i, nodes) in units.iter().enumerate() {
        if indegree[i] == 0 {
            ready.push(Reverse((nodes[0], i)));
        }
    }
    let mut topology = Vec::with_capacity(units.len());
    while let Some(Reverse((_, unit))) = ready.pop() {
        topology.push(unit);
        for &next in &edges[unit] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.push(Reverse((units[next][0], next)));
            }
        }
    }
    if topology.len() != units.len() {
        return build_convex_clusters(graph, order, supported, float64, compile_class);
    }
    let mut remap = vec![0; units.len()];
    for (position, &unit) in topology.iter().enumerate() {
        remap[unit] = position;
    }
    let next: Vec<Vec<usize>> = topology
        .iter()
        .map(|&unit| {
            let mut next: Vec<_> = edges[unit].iter().map(|&i| remap[i]).collect();
            next.sort_unstable();
            next
        })
        .collect();
    let mut previous = vec![Vec::new(); units.len()];
    for (i, next) in next.iter().enumerate() {
        for &j in next {
            previous[j].push(i);
        }
    }
    let ok: Vec<_> = topology.iter().map(|&i| supported[units[i][0]]).collect();
    let wide: Vec<_> = topology.iter().map(|&i| float64[units[i][0]]).collect();
    let classes: Vec<_> = topology
        .iter()
        .map(|&i| compile_class[units[i][0]])
        .collect();
    cluster_edges(&ok, &wide, &classes, &next, &previous)
        .into_iter()
        .map(|cluster| {
            let mut expanded: Vec<_> = cluster
                .into_iter()
                .flat_map(|i| units[topology[i]].iter().copied())
                .collect();
            expanded.sort_unstable();
            expanded.into_iter().map(|i| order[i]).collect()
        })
        .collect()
}

pub(crate) fn infer_layer_boundary_values(
    graph: &Graph,
    order: &[NodeId],
    attention_anchors: &[NodeId],
) -> HashSet<ValueId> {
    let positions: HashMap<NodeId, usize> = order
        .iter()
        .enumerate()
        .map(|(index, &node)| (node, index))
        .collect();
    let anchors = attention_anchors
        .iter()
        .filter_map(|node| positions.get(node).copied())
        .collect::<Vec<_>>();
    let op_types = order
        .iter()
        .map(|&node| graph.node(node).op_type.clone())
        .collect::<Vec<_>>();
    let successors = order
        .iter()
        .map(|&node| {
            graph
                .successors(node)
                .into_iter()
                .filter_map(|successor| positions.get(&successor).copied())
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let predecessors = order
        .iter()
        .map(|&node| {
            graph
                .predecessors(node)
                .into_iter()
                .filter_map(|predecessor| positions.get(&predecessor).copied())
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    infer_boundary_indices(&op_types, &successors, &predecessors, &anchors)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|index| graph.node(order[index]).outputs.first().copied())
        .collect()
}

pub(crate) fn split_annotated_layer_clusters(
    graph: &Graph,
    clusters: Vec<Vec<NodeId>>,
    layer_boundary_outputs: &HashSet<ValueId>,
    layers_per_partition: usize,
) -> Vec<Vec<NodeId>> {
    let mut split = Vec::with_capacity(clusters.len());
    for cluster in clusters {
        let mut part = Vec::new();
        let mut layers_in_part = 0usize;
        for node in cluster {
            part.push(node);
            let layer_end = graph
                .node(node)
                .outputs
                .iter()
                .any(|output| layer_boundary_outputs.contains(output));
            if layer_end {
                layers_in_part += 1;
                if layers_in_part == layers_per_partition {
                    split.push(std::mem::take(&mut part));
                    layers_in_part = 0;
                }
            }
        }
        if !part.is_empty() {
            split.push(part);
        }
    }
    split
}

fn infer_boundary_indices(
    op_types: &[String],
    successors: &[Vec<usize>],
    predecessors: &[Vec<usize>],
    attention_anchors: &[usize],
) -> Option<Vec<usize>> {
    if attention_anchors.len() < 2 {
        return None;
    }
    let mut boundaries = Vec::with_capacity(attention_anchors.len() - 1);
    for pair in attention_anchors.windows(2) {
        let (current, next) = (pair[0], pair[1]);
        let mut downstream = vec![false; op_types.len()];
        let mut stack = vec![current];
        while let Some(index) = stack.pop() {
            for &successor in &successors[index] {
                if successor <= next && !downstream[successor] {
                    downstream[successor] = true;
                    stack.push(successor);
                }
            }
        }
        let mut upstream = vec![false; op_types.len()];
        let mut stack = vec![next];
        while let Some(index) = stack.pop() {
            for &predecessor in &predecessors[index] {
                if predecessor >= current && !upstream[predecessor] {
                    upstream[predecessor] = true;
                    stack.push(predecessor);
                }
            }
        }
        boundaries.push(
            (current + 1..next)
                .rev()
                .find(|&index| op_types[index] == "Add" && downstream[index] && upstream[index])?,
        );
    }
    Some(boundaries)
}

fn convex_bitsets(members: &[u64], descendants: &[u64], ancestors: &[u64]) -> bool {
    members
        .iter()
        .zip(descendants)
        .zip(ancestors)
        .all(|((&members, &descendants), &ancestors)| descendants & ancestors & !members == 0)
}

fn cluster_edges(
    supported: &[bool],
    float64: &[bool],
    compile_class: &[CompilePartitionClass],
    succ: &[Vec<usize>],
    pred: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let n = supported.len();
    debug_assert_eq!(n, float64.len());
    debug_assert_eq!(n, compile_class.len());
    debug_assert_eq!(n, succ.len());
    debug_assert_eq!(n, pred.len());
    let words = n.div_ceil(64);
    let mut indeg: Vec<usize> = pred.iter().map(Vec::len).collect();
    let mut stack: Vec<usize> = (0..n).filter(|&index| indeg[index] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(node) = stack.pop() {
        order.push(node);
        for &successor in &succ[node] {
            indeg[successor] -= 1;
            if indeg[successor] == 0 {
                stack.push(successor);
            }
        }
    }
    if order.len() != n {
        order = (0..n).collect();
    }

    let mut reach = vec![vec![0u64; words]; n];
    for &node in order.iter().rev() {
        for &successor in &succ[node] {
            bit_set(&mut reach[node], successor);
            let successor_reach = reach[successor].clone();
            bit_or_into(&mut reach[node], &successor_reach);
        }
    }

    // A cluster is convex exactly when every node both reachable from it and
    // able to reach it is already a member. Cache ancestor unions alongside
    // descendant unions so each tentative merge checks one bitset, not all nodes.
    let mut ancestor_bits = vec![vec![0u64; words]; n];
    for &node in &order {
        for &predecessor in &pred[node] {
            bit_set(&mut ancestor_bits[node], predecessor);
            let ancestors = ancestor_bits[predecessor].clone();
            bit_or_into(&mut ancestor_bits[node], &ancestors);
        }
    }

    let mut parent: Vec<usize> = (0..n).collect();
    let mut cluster_bits = vec![vec![0u64; words]; n];
    let mut reach_bits = vec![vec![0u64; words]; n];
    for index in 0..n {
        if supported[index] {
            bit_set(&mut cluster_bits[index], index);
            reach_bits[index] = reach[index].clone();
        }
    }

    let mut edges = Vec::new();
    for node in 0..n {
        if !supported[node] {
            continue;
        }
        for &successor in &succ[node] {
            if supported[successor]
                && float64[node] == float64[successor]
                && compile_class[node] == compile_class[successor]
            {
                edges.push((node, successor));
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &(left, right) in &edges {
            let left_root = uf_find(&mut parent, left);
            let right_root = uf_find(&mut parent, right);
            if left_root == right_root {
                continue;
            }
            let mut merged = cluster_bits[left_root].clone();
            bit_or_into(&mut merged, &cluster_bits[right_root]);
            let mut merged_reach = reach_bits[left_root].clone();
            bit_or_into(&mut merged_reach, &reach_bits[right_root]);
            let mut merged_ancestors = ancestor_bits[left_root].clone();
            bit_or_into(&mut merged_ancestors, &ancestor_bits[right_root]);
            if !convex_bitsets(&merged, &merged_reach, &merged_ancestors) {
                continue;
            }
            parent[right_root] = left_root;
            cluster_bits[left_root] = merged;
            reach_bits[left_root] = merged_reach;
            ancestor_bits[left_root] = merged_ancestors;
            changed = true;
        }
    }

    let mut grouped = HashMap::<usize, Vec<usize>>::new();
    for (node, &is_supported) in supported.iter().enumerate() {
        if is_supported {
            grouped
                .entry(uf_find(&mut parent, node))
                .or_default()
                .push(node);
        }
    }
    let mut clusters = grouped
        .into_values()
        .map(|mut cluster| {
            cluster.sort_unstable();
            cluster
        })
        .collect::<Vec<_>>();

    clusters.sort_by_key(|cluster| cluster[0]);

    let mut qid = vec![usize::MAX; n];
    for (cluster_id, cluster) in clusters.iter().enumerate() {
        for &node in cluster {
            qid[node] = cluster_id;
        }
    }
    let mut next = clusters.len();
    for id in &mut qid {
        if *id == usize::MAX {
            *id = next;
            next += 1;
        }
    }
    let mut qsucc = vec![HashSet::new(); next];
    let mut qindeg = vec![0usize; next];
    for node in 0..n {
        for &successor in &succ[node] {
            let (from, to) = (qid[node], qid[successor]);
            if from != to && qsucc[from].insert(to) {
                qindeg[to] += 1;
            }
        }
    }
    let mut stack: Vec<usize> = (0..next).filter(|&node| qindeg[node] == 0).collect();
    let mut visited = 0usize;
    while let Some(node) = stack.pop() {
        visited += 1;
        for &successor in &qsucc[node] {
            qindeg[successor] -= 1;
            if qindeg[successor] == 0 {
                stack.push(successor);
            }
        }
    }
    if visited != next {
        return topological_colour_runs(supported, float64, compile_class, succ, pred);
    }
    // Schedule existing convex regions as atomic units. Only shapeless regions
    // may coalesce across independent branches: shape-keyed and eager regions
    // retain their input boundaries and cache keys.
    let mut units = clusters.clone();
    units.extend(
        (0..n)
            .filter(|&node| !supported[node])
            .map(|node| vec![node]),
    );
    let eligible: Vec<_> = units
        .iter()
        .map(|unit| {
            let node = unit[0];
            supported[node] && compile_class[node] == CompilePartitionClass::Shapeless
        })
        .collect();
    let wide: Vec<_> = units.iter().map(|unit| float64[unit[0]]).collect();
    let classes = vec![CompilePartitionClass::Shapeless; next];
    let edges: Vec<Vec<_>> = qsucc
        .iter()
        .map(|edges| edges.iter().copied().collect())
        .collect();
    let mut parents = vec![Vec::new(); next];
    for (from, successors) in edges.iter().enumerate() {
        for &to in successors {
            parents[to].push(from);
        }
    }
    let mut runs: Vec<Vec<usize>> =
        topological_colour_runs(&eligible, &wide, &classes, &edges, &parents)
            .into_iter()
            .map(|run| {
                let mut nodes: Vec<_> = run
                    .into_iter()
                    .flat_map(|unit| units[unit].iter().copied())
                    .collect();
                nodes.sort_unstable();
                nodes
            })
            .collect();
    runs.extend(
        clusters
            .iter()
            .filter(|cluster| !eligible[qid[cluster[0]]])
            .cloned(),
    );
    if runs.len() < clusters.len() {
        runs.sort_by_key(|cluster| cluster[0]);
        return runs;
    }
    clusters.sort_by_key(|cluster| cluster[0]);
    clusters
}

/// Keep a ready colour running until its queue empties. Each run remains a
/// contiguous interval of a valid topological order, hence a convex region.
fn topological_colour_runs(
    supported: &[bool],
    float64: &[bool],
    compile_class: &[CompilePartitionClass],
    succ: &[Vec<usize>],
    pred: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    const CLASSES: usize = 3;
    const UNSUPPORTED: usize = 2 * CLASSES;
    let colour = |node: usize| {
        if !supported[node] {
            return UNSUPPORTED;
        }
        usize::from(float64[node]) * CLASSES
            + match compile_class[node] {
                CompilePartitionClass::Shapeless => 0,
                CompilePartitionClass::ShapeKeyed => 1,
                CompilePartitionClass::Eager => 2,
            }
    };
    let mut ready: Vec<BinaryHeap<Reverse<usize>>> =
        (0..=UNSUPPORTED).map(|_| BinaryHeap::new()).collect();
    let mut indegree: Vec<_> = pred.iter().map(Vec::len).collect();
    for (node, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            ready[colour(node)].push(Reverse(node));
        }
    }
    let mut current: Option<usize> = None;
    let mut previous = None;
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    loop {
        let next = current.filter(|&kind| !ready[kind].is_empty()).or_else(|| {
            ready
                .iter()
                .enumerate()
                .filter_map(|(kind, queue)| queue.peek().map(|node| (kind, node.0)))
                .min_by_key(|&(_, node)| node)
                .map(|(kind, _)| kind)
        });
        let Some(kind) = next else { break };
        let Reverse(node) = ready[kind].pop().unwrap();
        current = Some(kind);
        if supported[node] {
            if previous != Some(kind) {
                clusters.push(Vec::new());
            }
            clusters.last_mut().unwrap().push(node);
            previous = Some(kind);
        } else {
            previous = None;
        }
        for &successor in &succ[node] {
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                ready[colour(successor)].push(Reverse(successor));
            }
        }
    }
    clusters
}

fn uf_find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

fn bit_set(bits: &mut [u64], index: usize) {
    bits[index >> 6] |= 1u64 << (index & 63);
}

fn bit_or_into(destination: &mut [u64], source: &[u64]) {
    for index in 0..destination.len() {
        destination[index] |= source[index];
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_contiguous_clusters, build_convex_clusters, cluster_edges, convex_bitsets,
        infer_boundary_indices, topological_colour_runs,
    };
    use crate::registry::CompilePartitionClass::{Eager, ShapeKeyed, Shapeless};
    use onnx_runtime_ir::{DataType, Graph, Node};

    fn region_graph(parents: &[&[usize]]) -> (Graph, Vec<onnx_runtime_ir::NodeId>) {
        let mut graph = Graph::new();
        let input = graph.create_named_value("input", DataType::Float32, Vec::new());
        let values: Vec<_> = (0..parents.len())
            .map(|i| graph.create_named_value(format!("v{i}"), DataType::Float32, Vec::new()))
            .collect();
        let mut nodes = Vec::new();
        for (i, parents) in parents.iter().enumerate() {
            let inputs = if parents.is_empty() {
                vec![Some(input)]
            } else {
                parents.iter().map(|&p| Some(values[p])).collect()
            };
            nodes.push(graph.insert_node(Node::new(
                onnx_runtime_ir::NodeId(0),
                "Compute",
                inputs,
                vec![values[i]],
            )));
        }
        (graph, nodes)
    }

    #[test]
    fn atomic_regions_keep_weight_prefixes_with_their_products() {
        let (graph, order) = region_graph(&[&[], &[], &[], &[0, 2], &[3], &[1, 4]]);
        let classes = vec![Eager, Eager, Shapeless, Eager, Shapeless, Eager];
        let regions = vec![vec![order[0], order[3]], vec![order[1], order[5]]];
        let clusters = super::build_convex_clusters_with_regions(
            &graph,
            &order,
            &[true; 6],
            &[false; 6],
            &classes,
            &regions,
        );
        for region in regions {
            assert!(
                clusters
                    .iter()
                    .any(|c| region.iter().all(|n| c.contains(n)))
            );
        }
        assert_eq!(clusters.len(), 4);
    }

    #[test]
    fn atomic_regions_preserve_cache_and_dtype_boundaries() {
        let (graph, order) = region_graph(&[&[], &[0]]);
        for (classes, wide) in [
            (vec![Eager, ShapeKeyed], vec![false, false]),
            (vec![Eager, Eager], vec![false, true]),
        ] {
            let clusters = super::build_convex_clusters_with_regions(
                &graph,
                &order,
                &[true; 2],
                &wide,
                &classes,
                &[order.clone()],
            );
            assert_eq!(clusters.len(), 2);
        }
    }

    #[test]
    fn nonconvex_atomic_region_is_declined() {
        let (graph, order) = region_graph(&[&[], &[0], &[1]]);
        let classes = vec![Eager, Shapeless, Eager];
        let clusters = super::build_convex_clusters_with_regions(
            &graph,
            &order,
            &[true; 3],
            &[false; 3],
            &classes,
            &[vec![order[0], order[2]]],
        );
        assert_eq!(clusters.len(), 3);
    }

    #[test]
    fn cyclic_atomic_quotient_uses_ordinary_convex_groups() {
        let (graph, order) = region_graph(&[&[], &[], &[0], &[1]]);
        let clusters = super::build_convex_clusters_with_regions(
            &graph,
            &order,
            &[true; 4],
            &[false; 4],
            &[Eager; 4],
            &[vec![order[0], order[3]], vec![order[1], order[2]]],
        );
        assert_eq!(
            clusters,
            vec![vec![order[0], order[2]], vec![order[1], order[3]]]
        );
    }

    #[test]
    fn convex_bitsets_match_path_definition_exhaustively() {
        const N: usize = 5;
        let edges = (0..N)
            .flat_map(|a| (a + 1..N).map(move |b| (a, b)))
            .collect::<Vec<_>>();
        for edge_mask in 0..1usize << edges.len() {
            let mut reach = [0u64; N];
            for (i, &(a, b)) in edges.iter().enumerate() {
                if edge_mask & (1 << i) != 0 {
                    reach[a] |= 1 << b;
                }
            }
            for k in (0..N).rev() {
                for a in 0..k {
                    if reach[a] & (1 << k) != 0 {
                        reach[a] |= reach[k];
                    }
                }
            }
            for members in 0..1u64 << N {
                let mut descendants = 0;
                let mut ancestors = 0;
                for (node, &reachable) in reach.iter().enumerate() {
                    if members & (1 << node) != 0 {
                        descendants |= reachable;
                    }
                    if reachable & members != 0 {
                        ancestors |= 1 << node;
                    }
                }
                let path_definition = reach.iter().enumerate().all(|(node, &reachable)| {
                    members & (1 << node) != 0
                        || descendants & (1 << node) == 0
                        || reachable & members == 0
                });
                assert_eq!(
                    convex_bitsets(&[members], &[descendants], &[ancestors]),
                    path_definition,
                    "edges={edge_mask}, members={members}"
                );
            }
        }
    }

    fn chain(n: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut succ = vec![Vec::new(); n];
        let mut pred = vec![Vec::new(); n];
        for index in 0..n.saturating_sub(1) {
            succ[index].push(index + 1);
            pred[index + 1].push(index);
        }
        (succ, pred)
    }

    #[test]
    fn convex_clustering_handles_long_decoder_chains() {
        let n = 10_000;
        let (succ, pred) = chain(n);
        assert_eq!(
            cluster_edges(
                &vec![true; n],
                &vec![false; n],
                &vec![Shapeless; n],
                &succ,
                &pred
            ),
            vec![(0..n).collect::<Vec<_>>()]
        );
    }

    #[test]
    fn convex_clustering_preserves_colours_and_contiguity() {
        let n = 8;
        let (succ, pred) = chain(n);
        let supported = vec![true; n];
        let float64 = vec![false, false, true, true, true, true, false, false];
        assert_eq!(
            cluster_edges(&supported, &float64, &vec![Shapeless; n], &succ, &pred),
            vec![vec![0, 1], vec![2, 3, 4, 5], vec![6, 7]]
        );
        let order = (0..n)
            .map(|index| onnx_runtime_ir::NodeId(index as u32))
            .collect::<Vec<_>>();
        assert_eq!(
            build_contiguous_clusters(&order, &supported, &float64, &vec![Shapeless; n]),
            vec![
                vec![onnx_runtime_ir::NodeId(0), onnx_runtime_ir::NodeId(1)],
                vec![
                    onnx_runtime_ir::NodeId(2),
                    onnx_runtime_ir::NodeId(3),
                    onnx_runtime_ir::NodeId(4),
                    onnx_runtime_ir::NodeId(5)
                ],
                vec![onnx_runtime_ir::NodeId(6), onnx_runtime_ir::NodeId(7)]
            ]
        );
    }

    #[test]
    fn ir_topology_matches_legacy_adjacency_clustering() {
        let mut graph = Graph::new();
        let input = graph.create_named_value("input", DataType::Float32, Vec::new());
        let values = (0..5)
            .map(|index| {
                graph.create_named_value(format!("value_{index}"), DataType::Float32, Vec::new())
            })
            .collect::<Vec<_>>();
        let mut previous = input;
        for output in &values {
            graph.insert_node(Node::new(
                onnx_runtime_ir::NodeId(0),
                "Identity",
                vec![Some(previous)],
                vec![*output],
            ));
            previous = *output;
        }
        let order = graph.topological_order().unwrap();
        let (successors, predecessors) = chain(order.len());
        let supported = vec![true; order.len()];
        let float64 = vec![false, true, true, false, false];
        let classes = vec![Shapeless; order.len()];
        let legacy = cluster_edges(&supported, &float64, &classes, &successors, &predecessors);
        let ir = build_convex_clusters(&graph, &order, &supported, &float64, &classes)
            .into_iter()
            .map(|cluster| {
                cluster
                    .into_iter()
                    .map(|node| {
                        order
                            .iter()
                            .position(|&candidate| candidate == node)
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(ir, legacy);
    }

    #[test]
    fn compile_boundaries_remain_separate() {
        let (succ, pred) = chain(5);
        assert_eq!(
            cluster_edges(
                &[true; 5],
                &[false; 5],
                &[Shapeless, ShapeKeyed, Eager, ShapeKeyed, Shapeless],
                &succ,
                &pred,
            ),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
        );
    }

    #[test]
    fn disconnected_compatible_nodes_share_one_acyclic_region() {
        let empty = vec![vec![]; 4];
        assert_eq!(
            cluster_edges(&[true; 4], &[false; 4], &[Shapeless; 4], &empty, &empty),
            vec![vec![0, 1, 2, 3]]
        );
    }

    #[test]
    fn regrouping_preserves_independent_shape_keyed_cache_boundaries() {
        let succ = vec![vec![2], vec![], vec![], vec![]];
        let pred = vec![vec![], vec![], vec![0], vec![]];
        assert_eq!(
            cluster_edges(
                &[true; 4],
                &[false; 4],
                &[ShapeKeyed, ShapeKeyed, ShapeKeyed, Shapeless],
                &succ,
                &pred,
            ),
            vec![vec![0, 2], vec![1], vec![3]],
        );
    }

    #[test]
    fn ready_colour_runs_group_independent_interleaved_branches() {
        let succ = vec![vec![2], vec![3], vec![4], vec![5], vec![], vec![]];
        let pred = vec![vec![], vec![], vec![0], vec![1], vec![2], vec![3]];
        assert_eq!(
            topological_colour_runs(
                &[true; 6],
                &[false; 6],
                &[Shapeless, Eager, Shapeless, Eager, Shapeless, Eager],
                &succ,
                &pred,
            ),
            vec![vec![0, 2, 4], vec![1, 3, 5]]
        );
        let (succ, pred) = chain(3);
        assert_eq!(
            topological_colour_runs(
                &[true, false, true],
                &[false; 3],
                &[Shapeless; 3],
                &succ,
                &pred
            ),
            vec![vec![0], vec![2]]
        );
    }

    #[test]
    fn interleaved_convex_groups_keep_every_supported_node() {
        // Both {0,3} and {1,2} are convex, but contracting both creates a cycle.
        let succ = vec![vec![2, 3], vec![2, 3], vec![], vec![]];
        let pred = vec![vec![], vec![], vec![0, 1], vec![0, 1]];
        let classes = [Shapeless, ShapeKeyed, ShapeKeyed, Shapeless];
        let groups = cluster_edges(&[true; 4], &[false; 4], &classes, &succ, &pred);
        let mut nodes = groups.iter().flatten().copied().collect::<Vec<_>>();
        nodes.sort_unstable();
        assert_eq!(nodes, vec![0, 1, 2, 3]);
        assert_ne!(groups, vec![vec![0, 3], vec![1, 2]]);
        for group in groups {
            assert!(group.iter().all(|&node| classes[node] == classes[group[0]]));
        }
    }

    #[test]
    fn branched_partitions_cover_supported_nodes_with_an_acyclic_quotient() {
        for seed in 1u64..=256 {
            let n = 7;
            let mut state = seed;
            let mut next_bit = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                state >> 63 != 0
            };
            let mut succ = vec![Vec::new(); n];
            let mut pred = vec![Vec::new(); n];
            for (from, successors) in succ.iter_mut().enumerate() {
                for (to, predecessors) in pred.iter_mut().enumerate().skip(from + 1) {
                    if next_bit() {
                        successors.push(to);
                        predecessors.push(from);
                    }
                }
            }
            let supported = (0..n)
                .map(|_| seed % 2 == 0 || next_bit())
                .collect::<Vec<_>>();
            let wide = (0..n)
                .map(|_| seed % 3 == 0 && next_bit())
                .collect::<Vec<_>>();
            let classes = (0..n)
                .map(|_| if next_bit() { ShapeKeyed } else { Shapeless })
                .collect::<Vec<_>>();
            for groups in [
                cluster_edges(&supported, &wide, &classes, &succ, &pred),
                topological_colour_runs(&supported, &wide, &classes, &succ, &pred),
            ] {
                let mut owner = vec![usize::MAX; n];
                for (id, group) in groups.iter().enumerate() {
                    for &node in group {
                        assert!(supported[node]);
                        assert_eq!(owner[node], usize::MAX);
                        assert_eq!(wide[node], wide[group[0]]);
                        assert_eq!(classes[node], classes[group[0]]);
                        owner[node] = id;
                    }
                }
                let mut count = groups.len();
                for (node, id) in owner.iter_mut().enumerate() {
                    assert_eq!(*id != usize::MAX, supported[node]);
                    if *id == usize::MAX {
                        *id = count;
                        count += 1;
                    }
                }
                let mut outgoing = vec![Vec::new(); count];
                let mut incoming = vec![0; count];
                for (node, successors) in succ.iter().enumerate() {
                    for &to in successors {
                        let (a, b) = (owner[node], owner[to]);
                        if a != b {
                            outgoing[a].push(b);
                            incoming[b] += 1;
                        }
                    }
                }
                let mut ready = (0..count)
                    .filter(|&id| incoming[id] == 0)
                    .collect::<Vec<_>>();
                let mut seen = 0;
                while let Some(id) = ready.pop() {
                    seen += 1;
                    for &to in &outgoing[id] {
                        incoming[to] -= 1;
                        if incoming[to] == 0 {
                            ready.push(to);
                        }
                    }
                }
                assert_eq!(seen, count, "cyclic quotient for seed {seed}");
            }
        }
    }

    #[test]
    fn uniformly_coloured_graphs_remain_maximal() {
        let (succ, pred) = chain(8);
        assert_eq!(
            cluster_edges(&[true; 8], &[false; 8], &[Shapeless; 8], &succ, &pred,),
            vec![(0..8).collect::<Vec<_>>()]
        );
        let succ = vec![vec![1, 2], vec![3], vec![3], vec![]];
        let pred = vec![vec![], vec![0], vec![0], vec![1, 2]];
        assert_eq!(
            cluster_edges(&[true; 4], &[true; 4], &[Shapeless; 4], &succ, &pred,),
            vec![vec![0, 1, 2, 3]]
        );
    }

    #[test]
    fn shape_keyed_regions_and_contiguous_fallback_preserve_boundaries() {
        let (succ, pred) = chain(5);
        assert_eq!(
            cluster_edges(
                &[true; 5],
                &[false; 5],
                &[Shapeless, Shapeless, ShapeKeyed, Shapeless, Shapeless],
                &succ,
                &pred,
            ),
            vec![vec![0, 1], vec![2], vec![3, 4]]
        );
        let order = (0..5)
            .map(|index| onnx_runtime_ir::NodeId(index as u32))
            .collect::<Vec<_>>();
        assert_eq!(
            build_contiguous_clusters(
                &order,
                &[true; 5],
                &[false; 5],
                &[Shapeless, ShapeKeyed, ShapeKeyed, Shapeless, Shapeless],
            ),
            vec![
                vec![onnx_runtime_ir::NodeId(0)],
                vec![onnx_runtime_ir::NodeId(1), onnx_runtime_ir::NodeId(2)],
                vec![onnx_runtime_ir::NodeId(3), onnx_runtime_ir::NodeId(4)]
            ]
        );
    }

    #[test]
    fn structural_boundaries_use_the_last_residual_add() {
        let layers = 24;
        let mut op_types = Vec::new();
        let mut successors = Vec::new();
        let mut predecessors = Vec::new();
        let mut anchors = Vec::new();
        for _ in 0..layers {
            let anchor = op_types.len();
            anchors.push(anchor);
            op_types.extend(["GroupQueryAttention", "Add", "MatMul", "Add"].map(String::from));
            successors.extend([vec![], vec![], vec![], vec![]]);
            predecessors.extend([vec![], vec![], vec![], vec![]]);
        }
        for &anchor in anchors.iter().take(layers - 1) {
            let next = anchor + 4;
            for (from, to) in [
                (anchor, anchor + 1),
                (anchor + 1, anchor + 2),
                (anchor + 2, anchor + 3),
                (anchor + 3, next),
            ] {
                successors[from].push(to);
                predecessors[to].push(from);
            }
        }
        let boundaries =
            infer_boundary_indices(&op_types, &successors, &predecessors, &anchors).unwrap();
        assert_eq!(boundaries.len(), layers - 1);
        assert_eq!(boundaries[0], 3);
        assert_eq!(boundaries[layers - 2], (layers - 2) * 4 + 3);
    }
}
