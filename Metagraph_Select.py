import numpy as np
from typing import List, Tuple, Set
from itertools import combinations


def parse_multiple_metagraphs(filepath: str) -> List[Tuple[List[int], List[Tuple[int, int]]]]:
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    metagraphs = []
    current_node_types = None
    current_edge_pairs = []
    in_metagraph = False

    for line in lines:
        if line.startswith('#'):
            if in_metagraph and current_node_types is not None:

            in_metagraph = True
            current_node_types = None
            current_edge_pairs = []
        elif line.startswith('T'):
            if not in_metagraph:
                continue
            parts = line.split()
            current_node_types = list(map(int, parts[1:]))
        elif line.startswith('E'):
            if not in_metagraph or current_node_types is None:
                continue
            parts = list(map(int, line.split()[1:]))
            for i in range(0, len(parts), 2):
                u, v = parts[i], parts[i + 1]
                current_edge_pairs.append((u, v))

    if in_metagraph and current_node_types is not None:
        metagraphs.append((current_node_types, current_edge_pairs))


    result = []
    for node_types, edge_pairs in metagraphs:
        type_of_node = {idx: node_types[idx] for idx in range(len(node_types))}
        undirected_etype_pairs = []
        for u, v in edge_pairs:
            tu, tv = type_of_node[u], type_of_node[v]
            if tu > tv:
                tu, tv = tv, tu
            undirected_etype_pairs.append((tu, tv))
        result.append((node_types, undirected_etype_pairs))

    return result


def infer_type_universes(all_node_types_list: List[List[int]],
                         all_edge_pairs_list: List[List[Tuple[int, int]]]):
    node_set: Set[int] = set()
    edge_set: Set[Tuple[int, int]] = set()
    for node_types in all_node_types_list:
        node_set.update(node_types)
    for edge_pairs in all_edge_pairs_list:
        edge_set.update(edge_pairs)
    node_type_order = sorted(node_set)
    edge_type_order = sorted(edge_set)
    return node_type_order, edge_type_order



def to_distribution(node_types: List[int],
                    edge_pairs: List[Tuple[int, int]],
                    node_type_order: List[int],
                    edge_type_order: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    node_dist = [0] * len(node_type_order)
    type_to_idx = {t: i for i, t in enumerate(node_type_order)}
    for t in node_types:
        node_dist[type_to_idx[t]] += 1

    edge_dist = [0] * len(edge_type_order)
    etype_to_idx = {e: i for i, e in enumerate(edge_type_order)}
    for e in edge_pairs:
        edge_dist[etype_to_idx[e]] += 1

    return np.array(node_dist, dtype=int), np.array(edge_dist, dtype=int)

def select_diverse_metagraphs(distributions: List[Tuple[np.ndarray, np.ndarray]],
                              k1: int) -> List[int]:
    n = len(distributions)
    if k1 > n:
        raise ValueError(f"k1 ({k1}) 不能大于元图总数 ({n})")
    if k1 < 2:
        raise ValueError("k1 必须至少为 data")

    # 计算差异矩阵
    diff_matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        node_diff = np.sum(np.abs(distributions[i][0] - distributions[j][0]))
        edge_diff = np.sum(np.abs(distributions[i][1] - distributions[j][1]))
        diff = node_diff + edge_diff
        diff_matrix[i, j] = diff
        diff_matrix[j, i] = diff

    # 初始选择：差异最大的两个
    flat_idx = np.argmax(diff_matrix)
    i1, i2 = divmod(flat_idx, n)
    selected = [i1, i2]

    remaining = set(range(n)) - {i1, i2}

    # 迭代添加
    for _ in range(k1 - 2):
        best_idx = -1
        best_avg = -1.0
        for idx in remaining:
            avg_diff = np.mean([diff_matrix[idx, s] for s in selected])
            if avg_diff > best_avg:
                best_avg = avg_diff
                best_idx = idx
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected



def select_diverse_from_file(filepath: str, k1: int, output=True):
    # 解析所有元图
    metagraphs_raw = parse_multiple_metagraphs(filepath)
    if len(metagraphs_raw) == 0:
        raise ValueError(f"文件 {filepath} 中没有解析到任何元图")

    all_node_types = [m[0] for m in metagraphs_raw]
    all_edge_pairs = [m[1] for m in metagraphs_raw]

    node_order, edge_order = infer_type_universes(all_node_types, all_edge_pairs)

    distributions = []
    for node_types, edge_pairs in metagraphs_raw:
        dist = to_distribution(node_types, edge_pairs, node_order, edge_order)
        distributions.append(dist)

    selected_indices = select_diverse_metagraphs(distributions, k1)

    if output:

        for idx in selected_indices:
            node_dist, edge_dist = distributions[idx]

    return selected_indices, [distributions[i] for i in selected_indices]

if __name__ == "__main__":
    file_path = "IMDB/data/imdb.gdb"
    k1 = 5
    selected_idx, _ = select_diverse_from_file(file_path, k1)