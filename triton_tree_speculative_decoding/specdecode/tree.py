from dataclasses import dataclass, field

@dataclass
class Node:
    token_id: int
    children: list["Node"] = field(default_factory=list)

@dataclass
class FlatTree:
    token_ids: list[int]
    parents: list[int]
    depths: list[int]

def flatten(root: Node) -> FlatTree:
    token_ids, parents, depths = [], [], []
    stack = [(root, -1, 0)]
    while stack:
        node, parent, depth = stack.pop()
        idx = len(token_ids)
        token_ids.append(node.token_id)
        parents.append(parent)
        depths.append(depth)
        for child in reversed(node.children):
            stack.append((child, idx, depth + 1))
    return FlatTree(token_ids, parents, depths)
