import torch
from specdecode.tree import Node, flatten
from specdecode.controller import DraftController
from specdecode.reference import verify_candidates

def test_flatten():
    tree = Node(1, [Node(2, [Node(4)]), Node(3)])
    flat = flatten(tree)
    assert flat.token_ids == [1, 2, 4, 3]
    assert flat.parents == [-1, 0, 1, 0]

def test_controller():
    c = DraftController(initial=4)
    assert c.update(.9, 4) == 5
    assert c.update(.2, 20) == 4

def test_reference():
    logits = torch.tensor([[0., 2., 1.], [3., 1., 0.]])
    tokens = torch.tensor([1, 0])
    assert verify_candidates(logits, tokens).all()
