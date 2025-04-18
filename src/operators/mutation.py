from copy import deepcopy
from parse_tree import FunctionNode


def subtree_mutation(tree, function_set, terminal_rules, max_depth):
    mutated_tree = deepcopy(tree)
    node_to_replace, parent = mutated_tree.get_random_node()
    new_subtree = FunctionNode.from_function_set(
        function_set, terminal_rules, max_depth, terminal_prob=0.1
    )

    if parent is None:
        mutated_tree.root = new_subtree
    else:
        for i, child in enumerate(parent.children):
            if child is node_to_replace:
                parent.children[i] = new_subtree
                break

    return mutated_tree
