from copy import deepcopy


def crossover(tree1, tree2):
    # Select a random node from each tree
    node1, parent1 = tree1.get_random_node()
    node2, parent2 = tree2.get_random_node()
    # Ensure the nodes are not the same
    while node1 is node2:
        node2, parent2 = tree2.get_random_node()
    # Swap the subtrees
    if parent1 is None:
        tree1.root = deepcopy(node2)
    else:
        for i, child in enumerate(parent1.children):
            if child is node1:
                parent1.children[i] = deepcopy(node2)
                break
    if parent2 is None:
        tree2.root = deepcopy(node1)
    else:
        for i, child in enumerate(parent2.children):
            if child is node2:
                parent2.children[i] = deepcopy(node1)
                break
    return tree1, tree2
