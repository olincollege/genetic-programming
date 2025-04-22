"""
Genetic operators for tree-based genetic programming
"""

from copy import deepcopy
from parse_tree import ParseTree, FunctionNode, TerminalNode, TerminalGenerationRules


class GeneticOperators:
    """
    A class containing various genetic operators for tree-based genetic programming.
    """

    @staticmethod
    def subtree_mutation(
        tree: "ParseTree",
        function_set: list[str],
        terminal_rules: "TerminalGenerationRules",
        max_depth: int,
    ) -> "ParseTree":
        """
        Replace a random subtree in the tree with a new subtree generated from the function set.

        Args:
            tree (ParseTree): The parse tree to mutate.
            function_set (list): The set of functions to use for mutation.
            terminal_rules (TerminalGenerationRules): The rules for generating terminal nodes.
            max_depth (int): The maximum depth of the new subtree.

        Returns:
            ParseTree: The mutated parse tree.
        """
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

    @staticmethod
    def leaf_replacement(
        tree: "ParseTree", terminal_rules: "TerminalGenerationRules"
    ) -> "ParseTree":
        """
        Replace a random leaf node in the tree with a new terminal node.
        Args:
            tree (ParseTree): The parse tree to mutate.
            terminal_rules (TerminalGenerationRules): The rules for generating terminal nodes.

        Returns:
            ParseTree: The mutated parse tree.
        """
        mutated_tree = deepcopy(tree)
        leaf_node, parent = mutated_tree.get_random_node("leaf")

        new_leaf = TerminalNode.from_terminal_set(terminal_rules)

        if parent is None:
            mutated_tree.root = new_leaf
        else:
            # Find and replace the leaf node with the new terminal node
            for i, child in enumerate(parent.children):
                if child is leaf_node:
                    parent.children[i] = new_leaf
                    break

        return mutated_tree

    @staticmethod
    def crossover(
        tree1: "ParseTree", tree2: "ParseTree"
    ) -> tuple["ParseTree", "ParseTree"]:
        """
        Perform crossover between two parse trees.
        Selects a random node from each tree and swaps their subtrees.

        Args:
            tree1 (ParseTree): The first parse tree.
            tree2 (ParseTree): The second parse tree.

        Returns:
            tuple: The modified parse trees after crossover.
        """
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
