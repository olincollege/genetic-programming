"""
Parse trees for genetic programming.
"""

import random


class ParseTree:
    """
    A parse tree, representing a mathematical expression in a tree structure.
    Must be at least depth 1.

    Attributes:
        root (FunctionNode): The root node of the parse tree.
    """

    def __init__(self, root):
        self.root: FunctionNode = root

    def __repr__(self):
        """
        Returns:
            str: The parse tree in prefix notation (FUNCTION ARG1 ARG2).
        """
        return repr(self.root)

    @staticmethod
    def generate_full(
        function_set: list[str], terminal_set: list[str], depth: int
    ) -> "ParseTree":
        """
        Generates a full parse tree of a given depth.

        Args:
            function_set (list[str]): The set of functions to use.
            terminal_set (list[str]): The set of terminals to use.
            depth (int): The depth of the tree. Expected to be 1 or greater

        Returns:
            ParseTree: A full parse tree with the given depth.
        """
        return ParseTree(
            FunctionNode.from_function_set(function_set, terminal_set, depth, 0.0)
        )

    @staticmethod
    def generate_grow(
        function_set: list[str],
        terminal_set: list[str],
        depth: int,
        terminal_prob: float,
    ) -> "ParseTree":
        """
        Generates a parse tree by "growing" it, randomizing between functions
        and terminals for each node up to the given depth. Not guaranteed to
        reach the given depth.

        Args:
            function_set (list[str]): The set of functions to use.
            terminal_set (list[str]): The set of terminals to use.
            depth (int): The maximium depth of the tree. Expected to be 1 or greater
            terminal_prob (float): The probability of a node being a terminal node.
                Expected to be between 0 and 1.

        Returns:
            ParseTree: A "grown" parse tree smaller or equal to the given depth.
        """
        return ParseTree(
            FunctionNode.from_function_set(
                function_set, terminal_set, depth, terminal_prob
            )
        )


class ParseNode:
    """
    A node in the parse tree. Can be a function or a terminal.

    Attributes:
        value (str): The value of the node, either a function or a terminal.
    """

    def __init__(self, value):
        self.value: str = value

    def __repr__(self):
        pass


class FunctionNode(ParseNode):
    """
    A function node in the parse tree. Represents a function with its arguments
    as children.

    Args:
        value (str): The function.
        arity (int): The number of arguments the function takes.
        children (list[ParseNode]): The arguments of the function.
    """

    def __init__(self, value, children):
        super().__init__(value)
        self.arity: int = self.arity_map(value)
        self.children: list[ParseNode] = children

    def __repr__(self):
        """
        Returns:
            str: The subtree of the FunctionNode in prefix notation
            (FUNCTION ARG1 ARG2).
        """
        args = " ".join([repr(child) for child in self.children])
        return f"({self.value} {args})"

    @staticmethod
    def arity_map(function: str) -> int:
        """
        Maps a function to its arity (number of arguments).

        Args:
            function (str): The function to map.

        Returns:
            int: The arity of the function.

        Raises:
            ValueError: If the function is not in the hardcoded list of
            available functions.
        """
        match function:
            case "+" | "-" | "*" | "/":
                return 2
            case "sin" | "cos" | "exp" | "ln":
                return 1
            case _:
                raise ValueError(f"Unknown function: {function}")

    @staticmethod
    def from_function_set(
        function_set: list[str],
        terminal_set: list[str],
        depth: int,
        terminal_prob: float,
    ) -> "FunctionNode":
        """
        Randomly generates a parse subtree. The function is chosen from the
        function set, and the children are randomly functions or terminals.

        Args:
            function_set (list[str]): The set of functions to use.
            terminal_set (list[str]): The set of terminals to use.
            depth (int): The maximium depth of the tree. Expected to be 1 or greater
            terminal_prob (float): The probability of a node being a terminal node.
                Expected to be between 0 and 1. A probability of 0 guarantees
                the subtree to be full.

        Returns:
            FunctionNode: The root of the generated subtree.
        """
        out = FunctionNode(
            random.choice(function_set),
            [],
        )
        # Randomize number of children based on arity of function
        children = []
        for _ in range(out.arity):
            if depth <= 1 or random.random() < terminal_prob:
                # Terminal
                children.append(TerminalNode.from_terminal_set(terminal_set))
            else:
                # Function
                children.append(
                    FunctionNode.from_function_set(
                        function_set, terminal_set, depth - 1, terminal_prob
                    )
                )
        out.children = children
        return out


class TerminalNode(ParseNode):

    def __repr__(self):
        """
        Returns:
            str: The value of the terminal node.
        """
        return self.value

    @staticmethod
    def from_terminal_set(terminal_set: list[str]) -> "TerminalNode":
        return TerminalNode(random.choice(terminal_set))
