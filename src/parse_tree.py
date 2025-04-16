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
        function_set: list[str], terminal_rules: "TerminalGenerationRules", depth: int
    ) -> "ParseTree":
        """
        Generates a full parse tree of a given depth.

        Args:
            function_set (list[str]): The set of functions to use.
            terminal_rules (TerminalGenerationRules): The rules for generating
                terminals, a set of literals, and a range for generating random
                constants.
            depth (int): The depth of the tree. Expected to be 1 or greater

        Returns:
            ParseTree: A full parse tree with the given depth.
        """
        return ParseTree(
            FunctionNode.from_function_set(function_set, terminal_rules, depth, 0.0)
        )

    @staticmethod
    def generate_grow(
        function_set: list[str],
        terminal_rules: "TerminalGenerationRules",
        depth: int,
        terminal_prob: float,
    ) -> "ParseTree":
        """
        Generates a parse tree by "growing" it, randomizing between functions
        and terminals for each node up to the given depth. Not guaranteed to
        reach the given depth.

        Args:
            function_set (list[str]): The set of functions to use.
            terminal_rules (TerminalGenerationRules): The rules for generating
                terminals, a set of literals, and a range for generating random
                constants.
            depth (int): The maximium depth of the tree. Expected to be 1 or greater
            terminal_prob (float): The probability of a node being a terminal node.
                Expected to be between 0 and 1.

        Returns:
            ParseTree: A "grown" parse tree smaller or equal to the given depth.
        """
        return ParseTree(
            FunctionNode.from_function_set(
                function_set, terminal_rules, depth, terminal_prob
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
        terminal_rules: "TerminalGenerationRules",
        depth: int,
        terminal_prob: float,
    ) -> "FunctionNode":
        """
        Randomly generates a parse subtree. The function is chosen from the
        function set, and the children are randomly functions or terminals.

        Args:
            function_set (list[str]): The set of functions to use.
            terminal_rules (TerminalGenerationRules): The rules for generating
                terminals, a set of literals, and a range for generating random
                constants.
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
                children.append(TerminalNode.from_terminal_set(terminal_rules))
            else:
                # Function
                children.append(
                    FunctionNode.from_function_set(
                        function_set, terminal_rules, depth - 1, terminal_prob
                    )
                )
        out.children = children
        return out


class TerminalNode(ParseNode):
    """
    A terminal node in the parse tree. Either a variable or a randomly
    generated constant.
    """

    def __repr__(self):
        """
        Returns:
            str: The value of the terminal node.
        """
        return self.value

    @staticmethod
    def from_terminal_set(rules: "TerminalGenerationRules") -> "TerminalNode":
        """
        Randomly generates a terminal node based on the given rules. Each
        literal has the same chance of being chosen as generating a random
        constant. For example, if the literals are ["X", "Y"], there is a 1/3
        chance of choosing "X", 1/3 chance of choosing "Y", and 1/3 chance of
        generating a random constant.

        If only the `ints_only` flag is set to True, the generated constant is
        simply truncated to an integer.

        Args:
            rules (TerminalGenerationRules): The rules for generating terminals,
                a set of literals, and a range for generating random constants.

        Returns:
            TerminalNode: The generated terminal node.
        """
        options = len(rules.literals)
        if rules.no_random_constants:
            options -= 1
        res = random.randint(0, options)
        if res < len(rules.literals):
            # Literal
            return TerminalNode(rules.literals[res])
        # Random constant
        const = random.uniform(rules.constants_range[0], rules.constants_range[1])
        if rules.ints_only:
            const = int(const)
        else:
            const = round(const, rules.decimal_places)
        return TerminalNode(str(const))


class TerminalGenerationRules:
    """
    Rules for generating terminal nodes in the parse tree. Terminal nodes will
    either randomly select from the given literals or generate a random constant.

    Args:
        literals (list[str]): The set of literals to use.
        constants_range (tuple[float, float]): The minimum and maximum for
            generating random constants.
        decimal_places (int): Defaults to 4. The number of decimal places for
            the random constants.
        ints_only (bool): Defaults to False. If True, the generated constants
            will only be integers.
        no_random_constants (bool): Defaults to False. If True, terminals will
            only be chosen from literals.
    """

    def __init__(
        self,
        literals: list[str],
        constants_range: tuple[float, float],
        decimal_places: int = 4,
        ints_only: bool = False,
        no_random_constants: bool = False,
    ):
        self.literals = literals
        self.constants_range = constants_range
        self.decimal_places = decimal_places
        self.ints_only = ints_only
        self.no_random_constants = no_random_constants
