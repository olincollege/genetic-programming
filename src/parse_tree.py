import random


class ParseTree:
    def __init__(self, root):
        self.root: ParseNode = root

    def __repr__(self):
        return repr(self.root)

    @staticmethod
    def generate_full(
        function_set: list[str], terminal_set: list[str], depth: int
    ) -> "ParseTree":
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
        return ParseTree(
            FunctionNode.from_function_set(
                function_set, terminal_set, depth, terminal_prob
            )
        )


class ParseNode:
    def __init__(self, value):
        self.value: str = value

    def __repr__(self):
        pass


class FunctionNode(ParseNode):

    def __init__(self, value, children):
        super().__init__(value)
        self.arity: int = self.arity_map(value)
        self.children: list[ParseNode] = children

    def __repr__(self):
        args = " ".join([repr(child) for child in self.children])
        return f"({self.value} {args})"

    @staticmethod
    def arity_map(function) -> int:
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
        return self.value

    @staticmethod
    def from_terminal_set(terminal_set: list[str]) -> "TerminalNode":
        return TerminalNode(random.choice(terminal_set))
