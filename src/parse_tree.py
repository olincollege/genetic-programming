class ParseTree:
    def __init__(self):
        self.root: ParseNode = None

    def __repr__(self):
        return repr(self.root)


class ParseNode:
    def __init__(self):
        self.value: str = None

    def __repr__(self):
        pass


class FunctionNode(ParseNode):

    def __init__(self):
        super().__init__()
        self.arity: int
        self.children: list[ParseNode] = []

    def __repr__(self):
        return " ".join([repr(child) for child in self.children])

    @staticmethod
    def arity_map(self, function):
        match function:
            case "+", "-", "*", "/":
                return 2
            case "sin", "cos", "exp", "ln":
                return 1


class TerminalNode(ParseNode):
    def __repr__(self):
        return self.value
