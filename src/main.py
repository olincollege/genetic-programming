from parse_tree import *

F = ["+", "-", "*", "/"]
# F = ["+", "-", "*", "/", "sin", "cos", "ln"]
T = TerminalGenerationRules(
    ["X", "Y"], (-10, 10), ints_only=False, no_random_constants=False
)

V = {"X": 1, "Y": 2}

t = ParseTree.generate_full(F, T, 4)
# u = ParseTree.generate_grow(F, T, 3, 0.5)

print(t)
print(t.evaluate(V))
# print(u)
