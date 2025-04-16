from parse_tree import *

F = ["+", "-", "*", "/"]
# F = ["+", "-", "*", "/", "sin", "cos", "exp", "ln"]
T = ["X", "Y", "1", "2", "3"]

t = ParseTree.generate_full(F, T, 3)
u = ParseTree.generate_grow(F, T, 3, 0.5)

print(t)
print(u)
