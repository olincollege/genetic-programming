# Genetic Programming

Enricco Gemha, Ian Lum, Sally Lee

Final project for Advanced Algorithms, exploring genetic programming

## Background of Algorithm

Genetic programming(GP) applies the principles of biological evolution to
generate computer programs [1]. GP follows the Darwinian survival of the
fittest, as well as utilizing genetic operations like random mutation and
crossover as it evolves over multiple generations.

There are a few variants of genetic programming, each representing the evolved
program in different ways. These include tree-based GP, stack-based GP, and
Linear GP [2]. We implemented tree-based GP, the most common variant, to solve
our problem. In tree-based GP, programs are represented as parse trees which are
evaluated recursively to produce multivariate expressions. The terminal nodes
(or leaf nodes) in a parse tree are variables or constants and the internal
nodes are operators such as addition, multiplication, trigonometry function, and
more. ![Tree based Genetic Programming](docs/img/tree_based_gp.png)

Genetic programming is classically implemented in Lisp due to its prefix
notation cleanly being able to represent parse trees [3]. However, for our
implementation, we used Python, and Swift.

GP is commonly used to solve automatic programming and machine learning problems
[1]. In this project, we use GP to classify species of iris flowers from the
classic [UCI iris dataset](https://www.kaggle.com/datasets/uciml/iris). The
dataset contains the length and width of the sepal and petal for 150 irises,
along with each flower’s species—either setosa, versicolor, or virginica.

Our goal is to evolve mathematical expressions that take the sepal and petal
dimensions as inputs, then output a number that maps to one of three species of
iris. From there, we will perform a parameter sweep on the various parameters of
GP (mutation rate, crossover rate, etc.) to see how each affects the accuracy of
the evolved expression.

## How It Works

### Parse tree implementation

#### Node structure

We wrote our own implementation of parse trees that were able to be randomly
generated from a function set and terminal (leaf node) generation rules.

The internal (non-leaf) nodes randomly select a function from addition,
subtraction, multiplication, division, sine, cosine, and natural logarithm. The
exponential function was also implemented but can easily cause integer overflow
errors, and thus was not used.

The terminal or leaf nodes are randomized given a set of rules, these being:

- literals: A set of literals, either variables like X or Y, or specific numbers
- constants_range: A tuple range from which numbers can be randomly selected
  from
- ints_only: A boolean controlling whether randomly generated constants can have
  fractional values
- no_random_constants: A boolean controlling whether randomly generated
  constants are included at all.
- decimal_places: The number of decimal places for randomly generated constants,
  defaults to 4.

For each terminal node, it has an equal chance of choosing from any of the
literals and randomly generating a constant. For example, if the literals are
["X", "Y"], there is a 1/3 chance of choosing "X", 1/3 chance of choosing "Y",
and 1/3 chance of generating a random constant.

#### Tree generation

![Full Tree Generation](docs/img/full_tree_generation.png)[3]

Parse trees can either be generated with the “full” or “grow” method. With the
“full” method, the resulting parse tree is guaranteed to be a full tree at a
given depth. With the “grow” method, branches have a chance to terminate early,
before the max depth. This chance is given by the terminal_prob argument.

Note: The root node is defined to be at depth 0, with the children of the root
note being depth 1. Thus, the trees in the diagrams above are depth 2 trees.

#### Tree evaluation

To evaluate a parse tree, a dictionary mapping variables to numeric values is
taken as input. Nodes evaluate their values recursively, with the base case
being terminal nodes that simply return their numeric value.
![Tree Evaluation](docs/img/tree_evaluation.png) An example parse tree
evaluation. (1) variables are mapped to their numeric values. (2) the
subtraction node is evaluated as 2-3. (3) the addition node is evaluated as 4 +
(-1). (4) an output value of 3 is calculated.

### GP Algorithm

The genetic programming algorithm performs the following steps:

1. Randomize initial population of parse trees
2. Survive a number of the best performing individuals (champions) to the next
   generation
3. For the rest of the population, select parents and perform crossover to spawn
   offspring
4. The new population of champions and offspring all have a chance to mutate
5. Repeat from #2 for a number of generations

#### Classification

Parse trees classified the three irises by taking sepal length, sepal width,
petal length, petal width and outputting a single number, $`A`$. The value of
$`A`$ is converted to a species via the following inequalities:

$$
\begin{cases}
A < 0.33 & \rightarrow \text{Setosa} \\
0.33 \leq A < 0.66 & \rightarrow \text{Versicolor} \\
0.66 \leq A & \rightarrow \text{Virginica} \\
\end{cases}
$$

The thresholds of 0.33 and 0.66 were arbitrarily defined and other thresholds
would work similarly well. While these thresholds in a sense expect $`A`$ values
from 0-1, there is no constraint limiting output to that range, though programs
that produce outputs around that range will naturally perform better and thus
persist in the survival of the fittest. If thresholds of 50 and 100 were
selected, then the best performing parse trees would be those that produce
output on that magnitude.

#### Fitness function

The fitness of each parse tree was evaluated by having the given tree attempt to
classify all irises in the training portion of the dataset (a random 80% of the
original dataset). Each iris that would add 1 to an individual’s fitness.

#### Initial Population Generation

The following parameters were used for the generation of our initial population:

- Population size: 50
- Function set: addition, subtraction, multiplication, division
- Terminal rules:
  - Literals: sepal length, sepal width, petal length, petal width
  - Constants range: (-10, 10)
  - Integers only: false
  - No random constants: false
- Tree generation method: Grow
- Max depth: 3
- Terminal probability: 0.2

#### Champion survival

The best performing individuals (champions) are selected to survive to the next
generation. The number of champions that are selected is determined by the
variable champion_survival_percentage, which we have set to 0.1.

#### Parent selection

The remainder of the population consists of offspring, generated via crossover.
To select the parents that are crossed over to generate these offspring,
roulette parent selection is used. Any individual from the population can be
selected as a parent, but individuals are weighted based on their fitness,
meaning more fit individuals are more likely to be selected.

#### Crossover

The two parent parse trees are crossed over to generate an offspring. To do
this, a subtree from one parent is randomly chosen to replace the subtree of the
other parent. Note that this can result in a tree that exceeds the max depth.
Crossover is not guaranteed to happen, and is dictated by the variable
crossover_rate, which we have set to 0.9. If crossover doesn’t occur, the
offspring is simply a copy of one of the parents.
![Crossover](docs/img/crossover.png)

#### Mutation

Champions and offspring all have a chance to mutate, dictated by the variable
mutation_rate, which we have set to 0.1. To mutate a parse tree, a subtree on
the parse tree is randomly replaced by a new randomly generated parse tree.
Similarly to crossover, this operation can result in a tree that exceeds the max
depth. ![Mutation](docs/img/mutation.png)

## Results

The genetic programming algorithm was run for 50 generations, using
`random.seed(2)` for reproducibility. The fittest program from this evolution
was the following tree:

![Best Tree](docs/img/best_tree.png)

This parse tree had an accuracy of 93% and the following classification report:

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | :-------: | :----: | :------: | :-----: |
| Iris-setosa      |   1.00    |  1.00  |   1.00   |   10    |
| Iris-versicolor  |   1.00    |  0.80  |   0.89   |   10    |
| Iris-virginica   |   0.83    |  1.00  |   0.91   |   10    |
|                  |           |        |          |         |
| **Accuracy**     |    N/A    |  N/A   |   0.93   |   30    |
| **Macro avg**    |   0.94    |  0.93  |   0.93   |   30    |
| **Weighted avg** |   0.94    |  0.93  |   0.93   |   30    |

To further visualize the how well each species was classified, the results are
plotted in a confusion matrix:

![Confusion Matrix](docs/img/confusion_matrix.png)

Looking at the confusion matrix, we see again that the program properly
classified all setosas and virginicas, but misclassified two versicolors as
virginicas. This error in classifying versicolors makes sense, as they were the
"middle" category of the classification. The GP programs output an number, and
only if that number is between the low and high thresholds is it then classified
as versicolor. Any value below under the low threshold is classified as setosa,
and any value above the high threshold is classified as virginica. This setup
makes it "harder" for virginicas to be classified as they have both an upper and
lower bound.

Further analysis of these incorrectly classified irises is in the _Analysis_
section.

### Parameter Sweep

To understand how each parameter of the genetic programming algorithm impacts
the accuracy of the best program, we performed a parameter sweep. For each set
of unique parameters, the program was run 10 times, and the average accuracy was
taken. Default values were chosen for parameters that were not being swept,
based on standards of literature and former trial and error when developing the
algorithm, trying to balance runtime and accuracy. The default values are:

- `function_set`: ["+", "-", "*", "/"]
- `terminal_rules`:
  - `literals`: ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm",
    "PetalWidthCm"]
  - `constants_range`: (-10, 10)
  - `ints_only`: False
  - `no_random_constants`: False
- `max_depth`: 3
- `terminal_probability`: 0.2
- `population_size`: 50
- `generations`: 20
- `crossover_rate`: 0.9
- `mutation_rate`: 0.1
- `champion_survival_percentage`: 0.1

Note: We had set a rather high mutation rate of 0.1 (10%) for much of the
development as well as for the parameter sweep. However, later research showed
that mutation rate should be around 0.5% to 1% [4]. Rather that re-sweeping with
1% as the default, we chose just to sweep mutation rates from 0 to 10%.

For other parameters, values generally centered around the default value were
chosen to swept. These values are:

- `population_size`: 20, 40, 60, 80, 100
- `generations`: 10, 30, 50, 70
- `crossover_rate`: 0.6, 0.7, 0.8, 0.9, 1.0
- `mutation_rate`: 0, 0.005, 0.01, 0.05, 0.1
- `champion_survival_percentage`: 0.1, 0.2, 0.3, 0.4, 0.5
- `max_depth`: 1, 2, 3, 4

## Analysis

To further analyze which versicolor was predicted as virginica from our result,
we plotted all irises by the dimensions of various parts of the flower. In the
first graph, they are plotted based on the length and width of the petal. In the
second graph, they are plotted based on the length and width of the sepal.
Points are colored based on the species that the best program predicted them to
be. The correct predictions are smaller and transparent, while the incorrect
predictions are larger and opaque.

When looking at petal dimension, we can see that the two versicolors that were
incorrectly classified as virginicas are towards the cluster of other
virginicas. Furthermore, when looking at sepal dimension, we can see that the
two misclassified versicolors are actually within the cluster of virginicas.
![Predicted Species by Petal Dimensions](docs/img/predicted_species_petal.png)
![Predicted Species by Sepal Dimensions](docs/img/predicted_species_sepal.png)

From these results, we were able to make sure that our program is not only
performing well just in terms of the accuracy score but the predictions made
sense even visually.

### Parameter Sweep Analysis

![Sweep Population Size](docs/img/sweep_population_size.png) The first sweep was
on the population size from 20 populations to a hundred, it roughly showed a
stable increase from about 0.85 to about 0.95. From this result, we can see that
the more the population size, the higher the accuracy is and we can assume that
it might even show a further increase as we increase the population size. This
result corresponded to our initial assumption that there would be a better
result if we increased our population size.

![Sweep Generations](docs/img/sweep_generations.png) The second sweep was on the
number of generations from 10 to 70. Overall, we see an increase in accuracy
from approximately 0.84 to approximately 0.95 as the number of generations was
increased. In this case, we see a rapid increase in accuracy from 10 to 30
generations, then a small decrease at 50 generations, then improving again at 70
generations. Although there is the slight dip around 50 generations, the overall
trend is that increasing the number of generations seems to improve model
performance. This finding affirms our original hypothesis that the more
generations used, the more evolutionary optimization could be done and hence the
more accurate.

![Sweep Crossover Rate](docs/img/sweep_crossover_rate.png) The third sweep
examined the effect of modifying the crossover rate from 0.6 to 1.0. There was a
continued improvement in accuracy as the crossover rate increased up to an
accuracy of approximately 0.93 when the crossover rate was set at 0.9. When the
crossover rate was constrained closer to 1.0, however, the accuracy declined all
the way back down to approximately 0.88. This suggests that while increasing the
crossover rate generally increases accuracy, doing crossover all the time could
potentially decrease performance. This result aligns with our intuition and our
initial research that crossover helps exploration of better solutions, but it
also suggests that there is a need to balance crossover and mutation so that
over-exploration and loss of diversity are avoided.

![Sweep Mutation Rate](docs/img/sweep_mutation_rate.png) The fourth sweep
experimented with the effect of altering the mutation rate from 0.0 to 0.5. We
observed that introducing a small amount of mutation (approximately 0.005)
caused a significant boost in accuracy, from approximately 0.81 to nearly 0.90.
But as the mutation rate increased further beyond this, accuracy gradually
declined. This suggests that some level of mutation is helpful to provide
diversity and avoid premature convergence, but a very high level of mutation
disrupts the optimization process and leads to suboptimal performance. These
results confirm our initial research that mutation rate with a significantly
small number (less than 1%) usually gives the best outcome.

![Sweep Champion Survival](docs/img/sweep_champion_survival_percentage.png) The
fifth sweep examined the effect of altering the champion survival percentage
from 0.0 to 0.5. Interestingly, the findings showed some fluctuation rather than
a clear upward or downward trend. Accuracy was very high at 0.0 survival (around
0.88), but decreased when survival percentages of 0.1 and 0.2 were employed.
Later, accuracy was regained and even surpassed the initial level at survival
rates of 0.3 and 0.4, reaching a maximum of approximately 0.90. This suggests
that an acceptable level of champion survival can be used to retain top players
and improve performance, but too little or poorly tuned survival percentages can
be detrimental. Overall, the findings indicate that accurate tuning of champion
survival is necessary for optimal performance.

![Sweep Max Depth](docs/img/sweep_max_depth.png) The sixth and our last sweep
took into account the maximum depth of trees ranging from 1 to 4. As the maximum
depth increased from 1 to 3, the accuracy consistently increased to the peak of
around 0.93 at depth 3. However, with the subsequent increase in the depth to 4,
the accuracy fell dramatically to around 0.87. This trend demonstrates that
allowing for deeper trees early on increases model expressiveness and leads to
better performance, but beyond some point, deeper trees can lead to overfitting
or unnecessary complexity, ultimately hurting generalization. This result aligns
with the common intuition that deeper structures can capture more complex
patterns at the cost of overfitting.

### Multi-Language Implementation Comparison

In order to understand the impact programming language choice has on GP
implementation and performance, we created parallel versions in Python and
Swift. This allowed us to quantitatively measure execution speed differences
between an interpreted language (Python) and a compiled language (Swift), while
exploring each language's different solution architecture. Python offers dynamic
typing and an extensive library ecosystem, for rapid development and testing,
while Swift has a strong typing, value semantics, and protocol-oriented design,
which despite being harder to implement, delivers a more “safe” development
environment.

Accuracy: 0.8333333333333334

Classification Report: precision recall f1-score support

    Iris-setosa       1.00      1.00      1.00         9

Iris-versicolor 0.67 1.00 0.80 10 Iris-virginica 1.00 0.55 0.71 11

       accuracy                           0.83        30
      macro avg       0.89      0.85      0.84        30

weighted avg 0.89 0.83 0.83 30

This comparative implementation made clear how Swift's protocols versus Python's
class inheritance led to slightly different approaches for handling genetic
operations. The high-level structure of each implementation can be seen below:

![Diagram of the GP implementation in Python and Swift](docs/img/diagram_swift_python.png)

Some of the key implementations made in the Swift version were enabling
polymorphic behavior between `FunctionNode` and `TerminalNode`. Another
important point was the development of a custom `copy()` methods to handle
proper tree duplication during the genetic operations. The methods are:

- For `TerminalNode`, creating a new node with the same value
- For `FunctionNode`, recursively copying each child node first, then creating a
  new function node with those copies
- For `ParseTree`, copying the root node and creating a new tree with it

The Swift implementation also tackles the multi-class Iris classification
challenge using a "one-vs-all" approach. Rather than evolving a single
expression tree to differentiate between all three species simultaneously, the
system trains separate classifier trees for each Iris species (setosa,
versicolor, and virginica). Each tree is evolved to output a high confidence
value when presented with its target class and a low value otherwise. During
classification, all three trees evaluate the same input features, generating
confidence scores that indicate how strongly each classifier believes the sample
belongs to its respective class. The final classification is determined by
selecting the class corresponding to the tree that produced the highest
confidence score, effectively letting the classifiers "vote" on the most likely
species. This approach simplifies the evolutionary process by transforming a
complex three-way classification problem into three more manageable binary
classification tasks.

#### Results

Our comparative implementation of genetic programming in Python and Swift showed
many interesting findings.

#### Performance Benchmarks

We measured the execution time and classification accuracy across different
population sizes and generation counts for both language implementations:

![Genetic Programming Runtime Comparison](docs/img/gp_runtime_comparison.png)

**Runtime Analysis:**

- **Python** has significantly faster execution times, running approximately
  2-5x faster than Swift
- **Swift** shows more scaling issues as population size and generation count
  increase
- Both implementations show linear scaling with population size, but Swift's
  slope is steeper
- With generations, Swift has a near-quadratic behavior while Python shows more
  linear growth

![Genetic Programming Accuracy Comparison](docs/img/gp_accuracy_comparison.png)

**Accuracy Analysis:**

- **Swift** achieves consistently higher classification accuracy, particularly
  at larger population sizes
- The sweet spot for Swift appears to be around 100 population size and 10
  generations, achieving nearly 80% accuracy
- Python's accuracy performance is more erratic, with significant variability
  between runs
- Important to note that increasing generations doesn't consistently improve
  accuracy in either implementation

#### Implementation Efficiency Tradeoffs

The benchmark results reveal a tradeoff in GP:

1. **Development vs. Runtime Efficiency:**

   - Python's flexible typing makes it easier to get the code done and delivers
     quick runtimes
   - Swift's strict typing and protocol-orientation required more upfront effort
     and has a worse runtime performance

2. **Accuracy vs. Speed:**

   - Swift's implementation achieved higher accuracy at the cost of longer
     execution times
   - Python traded accuracy for significantly faster execution

3. **Parameter Sensitivity:**
   - Swift showed better performance with increased population diversity and
     size
   - Python performed better with shorter generation runs, which might point out
     to overfitting

The optimal configuration appears to be language-dependent, with Swift
benefiting from larger populations but fewer generations, while Python performs
adequately with smaller populations and delivers quick runtimes.

## Next Steps

## Bibliography

[1] W. Banzhaf, “Artificial Intelligence: Genetic Programming,” International
Encyclopedia of the Social &amp; Behavioral Sciences, pp. 789–792, 2001.
doi:10.1016/b0-08-043076-7/00557-x

[2] K. Staats, “About GP,” Genetic Programming, https://geneticprogramming.com/
(accessed Apr. 28, 2025).

[3] W. B. Langdon, R. Poli, N. F. McPhee, and J. R. Koza, “Genetic programming:
An introduction and tutorial, with a survey of techniques and applications,”
Studies in Computational Intelligence, pp. 927–1028, 2008.
doi:10.1007/978-3-540-78293-3_22

[4] M. Obitko, “XIII. Recommendations,” Recommendations - Introduction to
Genetic Algorithms,
https://www.obitko.com/tutorials/genetic-algorithms/recommendations.php
(accessed Apr. 28, 2025).
