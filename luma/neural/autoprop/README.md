
<img src="https://raw.githubusercontent.com/ChanLumerico/luma/main/img/title/ap_dark.png" alt="logo" height="50%" width="50%">

Auto propagation system for complex neural networks of Luma Python library

## What is AutoProp?

AutoProp is a specialized package designed for automated dynamic graph construction within `Luma`'s neural components. 

It facilitates seamless automatic feed-forward and backpropagation processes, streamlining neural network operations and enhancing flexibility in model building.

## Why is Automated Propagation Needed?

- **Dynamic Graph Construction**: Automatically builds the computational graph, allowing flexible and non-linear architectures without manual intervention.
- **Automatic Feed-forwarding**: Ensures correct data flow between layers, handling dependencies and the order of operations during forward passes.
- **Backpropagation & Gradient Calculation**: Automates gradient computation using the chain rule, ensuring efficient and accurate backpropagation through the graph.
- **Error Handling & Optimization**: Detects errors early and applies optimizations (e.g., gradient clipping, batch normalization) automatically, reducing manual adjustments.

---

## How to Use `LayerNode`

`LayerNode` is an abstraction representing nodes in a neural computational graph, encapsulating individual neural components such as layers or blocks.

### 1️⃣ Insert a Neural Component

`LayerNode` has three arguments:

- **layer**: `LayerLike`
- **merge_mode**: `MergeMode`, default=*MergeMode.SUM*
- **name**: `str`, optional, default=*None*

```python
node = LayerNode(
    layer=Conv2D(3, 6, 3),
    ...,
)
```

Pass `LayerLike` instance to the argument `layer` to encapsulate the neural component as a new neural node. 

### 2️⃣ Set Merge Mode

A neural node can have multiple incoming branches from previous nodes, requiring it to resolve how to merge the incoming tensors during the propagation process. 

This involves selecting or applying a specific merging strategy, such as concatenation, addition, or averaging, to combine the inputs into a single tensor before passing it through the node for further computation.

For this purpose, `AutoProp` has a dedicated class `MergeMode` that can specify the merging strategy.

**Enum Class `MergeMode`**

| Mode | Operation |
| --- | --- |
| `SUM` *(Default)* | Sum the tensors |
| `CHCAT` | Concatenate over the channels of the tensors |
| `HADAMARD` | Perform Hadamard(element-wise) product over the tensors |
| `AVG` | Perform Element-wise averaging over the tensors |
| `MAX` | Select the maximum elements among those of the tensors |
| `MIN` | Select the minimum elements among those of the tensors |
| `DOT` | Perform dot products over the tensors sequentially |
| `SUB` | Subtract the tensors |

```python
node = LayerNode(
    layer=Conv2D(3, 6, 3),
    merge_mode=MergeMode.CHCAT,
    ...,
)
```

### 3️⃣ Set Nickname

`LayerNode` allows setting a nickname for each node to distinguish it from other potentially existing nodes. 

This feature helps in identifying and referencing specific nodes within a neural computational graph, especially in complex architectures where multiple nodes of the same type or function may exist.

```python
node = LayerNode(
    layer=Conv2D(3, 6, 3),
    merge_mode=MergeMode.CHCAT,
    name="my_node",
)
```

If not set, ‘LayerNode’ is automatically assigned as a nickname in default.

---

## How to use `LayerGraph`

`LayerGraph` is a graph representation of neural nodes, providing support for dynamic structural modifications. 

It facilitates automatic feed-forward and backpropagation processes, making it a central component of the `AutoProp` system. 

This allows flexible, adaptive network design and efficient computational graph management.

### 1️⃣ Instantiation

`LayerGraph` has three arguments:

- **graph**: `Dict[LayerNode, List[LayerNode]]`
- **root**: `LayerNode`
- **term**: `LayerNode`

```python
A = LayerNode(..., name="A")
B = LayerNode(..., name="B")
C = LayerNode(..., name="C")
D = LayerNode(..., name="D")

module = LayerGraph(
    graph={
        A: [B, C],
        B: [D],
        C: [D],
    },
    root=A,
    term=D,
)
```

The argument `graph` requires a dictionary of `LayerNode`, formatted as an adjacency list. 

In this structure, each dictionary key represents a node, and its corresponding value contains a list of nodes that follow it. 

Additionally, when constructing the graph, you must specify the root node and the terminal node using the arguments `root` and `term`, respectively.

### 2️⃣ Build the Graph

Call the `build` method to finalize the graph using the defined nodes and the specified root and terminal nodes.

```python
module.build()
```

### 3️⃣ Perform Forward and Backward Propagations

Use the `forward` and `backward` methods to perform feed-forwarding and backpropagation, respectively, on the neural graph you’ve constructed.

## Examples

**Linearly Connected Graph**

Structure:

```
Conv2D --> BatchNorm2D --> ReLU
```

Implementation:

```python
conv = LayerNode(Conv2D(3, 6, 3))
bn = LayerNode(BatchNorm2D(6))
relu = LayerNode(Activation.ReLU())

module = LayerGraph(
    graph={conv: [bn], bn: [relu]},
    root=conv,
    term=relu,
)
module.build()
```

**Shortcut Connection**

Structure:

```
Root -+-> Conv2D --> ReLU -+-> Term
      |                    |
      +--------------------+
```

Implementation:

```python
root = LayerNode(Identity())
conv = LayerNode(Conv2D(3, 6, 3))
relu = LayerNode(Activation.ReLU())
term = LayerNode(
    Identity(), MergeMode.SUM,
)

module = LayerGraph(
    graph={
        conv: [relu, term], 
        relu: [term],
    },
    root=conv,
    term=term,
)
module.build()
```

**Channel Concatenation**

Structure:

```
      +-> Conv2D --> ReLU -+
      |                    |
Root -+-> Conv2D --> ReLU -+-> Term
      |                    |
      +-> Conv2D --> ReLU -+
```

Implementation:

```python
root = LayerNode(Identity())
branch_1 = LayerNode(
    Sequential(
        Conv2D(3, 6, 3),
        Activation.ReLU(),
    ),
)
branch_2 = LayerNode(
    Sequential(
        Conv2D(3, 12, 3),
        Activation.ReLU(),
    ),
)
branch_3 = LayerNode(
    Sequential(
        Conv2D(3, 18, 3),
        Activation.ReLU(),
    ),
)
term = LayerNode(
    Identity(),
    MergeMode.CHCAT,
)

module = LayerGraph(
    graph={
        root: [branch_1, branch_2, branch_3],
        branch_1: [term],
        branch_2: [term],
        branch_3: [term],
    },
    root=root,
    term=term,
)
module.build()
```

---

## Summary

`AutoProp` is a core component of the Luma framework, designed to automate dynamic graph construction for neural network components. 

It simplifies model building by managing the creation and modification of neural graphs, supporting automatic feed-forwarding and backpropagation. 

Key features include:

- **LayerNode**: Represents individual neural layers or blocks in the graph.
- **LayerGraph**: Manages the neural graph structure, allowing dynamic changes and defining node relationships through adjacency lists.
- **Graph Construction**: Requires setting `root` and `term` nodes, with the `build` method finalizing the structure.
- **Automated Propagation**: Handles feed-forward and backpropagation with `forward` and `backward` methods, ensuring efficient data flow and gradient calculation.
