import graphviz

def visualize_graph(output_node, filename="computational_graph"):
    """
    Visualizes the computational graph using graphviz.

    Args:
        output_node: The final Var node in the graph (e.g., the loss).
        filename: The name of the file to save the graph to (without extension).
    """
    dot = graphviz.Digraph(comment='Computational Graph')

    visited = set()

    def build_graph(node):
        if node in visited:
            return
        visited.add(node)

        # Add a node for the current Var
        node_label = f"Var\nshape={node.shape}\nrequires_grad={node.requires_grad}"
        if node.arr.size <= 10: #Show value for small arrays
            node_label += f"\nvalue={node.arr}"

        dot.node(str(id(node)), label=node_label)

        # Recursively add edges and nodes for inputs
        for input_node, _ in node.parents:
            # Add an edge from the input to the current node
            dot.edge(str(id(input_node)), str(id(node)))
            build_graph(input_node) #Recursive call

    build_graph(output_node)
    dot.render(filename, format='png') #Save and create png
    print(f"Computational graph saved to {filename}.png")
