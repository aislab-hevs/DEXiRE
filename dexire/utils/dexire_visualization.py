import graphviz as gp
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any
from .probabilistic_ranking import probabilistic_ranking

def generate_nn_path_graph(model: tf.keras.Model, 
                           act_path: Dict[Any], 
                           format: str='pdf') -> gp.Digraph:
    """Generate visualization of the activation path in the neurons.

    :param model: Model to explain the activations.
    :type model: tf.keras.Model
    :param act_path: Dictionary with the activation path and a metric per layer.
    :type act_path: Dict[Any]
    :param format: Output format, defaults to 'pdf'
    :type format: str, optional
    :return: Graphviz object with the activation path in the model.
    :rtype: gp.Digraph
    """
    layers = model.layers
    out_layer_idx = len(layers) - 1
    layer_nodes = {}
    important_nodes = {}
    # Create graph
    g = gp.Digraph(format=format)
    g.attr(rankdir="LR")
    g.attr(splines="line")
    g.attr(nodesep=".05")
    g.attr(ranksep="1.0")
    #g.attr("node", label="")
    # create subgraphs
    preffix = ''
    for idx, layer in enumerate(layers):
        # Check if the idx layer is on the activation path
        if idx not in act_path.keys(): continue
        important_nodes[idx] = []
        if idx == 0:
            preffix = 'x'
            c = gp.Digraph(name="in_layer", node_attr={"style":"solid","color":"blue4", "shape":"circle"})
            c.attr(color="green")
            c.attr(label="Input layer")
        elif idx == out_layer_idx:
            preffix = 'y'
            c = gp.Digraph(name="out_layer", node_attr={"style":"solid","color":"blue4", "shape":"circle"})
            c.attr(color="white")
            c.attr(label="Output layer")
            # sort 
            sl = probabilistic_ranking(act_path[f'layer_{idx-1}'], key='value')
            #print(sl)
            id_node = sl[0][0].split('_')[-1]
            val_node = sl[0][1]['value']
        else:
            preffix = f'h_{idx}'
            c = gp.Digraph(name=f"h_{idx-1}", node_attr={"style":"solid","color":"blue4", "shape":"circle"})
            c.attr(color="white")
            c.attr(label=f"hidden layer {idx-1}")
            # sort 
            sl = probabilistic_ranking(act_path[f'layer_{idx-1}'])
            id_node = sl[0][0].split('_')[-1]
            val_node = sl[0][1]['value']
        # add nodes 
        layer_nodes[idx] = []
        for i in range(layers[idx].output.shape[-1]):
            if idx != 0 and str(i) == id_node:
                c.node(f"{preffix}_{i}", color='red', label=f'{preffix}_{i}={val_node}')
                important_nodes[idx].append(f"{preffix}_{i}")
            else:
                c.node(f"{preffix}_{i}")
            layer_nodes[idx].append(f"{preffix}_{i}")
        # add subgraph
        g.subgraph(c)
    # add connections 
    #print(important_nodes)
    for i in range(len(layers)-1):
        for n1 in layer_nodes[i]:
            for n2 in layer_nodes[i+1]:
                if i==0 and n2 in important_nodes[i+1]:
                    g.edge(n1, n2, color='red')
                elif n2 in important_nodes[i+1] and n1 in important_nodes[i]:
                    g.edge(n1, n2, color='red')
                else:
                    g.edge(n1, n2, color='grey')
    return g