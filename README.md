HGMG: Enhancing heterogeneous graph embedding by meta-graph learning
This repository contains the implementation of HGMG (Heterogeneous Graph Meta-Graph Model), 
a deep learning framework for heterogeneous graph representation learning and node classification. 
The model integrates multiple meta-graphs, instance-level embeddings, 
and similarity-based graph neural networks to capture complex structural information in heterogeneous graphs.

Overview
Meta-graph filtering based on meta-graph structure：Metagraph_Select.py
Node embedding based on meta-graph context:Node_emb_model.py
Node embedding enhancement based on meta-graph instance:Instance_emb_model.py
Meta-Graph Fusion:Metagraph_fusion
Node embedding based on node similarity：SimNode_emb_model.py

If you want to get the meta graph structure, please first download the original GraMi code from 
[https://github.com/ehab-abdelhamid/GraMi](https://smufang.github.io/code/GraMi%20with%20automorphism.zip).
You need to generate a .lg file from the dataset, and then use the .lg file in the Grami algorithm. You will obtain a .q file.
If you want to obtain the meta graph instances, please first download the Symiso code from https://smufang.github.io/code/symiso.zip,
and use command ./symiso data=dblp.lg query=dblp.q maxfreq=100000000 subgraph=dblp.gdb stats=output.dblp.

Command line arguments
data=<String>
The input graph filename. The file is in the Labeled Graph Format. The graph is treated as undirected, and edge types are not considered at the moment.

query=<String>
The input filename for a list of query metagraphs, in the Metagraph Query Format. These query metagraphs can be mined from the input graph using a modified version of GRAMI.

maxfreq=<Integer>
The maximum number of instances to match, for each query metagraph. The program immediately moves on to the next query after the specified maximum number of instances are found.

subgraph=<String>
The filename to output the metagraph database, which contains a list of processed metagraphs. The file is in the Metagraph Database Format.

stats=<String>
The directory name to output matched instances of each metagraph. Make sure you manually create the the directory before running. 
One instance file per metagraph.
Instance filenames are named according to the ID of each processed metagraph (see subgraph above).
Each line in an instance file representing one instance, containing tab delimited node IDs of the input graph, in the order according to the order of nodes in the processed metagraph (see subgraph above).

Running the IMDB Example: python Run_IMDB.py --epochs 300 --lr 0.01 --hidden 128
