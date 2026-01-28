GNN Resources. Thanks to Claude AI.

## **Hands-On Tutorials & Code**

**PyTorch Geometric (PyG) - Official Tutorials**
- [Introduction to GNNs](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
- [Heterogeneous Graph Learning](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html) - Critical for your project
- [Link Prediction on MovieLens](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_link_pred.py) - Recommendation example
- Their [examples directory](https://github.com/pyg-team/pytorch_geometric/tree/master/examples) has many recommendation implementations

**Hands-On Recommendation Tutorials**
- [Building a Recommendation System with GNNs (PyG)](https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377) - Stanford CS224W blog
- [Graph Neural Networks for Recommendation (Colab)](https://colab.research.google.com/drive/1N3LvAO0AXV4kBPbTMX866OwJM9YS6Ji2) - Interactive notebook
- [DGL Tutorial: Recommendation with Heterogeneous Graphs](https://docs.dgl.ai/en/latest/tutorials/blitz/4_link_predict.html) - Alternative library to PyG

**Deep Graph Library (DGL) Resources**
- [User-Item Recommendation Tutorial](https://docs.dgl.ai/en/latest/guide/training-link.html)
- [Amazon Product Recommendation Example](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero)

## **Academic Papers**

**Foundational GNN Papers**
- [Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling, 2017)](https://arxiv.org/abs/1609.02907) - The GCN paper
- [Graph Attention Networks (Veličković et al., 2018)](https://arxiv.org/abs/1710.10903) - GAT
- [Inductive Representation Learning on Large Graphs (Hamilton et al., 2017)](https://arxiv.org/abs/1706.02216) - GraphSAGE

**GNN for Recommendation (Key Papers)**
- [Neural Graph Collaborative Filtering (Wang et al., 2019)](https://arxiv.org/abs/1905.08108) - NGCF, highly cited
- [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (He et al., 2020)](https://arxiv.org/abs/2002.02126) - Simplified, effective
- [Graph Convolutional Matrix Completion (Berg et al., 2017)](https://arxiv.org/abs/1706.02263) - Early work
- [PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems (Ying et al., 2018)](https://arxiv.org/abs/1806.01973) - Pinterest's production system

**Heterogeneous Graphs (Important for Your Project)**
- [Heterogeneous Graph Transformer (Hu et al., 2020)](https://arxiv.org/abs/2003.01332) - HGT
- [Relational Graph Convolutional Networks (Schlichtkrull et al., 2018)](https://arxiv.org/abs/1703.06103) - R-GCN, handles multiple edge types

**Music Recommendation Specific**
- [Graph Neural Networks for Music Recommendation (Oramas et al., 2018)](https://arxiv.org/abs/1809.07276)
- [Collaborative Filtering with Graph Information (Monti et al., 2017)](https://arxiv.org/abs/1706.07684)

**Survey Papers**
- [Wu et al. (2021) - Comprehensive Survey of GNNs](https://arxiv.org/abs/1901.00596) - You already have this
- [Graph Neural Networks for Recommender Systems (Gao et al., 2022)](https://arxiv.org/abs/2011.02260) - Recent survey focused on recommendation

## **Blog Posts & Articles**

**Conceptual Introductions**
- [Intro to Graph Neural Networks (Distill.pub)](https://distill.pub/2021/gnn-intro/) - Beautiful interactive visualizations
- [Understanding Graph Convolutional Networks (Towards Data Science)](https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b)
- [Graph Neural Networks: A Review of Methods (Medium)](https://medium.com/dair-ai/an-illustrated-guide-to-graph-neural-networks-d5564a551783)

**Recommendation-Focused**
- [Graph Neural Networks for Recommendation Systems (Neptune.ai)](https://neptune.ai/blog/graph-neural-network-recommendation-systems)
- [Building a Music Recommender with GNNs (Towards Data Science)](https://towardsdatascience.com/how-to-build-a-music-recommendation-system-using-graph-neural-networks-5c4c3e19e5b6)
- [Pinterest's PinSage Engineering Blog](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)

## **Video Lectures**

**Stanford CS224W (You mentioned this)**
- [Lecture 14: Traditional Graph ML](https://www.youtube.com/watch?v=7JELX6DiUxQ&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=14)
- [Lecture 15: Graph Neural Networks](https://www.youtube.com/watch?v=JAB_plj2rbA&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=15)
- [Lecture 16: Deep Generative Models for Graphs](https://www.youtube.com/watch?v=eybCCtNKwzA&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=16)

**Other Video Resources**
- [Graph Neural Networks (Microsoft Research)](https://www.youtube.com/watch?v=cWIeTMklzNg) - Petar Veličković
- [MIT 6.S191: Graph Neural Networks](https://www.youtube.com/watch?v=GXhBEj1ZtE8)

## **Books**

**Free Online**
- [Graph Representation Learning (Hamilton, 2020)](https://www.cs.mcgill.ca/~wlh/grl_book/) - Comprehensive, free PDF
- [Geometric Deep Learning (Bronstein et al., 2021)](https://geometricdeeplearning.com/) - Theoretical foundations

**Textbooks**
- *Understanding Deep Learning* (Prince, 2023) - Chapter 12 as you mentioned
- *Deep Learning on Graphs* (Ma & Tang, 2021) - Specifically focused on graph DL

## **Datasets for Practice**

**Music/Collaboration Datasets**
- **MusicBrainz** - What you're using
- **Last.fm** - User-artist-tag network
- **Million Song Dataset** - Collaborative filtering data
- **Spotify API** - Can build your own graphs

**Standard GNN Benchmarks**
- **Cora/CiteSeer/PubMed** - Citation networks (node classification)
- **MovieLens** - User-movie ratings (recommendation)
- **Amazon Product Co-Purchase** - Product network
- **OGB (Open Graph Benchmark)** - [ogb.stanford.edu](https://ogb.stanford.edu/) - Standard benchmarks including recommendation

## **Specific to Your Jazz Project**

**Collaborative Filtering + Graphs**
- [Collaborative Filtering via Graph Neural Networks (Chen et al., 2020)](https://arxiv.org/abs/2001.07614)
- [Social Collaborative Filtering (Ma et al., 2019)](https://arxiv.org/abs/1902.09362) - Uses social network structure

**Temporal Graphs (Jazz evolves over time)**
- [Temporal Graph Networks (Rossi et al., 2020)](https://arxiv.org/abs/2006.10637)
- [DyRep: Learning Representations over Dynamic Graphs (Trivedi et al., 2019)](https://openreview.net/forum?id=HyePrhR5KX)

## **Implementation Libraries**

- **PyTorch Geometric** - [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/)
- **DGL (Deep Graph Library)** - [dgl.ai](https://www.dgl.ai/)
- **Spektral** (Keras-based) - [graphneural.network](https://graphneural.network/)
- **StellarGraph** - [stellargraph.readthedocs.io](https://stellargraph.readthedocs.io/)

## **Recommended Learning Path**

**Week 1-2: Foundations**
1. Distill.pub GNN intro (visual understanding)
2. Stanford CS224W Lectures 14-15
3. PyG Introduction tutorial

**Week 3-4: Implementation**
1. PyG heterogeneous graph tutorial
2. MovieLens recommendation example
3. Implement on small dataset (Cora or MovieLens)

**Week 5-6: Deep Dive on Recommendation**
1. Read LightGCN paper
2. Neural Graph Collaborative Filtering paper
3. Implement basic collaborative filtering with GNN

**Week 7-8: Your Project**
1. Map MusicBrainz to graph structure
2. Build PyG dataset loader
3. Adapt recommendation architecture to jazz collaboration network

## **Key Papers for Your Proposal**

If you want to cite 2-3 papers in your email to show you've done homework:
1. **LightGCN (He et al., 2020)** - Simple, effective, widely adopted
2. **Neural Graph Collaborative Filtering (Wang et al., 2019)** - Foundational work
3. **PinSage (Ying et al., 2018)** - Real-world production system

These show you understand the field and have realistic technical grounding.
