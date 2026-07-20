# JazzGraph

Most music recommendation systems ask what other users like you enjoy. [JazzGraph](http://philosofool.github.io/jazz-graph) asks Thelonious Monk.
By modeling the social network relationships among jazz musicians, JazzGraph reveals stylistically similar jazz performances that musicians themselves chose.

Please see the complete write-up at [https://philosofool.github.io/jazz-graph] for details on the project.

## Files

**Source (`src/jazz_graph/`)**
- [model/model.py](src/jazz_graph/model/model.py) — heterogeneous GNN (SAGE/GAT) that produces embeddings for artists, songs, and performances
- [recommendation/recommender.py](src/jazz_graph/recommendation/recommender.py) — cosine- and dot-product similarity search over learned embeddings to produce ranked recommendations
- [recommendation/playlist.py](src/jazz_graph/recommendation/playlist.py) — ingests a user's Spotify listening history and maps tracks to graph nodes
- [data/graph_builder/](src/jazz_graph/data/graph_builder/) — builds the PyG `HeteroData` graph from parquet tables (artists, songs, performances)
- [training/loop.py](src/jazz_graph/training/loop.py) — training loop using PyTorch Ignite with NT-Xent contrastive loss and early stopping
- [training/loss.py](src/jazz_graph/training/loss.py) — NT-Xent loss variants, including a masked multi-positive version
- [etl/](src/jazz_graph/etl/) — extracts performer/recording data from Discogs XML dumps and loads it into the database
- [metrics/](src/jazz_graph/metrics/) — alignment/uniformity embedding metrics and ranking evaluation

**Notebooks (`notebooks/`)**
- [recommendation.ipynb](notebooks/recommendation.ipynb) — end-to-end demo: load embeddings, match a Spotify history, get recommendations
- [modeling.ipynb](notebooks/modeling.ipynb) — model training walkthrough
- [explore.ipynb](notebooks/explore.ipynb) — graph exploration and data analysis

**Database (`queries/`)**
- SQL migrations and views that build the jazz recordings schema from raw Discogs data

**Scripts (`scripts/`)**
- One-off data pipeline scripts: extract Discogs data, build parquet tables, run training experiments
