# Jazz Graph

TODO: REmove these notes about what to add.
1. Visualize the graph in places.
2. Flow chart (System overview)
3. Results table.
4. Data tables (nodes etc.)

## System Architecture

![System Diagram](/documents/system_diagram.png)

The JazzGraph system includes a data pipeline (left), machine learning system (middle) and a recommendation system (right side). The ML system provides embeddings representing the similarity of jazz musical performances. The recommender uses these embeddings to generate recommendations.

## Jazz Collaboration for Recommendation

This project leverages collaboration with in jazz musical works to create a recommendation system. The main goals of the project are (1) to establish a large corpus of data about jazz music and collect that data in a graph format, (2) to develop a graph neural network that learns representation of jazz performances and (3) to evaluate a recommendation system which uses those representations to make recommendations.

Jazz music is highly collaborative. Jazz musicians collaborate directly when performing songs. Indirectly, jazz musicians play songs written by other jazz musicians. It is not an exaggeration to say that the most famous musicians in jazz from the 1950s and 1960s all played together at one time or another. It is fair to say the one could learn about new jazz simply starting with "Kind of Blue," the most famous and best selling jazz album of all time, and pick any artist on it to find another album of jazz. It is also not an exaggeration to say the most famous jazz songs have all been played by the most famous jazz musicians, often several times. Jazz performances involve rearrangement, improvisation, and substantial novelty as well. Many famous jazz performances, such as John Coltrane's "My Favorite Things" are stylistic reimaginings of the originals.

Jazz thus forms an elaborate social network and readily represented as a graph. Analysis and modeling of graphs is an active area of computer science, mathematics and data science.

The aim to produce a recommender system based on jazz collaboration differs form many recommender system approaches in some ways.
1. The system uses features of the subject matter to produce representations of similarity that may be used for recommendation.
1. Music and other entertainment content, in particular, have often relied on systems of aggregating knowledge and preference from human users to make recommendations. For example, collaborative filtering is often a matrix factorization problem where user-item interactions are the modeled data. In such a context, user-item interaction are either (1) a proxy for the domain features or (2) considered the primary target of representation. With 1, knowledge about musical similarity is assumed to be captured in user-item interactions. With 2, musical similarity is beside the point--the goal is learning something which represents something else, such as probabilities of user engagement. (We do not wish to overstate the reliance on user-item interactions. It is well known that large entertainment services incorporate diverse sources of data in their recommender systems.)

JazzGraph takes a slightly different approach. It assumes that the subject domain contains sufficiently rich information that similarity can be encoded by learning from the domain's structure alone. In short, it assumes that a graph of artists, songs and performances is sufficiently rich that it is possible to learn representation of the domain sufficient to generate informed recommendations. In this regard, it is arguably more like human cognition regarding musical recommendation than collaborative filtering: human experts know the music and recommend music based on musical features; they don't (or many would be ashamed to admit to) making recommendations simply based on popularity and other's taste.
(Again, we should not exaggerate the point: knowledge of who plays with whom is not the same as knowing a song. Nevertheless, it is knowledge of the subject domain, rather than (for example) user preferences. Another perspective on the present work is as an investigation of whether domain knowledge encoded in user preferences aligns with domain knowledge encoded in collaboration.)

## Data

### Data Model

We model jazz collaboration as a heterogenous graph. Within the graph, there are three primary entity types:
- Artists: people, primarily musicians who play songs. Additionally, composers of songs are included in the graph even when they are not also performers. Examples include Duke Ellington and John Coltrane.
- Songs: particular musical compositions that artists perform. Examples include "Take the A Train" and "My Favorite Things."
- Performances: events where musicians (or a single musician) play a song. Live performances and studio recording sessions are both performances.

(While our data about performances is captured in the published recordings of performances, it is important to distinguish a performance from a release. A release is a particular publication of a performance. A recording of a performance may be released and re-released numerous times, as, for example, when many record labels re-released their recordings on CD in the 1980s and 1990s.)

There are three relations that characterize the data and connect the node types:
- artist -*performs*-> performance: In jazz, this usually means playing an instrument as part of the performance. Which instrument is played is a feature of the performance, when that information is include. (That is, it's an edge feature.)
- performance -*performing*-> song: A performance is usually of some musical composition and this reflects which song is played in a performance.
- artist -*composed*-> song: Usually a song has a composer, the person or people who initially composed the music.

The following image displays some connectedness typical of famous jazz albums in the same era. It depicts four famous jazz albums, "Speak No Evil", "Sunday at the Village Vanguard," "Inner Urge" and "The Blues and the Abstract Truth."


TODO: clean this up; fewer artist names, maybe add some song names.
![data_model](/documents/poster_graph.png)

### Data Sources

To complete the project, it was necessary to extract the data about jazz collaborations from publicly available sources.

Musicbrainz is a large public SQL database of musical recordings. It contains detailed information, including songs and performers, for many of the recordings in the catalog. It lacks two keys components needed for this project. A concept of a master recording or release that organizes re-released recordings under a single parent. (As an example, recordings released on CD in the 1980s and a vinyl release of the same from the 1950s to not share a single parent id.) Additionally, Musicbrainz does not have style or genre information which isolates jazz recordings from other genres. Musicbrainz is a publicly maintained database and therefore contains inconsistencies that result from imperfect governance of the data.

Discogs is makes available XML files of its data. There are two XML files in Discogs of interest for this project. First, the releases table in Discogs includes the a parent object which duplicate releases share. Second, it contains data on releases themselves, including artist, track listings, and genres. Discogs does not have the two major weaknesses of Musicbrainz.  However, the data is not organized into relational tables. Thus, performers associated with a recording are not linked to a table with unique entries for each artist; instead, a release will indicate who played on it with string data. This means that it is difficult to reliably build performance edges for the graph.

To build the graph of jazz collaboration, we thus need to merge these two data sources by fuzzy matching on artist and title strings. In summary, this means identifying Discogs jazz master recordings and matching the earliest release in Musicbrainz with the same album title and artist name. This allows one to construct a table of jazz recordings in Musicbrainz and then use that database to construct the performing and performs edges.

### Process

The process for combining these two datasets involved two key steps. First, it is necessary to identify a first release for each recording id. A recording id in Musicbrainz corresponds to a single distinct recorded performance which may appear on multiple different releases, including compilation and releases on different media. A release is typically an album in a medium and thus we can create a table of first releases by selecting distinct recording ids ordered by album. This table includes all recording ids (Jazz or not) within Musicbrainz.

Once this table has been created, we match entries in it by artist name and album name in the discogs master releases table, using only those discogs releases classified as Jazz. Discogs genre is a multilabel classification. There are numerous recordings classified as belonging to multiple genres and including Jazz. Some releases in Discogs are multilabel because they are compilation releases akin to "Great Recordings of the 1950s" which may span several genres of music; in such cases, Discogs appears to apply a label it applies to any track on the release. In other cases, multilabeling seems to be a matter of music in multiple genres. It is surprisingly common, for example, that a recording is both "Electronic Music" and "Jazz." For simplicity, I used only Discogs releases with a single positive label "Jazz."

The matching between the two sources raises a number of common problems with string matching. Variations such as curly and straight quotes, capitalization, roman numerals and abbreviations represent some differences which matching the strings. An additional variation occurs because metadata relating to the recording is sometimes included in the title. (The normalization procedure here was also used downstream to match recordings in Spotify, where the addition of metadata in titles is especially common.) Among the issues of metadata, we find parentheticals like "(Legacy Edition)" "(5.0 Miz)" and "(Rudy van Gelder Edition)" which are basically meaningless to the song identity and not replicated across entries. We also find things like "(Live at the Village Vanguard)" or "(Take 2)" which indicate something about the performance and may distinguish from another version in the same release. Finally, we find alternative titles such as "Manha de Carnival (Black Orpheus)" or "Corcovado (Quiet Night of Quite Stars)" where an alternate title or chorus lyrics are included. Several iterations of normalization strategy were worked out to handle various cases. For example, John Coltrane's album "A Love Supreme" variously include abbreviations, Roman/Arabic numerals and often includes "A Love Supreme: ..." in the title of each song.

Well before approaching a final version of the dataset, I added guardrails to prevent modifying the dataset with an unclean Git tree. This means that the state of normalization when data was update can be seen by looking at the commit when the data was updated. The exact details of normalization are best left to the code itself. TODO: provide link or file name of the normalization functions.

After matching all recording ids in Musicbrainz corresponding to a jazz release in discogs, I create a one-many join table in the Musicbrainz database from recording_id to discogs_id. An inner join (or suitable left/right join) on this table assures than only entries that were matched is included in query results, providing a convenient way to filter any Musicbrainz query to jazz releases.

Because Musicbrainz is heavily normalized, it is difficult to describe in full detail the steps needed to join, e.g., performers with recordings, in Musicbrains. Thus, I describe the conceptual structure of join below. The files in `queries` represent sequences of updates to the database, which create views of the data for convenience in the steps described below. The queries to write the necessary graph entities are in `scripts/create_parquet_tables.py`.

A relational table structure is a type of graph; foreign keys describe an edge between the primary entities in two tables. For example, if we join a recordings table with a performers table where recordings.recording_id is performers.recording_id, we have an edge that defines an "artist -performs-> recording" edge. We can leverage this to produce three tables of nodes:
1. recording_ids joined to discogs creates the performance nodes;
1. Linking the jazz recordings with their songs and deduplicating creates a table of song nodes which includes the composer.
1. joining the performance node table with a performance to performer edge provides a table of artists; I union this table with a the composers of the song table. This gives an artist table (including both performers and composers.) Note that the union operation is deduplicating.

We have several edge tables that map the id fields in the above tables in a many-many relation and this all exists within Musicbrainz:
1. The artist-performance table links artist_id in artist with recording (performance) id in performances.
1. The performance-song table links the song_id and performance_id, corresponding to which song is performed.
1. The song-artist table links artist_id (composers) with song_id.
Note that edge tables values are the indexes of corresponding node's row in each table, not the ids themselves.

I discuss the specific features with each node in the training section below.

### Summarize

The above tables were written to parquet files and each file can be extracted with limited modification into a Pytorch Geometric heterogenous dataset. After extracting the data, I removed any isolated nodes, which correspond to performances without performer or song edges. I treated the graph as undirected.

|   |  Node Count  |
|:--|-------------:|
|Artist      | 24,470  |
|Performance | 130,695 |
|Song        | 31,828  |

Summary of node degrees by edge type:

|       |    composed |   performs |    performing |   rev_composed |   rev_performs |   rev_performing |
|:------|------------:|-----------:|--------------:|---------------:|---------------:|-----------------:|
| mean  |     1.48    |    21.11   |      0.66     |       1.13     |        3.95    |          2.72    |
| min   |     0       |     0      |      0        |       1        |        0       |          1       |
| 50%   |     0       |     8      |      1        |       1        |        3       |          1       |
| 75%   |     1       |    16      |      1        |       1        |        5       |          1       |
| 90%   |     3       |    44      |      1        |       1        |        8       |          4       |
| 97%   |    11       |   126      |      1        |       3        |       17       |         14       |
| max   |   288       |  1788      |     13        |       9        |       81       |        321       |

As a helpful guide to the meaning in this table, the max row indicates, from left to right,
some artist composed 288 songs in the data,
some performer was part of 1788 performances,
some performance (either a medley, or perhaps a data issue) combined 13 songs,
some song had 9 different composers involved,
some performance include 81 musicians,
and some song was performed in 321 different performances in the data.
The percentiles give a sense of the overall connectedness within each of these categories.

### "Issues"

Maybe not the best term, but acknowledge data quality in source material, effects of fuzzy matching and other matters of the overall graph quality.

i. Potential for duplicate records (nodes).
ii. Potential for missing records (missing nodes, missing edges).
iii. Potential for incomplete information (missing edges, incomplete node attributes).
iv. Combinations of effects: where a node is duplicated, a portion of relevant edges may direct to either of the nodes.

## Modeling

### GNN Approaches and Challenges


There are a number of models for GNN learning. The basic approach of all GNNs is to characterize graph nodes with their feature information; the feature information can be features from the data, learned features (i.e. embeddings) or a combination of these. The process of learning about the graph involves learning a representation of a node with its neighborhood. For example, the GraphSAGE algorithm learns a representation $h_1$ of a nodes neighborhood by transforming the feature representation of a node and it's neighbors with a learned transformation and then aggregating these to a representation of the node's neighborhood. Through additional layers, a representation of a node, its neighbors, and it's neighbors' neighbors can be learned. Roughly:
$$h_{x1} = agg(W_1.x_i)$$
where $W_1$ is the learned weight matrix, $N_x$ is the set of nodes that are neighbors of $x$, and $agg$ is an aggregation function, such as summation. In more sophisticated models, the aggregation can be informed by features of edges, which (of course) determine which nodes are connected to $x_i$.

The goal of the Graph Neural Network (GNN) is to learn representations of musical performances from the graph of collaboration.
The representations will be used in the downstream recommendation task. Thus, it is necessary to find a task for learning representations which generates a useful concept of similarity. Following standard deep learning practice, all the approaches for this I considered involved learning an embedding to capture the latent space which characterizes the features for learning.

There are many options for such a task, though the available data limits what might be done. Task are naturally divided into two classes, supervised and unsupervised (or self-supervised) models.

One feature available for supervised learning is the jazz substyle information in the graph. Each performance has a multilabel feature corresponding to the jazz sub-style(s) it exemplifies. One advantage of using this data is that music style presumably directly correlates to listener preferences, so that a listen who likes a lot of bebop would presumably like other songs which are also bepop. With this in mind, the first version of the model that I built learned embeddings to classify performances according to style. This model did poorly in the B-side experiment task (see below.) There are a few likely reasons why this model did not succeed. First is that the loss function did not constrain the node embeddings into a space where the dot product or cosine similarity represented degree of similarity between two performances. Second, the style information itself seems incomplete (many performances have no substyle.) Finally, the styles maybe too coarse to learn embeddings the helpfully differentiate in a spectrum of similarity. In the future work section below, I suggest addition ways the label information might be used to learn, directly, when two performances are similar.

Many recommendations systems use graphs, which are a natural extension of matrix factorizations problems in collaborative filtering. In those problems, an item and query have an edge between them if an appropriate interaction exists. For example, in movie recommendation, a user and a movie might be linked if the user gave the movie a positive review. In the graph context, the probability of a link between an item and query can be used to order items for recommendation. With this in mind, I also tried some link prediction tasks; two performances are similar if they share similar probability distributions over actual and hypothetical links. I investigated two potential link prediction tasks; first, performance-artist links, second, performance-album links. in both cases, the model also did poorly on the B-side experiment and recommendation task. But, the model was exceptionally good at finding the links. Indeed, that I spent a fair amount of time confirming that there was not leakage to the dev set because perfect prediction is often a symptom of leakage rather than model quality. Leakage is a common problem in link prediction learning for graphs because information can "sneak" along edges in subtle ways, including reverse edges of undirected graphs. Leakage was not the sources of the problem--the issue appears to be that certain links are just too easy to predict with enough information. (A common feature of our jazz graph is that two performances on the same album will share exactly the same artists, making the artists in one performance exceptional signal about the artists in another. Random samples of negative edges must be carried out very carefully to make the task informative enough to be challenging.) The combination of high performance on these training tasks but low performance on the downstream recommendation metrics suggests that the task is inappropriate to learning useful embeddings.

A self-supervised task for graph learning is also promising. In self-supervised learning tasks, a model is required to predict the graph structure itself. This is similar to token prediction tasks used to learn embeddings for words. For practical reasons, mostly relating to the code already engineered for the two supervised learning tasks, I decided to use a version of SimCLR for self-supervised learning. SimCLR was originally applied to image learning. A SimCLR model involves learning are representation h of an entity one seeks to represent and a representation z used for the training task. The training task in SimCLR involves created two augmented views of each sample and then learning to identify (1) which samples are augmentations of the same entity and (2) to spread the representations of different entities uniformly in the embedding space.

I tried two different augmentation approaches. First, an edge ablation task where random edges are removed from the graph. Second, a shared album task where two performances should have similar representations if they are on the same album. In some ways, these are similar to the edge prediction task. However, they add a component of contrastive learning. (EXPAND?)

Self-supervised learning resulted in performance embeddings that succeeded in the recommendation task, both for B-side and Spotify recommendations.

### Training and Model Architecture

In order to train on a large graph, it is necessary to sample the graph. Sampling is accomplished by ordering the nodes in a batch and then completing a random walk of constrained depth around each node in the batch. Initial attempts for random ordering of nodes resulted in no learning, however, this was traced back to extreme sparsity in the sampled graphs--placing recordings from the 1940 and 2020s in the same sample meant almost no relevant connection between nodes. A simple solution to this problems to add a jitter to the release year for each node at the start of each epoch and then order by this jittered year. This eliminated the sparsity problem and models were able to successfully learn from the jittered samples.

The minimum layer depth for learning from jazz collaborations is 2. Each performance node should learn not only from the features of artist who played on them but also from that artists performance neighborhood. In the collaboration graph, there are no direct performance-performance edges: to reach one performance from another, it is necessary to pass through an artist or song node. To reach a performance node from a performance node while moving through a song node we also need two hops. I selected a layer depth of three, allowing a more rich collection of graph information to reach each performance representation. I did not perform experiments to verify this decision, which remains for future work.

Models were trained without any informative features ("no feature models") and with two potentially important available features, the substyle information and an edge feature representing the instrument played ("with features.") In the no feature models each layer was a GraphSAGE convolution with 64 dimensional outputs. In the feature models, an attentional layer was used since SAGE models don't natively support edge features. Adding features to the model provided a significant boost to performance.

## Recommendation

Recommenders need to translate a collection of seed value into a collection of scored recommendations.

A recommender finds the dot product off all known performances with the embeddings of the seed performances, creating an n x m matrix, n = number of known recordings, m = number of seed recommendations. Each row, then, represents a single recording and it's similarity to all seed listing values. A score for each recording is generated by aggregating these scores.

I experimented with three different aggregation functions, sum, max and softmax. Summation weights all performances in the seed value equally; conceptually, this can be seen as promoting recommendations which are like the mean seed. To clarify this effect, imagine that the embedding primarily characterize whether a song is bebop or modal jazz; then, if 80% of all seeds are bebop, we would expect bepob performances to score highly while model performances score weakly, since the model performances will be similar to only 20% of seeds and each seed is equally weight. Max aggregation promotes performances which are highly similar to exactly one seed value. Conceptually, if some performances are avant-guard and avant-garde performances are highly similar to one another while others are hard bop and hard bop performances tend to be just somewhat similar to one another, max aggregation will have tendency to promote avant-garde recordings even if only a small proportion of seeds are avant-garde. Softmax aggregation takes each row of per-seed scores and weights it by the softmax of the row and then sums the score. The effect is up signaling performances that are highly similar to some seed value, and this may be seen as balancing the tendencies of max and sum aggregation.

Given an input model, how do we turn model outputs into a recommendation?

Also, two baselines.

## Evaluation and Results

We evaluate the quality of the recommendations by relevance and diversity. There is direct measure for a self-supervised task to assess whether it generates relevant recommendations. As a proxy, I used my own Spotify listening history. First, we split the listening history by album so that performances on an album are split between two sets. The reason for album splitting is straightforward: given that two songs are on the same album, it is almost trivial to know that I should recommend any unseeded performances that share an album with a seeded performance. The task is significantly more challenging if the system needs to understand that performance from different albums, which are more likely to be disjoint in some features, are similar.

What are the procedures for evaluation? Why use those?

Provide a table of results for the baselines and the models that were trained
| Task                    | Features          | Pooling   |   Novel Recall |   Familiar Recall |   Novelty |   BSide Mean |
|:------------------------|:----------------- |:----------|---------------:|------------------:|-----------:|----------------------:|
| Combined Loss           | Instrument, Style | sum       |       0.276 |          0.328       |  0.806  |              0.822 |
| Combined Loss           | Instrument, Style | max       |       0.297 |       1              |  0.840  |          **0.839**  |
| Combined Loss           | Instrument, Style | softmax   |       0.276 |          0.318       |  0.807  |              0.822 |
| Edge Ablation           | Instrument, Style | sum       |       0.314 |          0.365       |  0.910  |              0        |
| Edge Ablation           | Instrument, Style | max       |       0.239 |       1              | **0.917** |            0        |
| Edge Ablation           | Instrument, Style | softmax   |       0.323 |          0.393       |  0.910  |              0        |
| Match Album             | None              | sum       |       0.278 |          0.378       |  0.850  |              0.434 |
| Match Album             | None              | max       |       0.173 |       1              |  0.835  |              0.345 |
| Match Album             | None              | softmax   |       0.314 |          0.395       |  0.840  |              0.411 |
| Match Album             | Instrument, Style | sum       |       0.485 |          0.544       |  0.805  |              0.548 |
| Match Album             | Instrument, Style | max       |       0.295 |       1              |  0.829  |              0.523 |
| Match Album             | Instrument, Style | softmax   |   **0.518** |      **0.608**       |  0.804  |              0.548 |
| Random Walk Baseline    | ---               | ---       |       0.194 |          0.202       |  0.523  |              0.132 |
| Artist Weighted Baseline | ---              | ---       |       0.420 |          0.405       |  0.011  |              0        |

## Future Work
