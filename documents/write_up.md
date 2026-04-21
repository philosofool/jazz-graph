# Jazz Graph

## System Architecture

![System Diagram](/documents/system_diagram.png)

The JazzGraph system includes a data pipeline (left), machine learning system (middle) and a recommendation system (right side). The pipeline builds a graph from two public data sources. The ML system provides embeddings representing the similarity of jazz musical performances. The recommender uses these embeddings to generate recommendations.

The complete repository for JazzGraph is at https://www.github.com/philosofool/jazz-graph.

## Jazz Collaboration for Recommendation

This project leverages collaboration with in jazz musical works to create a recommendation system. The main goals of the project are (1) to establish a large corpus of data about jazz music and collect that data in a graph format, (2) to develop a graph neural network that learns representation of jazz performances and (3) to build and evaluate a recommendation system which uses those representations to make recommendations.

Jazz music is highly collaborative. Jazz musicians collaborate directly when performing songs. Indirectly, jazz musicians play songs written by other jazz musicians. Most famous musicians in jazz from the 1950s and 1960s all played together at one time or another. One could learn about new jazz simply starting with "Kind of Blue," the most famous and best selling jazz album of all time, and pick any artist on it to find another album of jazz. The most famous jazz songs have been performed by most famous jazz musicians, often several times. Jazz performances involve rearrangement, improvisation, and substantial novelty as well. Many famous jazz performances, such as John Coltrane's "My Favorite Things" are stylistic re-imaginings of the originals.

Jazz thus forms an elaborate social network, readily represented as a graph. Analysis and modeling of graphs is an active area of computer science, mathematics and data science.

The aim to produce a recommender system based on jazz collaboration differs from many recommender system approaches in some ways.
JazzGraph uses topological features of the subject matter to produce representations of similarity that may be used for recommendation.
Music and other entertainment content have often relied on systems of aggregating knowledge and preference from human users to make recommendations. For example, collaborative filtering is often a matrix factorization problem where user-item interactions are the modeled data.
In such a context, user-item interaction are either (1) a proxy for the domain similarities or (2) considered the primary target of representation. With (1), knowledge about musical similarity is assumed to be captured in user-item interactions. With (2), musical similarity is not be the aim--rather, something else, such as probabilities of user engagement, are the target of learning. (I do not wish to overstate the reliance on user-item interactions. It is well known that large entertainment services incorporate diverse sources of data in their recommender systems.)

JazzGraph takes a slightly different approach. It assumes that the subject domain contains sufficiently rich information that similarity can be encoded by learning from the domain's structure alone. In short, it assumes that a graph of artists, songs and performances is sufficiently rich that it is possible to learn representation of the domain sufficient to generate informed recommendations. In this regard, it is arguably more like human cognition regarding musical recommendation than collaborative filtering: human experts know the music and recommend music based on musical features; they don't (or many would be ashamed to admit to) making recommendations simply based on popularity and other's taste.
(Again, I should not exaggerate the point: knowledge of who plays with whom is not the same as knowing a song or performance. Nevertheless, it is knowledge of the subject domain, rather than (for example) user preferences. Another perspective on the present work is as an investigation of whether domain knowledge encoded in user preferences aligns with domain knowledge encoded in collaboration.)

## Data

### Data Model

I model jazz collaboration as a heterogenous graph. Within the graph, there are three primary entity types:
- Artists: people, primarily musicians who perform songs. Additionally, composers of songs are included in the graph even when they are not also performers. Examples include Duke Ellington and John Coltrane.
- Songs: particular musical compositions that artists perform. Examples include "Take the A Train" and "My Favorite Things."
- Performances: events where musicians (or a single musician) play a song. Live performances and studio recording sessions are both performances.

(The data about performances is captured in the published recordings of performances, but it is important to distinguish a performance from a release. A release is a particular publication of a performance. A recording of a performance may be released and re-released numerous times, as, for example, when many record labels re-released their catalogs on CD in the 1980s and 1990s.)

There are three relations that characterize the data and connect the node types:
- artist -*performs*-> performance: In jazz, this usually means playing an instrument as part of the performance. Which instrument is played is a feature of the relation, when that information is include. (That is, it's an edge feature.)
- performance -*performing*-> song: A performance is usually of some musical composition and this reflects which song is played in a performance.
- artist -*composed*-> song: Usually a song has a composer, the person or people who initially composed the music.

The following image displays some connectedness typical of famous jazz albums in the same era. It depicts four famous jazz albums, "Speak No Evil", "Sunday at the Village Vanguard," "Inner Urge" and "The Blues and the Abstract Truth." These recordings, which are also used for qualitative evaluations, are summarized in Appendix C.


![data_model](/documents/poster_graph.png)

### Data Sources

To complete the project, it was necessary to extract the data about jazz collaborations from publicly available sources.

[Musicbrainz](https://musicbrainz.org/) is a large public SQL database of musical recordings. It contains detailed information, including songs and performers, for many of the recordings in the catalog. It lacks two keys components needed for this project. A concept of a master recording or release that organizes re-released recordings under a single parent. (As an example, recordings released on CD in the 1980s and a vinyl release of the same from the 1950s do not share a single parent id.) Additionally, Musicbrainz does not have style or genre information which isolates jazz recordings from other genres. Musicbrainz is a publicly maintained database and therefore contains inconsistencies that result from imperfect governance of the data.

[Discogs](https://www.discogs.com/) is makes available XML files of its data. There are two XML files in Discogs of interest for this project. First, the releases table in Discogs includes the a parent object which duplicate releases share. Second, it contains data on releases themselves, including artist, track listings, and genres. Discogs does not have the two major weaknesses of Musicbrainz.  However, the data is not organized into relational tables. Thus, performers associated with a recording are not linked to a table with unique entries for each artist; instead, a release will indicate who played on it with string data. Songs similarly have no relational information and composers are not part of the data schema. This means that it is difficult to reliably build the edges for the graph.

To build the graph of jazz collaboration, we thus need to merge these two data sources by fuzzy matching on artist and title strings. In summary, this means identifying Discogs jazz master recordings and matching the earliest release in Musicbrainz with the same album title and artist name. This allows one to construct a table of jazz recordings in Musicbrainz and then use that database to construct the performing and performs edges.

### Process

The process for combining these two datasets involved two key steps. First, it is necessary to identify a first release for each recording id. A recording id in Musicbrainz corresponds to a single distinct recorded performance which may appear on multiple different releases, including compilation and releases on different media. A release is typically an album in a medium and thus we can create a table of first releases by selecting distinct recording ids ordered by album. The first releases table includes all recording ids (Jazz or not) within Musicbrainz.

Using a uniform normalization logic, I match entries in first releases and Discogs master releases by artist name and album name, using only those Discogs releases classified as Jazz. Discogs genre is a multi-label classification applied to releases, not individual tracks on them. There are numerous recordings classified as belonging to multiple genres, including Jazz. Some releases in Discogs are multi-label because they are compilation releases akin to "Great Recordings of the 1950s" which may span several genres of music; in such cases, Discogs applies a label to an album it applies to any track on the release. In other cases, multi-labeling occurs when tracks overlap multiple genres. It is surprisingly common, for example, that a recording is both "Electronic Music" and "Jazz." For simplicity, I used only Discogs releases with a single positive label "Jazz."

The matching between the two sources raises a number of common problems with string matching. Variations such as curly and straight quotes, capitalization, roman numerals and abbreviations represent some differences when matching strings. An additional variation occurs because metadata relating to the recording is sometimes included in the title.
Among the issues of metadata, there parenthetical expressions like "(Legacy Edition)" "(5.0 Mix)" and "(Rudy van Gelder Edition)" which are basically meaningless to the song identity and not replicated across data sources. There are also parentheticals like "(Live at the Village Vanguard)" or "(Take 2)" which indicate something about the performance and may distinguish from another version in the same release. Finally, some songs include alternative titles such as "Manha de Carnaval (Black Orpheus)" or "Corcovado (Quiet Night of Quite Stars)" where an alternate title or chorus lyrics are included. Several iterations of normalization strategy were worked out to handle various cases. For example, John Coltrane's album "A Love Supreme" variously includes abbreviations, Roman/Arabic numerals and often includes "A Love Supreme: ..." in the title of each song.

The normalization procedure here was also used downstream to match recordings in Spotify, where the addition of metadata in titles is especially common.

Well before approaching a final version of the dataset, I added guardrails to prevent modifying the data with an unclean git tree. This means that the state of normalization can be seen by looking at the commit when the data was updated.
See [`data_normalization.py`](../src/jazz_graph/clean/data_normalization.py) and [`test_data_normalization.py`](../tests/clean/test_data_normalization.py) for exact process and examples, most of which are extracted from actual cases.

After matching all recording ids in Musicbrainz corresponding to a jazz release in Discogs, I create a one-many join table in the Musicbrainz database from Musicbrainz's recording id to Discogs id. An inner join (or suitable left/right join) on this table assures than only entries that were matched is included in query results, providing a convenient way to filter any Musicbrainz query to jazz releases.

Because Musicbrainz is heavily normalized, it is difficult to describe in full detail the steps needed to join, e.g., performers with recordings, in Musicbrainz. Thus, I describe the conceptual structure of join below. The files in [`queries`]('../queries') represent sequences of updates to the database, which create views of the data for convenience in the steps described below. The queries to write the necessary graph entities are in [`create_parquet_tables.py`](../scripts/create_parquet_tables.py).

A relational table structure is a type of graph; foreign keys describe an edge between the primary entities in two tables, for example. For example, if we join a recordings table with a performers table where recordings.recording_id is performers.recording_id, we have an edge that defines an "artist -performs-> performance" edge. We can leverage this to produce three tables of nodes:
1. recording_ids inner joined to Discogs creates the performance nodes;
1. Linking the jazz performances with their songs and deduplicating creates a table of song nodes which includes the composer.
1. joining the jazz recordings table with a performance to performer edge provides a table of artists; a union of this table with the composers of in the song table. This gives an artist table (including both performers and composers.) Note that the union operation is deduplicating.

We have several edge tables that map the id fields in the above tables in a many-many relation and this all exists within Musicbrainz:
1. The artist-performance table links artist_id in artist with recording (performance) id in performances.
1. The performance-song table links the song_id and performance_id, corresponding to which song is performed.
1. The song-artist table links artist_id (composers) with song_id.
Note that edge tables values are the indexes of corresponding node's row in each table, not the ids themselves.

I discuss the specific features with each node in the training section below.

### Summary

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

As a helpful guide to the meaning in this table, the max row indicates, from left to right:
some artist composed 288 songs in the data;
some performer was part of 1788 performances;
some performance (either a medley, or perhaps a data issue) combined 13 songs;
some song had 9 different composers involved;
some performance include 81 musicians;
and some song was performed in 321 different performances in the data.
The percentiles give a sense of the overall connectedness within each of these categories.

The effects of fuzzy matching and source data quality present opportunities for improvement to JazzGraph. I discuss some of these issues in the future work section below.

## Modeling

### GNN Approaches


There are a number of models for Graph Neural Network (GNN) learning. The basic approach of all GNNs is to characterize graph nodes with their feature information; the feature information can be features from the data, learned features (i.e. embeddings) or a combination of these. The process of learning about the graph involves learning a representation of a node with its neighborhood. For example, the GraphSAGE algorithm learns a representation $h_1$ of a node's neighborhood by transforming the feature representation of a node and it's neighbors with a learned transformation and then aggregating these to a representation of the node's neighborhood. Through additional layers, a representation of a node, its neighbors, and it's neighbors' neighbors can be learned. Roughly:
$$h_{x_i} = agg(W_1x_j, \forall x_j \in N_x)$$
where $h_{x_i}$ is the representation of node $x_i$, $W_1$ is the learned weight matrix, $N_x$ is the set of nodes that are neighbors of $x$, and $agg$ is an aggregation function, such as summation. In more sophisticated models, the aggregation can be informed by features of edges, which (of course) determine which nodes are connected to $x_i$. $h_1$ is a representation of a node and its one hop neighborhood; by successively applying a graph convolution to the previous layer's output, we include information from one additional hop in the graph.

The goal is to learn representations of musical performances from the graph of collaboration.
The representations will be used in the downstream recommendation task. Thus, it is necessary to find a task for learning representations which generates a useful concept of similarity. Following standard deep learning practice, all the approaches for this I considered involved learning an embedding to capture the latent space which characterizes the features for learning; the expectation is the the dot product of two node's embeddings is larger when similarity is greater.

There are many options for such a task, though the available data limits what might be done. Task are naturally divided into two classes, supervised and unsupervised (or self-supervised) models.

One label available for supervised learning is the jazz sub-style information in the graph. Each performance has a multi-label feature corresponding to the jazz sub-style(s) it exemplifies. One advantage of using this data is that music style presumably directly correlates to listener preferences, so that a listener who likes a lot of bebop would presumably like other songs which are also bebop. With this in mind, the first version of the model that I built learned embeddings to classify performances according to style. This model did poorly in the B-side experiment and recommendation tasks (see below.) There are a few likely reasons why this model did not succeed. First is that the loss function did not constrain the node embeddings into a space where the dot product or cosine similarity represented degree of similarity between two performances. Second, the style information itself seems incomplete (many performances have no sub-style.) Finally, the styles maybe too coarse to learn embeddings that helpfully differentiate in a spectrum of similarity.

Many recommendations systems use graphs, which are a natural extension of matrix factorizations problems in collaborative filtering. In those problems, an item and query have an edge between them if an appropriate interaction exists. For example, in movie recommendation, a user and a movie might be linked if the user gave the movie a positive review. In the graph context, the probability of a link between an item and query can be used to order items for recommendation. With this in mind, I also tried some link prediction tasks; two performances are similar if they share similar probability distributions over actual and hypothetical links. I investigated two potential link prediction tasks; first, performance-artist links, second, performance-album links. in both cases, the model also did poorly on the B-side experiment and recommendation task. But, the model was exceptionally good at finding the links.
Leakage is a common problem in link prediction learning for graphs because information can "sneak" along edges in subtle ways, including reverse edges of undirected graphs.
Because perfect prediction is often a symptom of leakage rather than model quality, I took care to confirm that there was not leakage to the development set.
Leakage was not the source of the problem--the issue appears to be that certain links are just too easy to predict with enough information. (A common feature of our jazz graph is that two performances on the same album will share exactly the same artists, making the artists in one performance exceptional signal about the artists in another. Random samples of negative edges must be carried out very carefully to make the task informative enough to be challenging.) The combination of high performance on these training tasks but low performance on the downstream recommendation metrics suggests that the task is inappropriate to learning useful embeddings.

A self-supervised task for graph learning is also promising. In self-supervised learning tasks, a model is required to predict the graph structure itself. This is similar to token prediction tasks used to learn embeddings for words. For practical reasons, mostly relating to the code already engineered for the two supervised learning tasks, I decided to use a version of [SimCLR](https://arxiv.org/abs/2002.05709) for self-supervised learning. SimCLR was originally applied to image learning. A SimCLR model involves learning are representation $h$ of an entity one seeks to represent and a representation $z$ used for the training task. The training task in SimCLR involves created two augmented views of each sample and then learning to identify (1) which samples are augmentations of the same entity and (2) to spread the representations of different entities uniformly in the embedding space.

I tried two different augmentation approaches. First, an edge ablation task where random edges are removed from the graph. Second, a shared album task where two performances should have similar representations if they are on the same album. I discuss these tasks more in the training section below.

Self-supervised learning resulted in performance embeddings that outperformed baselines in the recommendation task.

### Training and Model Architecture

In order to train on a large graph, it is necessary to sample the graph. Sampling is accomplished by ordering the nodes in batches and then sampling each nodes' neighbors to constrained depth around each node in the batch. Initial attempts for random ordering of nodes resulted in no learning, however, this was traced back to extreme sparsity in the sampled graphs--placing recordings from the 1940 and 2020s in the same sample meant almost no relevant connection between nodes. A simple solution to this problem was to add a jitter to the release year for each node at the start of each epoch and then order by this jittered year; I chose a uniform random jitter of +/-4 years. This eliminated the sparsity problem and models were able to successfully learn from the jittered samples.

The minimum layer depth for learning from jazz collaborations is 2. Each performance node should learn not only from the features of artist who played on them but also from that artist's performance neighborhood. In the collaboration graph, there are no direct performance-performance edges: to reach one performance from another, it is necessary to pass through an artist or song node. To reach a performance node from a performance node while moving through a song node we also need two hops. I selected a layer depth of three, allowing a more rich collection of graph information to reach each performance representation. I did not perform experiments to verify this decision.

Models were trained without any informative features and with two potentially important available features, the sub-style label on performance nodes and an edge feature representing the instrument played. In the no feature models each layer was a GraphSAGE convolution with 64 dimensional outputs. In the feature models, I used an attentional layer, GATv2Conv, since SAGE models don't natively support edge features. Models learn a 64 dimensional embedding for each node. The hidden dimensions of the graph convolutions are 128 dimensional; in the models with features, the feature embeddings are 64 dimensional and concatenated in the first graph convolution with the hidden layer embedding.

Summary of Features:

| Feature Name | Cardinality | Embedding dim | Notes |
|:-------------|------------:|--------------:| :---- |
|Sub-style     |         20  |            64 | Multi-label |
|Instrument    |         28  |            64 | Multiple edges per artist if relevant, not multiple labels per edge |


Appendix A contains a descriptions of features and their distributions.

I tried three tasks for training the SimCLR model, corresponding to different augmentation approaches. The first augmentation was edge ablation; each edge in the graph had a 20% chance of removal and the model needs to recognize that two differently augmented nodes are the same. The second augmentation matches performances of the same album to each other, which is a more directly semantic model of the nodes' similarity. (This perhaps stretches the concept of an augmentation, but the training process is essentially the same.) Finally, I tried a dual loss approach which required the model to do both tasks and applied a weighted loss.

## Recommendation

Recommenders need to translate a collection of seed value into a collection of scored recommendations.

The recommenders using the GNN representations compute the dot product of all performances' embeddings with the embeddings of the seed performances, creating an $n \times m$ matrix, $n$ is the number of known recordings, $m$ is the number of seed recommendations. Each row, then, represents a single recording and its similarities to all seed listing values. A score for each recording is generated by aggregating these similarity scores.

I experimented with three different aggregation functions: sum, max and softmax. Summation weights all performances in the seed value equally; conceptually, this can be seen as promoting recommendations which are like the majority of seeds. To clarify this effect, imagine that the embedding primarily characterize whether a song is bebop or modal jazz; then, if 80% of all seeds are bebop, we would expect bebop performances to score highly while modal performances score weakly, since the modal performances will be similar to only 20% of seeds and each seed is equally weighted. Max aggregation promotes performances which are highly similar to exactly one seed value. Conceptually, if some performances are avant-garde and avant-garde performances are highly similar to one another while others are hard bop and hard bop performances tend to be just somewhat similar to one another, max aggregation will have tendency to promote avant-garde recordings even if only a small proportion of seeds are avant-garde. Softmax aggregation takes each row of per-seed scores and weights it by the softmax of the row and then sums the score. The effect is up signaling performances that are highly similar to some seed value, and this may be seen as balancing the tendencies of max and sum aggregation.

In addition to the GNN base models, I wrote two baseline recommender systems for comparison. The random walk baseline takes a seed recommendation and does a two-hop walk for the seed performance to a performer of the seed to a performance from that performer. This is repeated 10 times for each seed. It represents a simple graph traversal for finding quality recommendations which any trained model should be able to beat. The artist weighted recommender assumes that every performance by a seed artist is relevant and scores all performances as the relative frequency of the artist in data.

## Evaluation and Results

I constructed three measures of recommendation quality: novel recall, novelty, and a b-side experiment.

The first two use a listener's Spotify history to generate recommendations and then examine the qualities of the recommendations. Publicly available, granular data representing user engagement with music is scarce. As a proxy for a broader listening histories, I used my own Spotify listening history. With the same normalization process used to extract the dataset, the history is matched with artist, album and song title using against the Discogs sources. Each matched item is an eligible seed, and seeds were unweighted; for example, a performance listened to 100 times was weighted the same as a performance listened to once. I split the listening history into seed and holdout sets. Performances are split randomly by album, so that when two performances appear on the same album, they are in the same split. The reason for album splitting is that if two songs are on the same album, it is almost trivial to know that I should recommend any unseeded performances that share an album with a seeded performance. The task is significantly more challenging if the system needs to understand that performance from different albums, which are more likely to be disjoint in some features, are similar.

Novel recall is top-K recall on the held-out performances and is measures the relevance of the recommendations. We set K to a top 20% of ranked recommendations. Recall is an appropriate metric for this system. It measures the system's ability to detect relevance while not penalizing the system for indicating relevance outside the listener's history. Strong recall on held-out data suggests high relevance in the collection of recommendations, which is our target.  I also provide the familiar recall for all models, which indicates recall of seed performances in the top-K.

Novelty is the percentage of artists in the recommendations which were not in the seeds and is a measure of diversity and exploratory behavior. Novelty is a value in music recommendations; most users would like to have a system that helps them discover new music. Additionally, assuming that the recommendations are typically relevant, it measures the ability of the system to find similarities that are not immediately present in the data. However, high novelty in the presence of low overall relevance would be symptomatic of random recommendations.

A third metric is the "B-side experiment." The B-side experiment splits performance on a single album into seed and holdouts. It measures MAP at K for the held-out variants. (It has its name because the second side of a record is often called the b-side.) For this experiment, I report that average MAP at K on 6 albums by prominent Jazz artists for famous recordings; k was set to 20 plus the number of songs on the B-side. The six selected albums are summarized in Appendix B.

| Task                    | Features          | Pooling   |   Novel Recall |   Familiar Recall |   Novelty |   B-side Mean |
|:------------------------|:----------------- |:----------|---------------:|------------------:|-----------:|----------------------:|
| Combined Loss           | Instrument, Style | sum       |       0.276 |          0.328       |  0.806  |              0.822 |
| Combined Loss           | Instrument, Style | max       |       0.297 |       1              |  0.840  |            **0.839**  |
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

### Qualitative Assessment

Qualitative assessment provides an important source of information about the recommendations. Using the successful softmax, match album model above, I assessed the quality of recommendations (1) on my seed listening history, (2) on a specific small sample of albums I like and (3) a single performance ("Isotope") which is in the small sample. The small sample was also used for the subgraph used to illustrate the data model above.

In all of these, I focused on whether the model delivered recommendations that align with the musical preferences that are expressed by the sample, as I understand that. As part of this, I listened to novel performances in the recommendations as well as assessing familiar recordings in the recommendations.

It's worth noting that, as a jazz fan, it's expected that I like this music. However, misalignment like recommending Dixieland Jazz or vocal-heavy contemporary jazz would be an obvious mistake. I didn't discover any of that.

The results tend to cluster albums, though not perfectly. This is to be expected, given the training process. Song is not much of a signal in these (if it were, we would expect more shuffling between performance on different albums.) It's interesting that I seemed to like the results more as the number of seed values went down. That suggests a hypothesis: the recommender does better with singular signals than blended ones, however, this may be more of a weakness than a strength.

#### 1. Total Seed Assessment

This is a hard set to evaluate since it's essentially an unweighted matching of all the songs I have listen to that are jazz. Some of the seeds are songs I listened to more four years ago than today, but there is no temporal weighting. The theme to me is "very standard hard bop." If there were an easy way to test that Herbie Hancock is the dominant thread in this, I would test that. The feel aligns with a subset of jazz that I listen to; the jazz that it does align with is not very like the small sample jazz that I provided, and that suggests to me that it is maybe a little too much like the median performance of jazz than like my personal preferences.

Grade: B-. Needs to cover more styles of jazz; it's pretty obviously late 50's Miles Davis, early 60's Herbie Hancock, Thelonius Monk, and similar. Not as eclectic as my overall preferences.

#### 2. Small Sample Assessment

The recommendations here seemed to align well with the input values. Several are clear examples of hard bop, which align well with the Oliver Nelson seeds and (to a lesser extent) the Joe Henderson seed. "Max Roach, His Chorus and Orchestra" appeared in this; it's somewhat surprising because it includes a vocal chorus (no seed does) and is a larger ensemble than others in this selection. The alignment here is interesting because I loved the music but it was pretty unexpected given that it's not directly connected to any performance in the seeds. Whether that's an indication of randomness or finding deeper connection is an open question. There's a lot of Joe Henderson in the examples, suggesting that he is a strong signal. The included performances all range from 1959 to 1970, which makes a lot of sense from these seeds. Wes Montgomery recordings in the set feel misplaced; guitar and flute jazz that feels a little like Sonny Rollins to me is a little out of sync with these recordings.

Grade: B. I really like the music, but I don't see connections and some of it is stylistically a departure from the seeds.

#### 3. Single Performance Assessment

The single performance I selected was Joe Henderson's "Isotope" from his album "Inner Urge." Henderson was a tenor saxophonist and the performance includes pianist McCoy Tyner and drummer Elvin Jones, both members of tenor saxophonist John Coltrane's contemporaneous quartet.

The model did very well in the B-side experiment here: all songs from Inner Urge appear among the top recommendations. I expected any number of other Joe Henderson recordings from the same era as well as McCoy Tyner and Elvin Jones collaborations (there were many) to appear in this set. However, they were much less prominent that expected. Two novel (to me) artists Lester Bowie and The Don Rendell / Ian Carr Quintet appeared among the top. I consider the Lester Bowie recommendation an A in exploration; that song is an excellent piece of avant-garde jazz with stylistic feels that closely resemble Henderson. The Rendell/Carr stuff is also great. Other artists appearing in the list, including Bobby Hutcherson, Eric Dolphy and Andrew Hill, are great jazz artists whose work is worth knowing and who are close to Henderson--if a little more out there.

However, I think this test exposes model weakness more than strength. Given how the system is implemented, I would expect the basic result to be tenor saxophone quartets with artist overlap. Given such narrow seeds and model architecture, being very close to the neighborhood and features of "Isotope" probably reflects better on the model learning the features one expects than good exploration. If each song in a large seed set produces varied and exploratory results, how well can a recommender to aggregate the collection of each seed's suggestion?

Grade: A for exploration, but C for alignment. These were the most exciting recommendations of the three qualitative experiments, but they may represent a weakness rather than strength of the model. If the model were great at other metrics, I would love this result, but the big picture isn't quite right.

## Future Work

There are many opportunities to improve the system.

There are potential data quality issues which, if mitigated, may provide a performance boost.  The data fed the system has some liability to duplicate nodes.
Duplicate performance nodes may emerge from releases and compilation if the Musicbrainz data does not correctly associate the recording id in the re-release. Songs are often published with variant names, potentially creating duplicate entries. Missing performance-artist edges are likely for less common artists, especially when those artists did not publish as band leaders or published with multiple names. Missing song-performance edges are possible when a song has a variant title.

In addition to data quality issues, there are some features in the data which have not been used. Release year, in particular, is probably a strong signal of similarity. It was not included in this system because I observed many performances associated with a year much later than the original performance. This is likely because the CD release was the only one in Musicbrainz (late 80's was very common with these.) For artists with long careers, performances that are nearby in year may be more similar than distant years. Finding additional data outside of the datasets here, such as tempo and mood, would also be beneficial. An exciting feature possibility is to encode recordings themselves as features, though this confronts significant issues of copyright as well as complexity.

The validation data here is just the author's Spotify history. It would be useful to expand these data. Any user's spotify data is affected by the recommendation systems that Spotify uses. It would be useful to find other sources of data (Apple Music, etc.) to verify against those results.
There are also multiple ways to decide which listens in a history are used as seeds, how to weight seeds or to split them temporally. Ideally, a seed value is a performance that the user certainly likes; here, I assumed that if the user listened to it, they liked it. This is obviously an over-simplification, and alternatives approaches to classifying liked songs may be useful. Even with only one user's history, these may be useful alternative validation strategies.

It would also be helpful to include, even if manually generated, a collection of jazz music which is different from my tastes. For example, what would a manually created list of Nora Jones and Ella Fitzgerald music recommend? This would demonstrate that the observed pattern of "Recommendation 50's and 60's hard bop" observed in the first qualitative assessment won't emerge when the seed values are certainly different.

Besides listening histories, there are additional validations which might be added in the future. Validation that examine specific expected similarities, like the B-side experiment, would be useful. Exploring embedding distance directly would be helpful. For example, one would expect the average distance between performances of the same song to be smaller between those songs and randomly selected songs. Similar expectations emerge with the role of instruments, songs with the same composer, and so on. In general, for any edge type in the graph, one expects source nodes connected through a single destination node to be more similar than to randomly selected nodes of the destination type. Explicitly testing these expectations will provide insight into potential areas for improvement or regression and provide a sense of the difference between various approaches to modeling, data changes, or other alterations.

The GNN architectures here may not be optimal. SimCLR tries to spread the embeddings for samples (here, nodes) uniformly in the embedding space; however, this may not be true to the space of jazz performances, artist and songs; some regions of the space should be more dense than others.  Too late in development to add it, I discovered Metapath2Vec, variant of Node2Vec, which is based on Word2Vec. It uses metapaths like (artist -> performance -> artist) to take random walks of a graph. These random walks are treated like sequences in Word2Vec and the model constructs embeddings which predict the probability of other nodes being within the context window of the node. I suspect that this will be stronger than the SimCLR approach taken here.

The link prediction task which I tried did not produce useful embeddings. However, this is very likely due to the effect of randomly sampling negative edges. With over 24,470 artist in the graph, the vast majority of negative edges samples for link prediction between an artist and a performance are trivially easy to reject. Improving the negative sampling procedure may allow this approach to be more effective.

Within the SimCLR architecture here, it may also be useful to consider alternative augmentations. After the fact, the failure of the edge ablation task is somewhat obvious: in a graph, edge removal amounts to information destruction; augmentation by adding missing edges or changing node features might be more effective, especially combined with the suggestion to add features mentioned above. While albums are one concept of similarity, it's worth considering others that might be available. I will probably try Metapath2Vec and variants before doing these, however.


## Appendix A: Feature Summaries


There are 20 different sub-styles in the data. Their relative frequencies are given below:

| Jazz Sub-style     |    Percentage with label |
|:-------------------|:----- |
| Contemporary Jazz  | 8.03% |
| Swing              | 6.07% |
| Bop                | 5.53% |
| Big Band           | 4.91% |
| Post Bop           | 4.54% |
| Hard Bop           | 3.07% |
| Cool Jazz          | 2.85% |
| Fusion             | 2.39% |
| Vocal              | 2.37% |
| Soul-Jazz          | 2.23% |
| Free Jazz          | 2.08% |
| Easy Listening     | 1.96% |
| Modal              | 1.51% |
| Jazz-Funk          | 1.4% |
| Smooth Jazz        | 1.2%  |
| Avant-garde Jazz   | 1.0%    |
| Dixieland          | 0.95% |
| Free Improvisation | 0.91% |
| Latin Jazz         | 0.86% |
| Jazz-Rock          | 0.75% |

We see from the following table that many performances have no labels or more than one:
| Number of labels | Percentage of Samples |
|---|--------|
| 0 | 0.240% |
| 1 | 0.465% |
| 2 | 0.227% |
| 3 | 0.056% |
| 4 | 0.009% |
| 5 | 0.002% |
| 6 | 0.001% |
| 7 | 0.000% |

There were over three hundred instruments in the Musicbrainz data. I categorized some rare and similar instruments into groups; all groups are mutually exclusive.

The groups and the proportion of edges they label are given below:

| Instrument           |   Proportion |
|:---------------------|-------------:|
| drums                |        12.17% |
| piano (acoustic)     |        12.03% |
| trumpet              |         9.01% |
| other/world          |         8.77% |
| saxophone (tenor)    |         7.73% |
| bass (acoustic)      |         5.97% |
| trombone             |         5.94% |
| percussion           |         5.68% |
| guitar (acoustic)    |         5.42% |
| other personnel      |         4.51% |
| violin               |         3.66% |
| other wind           |         3.44% |
| saxophone (alto)     |         3.06% |
| piano (electric)     |         2.39% |
| other brass          |         2.06% |
| clarinet             |         1.33% |
| flute                |         1.19% |
| saxophone (baritone) |         1.15% |
| vibraphone           |         0.87% |
| organ                |         0.78% |
| saxophone (soprano)  |         0.67% |
| bass (electric)      |         0.63% |
| guitar (electric)    |         0.62% |
| vocals (other)       |         0.51% |
| vocals (lead)        |         0.29% |
| electronic/effects   |         0.08% |
| drums (electronic)   |         0.04% |

## Appendix B: B-side Experiment

The selection of albums for the B-side experiment was not systematic. I chose albums where qualitatively assessing recommendations would be easy for me and published around 1960 because while prototyping the model, I used data from that era only. The era in question is a highly connected period in Jazz.

| Artist | Album | Notes |
|-------|-------|-------|
|Miles Davis|Sketches of Spain| | Large ensemble, Davis is one of the most highly connected nodes.|
|Miles Davis|Kind of Blue|One performance (Freddie Freeloader) has a different pianist from the rest. This is the most recommended jazz album of all time.|
|Art Blakey and the Jazz Messengers|Mosaic||
|Dave Brubeck|Time Out| Popular jazz recommendation. Brubeck has many performances but connects mostly to the same sidemen in his work.|
|Charles Mingus|Mingus Ah Uhm|
|Ornette Coleman|The Shape of Jazz to Come| One very well connected song (Lonely Woman) but less well connected performer/performances than some others here. This was a highly challenging example for all models.|


## Appendix C: Qualitative Selection Results

The small sample, used in the second set of qualitative assessments, was this:

|   recording_id | Artist                   | Album                                                            | Performance Title                      |
|---------------:|:----------------|:---------------------------------|:-----------------------------|
|        3427034 | Joe Henderson   | Inner Urge                       | Night and Day                |
|       14805308 | Joe Henderson   | Inner Urge                       | Inner Urge                   |
|       14805309 | Joe Henderson   | Inner Urge                       | Isotope                      |
|       14805310 | Joe Henderson   | Inner Urge                       | El Barrio                    |
|       14805311 | Joe Henderson   | Inner Urge                       | You Know I Care              |
|        1662631 | Bill Evans Trio | Sunday at the Village Vanguard   | Gloria’s Step (take 2)       |
|         696865 | Bill Evans Trio | Sunday at the Village Vanguard   | Jade Visions (take 2)        |
|         159842 | Bill Evans Trio | Sunday at the Village Vanguard   | Alice in Wonderland (take 2) |
|         159841 | Bill Evans Trio | Sunday at the Village Vanguard   | Solar                        |
|       12541732 | Bill Evans Trio | Sunday at the Village Vanguard   | My Man’s Gone Now            |
|       12541735 | Bill Evans Trio | Sunday at the Village Vanguard   | All of You (take 2)          |
|         415771 | Wayne Shorter   | Speak No Evil                    | Witch Hunt                   |
|         415772 | Wayne Shorter   | Speak No Evil                    | Fee‐Fi‐Fo‐Fum                |
|         415774 | Wayne Shorter   | Speak No Evil                    | Dance Cadaverous             |
|         415775 | Wayne Shorter   | Speak No Evil                    | Speak No Evil                |
|         415777 | Wayne Shorter   | Speak No Evil                    | Infant Eyes                  |
|         415779 | Wayne Shorter   | Speak No Evil                    | Wild Flower                  |
|        5748097 | Oliver Nelson   | The Blues and the Abstract Truth | Stolen Moments               |
|         811575 | Oliver Nelson   | The Blues and the Abstract Truth | Hoe‐Down                     |
|         811576 | Oliver Nelson   | The Blues and the Abstract Truth | Cascades                     |
|         811577 | Oliver Nelson   | The Blues and the Abstract Truth | Yearnin’                     |
|         811578 | Oliver Nelson   | The Blues and the Abstract Truth | Butch and Butch              |
|         811579 | Oliver Nelson   | The Blues and the Abstract Truth | Teenie’s Blues               |



Results from using my full spotify history (top 30 recommendations):

|   recording_id | Artist                   | Album                                                            | Performance Title                      |
|---------------:|:-------------------------|:-----------------------------------------------------------------|:--------------------------------------|
|        7358212 | Lee Morgan               | Vol. 3                                                           | Tip-Toeing                            |
|        5511186 | Jackie McLean            | Let Freedom Ring                                                 | Rene                                  |
|         495915 | Thelonious Monk Quintet  | 5 by Monk by 5                                                   | Played Twice (take 2)                 |
|        9632965 | Lee Morgan               | Vol. 3                                                           | Domingo                               |
|        8585598 | Lee Morgan               | Vol. 3                                                           | Hasaan's Dream                        |
|         365672 | Miles Davis Sextet       | Someday My Prince Will Come                                      | Someday My Prince Will Come           |
|         495912 | Thelonious Monk Quintet  | 5 by Monk by 5                                                   | Straight, No Chaser                   |
|         495916 | Thelonious Monk Quintet  | 5 by Monk by 5                                                   | I Mean You                            |
|        5511185 | Jackie McLean            | Let Freedom Ring                                                 | I’ll Keep Loving You                  |
|         495914 | Thelonious Monk Quintet  | 5 by Monk by 5                                                   | Played Twice (take 1)                 |
|        5511184 | Jackie McLean            | Let Freedom Ring                                                 | Melody for Melonae                    |
|         495911 | Thelonious Monk Quintet  | 5 by Monk by 5                                                   | Jackie‐ing                            |
|         495917 | Thelonious Monk Quintet  | 5 by Monk by 5                                                   | Ask Me Now                            |
|         495913 | Thelonious Monk Quintet  | 5 by Monk by 5                                                   | Played Twice (take 3)                 |
|        7910939 | Tommy Turrentine         | Tommy Turrentine                                                 | Long as You're Living                 |
|        7358210 | Lee Morgan               | Vol. 3                                                           | Mesabi Chant                          |
|        6723097 | Wes Montgomery           | Full House                                                       | I've Grown Accustomed to Her Face     |
|        9553884 | Charlie Rouse            | Bossa Nova Bacchanal                                             | Back to the Tropics                   |
|         599006 | Grant Green              | Idle Moments                                                     | Django                                |
|        7910938 | Tommy Turrentine         | Tommy Turrentine                                                 | Time's Up                             |
|        7910942 | Tommy Turrentine         | Tommy Turrentine                                                 | Blues for J.P.                        |
|         599008 | Grant Green              | Idle Moments                                                     | Jean de Fleur (alternate take)        |
|        8387587 | Max Roach                | Drums Unlimited                                                  | In the Red (A Xmas Carol)             |
|        7910937 | Tommy Turrentine         | Tommy Turrentine                                                 | Webb City                             |
|        7910936 | Tommy Turrentine         | Tommy Turrentine                                                 | Gunga Din                             |
|       14350053 | The Kenny Dorham Quintet | Scandia Skies                                                    | Manha de Carnaval                     |
|        3002492 | Herbie Hancock           | My Point of View                                                 | Blind Man, Blind Man (alternate take) |
|        2073873 | Miles Davis              | In Person: Friday and Saturday Nights at the Blackhawk, Complete | Fran Dance                            |
|        3282589 | Thelonious Monk          | It’s Monk’s Time                                                 | Lulu’s Back in Town                   |
|         599009 | Grant Green              | Idle Moments                                                     | Django (alternate take)               |

Results using Joe Henderson, Isotope, as the seed (with seed removed):

|   recording_id | Artist                   | Album                                     | Performance Title                      |
|---------------:|:-----------------------------------|:--------------------------------|:------------------------|
|       14805308 | Joe Henderson                      | Inner Urge                      | Inner Urge              |
|       31407187 | The Don Rendell / Ian Carr Quintet | Shades of Blue                  | Just Blue               |
|       14805310 | Joe Henderson                      | Inner Urge                      | El Barrio               |
|       31407190 | The Don Rendell / Ian Carr Quintet | Shades of Blue                  | Blue Doom               |
|        5895422 | Lester Bowie                       | The Great Pretender             | It’s Howdy Doody Time   |
|       22843880 | Lester Bowie                       | The Great Pretender             | The Great Pretender     |
|       31407189 | The Don Rendell / Ian Carr Quintet | Shades of Blue                  | Garrison 64             |
|       14805311 | Joe Henderson                      | Inner Urge                      | You Know I Care         |
|       21839953 | Pete La Roca                       | Turkish Women at the Bath       | Dancing Girls           |
|       10492802 | The Don Rendell / Ian Carr Quintet | Dusk Fire                       | Tan Samfu               |
|        3915344 | Pete La Roca                       | Basra                           | Eiderdown               |
|        3427034 | Joe Henderson                      | Inner Urge                      | Night and Day           |
|         278907 | Eric Dolphy                        | At the Five Spot, Volume 1      | The Prophet             |
|       31407192 | The Don Rendell / Ian Carr Quintet | Shades of Blue                  | Big City Strut          |
|        8629708 | Andrew Hill                        | One for One                     | Without Malice          |
|       31407186 | The Don Rendell / Ian Carr Quintet | Shades of Blue                  | Latin Blue              |
|        5895426 | Lester Bowie                       | The Great Pretender             | Oh, How the Ghost Sings |
|       10492807 | The Don Rendell / Ian Carr Quintet | Dusk Fire                       | Dusk Fire               |
|       31407185 | The Don Rendell / Ian Carr Quintet | Shades of Blue                  | Blue Mosque             |
|        2575473 | Bobby Hutcherson                   | Stick-Up!                       | Blues Mind Matter       |
|        2575472 | Bobby Hutcherson                   | Stick-Up!                       | Verse                   |
|       10492805 | The Don Rendell / Ian Carr Quintet | Dusk Fire                       | Prayer                  |
|        3915339 | Pete La Roca                       | Basra                           | Malagueña               |
|        6085827 | Elvin Jones                        | Brother John                    | Harmonique              |
|        5777369 | Steve Lacy                         | The Straight Horn of Steve Lacy | Louise                  |
|        2575469 | Bobby Hutcherson                   | Stick-Up!                       | 8/4 Beat                |
|        2575470 | Bobby Hutcherson                   | Stick-Up!                       | Summer Nights           |
|        8629706 | Andrew Hill                        | One for One                     | One for One             |
|       31407191 | The Don Rendell / Ian Carr Quintet | Shades of Blue                  | Shades of Blue          |
|        5895425 | Lester Bowie                       | The Great Pretender             | Rose Drop               |

Results of the small sample, seeds removed:

|   recording_id | Artist                   | Album                                                            | Performance Title                      |
|---------------:|:------------------------------------|:-------------------------------------------------------------------------------------|:------------------------------------|
|       31487448 | Lucky Thompson                      | Plays Jerome Kern And No More                                                        | Why Do I Love You?                  |
|       19531578 | The Joe Henderson Quintet           | At the Lighthouse - "If You're Not Part of the Solution, You're Part of the Problem" | Caribbean Fire Dance                |
|        1130451 | John Coltrane                       | The Heavyweight Champion: The Complete Atlantic Recordings                           | Giant Steps (take 3) (incomplete)   |
|       15177263 | Joe Henderson Sextet                | The Kicker                                                                           | Mo' Joe                             |
|       19531586 | The Joe Henderson Quintet           | At the Lighthouse - "If You're Not Part of the Solution, You're Part of the Problem" | Blue Bossa                          |
|       15177259 | Joe Henderson Sextet                | The Kicker                                                                           | If                                  |
|         159726 | Miles Davis                         | Miles Davis’ Greatest Hits                                                           | So What                             |
|       12317340 | Joe Zawinul                         | Money in the Pocket                                                                  | Some More of Dat                    |
|        4389458 | Max Roach, His Chorus and Orchestra | It’s Time                                                                            | The Profit                          |
|       27857896 | John Coltrane                       | The Heavyweight Champion: The Complete Atlantic Recordings                           | Giant Steps, take 6 (alternate)     |
|       19531582 | The Joe Henderson Quintet           | At the Lighthouse - "If You're Not Part of the Solution, You're Part of the Problem" | ’round Midnight                     |
|        4389456 | Max Roach, His Chorus and Orchestra | It’s Time                                                                            | Sunday Afternoon                    |
|        4389455 | Max Roach, His Chorus and Orchestra | It’s Time                                                                            | Another Valley                      |
|        6663125 | The Cecil Taylor Quintet            | Hard Driving Jazz                                                                    | Shifting Down                       |
|       14358520 | Chet Baker                          | The Trumpet Artistry of Chet Baker                                                   | Russ Job                            |
|       15177262 | Joe Henderson Sextet                | The Kicker                                                                           | O Amor Em Paz                       |
|        6966939 | John Coltrane                       | Alternate Takes                                                                      | I'll Wait and Pray (alternate take) |
|       15177256 | Joe Henderson Sextet                | The Kicker                                                                           | Mamacita                            |
|       19531583 | The Joe Henderson Quintet           | At the Lighthouse - "If You're Not Part of the Solution, You're Part of the Problem" | Mode for Joe                        |
|       15177257 | Joe Henderson Sextet                | The Kicker                                                                           | The Kicker                          |
|       14358521 | Chet Baker                          | The Trumpet Artistry of Chet Baker                                                   | Tommy Hawk                          |
|       19531580 | The Joe Henderson Quintet           | At the Lighthouse - "If You're Not Part of the Solution, You're Part of the Problem" | A Shade Of Jade                     |
|        6924114 | Wes Montgomery                      | Movin' Along                                                                         | Tune-Up (take 4)                    |
|        6924121 | Wes Montgomery                      | Movin' Along                                                                         | Movin' Along (take 4)               |
|        4389454 | Max Roach, His Chorus and Orchestra | It’s Time                                                                            | It’s Time                           |
|        4389457 | Max Roach, His Chorus and Orchestra | It’s Time                                                                            | Living Room                         |
|       15177258 | Joe Henderson Sextet                | The Kicker                                                                           | Chelsea Bridge                      |
|        5319124 | Wes Montgomery                      | Movin' Along                                                                         | Tune Up                             |
|       19531587 | The Joe Henderson Quintet           | At the Lighthouse - "If You're Not Part of the Solution, You're Part of the Problem" | Closing Theme                       |
|        6924113 | Wes Montgomery                      | Movin' Along                                                                         | Movin' Along (take 5)               |