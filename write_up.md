# Jazz Graph

This project leverages collaboration with in jazz musical works to create a recommendation system. The main goals of the project are (1) to establish a large corpus of data about jazz music and collect that data in a graph format, (2) to develop a graph neural network that learns representation of jazz performances and (3) to evaluate a recommendation system which uses those representations to make recommendations.

Jazz music is highly collaborative. Jazz musicians collaborate directly when performing songs. Indirectly, jazz musicians play songs written by other jazz musicians. It is not an exaggeration to say that the most famous musicians in jazz from the 1950s and 1960s all played together at one time or another. It is fair to say the one could learn about new jazz simply starting with "Kind of Blue," the most famous and best selling jazz album of all time, and pick any artist on it to find another album of jazz. It is also not an exaggeration to say the the most famous jazz songs have all been played by the most famous jazz musicians, often several times. Jazz performances involve rearrangement, improvisation, and substantial novelty as well. Many famous jazz performances, such as John Coltrane's "My Favorite Things" are stylitstic reimaginings of the originals.

Jazz thus forms an elaborate social network and readily represnted as a graph. Analysis and modeling of graphs is an active area of computer science, mathematics and data science.

The aim to produce a recommender system based on jazz collaboration differs form many recommender system approaches in some ways.
1. The system uses features of the subject matter to produce representations of similarity that may be used for recommendation.
1. Music and other enteratinment content, in particular, have often relied on systems of aggregating knowledge and preference from human users to make recommendations. For example, collaborative filltering is often a matrix favorization problem where user-item interactions are the modeled data. In such a context, user-item interaction are either (1) a proxy for the domain features or (2) considered the primary target of represention. With 1, knowledge about musical similarity is assumed to be captured in user-item interactions. With 2, musiscal similarity is beside the point--the goal is learning something which represents something else, such as probabilities of user engagement. (We do not wish to overstate the reliance on user-item iteractions. It is well known that large enterainment services incorporate diverse sources of data in their recommender systems.)
JazzGraph takes a slightly different approach. It assumes that the subject domain contains sufficiently rich information that similarity can be encoded by learning from the domain's structure alone. In short, it assumes that a graph of artists, songs and performances is sufficiently rich that it is possible to learn representation of the domain sufficient to generate informed recommendations. In this regard, it is arguably more like human cognition regarding musical recommendation than collaborative filtering: human experts know the music and recommend music based on musical features; they don't (or many would be ashamed to admit to) making recommendations simply based on popularity and other's taste.
(Again, we should not exaggerate the point: knoweldge of who plays with whom is not the same as knowing a song. Nevertheless, it is knowledge of the subject domain, rather than (for example) user perferences. Another perspective on the present work is as an investigation of whether domain knowledge encoded in user preferences aligns with domain knowledge encoded in collaboration.)

## Data

### Data Model

We model jazz collaboration as a heterogenous graph. Within the graph, there are three primary entity types:
- Artists: people, primarily musicians who play songs. Additionally, composers of songs are included in the graph even when they are not also performers. Examples include Duke Ellington and John Coltrane.
- Songs: particular musical compositions that artists perform. Examples include "Take the A Train" and "My Favorite Things."
- Performances: events where musicians (or a single musican) play a song. Live performances and studio recording sessions are both performances.

(While our data about performances is captured in the published recordings of performances, it is important to distinguish a performance from a release. A release is a particular publication of a performance. A recording of a performance may be released and re-released numerous times, as, for example, when many record labels re-released their recordings on CD in the 1980s and 1990s.)

* Relations
* Features

### Data Sources

To complete the project, it was necessary to extract the data about jazz collaborations from publicly available sources.

Musicbrainz is a large public SQL database of musical recordings. It contains detailed information, including songs and performers, for many of the recordings in the catalog. It lacks two keys components needed for this project. A concept of a master recording or release that organizes re-released recordings under a single parent. (As an example, recordings released on CD in the 1980s and a vinyl release of the same from the 1950s to not share a single parent id.) Additionally, Musicbrainz does not have style or genre information which isolates jazz recordings from other genres. Musicbrainz is a publicly maintained database and therefore contains inconsistencies that result from imperfect governance of the data.

Discogs is makes available an XML files of its data. There are two XML files in Discogs of interest for this project. First, the releases table in Discogs includes the a parent object which duplicate releases share. Second, it contains data on releases themselves, including artist, trakclistings, and genres. Discogs does not have the two major weaknesses of Musicbrainz.  However, the data is not organized into relational tables. Thus, performers associated with a recording are not linked to a table with unique entries for each artist; instead, a release will indicate who played on it with string data. This means that it is difficult to reliably build performance edges for the graph.

To build the graph of jazz collaboration, we thus need to merge these two data sources by fuzzy matching on strings. In summary, this means identifying Discogs jazz master recordings and matching the earliest release in Musicbrainz with the same album title and artist name. This allows one to construct a table of jazz recordings in Musicbrainz and then use that database to construct the performing and performs edges.

### Process

A. Combine the sources finding the first known example of all recording_ids in Musicbrainz and selecting distinct records ordered by ascending release date. These rows were fuzzy matched by normalizing album title and artist name and aligning to the known jazz recordings in Discogs. These recording ids were added to a junction table and the relevant Discogs were added to a separate table. From this, a stable view of the jazz recordings can be read from the database.

Despite the simplicity of the plan, there were a number of iterations on this process. To ensure that the result of the data deduplicated relevant rows, did not have nulls values, was properly column sorted, I created schemas in Pandera to valid data before loading it. To prevent potential issues where the state of the database might be unclear, I prevented writing code to the database except when the working tree was clean, and I logged (in the database) the DDL and DML queries run against it. As a result, it is possible know

### Summarize

What the hell remains? How many nodes? How many edges? Etc.

Features? (Maybe just discuss features with training, since you mostl)

### "Issues"

Maybe not the best term, but acknowledge data quality in source material, effects of fuzzy matching and other matters of the overall graph quality.

i. Potential for duplicate records (nodes).
ii. Potential for missing records (missing nodes, missing edges).
iii. Potential for incomplete information (missing edges, incomplete node attributes).
iv. Combinations of effects: where a node is duplicated, a portion of relevant edges may direct to either of the nodes.

## Modeling

### GNN Approaches and Challenges

* GNN approaches.
* SimCLR

### Training

Graph sampling. Jitter Year.

### Recommendation

Given an input model, how do we turn model outputs into a recommendation?
Also, two baselines.

## Evaluation and Results

What are the procedures for evaluation? Why use those?

T

## Future Projects
