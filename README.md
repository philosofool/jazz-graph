# JazzGraph

_JazzGraph_ is a recommender system for using expert judgment to make music recommendations.
It uses the relationships of musical collaboration among jazz artists to encode similarity relations among performances and leverages those similarity relationships to make recommendations.
The model of similarity is a graph neural network.

Think of JazzGraph as the following: rather than ask other music listeners what songs to listen to, ask Thelonious Monk.

_JazzGraph_ begins as the capstone project for my Master of Science in Data Science. Below, we outline the initial proposal for completing the capstone.

## Files

[src/jazz-graph] python source code for project.
[notebooks](/notebooks/) contains interactive examples and demonstrations.
[queries](/queries/) contains database essentials.
[resources](/resources/) text files linking to helpful resources related to recommendation systems, graph deep learning, and PyG (pytorch-geometric) for graph applications.


## Concept

Create a graph neural network to learn interesting features of jazz as represented in a graph of jazz artist collaborations, and provide some interesting application of the model or features that it learns.

## Requirements

There should be one pull-request for each of the following requirements. Some of them should be fairly simiple; (1), for example, is probably a single markdown file that describes the entities and relations in the data. I expect 2, 4 and 5 to moderate sized PRs.
1. Create a data model expressive of collaboration in jazz; the model should be graph-like.
2. Extract a substantial amount of data in a useful format, in the model of 1. This should be documented (e.g., what are the inclusion criteria?) with working code that can replicate the results (e.g. how do we extract this from sources, load it to a database?)
3. Describe a task (e.g., classification, recommendation) and associated learning problem (evaluation criteria, surrogate loss function, etc.) suited to learning in with a graph neural network.
4. Implment in Pytorch a training loop that learns representations for completing the task described in 3. The complete python code for this should be included for replication of the results.
5. Complete some demonstration of the usefulness of the trained system.
6. Include references where appropriate to existing examples and research.

## Proposed implementation

I'm separating the requirements from this proposal so that the evalution criteria are clear even if the end results are different from this proposal; a proposal is a good way to show that something is feasible. The following would, I think, satisfy the requirements very well and I believe that this can be completed.

1. (Data Model) There are entities: artists, songs and performances. There are relationships: arists _compose_ songs; artists _perform_ performances; a performance _plays_ a song. This creates a heterogeneous graph of interconnected entities representing a significant component of jazz collaboration.
2. (Data Extraction, Processing) The data can be extracted from two public data sources, Discogs and Musicbrainz. It should be little problem to include a few hundred artists, thousands of performances and several thousand edges between performances and the performers and songs. Discogs is an extensive library of recordings with artists and style labeling, but does not include deep performance data.
For example, it won't include that there are numerous different performers with Thelonious Monk on "Genius of Modern Music," which was a complilation of five different recording sets, no two of which included the exact same performers.
Musicbrainz does much better on this score but (oddly) lacks genre labeling.
I propose to extract the above data, clean and transform it in Python, and load it to a Neo4j graph database, which is ideal for querying relationships among entities.
3. (Task, Task Criteria) I would like to build a graph recomender system. There are excellent examples of this in academic blogs and published papers. The goal is to learn relations of similarity between recorded performances based solely on who the performers, songs and composer are. Another possible task is sub-genre classification: is this performance bebop, modal, free jazz, hardbop, cool jazz, etc?
4. (Code a Model) This is fairly self explanatory: transform data from 2 above into a Pytorch geometric dataset, define the loss function and train a model. Provide some evaluations of success in the primary task (a metric which may be slightly different from the surrogate loss.)
5. (Demonstration/Application) I will wrap the model in code to make recommendations from a person's music listening history.
Getting a Spotify/Apple Music/etc. history to align with the data above would be a moderate size project in itself and I make no warranties about completing that as a step of the project.
However, I could probably do it for my own music history manually as a proof of concept.
It might also be instructive an fun to look at a few telling examples. ("I have only listened to Miles Davis' album _Kind of Blue_. What's next?" is an almost cliche jazz recommendation question--cool to see what the model would say.)
6. (Attribution and citation) I include proper citation of sources with each of the above.
