# JazzWork

A PyG based recommender system for jazz music.
JazzGraph uses the network of jazz collaboration to construct recommendations.
Where a traditional recommendation systems says
"I like the song 'Round Midnight', could you ask other users to recommend me some Jazz?"
JazzWork says
"I like the song 'Round Midnight', could you ask Theolonious Monk to recommend me some Jazz?"


## Concept

Consider "Round Midnight", one of the most recorded jazz songs of all time.
JazzGraph captures the rich structure around that song. It traverses not just who recorded it, but the web of influence and collaboration around it.
Moving through this network discovers the logic and inspiration which jazz musicians find in that song, and reveals paths of significance to jazz music itself.
Unlike traditional recommender systems, which are necessarily biased by commercial success,
Jazz Graph relies on the judge of the deepest experts on jazz: creators themselves.

## Files

[src/jazz-graph] python source code for project.
[notebooks](/notebooks/) contains interactive examples and demonstrations.
[queries](/queries/) contains database essentials.
[resources](/resources/) text files linking to helpful resources related to recommendation systems, graph deep learning, and PyG (pytorch-geometric) for graph applications.
