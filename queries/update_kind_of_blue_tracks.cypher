// Recordings and tracks; :Track <-:LISTS_TRACK :Recording

// Extract existing performances of songs. These are known to be unique
MATCH (Freddie:Performance) -[:PERFORMING]-> (:Song {title: "Freddie Freeloader"})
MATCH (Bluei:Performance) -[:PERFORMING]-> (:Song {title: 'Blue in Green'})
MATCH (AllB:Performance) -[:PERFORMING]-> (:Song {title: 'All Blues'})
MATCH (Sketches:Performance) -[:PERFORMING]-> (:Song {title: "Flamenco Sketches"})
MATCH (KindOfBlue:Recording, {title: "Kind of Blue"})

// Create tracks with these properties, if they don't exist, else just bind existing to variables
MERGE (FreddieT:Track {trackNumber: 2, title: "Freddie Freeloader"})
MERGE (BlueiT:Track {trackNumber: 3, title: 'Blue in Green'})
MERGE (AllBT:Track {trackNumber: 4, title: 'All Blues'})
MERGE (SketchesT:Track {trackNumber: 5, title: "Flamenco Sketches"})


// Create listings in Kind of Blue, if they don't exist
MERGE (KindOfBlue) -[:LISTS_TRACK]-> (FreddieT)
MERGE (KindOfBlue) -[:LISTS_TRACK]-> (BlueiT)
MERGE (KindOfBlue) -[:LISTS_TRACK]-> (AllBT)
MERGE (KindOfBlue) -[:LISTS_TRACK]-> (SketchesT)

// Create recording relationships, if they don't exist
MERGE (FreddieT) -[:RECORDS]-> (Freddie)
MERGE (BlueiT) -[:RECORDS]-> (Bluei)
MERGE (AllBT) -[:RECORDS]-> (AllB)
MERGE (SketchesT) -[:RECORDS]-> (Sketches)