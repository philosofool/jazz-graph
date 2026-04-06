DB schema. It's Neo4j, so very flexible, but these are expectations.

```
Node:Musician
  name: STR, required
  primary_instrument: STR, optional

Node:Performance
  date: DATE, optional
  venue: STR, optional

Node:StubPerformance
  // use with partial tracks, outtakes, etc. which are not complete performances
  // of a song, capture banter in the studio by little/no music, etc.
  date: DATE, optional
  notes: STR // flexible metadata, probably lacks enough structure for rigorous queries.

Node:Song
  title: STR, required

Node:Recording
  // one or more performances collected for publication.
  title: STR, required
  releaseDate: DATE, optional

Relationship::PERFORMING
  # (:Performance) -[:PERFORMING]-> (:Song)

Relationship::PERFORMS
  # (:Musician) -[:PERFORMS {instrument: STR}]-> (:Performance)
  instrument: STR | LIST[str], required.
  role: STR, optional: 'leader', 'sideman'

Relationship::RECORDS
  # (:Recording) [:RECORDS {trackNumber: INT}]-> (:Performance)
  trackNumber: INT, unique, required
  title: STR, optional  // use for alternate title of song, i.e., Song as titled on recording.

Relationship::COMPOSES
  # (:Musician) -[:COMPOSES]-> (:Song)
  date: DATE, optional

Relationship::REISSUES
  # (:Recording) -[:REISSUES]-> (:Performance|Recording)
  // indicates a republication of existing published work. e.g., compilation, deluxe editions, etc.
```
