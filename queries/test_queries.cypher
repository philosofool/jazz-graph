// Constraints

// required properties
CREATE CONSTRAINT musician_name_required FOR (n:Musician) REQUIRE n.name IS NOT NULL
CREATE CONSTRAINT performance_instrument_required FOR ()-[performs:PERFORMS]-() performs.instrument IS NOT NULL

// Required property types

CREATE CONSTRAINT performance_date FOR (n:Performance) REQUIRE n.date IS :: DATE
CREATE CONSTRAINT performance_stub_date FOR (n:PerformanceStub) REQUIRE n.date IS :: DATE
CREATE CONSTRAINT recording_date FOR (n:Recording) REQUIRE n.releaseDate IS :: DATE

// Unique properties
CREATE CONSTRAINT track_number_unique FOR ()-[records:RECORDS]-() records.trackNumber IS UNIQUE


// Kind of Blue

// Musicians
CREATE (BillE:Musician {name: 'Bill Evans'})
CREATE (WyntonK:Musician {name: 'Wynton Kelly'})
CREATE (CannonballA:Musician {name: 'Cannonball Adderly'})
CREATE (JohnC:Musician {name: 'John Coltrane'})
CREATE (MilesD:Musician {name: 'Miles Davis'})
CREATE (PaulC:Musician {name: 'Paul Chambers'})
CREATE (JimmyC:Musician {name: 'Jimmy Cobb'})

// Songs.
CREATE (SoWSong:Song {title: 'So What?'})
CREATE (MilesD) -[:COMPOSES]-> (SoWSong)

CREATE (FreddieFSong:Song {title: 'Freddie Freeloader'})
CREATE (MilesD) -[:COMPOSES]-> (FreddieFSong)

CREATE (BlueiSong:Song {title: 'Blue in Green'})
CREATE (MilesD) -[:COMPOSES]-> (BlueiSong)
CREATE (BillE) -[:COMPOSES]-> (BlueiSong)

CREATE (AllBSong:Song {title: 'All Blues'})
CREATE (MilesD) -[:COMPOSES]-> (AllBSong)

CREATE (FlamencoSSong:Song {title: 'Flamenco Sketches'})
CREATE (MilesD) -[:COMPOSES]-> (FlamencoSSong)
CREATE (BillE) -[:COMPOSES]-> (FlamencoSSong)

// Performances: Concrete events at a particular place and time, usually related to a song: :Performance -PERFORMING-> :Song
CREATE (SoW:Performance {date: date('1959-03-02')})
CREATE (FreddieF:Performance {date: date('1959-03-02')})
CREATE (AllB:Performance {date: date('1959-03-02')})
CREATE (Bluei:Performance {date: date('1959-04-22')})
CREATE (FlamencoS:Performance {date: date('1959-04-22')})

// Album
CREATE (KindOfBlue:Recording {title: 'Kind of Blue'})

// So What?
CREATE (BillE) -[:PERFORMS {instrument: 'piano'}]-> (SoW)
CREATE (JohnC) -[:PERFORMS {instrument: 'tenor saxophone'}]-> (SoW)
CREATE (CannonballA) -[:PERFORMS {instrument: 'alto saxophone'}]-> (SoW)
CREATE (MilesD) -[:PERFORMS {instrument: 'trumpet'}]-> (SoW)
CREATE (PaulC) -[:PERFORMS {instrument: 'bass'}]-> (SoW)
CREATE (JimmyC) -[:PERFORMS {instrument: 'drums'}]-> (SoW)
CREATE (SoW)-[:PERFORMING]->(SoWSong)
CREATE (KindOfBlue) -[:RECORDS {trackNumber: 1}]-> (SoW)

// Freddie Freeloader
CREATE (WyntonK) -[:PERFORMS {instrument: 'piano'}]-> (FreddieF)
CREATE (CannonballA) -[:PERFORMS {instrument: 'alto saxophone'}]-> (FreddieF)
CREATE (JohnC) -[:PERFORMS {instrument: 'tenor saxophone'}]-> (FreddieF)
CREATE (MilesD) -[:PERFORMS {instrument: 'trumpet'}]-> (FreddieF)
CREATE (JimmyC) -[:PERFORMS {instrument: 'drums'}]-> (FreddieF)
CREATE (PaulC) -[:PERFORMS {instrument: 'bass'}]-> (FreddieF)
CREATE (FreddieF)-[:PERFORMING]->(FreddieFSong)
CREATE (KindOfBlue) -[:RECORDS {trackNumber: 2}]-> (FreddieF)

// Blue in Green
CREATE (BillE) -[:PERFORMS {instrument: 'piano'}]-> (Bluei)
CREATE (JohnC) -[:PERFORMS {instrument: 'tenor saxophone'}]-> (Bluei)
// Adderley did not perform in Blue in Green.
CREATE (MilesD) -[:PERFORMS {instrument: 'trumpet'}]-> (Bluei)
CREATE (PaulC) -[:PERFORMS {instrument: 'bass'}]-> (Bluei)
CREATE (JimmyC) -[:PERFORMS {instrument: 'drums'}]-> (Bluei)
CREATE (Bluei)-[:PERFORMING]->(BlueiSong)
CREATE (KindOfBlue) -[:RECORDS {trackNumber: 3}]-> (Bluei)

// All Blues
CREATE (BillE) -[:PERFORMS {instrument: 'piano'}]-> (AllB)
CREATE (CannonballA) -[:PERFORMS {instrument: 'alto saxophone'}]-> (AllB)
CREATE (JohnC) -[:PERFORMS {instrument: 'tenor saxophone'}]-> (AllB)
CREATE (MilesD) -[:PERFORMS {instrument: 'trumpet'}]-> (AllB)
CREATE (PaulC) -[:PERFORMS {instrument: 'bass'}]-> (AllB)
CREATE (JimmyC) -[:PERFORMS {instrument: 'drums'}]-> (AllB)
CREATE (AllB)-[:PERFORMING]->(AllBSong)
CREATE (KindOfBlue) -[:RECORDS {trackNumber: 4}]-> (AllB)

// Flamenco Sketches
CREATE (PaulC) -[:PERFORMS {instrument: 'bass'}]-> (FlamencoS)
CREATE (CannonballA) -[:PERFORMS {instrument: 'alto saxophone'}]-> (FlamencoS)
CREATE (JohnC) -[:PERFORMS {instrument: 'tenor saxophone'}]-> (FlamencoS)
CREATE (MilesD) -[:PERFORMS {instrument: 'trumpet'}]-> (FlamencoS)
CREATE (BillE) -[:PERFORMS {instrument: 'piano'}]-> (FlamencoS)
CREATE (JimmyC) -[:PERFORMS {instrument: 'drums'}]-> (FlamencoS)
CREATE (FlamencoS)-[:PERFORMING]->(FlamencoSSong)
CREATE (KindOfBlue) -[:RECORDS {trackNumber: 5}]-> (FlamencoS)


// A Love Supreme
// Album
CREATE (ALoveSupreme:Recording {title: 'A Love Supreme', releaseDate: date('1965-01-12'), label: 'Impulse!'})

// Songs (the four-part suite)
CREATE (Acknowledgement:Song {title: 'Acknowledgement'})
CREATE (Resolution:Song {title: 'Resolution'})
CREATE (Pursuance:Song {title: 'Pursuance'})
CREATE (Psalm:Song {title: 'Psalm'})

// Musicians
CREATE (McCoyT:Musician {name: 'McCoy Tyner', instrument: 'piano'})
CREATE (JimmyG:Musician {name: 'Jimmy Garrison', instrument: 'bass'})
CREATE (ElvinJ:Musician {name: 'Elvin Jones', instrument: 'drums'})

// Performances
CREATE (AckPerf:Performance {date: date('1964-12-09')})
CREATE (ResPerf:Performance {date: date('1964-12-09')})
CREATE (PurPerf:Performance {date: date('1964-12-09')})
CREATE (PsaPerf:Performance {date: date('1964-12-09')})

// Acknowledgement
CREATE (JohnC)-[:PERFORMS {instrument: 'tenor saxophone', role: 'leader'}]->(AckPerf)
CREATE (McCoyT)-[:PERFORMS {instrument: 'piano'}]->(AckPerf)
CREATE (JimmyG)-[:PERFORMS {instrument: 'bass'}]->(AckPerf)
CREATE (ElvinJ)-[:PERFORMS {instrument: 'drums'}]->(AckPerf)
CREATE (AckPerf)-[:PERFORMING]->(Acknowledgement)
CREATE (JohnC)-[:COMPOSES]->(Acknowledgement)
CREATE (ALoveSupreme)-[:RECORDS {trackNumber: 1}]->(AckPerf)

// Resolution
CREATE (JohnC)-[:PERFORMS {instrument: 'tenor saxophone', role: 'leader'}]->(ResPerf)
CREATE (McCoyT)-[:PERFORMS {instrument: 'piano'}]->(ResPerf)
CREATE (JimmyG)-[:PERFORMS {instrument: 'bass'}]->(ResPerf)
CREATE (ElvinJ)-[:PERFORMS {instrument: 'drums'}]->(ResPerf)
CREATE (ResPerf)-[:PERFORMING]->(Resolution)
CREATE (JohnC)-[:COMPOSES]->(Resolution)
CREATE (ALoveSupreme)-[:RECORDS {trackNumber: 2}]->(ResPerf)

// Pursuance
CREATE (JohnC)-[:PERFORMS {instrument: 'tenor saxophone', role: 'leader'}]->(PurPerf)
CREATE (McCoyT)-[:PERFORMS {instrument: 'piano'}]->(PurPerf)
CREATE (JimmyG)-[:PERFORMS {instrument: 'bass'}]->(PurPerf)
CREATE (ElvinJ)-[:PERFORMS {instrument: 'drums'}]->(PurPerf)
CREATE (PurPerf)-[:PERFORMING]->(Pursuance)
CREATE (JohnC)-[:COMPOSES]->(Pursuance)
CREATE (ALoveSupreme)-[:RECORDS {trackNumber: 3}]->(PurPerf)

// Psalm
CREATE (JohnC)-[:PERFORMS {instrument: 'tenor saxophone', role: 'leader'}]->(PsaPerf)
CREATE (McCoyT)-[:PERFORMS {instrument: 'piano'}]->(PsaPerf)
CREATE (JimmyG)-[:PERFORMS {instrument: 'bass'}]->(PsaPerf)
CREATE (ElvinJ)-[:PERFORMS {instrument: 'drums'}]->(PsaPerf)
CREATE (PsaPerf)-[:PERFORMING]->(Psalm)
CREATE (JohnC)-[:COMPOSES]->(Psalm)
CREATE (ALoveSupreme)-[:RECORDS {trackNumber: 4}]->(PsaPerf)
