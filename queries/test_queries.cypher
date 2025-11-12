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

// Genius of Modern Music, Volume 1
// Album
CREATE (GeniusVol1:Recording {
  title: 'Genius of Modern Music, Volume 1',
  releaseDate: date('1951-09-01'),
  label: 'Blue Note'
})

// Songs
CREATE (RoundMidnight:Song {title: "'Round Midnight"})
CREATE (OffMinor:Song {title: 'Off Minor'})
CREATE (RubyMyDear:Song {title: 'Ruby, My Dear'})
CREATE (IShouldCare:Song {title: 'I Should Care'})
CREATE (AprilInParis:Song {title: 'April in Paris'})

// Musicians
CREATE (TheloMonk:Musician {name: 'Thelonious Monk', primary_instrument: 'piano'})
CREATE (IdreesS:Musician {name: 'Idrees Sulieman', primary_instrument: 'trumpet'})
CREATE (DannyQ:Musician {name: 'Danny Quebec West', primary_instrument: 'alto saxophone'})
CREATE (BillyS:Musician {name: 'Billy Smith', primary_instrument: 'alto saxophone'})
CREATE (GeneLR:Musician {name: 'Gene Ramey', primary_instrument: 'bass'})
CREATE (ArtB:Musician {name: 'Art Blakey', primary_instrument: 'drums'})
CREATE (MiltJ:Musician {name: 'Milt Jackson', primary_instrument: 'vibraphone' })
CREATE (JohnS:Musician {name: 'John Simmons', primary_instrument: 'bass'})
CREATE (ShadowW:Musician {name: 'Shadow Wilson', primary_instrument: 'drums'})

// Performances (sessions from 1947-1948)
CREATE (RMPerf:Performance {date: date('1947-10-24')})
CREATE (OMPerf:Performance {date: date('1947-10-24')})
CREATE (RMDPerf:Performance {date: date('1947-10-24')})
CREATE (ISCPerf:Performance {date: date('1948-07-02')})
CREATE (AIPPerf:Performance {date: date('1948-07-02')})

// 'Round Midnight (October 24, 1947)
CREATE (TheloMonk)-[:PERFORMS {instrument: 'piano', role: 'leader'}]->(RMPerf)
CREATE (IdreesS)-[:PERFORMS {instrument: 'trumpet'}]->(RMPerf)
CREATE (DannyQ)-[:PERFORMS {instrument: 'alto saxophone'}]->(RMPerf)
CREATE (GeneLR)-[:PERFORMS {instrument: 'bass'}]->(RMPerf)
CREATE (ArtB)-[:PERFORMS {instrument: 'drums'}]->(RMPerf)
CREATE (RMPerf)-[:PERFORMING]->(RoundMidnight)
CREATE (TheloMonk)-[:COMPOSES]->(RoundMidnight)
CREATE (GeniusVol1)-[:RECORDS {trackNumber: 20, title: "'Round Midnight"}]->(RMPerf)

// Off Minor (October 24, 1947)
CREATE (TheloMonk)-[:PERFORMS {instrument: 'piano', role: 'leader'}]->(OMPerf)
CREATE (IdreesS)-[:PERFORMS {instrument: 'trumpet'}]->(OMPerf)
CREATE (DannyQ)-[:PERFORMS {instrument: 'alto saxophone'}]->(OMPerf)
CREATE (GeneLR)-[:PERFORMS {instrument: 'bass'}]->(OMPerf)
CREATE (ArtB)-[:PERFORMS {instrument: 'drums'}]->(OMPerf)
CREATE (OMPerf)-[:PERFORMING]->(OffMinor)
CREATE (TheloMonk)-[:COMPOSES]->(OffMinor)
CREATE (GeniusVol1)-[:RECORDS {trackNumber: 15, title: 'Off Minor'}]->(OMPerf)

// Ruby, My Dear (October 24, 1947)
CREATE (TheloMonk)-[:PERFORMS {instrument: 'piano', role: 'leader'}]->(RMDPerf)
CREATE (IdreesS)-[:PERFORMS {instrument: 'trumpet'}]->(RMDPerf)
CREATE (DannyQ)-[:PERFORMS {instrument: 'alto saxophone'}]->(RMDPerf)
CREATE (GeneLR)-[:PERFORMS {instrument: 'bass'}]->(RMDPerf)
CREATE (ArtB)-[:PERFORMS {instrument: 'drums'}]->(RMDPerf)
CREATE (RMDPerf)-[:PERFORMING]->(RubyMyDear)
CREATE (TheloMonk)-[:COMPOSES]->(RubyMyDear)
CREATE (GeniusVol1)-[:RECORDS {trackNumber: 10, title: 'Ruby, My Dear'}]->(RMDPerf)

// I Should Care (July 2, 1948 - different personnel)
CREATE (TheloMonk)-[:PERFORMS {instrument: 'piano', role: 'leader'}]->(ISCPerf)
CREATE (BillyS)-[:PERFORMS {instrument: 'alto saxophone'}]->(ISCPerf)
CREATE (GeneLR)-[:PERFORMS {instrument: 'bass'}]->(ISCPerf)
CREATE (ArtB)-[:PERFORMS {instrument: 'drums'}]->(ISCPerf)
CREATE (ISCPerf)-[:PERFORMING]->(IShouldCare)
// I Should Care composed by Sammy Cahn, Axel Stordahl, Paul Weston (not Monk)
CREATE (GeniusVol1)-[:RECORDS {title: 'I Should Care'}]->(ISCPerf)

// April in Paris (July 2, 1948)
CREATE (TheloMonk)-[:PERFORMS {instrument: 'piano', role: 'leader'}]->(AIPPerf)
CREATE (BillyS)-[:PERFORMS {instrument: 'alto saxophone'}]->(AIPPerf)
CREATE (GeneLR)-[:PERFORMS {instrument: 'bass'}]->(AIPPerf)
CREATE (ArtB)-[:PERFORMS {instrument: 'drums'}]->(AIPPerf)
CREATE (AIPPerf)-[:PERFORMING]->(AprilInParis)
// April in Paris composed by Vernon Duke (not Monk)
CREATE (GeniusVol1)-[:RECORDS {trackNumber: 5, title: 'April in Paris'}]->(AIPPerf)

// // I Mean You (July 1948)
// Annoying data issue: this was reissued on a Milt Jackson anthology...
// CREATE (IMeanYou:Song {title: "I Mean You"})
// CREATE (IMeanYouPerf:Performance)
// CREATE (ColemanH:Musician {name: 'Coleman Hawkins'})

// CREATE (TheloMonk) -[:COMPOSES]-> (IMeanYou)
// CREATE (ColemanH) -[:COMPOSES]-> (IMeanYou)
// CREATE (TheloMonk)-[:PERFORMS {instrument: 'piano', role: 'leader'}]->(IMeanYouPerf)
// CREATE (MiltJ)-[:PERFORMS {instrument: 'vibraphone'}]->(IMeanYouPerf)
// CREATE (JohnS)-[:PERFORMS {instrument: 'bass'}]->(IMeanYouPerf)
// CREATE (ShadowW)-[:PERFORMS {instrument: 'drums'}]->(IMeanYouPerf)
// CREATE (IMeanYouPerf)-[:PERFORMING]->(IMeanYou)

// CREATE (GeniusVol1)-[:RECORDS {trackNumber: 5, title: 'I Mean You'}]->(IMeanYouPerf)
