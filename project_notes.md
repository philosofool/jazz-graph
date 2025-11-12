# Project Notes

The following notes are kept largely as WIP thoughts for project development.
Currently, there is just one person working on the project, so full management
software or even GitHub tools are unnecessary.

No warranties exist regarding the plans here.
The README is documentation of the project as concepts reach beta stage and we move 0 to 1.


## Jazz/Music Data Model

Some entities:
performances (recordings, events), works (songs, albums, publications), people (artists, composers, producers), instruments, labels, places (Blue Note, Smalls, Village Vanguard, New Orleans)

Questions:
Who played with John Coltrane?
Which tenors saxophonists played with Miles Davis?
How many albums did Ron Cater record?
Who are the artists that have played "So What?"
Which songs composed by Bill Evans have been recorded the most?
Which artists did Miles Davis play with before recording Kind of Blue?
Who all the pianists that played with someone who also played with McCoy Tyner?
What instruments did Eric Dolphy play on Point of Departure?
Which labels did Andrew Hill record with?
What songs were played at the Village Vanguard in 1961?
How many times has ... been reissued? (Also, filter duplicate/re-issued albums.)

Notes:
There's a good tutorial in Neo4j Desktop. Put `:welcome` into the query/search bar--I'm not sure which that is yet--
and it will show you the movies tutorial which has some good examples of basics.

```cypher

CREATE (:Musician {name: 'Bill Evans'})
```

## Fuzzy Test

Saturady, Nov. 8, 2025 I did a little exploration of Spotify recommendations.
I searched for "So What?" and selected the top result, which was "So What? (feat. John Coltrane...)"* which was, of course, the original recording (remastered, down sampled, etc.) from Kind of Blue.
What does Spotify follow with?
1. "So What"
2. "Giant Steps" Way obvious, very popular, less accessible the So What. C.
3. "Like It Is", Yusef Lateef (a 1968 recording from _The Blue Yusef Lateef_)--I didn't know this performance. This is an A recommendation; exploratory, should appeal to anyone who likes So What, etc.
4. "Circle", Miles Davis second Quintet (composer, Miles Davis) from _Miles Smiles_ 1967. Very modal, pretty obvious connection to seed song; it's more like Kind of Blue than either Sketches of Spain (1960) or Bitches Brew (1969). This is just avant guarde enough A-.
5. "Some Other Time" Bill Evans _Waltz for Debbie_.
6. "A Night in Tunisia" Jesus Molina, et al. _Departing_ A trio performance, 2020. (I didn't know this; would probably have guessed this as recent performance. Sounds a little Bad Plus plays Dizzy Gillespie.)
I don't want to say I didn't like this, but it was a little too rock out on a song that rocked before rock.
7. "Bright Size Life" Pat Metheny.
8. "I'll Wait and Pray" John Coltrane, _Coltrane Jazz_. Atlantic. (1961) Note on the album: it's several sessions with several musicians. A little bit of McCoy/Jones, Wynton Kelly/Chambers/etc. Composers were George Treadwell and Jerry Valentine.
9. "It Never Entered My Mind" Miles Davis. Spotify pull a complilation of Blue Note and Capitol Records recordings. This is presumably a 1954 recording with Blue Note (I'm seening a later recording with Prestige 1956) Originally a Sinatra recording.
10. "Move" Miles Davis, same compilation as 9! Originallyl _Birth of the Cool_ (1957).

Quick thoughts: this was a solid list of recommendations.
All good songs. It circled back a little too hard on Miles and double dipped Coltrane, with one of those selections completely uninspired. (Great song, but it's a peeve of mind to immediately get a top three most obvious song when requesting something by an artists. Like, if I say "Bob Dylan" and you reply with "Hurricane,"... sigh.) Overall, I would say this is good exploration of Miles' recordings, but that's actually not to hard (use a random number generator over Miles and you get an eclectic bunch of songs that are all really good. People who pan Miles generally do so to be snobs; Miles put together amazing musicians and all the above performances are great. )
How do I grade these recs? B? Better than C, but not B+. Underneath the misplaced elitism, people who pan Miles are right that _more Miles Davis_ is D tier exploration. Two Coltrane songs? Coltrane is the second most obvious choice of Jazz musician to show someone. Maybe Louis, Ella and Duke have similar name recognition, but he's a top five for sure and played with Miles a lot. So, I get some lesser known performances by some better known artists and a few more exploratory artist choices. Lateef is easily the best real exploration here; the Bill Evans pick is A tier for a Jazz novice and A tier in any Jazz play list, but it's like C tier exploration for the moderately schooled Jazz fan.

Now that I'm here, what are the grades?
A: Wow. This is just an amazing collection of songs, balance the feel of the seed with novelty, picking novel choices by familiar names, on-theme selections from novel sources and generally making me want to listen to more.
B: Good songs with moderate exploration, but too much reliance on known quantities or effects of popularity or repetition of the seed.
C: Lots of reliance on popularity, repeating the seed, but at least stays in the domain of the seed or the recommender's preference.
D: Recommender effect exists, but sometimes missing, where it's not clear how one would draw the line, except like "People like 'So What' and they like Bob Dylan, so you probably like Bob Dylan because you are a person."*
F: No obvious recommender effect. Random generation weighted by pure popularity.

### Round 2

1. Bill Withers, "Ain't No Sunshine" (1971)
2. Carole King, "It's Too Late" (1971)
3. Bill Withers, "Lean on Me" (1972)
(Predicting Stand By Me, Ben E. King) before this ends.
4. Harvest Moon, Neil Young (1992)
5. Paul Simon, "50 Ways to Leave your Lover" (1975)
6. "Sara Smile" Daryl Hall and John Oates (1975)
7. "Hopelessly Hoping" Crosby, Stills and Nash (no Young! 1969)*
8. Bill Withers, "Lovely Day" (1977)
(This is your Stand by Me chance, spotify!)
9. Fleatwood Mac, "Dreams"
(I'm not sure if I am wrong or Spotify is...)
10. "Old Man," Neil Young

This is a C. Seed is Bill Withers, and it started because my girlfriend has just left for a weekend trip to see some friends.
So, to sum this up, we have two things (1) more Bill Withers and (2) other music I like.
The line through Carole King from Withers to Neil Young is, in a word, overfitted.
Are we even trying not to just... pick songs that I already know? "Sara Smile" was new to me.
But I didn't think it was all that good. Everything else is just sort of adjacent because it is in the popular region of the musical graph and I've played it or the artist in question.
I like 90% of all the songs on here, and I love a few of them, but it doesn't feel novel.
Much less than the Jazz recommendations seeded by So What.


#### Notes
I hate it when spotify adds click bait to song title.
There's no song called "So What (feat. John Coltrane)"
You could easily just slim this down: "You probably like Bob Dylan because you are a person."
Universally recognized good songs and artists are not universally recognized recommendations.


## Recommendation

Our basic model is musicians, performances and songs.
Our goal is to recommend performances.
In an analogy with collaborative filters, performances are items.
Optimize Bayesian Personalized Ranking, treating performances and musicians as items and queries.
In addition to capturing performances and musicians, we also include songs in the graphs,
thereby allowing messages to pass through them.