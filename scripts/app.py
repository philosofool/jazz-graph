import gradio as gr
import pandas as pd
from jazz_graph.clean.data_normalization import normalize_title
from jazz_graph.recommendation.recommender import InferenceRecommender, LookupRecordings

class LookupInput:
    def __init__(self, recording_traits: pd.DataFrame):
        # recording traits should be indexed on recording_id

        self.recording_traits = recording_traits.copy()
        self.recording_traits['norm_album'] = self.recording_traits.album.apply(normalize_title)
        self.recording_traits['norm_song'] = self.recording_traits.title.apply(normalize_title)
        self.recording_traits['norm_artist'] = self.recording_traits.artist.apply(normalize_title)

    def match_recordings(self, record: dict) -> pd.DataFrame:
        fields = 'album', 'artist', 'song'
        df = self.recording_traits
        for field in fields:
            data = normalize_title(record.get(field))
            if field is None:
                continue
            norm_field = 'norm_' + field
            df = df[df[norm_field] == data]
        if len(df) == len(self.recording_traits):
            ...
        return df

class Recommend:
    def __init__(self, recommender: InferenceRecommender, recording_traits: pd.DataFrame):
        self.lookup = LookupInput(recording_traits)
        self.recording_traits = self.lookup.recording_traits[['artist', 'title', 'album', 'release_year']]
        self.recommender = recommender


    def recommend(self, artist, song, album):
        record = {'artist': artist, 'song': song, 'album': album}
        df = self.lookup.match_recordings(record)
        # TODO: handle no matches, all matches.
        if len(df) == len(self.lookup.recording_traits):
            ...
        elif len(df) == 0:
            ...
        rec_ids, *_ = self.recommender.get_recommendations(list(df.index))
        return self.recording_traits.loc[rec_ids]

demo = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Textbox(label="artist", info="Name of the artist who published the perfromance. Does not currently support filters by sidemen, i.e., use 'Miles Davis' for Kind of Blue, not 'Bill Evans'."),
        gr.Textbox(label="song", info="Name of the song. Many jazz songs are performed by several artists. For best results, also include artist or album name."),
        gr.Textbox(label="album", info="Name of the album the song was first on."),
    ],
    outputs=["text"],
    examples=[
        ['Joe Henderson', 'Isotope', 'Inner Urge'],
        ['Miles Davis', 'So What', 'Kind of Blue'],
    ],
    title="JazzGraph",
    description="Generate Jazz musical recommendations.",
    api_name="jazz_graph"
)

if __name__ == '__main__':
    demo.launch()