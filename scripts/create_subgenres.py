from collections import Counter
import jsonlines
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from jazz_graph.extract_discogs import is_jazz_album


def get_subgenres(path_to_releases) -> tuple[list[int], list[list], Counter]:
    jazz_subgenres = list()
    ids = []
    style_counts = Counter([])
    with jsonlines.open(path_to_releases) as f:
        for entry in f:
            if is_jazz_album(entry):
                styles = entry.get('styles', [])
                if not styles:
                    styles.append('None Listed')
                jazz_subgenres.append(styles)
                style_counts.update(styles)
                ids.append(entry['id'])

    return ids, jazz_subgenres, style_counts


def create_subgenres_table(path_to_releases):
    ids, jazz_subgenres, style_counts = get_subgenres(path_to_releases)
    common_styles = [style for style, _ in style_counts.most_common(21) if style != 'None Listed']
    labeler = MultiLabelBinarizer(classes=common_styles)
    style_matrix = labeler.fit_transform(jazz_subgenres)
    styles = labeler.classes_
    style_data = pd.DataFrame(style_matrix, columns=styles, index=ids)   # pyright: ignore
    style_data.to_parquet('/workspace/local_data/discogs_styles.parquet')
    print(f"Data for discogs styles written.\nIncludes styles {common_styles}.")


if __name__ == '__main__':
    path_to_releases = '/workspace/local_data/jazz_releases.jsonl'
    create_subgenres_table(path_to_releases)
    df = pd.read_parquet('/workspace/local_data/discogs_styles.parquet')
    print(df.head())