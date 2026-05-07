import torch

from jazz_graph.data.fetch import fetch_recording_traits

def analyze_batch_positives(batch, positive_mask):
    """How many positives per sample in this batch?"""
    # I had Claude AI write this.
    batch_size = positive_mask.size(0)
    mask_no_diag = positive_mask & ~torch.eye(batch_size, dtype=torch.bool)

    positives_per_sample = mask_no_diag.sum(dim=1)
    zero_positives = (positives_per_sample == 0).sum().item()

    print(f"Batch size: {batch_size}")
    print(f"Samples with 0 positives: {zero_positives} ({100 * zero_positives / batch_size:.1f}%)")
    print(f"Samples with 1+ positives: {(positives_per_sample > 0).sum().item()}")
    print(f"Mean positives per sample: {positives_per_sample.float().mean():.2f}")
    print(f"Max positives per sample: {positives_per_sample.max().item()}")

# Check if loss is dominated by easy negatives
def analyze_negative_difficulty(z, positive_mask, temperature=0.5):
    """Are negatives too easy?"""
    # I had Claude AI write this.
    sim_matrix = torch.mm(z, z.t()) / temperature

    # Positive similarities
    pos_sims = sim_matrix[positive_mask & ~torch.eye(len(z), dtype=torch.bool)]

    # Negative similarities
    neg_sims = sim_matrix[~positive_mask & ~torch.eye(len(z), dtype=torch.bool)]

    print(f"Positive similarities: mean={pos_sims.mean():.3f}, std={pos_sims.std():.3f}")
    print(f"Negative similarities: mean={neg_sims.mean():.3f}, std={neg_sims.std():.3f}")
    print(f"Gap: {pos_sims.mean() - neg_sims.mean():.3f}")


def analyze_model_embeddings(model, graph_data):
    album_experiments = [
        ('Miles Davis', 'Kind of Blue'),
        ('Miles Davis', 'Sketches of Spain'),
        ('Art Blakey & The Jazz Messengers', 'Mosaic'),
        ('Charles Mingus', "Mingus Ah Um"),  # lots of songs, should have some.
        ('The Dave Brubeck Quartet', "Time Out"),
        ('Ornette Coleman', 'The Shape of Jazz to Come')  # very unusual music--should probably be easy.
    ]
    recording_traits = fetch_recording_traits(use_proto=True).set_index('recording_id')

    def zero_diag(tensor):
        return tensor * (torch.ones_like(tensor) - torch.eye(tensor.size(0)))

    for artist, album in album_experiments:
        ids = recording_traits.query(f"artist == '{artist}'").query(f"album == '{album}'").index.to_numpy()
        mask = torch.isin(graph_data['performance'].x[:, 1], torch.tensor(ids))

        with torch.no_grad():
            performance_encoded = model(graph_data)['performance']
        expertiment_encoding = performance_encoded[mask]
        rand_selection = torch.randint(0, mask.size(0), (5,))
        other = performance_encoded[rand_selection]
        inner_album_similarity = zero_diag(expertiment_encoding @ expertiment_encoding.T).sum() / (expertiment_encoding.size(0) ** 2 - expertiment_encoding.size(0))
        print(f'{artist}, "{album}"')
        print(f"Mean inner-album similarity: {inner_album_similarity.item():.3f}")
        print(f"Mean random sample similarity: {(expertiment_encoding @ other.T).mean().item():.3f}")
        print(expertiment_encoding @ expertiment_encoding.T)
