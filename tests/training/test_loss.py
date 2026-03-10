import torch
from jazz_graph.training.loss import nt_xent_loss_with_masking

def loop_masked_nt_xent(z, postive_mask, temperature):
    """This is the correct algorithm: use to double check vectorized versions."""
    sim_matrix = torch.mm(z, z.t()) / temperature
    self_mask = torch.eye(4, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(self_mask, -float('inf'))
    # print(sim_matrix)
    losses = []
    for i in range(4):
        pos_mask = postive_mask[i] & ~self_mask[i]
        if pos_mask.sum() == 0:
            continue
        pos_sims = sim_matrix[i][pos_mask]
        all_sims = sim_matrix[i][~self_mask[i]]
        pos_exp_sum = torch.logsumexp(pos_sims, dim=0)
        all_exp_sum = torch.logsumexp(all_sims, dim=0)
        loss_i = -pos_exp_sum + all_exp_sum
        losses.append(loss_i)
    if not losses:
        return torch.tensor(0.)
    return torch.stack(losses).mean()


def test_nx_xent_with_masking():
    z = (torch.arange(1, 17) / 10).reshape(-1, 4)
    mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 1]
    ], dtype=torch.bool)
    temperature = .5
    result = nt_xent_loss_with_masking(z, mask, temperature)
    assert torch.abs(result) != torch.tensor(float('inf')), "The result should not be infite; otherwise suggests code bug in masking."
    assert not torch.isnan(result)
    assert result == loop_masked_nt_xent(z, mask, temperature)

    mask = torch.eye(4, dtype=torch.bool)
    result = nt_xent_loss_with_masking(z, mask, .5)
    print(result)
    assert torch.abs(result) != torch.tensor(float('inf'))
    assert not torch.isnan(result)
    assert result == 0., "The cost should be 0 if there are not inter-group matches."

if __name__ == '__main__':
    z = (torch.arange(1, 17) / 10).reshape(-1, 4)
    postive_mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 1]
    ], dtype=torch.bool)
    temperature = .5
    loop_masked_nt_xent(z, postive_mask, .5)
    loop_masked_nt_xent(z, torch.eye(4, dtype=torch.bool), .5)


    test_nx_xent_with_masking()