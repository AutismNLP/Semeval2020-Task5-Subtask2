import torch
import torch.nn as nn
from utils import lengths_to_masks


class CRF(nn.Module):
    def __init__(self, input_size, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.emission = nn.Linear(input_size, num_tags)
        self.transition = nn.Parameter(torch.FloatTensor(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        self.transition.data.zero_()

    def forward(self, input):
        # input: FloatTensor, (batch x len x input_size)
        num_tags = self.num_tags
        emission_scores = self.emission(input)                                  # (batch, len, num_tags)
        emission_scores = emission_scores.unsqueeze(2)                          # (batch, len, 1, num_tags)

        # emission:   (batch, len, 1       , num_tags)
        # transition: (            num_tags, num_tags)
        # broadcasting is automatically done
        scores = emission_scores + self.transition

        # e.g.
        # emission (one timestep):
        # e1, e2, e3, e4, e5
        # after broadcast:
        # e1, e2, e3, e4, e5
        # e1, e2, e3, e4, e5
        # e1, e2, e3, e4, e5
        # e1, e2, e3, e4, e5
        # e1, e2, e3, e4, e5
        # transition:
        # t11, t12, t13, t14, t15
        # t21, t22, t23, t24, t25
        # t31, t32, t33, t34, t35
        # t41, t42, t43, t44, t45
        # t51, t52, t53, t54, t55
        # sum:
        # e1+t11, e2+t12, e3+t13, e4+t14, e5+t15
        # e1+t21, e2+t22, e3+t23, e4+t24, e5+t25
        # e1+t31, e2+t32, e3+t33, e4+t34, e5+t35
        # e1+t41, e2+t42, e3+t43, e4+t44, e5+t34
        # e1+t51, e2+t52, e3+t53, e4+t54, e5+t55
        # row (dim 2 in scores): previous tag
        # col (dim 3 in scores): current tag
        return scores


def log_sum_exp(tensor, dim):
    m, _ = torch.max(tensor, dim=dim)
    return m + torch.log(torch.sum(torch.exp(tensor - m.unsqueeze(dim)), dim=dim))


class ViterbiLoss(nn.Module):
    def __init__(self, start_tag_idx, end_tag_idx, reduction='mean'):
        super(ViterbiLoss, self).__init__()
        assert reduction in ['none', 'mean']
        self.start_tag_idx = start_tag_idx
        self.end_tag_idx = end_tag_idx
        self.reduction = reduction

    def forward(self, scores, targets, lengths):
        # scores: FloatTensor, (batch x len x num_tags x num_tags), where dim 2 is the previous tag, dim 3 is the current tag
        # targets: LongTensor, (batch x len), where value for each timestep is (previous_tag * num_tags + current_tag)
        # lengths: LongTensor, (batch)
        # both scores, targets, lengths should be sorted in descending order with regard to length

        # TODO move to preprocess
        # e.g.
        # 1    [START, 1]
        # 3    [1, 3]
        # 2 -> [3, 2]
        # 5    [2, 5]
        # 4    [5, 4]
        # END  [4, END]

        batch_size, total_len, num_tags, _ = scores.size()

        flattened_scores = scores.view(batch_size, total_len, -1)                           # (batch, len, num_tags*num_tags)
        targets = targets.unsqueeze(2)                                                      # (batch, len, 1)
        scores_at_targets = torch.gather(input=flattened_scores, dim=2, index=targets)      # (batch, len, 1)
        scores_at_targets = scores_at_targets.squeeze(2)                                    # (batch, len)
        masks = lengths_to_masks(lengths, total_length=total_len).float()                   # (batch, len)
        gold_score = (scores_at_targets * masks).sum(dim=1)                                 # (batch)
        # according to https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling, there is not need to calculate log-exp for gold score

        scores_upto_t = torch.zeros(batch_size, num_tags).to(scores.device)                 # (batch, num_tags)
        for t in range(total_len):
            batch_size_t = lengths.gt(t).sum().item()
            if batch_size_t == 0:
                break
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag_idx, :]  # (batch, num_tags)
            else:
                scores_upto_t[:batch_size_t] = log_sum_exp(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1
                )
        all_path_scores = scores_upto_t[:, self.end_tag_idx]    # (batch)

        loss = all_path_scores - gold_score                     # (batch)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


def viterbi_decode(scores, lengths, start_tag_idx, end_tag_idx):
    # scores: FloatTensor, (batch x len x num_tags x num_tags)
    # start_tag_idx: int
    # end_tag_idx: int
    device = scores.device
    batch_size, total_len, num_tags, _ = scores.size()

    scores_upto_t = torch.zeros(batch_size, num_tags, device=device)
    backpointers = torch.ones(batch_size, total_len, num_tags, dtype=torch.long, device=device) * end_tag_idx
    for t in range(total_len):
        batch_size_t = lengths.gt(t).sum().item()
        if batch_size_t == 0:
            break
        if t == 0:
            scores_upto_t[:batch_size_t, :] = scores[:batch_size_t, t, start_tag_idx, :]
            backpointers[:batch_size_t, t] = torch.ones(batch_size_t, num_tags, dtype=torch.long, device=device) * start_tag_idx
        else:
            scores_upto_t[:batch_size_t, :], backpointers[:batch_size_t, t, :] = torch.max(
                scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t, :].unsqueeze(2),
                dim=1
            )

    decoded = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
    pointer = torch.ones(batch_size, 1, dtype=torch.long, device=device) * end_tag_idx
    for t in reversed(range(total_len)):
        decoded[:, t] = torch.gather(backpointers[:, t, :], dim=1, index=pointer).squeeze(1)
        pointer = decoded[:, t].unsqueeze(1)

    # sanity check
    assert decoded[:, 0].equal(torch.ones(batch_size, dtype=torch.long, device=device) * start_tag_idx)

    # remove START at the beginning, append END
    decoded = torch.cat(
        [decoded[:, 1:], torch.ones(batch_size, 1, dtype=torch.long, device=device) * end_tag_idx],
        dim=1
    )
    return decoded
