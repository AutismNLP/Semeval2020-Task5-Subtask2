import torch


def lengths_to_masks(lengths, total_length=None):
    # lengths: LongTensor, (batch)
    # total_length: int|None
    if total_length is None:
        total_length = lengths.max().item()
    """
    tensor([[  0,   1,   2,  ..., 149, 150, 151],
        [  0,   1,   2,  ..., 149, 150, 151],
        [  0,   1,   2,  ..., 149, 150, 151],
        ...,
        [  0,   1,   2,  ..., 149, 150, 151],
        [  0,   1,   2,  ..., 149, 150, 151],
        [  0,   1,   2,  ..., 149, 150, 151]])
    tensor([[36, 36, 36,  ..., 36, 36, 36],
        [62, 62, 62,  ..., 62, 62, 62],
        [58, 58, 58,  ..., 58, 58, 58],
        ...,
        [81, 81, 81,  ..., 81, 81, 81],
        [44, 44, 44,  ..., 44, 44, 44],
        [21, 21, 21,  ..., 21, 21, 21]])

    """
    masks = torch.arange(total_length, device=lengths.device).expand(lengths.size(0), -1).lt(lengths.view(-1, 1).expand(-1, total_length))
    return masks


def token_acc(pred, gold, token_pad_idx=-1):
    # pred: LongTensor, (batch x len)
    # gold: LongTensor, (batch x len)
    # token_pad_idx: int
    masks = gold.ne(token_pad_idx)
    num_correct = pred.eq(gold).masked_select(masks).sum().item()
    num_total = masks.sum().item()
    return num_correct / num_total


def array_equal(array1, array2):
    if len(array1) != len(array2):
        return False
    for item1, item2 in zip(array1, array2):
        if item1 != item2:
            return False
    return True
