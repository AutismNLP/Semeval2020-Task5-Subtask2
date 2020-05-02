import torch
from torch.utils.data import TensorDataset


TAG_TO_INDEX = {
    'B-cause': 1,
    'I-cause': 2,
    'B-effect': 3,
    'I-effect': 4,
    'O': 5,
    # 'M': 6
    # 'B-trigger': 5,
    # 'I-trigger': 6,
    # 'B-strength': 7,
    # 'I-strength': 8
}
INDEX_TO_TAG = { TAG_TO_INDEX[key] : key for key in TAG_TO_INDEX }
NUM_TAGS = len(TAG_TO_INDEX)


def build_dataset_for_bert(raw_dataset, token_to_id, tag_pad_idx=-1, total_length=None, device='cpu'):
    # raw_dataset: dict, dataset in json format
    # token_to_id: dict, token -> id
    all_tokens = []
    all_tags = []
    lengths = []
    for sample in raw_dataset:
        tokens = [token_to_id[token] if token in token_to_id else token_to_id['[UNK]'] for token in sample['tokens']]
        tags = [TAG_TO_INDEX[tag] for tag in sample['tags']]
        tokens = tokens[:(total_length-2)]
        tags = tags[:(total_length-2)]
        # add CLS and SEP
        tokens = [token_to_id['[CLS]']] + tokens + [token_to_id['[SEP]']]
        tags = [tag_pad_idx] + tags + [tag_pad_idx]     # the output for CLS and SEP will be ignored, because their ground-truth tag is PAD
        lengths.append(len(tokens))
        all_tokens.append(tokens)
        all_tags.append(tags)


    if total_length is None:
        total_length = max(lengths)

    assert total_length >= max(lengths)
    for tokens in all_tokens:
        while len(tokens) < total_length:
            tokens.append(token_to_id['[PAD]'])

    for tags in all_tags:
        while len(tags) < total_length:
            tags.append(tag_pad_idx)

    all_tokens = torch.LongTensor(all_tokens).to(device)
    all_tags = torch.LongTensor(all_tags).to(device)
    lengths = torch.LongTensor(lengths).to(device)
    dataset = TensorDataset(all_tokens, all_tags, lengths)
    # print(dataset[0])
    # exit(0)
    return dataset


def build_dataset_for_bert_crf(raw_dataset, token_to_id, total_length=None, device='cpu'):
    all_tokens = []
    all_tags = []
    all_crf_tags = []
    lengths = []

    for sample in raw_dataset:
        tokens = [token_to_id[token] if token in token_to_id else token_to_id["<unk>"] for token in sample["tokens"]]
        # tokens = tokens[:(total_length-2)]
        # add CLS and SEP
        tokens = [token_to_id['<s>']] + tokens + [token_to_id['</s>']]
        # adjust tags to CRF format
        # END aligns with [SEP], but no token aligns with [CLS]
        # len(tags) = len(tokens) - 1
        num_tags = NUM_TAGS + 2     # add START and END
        start_tag_idx = num_tags - 2
        end_tag_idx = num_tags - 1
        tags = [TAG_TO_INDEX[tag] for tag in sample['tags']]
        # tags = tags[:(total_length-2)]

        crf_tags = [start_tag_idx * num_tags + tags[0]]

        for i in range(len(tags) - 1):
            crf_tags.append(tags[i] * num_tags + tags[i + 1])
        crf_tags.append(tags[-1] * num_tags + end_tag_idx)
        # add to dataset
        lengths.append(len(tokens))
        all_tokens.append(tokens)
        all_tags.append(tags)
        all_crf_tags.append(crf_tags)

    if total_length is None:
        total_length = max(lengths)
    assert total_length >= max(lengths)
    for tokens in all_tokens:
        while len(tokens) < total_length:
            tokens.append(token_to_id['<pad>'])
    for tags in all_tags:
        while len(tags) < total_length - 2:
            tags.append(end_tag_idx)
    for crf_tags in all_crf_tags:
        while len(crf_tags) < total_length - 1:
            crf_tags.append(end_tag_idx * num_tags + end_tag_idx)

    all_tokens = torch.LongTensor(all_tokens).to(device)
    all_tags = torch.LongTensor(all_tags).to(device)
    all_crf_tags = torch.LongTensor(all_crf_tags).to(device)
    lengths = torch.LongTensor(lengths).to(device)
    dataset = TensorDataset(all_tokens, all_tags, all_crf_tags, lengths)
    return dataset
