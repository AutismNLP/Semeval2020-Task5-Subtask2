import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from crf import viterbi_decode
from model_bert_crf import BertCrfTagger
import json
from argparse import ArgumentParser
from bio_datasetyw import build_dataset_for_bert_crf, NUM_TAGS, INDEX_TO_TAG
import collections

def main(option):
    with open('roberta-vocab.txt') as f:
        vocab = f.read()
        vocab = json.loads(vocab)
        vocab = collections.OrderedDict(vocab)
        token_to_id = vocab
    #token_to_id = BertTokenizer.from_pretrained(option.bert_vocab).vocab
    raw_dataset = json.load(open(option.dataset, 'r'))
    dataset = build_dataset_for_bert_crf(raw_dataset, token_to_id, device=option.device)
    data_loader = DataLoader(dataset, batch_size=option.batch_size, shuffle=False)
    num_tags = NUM_TAGS + 2
    start_tag_idx = num_tags - 2
    end_tag_idx = num_tags - 1
    model = BertCrfTagger(option.encoded_size, num_tags, start_tag_idx, end_tag_idx)
    model = model.to(option.device)
    model = torch.nn.DataParallel(model)
    state_dict = torch.load(option.model, map_location=option.device)
    model.load_state_dict(state_dict)
    model.eval()

    result = raw_dataset
    for batch in data_loader:
        tokens, _, _, lengths = batch
        lengths, sorted_indices = lengths.sort(descending=True)
        tokens = tokens.index_select(dim=0, index=sorted_indices)
        _, sorted_indices_reverse = sorted_indices.sort(descending=False)
        with torch.no_grad():
            scores, _ = model(tokens, lengths)
        preds = viterbi_decode(scores, lengths, start_tag_idx, end_tag_idx)
        preds = preds.index_select(dim=0, index=sorted_indices_reverse)

        batch_size = tokens.size(0)
        raw_samples = raw_dataset[:batch_size]
        raw_dataset = raw_dataset[batch_size:]

        for raw_sample, pred in zip(raw_samples, preds):
            tags = []
            for i in range(len(raw_sample['tokens'])):
                if pred[i] == 6:
                    pred[i] = 5 
                tags.append(INDEX_TO_TAG[pred[i].item()])
                raw_sample['tags'] = tags

    json.dump(result, open(option.output_file, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='../data/dev1.json')
    parser.add_argument('--model', type=str, default='models/e1.pt')
    parser.add_argument('--bert_vocab', type=str, default='roberta-vocab.txt')
    parser.add_argument('--bert_model', type=str, default='/data/huggingface_transformers/roberta-base')
    parser.add_argument('--encoded_size', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_file', type=str, default='output/e1.json')
    option = parser.parse_args()

    main(option)
