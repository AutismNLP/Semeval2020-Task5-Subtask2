import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from pytorch_pretrained_bert import BertModel, BertTokenizer
import json
from argparse import ArgumentParser
from crf import CRF, ViterbiLoss, viterbi_decode
from bio_datasetyw import build_dataset_for_bert_crf, NUM_TAGS
from utils import lengths_to_masks, token_acc
from transformers import RobertaModel
import collections

class BertCrfTagger(nn.Module):
    def __init__(self, encoded_size, num_tags, start_tag_idx, end_tag_idx, loss_reduction='mean'):
        super(BertCrfTagger, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-large')
        self.crf = CRF(encoded_size, num_tags)
        self.criterion = ViterbiLoss(start_tag_idx, end_tag_idx, reduction=loss_reduction)

    def forward(self, tokens, lengths, crf_tags=None):
        # tokens: LongTensor, (batch x len), added CLS and SEP
        # lengths: LongTensor, (batch)
        # crf_tags: LongTensor, (batch x (len-1))
        # both tokens, lengths and crf_tags should be sorted in descending order by length

        token_type_ids = torch.zeros_like(tokens, device=tokens.device).long()          # (batch, len)
        attention_masks = lengths_to_masks(lengths, total_length=tokens.size(-1))       # (batch, len)
        encoded_layers, _ = self.roberta(tokens, token_type_ids=token_type_ids, attention_mask=attention_masks)   # (batch, len, encoded_size)
        # remove hidden for CLS
        features = encoded_layers[:, 1:, :]         # (batch, len-1, encoded_size)
        scores = self.crf(features)                 # (batch, len-1, num_tags, num_tags)
        # calc loss
        if crf_tags is None:
            loss = None
        else:
            loss = self.criterion(scores, crf_tags, lengths - 1)
        return scores, loss


def main(option):
    torch.manual_seed(option.random_seed)
    with open('roberta-vocab.txt') as f:
        vocab = f.read()
    vocab = json.loads(vocab)
    vocab = collections.OrderedDict(vocab)
    token_to_id = vocab
    raw_train_dataset = json.load(open(option.train_dataset, 'r'))
    #raw_dev_dataset = json.load(open(option.dev_dataset, 'r'))

    train_dataset = build_dataset_for_bert_crf(raw_train_dataset, token_to_id, option.max_seq_len, option.device)
    #dev_dataset = build_dataset_for_bert_crf(raw_dev_dataset, token_to_id, option.max_seq_len, option.device)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=option.batch_size, shuffle=True, drop_last=True)
    #dev_data_loader = DataLoader(dataset=dev_dataset, batch_size=option.batch_size, shuffle=True, drop_last=True)

    num_tags = NUM_TAGS + 2
    start_tag_idx = num_tags - 2
    end_tag_idx = num_tags - 1
    model = BertCrfTagger(option.encoded_size, num_tags, start_tag_idx, end_tag_idx, loss_reduction='none')
    model = model.to(option.device)
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=option.lr)

    dev_acc = 0
    for epoch in range(option.epochs):
        epoch += 1
        print('epoch %d' % (epoch,))

        print('train set')
        model.train()
        ttacc=0
        for i, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            tokens, tags, crf_tags, lengths = batch
            # sort by length in descending order
            lengths, sorted_indices = lengths.sort(descending=True)
            tokens = tokens.index_select(dim=0, index=sorted_indices)
            tags = tags.index_select(dim=0, index=sorted_indices)
            crf_tags = crf_tags.index_select(dim=0, index=sorted_indices)
            # forward
            scores, loss = model(tokens, lengths, crf_tags)
            loss = loss.mean()
            # backward
            loss.backward()
            optimizer.step()
            if i % option.train_report_every == 0:
                pred = viterbi_decode(scores, lengths - 1, start_tag_idx, end_tag_idx)
                pred = pred[:, 1:]              # remove START
                acc = token_acc(pred, tags, end_tag_idx)
                print('batch %d, loss=%.4f, acc=%.4f' % (i, loss.item(), acc))
                if acc >= ttacc:
                    ttacc = acc
                    print('save current model.')
                    torch.save(model.state_dict(), option.save_model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=19950125)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--train_dataset', type=str, default='../data/train1234.json')
    parser.add_argument('--dev_dataset', type=str, default='../data/dev5.json')
    parser.add_argument('--max_seq_len', type=int, default=113+2)
    # parser.add_argument('--bert_vocab', type=str, default='/users5/kliao/.pytorch_pretrained_bert/bert-base-chinese-vocab.txt')
    # parser.add_argument('--bert_model', type=str, default='/users5/kliao/.pytorch_pretrained_bert/bert-base-chinese')
    parser.add_argument('--bert_vocab', type=str,
                        default='./bert-base-uncased-vocab.txt')
    parser.add_argument('--bert_model', type=str, default='roberta-base')
    parser.add_argument('--tag_pad_idx', type=int, default=-1)
    parser.add_argument('--encoded_size', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_report_every', type=int, default=50)
    parser.add_argument('--dev_report_every', type=int, default=10)
    parser.add_argument('--save_model', type=str, default='models/e5.pt')
    option = parser.parse_args()

    main(option)
