# -*- coding: utf-8 -*-
import json
from argparse import ArgumentParser
from utils import array_equal



def decode_mentions(tokens, tags):
    mention_dict = {}
    mention_type = None
    mention_buffer = []
    state = 0
    # 0: wait for start of a mention
    # 1: within a mention
    # print(tags)
    # exit(0)
    for token, tag in zip(tokens, tags):
        if state == 0:
            if tag[:2] == 'B-':
                mention_type = tag[2:]
                mention_buffer.append(token)
                state = 1
            else:
                pass
        elif state == 1:
            if tag[:2] == 'I-' and tag[2:] == mention_type:
                mention_buffer.append(token)
            else:
                mention = ' '.join(mention_buffer)
                if mention_type not in mention_dict:
                    mention_dict[mention_type] = []
                mention_dict[mention_type].append(mention)
                mention_type = None
                mention_buffer = []
                if tag[:2] == 'B-':
                    mention_type = tag[2:]
                    mention_buffer.append(token)
                else:
                    state = 0
    return mention_dict


def main(option):
    gold_data = json.load(open(option.gold_file, 'r'))
    pred_data = json.load(open(option.pred_file, 'r'))

    # count_a = 0     # extracted, should extract, correct
    # count_b = 0     # extracted, should extract, wrong
    # count_c = 0     # extracted, should not extract
    # count_d = 0     # not extracted, should extract
    # count_e = 0     # not extracted, should not extract


    aaa = 0
    bbb = 0
    count_acc = 0

    count_b = 0       # Precision的分母 只要给出预测短语，挖去为空的数据集
    for gold_instance, pred_instance in zip(gold_data, pred_data):
        gold_tokens = gold_instance['tokens']
        pred_tokens = pred_instance['tokens']
        assert array_equal(gold_tokens, pred_tokens)
        gold_tags = gold_instance['tags']
        pred_tags = pred_instance['tags']
        gold_mention_dict = decode_mentions(gold_tokens, gold_tags)
        pred_mention_dict = decode_mentions(pred_tokens, pred_tags)
        # print(gold_mention_dict)
        # print(pred_mention_dict)
        if gold_mention_dict == pred_mention_dict and len(gold_mention_dict) != 0 :
            count_acc += 1

        if 'cause' in gold_mention_dict:
            gold_cause = gold_mention_dict['cause'][0]
        if 'effect' in gold_mention_dict:
            gold_effect = gold_mention_dict['effect'][0]

        if 'cause' in pred_mention_dict:
            pred_causes = pred_mention_dict['cause'][0]
        if 'effect' in pred_mention_dict:
            pred_effects = pred_mention_dict['effect'][0]
            aaa = aaa + 1

        if 'cause' in pred_mention_dict or 'effect' in pred_mention_dict:
            count_b += 1

    print(aaa)
    print("前件和后件完全匹配 count_acc", count_acc)
    print("语料库中语料个数 len(gold_data)", len(gold_data))
    print("ACC准确率 count_acc/len(gold_data):",count_acc / len(gold_data))
    #print("ACC准确率 count_acc/(len(gold_data)+80):", count_acc / (len(gold_data) + 80))

    precision = count_acc / count_b
    recall = count_acc / len(gold_data)
    f1 = 2 * precision * recall / (precision + recall)

    recall_copy = count_acc / (len(gold_data) + 80)
    f1_copy = 2 * precision * recall_copy / (precision + recall_copy)
    print("count_b:", count_b)
    print("count_acc / count_b",'precision: %.4f' % (precision,))
    print("count_acc / len(gold_data)", 'recall: %.4f' % (recall,))
    #print("count_acc / (len(gold_data)+80)", 'recall_copy: %.4f' % (recall_copy,))
    print('f1: %.4f' % (f1,))
    #print('f1_copy: %.4f' % (f1_copy,))

# 1939 - 1859 = 80



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gold_file', type=str, default='../data/dev5.json')
    parser.add_argument('--pred_file', type=str, default='output/e5.json')
    option = parser.parse_args()

    main(option)
