import argparse
import json
import AoANet_C.misc.utils as utils
from collections import defaultdict

def precook(s, n=4):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n+1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i: i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4):
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
    crefs = []
    for ref in refs:
        crefs.append(cook_refs(ref))
    return crefs

def compute_doc_freq(crefs):
    document_frequency = defaultdict(float)
    for refs in crefs:
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1
    return document_frequency

def build_dict(imgs, wtoi, params):
    wtoi['<eos>'] = 0
    count_imgs = 0
    refs_words = []
    refs_idxs = []
    for img in imgs:
        if(params['split'] == img['split']) or \
        (params['split'] == 'train' and img['split'] == 'restval') or \
        (params['split'] == img['split']):
            ref_words = []
            ref_idxs = []
            for sent in img['sentences']:
                # if hasattr(params, 'bps'):
                #     sent['tokens'] = params.bps.segment(' '.join(sent['tokens'])).strip().split(' ')
                tmp_tokens = sent['tokens'] + ['<eos>']
                tmp_tokens = [_ if _ in wtoi else 'UNK' for _ in tmp_tokens]
                ref_words.append(' '.join(tmp_tokens))
                ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
            refs_words.append(ref_words)
            refs_idxs.append(ref_idxs)
            count_imgs += 1
    print('total imges:', count_imgs)
    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_words, ngram_idxs, count_imgs

def main(params):
    imags = json.load(open(params['input_json'], 'r'))
    dict_json = json.load(open(params['dict_json'], 'r'))
    itow = dict_json['ix_to_word']
    wtoi = {w: i for i, w in itow.items()}
    imgs = imags['images']
    ngram_words, ngram_idxs, ref_len = build_dict(imgs, wtoi, params)
    utils.pickle_dump(
        {'document_frequency': ngram_words, 'ref_len': ref_len},
        open(params['output_pkl'] + '-words.p', 'wb')
    )
    utils.pickle_dump(
        {'document_frequency': ngram_idxs, 'ref_len': ref_len},
        open(params['output_pkl'] + '-idxs.p', 'wb')
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='E:/SceneGraphProject/AoANet_C/data/dataset_coco.json', help='input json file to process into hdf5')
    parser.add_argument('--dict_json', default='E:/SceneGraphProject/AoANet_C/data/cocotalk.json', help='output json file')
    parser.add_argument('--output_pkl', default='E:/SceneGraphProject/AoANet_C/data/coco-train', help='output pickle file')
    parser.add_argument('--split', default='train', help='test, val, train, all')
    args = parser.parse_args()
    params = vars(args)
    main(params)

