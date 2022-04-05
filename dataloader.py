import torch.utils.data as data
import json
import h5py

class DataLoader(data.Dataset):
def __init__(self, opt):
    self.opt = opt
    self.batch_size = self.opt.batch_size
    self.seq_per_img = opt.seq_per_img
    self.use_fc = getattr(opt, 'use_fc', True)
    self.use_att = getattr(opt, 'use_att', True)
    # self.use_box = getattr(opt, 'use_box', 0)
    # self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
    # self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)
    print('DataLoader loading json file:', opt.input_json)
    self.info = json.load(open(self.opt.input_json))
    if 'ix_to_word' in self.info:
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print("vocab size is", self.vocab_size)
    print('DataLoader loading h5 file:', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
    if self.opt.input_label_h5 != 'none':
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
        seq_size = self.h5_label_file['labels'].shape
        self.label = self.h5_label_file['labels'][:]
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]
    # else:
    #     self.seq_length = 1
