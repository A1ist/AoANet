import torch.utils.data as data
import json
import h5py
import numpy as np
import lmdb
import os
import torch

class HybridLoader:
    def __init__(self, db_path, ext):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(x)
        else:
            self.loader = lambda x: np.load(x)['feat']
        if db_path.endwith('.lmdb'):
            self.db_type = 'lmdb'
            self.env = lmdb.open(
                db_path,
                subdir=os.path.isdir(db_path),
                readonly=True,
                readahead=False,
                lock=False,
                meminit=False
            )
        elif db_path.endwith('.pth'):
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print("HybridLoader:ext is ignored")
        else:
            self.db_type = 'dir'
class BlobFetcher():
    def __init__(self, split, dataloader, if_shuffle=False):
        self.split = split
        self.dataloder = dataloader
        self.if_shuffle = if_shuffle


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
        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy')
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz')
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy')
        self.num_images = len(self.info['images'])
        print("read %d image features" % self.num_images)
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.num_images)):
            img = self.info['images'][ix]
            if not 'split' in img:
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:
                self.split_ix['train'].append(ix)
        print('assign %d images to split train' % len(self.split_ix['train']))
        print('assign %d images to split val' % len(self.split_ix['val']))
        print('assign %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0,'val': 0, 'test': 0}
        self._prefetch_process = {}
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == "train")
        # def cleanup():
        #     print('Terminating BlobFetcher')
        #     for split in self.iterators.keys():
        #         del self._prefetch_process[split]
        # import atexit
        # atexit.register(cleanup)

    def get_vocab(self):
        return self.ix_to_word


