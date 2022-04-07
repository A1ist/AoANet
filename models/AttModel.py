
import torch.nn as nn
from .CptionModel import CaptionModel

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.ss_prob = 0.0
        self.embed = nn.Sequential(
            nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm)
        )
        self.fc_embed = nn.Sequential(
            nn.Linear(self.fc_feat_size, self.rnn_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm)
        )
        self.att_embed = nn.Sequential(
            *(
                # ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (
                    nn.Linear(self.att_feat_size, self.rnn_size),
                    nn.ReLU(),
                    nn.Dropout(self.drop_prob_lm)
                )
                # +((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else())
            )
        )
        self.logit_layer = getattr('logit_layers', 1)
        if self.logit_layer == 1:
            self.logit = nn.Liner(self.rnn_size, self.vocab_size + 1)
        # else:
        #     self.logit = [
        #         [nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)
        #     ]
        #     self.logit = nn.Sequential(
        #         *(
        #             reduce(lambda x,y: x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]
        #         )
        #     )
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.vocab = opt.vocab
        self.bad_endings_ix = [int(k) for k, v in self.vocab.items() if v in bad_endings]