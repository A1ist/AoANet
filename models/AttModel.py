import torch.nn as nn
from .CaptionModel import CaptionModel
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.functional as F
import numpy as np

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is', 'are', 'am']

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

    def clip_att(self, att_feats, att_masks):
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    # def _sample_beam(self, fc_feats, att_feats, att_masks, opt):

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        weight_reset = weight.new_zeros(self.num_layers, bsz, self.rnn_size)
        weights = (weight_reset, weight_reset)
        return weights

    def get_logprobs_states(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        xt = self.embed(it)
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        batch_size = fc_feats.size[0]
        state = self.init_hidden(batch_size)
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        trigrams = []
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            logprobs, state = self.get_logprobs_states(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, att_feats)
            # if decoding_constraint and t > 0:
            #     tmp = logprobs.new_zeros(logprobs.size())
            #     tmp.scatter_(1, seq[:, t-1].data.unsqueese(1), float('-inf'))
            #     logprobs = logprobs + tmp
            # if remove_bad_endings and t > 0:
            #     tmp = logprobs.new_zeros(logprobs.size())
            #     prev_bad = np.isin(seq[:, t-1].data.cpu().numpy(), self.bad_endings_ix)
            #     tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
            #     logprobs = logprobs + tmp
            # if block_trigrams and t >= 3:
            #     prev_two_batch = seq[:, t-3: t-1]
            #     for i in range(batch_size):
            #         prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
            #         current = seq[i][t-1]
            #         if t == 3:
            #             trigrams.append({prev_two: [current]})
            #         elif t > 3:
            #             if prev_two in trigrams[i]:
            #                 trigrams[i][prev_two].append(current)
            #             else:
            #                 trigrams[i][prev_two] = [current]
            #     prev_two_batch = seq[:, t-2: t]
            #     mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()
            #     for i in range(batch_size):
            #         prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
            #         if prev_two in trigrams[i]:
            #             for prev_two in trigrams[i]:
            #                 mask[i, j] += 1
            #     alpha = 2.0
            #     logprobs = logprobs + (mask * -0.693 * alpha)
            # if t == self.seq_length:
            #     break

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_fist=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_warpper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(
            att_feats,
            att_masks.data.long().sum(1)
        )
        unsort_sequnce = pad_unsort_packed_sequence(
            PackedSequence(module(packed[0]), packed[1]),
            inv_ix
        )
        return unsort_sequnce
    else:
        return module(att_feats)
