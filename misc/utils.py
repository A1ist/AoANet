import six
from six.moves import cPickle
import torch.nn as nn
import torch
import torch.optim as optim

def if_use_feat(caption_model):
    if caption_model in ['show_tell', 'all_img', 'fc', 'newfc']:
        use_fc, use_att = True, False
    elif caption_model == 'language_model':
        use_fc, use_att = False, False
    elif caption_model in ['topdown', 'aoa']:
        use_fc, use_att = True, True
    else:
        use_fc, use_att = False, True
    return use_fc, use_att

def pickle_load(f):
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)

class LabelSmoothing(nn.Module):
    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()
    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = -input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

class NoamOpt(object):
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

def get_std_opt(model, factor=1, warmup=2000):
    Noam_Opt = NoamOpt(
        model.model.tat_embed[0].d_model,
        factor,
        warmup,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )
    return Noam_Opt

def build_optimizer(params, opt):
    if opt.optim == 'adma':
        optimer = optim.Adam(
            params,
            opt.learing_rate,
            (opt.optim_alpha, opt.optim_beta),
            opt.optim_epsilon,
            weight_decay=opt.weight_decay
        )
        return optimer
    # elif opt.optim == 'rmsprop':
    #     optimer = optim.RMSprop(
    #         params,
    #         opt.learing_rate,
    #         opt.optim_alpha,
    #         opt.optim_epsilon,
    #         weight_decay=opt.weight_decay
    #     )
    #     return optimer
    # elif opt.optim == 'adagrad':
    #     optimer = optim.Adagrad(
    #         params,
    #         opt.learing_rate,
    #         weight_decay=opt.weight_decay
    #     )
    #     return optimer
    # elif opt.optim == 'sgd':
    #     optimer = optim.SGD(
    #         params,
    #         opt.learing_rate,
    #         weight_decay=opt.weight_decay
    #     )
    #     return optimer
    # elif opt.optim == 'sgdm':
    #     optimer = optim.SGD(
    #         params,
    #         opt.learing_rate,
    #         opt.optim_alpha,
    #         weight_decay=opt.weight_decay
    #     )
    #     return optimer
    # elif opt.optim == 'sgdmom':
    #     optimer = optim.SGD(
    #         params, opt.learing_rate,
    #         opt.optim_alpha,
    #         weight_decay=opt.weight_decay,
    #         nesterov=True
    #     )
    #     return optimer
    # else:
    #     raise Exception("bad option opt.optim: {}".format(opt.optim))

def get_lr(optimizer):
    for group in optimizer.param_grous:
        return group['lr']

class ReduceLROnPlateau(object):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps
        )
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)

def pickle_dump(obj, f):
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_grous:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)
