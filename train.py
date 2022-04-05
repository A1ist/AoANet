import misc.utils as utils
from dataloader import *


def train(opt):
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)
    # if opt.use_box:
    #     opt.att_feat_size = opt.att_feat_size + 5
    acc_steps = getattr(opt, 'acc_steps', 1)
    loader = DataLoader(opt)

