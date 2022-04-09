from AoAModel import *
import os
import torch

def setup(opt):
    if opt.caption_model =='aoa':
        model = AoAModel(opt)
    # elif opt.caption_model == 'fc':
    #     model = FCModel(opt)
    # elif opt.caption_model =='language_model':
    #     model = LMModel(opt)
    # elif opt.caption_model =='newfc':
    #     model = NewFCModel(opt)
    # elif opt.caption_model =='show_tell':
    #     model = ShowTellModel(opt)
    # elif opt.caption_model =='att2in':
    #     model = Att2inModel(opt)
    # elif opt.caption_model =='att2in2':
    #     model = Att2in2Model(opt)
    # elif opt.caption_model =='att2all2':
    #     model = Att2all2Model(opt)
    # elif opt.caption_model =='adaatt':
    #     model = AdaAttModel(opt)
    # elif opt.caption_model =='adaattmo':
    #     model = AdaAttMoModel(opt)
    # elif opt.caption_model =='topdown':
    #     model = TopDownModel(opt)
    # elif opt.caption_model =='stackatt':
    #     model = StackAttModel(opt)
    # elif opt.caption_model =='denseatt':
    #     model = TransFormerModel(opt)
    # elif opt.caption_model =='transformer':
    #     model = LMModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))
    if vars(opt).get('star_from', None) is not None:
        assert os.path.isdir(opt.star_from), "%s must be a path" % opt.start_from
        assert os.path.isfile(opt.path.jion(opt.start_from, "infos_" + opt.id + ".pkl")), "infos.pkl file does not exist in path %s" % opt.start_from
        model.load_state_dict(torch.load(os.path.jion(opt.start_from, 'model.pth')))

