import misc.utils as utils
from dataloader import *
import models
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def train(opt):
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)
    # if opt.use_box:
    #     opt.att_feat_size = opt.att_feat_size + 5
    acc_steps = getattr(opt, 'acc_steps', 1)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)
    infos = {}
    histories = {}
    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos'+opt.id+'.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl', 'rb')) as f:
                histories = utils.pickle_load(f)
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix
        infos['vocab'] = loader.get_vocab()
    infos['opt'] = opt
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.loader_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    opt.vocab = loader.get_vocab()
    model = models.setup()





