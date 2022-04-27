import misc.utils as utils
from dataloader import *
import models
import torch.nn as nn
from misc.loss_wrapper import LossWrapper
import opts
from misc.rewards import init_scorer
import time
from misc.loss_wrapper import LossWrapper

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
    model = models.setup(opt).cuda()
    del opt.vocab
    de_model = nn.DataParallel(model)
    lw_model = LossWrapper(model, opt)
    dp_lw_model = nn.DataParallel(lw_model)
    epoch_done = True
    dp_lw_model.train()
    if opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    # elif opt.noamopt:
    #     assert opt.caption_model in ['transformer', 'aoa'], 'noamopt can only work with transformer'
    #     optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
    #     optimizer._step = iteration
    # else:
    #     optimizer = utils.build_optimizer(model.parameters(), opt)
    if vars(opt).get('star_from', None) is not None and os.path.isfile(os.path.jion(opt.start_from, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, "optimizer.pth")))

    try:
        while True:
            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.learning_lr)
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    sc_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    sc_flag = False
                epoch_done = False
            start = time.time()
            if (opt.use_warmup == 1) and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * (iteration + 1) / opt.noamopt_warmup
                utils.set_lr(optimizer, opt.current_lr)
            data = loader.get_batch('train')
            print('Read data:', time.time() - start)
            if (iteration % acc_steps == 0):
                optimizer.zero_grad()
            torch.cuda.synchronize()
            start = time.time()
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp
            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag)
            loss = model_out['loss'].mean()
            loss_sp = loss / acc_steps
            loss_sp.backward()
            if((iteration + 1) % acc_steps == 0):
                utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
            torch.cuda.synchronize()
            train_loss = loss.item()
            end = time.time()
            if not sc_flag:
                print("iter {}  (epoch {}), train_loss = {: .3f}, time/batch = {: .3f}".format(iteration, epoch, train_loss, end-start))
            else:
                print("iter {}  (epoch {}), avg_reward = {: .3f}, time/batch = {: .3f}".format(iteration, epoch, model_out['reward'].mean, end-start))


    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')




if __name__ == '__main__':
    opt = opts.parse_opt()
    train(opt)