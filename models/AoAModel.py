import torch.nn as nn
from .AttModel import AttModel
from .TransformerModel import LayerNorm, clones

class MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0, dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        self.d_k = d_model * scale // h
        self.h = h
        self.project_k_v = project_k_v
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        self.linear = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)
        self.output_layer = nn.Linear(d_model * scale, d_model)
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(
                nn.Linear((1 + scale) * d_model, 2 * d_model),
                nn.GLU()
            )
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dorpout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x
        if self.use_aoa or not use_output_layer:
            del self.output_layer
            self.output_layer = lambda x: x
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

class AOA_Refiner_Core(nn.Module):
    def __init__(self, opt):
        super(AOA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(opt.num_heads,)

class AoAModel(AttModel):
    def __init__(self, opt):
        super(AoAModel, self).__init__(opt)
        self.num_layers = 2
        self.use_mean_feats = getattr(opt, 'mean_feats', 1)
        if opt.use_mutil_head == 2:
            del self.ctx2att
            self.ctx2att = nn.Linear(opt.rnn_size, 2 * opt.use_mutil_head_scale * opt.rnn_size)
        if self.use_mean_feats:
            del self.fc_embed
        if opt.refine:
            self.refiner = AoA_Refiner_Core(opt)
        else:
            self.refiner = lambda x, y: x
        self.core = AoA_Decder_Core(opt)