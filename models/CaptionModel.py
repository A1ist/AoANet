import torch.nn as nn
import torch
import torch.nn.functional as F

class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        # elif sample_method == 'gumbel':
        #     def sample_gumbel(shape, eps=1e-20):
        #         U = torch.rand(shape).cuda()
        #         return -torch.log(-torch.log(U + eps) + eps)
        #
        #     def gumbel_softmax_sample(logits, temperature):
        #         y = logits + sample_gumbel(logits.size())
        #         return F.log_softmax(y / temperature, dim=-1)
        #
        #     _logprobs = gumbel_softmax_sample(logprobs, temperature)
        #     _, it = torch.max(_logprobs.data, 1)
        #     sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
        else:
            logprobs = logprobs / temperature
        #     if sample_method.startswith('top'):
        #         top_num = float(sample_method[3:])
        #         if 0 < top_num < 1:
        #             probs = F.softmax(logprobs, dim=1)
        #             sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
        #             _cumsum = sorted_probs.cumsum(1)
        #             mask = _cumsum < top_num
        #             mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
        #             sorted_probs = sorted_probs * mask.float()
        #             sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
        #             logprobs.scatter_(1, sorted_indices, sorted_probs.log())
        #         else:
        #             the_k = int(top_num)
        #             tmp = torch.empty_like(logprobs).fill_(float('-inf'))
        #             topk, indices = torch.topk(logprobs, the_k, dim=1)
        #             tmp = tmp.scatter(1, indices, topk)
        #             logprobs = tmp
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))
        return it, sampleLogprobs