from AOANet_C.metrics.ciderD.ciderD import CiderD
from AOANet_C.metrics.bleu.bleu import Bleu

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)