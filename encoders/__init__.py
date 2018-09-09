from .lf import LateFusionEncoder
from .hr import HierarchicalRecurrentEncoder

name_enc_map = {
    'lf-ques-im-hist': LateFusionEncoder,
    'hr-ques-im-hist': HierarchicalRecurrentEncoder
}

def EncoderArgs(encoder, parser):
    name_enc_map[encoder].add_cmdline_args(parser)
    

def Encoder(model_args):
    return name_enc_map[model_args.encoder](model_args)

