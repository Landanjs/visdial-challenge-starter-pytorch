import torch
import torch.nn as nn

from utils import DynamicRNN

class HierarchicalRecurrentEncoder(nn.Module):

    def add_cmdline_args(parser):
        parser.add_argument_group('HRE specific arguments')
        parser.add_argument('-img_feature_size', default=4096,
                                help='Channel size of image feature')
        parser.add_argument('-embed_size', default=300,
                                help='Size of the input word embedding')
        parser.add_argument('-rnn_hidden_size', default=512,
                                help='Size of the multimodal embedding')
        parser.add_argument('-num_layers', default=2,
                                help='Number of layers in LSTM')
        parser.add_argument('-max_history_len', default=60,
                                help='Size of the multimodal embedding')
        parser.add_argument('-dropout', default=0.5, help='Dropout')
        parser.add_argument('-attend_hist', action='store_true',
                                help='Attention-over-history mechanism')
        
    def __init__(self, args):
        super().__init__()
        self.args = args
        print(args.dropout)
        self.word_embed = nn.Embedding(args.vocab_size, args.embed_size, padding_idx=0)

        self.ques_img_rnn = nn.LSTM(args.embed_size + args.img_feature_size,
                                    args.rnn_hidden_size, args.num_layers,
                                    batch_first=True, dropout=args.dropout)

        self.hist_rnn = nn.LSTM(args.embed_size, args.rnn_hidden_size, args.num_layers,
                                batch_first=True, dropout=args.dropout)

        self.dialog_rnn = nn.LSTM(args.rnn_hidden_size*2, args.rnn_hidden_size,
                                  args.num_layers, batch_first=True, dropout=args.dropout)

        self.ques_img_rnn = DynamicRNN(self.ques_img_rnn)
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.dialog_rnn = DynamicRNN(self.dialog_rnn)

    def forward(self, batch):
        # extract data
        img = batch['img_feat']                                                       # batch x feat_size
        ques = batch['ques']                                                          # batch x num_rounds x max_q_len
        hist = batch['hist']                                                          # batch x num_rounds x max_h_len
        batch_size, num_rounds, max_q_len = ques.shape

        # each round can be treated as an independent sample
        ques = ques.view(-1, ques.size(2))                                            # batch * num_rounds x max_q_len
        ques_embed = self.word_embed(ques)                                            # batch * num_rounds x max_q_len x embed_size


        ques_len = batch['ques_len'].view(-1)

        # can I just repeat the image vectors? depends if DynamicRNN will caught of the sections that should be padded
        #    can I use expand instead of repeat? effect on gradient?

        # concatenate image to each word embedding in each question
        expand_img = torch.zeros(batch_size*num_rounds, max_q_len, img.size(-1)).cuda()
        for sample in range(expand_img.size(0)):
            expand_img[sample, :ques_len[sample]] = img[sample // num_rounds]
        ques_img = torch.cat([ques_embed, expand_img], dim=-1)
        ques_img_embed = self.ques_img_rnn(ques_img, batch['ques_len'])               # batch * num_rounds x hidden_state_size

        # LSTM embedding of every previous round
        hist = hist.view(-1, hist.size(2))                                            # batch * num_rounds x max_h_len
        hist_embed = self.word_embed(hist)                                            # batch * num_rounds x max_h_len x embed_size
        hist_embed = self.hist_rnn(hist_embed, batch['hist_len'])                     # batch * num_rounds x hidden_state_size

        # concatenate image + question embedding with history embedding for every previous round
        ques_img_hist = torch.cat([ques_img_embed, hist_embed], -1)
        ques_img_hist = ques_img_hist.view(-1, num_rounds, 1, ques_img_hist.size(-1)) # batch x num_rounds x hidden_state_size * 2

        # initialize input to the dialog LSTM
        dialog = torch.zeros(batch_size, num_rounds, num_rounds,
                             hist_embed.size(-1) + ques_img_embed.size(-1)).cuda()
        
        # dialog LSTM input sequence length size for each sample
        dialog_len = torch.arange(1, num_rounds + 1).unsqueeze(0).cuda()
        dialog_len = dialog_len.repeat(batch_size, 1)                                 # batch_size x num_rounds

        # define the inputs to the dialog LSTM
        for round in range(num_rounds):
            dialog[:, round:, round, :] = ques_img_hist[:, round]

        # batch * num_rounds x num_rounds x hidden_state_size
        dialog = dialog.reshape(-1, num_rounds, dialog.size(-1)) 
        dialog_embed = self.dialog_rnn(dialog, dialog_len)
        return dialog_embed
        
            
        
        
        
        

        
        
