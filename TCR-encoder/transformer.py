import torch
import torch.nn as nn
from torch import softmax
from torch.nn.init import xavier_uniform_
from positional_encoding import PositionalEncoding
# from TransformerEncoderLayer_R1 import TransformerEncoderLayer_R1

import globalvar as gl
device = gl.get_value('cuda')

class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model=128, nhead=2, num_encoder_layers=2,
                 num_decoder_layers=6, dim_feedforward=512, dropout=0.1):
        super(Transformer, self).__init__()

        # Preprocess
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0).to(device)
        # with torch.no_grad():
        #     self.embedding.weight[0] = torch.zeros(self.d_model)
        self.pos_encoder_src = PositionalEncoding(d_model=128).to(device)
        # tgt
        # self.pos_encoder_tgt = PositionalEncoding(d_model=128).to(device)

        # Encoder
        encoder_layer = TransformerEncoderLayer_R1(d_model, nhead, dim_feedforward, dropout).to(device)
        encoder_norm = nn.LayerNorm(d_model).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm).to(device)

        # Decoder
        # decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,dropout).to(device)
        # decoder_norm = nn.LayerNorm(d_model).to(device)
        # self.decoder = nn.TransformerDecoder(decoder_layer,num_decoder_layers,decoder_norm).to(device)
        self.output_layer = nn.Linear(d_model, vocab_size).to(device)

        self._reset_parameters()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead


    def forward(self, org_src,tgt,src_mask = None,tgt_mask = None,
                memory_mask = None,src_key_padding_mask = None,
                tgt_key_padding_mask = None,memory_key_padding_mask = None):

        # word embedding
        # emb_src = nn.Embedding(self.vocab_size,self.d_model, padding_idx=0).to(device)(org_src)

        emb_src = self.embedding(org_src)
        # tgt = self.embedding(tgt)
        #
        # # shape check
        # if src.size(1) != tgt.size(1):
        #     raise RuntimeError("the batch number of src and tgt must be equal")
        # if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
        #     raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        # position encoding
        src = self.pos_encoder_src(emb_src)
        # tgt = self.pos_encoder_tgt(tgt)

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
        #                       tgt_key_padding_mask=tgt_key_padding_mask,
        #                       memory_key_padding_mask=memory_key_padding_mask)
        output = self.output_layer(memory)
        return output
        # return softmax(output, dim=2)


    def generate_square_subsequent_mask(sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)