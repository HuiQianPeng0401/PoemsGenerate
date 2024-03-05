import math
import torch
from torch import nn as nn
from torch import Tensor,device


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model,
            *,
            dropout=0.1,
            max_len=500
    ):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout)
        position=torch.arange(max_len).unsqueeze(1)
        den=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pos=torch.zeros(max_len,d_model)
        pos[:,0::2]=torch.sin(position*den)
        pos[:,1::2]=torch.cos(position*den)
        pos=pos.unsqueeze(0)
        self.register_buffer("pos",pos)


    def forward(self,x):
        x = x + self.pos[:,:x.size(1)]
        return self.dropout(x)



class PoetryModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            device,
            *,
            embed_size=512,
            n_head=8,
            n_layer=4,
            hidden_size=512
    ):
        super(PoetryModel,self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_size,device=device)
        self.embed_size=embed_size
        self.sq=math.sqrt(self.embed_size)
        self.transformer=nn.Transformer(
            embed_size,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            batch_first=True,
            dim_feedforward=hidden_size,
            device=device
        )
        self.generator=nn.Linear(embed_size,vocab_size)
        self.pos_encoding=PositionalEncoding(embed_size)

    def forward(
            self,
            src,
            tgt,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
    ):
        src=self.embed(src)*self.sq
        src=self.pos_encoding(src)
        tgt=self.embed(tgt)*self.sq
        tgt=self.pos_encoding(tgt)
        out=self.transformer.forward(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        out=self.generator.forward(out)
        return out

    def encode(
            self,
            src,
            noise=True,
            noise_intensity=1,
    ):
        embeded=self.embed.forward(src)
        if noise:
            embeded+=torch.rand_like(embeded)*noise_intensity
        return self.transformer.encoder.forward(
            self.pos_encoding.forward(embeded*self.sq)
        )

    def decode(
            self,
            tgt,
            memory,
    ):
        return self.generator.forward(
            self.transformer.decoder.forward(
                self.pos_encoding.forward(self.embed(tgt)*self.sq),
                memory,
            )
        )