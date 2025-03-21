import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Callable
import numpy as np
from chessbot_byte.configs import parent_config, model_config
import math

dtype = parent_config.dtype
device = parent_config.device


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class TransformerEncoderLayer(nn.Module):

    __constants__ = ['norm_first']

    def __init__(self, d_model: int = model_config.d_model,
                 nhead: int = model_config.nhead,
                 dim_feedforward: int = model_config.dim_feedforward,
                 dropout: float = model_config.dropout,
                 activation: Union[str, Callable[[Tensor],
                                                 Tensor]] = model_config.activation,
                 layer_norm_eps: float = model_config.layer_norm_eps,
                 batch_first: bool = model_config.batch_first,
                 norm_first: bool = model_config.norm_first,
                 bias: bool = model_config.bias,
                 num_experts: int = model_config.num_experts,
                 num_experts_per_tok: int = model_config.num_experts_per_tok) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               bias=bias, batch_first=batch_first,
                                               **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward,
                                 bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model,
                                 bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps,
                                  bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps,
                                  bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # creating the experts -
        self.experts = nn.ModuleList(
            [google_expert(d_model, dim_feedforward, bias) for i in range(num_experts)])

        # creating gatingnetwork
        self.gatingNetwork = GatingNetwork(d_model, num_experts)
        self.num_experts_per_tok = num_experts_per_tok
        # Legacy string support for activation function.

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask,
                                   src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask,
                           src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
#     def _ff_block(self, x: Tensor , num_experts_per_tok) -> Tensor:
#         gating_scores = F.softmax(self.gatingNetwork(x) , dim = -1)
#         top_scores, top_indices = gating_scores.topk(self.num_experts_per_tok , dim = -1 , sorted = False)

#         top_mask = torch.zeros_like(gating_scores)
#         top_mask = top_mask.scatter(-1, top_indices, 1)
#         output = []

#         for i in range(len(top_mask)):
#             if(top_mask[i] == 1):
#                 output.append(top_scores[i]*self.experts[i](x))

#         result = torch.sum(torch.stack(output) , dim = 0)
#         return result
# #         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
# #         return self.dropout2(x)hat

    def _ff_block(self, x: Tensor) -> Tensor:
        # gating_scores = F.softmax(self.gatingNetwork(x), dim=-1)

        gating_scores = self.gatingNetwork(x)
        top_scores, top_indices = gating_scores.topk(
            self.num_experts_per_tok, dim=-1, sorted=False)
        print('gating scores', gating_scores.shape)
        print('top score', top_scores.shape)
        print('top indices', top_indices.shape)
        print(torch.zeros_like(x).shape)
        
        output = torch.zeros_like(x)

        for i, index in enumerate(top_indices):
            print(index.shape)
            expert_outputs = [top_scores[i][j] * self.experts[index[j]]
                              (x[i].unsqueeze(0)) for j in range(self.num_experts_per_tok)]
            output[i] = sum(expert_outputs)

        return output


class GatingNetwork(nn.Module):
    def __init__(self,
                 input_dim=model_config.d_model,
                 num_experts=model_config.num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)


class expert(nn.Module):
    def __init__(self, d_model=model_config.d_model,
                 dim_feedforward=model_config.dim_feedforward,
                 bias=model_config.expert_bias,
                 dropout_factor=model_config.expert_dropout_factor,
                 activation=model_config.expert_activation):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1 = nn.Linear(d_model, dim_feedforward,
                                 bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout_factor)
        self.linear2 = nn.Linear(dim_feedforward, d_model,
                                 bias=bias, **factory_kwargs)
        self.dropout2 = nn.Dropout(dropout_factor)
        self.activation = activation

    def forward(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)


class google_expert(nn.Module):
    def __init__(self, d_model=model_config.d_model,
                 dim_feedforward=model_config.dim_feedforward,
                 bias=model_config.expert_bias):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1 = nn.Linear(d_model, dim_feedforward,
                                 bias=bias, **factory_kwargs)
        self.linear2 = nn.Linear(
            d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.linear_out = nn.Linear(
            dim_feedforward, d_model, bias, **factory_kwargs)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x = F.silu(x1)*x2
        x = self.linear_out(x)
        return x


class cust_embeddings(nn.Module):
    def __init__(self, embedding_dim=model_config.d_model,
                 emb_init_scale=model_config.emb_init_scale,
                 NUM_ACTIONS=model_config.NUM_ACTIONS,
                 SEQUENCE_LENGTH=model_config.SEQUENCE_LENGTH,
                 use_sinosoidal=model_config.use_sinosoidal,
                 max_timescale=model_config.max_timescale):  # max_timescale 1e4
        super(cust_embeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = SEQUENCE_LENGTH + 2
        self.emb_init_scale = emb_init_scale
        self.max_timescale = max_timescale
        vocab_size = NUM_ACTIONS  # same is done in the google repo , idk why though

        if (use_sinosoidal):
            pos_embeddings = self._get_sinusoidal_position_encoding(
                self.sequence_length, embedding_dim, max_timescale)
            self.register_buffer("pos_embeddings", pos_embeddings)
        else:
            self.pos_embeddings = nn.Embedding(
                num_embeddings=self.sequence_length, embedding_dim=embedding_dim).to(device)  # 79 is the max seq length

        self.embeddings_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim).to(device)
        self.init_weights()

    def init_weights(self):
        init.trunc_normal_(self.embeddings_layer.weight,
                           std=self.emb_init_scale)

    def _get_sinusoidal_position_encoding(self, sequence_length, embedding_dim, max_timescale):
        """
        Creates sinusoidal encodings as in the original Transformer paper.

        For each position pos and dimension i:
          PE(pos, 2i)   = sin(pos / (max_timescale^(2i/embedding_dim)))
          PE(pos, 2i+1) = cos(pos / (max_timescale^(2i/embedding_dim)))

        Returns:
            A tensor of shape (sequence_length, embedding_dim).
        """
        # Create a tensor of positions (shape: [sequence_length, 1])
        position = torch.arange(
            0, sequence_length, dtype=torch.float32).unsqueeze(1).to(device)
        # Compute the scaling term for each even index in the embedding dimensions.
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) *
            (-torch.log(max_timescale) / embedding_dim)
        ).to(device)
        # Calculate the sinusoidal input by outer product (broadcast multiplication)
        sinusoid_inp = position * div_term.unsqueeze(0)
        pos_encoding = torch.zeros(sequence_length, embedding_dim)
        pos_encoding[:, 0::2] = torch.sin(sinusoid_inp)
        pos_encoding[:, 1::2] = torch.cos(sinusoid_inp)
        return pos_encoding

    def forward(self, x):

        pos_embed = self.pos_embeddings(torch.arange(self.sequence_length).to(device)).to(device)
        embed = self.embeddings_layer(x)*math.sqrt(self.embedding_dim)
        return pos_embed+embed


def shift_right(sequences: torch.Tensor) -> torch.Tensor:
    """Right-shift the one-hot encoded input by padding on the temporal axis."""    # assuming this is wrong , but bc google thori galat hoga
    bos_array = torch.zeros((sequences.size(0), 1), dtype=sequences.dtype)
    # Concatenate the bos_array with sequences along the temporal axis
    padded_sequences = torch.cat([bos_array, sequences], dim=1)
    # Return the padded sequences, excluding the last element along the temporal axis
    return padded_sequences[:, :-1]


class decoder(nn.Module):
    def __init__(self,
                 input_size=model_config.d_model,
                 output_size=model_config.num_return_buckets,
                 decoder_layernorm=model_config.decoder_layernorm):

        super().__init__()
        self.use_layer_norm = decoder_layernorm
        self.layer_norm = nn.LayerNorm(input_size)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        if (self.use_layer_norm):
            x = self.layer_norm(x)
        x = self.linear(x)
        return x


class chessbot_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = cust_embeddings().to(device)
        self.encoders = nn.ModuleList(
            [TransformerEncoderLayer().to(device) for _ in range(model_config.encoder_layers)])
        self.decoder = decoder().to(device)

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.encoders:
            x = layer(x)

        logits = self.decoder(x)
    # Apply log softmax
        return F.log_softmax(logits, dim=-1)
