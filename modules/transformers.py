import torch
import torch.nn
from torch.nn import functional as F, Embedding, LayerNorm, Linear
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def generate_attn_mask(sz):
    """The function generates an attention mask for autoregressive tasks. Masked
    positions are filled with float('-inf'). Unmasked positions are filled with
    0s.
    """

    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask\
        .float()\
        .masked_fill(mask == 0, float('-inf'))\
        .masked_fill(mask == 1, float(0.0))
    
    return mask


class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    """The class is an extension of torch.nn.TransformerEncoderLayer capable of
    returning attention weights. See the torch.nn.TransformerEncoderLayer source
    for the documentation.
    """

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        need_weights=False # output attention weights?
    ):
        src2, weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return (src, weights) if need_weights else src

class TransformerEncoder(torch.nn.TransformerEncoder):
    """The class is an extension of torch.nn.TransformerEncoder capable of
    returning attention weights. See the torch.nn.TransformerEncoder source
    for the documentation.
    """

    def forward(
        self,
        src,
        mask=None,
        src_key_padding_mask=None,
        need_weights=False # output attention weights?
    ):
        output = src

        if need_weights:
            weights = torch.empty(
                src.shape[1], # batch size
                len(self.layers), # number of layers
                src.shape[0], # number of positions
                src.shape[0], # number of positions
                device=src.device
            )

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                need_weights=need_weights
            )
            if need_weights:
                weights[:,i] = output[1]
                output = output[0]

        if self.norm is not None:
            output = self.norm(output)

        return (output, weights) if need_weights else output

class ARTransformer(Module):
    """The class implements an encoder-only adaptation of the "Attention Is All
    You Need" model capable of dealing with time sequences of frames. The model
    is designed to work in an autoregressive fashion.
    """

    def __init__(
        self,
        num_embeddings, # size of the data vocabulary
        num_positions, # maximum input sequence length
        d_model=512, # number of expected features in the encoder
        nhead=8, # number of heads in the multiheadattentions
        num_encoder_layers=6, # number of layers in the encoder
        dim_feedforward=2048, # dimension of the feedforward networks
        dropout=0.1,
        activation='relu' # 'relu' or 'gelu'
    ):
        super(ARTransformer, self).__init__()

        # learnable embeddings
        self.sos = Parameter(torch.empty(d_model)) # Start of String (SoS)
        torch.nn.init.normal_(self.sos)
        self.embedding = Embedding(num_embeddings, d_model)
        
        # learnable positions
        self.positions = Parameter(torch.cat([
            torch.tensor([num_positions - 1]), # SoS position
            torch.arange(num_positions - 1) # remaining positions
        ]), requires_grad=False)
        self.positional_encoding = Embedding(num_positions, d_model)

        # attention mask
        self.attn_mask = Parameter(
            generate_attn_mask(num_positions),
            requires_grad=False
        )

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation
        )
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm
        )

        # final linear layer
        self.linear = Linear(d_model, num_embeddings)
        
    def forward(
        self,
        src, # input sequence
        need_weights=False # output attention weights?
    ):
        embeddings = self.embedding(src)
        embeddings = torch.cat([ # add the SoS token and shift
            self.sos.expand(1, src.shape[1], - 1),
            embeddings[:-1]
        ])

        embeddings += self.positional_encoding(self.positions[:src.shape[0]])\
            .unsqueeze(1)\
            .expand_as(embeddings)
        
        output = self.encoder(
            embeddings,
            mask=self.attn_mask[:src.shape[0],:src.shape[0]],
            need_weights=need_weights
        )
        if need_weights:
            weights = output[1]
            output = output[0]

        logits = self.linear(output)
        
        return (logits, weights) if need_weights else logits

class ARTransformerNum(ARTransformer):
    """The class is an extension of ARTransformer capable of dealing with
    additional numerosity features. In particular, different numerosities are
    associated with different learnable SoS tokens.
    """

    def __init__(
        self,
        num_num, # number of different numerosities
        num_embeddings,
        num_positions,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu'
    ):
        super(ARTransformerNum, self).__init__(
            num_embeddings,
            num_positions,
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            activation
        )

        self.d_model = d_model

        # replace the old SoS
        self.__delattr__('sos')
        self.sos = Embedding(num_num, d_model)

    def forward(
        self,
        num, # numerosity
        src, # input sequence
        need_weights=False
    ):
        if not (num.dtype == torch.long or num.size(- 1) == self.d_model):
            raise RuntimeError(
                'numerosity features or embeddings must be provided'
            )
        
        embeddings = self.embedding(src)
        if num.dtype == torch.long: # numerosity features
            # compute numerosity embeddings
            sos = self.sos(num).unsqueeze(0)
        else: # numerosity embeddings
            # directly inject embeddings
            sos = num.unsqueeze(0)
        embeddings = torch.cat([ # add the SoS token and shift
            sos,
            embeddings[:-1]
        ])

        embeddings += self.positional_encoding(self.positions[:src.shape[0]])\
            .unsqueeze(1)\
            .expand_as(embeddings)
        
        output = self.encoder(
            embeddings,
            mask=self.attn_mask[:src.shape[0],:src.shape[0]],
            need_weights=need_weights
        )
        if need_weights:
            weights = output[1]
            output = output[0]

        logits = self.linear(output)
        
        return (logits, weights) if need_weights else logits

    def predict(self, num, src):
        """The method conditionally predicts each frame pixel from the (ground
        truth) previous ones and the provided frame numerosity.
        """

        self.eval()

        with torch.no_grad():
            logits = self(num, src)
        probabilities = F.softmax(logits, dim=- 1)
        predictions = torch.argmax(probabilities, dim=- 1)
        
        return predictions

    def generate(self, num, src):
        """The method autoregressively generates a novel batch from the learned
        conditional distribution. Frames are sampled in accordance with the
        provided numerosities.
        """

        self.eval()
        
        with torch.no_grad():
            for i in range(src.shape[0]):
                logits = self(num, src[:i+1])
                probabilities = F.softmax(logits[i], dim=- 1)
                predictions = torch.multinomial(probabilities, num_samples=1)\
                    .squeeze()
                src[i] = predictions

        return src