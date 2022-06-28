import numpy as np
import torch
import torch.nn as nn


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class AGE(nn.Module):
    def __init__(self, args):
        super(AGE, self).__init__()
        self.model_name = 'AGE'
        self.args = args
        self.device = args.experiment.device
        self.model_args = args.model
        self.n = self.args.model.input_size

        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.hidden_size = self.model_args.hidden_size
        self.embed = nn.Linear(self.n, self.hidden_size)
        # Initialise the Encoder
        enc_args = self.model_args.encoder
        enc_args.input_size = self.n
        enc_args.hidden_size = self.hidden_size
        layer = torch.nn.TransformerEncoderLayer(self.hidden_size,
                                                 nhead=enc_args.num_attn_heads,
                                                 dim_feedforward=enc_args.dim_feedforward,
                                                 dropout=enc_args.dropout,
                                                 batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_args.num_layers)
        dec_args = self.model_args.decoder
        layer = torch.nn.TransformerDecoderLayer(self.hidden_size,
                                                 nhead=dec_args.num_attn_heads,
                                                 dim_feedforward=dec_args.dim_feedforward,
                                                 dropout=dec_args.dropout,
                                                 batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, dec_args.num_layers)
        self.out_lin = nn.Linear(self.hidden_size, self.n)

        self.idx = [[i, j] for i in range(self.n - 1) for j in range(i+1)]
        self.idx = np.array(self.idx)

    def _inference(self, x, y_adj):
        # x = self.target_embed(x)
        x = self.encoder(self.embed(x))
        mask = generate_square_subsequent_mask(self.n-1).to(x.device)
        y_adj = self.decoder(self.embed(y_adj), x, tgt_mask=mask)
        return self.out_lin(y_adj)

    def forward(self, data):
        y = data['y'].to(self.device) if 'y' in data else None
        y_lab = data['y_lab'].to(self.device) if 'y_lab' in data else None
        x = data['x'].to(self.device) if 'x' in data else None
        adj = data['adj'].to(self.device) if 'adj' in data else None
        is_sampling = data['is_sampling'] if 'is_sampling' in data else None

        if not is_sampling:
            A_pred_logits = self._inference(x, y) # B * (n-1) * n
            # Get the lower triangle of A_pred
            edges_pred = A_pred_logits[:, self.idx[:, 0], self.idx[:, 1]]
            loss = self.loss_fn(edges_pred, y_lab)
            return loss
        else:
            return self.sampling(adj)

    def sampling(self, x):
        B = x.shape[0] # Batch size
        n = self.n # Number of nodes - 1

        encoder_outputs = self.encoder(self.embed(x))
        outputs = torch.zeros((B, n, n)).to(x.device)
        # Set the start of sequence token
        outputs[:, 0] = 1
        for i in range(1, n):
            mask = generate_square_subsequent_mask(i).to(x.device)
            decoder_outputs = self.decoder(self.embed(outputs[:, :i]), encoder_outputs, tgt_mask=mask)
            decoder_outputs = torch.sigmoid(self.out_lin(decoder_outputs[:, -1].unsqueeze(1)))
            outputs[:, i:i + 1, :i] = torch.bernoulli(decoder_outputs[:, :, :i])
        outputs[:, 0] = 0  # Remove the SOS token.
        return outputs

    def recursive_forward(self, x, T):
        ' Returns a list with T sampled timesteps of the network.'
        out_ts = []
        for t in range(T):
            x = self.sampling(x)
            out_ts.append(x + x.transpose(1, 2))
        return out_ts

