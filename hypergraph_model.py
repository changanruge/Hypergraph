import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
        X: [n_node, dim]
        path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)
        return x


class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        b, c, l = x.shape[0], x.shape[1], x.shape[2]
        x = x.transpose(1, 2).contiguous()  # [B, L, C]
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = (distance < self.threshold).float().to(x.device)
        x = self.hgconv(x, hg) + x
        x = x.transpose(1, 2).contiguous()  # [B, C, L]
        x = self.act(self.bn(x))
        return x


class HypergraphModel(nn.Module):
    """
    HyperGraph模型 - 用于分类任务
    """

    def __init__(self, seq_len=116, enc_in=116, num_class=2, d_model=32, dropout=0.5):
        super(HypergraphModel, self).__init__()
        self.seq_len = seq_len
        self.channels = enc_in
        self.hidden_dim = d_model
        self.dropout_p = dropout

        # 使用 HyperComputeModule 作为 encoder
        self.encoder_layer = HyperComputeModule(
            c1=self.channels,
            c2=self.channels,
            threshold=0.5
        )

        self.projection = nn.Linear(self.channels * self.seq_len, num_class)

    def encoder(self, x):
        # x: [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        out = self.encoder_layer(x)
        return out.permute(0, 2, 1)  # [B, L, C]

    def forward(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        return self.projection(output)
