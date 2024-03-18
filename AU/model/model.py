import torch
import torch.nn as nn
import yaml
from munch import DefaultMunch

from .tcn import TemporalConvNet
from .trans_encoder import TransEncoder


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        tcn_channels = [768, 256, 128]
        self.temporal = TemporalConvNet(
            num_inputs=tcn_channels[0],
            num_channels=tcn_channels,
            kernel_size=cfg.Model.kernel_size,
            dropout=cfg.Solver.dropout,
            attention=False,
        )

        self.encoder = TransEncoder(
            inc=tcn_channels[-1],
            outc=cfg.Model.out_dim,
            dropout=cfg.Solver.dropout,
            nheads=cfg.Model.num_head,
            nlayer=cfg.Model.num_layer,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.Model.out_dim, cfg.Model.out_dim // 2),
            nn.BatchNorm1d(cfg.Model.out_dim // 2),
            nn.Linear(cfg.Model.out_dim // 2, 12),
        )

    def forward(self, x):
        if x.shape[2] == 1:
            x = x[:, :, 0, :]
        bs, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = self.encoder(x)
        x = torch.transpose(x, 1, 0)
        x = torch.reshape(x, (bs * seq_len, -1))

        x = self.head(x)
        return x


if __name__ == "__main__":
    config_path = "config/config.yml"
    yaml_dict = yaml.load(
        open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    cfg = DefaultMunch.fromDict(yaml_dict)
    model = Model(cfg)

    print(model)
