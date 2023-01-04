# (c) Stefano B. Blumberg, do not redistribute or modify this file and helper files


import torch


def return_act_func(act_func):
    """Returns an activation function."""
    if act_func == "relu":
        return torch.nn.ReLU()
    elif act_func == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_func is None:
        return torch.nn.Identity()
    else:
        assert False, "Choose an activation option"


class fcnet_pt(torch.nn.Module):
    """Fully-connected network."""

    def __init__(
        self,
        in_dim,
        out_dim,
        inter_units,
        inter_act_fn="relu",
        final_act_fn=None,
        inp_loss_affine_0=None,
        out_loss_affine_0=None,
    ):
        """
        Args:
        in_dim (int): Number of input units
        out_dim: Number of output units
        inter_units (List[int]): Number of units for intermediate layers
        inter_act_fn (str): Activation function for intermediate layers
        final_act_fn: Actication function on last layers
        inp_loss_affine_0: Affine normalizer multiplier for input
        out_loss_affine_0: Affine normalizer multiplier for output
        """
        super().__init__()
        if inp_loss_affine_0 is None:
            inp_affine_0_prod = 1
        else:
            inp_affine_0_prod = inp_loss_affine_0
        if out_loss_affine_0 is None:
            out_affine_0_prod = 1
        else:
            out_affine_0_prod = out_loss_affine_0
        self.register_buffer("inp_affine_0_prod", torch.tensor(inp_affine_0_prod))
        self.register_buffer("out_affine_0_prod", torch.tensor(out_affine_0_prod))

        layers = []
        if len(inter_units) == 0:
            layers.append(torch.nn.Linear(in_dim, out_dim))
        elif inter_units[0] == -1:
            pass
        else:
            for i, num_outputs in enumerate(inter_units):
                if i == 0:
                    in_features = in_dim
                else:
                    in_features = inter_units[i - 1]

                layers.append(torch.nn.Linear(in_features, num_outputs))
                if inter_act_fn is not None:
                    layers.append(return_act_func(inter_act_fn))

            layers.append(torch.nn.Linear(inter_units[-1], out_dim))

        if final_act_fn is not None:
            layers.append(return_act_func(final_act_fn))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.inp_affine_0_prod is not None:
            x = x / self.inp_affine_0_prod
        x = self.layers(x)
        if self.out_affine_0_prod is not None:
            x = x * self.out_affine_0_prod
        return x


class DownsamplingMultLayer(torch.nn.Module):
    """Subsampling layer also holds JOFSTO variables."""

    def __init__(self, n_features, train_x_median):
        super().__init__()
        self.register_buffer("sigma", torch.zeros(n_features))
        self.register_buffer("sigma_mult", torch.tensor(1.0))
        self.register_buffer("m", torch.ones(n_features))
        self.register_buffer("sigma_bar", torch.ones(n_features))
        self.register_buffer("sigma_average", torch.zeros(n_features))
        self.register_buffer("train_x_median", torch.tensor(train_x_median))

    def assign(self, **kwargs):
        for key, val in kwargs.items():
            if not torch.is_tensor(val):
                val = torch.tensor(val)
            old_val = getattr(self, str(key))
            val = val.clone().to(old_val.device)
            setattr(self, key, val)

    def get(self, *args):
        ret = []
        for key in args:
            out = getattr(self, str(key))
            ret.append(out.clone())
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def forward(self, x_inp, score_inp=1):
        if isinstance(score_inp, torch.Tensor):
            self.sigma = (torch.mean(score_inp, axis=0)).detach()
        score_tot = self.sigma_mult * score_inp + (1 - self.sigma_mult) * self.sigma_bar
        subsample = self.m * x_inp + (1 - self.m) * self.train_x_median
        out = score_tot * subsample
        return out


### Options for activation function


class Sigmoid(torch.nn.Module):
    def __init__(self, mult=1):
        super().__init__()
        self.register_buffer("mult", torch.tensor(mult))

    def forward(self, x):
        return torch.sigmoid(x) * self.mult


class SigmoidRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_low = torch.clamp(x, min=None, max=0)
        y = torch.nn.functional.sigmoid(x_low) * 2
        x_high = torch.clamp(x, min=0, max=None)
        y = y + x_high
        return y


class Exp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


def get_score_activation(score_activation):
    if score_activation == "doublesigmoid":
        return Sigmoid(mult=2)
    elif score_activation == "sigmoidrelu":
        return SigmoidRelu()
    elif score_activation == "exp":
        return Exp()
    elif score_activation == "relu":
        return torch.nn.ReLU()
    else:
        assert False
