# (c) Stefano B. Blumberg, do not redistribute or modify this file and helper files


import torch

from .layers import DownsamplingMultLayer, fcnet_pt, get_score_activation


class jofsto(torch.nn.Module):
    """Super class for JOFSTO-specific optimization/training."""

    def __init__(
        self,
        C_i_values,
        C_i_eval,
        epochs,
        epochs_decay,
        epochs_fix_sigma,
        epochs_decay_sigma,
    ):
        """Arguments from paper, see jofsto_train_eval in config file."""
        super().__init__()
        self.C_i_values = C_i_values
        self.n_features = self.C_i_values[0]

        self.epochs_decay = epochs_decay
        self.alpha_m = 1.0 / epochs_decay

        self.epochs_fix_sigma = epochs_fix_sigma
        self.epochs_decay_sigma = epochs_decay_sigma
        self.alpha_sigma = 0.5 / epochs_decay_sigma

        assert epochs_fix_sigma + epochs_decay_sigma + epochs_decay < epochs
        for C_i in C_i_eval:
            if C_i not in C_i_values:
                print(f"Put {C_i} in C_i_eval arguments")

        self.t = 0  # Time step of outer loop

    def assign(self, **kwargs):
        self.downsampling_mult_layer.assign(**kwargs)

    def get(self, *args):
        return self.downsampling_mult_layer.get(*args)

    def on_train_begin(self):
        """Call at beginning of each step t=1,...T of JOFSTO training."""
        self.t += 1
        self.epoch = 0
        m = self.get("m")
        print(f"m has {len(torch.where(m == 1)[0])} ones and {len(torch.where(m == 0)[0])} zeros")
        # Number of measurements to remove this step
        if self.t == 1:
            self.D_t = self.n_features - self.C_i_values[0]
        else:
            self.D_t = self.C_i_values[self.t - 2] - self.C_i_values[self.t - 1]
        if self.t == 1:
            sigma_mult = 1
        else:
            sigma_mult = 0.5
        self.assign(sigma_mult=sigma_mult)
        print(f"sigma_mult {sigma_mult}")
        self.set_m_decay()

    def on_epoch_begin(self):
        """Call at the beginning of each epoch."""
        self.epoch += 1
        m, sigma_mult = self.get("m", "sigma_mult")

        if (self.epoch == self.epochs_fix_sigma) and self.t > 1:
            print("Trigger epochs_fix_sigma", flush=True)
            sigma_average, sigma_bar = self.get("sigma_average", "sigma_bar")
            sigma_bar = 0.5 * (sigma_bar + sigma_average)
            self.assign(sigma_bar=sigma_bar)
            print("sigma_bar = 0.5*(sigma_bar+sigma_average)")

        if self.epoch >= self.epochs_fix_sigma and self.t > 1:
            if sigma_mult > 0:
                sigma_mult = sigma_mult - self.alpha_sigma
                sigma_mult = torch.max(sigma_mult, torch.tensor(0).type_as(sigma_mult))
                print("Decay sigma_mult", float(sigma_mult))
                self.assign(sigma_mult=sigma_mult)

        if (self.epoch >= self.epochs_fix_sigma + self.epochs_decay_sigma) and self.t > 1:
            if torch.sum(self.m_decay > 0) and torch.max(m[torch.where(self.m_decay == 1)]) > 0:
                print("Decay measurements")
                m = m - self.alpha_m * self.m_decay
                m = torch.max(m, torch.tensor(0).type_as(m))
                self.assign(m=m)

        self.sigma_average_list = []
        self.no_batches = 0

    def on_batch_begin_train(self):
        """Call at batch beginning to count number of batches."""
        self.no_batches += 1

    def on_batch_end_train(self):
        """Call at batch end to cache learnt score in batch."""
        sigma = self.get("sigma")
        self.sigma_average_list.append(sigma)

    def on_epoch_end(self):
        """Call at epoch end to compute averaged score across batch."""
        sigma_average_list = torch.stack(self.sigma_average_list)
        sigma_average_list = torch.mean(sigma_average_list, axis=0)
        self.assign(sigma_average=sigma_average_list)

    def on_train_end(self, logs=None):
        """Call at the end of step t=1,...,T to cache averaged score."""
        if self.t == 1:
            self.assign(sigma_bar=self.get("sigma_average"))

    def set_m_decay(self):
        """Choose and set features to remove m_{t-1}-m_{t}."""
        m, sigma_bar = self.get("m", "sigma_bar")
        self.m_decay = torch.zeros(self.n_features, dtype=torch.int).to(m.device)
        # Decay self.D[t] measurements in NAS step t
        # D_t = self.D_t # D_{t} in paper
        if self.D_t > 0:
            # Find indices smallest sigma_bar where m!=0
            assert torch.sum((0 < m) & (m < 1)) == 0, "m has values between 0,1"

            m_decay_options = torch.where(m == 1)
            m_decay_options_sigma = sigma_bar[m_decay_options]
            m_decay_options_sigma = torch.argsort(m_decay_options_sigma)[: self.D_t]
            D = m_decay_options[0][m_decay_options_sigma]
            self.m_decay[D] = 1
        print(f"Decay: {float(torch.sum(self.m_decay))} measurements in NAS step {self.t}")


class jofsto_network(jofsto):
    def __init__(
        self,
        num_units_score,
        num_units_task,
        score_activation,
        n_features,
        out_units,
        train_x_median,
        loss_affine_x,
        loss_affine_y,
        jofsto_train_eval,
    ):
        """Define Scoring and Prediction Networks with forward pass.

        Args:
        num_units_score, num_units_task, score_activation: Config file network parameters
        n_features, out_units, train_x_median, loss_affine_x, loss_affine_y: Data features
        jofsto_train_eval: Config file jofsto_train_eval parameters
        """
        super().__init__(**jofsto_train_eval)
        self.score_net = fcnet_pt(
            in_dim=n_features,
            out_dim=n_features,
            inter_units=num_units_score,
            inter_act_fn="relu",
            final_act_fn=None,
            inp_loss_affine_0=loss_affine_x[0],
            out_loss_affine_0=None,
        )

        self.task_net = fcnet_pt(
            in_dim=n_features,
            out_dim=out_units,
            inter_units=num_units_task,
            inter_act_fn="relu",
            final_act_fn=None,
            inp_loss_affine_0=loss_affine_x[0],
            out_loss_affine_0=loss_affine_y[0],
        )

        self.score_activation = get_score_activation(score_activation)
        self.downsampling_mult_layer = DownsamplingMultLayer(
            n_features=n_features,
            train_x_median=train_x_median,
        )
        self.loss_fct = torch.nn.MSELoss()

    def forward_score(self, x_inp):
        score = self.score_net(x_inp)
        score = self.score_activation(score)
        return score

    def forward_eval(self, x_inp, score):
        x_subsampled_weighted = self.downsampling_mult_layer(x_inp, score)
        x = self.task_net(x_subsampled_weighted)
        return x

    def forward(self, x):
        score = self.forward_score(x)
        y = self.forward_eval(x, score)
        return y

    def forward_and_loss(self, x, y):
        y_hat = self.forward(x)
        loss = self.loss_fct(y, y_hat)
        return loss

    def forward_and_backward(self, x, y):
        loss = self.forward_and_loss(x, y)
        loss.backward()
        return loss
