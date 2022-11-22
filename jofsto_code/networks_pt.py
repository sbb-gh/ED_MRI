# (c) Stefano B. Blumberg, do not redistribute or modify this file and helper files


import torch
#import .layers_pt as layers_pt
from .layers_pt import fcnet_pt, get_score_activation, DownsamplingMultLayer

class jofsto(torch.nn.Module):
    def __init__(
        self,
        C_i_values,
        save_model_path,
        n_features,
        epochs,
        epochs_decay,
        epochs_fix_sigma,
        epochs_decay_sigma,
        **kwargs
    ):
        super().__init__()
        self.C_i_values = C_i_values
        self.save_model_path = save_model_path
        self.n_features = n_features
        self.epochs_decay = epochs_decay
        assert epochs_decay <= epochs
        self.alpha_m = 1.0 / epochs_decay

        self.epochs_fix_sigma = epochs_fix_sigma
        assert epochs_fix_sigma <= epochs
        self.epochs_decay_sigma = epochs_decay_sigma
        self.alpha_sigma = 0.5 / epochs_decay_sigma

        self.t = 0

    def assign(self, **kwargs):
        self.downsampling_mult_layer.assign(**kwargs)

    def get(self, *args):
        return self.downsampling_mult_layer.get(*args)

    def on_train_begin(self):
        self.t += 1
        self.epoch = 0
        m = self.get("m")
        print(
            "m has",
            len(torch.where(m == 1)[0]),
            "ones",
            len(torch.where(m == 0)[0]),
            "zeros",
        )
        if self.t == 1:
            self.D_t = self.n_features - self.C_i_values[0]
        else:
            self.D_t = self.C_i_values[self.t - 2] - self.C_i_values[self.t - 1]

        if self.t == 1:
            sigma_mult = 1
        else:
            sigma_mult = 0.5
        self.assign(sigma_mult=sigma_mult)
        print("sigma_mult", sigma_mult)
        self.set_m_decay()

    def on_epoch_begin(self):
        self.epoch += 1
        m, sigma_mult = self.get("m", "sigma_mult")

        if (self.epoch == self.epochs_fix_sigma) and self.t > 1:
            print("Trigger epochs_fix_sigma", flush=True)
            sigma_average, sigma_bar = self.get("sigma_average", "sigma_bar")
            sigma_bar = 0.5 * (sigma_bar + sigma_average)
            # self.assign(sigma_mult=0.0,sigma_bar=sigma_bar); print("sigma_mult=0.0, sigma_bar = 0.5*(sigma_bar+sigma_average)")
            self.assign(sigma_bar=sigma_bar)
            print("sigma_bar = 0.5*(sigma_bar+sigma_average)")

        if self.epoch >= self.epochs_fix_sigma and self.t > 1:
            if sigma_mult > 0:
                sigma_mult = sigma_mult - self.alpha_sigma
                sigma_mult = torch.max(sigma_mult, torch.tensor(0).type_as(sigma_mult))
                print("Decay sigma_mult", float(sigma_mult))
                self.assign(sigma_mult=sigma_mult)

        if (
            self.epoch >= self.epochs_fix_sigma + self.epochs_decay_sigma
        ) and self.t > 1:
            if (
                torch.sum(self.m_decay > 0)
                and torch.max(m[torch.where(self.m_decay == 1)]) > 0
            ):
                print("Decay measurements")
                m = m - self.alpha_m * self.m_decay
                m = torch.max(m, torch.tensor(0).type_as(m))
                self.assign(m=m)

        self.sigma_average_list = []
        self.no_batches = 0

    def on_batch_begin_train(self):
        self.no_batches += 1

    def on_batch_end_train(self):
        sigma = self.get("sigma")
        self.sigma_average_list.append(sigma)

    def on_epoch_end(self):
        sigma_average_list = torch.stack(self.sigma_average_list)
        sigma_average_list = torch.mean(sigma_average_list, axis=0)
        m = self.get("m")
        # print(sigma_average_list*m)
        self.assign(sigma_average=sigma_average_list)

    def on_train_end(self, logs=None):
        if self.t == 1:
            self.assign(sigma_bar=self.get("sigma_average"))

    def set_m_decay(self):
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
        print(
            "Decay:", float(torch.sum(self.m_decay)), "measurements in NAS step", self.t
        )  # could also print measurements to decay


class jofsto_network(jofsto):
    def __init__(
        self,
        n_features,
        out_units,
        num_units_score,
        num_units_task,
        score_activation,
        train_x_median,
        loss_affine_x=(1, 0),
        loss_affine_y=(1, 0),
        **kwargs
    ):
        """Scorer + Predictor Network, End-To-End"""

        super().__init__(n_features=n_features, **kwargs)
        self.score_net = fcnet_pt(
            in_dim=n_features,
            out_dim=n_features,
            inter_units=num_units_score,
            inter_act_fn="relu",
            final_act_fn=None,
            dropout_rate=None,
            inp_loss_affine_0=loss_affine_x[0],
            out_loss_affine_0=None,  # weight_init=None, seed=seed,
        )

        self.task_net = fcnet_pt(
            in_dim=n_features,
            out_dim=out_units,
            inter_units=num_units_task,
            inter_act_fn="relu",
            final_act_fn=None,
            dropout_rate=None,
            inp_loss_affine_0=loss_affine_x[0],
            out_loss_affine_0=loss_affine_y[0],  # weight_init=None, seed=seed,
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
        # print(torch.mean(score,axis=0))
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
