# (c) Stefano B. Blumberg, do not redistribute or modify this file and helper files


import copy
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader

from .networks import jofsto_network


class Dataset(torch.utils.data.Dataset):
    """Simple input-target Pytorch dataset."""

    def __init__(self, data_x, data_y=None):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        if self.data_y is not None:
            return self.data_x[index, :], self.data_y[index, :]
        else:
            return self.data_x[index, :]

    def __len__(self):
        return self.data_x.shape[0]


class Trainer:
    """JOFSTO training class."""

    def __init__(
        self,
        jofsto_train_eval,
        network,
        data_features_norm,
        train_pytorch,
        other_options,
    ):
        """
        Args:
        jofsto_train_eval, network, train_pytorch, other_options: Top-level arguments in config file.
        data_features_norm: Information about the data passed to networks.
        """
        self.jofsto_train_eval = jofsto_train_eval
        self.network = network
        self.data_features_norm = data_features_norm
        self.train_pytorch = train_pytorch
        self.other_options = other_options

    def create_model(self):
        model = jofsto_network(
            **self.network, **self.data_features_norm, jofsto_train_eval=self.jofsto_train_eval
        )
        print(model)
        self.model = model.to(self.device)

    def create_optimizer(self):
        # Only have optimizer for now
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), **self.train_pytorch["optimizer_params"]
        )

    def create_dataloaders(self, train_x, train_y, val_x, val_y):
        """Create PyTorch dataloaders for training and validation."""
        train_data = Dataset(train_x, train_y)
        val_data = Dataset(val_x, val_y)
        self.train_loader = DataLoader(train_data, **self.train_pytorch["dataloader_params"])
        self.val_loader = DataLoader(val_data, **self.train_pytorch["dataloader_params"])

    def train_val_epoch(self, epoch):
        """Train and validate a single JOFSTO epoch."""
        train_losses = []
        self.model.train()
        self.model.on_epoch_begin()
        start_epoch = timeit.default_timer()
        # Forwards and backwards pass for training epoch
        for i, (x, y) in enumerate(self.train_loader):
            self.model.on_batch_begin_train()
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.forward_and_backward(x, y)
            self.optimizer.step()
            train_losses.append(float(loss))
            self.model.on_batch_end_train()
        # Forwards and backwards pass for validation epoch
        with torch.no_grad():
            val_losses = []
            self.model.eval()
            for i, (x, y) in enumerate(self.val_loader):
                x, y = x.to(self.device), y.to(self.device)
                loss = self.model.forward_and_loss(x, y)
                val_losses.append(float(loss))
        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)

        self.model.on_epoch_end()
        print(
            "Epoch:{:.0f} train_loss:{:.3f} val_loss:{:.3f} time:{:.3f}".format(
                epoch, mean_train, mean_val, timeit.default_timer() - start_epoch
            )
        )
        return mean_train, mean_val

    def early_stopping(self, epoch, epoch_val):
        """Early stopping on validation performance for training."""
        patience = 20
        start_es = (
            self.jofsto_train_eval["epochs_fix_sigma"]
            + self.jofsto_train_eval["epochs_decay_sigma"]
            + self.jofsto_train_eval["epochs_decay"]
            + 1
        )
        if epoch >= start_es:
            if epoch_val[-1] == min(epoch_val[start_es:]):
                print("Cached best_state_dict")
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
        if len(epoch_val) > (start_es + patience):
            if epoch_val[-patience - 1] < min(epoch_val[-patience:]):
                return True
        return False

    def train_val_step(self):
        """Perform JOFSTO step t."""
        epoch_val = []
        timer_t = timeit.default_timer()
        self.model.on_train_begin()
        self.best_state_dict = None
        for epoch in range(self.jofsto_train_eval["epochs"]):
            mean_train, mean_val = self.train_val_epoch(epoch=epoch)
            epoch_val.append(mean_val)
            if self.early_stopping(epoch, epoch_val):
                break

        self.model.load_state_dict(self.best_state_dict)
        self.model.on_train_end()
        print(
            "Finished training step{:.0f} m {:.0f}  step time:{:.3f}".format(
                self.model.t,
                float(torch.sum(self.model.get("m") == 1)),
                timeit.default_timer() - timer_t,
            )
        )

    def eval_step(self, data_x, data_y):
        """Evaluation on test data."""
        m, sigma_bar, sigma_mult = self.model.get("m", "sigma_bar", "sigma_mult")
        print(f"m {len(torch.where(m==1)[0])} ones {len(torch.where(m==0)[0])} zeros")
        data_x = data_x * m.cpu().numpy()  # just to make sure

        self.model.eval()
        with torch.no_grad():
            out = []
            loader = torch.utils.data.DataLoader(
                Dataset(data_x),
                batch_size=self.train_pytorch["dataloader_params"]["batch_size"],
                shuffle=False,
            )
            for i, x in enumerate(loader):
                x = x.to(self.device)

                y_pred = self.model.forward_eval(x, score=1)
                out.append(y_pred.cpu())
            pred_all = torch.cat(out).detach()
            loss = self.model.loss_fct(torch.tensor(data_y), pred_all)
        return float(loss), pred_all.numpy()

    def train_val_test_all(self, train_x, train_y, val_x, val_y, test_x, test_y):
        """Complete optimization for JOFSTO."""
        if self.other_options["no_gpu"]:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Run training on: {self.device}", flush=True)

        self.create_model()
        self.create_optimizer()
        self.create_dataloaders(train_x, train_y, val_x, val_y)

        results = dict()

        # Step t=1,...,T feature set size C_i
        for t, C_i in enumerate(self.jofsto_train_eval["C_i_values"], 1):
            self.train_val_step()

            if C_i in self.jofsto_train_eval["C_i_eval"]:

                val_joint, val_pred = self.eval_step(val_x, val_y)
                test_joint, test_pred = self.eval_step(test_x, test_y)

                m, sigma_bar = self.model.get("m", "sigma_bar")
                m, sigma_bar = m.cpu().numpy(), sigma_bar.cpu().numpy()
                measurements = np.where(m == 1)[0]

                print(f"val_joint {val_joint} test_joint {test_joint}")
                results[C_i] = dict(
                    val_joint=val_joint,
                    test_joint=test_joint,
                    measurements=measurements,
                    sigma_bar=sigma_bar,
                )
                if self.other_options["save_output"]:
                    results[C_i]["test_output"] = test_pred

        print("End of Training")
        return results
