import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from tadred import layers

from tadred import data_processing

from omegaconf import OmegaConf

class ModelFittingDataset(torch.utils.data.Dataset):
    """Simple input-target dataset for model fitting."""

    def __init__(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray | None = None,
    ):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        if self.data_y is None:
            return self.data_x[index, :]

        return (
            self.data_x[index, :],
            self.data_y[index, :],
        )

    def __len__(self):
        return self.data_x.shape[0]

    
class ModelFittingNetwork(layers.FCN):
    """
    Conventional neural-network model fitting using the same FCN
    architecture as the TADRED task network.
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        hidden_units,
        inp_loss_affine_0,
        out_loss_affine_0,
    ):
        super().__init__(
            in_dim=n_inputs,
            out_dim=n_outputs,
            inter_units=hidden_units,
            inter_act_fn="relu",
            final_act_fn="identity",
            inp_loss_affine_0=inp_loss_affine_0,
            out_loss_affine_0=out_loss_affine_0,
        )



class ModelFittingTrainer:
    """
    Trainer for conventional neural-network model fitting.

    Matches the TADRED task-network training setup as closely as possible:
        - same FCN implementation
        - same hidden layers
        - ReLU intermediate activations
        - identity output activation
        - same affine normalization
        - MSE loss
        - same Adam configuration
        - same DataLoader configuration
        - same number of epochs
    """

    def __init__(
        self,
        hidden_units,
        train_pytorch,
        epochs,
        no_gpu=False,
        patience=20,
    ):
        self.hidden_units = hidden_units
        self.train_pytorch = train_pytorch
        self.epochs = epochs
        self.no_gpu = no_gpu
        self.patience = patience


    def _create_model(
        self,
        n_inputs,
        n_outputs,
    ):
        self.model = ModelFittingNetwork(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            hidden_units=self.hidden_units,
            inp_loss_affine_0=self.loss_affine_x[0],
            out_loss_affine_0=self.loss_affine_y[0],
        ).to(self.device)

        self.loss_fct = torch.nn.MSELoss()


    def _create_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            **self.train_pytorch.optimizer_params,
        )


    def _create_dataloaders(
        self,
        train_x,
        train_y,
        val_x,
        val_y,
    ):
        train_data = ModelFittingDataset(
            train_x,
            train_y,
        )

        val_data = ModelFittingDataset(
            val_x,
            val_y,
        )

        dataloader_params = OmegaConf.to_container(
            self.train_pytorch.dataloader_params
        )

        self.train_loader = DataLoader(
            train_data,
            **dataloader_params,
        )

        self.val_loader = DataLoader(
            val_data,
            **dataloader_params,
        )


    def _train_epoch(self):
        self.model.train()

        losses = []

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss = self.loss_fct(
                y,
                y_pred,
            )

            loss.backward()

            self.optimizer.step()

            losses.append(
                float(loss)
            )

        return float(
            np.mean(losses)
        )


    def _val_epoch(self):
        self.model.eval()

        losses = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)

                loss = self.loss_fct(
                    y,
                    y_pred,
                )

                losses.append(
                    float(loss)
                )

        return float(
            np.mean(losses)
        )


    def fit(
        self,
        train_x,
        train_y,
        val_x,
        val_y,
    ):
        train_x = np.asarray(
            train_x,
            dtype=np.float32,
        )

        train_y = np.asarray(
            train_y,
            dtype=np.float32,
        )

        val_x = np.asarray(
            val_x,
            dtype=np.float32,
        )

        val_y = np.asarray(
            val_y,
            dtype=np.float32,
        )

        if self.no_gpu:
            self.device = "cpu"
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        # -----------------------------------------
        # Same normalization used by TADRED
        # -----------------------------------------

        self.loss_affine_x = data_processing.calc_affine_norm(
            train_x,
            data_normalization="original-measurement",
        )

        self.loss_affine_y = data_processing.calc_affine_norm(
            train_y,
            data_normalization="original-measurement",
        )

        self._create_model(
            n_inputs=train_x.shape[1],
            n_outputs=train_y.shape[1],
        )

        self._create_optimizer()

        self._create_dataloaders(
            train_x,
            train_y,
            val_x,
            val_y,
        )

        best_val_loss = np.inf
        best_state_dict = None

        epochs_without_improvement = 0

        for epoch in range(
            self.epochs
        ):
            train_loss = (
                self._train_epoch()
            )

            val_loss = (
                self._val_epoch()
            )

            if epoch % 50 == 0:
                print(
                    f"Model fit training "
                    f"Epoch:{epoch} "
                    f"train_loss:{train_loss:.3f} "
                    f"val_loss:{val_loss:.3f}"
                )           

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                best_state_dict = copy.deepcopy(
                    self.model.state_dict()
                )

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= self.patience
            ):
                print(
                    f"Early stopping at epoch {epoch}"
                )
                break

        if best_state_dict is not None:
            self.model.load_state_dict(
                best_state_dict
            )

        return self.model


    def predict(
        self,
        data,
    ):
        data = np.asarray(
            data,
            dtype=np.float32,
        )

        self.model.eval()

        loader = DataLoader(
            ModelFittingDataset(data),
            batch_size=(
                self.train_pytorch
                .dataloader_params
                .batch_size
            ),
            shuffle=False,
        )

        outputs = []

        with torch.no_grad():
            for x in loader:
                x = x.to(
                    self.device
                )

                y_pred = self.model(x)

                outputs.append(
                    y_pred.cpu()
                )

        return (
            torch.cat(outputs)
            .numpy()
        )