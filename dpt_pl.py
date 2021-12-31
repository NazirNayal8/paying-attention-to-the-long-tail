import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import wandb
from torch import Tensor
from torch.optim import Adam, SGD
from models.dpt.models import DPTSegmentationModel
from torchmetrics import Accuracy, JaccardIndex, F1, Precision, Recall



class DPT(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

        self.model = DPTSegmentationModel(self.hparams)

        # Loss Function
        self.loss_func = self.get_loss_function()

        # Metrics

        # Accuracy
        self.pixel_acc_train = Accuracy(num_classes=self.hparams.num_classes)
        self.pixel_acc_valid = Accuracy(num_classes=self.hparams.num_classes)

        # mIoU
        self.mIoU_train = JaccardIndex()


    def get_loss_function(self) -> nn.Module:
        """
        Get Loss Function
        :return:
        """
        if self.hparams.loss_function == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise Exception(f"Unsupported Loss Function: {self.hparams.loss_function}")

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Expected Shape: (N, 3, H, W)
            where:
            - N: batch size
            - H: height
            - W: width
        :return: Expected Shape (N, )
        """
        return self.model(x)

    def compute_loss(self, logits: Tensor, y: Tensor):
        """
        :param logits:
        :param y:
        :return:
        """

        return self.loss_func(logits, y)

    def configure_optimizers(self):
        """
        Create Optimizers
        :return:
        """
        if self.hparams.optimizer == 'Adam':
            return Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'SGD':
            return SGD(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise Exception(f"Unsupported Optimizer type: {self.hparams.optimizer}")

    def training_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)

        self.log('train/loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True)

        results = {
            'loss': loss,
            'x': x.cpu(),
            'y': y.cpu()
        }
        return results

    def training_epoch_end(self, training_step_outputs):

        self.eval()
        steps = len(training_step_outputs)
        ys = [None] * steps
        y_preds = [None] * steps

        with torch.no_grad():
            for i, result in enumerate(training_step_outputs):
                x, y = result['x'], result['y']
                batch = (x.to(self.device), y.to(self.device))

                y_pred, y = self.validation_step(batch, -1)

                ys[i] = y
                y_preds[i] = y_pred

        y = torch.cat(ys)
        y_pred = torch.cat(y_preds)

        if self.logger is not None:
            self.logger.experiment.log({
                "train/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=np.array(y.cpu().tolist()),
                    preds=np.array(y_pred.cpu().tolist()),
                    class_names=self.hparams.class_names)
            })

    def validation_step(self, batch, batch_idx):

        x, y = batch

        logits = self(x)
        loss = self.loss_func(logits, y)
        y_pred = torch.argmax(logits, dim=1)

        if batch_idx == -1:
            return y_pred, y

        results = {
            'loss': loss,
            'y': y,
            'y_pred': y_pred
        }

        return results

    def validation_epoch_end(self, val_step_outputs):

        steps = len(val_step_outputs)

        y = [None] * steps
        y_pred = [None] * steps

        for i, result in enumerate(val_step_outputs):
            y[i] = result['y']
            y_pred[i] = result['y_pred']

        y = torch.cat(y).cpu().tolist()
        y_pred = torch.cat(y_pred).cpu().tolist()

        if self.logger is not None:
            self.logger.experiment.log({
                "val/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=np.array(y), preds=np.array(y_pred),
                    class_names=self.hparams.class_names)
            })

    def test_step(self, batch, batch_idx):

        y_pred, y = self.validation_step(batch, -1)

        results = {
            'y': y,
            'y_pred': y_pred
        }
        return results

    def test_epoch_end(self, outputs):

        steps = len(outputs)

        y = [None] * steps
        y_pred = [None] * steps

        for i, result in enumerate(outputs):
            y[i] = result['y']
            y_pred[i] = result['y_pred']

        y = torch.cat(y).cpu().tolist()
        y_pred = torch.cat(y_pred).cpu().tolist()

        if self.logger is not None:
            self.logger.experiment.log({
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=np.array(y), preds=np.array(y_pred),
                    class_names=self.hparams.class_names)
            })

        # plot confusion matrix in sklearn style
        if self.logger is not None:
            self.logger.experiment.log({
                'test/confusion_matrix_sklearn': wandb.sklearn.plot_confusion_matrix(y, y_pred, self.hparams.class_names)
            })

    def setup(self, stage):
        pass


    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass