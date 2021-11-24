import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import wandb
import matplotlib.pyplot as plt
from models.vit import TimmViT
from torch import Tensor
from torchmetrics import Accuracy
from torch.optim import Adam
from data.cifar_imbalanced import IMBALANCECIFAR10, IMBALANCECIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader


class Transformer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

        self.transformer = self.get_model()

        # Loss Function
        self.loss_func = self.get_loss_function()

        # Metrics
        self.acc_train = Accuracy()
        self.acc_valid = Accuracy()
        self.acc_test = Accuracy()

        self.acc_per_class_train = nn.ModuleList([Accuracy() for _ in range(self.hparams.num_classes)])
        self.acc_per_class_valid = nn.ModuleList([Accuracy() for _ in range(self.hparams.num_classes)])
        self.acc_per_class_test = nn.ModuleList([Accuracy() for _ in range(self.hparams.num_classes)])

        # data
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def get_model(self) -> nn.Module:
        """
        Get model based on provider
        :return:
        """
        if self.hparams.model_provider == 'timm':
            return TimmViT(self.hparams.transformer_model, self.hparams.pretrained, self.hparams.num_classes)
        else:
            raise Exception(f"Unsupported Provider {self.hparams.model_provider}")

    def get_loss_function(self) -> nn.Module:
        """
        Get loss function
        :return:
        """
        if self.hparams.loss_function == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.hparams.loss_function == 'weighted_cross_entropy':
            return nn.CrossEntropyLoss(weight=Tensor(self.hparams.class_weights).to(self.device))
        else:
            raise Exception(f"Unsupported Loss Function: {self.hparams.loss_function}")

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Expected Shape: (N, 3, H, W)
            where:
            - N: batch size
            - H: height
            - W: width
        :return: Expected Shape: (N, num_classes)
        """
        return self.transformer(x)

    def loss_function(self, logits: Tensor, y: Tensor):
        """
        :param logits: (N, num_classes)
        :param y: (N,)
        :return: loss value, a scalar
        """
        return self.loss_func(logits, y)

    def configure_optimizers(self):
        """
        Create Optimizers
        :return:
        """
        if self.hparams.optimizer == 'Adam':
            return Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        else:
            raise Exception(f"Unsupported Optimizer type: {self.hparams.optimizer}")

    def training_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)

        self.log('loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True)

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

                self.acc_train(y_pred, y)
                # calculate per class accuracy
                for c in range(self.hparams.num_classes):
                    indexes = (y == c)
                    y_c = y[indexes]
                    y_pred_c = y_pred[indexes]
                    # if this batch does not have samples from this class then ignore
                    if y_c.numel() == 0 or y_pred_c.numel() == 0:
                        continue
                    self.acc_per_class_train[c](y_pred_c, y_c)

            self.log('train_accuracy', self.acc_train.compute())
            for c in range(self.hparams.num_classes):
                if self.acc_per_class_train[c].mode:
                    self.log(f'class_{c}_train_accuracy', self.acc_per_class_train[c].compute())

        y = torch.cat(ys)
        y_pred = torch.cat(y_preds)

        self.logger.experiment.log({
            "confusion_matrix_training": wandb.plot.confusion_matrix(
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

        self.log('val_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True)

        self.acc_valid(logits, y)

        for c in range(self.hparams.num_classes):
            indexes = (y == c)
            y_c = y[indexes]
            logits_c = logits[indexes]
            if y_c.numel() == 0 or logits_c.numel() == 0:
                continue
            self.acc_per_class_valid[c](logits_c, y_c)

        results = {
            'loss': loss,
            'y': y,
            'y_pred': y_pred
        }

        return results

    def validation_epoch_end(self, val_step_outputs):

        self.log('val_accuracy', self.acc_valid.compute())
        for c in range(self.hparams.num_classes):
            if self.acc_per_class_valid[c].mode:
                self.log(f'class_{c}_val_accuracy', self.acc_per_class_valid[c].compute())
        steps = len(val_step_outputs)

        y = [None] * steps
        y_pred = [None] * steps

        for i, result in enumerate(val_step_outputs):
            y[i] = result['y']
            y_pred[i] = result['y_pred']

        y = torch.cat(y).cpu().tolist()
        y_pred = torch.cat(y_pred).cpu().tolist()

        self.logger.experiment.log({
            "confusion_matrix_validation": wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.array(y), preds=np.array(y_pred),
                class_names=self.hparams.class_names)
        })

    def test_step(self, batch, batch_idx):

        y_pred, y = self.validation_step(batch, -1)
        self.acc_test(y_pred, y)
        for c in range(self.hparams.num_classes):
            indexes = (y == c)
            y_c = y[indexes]
            y_pred_c = y_pred[indexes]
            if y_c.numel() == 0 or y_pred_c.numel() == 0:
                continue
            self.acc_per_class_test[c](y_pred_c, y_c)

        results = {
            'y': y,
            'y_pred': y_pred
        }
        return results

    def test_epoch_end(self, outputs):

        self.log('test_accuracy', self.acc_test.compute())
        per_class_accuracies = []
        for c in range(self.hparams.num_classes):
            if self.acc_per_class_test[c].mode:
                acc = self.acc_per_class_test[c].compute()
                per_class_accuracies.extend([acc.item()])
                self.log(f'class_{c}_test_accuracy', acc)
        steps = len(outputs)

        plt.title('Per Class Accuracy')
        plt.xlabel('Classes Sorted by Frequency')
        plt.ylabel('Top-1 Accuracy')
        plt.plot(np.arange(self.hparams.num_classes), per_class_accuracies)

        self.logger.experiment.log({
            'per_class_test_accuracies': plt
        })

        y = [None] * steps
        y_pred = [None] * steps

        for i, result in enumerate(outputs):
            y[i] = result['y']
            y_pred[i] = result['y_pred']

        y = torch.cat(y).cpu().tolist()
        y_pred = torch.cat(y_pred).cpu().tolist()

        # plot confusion matrix in wandb style
        self.logger.experiment.log({
            "confusion_matrix_test": wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.array(y), preds=np.array(y_pred),
                class_names=self.hparams.class_names)
        })

        # plot confusion matrix in sklearn style
        self.logger.experiment.log({
            'confusion_matrix_sklearn_test': wandb.sklearn.plot_confusion_matrix(y, y_pred, self.hparams.class_names)
        })

    def setup(self, stage):

        cifar_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.hparams.img_size, self.hparams.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        cifar_transform_valid = transforms.Compose([
            transforms.Resize((self.hparams.img_size, self.hparams.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if self.hparams.dataset == 'cifar10':
            self.train_dataset = IMBALANCECIFAR10(
                root=self.hparams.data_root,
                imb_type=self.hparams.imb_type,
                imb_factor=self.hparams.imb_factor,
                rand_number=self.hparams.random_seed,
                train=True,
                transform=cifar_transform_train,
                download=True
            )
            self.valid_dataset = IMBALANCECIFAR10(
                root=self.hparams.data_root,
                imb_type=self.hparams.imb_type,
                imb_factor=self.hparams.imb_factor,
                rand_number=self.hparams.random_seed,
                train=False,
                transform=cifar_transform_valid,
                download=True,
                test=False,
                test_ratio=self.hparams.test_ratio
            )
            self.test_dataset = IMBALANCECIFAR10(
                root=self.hparams.data_root,
                imb_type=self.hparams.imb_type,
                imb_factor=self.hparams.imb_factor,
                rand_number=self.hparams.random_seed,
                train=False,
                transform=cifar_transform_valid,
                download=True,
                test=True,
                test_ratio=self.hparams.test_ratio
            )
        elif self.hparams.dataset == 'cifar100':
            self.train_dataset = IMBALANCECIFAR100(
                root=self.hparams.data_root,
                imb_type=self.hparams.imb_type,
                imb_factor=self.hparams.imb_factor,
                rand_number=self.hparams.random_seed,
                train=True,
                transform=cifar_transform_train,
                download=True
            )
            self.valid_dataset = IMBALANCECIFAR100(
                root=self.hparams.data_root,
                imb_type=self.hparams.imb_type,
                imb_factor=self.hparams.imb_factor,
                rand_number=self.hparams.random_seed,
                train=False,
                transform=cifar_transform_valid,
                download=True,
                test=False,
                test_ratio=self.hparams.test_ratio
            )
            self.test_dataset = IMBALANCECIFAR100(
                root=self.hparams.data_root,
                imb_type=self.hparams.imb_type,
                imb_factor=self.hparams.imb_factor,
                rand_number=self.hparams.random_seed,
                train=False,
                transform=cifar_transform_valid,
                download=True,
                test=True,
                test_ratio=self.hparams.test_ratio
            )
        else:
            raise Exception(f"Unsupported Dataset {self.hparams.dataset}")

    def train_dataloader(self):

        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          shuffle=True)

    def val_dataloader(self):

        return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):

        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)





