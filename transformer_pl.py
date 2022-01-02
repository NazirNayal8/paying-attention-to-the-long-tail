import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import wandb
import matplotlib.pyplot as plt
from models.vit import TimmViT
from torch import Tensor
from torchmetrics import Accuracy, Precision, Recall, F1, AveragePrecision
from torch.optim import Adam, SGD
from data.cifar_imbalanced import IMBALANCECIFAR10, IMBALANCECIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from metrics.classification import KShotAccuracy, KShotPrecision, KShotF1, KShotmAP
from losses.focal_loss import focal_loss
from losses.class_balanced_loss import ClassBalancedLoss


class Transformer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        config = self.add_run_dependent_parameters(config)

        self.save_hyperparameters(config)

        self.transformer = self.get_model()

        # Loss Function
        self.loss_func = self.get_loss_function()

        # Metrics

        # Accuracy
        self.acc_train = Accuracy()
        self.acc_valid = Accuracy()

        # Precision, Recall, F1, mAP
        self.precision_valid = Precision(num_classes=self.hparams.num_classes, average='macro')
        self.recall_valid = Recall(num_classes=self.hparams.num_classes, average='macro')
        self.f1_valid = F1(num_classes=self.hparams.num_classes, average='macro')

        self.precision_weighted_valid = Precision(num_classes=self.hparams.num_classes, average='weighted')
        self.recall_weighted_valid = Recall(num_classes=self.hparams.num_classes, average='weighted')
        self.f1_weighted_valid = F1(num_classes=self.hparams.num_classes, average='weighted')

        self.mAP_valid = AveragePrecision(num_classes=self.hparams.num_classes)

        # Per Class Metrics
        self.acc_per_class_train = Accuracy(num_classes=self.hparams.num_classes, average=None)
        self.acc_per_class_valid = Accuracy(num_classes=self.hparams.num_classes, average=None)

        self.precision_per_class_valid = Precision(num_classes=self.hparams.num_classes, average=None)
        self.f1_per_class_valid = F1(num_classes=self.hparams.num_classes, average=None)
        self.ap_per_class_valid = AveragePrecision(num_classes=self.hparams.num_classes, average=None)

        # Few-Medium-Many Shot Metrics
        few_shot_classes, medium_shot_classes, many_shot_classes = self.get_few_medium_many_shot_partitions()

        self.kshot_acc_train = KShotAccuracy(few_shot_classes, medium_shot_classes, many_shot_classes)

        self.kshot_acc_valid = KShotAccuracy(few_shot_classes, medium_shot_classes, many_shot_classes)
        self.kshot_precision_valid = KShotPrecision(few_shot_classes, medium_shot_classes, many_shot_classes)
        self.kshot_f1_valid = KShotF1(few_shot_classes, medium_shot_classes, many_shot_classes)
        self.kshot_map_valid = KShotmAP(few_shot_classes, medium_shot_classes, many_shot_classes)

        self.few_shot_classes = few_shot_classes
        self.medium_shot_classes = medium_shot_classes
        self.many_shot_classes = many_shot_classes

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

    def get_few_medium_many_shot_partitions(self):
        """
        Return few, medium and many shot class partitions depending on the chosen strategy in hparams.
        :return:
        """
        strategy = self.hparams.few_medium_many_partition_strategy
        if  strategy == '30-40-30':

            limit_30 = int(self.hparams.num_classes * 0.3)
            limit_70 = int(self.hparams.num_classes * 0.7)
            all_classes = np.arange(self.hparams.num_classes)
            few_shot_classes = all_classes[limit_70:]
            medium_shot_classes = all_classes[limit_30:limit_70]
            many_shot_classes = all_classes[:limit_30]

            return few_shot_classes, medium_shot_classes, many_shot_classes
        else:
            raise Exception(f"Few-Medium-Many Partition Strategy {strategy} undefined.")

    def add_run_dependent_parameters(self, config):
        """
        This function is used to add parameters to the config dict where these parameters dependent on the chosen
        hyperparameters.
        :param config:
        :return:
        """
        config = self.add_per_class_frequency(config)
        config = self.add_class_weights(config)
        return config

    def add_per_class_frequency(self, config):
        """
        This function calculates the per class frequency of the dataset and adds it to the config dict
        :param config:
        :return:
        """

        if config['dataset'] == 'cifar10':
            train_dataset = IMBALANCECIFAR10(
                root=config['data_root'],
                imb_type=config['imb_type'],
                imb_factor=config['imb_factor'],
                rand_number=config['random_seed'],
                train=True,
                transform=None,
                download=True
            )
        elif config['dataset'] == 'cifar100':
            train_dataset = IMBALANCECIFAR100(
                root=config['data_root'],
                imb_type=config['imb_type'],
                imb_factor=config['imb_factor'],
                rand_number=config['random_seed'],
                train=True,
                transform=None,
                download=True
            )
        else:
            raise Exception(f'Dataset {config["dataset"]} undefined.')

        config['per_class_frequency'] = np.array(train_dataset.per_class_frequency)
        return config

    def add_class_weights(self, config):
        """
        This function calculates the class weights based on a chosen pre-defined strategy and appends the weights to
        the config dictionary
        :param config:
        :return:
        """

        if 'per_class_frequency' not in config:
            raise Exception('Class Weights cannot be computed without per_class_frequency')

        per_class_frequency = config['per_class_frequency']

        if config['class_weights_strategy'] == 'frequency':

            class_weights = np.sum(per_class_frequency) / (config['num_classes'] * per_class_frequency)
            config['class_weights'] = class_weights
        else:
            raise Exception(f'Class Weights Computation Strategy {config["class_weights_strategy"]} undefined')
        return config

    def get_loss_function(self) -> nn.Module:
        """
        Get loss function
        :return:
        """
        if self.hparams.loss_function == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.hparams.loss_function == 'weighted_cross_entropy':
            return nn.CrossEntropyLoss(weight=Tensor(self.hparams.class_weights).to(self.device))
        elif self.hparams.loss_function == 'focal_loss':
            if self.hparams.class_weights is None:
                alpha = None
            else:
                alpha = torch.Tensor(self.hparams.class_weights).to(self.device)
            return focal_loss(alpha=alpha, gamma=self.hparams.focal_loss_gamma)
        elif self.hparams.loss_function == 'class_balanced_loss':
            return ClassBalancedLoss(
                loss_type=self.hparams.class_balanced_loss_type,
                beta=self.hparams.class_balanced_beta,
                num_classes=self.hparams.num_classes,
                per_class_frequency=self.hparams.per_class_frequency,
                focal_loss_gamma=self.hparams.focal_loss_gamma,
                device=self.device
            )
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
        elif self.hparams.optimizer == 'SGD':
            return SGD(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise Exception(f"Unsupported Optimizer type: {self.hparams.optimizer}")

    def mixup_data(self, x, y):
        """
        Returns mixed inputs, pairs of targets, and lambda'''
        """

        alpha = self.hparams.mixup_alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.shape[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def training_step(self, batch, batch_idx):

        x, y = batch

        if self.hparams.mixup and self.current_epoch < self.hparams.mixup_stop_epoch:
            x_mixed, y_a, y_b, lam = self.mixup_data(x, y)
            logits = self(x_mixed)
            loss = lam * self.loss_function(logits, y_a) + (1 - lam) * self.loss_function(logits, y_b)
        else:
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

                y_pred, y, logits = self.validation_step(batch, -1)

                ys[i] = y
                y_preds[i] = y_pred

                self.acc_train(y_pred, y)
                if self.hparams.per_class_metrics:
                    self.acc_per_class_train(y_pred, y)
                if self.hparams.kshot_metrics:
                    self.kshot_acc_train(y_pred, y)

            self.log('train/accuracy', self.acc_train.compute())

            if self.hparams.per_class_metrics:
                acc_per_class = self.acc_per_class_train.compute()
                for c in range(self.hparams.num_classes):
                    self.log(f'train/class_{self.hparams.class_names[c]}-{c}_accuracy', acc_per_class[c])

            if self.hparams.kshot_metrics:
                few_acc, medium_acc, many_acc = self.kshot_acc_train.compute()
                self.log('train/few_shot_acc', few_acc)
                self.log('train/medium_shot_acc', medium_acc)
                self.log('train/many_shot_acc', many_acc)

        y = torch.cat(ys).cpu().numpy()
        y_pred = torch.cat(y_preds).cpu().numpy()

        if self.hparams.per_class_metrics:
            self.logger.experiment.log({
                "train/confusion_matrix_per_class": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y,
                    preds=y_pred,
                    class_names=self.hparams.class_names)
            })

            self.logger.experiment.log({
                'train/confusion_matrix_per_class_sklearn':
                    wandb.sklearn.plot_confusion_matrix(y, y_pred, self.hparams.class_names)
            })

        if self.hparams.kshot_metrics:

            y_true = y
            preds = y_pred

            few_index = np.isin(y_true, self.few_shot_classes)
            medium_index = np.isin(y_true, self.medium_shot_classes)
            many_index = np.isin(y_true, self.many_shot_classes)

            y_true[few_index] = 0
            y_true[medium_index] = 1
            y_true[many_index] = 2

            few_index = np.isin(preds, self.few_shot_classes)
            medium_index = np.isin(preds, self.medium_shot_classes)
            many_index = np.isin(preds, self.many_shot_classes)

            preds[few_index] = 0
            preds[medium_index] = 1
            preds[many_index] = 2

            self.logger.experiment.log({
                "train/confusion_matrix_kshot": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=preds,
                    class_names=['Few Shot', 'Medium Shot', 'Many Shot'])
            })

            self.logger.experiment.log({
                'train/confusion_matrix_kshot_sklearn':
                    wandb.sklearn.plot_confusion_matrix(y_true, preds, ['Few Shot', 'Medium Shot', 'Many Shot'])
            })

        self.acc_per_class_train.reset()
        self.acc_train.reset()
        self.kshot_acc_train.reset()

    def validation_step(self, batch, batch_idx):

        x, y = batch

        logits = self(x)
        loss = self.loss_func(logits, y)

        y_pred = torch.argmax(logits, dim=1)

        if batch_idx == -1:
            return y_pred, y, logits

        self.log('val/loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True)

        # log accuracy
        self.acc_valid(logits, y)
        # log mAP
        self.mAP_valid(logits, y)
        # log precision and recall
        self.precision_valid(logits, y)
        self.recall_valid(logits, y)
        self.precision_weighted_valid(logits, y)
        self.recall_weighted_valid(logits, y)
        # log F1 Score
        self.f1_valid(logits, y)
        self.f1_weighted_valid(logits, y)

        if self.hparams.per_class_metrics:
            self.acc_per_class_valid(logits, y)
            self.f1_per_class_valid(logits, y)
            self.precision_per_class_valid(logits, y)
            self.ap_per_class_valid(logits, y)

        if self.hparams.kshot_metrics:
            self.kshot_acc_valid(y_pred, y)
            self.kshot_map_valid(logits, y)
            self.kshot_precision_valid(y_pred, y)
            self.kshot_f1_valid(y_pred, y)

        results = {
            'loss': loss,
            'y': y,
            'y_pred': y_pred,
            'logits': logits
        }

        return results

    def validation_epoch_end(self, val_step_outputs):

        # log accuracy
        self.log('val/accuracy', self.acc_valid.compute())
        self.acc_valid.reset()
        # log map
        self.log('val/mAP', self.mAP_valid.compute())
        self.mAP_valid.reset()
        # log precision, recall, F1 Score
        self.log('val/precision', self.precision_valid.compute())
        self.precision_valid.reset()
        self.log('val/recall', self.recall_valid.compute())
        self.recall_valid.reset()
        self.log('val/F1_score', self.f1_valid.compute())
        self.f1_valid.reset()
        # log precision, recall, F1 Score
        self.log('val/precision_weighted', self.precision_weighted_valid.compute())
        self.precision_weighted_valid.reset()
        self.log('val/recall_weighted', self.recall_weighted_valid.compute())
        self.recall_weighted_valid.reset()
        self.log('val/F1_score_weighted', self.f1_weighted_valid.compute())
        self.f1_weighted_valid.reset()

        # log per class accuracy
        if self.hparams.per_class_metrics:
            acc_per_class = self.acc_per_class_valid.compute()
            self.acc_per_class_valid.reset()
            for c in range(self.hparams.num_classes):
                self.log(f'val/class_{self.hparams.class_names[c]}-{c}_accuracy', acc_per_class[c])

            precision_per_class = self.precision_per_class_valid.compute()
            self.precision_per_class_valid.reset()
            for c in range(self.hparams.num_classes):
                self.log(f'val/class_{self.hparams.class_names[c]}-{c}_precision', precision_per_class[c])

            
            f1_per_class = self.f1_per_class_valid.compute()
            self.f1_per_class_valid.reset()
            for c in range(self.hparams.num_classes):
                self.log(f'val/class_{self.hparams.class_names[c]}-{c}_F1_score', f1_per_class[c])
            
            ap_per_class = self.ap_per_class_valid.compute()
            self.ap_per_class_valid.reset()
            for c in range(self.hparams.num_classes):
                self.log(f'val/class_{self.hparams.class_names[c]}-{c}_AP', ap_per_class[c])

        if self.hparams.kshot_metrics:

            # log accuracy
            few_acc, medium_acc, many_acc = self.kshot_acc_valid.compute()
            self.kshot_acc_valid.reset()
            self.log('val/few_shot_accuracy', few_acc)
            self.log('val/medium_shot_accuracy', medium_acc)
            self.log('val/many_shot_accuracy', many_acc)
            # log precision
            few_precision, medium_precision, many_precision = self.kshot_precision_valid.compute()
            self.kshot_precision_valid.reset()
            self.log('val/few_shot_precision', few_precision)
            self.log('val/medium_shot_precision', medium_precision)
            self.log('val/many_shot_precision', many_precision)
            # log F1 Score
            few_f1, medium_f1, many_f1 = self.kshot_f1_valid.compute()
            self.kshot_f1_valid.reset()
            self.log('val/few_shot_f1', few_f1)
            self.log('val/medium_shot_f1', medium_f1)
            self.log('val/many_shot_f1', many_f1)
            # log mAP
            few_map, medium_map, many_map = self.kshot_map_valid.compute()
            self.kshot_map_valid.reset()
            self.log('val/few_shot_map', few_map)
            self.log('val/medium_shot_map', medium_map)
            self.log('val/many_shot_map', many_map)

        steps = len(val_step_outputs)

        y = [None] * steps
        y_pred = [None] * steps
        logits = [None] * steps

        for i, result in enumerate(val_step_outputs):
            y[i] = result['y']
            y_pred[i] = result['y_pred']
            logits[i] = result['logits']

        y = torch.cat(y).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        logits = torch.cat(logits).cpu().numpy()

        if self.hparams.per_class_metrics:
            self.logger.experiment.log({
                "val_charts/confusion_matrix_per_class": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y, preds=y_pred,
                    class_names=self.hparams.class_names)
            })

            self.logger.experiment.log({
                'val_charts/confusion_matrix_per_class_sklearn':
                    wandb.sklearn.plot_confusion_matrix(y, y_pred, self.hparams.class_names)
            })

            self.logger.experiment.log({
                "val_charts/precision_recall_curve_per_class":  wandb.plot.pr_curve(
                    y,
                    logits,
                    labels=self.hparams.class_names
                )
            })

        if self.hparams.kshot_metrics:
            y_true = y
            preds = y_pred

            few_index = np.isin(y_true, self.few_shot_classes)
            medium_index = np.isin(y_true, self.medium_shot_classes)
            many_index = np.isin(y_true, self.many_shot_classes)

            y_true_cmp = np.zeros_like(y_true)

            y_true_cmp[few_index] = 0
            y_true_cmp[medium_index] = 1
            y_true_cmp[many_index] = 2

            few_index = np.isin(preds, self.few_shot_classes)
            medium_index = np.isin(preds, self.medium_shot_classes)
            many_index = np.isin(preds, self.many_shot_classes)

            preds_cmp = np.zeros_like(preds)

            preds_cmp[few_index] = 0
            preds_cmp[medium_index] = 1
            preds_cmp[many_index] = 2

            logits_cmp = np.zeros((logits.shape[0], 3))
            logits_cmp[:, 0] = np.max(logits[:, self.few_shot_classes], axis=1)
            logits_cmp[:, 1] = np.max(logits[:, self.medium_shot_classes], axis=1)
            logits_cmp[:, 2] = np.max(logits[:, self.many_shot_classes], axis=1)

            self.logger.experiment.log({
                "val_charts/confusion_matrix_kshot": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true_cmp,
                    preds=preds_cmp,
                    class_names=['Few Shot', 'Medium Shot', 'Many Shot'])
            })
            self.logger.experiment.log({
                'val_charts/confusion_matrix_kshot_sklearn':
                    wandb.sklearn.plot_confusion_matrix(y_true_cmp, preds_cmp, ['Few Shot', 'Medium Shot', 'Many Shot'])
            })

            self.logger.experiment.log({
                "val_charts/precision_recall_curve_kshot":  wandb.plot.pr_curve(
                    y_true_cmp,
                    logits_cmp,
                    labels=['Few Shot', 'Medium Shot', 'Many Shot']
                )
            })

    def test_step(self, batch, batch_idx):

        y_pred, y, logits = self.validation_step(batch, -1)
        
        results = {
            'y': y,
            'y_pred': y_pred,
            'logits': logits
        }
        return results

    def test_epoch_end(self, outputs):

         # Accuracy
        acc_test = Accuracy().to(self.device)

        # Precision, Recall, F1, mAP
        precision_test = Precision(num_classes=self.hparams.num_classes, average='macro').to(self.device)
        recall_test = Recall(num_classes=self.hparams.num_classes, average='macro').to(self.device)
        f1_test = F1(num_classes=self.hparams.num_classes, average='macro').to(self.device)

        mAP_test = AveragePrecision(num_classes=self.hparams.num_classes).to(self.device)

        # Per Class Metrics
        acc_per_class_test = Accuracy(num_classes=self.hparams.num_classes, average=None).to(self.device)

        precision_per_class_test = Precision(num_classes=self.hparams.num_classes, average=None).to(self.device)
        f1_per_class_test = F1(num_classes=self.hparams.num_classes, average=None).to(self.device)
        ap_per_class_test = AveragePrecision(num_classes=self.hparams.num_classes, average=None).to(self.device)

        # Few-Medium-Many Shot Metrics
        few_shot_classes, medium_shot_classes, many_shot_classes = self.get_few_medium_many_shot_partitions()

        kshot_acc_test = KShotAccuracy(few_shot_classes, medium_shot_classes, many_shot_classes).to(self.device)
        kshot_precision_test = KShotPrecision(few_shot_classes, medium_shot_classes, many_shot_classes).to(self.device)
        kshot_f1_test = KShotF1(few_shot_classes, medium_shot_classes, many_shot_classes).to(self.device)
        kshot_map_test = KShotmAP(few_shot_classes, medium_shot_classes, many_shot_classes).to(self.device)

        steps = len(outputs)

        y = [None] * steps
        y_pred = [None] * steps
        logits = [None] * steps

        for i, result in enumerate(outputs):
            
            y[i] = result['y'].to(self.device)
            y_pred[i] = result['y_pred'].to(self.device)
            logits[i] = result['logits'].to(self.device)

            print(logits[i].shape, y[i].shape)

            # global metrics
            acc_test(logits[i], y[i])
            precision_test(logits[i], y[i])
            recall_test(logits[i], y[i])
            f1_test(logits[i], y[i])
            mAP_test(logits[i], y[i])
            # per class metrics
            acc_per_class_test(logits[i], y[i])
            precision_per_class_test(logits[i], y[i])
            f1_per_class_test(logits[i], y[i])
            ap_per_class_test(logits[i], y[i])
            # k-shot metrics
            kshot_acc_test(y_pred[i], y[i])
            kshot_precision_test(y_pred[i], y[i])
            kshot_f1_test(logits[i], y[i])
            kshot_map_test(logits[i], y[i])
        


        y = torch.cat(y).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        logits = torch.cat(logits).cpu().numpy()

        self.log('test/accuracy', acc_test.compute())
        self.log('test/precision', precision_test.compute())
        self.log('test/recall', recall_test.compute())
        self.log('test/f1', f1_test.compute())
        self.log('test/mAP', mAP_test.compute())

        acc_per_class = acc_per_class_test.compute()
        for c in range(self.hparams.num_classes):
            self.log(f'test/class_{self.hparams.class_names[c]}-{c}_accuracy', acc_per_class[c])
            plt.figure(figsize=(8,4))
            plt.grid()
            plt.plot(np.arange(self.hparams.num_class) , acc_per_class)
            plt.title(rf'{self.hparams.dataset} $\rho = {int(1/self.hparams.imb_factor)}$ Per Class Accuracy')
            plt.xlabel('Classes Sorted by Frequency')
            plt.ylabel('Top-1 Accuracy')
            plt.savefig('figures/' + self.logger.experiment.name + '_per_class_accuracy.png')

        precision_per_class = precision_per_class_test.compute()
        for c in range(self.hparams.num_classes):
            self.log(f'test/class_{self.hparams.class_names[c]}-{c}_precision', precision_per_class[c])
            plt.figure(figsize=(8,4))
            plt.grid()
            plt.plot(np.arange(self.hparams.num_class) , acc_per_class)
            plt.title(rf'{self.hparams.dataset} $\rho = {int(1/self.hparams.imb_factor)}$ Per Class Precision')
            plt.xlabel('Classes Sorted by Frequency')
            plt.ylabel('Precision')
            plt.savefig('figures/' + self.logger.experiment.name + '_per_class_precision.png')
        
        f1_per_class = f1_per_class_test.compute()
        for c in range(self.hparams.num_classes):
            self.log(f'test/class_{self.hparams.class_names[c]}-{c}_F1_score', f1_per_class[c])
            plt.figure(figsize=(8,4))
            plt.grid()
            plt.plot(np.arange(self.hparams.num_class) , acc_per_class)
            plt.title(rf'{self.hparams.dataset} $\rho = {int(1/self.hparams.imb_factor)}$ Per Class F1 Score')
            plt.xlabel('Classes Sorted by Frequency')
            plt.ylabel('F1 Score')
            plt.savefig('figures/' + self.logger.experiment.name + '_per_class_f1_.png')
        
        ap_per_class = ap_per_class_test.compute()
        for c in range(self.hparams.num_classes):
            self.log(f'test/class_{self.hparams.class_names[c]}-{c}_AP', ap_per_class[c])
            plt.figure(figsize=(8,4))
            plt.grid()
            plt.plot(np.arange(self.hparams.num_class) , acc_per_class)
            plt.title(rf'{self.hparams.dataset} $\rho = {int(1/self.hparams.imb_factor)}$ Per Class Average Precision')
            plt.xlabel('Classes Sorted by Frequency')
            plt.ylabel('Average Precision')
            plt.savefig('figures/' + self.logger.experiment.name + '_per_class_ap.png')


        # log accuracy
        few_acc, medium_acc, many_acc = kshot_acc_test.compute()
        self.log('test/few_shot_accuracy', few_acc)
        self.log('test/medium_shot_accuracy', medium_acc)
        self.log('test/many_shot_accuracy', many_acc)
        # log precision
        few_precision, medium_precision, many_precision = kshot_precision_test.compute()
        self.log('test/few_shot_precision', few_precision)
        self.log('test/medium_shot_precision', medium_precision)
        self.log('test/many_shot_precision', many_precision)
        # log F1 Score
        few_f1, medium_f1, many_f1 = kshot_f1_test.compute()
        self.log('test/few_shot_f1', few_f1)
        self.log('test/medium_shot_f1', medium_f1)
        self.log('test/many_shot_f1', many_f1)
        # log mAP
        few_map, medium_map, many_map = kshot_map_test.compute()
        self.log('test/few_shot_map', few_map)
        self.log('test/medium_shot_map', medium_map)
        self.log('test/many_shot_map', many_map)

        # plot confusion matrix in wandb style
        self.logger.experiment.log({
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y, preds=y,
                class_names=self.hparams.class_names)
        })

        y_true = y
        preds = y_pred

        few_index = np.isin(y_true, self.few_shot_classes)
        medium_index = np.isin(y_true, self.medium_shot_classes)
        many_index = np.isin(y_true, self.many_shot_classes)

        y_true_cmp = np.zeros_like(y_true)

        y_true_cmp[few_index] = 0
        y_true_cmp[medium_index] = 1
        y_true_cmp[many_index] = 2

        few_index = np.isin(preds, self.few_shot_classes)
        medium_index = np.isin(preds, self.medium_shot_classes)
        many_index = np.isin(preds, self.many_shot_classes)

        preds_cmp = np.zeros_like(preds)

        preds_cmp[few_index] = 0
        preds_cmp[medium_index] = 1
        preds_cmp[many_index] = 2

        logits_cmp = np.zeros((logits.shape[0], 3))
        logits_cmp[:, 0] = np.max(logits[:, self.few_shot_classes], axis=1)
        logits_cmp[:, 1] = np.max(logits[:, self.medium_shot_classes], axis=1)
        logits_cmp[:, 2] = np.max(logits[:, self.many_shot_classes], axis=1)

        self.logger.experiment.log({
            "test_charts/confusion_matrix_kshot": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true_cmp,
                preds=preds_cmp,
                class_names=['Few Shot', 'Medium Shot', 'Many Shot'])
        })
        self.logger.experiment.log({
            'test_charts/confusion_matrix_kshot_sklearn':
                wandb.sklearn.plot_confusion_matrix(y_true_cmp, preds_cmp, ['Few Shot', 'Medium Shot', 'Many Shot'])
        })

        self.logger.experiment.log({
            "test_charts/precision_recall_curve_kshot":  wandb.plot.pr_curve(
                y_true_cmp,
                logits_cmp,
                labels=['Few Shot', 'Medium Shot', 'Many Shot']
            )
        })



    def setup(self, stage):

        cifar_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.hparams.img_size, self.hparams.img_size)),
            transforms.RandAugment(num_ops=self.hparams.randaug_num_ops, magnitude=self.hparams.randaug_magnitude),
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





