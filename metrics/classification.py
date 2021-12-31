import torch
import numpy as np
from torchmetrics import Metric, AveragePrecision
from torch import Tensor


class KShotAccuracy(Metric):
    """
    This class computes the accuracy based on partitioning a set of classes according to their frequencies. The class
    takes 3 lists where each list contains the ids of the classes in that particular group. The groups are few shot,
    medium shot, and many shot.
    """
    def __init__(self, few_shot_classes, medium_shot_classes, many_shot_classes):
        super().__init__()

        self.few_shot_classes = few_shot_classes
        self.medium_shot_classes = medium_shot_classes
        self.many_shot_classes = many_shot_classes

        self.add_state('few_correct', default=torch.tensor(0), persistent=True)
        self.add_state('few_count', default=torch.tensor(0), persistent=True)

        self.add_state('medium_correct', default=torch.tensor(0), persistent=True)
        self.add_state('medium_count', default=torch.tensor(0), persistent=True)

        self.add_state('many_correct', default=torch.tensor(0), persistent=True)
        self.add_state('many_count', default=torch.tensor(0), persistent=True)

    def update(self, output: Tensor, targets: Tensor):

        output = output.cpu().numpy()
        targets = targets.cpu().numpy()

        few_index = np.isin(targets, self.few_shot_classes)
        medium_index = np.isin(targets, self.medium_shot_classes)
        many_index = np.isin(targets, self.many_shot_classes)

        self.few_count += np.sum(few_index)
        self.medium_count += np.sum(medium_index)
        self.many_count += np.sum(many_index)

        self.few_correct += np.sum(output[few_index] == targets[few_index])
        self.medium_correct += np.sum(output[medium_index] == targets[medium_index])
        self.many_correct += np.sum(output[many_index] == targets[many_index])

    def compute(self):

        few_acc = self.few_correct / self.few_count if self.few_count > 0 else 0
        medium_acc = self.medium_correct / self.medium_count if self.medium_count > 0 else 0
        many_acc = self.many_correct / self.many_count if self.many_count > 0 else 0

        return few_acc, medium_acc, many_acc


class KShotPrecision(Metric):
    """
    This class computes the precision based on partitioning a set of classes according to their frequencies. The class
    takes 3 lists where each list contains the ids of the classes in that particular group. The groups are few shot,
    medium shot, and many shot.
    """
    def __init__(self, few_shot_classes, medium_shot_classes, many_shot_classes):
        super().__init__()

        self.few_shot_classes = few_shot_classes
        self.medium_shot_classes = medium_shot_classes
        self.many_shot_classes = many_shot_classes

        self.add_state('few_correct', default=torch.tensor(0), persistent=True)
        self.add_state('few_count', default=torch.tensor(0), persistent=True)

        self.add_state('medium_correct', default=torch.tensor(0), persistent=True)
        self.add_state('medium_count', default=torch.tensor(0), persistent=True)

        self.add_state('many_correct', default=torch.tensor(0), persistent=True)
        self.add_state('many_count', default=torch.tensor(0), persistent=True)

    def update(self, output: Tensor, targets: Tensor):

        output = output.cpu().numpy()
        targets = targets.cpu().numpy()

        few_index = np.isin(output, self.few_shot_classes)
        medium_index = np.isin(output, self.medium_shot_classes)
        many_index = np.isin(output, self.many_shot_classes)

        self.few_count += np.sum(few_index)
        self.medium_count += np.sum(medium_index)
        self.many_count += np.sum(many_index)

        self.few_correct += np.sum(output[few_index] == targets[few_index])
        self.medium_correct += np.sum(output[medium_index] == targets[medium_index])
        self.many_correct += np.sum(output[many_index] == targets[many_index])

    def compute(self):

        few_acc = self.few_correct / self.few_count if self.few_count > 0 else 0
        medium_acc = self.medium_correct / self.medium_count if self.medium_count > 0 else 0
        many_acc = self.many_correct / self.many_count if self.many_count > 0 else 0

        return few_acc, medium_acc, many_acc


class KShotF1(Metric):
    """
    This class computes the F1 Score based on partitioning a set of classes according to their frequencies. The class
    takes 3 lists where each list contains the ids of the classes in that particular group. The groups are few shot,
    medium shot, and many shot.
    """
    def __init__(self, few_shot_classes, medium_shot_classes, many_shot_classes):
        super().__init__()

        self.recall = KShotAccuracy(few_shot_classes, medium_shot_classes, many_shot_classes)
        self.precision = KShotPrecision(few_shot_classes, medium_shot_classes, many_shot_classes)

    def update(self, output: Tensor, targets: Tensor):

        self.recall.update(output, targets)
        self.precision.update(output, targets)

    def compute(self):

        few_recall, medium_recall, many_recall = self.recall.compute()
        few_precision, medium_precision, many_precision = self.precision.compute()

        few_f1 = 2 * (few_recall * few_precision) / (few_recall + few_precision) if (few_recall + few_precision) > 0 else 0
        medium_f1 = 2 * (medium_recall * medium_precision) / (medium_recall + medium_precision) if (medium_recall + medium_precision) >  0 else 0
        many_f1 = 2 * (many_recall * many_precision) / (many_recall + many_precision) if (many_recall + many_precision) > 0 else 0

        return few_f1, medium_f1, many_f1

    def reset(self) -> None:
        self.recall.reset()
        self.precision.reset()


class KShotmAP(Metric):
    """
    This class computes the mean Average Precision based on partitioning a set of classes according to their frequencies. The class
    takes 3 lists where each list contains the ids of the classes in that particular group. The groups are few shot,
    medium shot, and many shot.
    """
    def __init__(self, few_shot_classes, medium_shot_classes, many_shot_classes):
        super().__init__()

        self.few_shot_classes = few_shot_classes
        self.medium_shot_classes = medium_shot_classes
        self.many_shot_classes = many_shot_classes

        self.map = AveragePrecision(num_classes=3, average=None)

    def update(self, output: Tensor, targets: Tensor):

        output = output.cpu().numpy()
        targets = targets.cpu().numpy()

        output_cmp = np.zeros((output.shape[0], 3))
        output_cmp[:, 0] = np.max(output[:, self.few_shot_classes], axis=1)
        output_cmp[:, 1] = np.max(output[:, self.medium_shot_classes], axis=1)
        output_cmp[:, 2] = np.max(output[:, self.many_shot_classes], axis=1)

        few_index = np.isin(targets, self.few_shot_classes)
        medium_index = np.isin(targets, self.medium_shot_classes)
        many_index = np.isin(targets, self.many_shot_classes)

        targets_cmp = np.zeros_like(targets)

        targets_cmp[few_index] = 0
        targets_cmp[medium_index] = 1
        targets_cmp[many_index] = 2

        self.map(torch.Tensor(output_cmp), torch.Tensor(targets_cmp))

    def compute(self):

        few_map, medium_map, many_map = self.map.compute()
        return few_map, medium_map, many_map

    def reset(self) -> None:

        self.map.reset()

