import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from termcolor import colored

class ResCoreElement(nn.Module):
    """
    Residual Core element used inside the NN. Control the number of filters
    and batch normalization.
    """
    def __init__(self,
                 input_size,
                 num_filters,
                 use_batchnorm=True,
                 use_leaky=True,
                 leaky_p=0.2):
                super(ResCoreElement, self).__init__()
                self.use_bn = use_batchnorm
                self.use_lr = use_leaky
                self.leaky_p = leaky_p

                self.conv1 = nn.Conv3d(input_size,
                                       num_filters,
                                       kernel_size=3,
                                       padding=1)
                self.conv2 = nn.Conv3d(input_size,
                                       num_filters,
                                       kernel_size=1,
                                       padding=0)
                self.bn_add = nn.BatchNorm3d(num_filters)

    def forward(self, x):
        """
        include residual model
        """
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_sum = self.bn_add(x_1 + x_2) if self.use_bn is True else x_1 + x_2
        return F.leaky_relu(x_sum,
                            self.leaky_p) if self.use_lr else F.relu(x_sum)


class ConvElement(nn.Module):
    """
    Residual Core element used inside the NN. Control the number of filters
    and batch normalization.
    """
    def __init__(self,
                 input_size,
                 num_filters,
                 use_leaky=True,
                 stride=1,
                 leaky_p=0.2):
                super(ConvElement, self).__init__()
                self.use_lr = use_leaky
                self.leaky_p = leaky_p
                self.conv1 = nn.Conv3d(input_size,
                                       num_filters,
                                       kernel_size=3,
                                       padding=1,
                                       stride=stride)

    def forward(self, x):
        """
        include residual model
        """
        x_1 = self.conv1(x)
        return F.leaky_relu(x_1, self.leaky_p) if self.use_lr else F.relu(x_1)


class Pooling3D(nn.Module):
    """
    3D pooling layer by striding.
    """
    def __init__(self, input_size,
                 use_batchnorm=True,
                 use_leaky=True,
                 leaky_p=0.2):
        super(Pooling3D, self).__init__()
        self.use_bn = use_batchnorm
        self.use_lr = use_leaky
        self.leaky_p = leaky_p
        self.conv1 = nn.Conv3d(input_size,
                               input_size,
                               kernel_size=2,
                               stride=2)
        self.bn = nn.BatchNorm3d(input_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x) if self.use_bn is True else x
        return F.leaky_relu(x, self.leaky_p) if self.use_lr else F.relu(x)


class UpdateStatus(object):
    """
    Update training/testing colors
    Register new elements iteratively by setting them by name

    """
    def __init__(self, pat_interval=0):
        super(UpdateStatus, self).__init__()

        # network parameters
        self.elements = {}

    def register_new_element(self, element_name, mode):
        """
        Register new elements to show. We control each element
        using a dictionary with two elements:
        - value: actual value to update
        - mode: value type, so accuracy is incremental and losses decremental
        """
        if mode == 'incremental':
            self.elements[element_name] = {'value': 0, 'mode': 'incremental'}
        else:
            self.elements[element_name] = {'value': np.inf,
                                           'mode': 'decremental'}

    def update_element(self, element_name, current_value):
        """
        update element
        """

        update = current_value
        if element_name in self.elements.keys():
            update = self.process_update(element_name, current_value)
        else:
            print('ERROR:', element_name,
                  "element is not currently registered")
        return update

    def process_update(self, element_name, current_value):
        """
        update the value for a particu
        """
        best_value = self.elements[element_name]['value']
        mode = self.elements[element_name]['mode']
        update = '{:.4f}'.format(current_value)

        if (mode == 'decremental') and (best_value > current_value):
            update = colored(update, 'green')
            self.elements[element_name]['value'] = current_value
        elif (mode == 'incremental') and (best_value < current_value):
            update = colored(update, 'green')
            self.elements[element_name]['value'] = current_value
        else:
            update = colored(update, 'red')

        return update


class EarlyStopping(object):
    """
    Control early stopping with several parameters
    check early stopping conditions and save the model. If the
    current validation loss is lower than the previous one, the
    model is saved back and the early stopping iteration
    is set to 0. If not, the number of iterations without
    decrease in the val_loss is update. When the number
    iterations is > than patience, training is stopped.

    """
    def __init__(self, epoch=1, metric='acc', patience=20):
        super(EarlyStopping, self).__init__()

        self.epoch = epoch
        self.patience = patience
        self.patience_iter = 0
        self.metric = metric
        self.best = None

        # initialize the best value
        self.__initialize_best()

    def __initialize_best(self):
        """
        Initialize the best value taking into account the kind of metric
        """
        if self.metric == 'acc':
            self.best = 0
        if self.metric == 'dsc':
            self.best = 0
        if self.metric == 'loss':
            self.best = np.inf

    def __compare_metric(self, current_value):
        """
        Check status for the current epoch
        """
        if self.metric == 'acc':
            is_best = current_value > self.best
        if self.metric == 'dsc':
            is_best = current_value > self.best
        if self.metric == 'loss':
            is_best = current_value < self.best

        return is_best

    def save_epoch(self, current_value, current_epoch):
        """
        check if the current_value for a given epoch
        """

        self.epoch = current_epoch
        is_best = self.__compare_metric(current_value)
        if is_best:
            self.best = current_value
            self.patience_iter = 0
        else:
            self.patience_iter += 1

        return is_best

    def stop_model(self):
        """
        check if the maximum number of iterations has been raised
        """
        return self.patience_iter > self.patience

    def get_best_value(self):
        """
        Return best value
        """
        return self.best
