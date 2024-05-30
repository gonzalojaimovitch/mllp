import sys
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.nn import init
from torch.autograd import Variable
import math
from copy import deepcopy

from mllp.utils import UnionFind

THRESHOLD = 0.5

"""Adaptation to L0 Reg

- Add constants limit_a, limit_b, epsilon
"""

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class L0ConjunctionLayer(nn.Module):
    """The conjunction layer is used to learn the rules."""

    """Adaptation to L0 Reg:

    - Add L0 functions: reset_parameters, constrain_parameters, cdf_qz, quantile_concrete, _reg_w, regularization, count_expected_flops_and_l0, get_eps, sample_z, sample_weights

    __init__
    - Remove Random Binarization Rate
    - Change "n" for "out_features" for compatibility with the L0 functions
    - Change "input_dim" for "out_features" for compatibility with the L0 functions
    - Change "W" for "weights" for compatibility with the L0 functions. The initialization is also updated to the L0 repo one (Kaiming). Note that the weights are transposed in comparison to the original MLLP implementation.
    - Remove "randomly_binarize_layer"
    - Include "bias", "weight_decay", "droprate_init", "temperature", "lamba", "local_rep", "**kwargs"

    forward
    - Add L0 Weight Sampling
    - Change "x" for "input" as argument for the adaptation
    - Change "x" for "processed_input" in the activation for the adaptation (now the logit is obtained after sampling weights)
    - Bias is commented

    binarized_forward
    - Change W for weights for L0 compatibility

    reset_parameters
    - Comment Kaiming initialization since the weights are initialized as in the MLLP repo
    """

    def __init__(self, in_features, out_features, use_not=False, bias=True, weight_decay=1., droprate_init=0.5, temperature=2./3., lamba=1., local_rep=False, **kwargs): # UPDATED
        super(L0ConjunctionLayer, self).__init__()
        self.use_not = use_not
        self.node_activation_cnt = None

        self.in_features = in_features if not use_not else in_features * 2 # UPDATED
        self.out_features = out_features # UPDATED
        self.prior_prec = weight_decay # NEW
        self.weights = Parameter(0.1 * torch.rand(in_features, out_features)) # UPDATED
        self.sampled_weights = None # NEW
        self.qz_loga = Parameter(torch.Tensor(in_features)) # NEW
        self.temperature = temperature # NEW
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5 # NEW
        self.lamba = lamba # NEW
        self.use_bias = False # NEW
        self.local_rep = local_rep # NEW
        if bias: # NEW
            self.bias = Parameter(torch.Tensor(out_features)) # NEW
            self.use_bias = True # NEW
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor # NEW
        self.reset_parameters() # NEW

    def forward(self, input): # UPDATED
        if self.use_not:
            input= torch.cat((input, 1 - input), dim=1)
        if self.local_rep or not self.training: # NEW
            z = self.sample_z(input.size(0), sample=self.training) # NEW
            xin = input.mul(z) # NEW
            processed_input = xin # NEW
            # output = xin.mm(self.weights) # NEW
        else:
            weights = self.sample_weights() # NEW
            processed_input = input # NEW
            # output = input.mm(weights) # NEW
        output = torch.prod((1 - (1 - processed_input)[:, :, None] * weights[None, :, :]), dim=1) # NEW
        # if self.use_bias: # NEW
        #     output.add_(self.bias) # NEW
        return output

    def binarized_forward(self, x): # UPDATED
        with torch.no_grad():
            if self.use_not:
                x = torch.cat((x, 1 - x), dim=1)
            x = x.type(torch.int)
            Wb = torch.where(self.weights > THRESHOLD, torch.ones_like(self.weights), torch.zeros_like(self.weights)).type(torch.int) # UPDATED
            return torch.prod((1 - (1 - x)[:, :, None] * Wb[None, :, :]), dim=1)

    def reset_parameters(self): # NEW
        # init.kaiming_normal(self.weights, mode='fan_out')

        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs): # NEW
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x): # NEW
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x): # NEW
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self): # NEW
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw_col = torch.sum(- (.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        logpb = 0 if not self.use_bias else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
        return logpw + logpb

    def regularization(self): # NEW
        return self._reg_w()

    def count_expected_flops_and_l0(self): # NEW
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        ppos = torch.sum(1 - self.cdf_qz(0))
        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = ((self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]) + 1  # for rows
        num_instances_per_filter *= ((self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1]) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos

        if self.use_bias:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size): # NEW
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True): # NEW
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.in_features).expand(batch_size, self.in_features)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self): # NEW
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.in_features)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1) * self.weights
        


class L0DisjunctionLayer(nn.Module):
    """The disjunction layer is used to learn the rule sets."""

    """Adaptation to L0 Reg:

    - Add L0 functions: reset_parameters, constrain_parameters, cdf_qz, quantile_concrete, _reg_w, regularization, count_expected_flops_and_l0, get_eps, sample_z, sample_weights

    __init__
    - Remove Random Binarization Rate
    - Change "n" for "out_features" for compatibility with the L0 functions
    - Change "input_dim" for "out_features" for compatibility with the L0 functions
    - Change "W" for "weights" for compatibility with the L0 functions. The initialization is also updated to the L0 repo one (Kaiming). Note that the weights are transposed in comparison to the original MLLP implementation.
    - Remove "randomly_binarize_layer"
    - Include "bias", "weight_decay", "droprate_init", "temperature", "lamba", "local_rep", "**kwargs"

    forward
    - Add L0 Weight Sampling
    - Change "x" for "input" as argument for the adaptation
    - Change "x" for "processed_input" in the activation for the adaptation (now the logit is obtained after sampling weights)
    - Bias is commented

    binarized_forward
    - Change W for weights for L0 compatibility

    reset_parameters
    - Comment Kaiming initialization since the weights are initialized as in the MLLP repo
    """
    def __init__(self, in_features, out_features, use_not=False, bias=True, weight_decay=1., droprate_init=0.5, temperature=2./3., lamba=1., local_rep=False, **kwargs): # UPDATED
        super(L0DisjunctionLayer, self).__init__()
        self.use_not = use_not
        self.node_activation_cnt = None

        self.in_features = in_features if not use_not else in_features * 2 # UPDATED
        self.out_features = out_features # UPDATED
        self.prior_prec = weight_decay # NEW
        self.weights = Parameter(0.1 * torch.rand(in_features, out_features)) # UPDATED
        self.qz_loga = Parameter(torch.Tensor(in_features)) # NEW
        self.temperature = temperature # NEW
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5 # NEW
        self.lamba = lamba # NEW
        self.use_bias = False # NEW
        self.local_rep = local_rep # NEW
        if bias: # NEW
            self.bias = Parameter(torch.Tensor(out_features)) # NEW
            self.use_bias = True # NEW
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor # NEW
        self.reset_parameters() # NEW

    def forward(self, input): # UPDATED
        if self.use_not:
            input= torch.cat((input, 1 - input), dim=1)
        if self.local_rep or not self.training: # NEW
            z = self.sample_z(input.size(0), sample=self.training) # NEW
            xin = input.mul(z) # NEW
            processed_input = xin # NEW
            # output = xin.mm(self.weights) # NEW
        else:
            weights = self.sample_weights() # NEW
            processed_input = input # NEW
            # output = input.mm(weights) # NEW
        output = 1 - torch.prod(1 - processed_input[:, :, None] * weights[None, :, :], dim=1) # UPDATED
        # if self.use_bias: # NEW
        #     output.add_(self.bias) # NEW
        return output

    def binarized_forward(self, x): # UPDATED
        with torch.no_grad():
            if self.use_not:
                x = torch.cat((x, 1 - x), dim=1)
            x = x.type(torch.int)
            Wb = torch.where(self.weights > THRESHOLD, torch.ones_like(self.weights), torch.zeros_like(self.weights)).type(torch.int) # UPDATED
            return 1 - torch.prod(1 - x[:, :, None] * Wb[None, :, :], dim=1)

    def reset_parameters(self): # NEW
        # init.kaiming_normal(self.weights, mode='fan_out')

        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs): # NEW
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x): # NEW
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x): # NEW
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self): # NEW
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw_col = torch.sum(- (.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        logpb = 0 if not self.use_bias else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
        return logpw + logpb

    def regularization(self): # NEW
        return self._reg_w()

    def count_expected_flops_and_l0(self): # NEW
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        ppos = torch.sum(1 - self.cdf_qz(0))
        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = ((self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]) + 1  # for rows
        num_instances_per_filter *= ((self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1]) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos

        if self.use_bias:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size): # NEW
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True): # NEW
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.in_features).expand(batch_size, self.in_features)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self): # NEW
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.in_features)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1) * self.weights


class L0MLLP(nn.Module):
    """The Multilayer Logical Perceptron (MLLP) used for Concept Rule Sets (CRS) learning.

    For more information, please read our paper: Transparent Classification with Multilayer Logical Perceptrons and
    Random Binarization."""

    """
    L0 Adaptation Reg:
    __init__
    - Remove Random Binarization Rate
    - Use L0ConjunctionLayer and L0DisjunctionLayer
    - Fixed Lamba to 1.0 (the original repo seems to pass an array with the value for each layer, but by default is 1.0)

    forward
    - Remove Random Binarization Rate

    train
    - Remove Weight Decay from Adam (to imitate the L0 repo, that apparently includes this in the regularization function)
    - Include "loss_function" to include L0 Regularization (that includes itself the Weight Decay)
    - Add "layers" attribute
    - Add regularization, update_ema, load_ema_params, load_params and get_params functions
    - Add N, beta_ema, weight_decay, lambas, local_rep and temperature
    - Add constrain parameters and update ema after each batch

    get_rules
    - Transpose weights (since they were transposed to match the L0 implementation)
    - Consider only the "weights" parameters
    """

    def __init__(self, dim_list, device, use_not=False, log_file=None, N=50000, beta_ema=0.999,
                 weight_decay=1, lamba=0.1, local_rep=False, temperature=2./3.): # UPDATED
        """

        Parameters
        ----------
        dim_list : list
            A list specifies the number of nodes (neurons) of all the layers in MLLP from bottom to top. dim_list[0]
            should be the dimensionality of the input data and dim_list[1] should be the number of class labels.
        device : torch.device
            Run on which device.
        use_not : bool
            Whether use the NOT (~) operator in logical rules.
        log_file : str
            The path of the log file. If log_file is None, use sys.stdout as the output stream.
        """

        super(L0MLLP, self).__init__()

        log_format = '[%(levelname)s] - %(message)s'
        if log_file is None:
            logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=log_format)
        else:
            logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format=log_format)

        self.N = N # NEW
        self.beta_ema = beta_ema # NEW
        # self.weight_decay = self.N * weight_decay # NEW
        self.weight_decay = weight_decay # NEW
        self.lamba = lamba # NEW
        self.dim_list = dim_list
        self.device = device
        self.use_not = use_not
        self.enc = None
        self.conj = []
        self.disj = []

        for i in range(0, len(dim_list) - 2, 2):
            conj = L0ConjunctionLayer(dim_list[i], dim_list[i+1], use_not=use_not, droprate_init=0.2 if i == 0 else 0.5, weight_decay=self.weight_decay,
                               lamba=lamba, local_rep=local_rep, temperature=temperature)
            disj = L0DisjunctionLayer(dim_list[i + 1], dim_list[i + 2], use_not=False, droprate_init=0.5, weight_decay=self.weight_decay,
                               lamba=lamba, local_rep=local_rep, temperature=temperature)
            self.add_module('conj{}'.format(i), conj)
            self.add_module('disj{}'.format(i), disj)
            self.conj.append(conj)
            self.disj.append(disj)

        self.layers = self.conj + self.disj # NEW

        if beta_ema > 0.: # NEW
            print('Using temporal averaging with beta: {}'.format(beta_ema)) # NEW
            self.avg_param = deepcopy(list(p.data for p in self.parameters())) # NEW
            if torch.cuda.is_available(): # NEW
                self.avg_param = [a.cuda() for a in self.avg_param] # NEW
            self.steps_ema = 0. # NEW

    def forward(self, x): # UPDATED
        for conj, disj in zip(self.conj, self.disj):
            x = conj(x) # UPDATED
            x = disj(x) # UPDATED
        return x

    def binarized_forward(self, x):
        """Equivalent to using the extracted Concept Rule Sets."""
        with torch.no_grad():
            for conj, disj in zip(self.conj, self.disj):
                x = conj.binarized_forward(x)
                x = disj.binarized_forward(x)
        return x

    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for name, param in self.named_parameters():
            if "weights" in name:
                param.data.clamp_(0, 1)

    def data_transform(self, X, y):
        X = X.astype(np.float32)
        if y is None:
            return torch.tensor(X)
        y = y.astype(np.float32)
        logging.debug('{}'.format(y.shape))
        logging.debug('{}'.format(y[:20]))
        return torch.tensor(X), torch.tensor(y)  # Do not put all the data in GPU at once.

    def regularization(self): # NEW
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self): # NEW
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self): # NEW
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self): # NEW
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params): # NEW
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self): # NEW
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        if epoch % lr_decay_epoch == 0:
            logging.info('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train(self, X=None, y=None, X_validation=None, y_validation=None, data_loader=None, epoch=50, lr=0.01, lr_decay_epoch=100,
              lr_decay_rate=0.75, batch_size=64, weight_decay=0.0): # UPDATED
        """

        Parameters
        ----------
        X : numpy.ndarray, shape = [n_samples, n_features]
            The training input instances. All the values should be 0 or 1.
        y : numpy.ndarray, shape = [n_samples, n_classes]
            The class labels. All the values should be 0 or 1.
        X_validation : numpy.ndarray, shape = [n_samples, n_features]
            The input instances of validation set. The format of X_validation is the same as X.
            if X_validation is None, use the training set (X) for validation.
        y_validation : numpy.ndarray, shape = [n_samples, n_classes]
            The class labels of validation set. The format of y_validation is the same as y.
            if y_validation is None, use the training set (y) for validation.
        epoch : int
            The total number of epochs during the training.
        lr : float
            The initial learning rate.
        lr_decay_epoch : int
            Decay learning rate every lr_decay_epoch epochs.
        lr_decay_rate : float
            Decay learning rate by a factor of lr_decay_rate.
        batch_size : int
            The batch size for training.
        weight_decay : float
            The weight decay (L2 penalty).

        Returns
        -------
        loss_log : list
            Training loss of MLLP during the training.
        accuracy : list
            Accuracy of MLLP on the validation set during the training.
        accuracy_b : list
            Accuracy of CRS on the validation set during the training.
        f1_score : list
            F1 score (Macro) of MLLP on the validation set during the training.
        f1_score_b : list
            F1 score (Macro) of CRS on the validation set during the training.

        """

        torch.autograd.set_detect_anomaly(True)

        if (X is None or y is None) and data_loader is None:
            raise Exception("Both data set and data loader are unavailable.")
        if data_loader is None:
            X, y = self.data_transform(X, y)
            if X_validation is not None and y_validation is not None:
                X_validation, y_validation = self.data_transform(X_validation, y_validation)
            data_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        loss_log = []
        accuracy = []
        accuracy_b = []
        f1_score = []
        f1_score_b = []

        self.weight_decay = weight_decay # NEW

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr) # UPDATED

        # define loss function (criterion) and optimizer
        def loss_function(output, target_var): # NEW
            loss = criterion(output, target_var)
            total_loss = loss + self.regularization()
            if torch.cuda.is_available():
                total_loss = total_loss.to(self.device)
            return total_loss

        for epo in tqdm(range(epoch)):
            optimizer = self.exp_lr_scheduler(optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate,
                                              lr_decay_epoch=lr_decay_epoch)
            running_loss = 0.0
            cnt = 0
            for X, y in data_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()  # Zero the gradient buffers.
                y_pred = self.forward(X)
                loss = loss_function(y_pred, y) # UPDATED
                running_loss += loss.item()
                loss.backward()
                if epo % 100 == 0 and cnt == 0:
                    for param in self.parameters():
                        logging.debug('{}'.format(param.grad))
                    cnt += 1
                optimizer.step()
                
                for k, layer in enumerate(self.layers):  # NEW
                    layer.constrain_parameters() # NEW

                if self.beta_ema > 0.: # NEW
                    self.update_ema() # NEW

                self.clip()

            logging.info('epoch: {}, loss: {}'.format(epo, running_loss))
            loss_log.append(running_loss)
            # Change the set of weights to be binarized every epoch.

            # Test the validation set or training set every 5 epochs.
            if epo % 5 == 0:
                if X_validation is not None and y_validation is not None:
                    acc, acc_b, f1, f1_b = self.test(X_validation, y_validation, False)
                    set_name = 'Validation'
                else:
                    acc, acc_b, f1, f1_b = self.test(X, y, False)
                    set_name = 'Training'
                logging.info('-' * 60)
                logging.info('On {} Set:\n\tAccuracy of MLLP Model: {}'
                             '\n\tAccuracy of CRS  Model: {}'.format(set_name, acc, acc_b))
                logging.info('On {} Set:\n\tF1 Score of MLLP Model: {}'
                             '\n\tF1 Score of CRS  Model: {}'.format(set_name, f1, f1_b))
                logging.info('-' * 60)
                accuracy.append(acc)
                accuracy_b.append(acc_b)
                f1_score.append(f1)
                f1_score_b.append(f1_b)
        return loss_log, accuracy, accuracy_b, f1_score, f1_score_b

    def test(self, X, y, need_transform=True):
        if need_transform:
            X, y = self.data_transform(X, y)
        with torch.no_grad():
            X = X.to(self.device)
            test_loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=False)

            y = y.cpu().numpy().astype(int)
            y = np.argmax(y, axis=1)
            data_num = y.shape[0]
            slice_step = data_num // 40 if data_num >= 40 else 1
            logging.debug('{} {}'.format(y.shape, y[:: slice_step]))

            # Test the model batch by batch.
            # Test the MLLP.
            y_pred_list = []
            for X, in test_loader:
                y_pred_list.append(self.forward(X))
            y_pred = torch.cat(y_pred_list)
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            logging.debug('{} {}'.format(y_pred.shape, y_pred[:: slice_step]))

            # Test the CRS.
            y_pred_b_list = []
            for X, in test_loader:
                y_pred_b_list.append(self.binarized_forward(X))
            y_pred_b = torch.cat(y_pred_b_list)
            y_pred_b = y_pred_b.cpu().numpy()
            logging.debug('y_pred_b: {} {}'.format(y_pred_b.shape, y_pred_b[:: (slice_step * 2)]))
            y_pred_b = np.argmax(y_pred_b, axis=1)
            logging.debug('{} {}'.format(y_pred_b.shape, y_pred_b[:: slice_step]))

            accuracy = metrics.accuracy_score(y, y_pred)
            accuracy_b = metrics.accuracy_score(y, y_pred_b)

            f1_score = metrics.f1_score(y, y_pred, average='macro')
            f1_score_b = metrics.f1_score(y, y_pred_b, average='macro')
        return accuracy, accuracy_b, f1_score, f1_score_b

    def detect_dead_node(self, X, need_transform=True):
        if need_transform:
            X = self.data_transform(X, None)
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=False)
            for conj, disj in zip(self.conj, self.disj):
                conj.node_activation_cnt = torch.zeros(conj.out_features, dtype=torch.long, device=self.device)
                disj.node_activation_cnt = torch.zeros(disj.out_features, dtype=torch.long, device=self.device)

            # Test the model batch by batch.
            for x, in test_loader:
                x = x.to(self.device)
                for conj, disj in zip(self.conj, self.disj):
                    x = conj.binarized_forward(x)
                    conj.node_activation_cnt += torch.sum(x, dim=0)
                    x = disj.binarized_forward(x)
                    disj.node_activation_cnt += torch.sum(x, dim=0)

    def get_rules(self, X=None):
        """Extract rules from parameters of MLLP."""
        # If X is not None, detect the dead nodes using X.
        if X is not None:
            self.detect_dead_node(X)
            activation_cnt_list = [np.sum(np.concatenate([X, 1 - X], axis=1) if self.use_not else X, axis=0)]
            for conj, disj in zip(self.conj, self.disj):
                activation_cnt_list.append(conj.node_activation_cnt.cpu().numpy())
                activation_cnt_list.append(disj.node_activation_cnt.cpu().numpy())
        else:
            activation_cnt_list = None

        # Get the rules from the top layer to the bottom layer.
        param_list = list(param for name, param in self.named_parameters() if "weights" in name) # UPDATED
        n_param = len(param_list)
        mark = {}
        rules_list = []
        for i in reversed(range(n_param)):
            param = param_list[i]
            W = param.T.cpu().detach().numpy() # UPDATED
            rules = defaultdict(list)
            num = self.dim_list[i]
            for k, row in enumerate(W):
                if i != n_param - 1 and ((i, k) not in mark):
                    continue
                if X is not None and activation_cnt_list[i + 1][k] < 1:
                    continue
                found = False
                for j, wj in enumerate(row):
                    if X is not None and activation_cnt_list[i][j % num] < 1:
                        continue
                    if wj > THRESHOLD:
                        rules[k].append(j)
                        mark[(i - 1, j % num)] = 1
                        found = True
                if not found:
                    rules[k] = []
            rules_list.append(rules)
        return rules_list

    def eliminate_redundant_rules(self, rules_list):
        """Eliminate redundant rules to simplify the extracted CRS."""
        rules_list = copy.deepcopy(rules_list)
        for i in reversed(range(len(rules_list))):
            # Eliminate the redundant part of each rule from bottom to top.
            if i != len(rules_list) - 1:
                num = self.dim_list[len(self.dim_list) - i - 2]
                for k, v in rules_list[i].items():
                    mark = {}
                    new_rule = []
                    for j1 in range(len(v)):
                        if j1 in mark:
                            continue
                        for j2 in range(j1 + 1, len(v)):
                            if j2 in mark:
                                continue
                            if j1 // num != j2 // num:
                                continue
                            s1 = set(rules_list[i + 1][v[j1 % num]])
                            s2 = set(rules_list[i + 1][v[j2 % num]])
                            if s1.issuperset(s2):
                                mark[j1] = 1
                                break
                            elif s1.issubset(s2):
                                mark[j2] = 1
                        if j1 not in mark:
                            new_rule.append(v[j1])
                    rules_list[i][k] = sorted(list(set(new_rule)))

            # Merge the identical nodes.
            union_find = UnionFind(rules_list[i].keys())
            kv_list = list(rules_list[i].items())
            n_kv = len(kv_list)
            if i > 0:
                for j1 in range(n_kv):
                    k1, v1 = kv_list[j1]
                    for j2 in range(j1 + 1, n_kv):
                        k2, v2 = kv_list[j2]
                        if v1 == v2:
                            union_find.union(k1, k2)
                # Update the upper layer.
                for k, v in rules_list[i - 1].items():
                    for j in range(len(v)):
                        v[j] = union_find.find(v[j])
                    rules_list[i - 1][k] = sorted(list(set(v)))
        # Get the final simplified rules.
        new_rules_list = []
        mark = {}
        for i in range(len(rules_list)):
            num = self.dim_list[len(self.dim_list) - i - 2]
            rules = defaultdict(list)
            for k, v in rules_list[i].items():
                if i != 0 and ((i, k) not in mark):
                    continue
                for j in v:
                    mark[(i + 1, j % num)] = 1
                    rules[k].append(j)
            new_rules_list.append(rules)
        return new_rules_list

    def get_name(self, i, j, X_fname=None, y_fname=None):
        nl = len(self.dim_list)
        num = self.dim_list[nl - i - 1]
        if j >= num:
            j -= num
            prefix = '~'
        else:
            prefix = ' '
        if X_fname is not None and i == nl - 1:
            name = X_fname[j]
        elif y_fname is not None and i == 0:
            name = y_fname[j]
        else:
            name = '{}{},{}'.format('s' if i % 2 == 0 else 'r', (nl - 2 - i) // 2 + 1, j)
        name = prefix + name
        return name

    def concept_rule_set_print(self, X=None, X_fname=None, y_fname=None, file=sys.stdout, eliminate_redundancy=True):
        """Print the Concept Rule Sets extracted from the trained Multilayer Logical Perceptron."""
        if eliminate_redundancy:
            rules_list = self.eliminate_redundant_rules(self.get_rules(X))
        else:
            rules_list = self.get_rules(X)
        for i in range(0, len(rules_list), 2):
            rules_str = defaultdict(list)
            for k, v in rules_list[i + 1].items():
                for j in v:
                    rules_str[k].append(self.get_name(i + 2, j, X_fname=X_fname, y_fname=y_fname))
            rule_sets = defaultdict(list)
            num = self.dim_list[len(self.dim_list) - i - 2]
            for k, v in rules_list[i].items():
                for j in v:
                    if j >= num:
                        jn = j - num
                        prefix = '~'
                    else:
                        prefix = ' '
                        jn = j
                    rule_sets[self.get_name(i, k, X_fname=X_fname, y_fname=y_fname)].append(
                        '{:>10}:\t{}{}'.format(self.get_name(i + 1, j, X_fname=X_fname, y_fname=y_fname), prefix,
                                               rules_str[jn]))
            print('-' * 90, file=file)
            for k, v in rule_sets.items():
                print('{}:'.format(k), file=file)
                for r in v:
                    print('\t', r, file=file)
