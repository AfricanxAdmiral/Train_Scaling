import numpy as np
import torch
from torch import optim
import torch.nn as nn
from scipy.spatial.distance import cdist
from torch.nn import functional as F
from tqdm import tqdm
import copy
#from inclearn.lib import factory, network, utils
#from inclearn.models.base import IncrementalLearner
from lib import factory, network, utils
from models.base import IncrementalLearner
import math
from lib.center_loss import CenterLoss

EPSILON = 1e-8


class ICarl(IncrementalLearner):
    """Implementation of iCarl.

    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args):
        super().__init__()

        self._device = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]
        self.epslon = 0.0000000001
        self._memory_size = args["memory_size"]
        self._n_classes = 0
        self._temp = args["temperature"]
        self.use_bias = True
        self.normalize = False
        self._network = network.BasicNet(args["convnet"], device=self._device, use_bias=self.use_bias, normalize = self.normalize)
        self.task_fc = None
        self._examplars = {}
        self._means = None
        self._frequency = 10
        self._old_model = None
        self._temp_model = None
        self.feature_mean = None
        self._herding_matrix = []
        self._exemplar_mean_withoutNorm = None
        
        
    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        print("Now {} examplars per class.".format(self._memory_per_class))
        
        self._optimizer = factory.get_optimizer(self._network.parameters(), self._opt_name, self._lr, self._weight_decay)
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, self._scheduling, gamma=self._lr_decay)
        
        
    def _train_task(self, train_loader, val_loader):
        print("nb ", len(train_loader.dataset))
        for epoch in range(self._n_epochs):
            self.epoch = epoch
            _loss, _closs, _dloss, val_loss = 0., 0., 0., 0.

            prog_bar = tqdm(train_loader)
            for i, (inputs, targets, index) in enumerate(prog_bar, start=1):
                
                self._optimizer.zero_grad()

                loss, closs, dloss = self._forward_loss(inputs, targets, index)

                if not utils._check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                
                self._optimizer.step()

                _loss += loss.item()
                _closs +=closs
                _dloss +=dloss
                
                if val_loader is not None and i == len(train_loader):
                    for inputs, targets in val_loader:
                        val_loss += self._forward_loss(inputs, targets).item()

                prog_bar.set_description(
                    "Task {}/{}, Epoch {}/{} => Total loss: {}, Classify: {}, Dist: {}".format(
                        self._task + 1, self._n_tasks,
                        epoch + 1, self._n_epochs,
                        round(_loss / i, 3),
                        round(_closs / i, 3),
                        round(_dloss / i, 3)
                    )
                )
                
            self._scheduler.step()
    def _forward_loss(self, inputs, targets, index):
        inputs, targets, index = inputs.to(self._device), targets.to(self._device), index.to(self._device)
        targets = utils.to_onehot(targets, self._n_classes).to(self._device)
        logits = self._network(inputs)

        return self._compute_loss(inputs, logits, targets, index)

    def _after_task(self, inc_dataset):
        self._old_model = self._network.copy().freeze()
        self.build_examplars(inc_dataset)

    def _save_model(self, path):
        torch.save(self._old_model.state_dict(), path)
        
        
    def _eval_task(self, data_loader):
        y_nearest_pred, y_nearest_true, y_nearest_top5 = compute_nearest_accuracy(self._network, data_loader, self._exemplar_mean_withNorm, self._n_classes/self._task_size, self._task_size)
        y_classifier_pred, y_classifier_true, y_classifier_top5 =  compute_classifier_accuracy(self._network, data_loader, self._exemplar_mean_withNorm)
        return y_classifier_pred, y_classifier_true, y_nearest_pred, y_nearest_true, y_classifier_top5, y_nearest_top5

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, inputs, logits, targets, index):
        if self._old_model is None:
            c_loss = 0
            d_loss = 0
            #_, one_targets = torch.max(targets, 1)
            #loss = nn.CrossEntropyLoss()(logits, one_targets)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            
        else:
            c_loss = 0
            d_loss = 0
            #_, one_targets = torch.max(targets, 1)
            #loss = nn.CrossEntropyLoss()(logits, one_targets)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            
            weight1 = 0.1
            weight2= 1
            
            new_feature = self._network.extract(inputs)
            old_feature = self._old_model.extract(inputs).detach()

            new_theta = self._network.classifier.weight[:-self._task_size,:]
            old_theta = self._old_model.classifier.weight.detach()
            """
            #Euclidean
            #################v3#################################################################
            new_bias = self._network.classifier.bias[:-self._task_size,:]
            old_bias = self._old_model.classifier.bias.detach()
            
            new_feature = torch.cat((new_feature, (torch.ones(len(new_feature),1)).to(self._device)), dim=1)
            new_theta = torch.cat((new_theta, new_bias), dim=1)
            old_feature = torch.cat((old_feature, (torch.ones(len(old_feature),1)).to(self._device)), dim=1)
            old_theta = torch.cat((old_theta, old_bias), dim=1)
            #################v3#################################################################
            """
            new_dist = self._Euclidean(F.normalize(new_feature, p=2, dim=1),F.normalize(new_theta, p=2, dim=1))
            old_dist = self._Euclidean(F.normalize(old_feature, p=2, dim=1),F.normalize(old_theta, p=2, dim=1))  
            loss1 = torch.pow(new_dist-old_dist,2)
            weight = self._weight(old_dist)
            #weight = 1
            
            loss1 = weight * loss1
            loss1 = torch.sum(loss1)/len(new_dist)
            #Euclidean

            oldclass_num = self._n_classes-self._task_size
            
            #005009 00250045
            #weight1 = 0.1* oldclass_num/self._n_classes
            ##007009  0035045
            weight1 = 0.1 * math.sqrt(oldclass_num/self._n_classes)
            #weight1 = 0.4
            
            loss += weight1 * loss1            
            d_loss = loss1.item()
                  
        return loss, c_loss, d_loss

    def _weight(self, dist):
        return torch.exp(-0.5*dist)
        #return torch.exp(-2*dist)
        
    def _Euclidean(self, input1, input2, dim=1, weight=1):
        #Euclidean
        distmat = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(len(input1), len(input2)) + \
                  torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(len(input2), len(input1)).t()
        distmat.addmm_(1, -2, input1, input2.t())
        return distmat
        #return torch.sqrt(torch.sum(weight * torch.pow(input1-input2+self.epslon,2),dim=dim))
    
    def _compute_predictions(self, data_loader):
        preds = torch.zeros(self._n_train_data, self._n_classes, device=self._device)

        for idxes, inputs, _ in data_loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            preds[idxes] = self._network(inputs).detach()

        return torch.sigmoid(preds)

    def _classify(self, data_loader):
        if self._means is None:
            raise ValueError(
                "Cannot classify without built examplar means,"
                " Have you forgotten to call `before_task`?"
            )
        if self._means.shape[0] != self._n_classes:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(self._means.shape[0]) +
                " with the number of classes ({}).".format(self._n_classes)
            )

        ypred = []
        ytrue = []

        for _, inputs, targets in data_loader:
            inputs = inputs.to(self._device)

            features = self._network.extract(inputs).detach()
            preds = self._get_closest(self._means, F.normalize(features))

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)
    
    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        return self._memory_size // self._n_classes

    # -----------------
    # Memory management
    # -----------------

    def build_examplars(self, inc_dataset):
        print("Building & updating memory.")

        self._data_memory, self._targets_memory = [], []
        self._class_weight = np.zeros((100, self._network.features_dim))
        self._exemplar_mean_withoutNorm = np.zeros((100, self._network.features_dim))
        self._exemplar_mean_withNorm = np.zeros((100, self._network.features_dim))
        self._data_mean_withoutNorm = np.zeros((100, self._network.features_dim))
        
        
        for class_idx in range(self._n_classes):
            inputs, loader = inc_dataset.get_custom_loader(class_idx, mode="test")
            features, targets = extract_features(
                self._network, loader
            )
            features_flipped, _ = extract_features(
                self._network, inc_dataset.get_custom_loader(class_idx, mode="flip")[1]
            )
            
            if class_idx >= self._n_classes - self._task_size:
                self._herding_matrix.append(select_examplars(
                    features, self._memory_per_class
                ))

            mean_withoutNorm, examplar_mean, alph = compute_examplar_mean(
                features, features_flipped, self._herding_matrix[class_idx], self._memory_per_class
            )
            self._data_memory.append(inputs[np.where(alph == 1)[0]])
            self._targets_memory.append(targets[np.where(alph == 1)[0]])
            ###########data mean##################################################
            #training_mean = np.mean(features, axis=0)
            #self._data_mean_withoutNorm[class_idx, :] = training_mean
            #self._data_mean_withoutNorm[class_idx, :] /= np.linalg.norm(self._data_mean_withoutNorm[class_idx, :])
            ###########data mean##################################################
            #self._class_weight[class_idx, :] = self._network.classifier.weight[class_idx, :].detach().cpu()
            #self._class_weight[class_idx, :] /= np.linalg.norm(self._class_weight[class_idx, :])
            
            self._exemplar_mean_withNorm[class_idx, :] = examplar_mean
            self._exemplar_mean_withoutNorm[class_idx, :] = mean_withoutNorm
            
            self._exemplar_mean_withNorm[class_idx, :] /= np.linalg.norm(self._exemplar_mean_withNorm[class_idx, :])
            self._exemplar_mean_withoutNorm[class_idx, :] /= np.linalg.norm(self._exemplar_mean_withoutNorm[class_idx, :])

        self._data_memory = np.concatenate(self._data_memory)
        self._targets_memory = np.concatenate(self._targets_memory)

    def get_memory(self):
        return self._data_memory, self._targets_memory


def extract_features(model, loader):
    targets, features = [], []

    for _inputs, _targets, index in loader:
        _targets = _targets.numpy()
        _features = model.extract(_inputs.to(model.device)).detach().cpu().numpy()

        features.append(_features)
        targets.append(_targets)

    return np.concatenate(features), np.concatenate(targets)


def select_examplars(features, nb_max):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
        np.sum(herding_matrix != 0) == min(nb_max, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    return herding_matrix


def compute_examplar_mean(feat_norm, feat_flip, herding_mat, nb_max):
    D = feat_norm.T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)

    D2 = feat_flip.T
    D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

    alph = herding_mat
    alph = (alph > 0) * (alph < nb_max + 1) * 1.

    alph_mean = alph / np.sum(alph)

    mean = (np.dot(D, alph_mean) + np.dot(D2, alph_mean)) / 2
    mean /= np.linalg.norm(mean)
    
    #
    D = feat_norm.T
    D2 = feat_flip.T
    mean_withoutNorm = (np.dot(D, alph_mean) + np.dot(D2, alph_mean)) / 2
    #
    return mean_withoutNorm, mean, alph


def compute_nearest_accuracy(model, loader, class_means, class_size, task_size):
    features, targets_ = extract_features(model, loader)

    targets = np.zeros((targets_.shape[0], 100), np.float32)
    targets[range(len(targets_)), targets_.astype('int32')] = 1.
    features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

    # Compute score for iCaRL
    sqd = cdist(class_means[:int(class_size)*task_size,:], features, 'sqeuclidean')
    score_icarl = (-sqd).T
    pred_top5=pred_topk(score_icarl, targets_)
    
    return np.argsort(score_icarl, axis=1)[:, -1], targets_, pred_top5

def compute_classifier_accuracy(model, loader, class_means):
    targets, predict = [], []
    softmax = torch.nn.Softmax(dim=1)
    _all_predict = []
    for _inputs, _targets, index in loader:
        _targets = _targets.numpy()
        _predict = model.classifier(model.extract(_inputs.to(model.device))).detach()
        _predict = softmax(_predict).cpu().numpy()
        predict.append(np.argsort(_predict, axis=1)[:, -1])
        _all_predict.append(_predict)
        targets.append(_targets)
        
    predict_ = np.concatenate(predict)
    targets_ = np.concatenate(targets)
    all_predict_ = np.concatenate(_all_predict)
    pred_top5=pred_topk(all_predict_, targets_)
    return predict_, targets_, pred_top5

def pred_topk(pred, target):
    pred, target = torch.tensor(pred), torch.tensor(target)
    _, top5 = pred.topk(5, 1, True, True)
    top5 = top5.t()
    correct = top5.eq(target.view(1, -1).expand_as(top5))
    correct = correct.sum(dim=0)
    target[correct==0] = target[correct==0]-1
    return target.numpy()
