import copy

import torch
from torch import nn

#from inclearn.lib import factory
from lib import factory
from lib import normalized_fc
from lib import task_fc

class BasicNet(nn.Module):

    def __init__(
        self, convnet_type, use_bias=False, init="kaiming", use_multi_fc=False, device=None,normalize=False
    ):
        super(BasicNet, self).__init__()

        self.use_bias = use_bias
        self.init = init
        self.use_multi_fc = use_multi_fc
        self.normalize = normalize
        self.convnet = factory.get_convnet(convnet_type, nf=64, zero_init_residual=True)
        self.classifier = None
        self.task_classifier = None
        self.n_classes = 0
        self.device = device

        self.to(self.device)

    def forward(self, x, gamma_beta=None, ss_b=False, label=None):
        if self.classifier is None:
            raise Exception("Add some classes before training.")
        # to ss generator
        if ss_b is True:
            features, x_1, x_2, x_3 = self.convnet(x, ss=ss_b, resnet_label=label, gamma_beta=gamma_beta)
        elif ss_b is False:    
            features = self.convnet(x, resnet_label=label, gamma_beta=gamma_beta)
        
        if self.use_multi_fc:
            logits = []
            for clf_name in self.classifier:
                logits.append(self.__getattr__(clf_name)(features))
            logits = torch.cat(logits, 1)
        else:
            logits = self.classifier(features)
            
        if ss_b is True:
            return logits, x_1, x_2, x_3     
        return logits

    @property
    def features_dim(self):
        return self.convnet.out_dim

    def extract(self, x, label = None, gamma_beta=None):
        return self.convnet(x, resnet_label=label, gamma_beta=gamma_beta)
    
    # extract the feature with label to directly use the scaling and shifting
    def gb_extract(self, x, label):
        return self.convnet(x, resnet_label=label)
    
    def extract_pre(self, x):
        return self.convnet.previous_feature(x)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        if self.use_multi_fc:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.classifier is None:
            self.classifier = []

        new_classifier = self._gen_classifier(in_dim = self.convnet.out_dim, out_dim = n_classes, normalize = self.normalize)
        name = "_clf_{}".format(len(self.classifier))
        self.__setattr__(name, new_classifier)
        self.classifier.append(name)

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(in_dim = self.convnet.out_dim, out_dim = self.n_classes + n_classes, normalize = self.normalize)

        if self.classifier is not None:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier
    
    def _add_task_fc(self, n_classes):
        """
        if self.task_classifier is not None:
            weight = copy.deepcopy(self.task_classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.task_classifier.bias.data)
        """
        task_classifier = self._gen_classifier(in_dim = self.n_classes, out_dim = (self.n_classes)//n_classes, normalize = self.normalize)
        """
        if self.task_classifier is not None:
            task_classifier.weight.data[:(self.n_classes-n_classes)//n_classes] = weight
            if self.use_bias:
                task_classifier.bias.data[:(self.n_classes-n_classes)//n_classes] = bias
        """
        del self.task_classifier
        self.task_classifier = task_classifier
        
    def _gen_classifier(self, in_dim, out_dim, normalize=False):
        if normalize == False:
            classifier = nn.Linear(in_dim, out_dim, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.)
        else:
            classifier = normalized_fc.CosineLinear(in_dim, out_dim,self.device, bias=self.use_bias, eta=True).to(self.device)

        return classifier
