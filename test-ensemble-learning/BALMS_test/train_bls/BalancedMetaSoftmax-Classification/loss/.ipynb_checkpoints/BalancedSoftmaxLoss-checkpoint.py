"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, freq_path):
        super(BalancedSoftmax, self).__init__()
        with open(freq_path, 'r') as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    
    sample_per_class_ = None
    tau = 0.0
    filename = f'tau_log_prior_{tau}.pt'
    spc = torch.load(filename)
#     sample_per_class_ = sample_per_class_.long()
#     print("_",sample_per_class_.dtype,"000",sample_per_class.dtype)
#     print(sample_per_class_)
#     print("--------------------------------------------------------------------------------")
#     spc = sample_per_class_.type_as(logits)
#     spc = sample_per_class.type_as(logits)
#     spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
#     import ipdb 
#     ipdb.set_trace()
    logits = logits + spc
    
    print(logits)
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

# def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
#     """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
#     Args:
#       labels: A int tensor of size [batch].
#       logits: A float tensor of size [batch, no_of_classes].
#       sample_per_class: A int tensor of size [no of classes].
#       reduction: string. One of "none", "mean", "sum"
#     Returns:
#       loss: A float tensor. Balanced Softmax Loss.
#     """
    



# #     print(sample_per_class)
#     spc = sample_per_class.type_as(logits)
#     spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
#     logits = logits + spc.log()
#     loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
#     return loss


def create_loss(freq_path):
    print('Loading Balanced Softmax Loss.')
    return BalancedSoftmax(freq_path)




# ### 0911 更改增加BALPOE
# class BSExpertLoss(nn.Module):
#     r"""
#     References:

#     """

#     def __init__(self, cls_num_list=None, tau_list=(0, 1, 2), eps=1e-9, **kwargs):
#         super().__init__()
#         self.base_loss = F.cross_entropy

#         self.register_buffer('bsce_weight', torch.tensor(cls_num_list).float())
#         self.register_buffer('tau_list', torch.tensor(tau_list).float())
#         self.num_experts = len(tau_list)
#         self.eps = eps

#         assert self.num_experts >= 1

#     def forward(self, output_logits, targets, extra_info=None, return_expert_losses=False):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (batch_size)
#         """
#         if extra_info is None:
#             return self.base_loss(output_logits, targets)  # output_logits indicates the final prediction

#         logits = extra_info['logits']
#         assert len(logits.shape) == 3
#         assert logits.shape[0] == self.num_experts

#         expert_losses = dict()
#         loss = 0.0

#         for idx in range(self.num_experts):
#             adjusted_expert_logits = logits[idx] + self.get_bias_from_index(idx)
#             expert_losses[f'loss_e_{idx}'] = expert_loss = self.base_loss(adjusted_expert_logits, targets)
#             loss = loss + expert_loss

#         loss = loss / self.num_experts

#         if return_expert_losses:
#             return loss, expert_losses
#         else:
#             return loss

#     def get_default_bias(self, tau=1):
#         prior = self.bsce_weight
#         prior = prior / prior.sum()
#         log_prior = torch.log(prior + self.eps)
#         print("------------------------------------------------tau * log_prior",tau * log_prior)
#         tau_log_prior = tau * log_prior
#         filename = f'tau_log_prior_{tau}.pt'
#         torch.save(tau_log_prior, filename)
#         return tau * log_prior

#     def get_bias_from_index(self, e_idx):
#         tau = self.tau_list[e_idx]
#         return self.get_default_bias(tau)