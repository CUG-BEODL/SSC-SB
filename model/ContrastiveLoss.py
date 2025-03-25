# Reference: https://theaisummer.com/simclr/


import torch
import torch.nn as nn
import torch.nn.functional as F


def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size, batch_size, dtype=bool)).float()

    def calc_similarity_batch(self, representation):
        return F.cosine_similarity(representation.unsqueeze(0), representation.unsqueeze(1), dim=2)

    def forward(self, feature, label):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        z = F.normalize(feature, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z)  # (128, 128)

        l1 = torch.tile(label, dims=[self.batch_size]).reshape(self.batch_size, self.batch_size)
        l2 = l1.T
        mask_negative = device_as(torch.logical_xor(l1, l2), similarity_matrix)

        mask_positive = device_as(torch.logical_not(mask_negative), similarity_matrix)
        mask_positive = device_as(
            mask_positive.logical_and(device_as(~torch.eye(mask_positive.size(0)).bool(), similarity_matrix)),
            similarity_matrix)
        moninator = mask_positive * torch.exp(similarity_matrix / self.temperature)

        denominator = device_as(mask_negative, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log((torch.sum(moninator, dim=1) + 1e-8) / (
                    torch.sum(denominator, dim=1) + torch.sum(moninator, dim=1)))  # 每个样本的loss

        loss = torch.sum(all_losses) / self.batch_size

        mean_pos = torch.mul(similarity_matrix, device_as(mask_positive, similarity_matrix)).mean()

        mean_neg = torch.mul(similarity_matrix, device_as(mask_negative, similarity_matrix)).mean()

        return loss, mean_pos, mean_neg
