import torch
from torch import nn
from torch.nn import functional as F

class Contrastive(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
 
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer(
            "negatives_mask", 
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        )

    def forward(self, embed_i, embed_j):
        embed_i = F.normalize(embed_i, dim=1)
        embed_j = F.normalize(embed_j, dim=1)
        embed = torch.cat([embed_i, embed_j], dim=0)
        pairwise_sim = F.cosine_similarity(embed, embed.unsqueeze(1), dim=2)
        sim_ij = torch.diag(pairwise_sim, self.batch_size)
        sim_ji = torch.diag(pairwise_sim, -self.batch_size)
        positive_sim = torch.cat([sim_ij, sim_ji], dim=0)
        pos_scores = torch.exp(positive_sim / self.temperature)
        all_scores = self.negatives_mask * torch.exp(pairwise_sim / self.temperature)
        all_scores_summed = torch.sum(all_scores, dim=1)
        loss_per_pair = -torch.log(pos_scores / all_scores_summed)
        ave_loss = torch.sum(loss_per_pair) / (2 * self.batch_size)
        
        return ave_loss
