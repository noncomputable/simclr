from torch import nn

class Embeddor(nn.Module):
    def __init__(self, base_model, base_output_size, embed_size):
        super().__init__()
       
        self.base_model = base_model 
        self.projector = nn.Sequential(
            nn.Linear(in_features=base_output_size, out_features=embed_size),
            nn.ReLU(),
            nn.Linear(in_features=embed_size, out_features=embed_size)
        )

    def forward(self, image):
        base_embed = self.base_model(image)
        embed = self.projector(base_embed)
        return embed
