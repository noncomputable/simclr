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

class Classifier(nn.Module):
    def __init__(self, embeddor, freeze_base, n_classes):
        super().__init__()
        
        self.embeddor = embeddor.base_model
        
        if freeze_base:
            for param in self.embeddor.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(in_features=embeddor.projector[0].in_features,
                                    out_features=n_classes)

    def forward(self, image):
        emb = self.embeddor(image)
        class_ = self.classifier(emb)
        
        return class_
