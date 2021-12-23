import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torch.multiprocessing import cpu_count
from torch.optim import Adam

from models import Embeddor
from loss import Contrastive
from dataset import AugmentedDataset

class SimCLREmbeddor(pl.LightningModule):
    def __init__(self, dataset, base_model, base_output_size, embed_size, temperature,
                 batch_size, train_count, val_count, lr):
        super().__init__()
        
        self.dataset = dataset 
        self.save_hyperparameters(ignore=["dataset", "base_model"])
        self.model = Embeddor(base_model, self.hparams.base_output_size, self.hparams.embed_size)
        self.loss = Contrastive(self.hparams.batch_size, self.hparams.temperature)

    def train_dataloader(self):
        train_indices = list(range(self.hparams.train_count))
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            sampler=SubsetRandomSampler(train_indices),
            drop_last=True
        )
         
        return dataloader

    def val_dataloader(self):
        val_indices = list(range(self.hparams.train_count + 1, self.hparams.train_count + self.hparams.val_count))
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=SequentialSampler(val_indices),
            drop_last=True
        )
        
        return dataloader

    def forward(self, image):
        return self.model(image)

    def step(self, batch, mode = "train"):
        (img_i, img_j), _ = batch
        emb_i = self.forward(img_i)
        emb_j = self.forward(img_j)
        loss = self.loss(emb_i, emb_j)
        loss_key = f"{mode}_loss"
        self.log(loss_key, loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)    
    
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def validation_end(self, outputs):
        if len(outputs) == 0:
            return {"val_loss": torch.tensor(0)}
        else:
            loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
