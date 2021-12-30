import pytorch_lightning as pl

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torch.multiprocessing import cpu_count
from torch.optim import Adam

from models import Embeddor, Classifier
from loss import Contrastive
from dataset import AugmentedDataset

class SimCLREmbeddor(pl.LightningModule):
    def __init__(self, dataset, base_model, base_output_size, embed_size, temperature,
                 batch_size, train_count, val_count, lr):
        super().__init__()
        
        self.save_hyperparameters(ignore=["dataset", "base_model"])
        self.dataset = dataset 
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

class SimCLRClassifier(pl.LightningModule):
    def __init__(self, dataset_class, embeddor_model, n_classes, freeze_base,
                 batch_size, epochs, lr):
        super().__init__()
        
        self.save_hyperparameters(ignore=["dataset", "embeddor_model"])
        self.dataset_class = dataset_class
        self.model = Classifier(embeddor_model, self.hparams.freeze_base, self.hparams.n_classes)
        self.loss = nn.CrossEntropyLoss()
    
    def get_dataloader(self, split):
        dataset = self.dataset_class("./data", split=split)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle="split"=="train",
            drop_last=False
        )

        return dataloader
    
    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("test")

    def forward(self, image):
        return self.model(image)        

    def step(self, batch, mode = "train"):
        img, true_cls = batch
        pred_cls = self.forward(img)
        loss = self.loss(pred_cls, true_cls)
        loss_key = f"{mode}_loss"
        self.log(loss_key, loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)    
    
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def validation_end(self, outputs):
        if len(outputs) == 0:
            return {"val_loss": torch.tensor(0)}
        else:
            loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, self.hparams.epochs)

        return [optimizer], [scheduler]
