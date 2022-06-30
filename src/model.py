import torch
from torch import nn

import pytorch_lightning as pl
import torchmetrics


class DrugTargetCoembeddingLightning(pl.LightningModule):
    def __init__(
        self,
        drug_dim=2048,
        target_dim=100,
        latent_dim=1024,
        activation=nn.ReLU,
        classify=True,
        lr=1e-4,
    ):
        super().__init__()

        self.drug_dim = drug_dim
        self.target_dim = drug_dim
        self.latent_dim = latent_dim
        self.activation = activation

        self.classify = classify
        self.lr = lr

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_dim, self.latent_dim), self.activation()
        )

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_dim, self.latent_dim), self.activation()
        )

        if self.classify:
            self.val_accuracy = torchmetrics.Accuracy()
            self.val_aupr = torchmetrics.AveragePrecision()
            self.val_auroc = torchmetrics.AUROC()
            self.val_f1 = torchmetrics.F1Score()
            self.metrics = {
                "acc": self.val_accuracy,
                "aupr": self.val_aupr,
                "auroc": self.val_auroc,
                "f1": self.val_f1,
            }
        else:
            self.val_mse = torchmetrics.MeanSquaredError()
            self.val_pcc = torchmetrics.PearsonCorrCoef()
            self.metrics = {"mse": self.val_mse, "pcc": self.val_pcc}

    def forward(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        if self.classify:
            similarity = nn.CosineSimilarity()(
                drug_projection, target_projection
            )
        else:
            similarity = torch.bmm(
                drug_projection.view(-1, 1, self.latent_dim),
                target_projection.view(-1, self.latent_dim, 1),
            ).squeeze()

        return similarity

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        drug, protein, label = train_batch
        similarity = self.forward(drug, protein)

        if self.classify:
            sigmoid = torch.nn.Sigmoid()
            similarity = torch.squeeze(sigmoid(similarity))
            loss_fct = torch.nn.BCELoss()
        else:
            loss_fct = torch.nn.MSELoss()

        loss = loss_fct(similarity, label)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, train_batch, batch_idx):
        drug, protein, label = train_batch
        similarity = self.forward(drug, protein)

        if self.classify:
            sigmoid = torch.nn.Sigmoid()
            similarity = torch.squeeze(sigmoid(similarity))
            loss_fct = torch.nn.BCELoss()
        else:
            loss_fct = torch.nn.MSELoss()

        loss = loss_fct(similarity, label)
        self.log("val/loss", loss)
        return {"loss": loss, "preds": similarity, "target": label}

    def validation_step_end(self, outputs):
        for name, metric in self.metrics.items():
            metric(outputs["preds"], outputs["target"])
            self.log(f"val/{name}", metric)
