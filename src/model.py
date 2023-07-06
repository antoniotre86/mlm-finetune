import pytorch_lightning as pl
from transformers import AutoModelForMaskedLM, AutoConfig
from transformers.optimization import AdamW
from typing import Tuple
import torch.nn.functional as F


class MLMLightningModule(pl.LightningModule):
    def __init__(
            self,
            pretrained_path: str,
            learning_rate: float = 5e-5,
            adam_betas: Tuple[float, float] = (0.9, 0.999),
            adam_epsilon: float = 1e-8
    ):
        super().__init__()
        self.save_hyperparameters()
        model_config = AutoConfig.from_pretrained(pretrained_path, return_dict=True)
        # self.model = AutoModelForMaskedLM.from_pretrained(pretrained_path, config=model_config)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_path)
        self.vocab_size = model_config.vocab_size

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        # logits = self.model(**batch).logits
        # loss = F.cross_entropy(logits.view(-1, self.vocab_size), batch["labels"].view(-1))
        loss = self.model(**batch).loss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('valid_loss', loss, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.hparams.learning_rate,
                          betas=self.hparams.adam_betas,
                          eps=self.hparams.adam_epsilon,)
        return optimizer
