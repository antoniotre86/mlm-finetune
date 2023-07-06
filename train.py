from mlm_finetune.data import MaskedLMDataModule
import pytorch_lightning as pl

from mlm_finetune.model import MLMLightningModule

SEED = 123


def main():
    pl.seed_everything(SEED)
    # pretrained_path = r"C:\Users\anton\workspace\models\distilbert-base-uncased"
    pretrained_path = "distilbert-base-uncased"
    data_module = MaskedLMDataModule(
        r"C:\Users\anton\data\nifty\sample.zip",
        pretrained_path,
        val_size=0.2,
        mlm_probability=0.1,
        chunk_size=128
    )
    model = MLMLightningModule(
        pretrained_path=pretrained_path,
        learning_rate=5e-5,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8
    )
    trainer = pl.Trainer()
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
