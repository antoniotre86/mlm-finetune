from src.data import MaskedLMDataModule, BATCH_SIZE


def test_data_module():
    chunk_size = 128
    data_module = MaskedLMDataModule(
        r"C:\Users\anton\data\nifty\sample.zip",
        r"C:\Users\anton\workspace\models\distilbert-base-uncased",
        val_size=0.2,
        mlm_probability=0.1,
        chunk_size=chunk_size
    )
    data_module.setup(None)
    for batch in data_module.train_dataloader():
        break
    assert batch["input_ids"].shape[0] == BATCH_SIZE
    assert batch["input_ids"].shape[1] == chunk_size

