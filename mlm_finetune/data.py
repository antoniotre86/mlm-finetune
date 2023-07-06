import glob
import os
import tempfile
import zipfile
from typing import List

import pytorch_lightning as pl
from transformers import AutoTokenizer, DataCollatorForWholeWordMask, DataCollatorForLanguageModeling
from torch.utils.data import random_split, DataLoader, Subset
from datasets import Dataset, NamedSplit

DATALOADER_NUM_WORKERS = 4

BATCH_SIZE = 8


class MaskedLMDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_path: str,
            pretrained_path: str,
            val_size: float,
            mlm_probability: float,
            chunk_size: int = 128,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.val_size = val_size
        self.chunk_size = chunk_size
        self.mlm_probability = mlm_probability

    def setup(self, stage):
        data_raw: List[str] = load_texts(self.data_path)
        train_data, validation_data = random_split(data_raw, (1 - self.val_size, self.val_size))
        train_data = Dataset.from_dict({"text": train_data}, split=NamedSplit("train"))
        validation_data = Dataset.from_dict({"text": validation_data}, split=NamedSplit("validation"))
        train_data_tokenized = train_data.map(self.tokenize_, batched=True, remove_columns=["text"])
        validation_data_tokenized = validation_data.map(self.tokenize_, batched=True, remove_columns=["text"])
        train_data_tokenized = train_data_tokenized.map(self.concatenate_and_chunk, batched=True)
        validation_data_tokenized = validation_data_tokenized.map(self.concatenate_and_chunk, batched=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_probability)
        self.train_dataset = train_data_tokenized
        self.validation_dataset = validation_data_tokenized
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=self.data_collator,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=self.data_collator,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def tokenize_(self, examples):
        result = self.tokenizer(examples["text"])
        # if self.tokenizer.is_fast:
        #     result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    def concatenate_and_chunk(self, tokenized):
        # Concatenate all texts
        concatenated_examples = {k: sum(tokenized[k], []) for k in tokenized.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(tokenized.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // self.chunk_size) * self.chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i: i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result


def load_texts(path: str) -> List[str]:
    zf = zipfile.ZipFile(path)
    data = []
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        for fname in glob.glob(os.path.join(tempdir, "**/*.txt"), recursive=True):
            with open(fname, encoding="utf-8") as f:
                try:
                    text = f.read()
                    data.append(text)
                except UnicodeDecodeError:
                    continue
    return data

