import os
import pytorch_lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader


class Task1DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./Dataset"):
        super().__init__()
        self.data_dir = data_dir

    def _load_preference_data(self, path):
        buf = []
        with open(path) as f:
            for line in f:
                user, iid = line.strip().split(",")
                buf.append([user, iid])

        return buf

    def setup(self, stage: str):
        if stage == "fit":
            user_itemset = self._load_preference_data(os.path.join(self.data_dir, "user_itemset_training.csv"))
        # elif


    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def predict_dataloader(self):
        pass


class Task2DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./Dataset"):
        super().__init__()
