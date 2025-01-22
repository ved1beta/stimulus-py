"""This file provides the class API for handling the data in pytorch using the Dataset and Dataloader classes"""

from typing import Tuple

from torch.utils.data import Dataset

import src.stimulus.data.csv as csv
import src.stimulus.data.experiments as experiments


class TorchDataset(Dataset):
    """Class for creating a torch dataset"""

    def __init__(self, config_path: str, csv_path: str, encoder_loader: experiments.EncoderLoader, split: Tuple[None, int] = None) -> None:

        self.loader = csv.DatasetLoader(
            config_path=config_path,
            csv_path=csv_path,
            encoder_loader=encoder_loader,
            split=split,
        )

    def __len__(self) -> int:
        return len(self.loader)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, dict]:
        return (
            self.loader[idx]
        )
