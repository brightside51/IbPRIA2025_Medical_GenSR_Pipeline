from dataset import DEFAULTDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(cfg):
    if cfg.dataset.name == 'DEFAULT':
        train_dataset = DEFAULTDataset(root_dir=cfg.dataset.root_dir)
        val_dataset = DEFAULTDataset(root_dir=cfg.dataset.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler  # Return datasets if available
    else:
        raise ValueError(f'{cfg.dataset.name} Dataset is not available')