from dataloader.localdataset_lpm_real_lr import LocalImageDataset_LPM
from dataloader.testdataset_real_lr import TestDataset


def build_dataset(train_data_path: str, valid_data_path: str, final_reso: int):
    train_set = LocalImageDataset_LPM(pngtxt_dir=train_data_path, image_size=final_reso)
    val_set = TestDataset(valid_data_path, resize_bak=True)
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}')
    return train_set, val_set
