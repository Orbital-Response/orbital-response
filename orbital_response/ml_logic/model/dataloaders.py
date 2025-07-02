from orbital_response.ml_logic.model.data_prep import *
from orbital_response.ml_logic.model.config import *
from orbital_response.ml_logic.model.imports import *


train_secondary_dataset = AugmentedSecondaryCropDataset(
    image_dir="../data/filtered/data_secondary/split/train/images",
    mask_dir="../data/filtered/data_secondary/split/train/masks",
    crop_size=(224, 224),
    min_ratio=0.1
)
train_secondary_dataset_damage = AugmentedSecondaryCropDatasetDamage(
    image_dir="../data/filtered/data_secondary/split/train/images",
    mask_dir="../data/filtered/data_secondary/split/train/masks",
    building_mask_dir="../data/filtered/data_secondary/split/train/building_masks",
    crop_size=(224, 224),
    min_ratio=0.1
)

val_secondary_dataset = SecondaryDataset(root_dir="../data/filtered/data_secondary/split/val", mask_transform=mask_transform)
val_secondary_dataset_damage = SecondaryDatasetDamage(root_dir="../data/filtered/data_secondary/split/val", mask_transform=mask_transform, include_building_mask=True)

test_secondary_dataset = SecondaryDataset(root_dir="../data/filtered/data_secondary/split/test/", mask_transform=mask_transform)
test_secondary_dataset_damage = SecondaryDatasetDamage(root_dir="../data/filtered/data_secondary/split/test", mask_transform=mask_transform, include_building_mask=True)


############ Buildings detector dataloaders #########
train_dataloader = DataLoader(train_secondary_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=num_workers)
val_dataloader = DataLoader(val_secondary_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=num_workers)
test_dataloader = DataLoader(test_secondary_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=num_workers)

############ Damage destruction detector dataloaders #########
train_dataloader_damage = DataLoader(train_secondary_dataset_damage,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=num_workers)
val_dataloader_damage = DataLoader(val_secondary_dataset_damage,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=num_workers)
test_dataloader_damage = DataLoader(test_secondary_dataset_damage,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=num_workers)
