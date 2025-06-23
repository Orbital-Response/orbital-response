from orbital_response.ml_logic.model.imports import *

class MaskTransform:
    def __init__(self, size=(1024, 1024)):
        self.size = size

    def __call__(self, mask):
        # Nearest to mantain classes
        mask = TF.resize(mask, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        mask_np = np.array(mask)
        mask_bin = (mask_np > 0).astype(np.uint8)
        return torch.from_numpy(mask_bin).long()
class MaskTransformDamage:
    def __init__(self, size=(1024, 1024)):
        self.size = size

    def __call__(self, mask):
        # Nearest to mantain classes
        mask = TF.resize(mask, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        mask_np = np.array(mask)
        mask_bin = (mask_np > 2).astype(np.uint8)
        return torch.from_numpy(mask_bin).long()
    
mask_transform = MaskTransform(size=(1024, 1024))
mask_transform = MaskTransformDamage(size=(1024, 1024))

class SecondaryDataset(Dataset):
    def __init__(self, root_dir, mask_transform=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.mask_transform = mask_transform

        self.resize = transforms.Resize((1024, 1024))
        self.to_tensor = transforms.ToTensor()
        ############### NEEDED SINCE THE ORIGINAL TRAINED MODEL WAS NORMALIZED ##############
        self.normalize = transforms.Normalize(mean=[0.485]*6, std=[0.229]*6)

        all_mask_files = os.listdir(self.mask_dir)
        self.ids = sorted(set(
            f.replace("_mask.png", "").replace("_mask.png", "")
            for f in all_mask_files if f.endswith(".png")
        ))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]

        pre_img = Image.open(os.path.join(self.image_dir, f"{image_id}_pre_disaster.png")).convert("RGB")
        post_img = Image.open(os.path.join(self.image_dir, f"{image_id}_post_disaster.png")).convert("RGB")
        mask_img = Image.open(os.path.join(self.mask_dir, f"{image_id}_mask.png"))

        # Resize & Tensor
        pre = self.to_tensor(self.resize(pre_img))
        post = self.to_tensor(self.resize(post_img))
        image = torch.cat([pre, post], dim=0)  # [6, H, W]
        image = self.normalize(image)

        if self.mask_transform:
            mask = self.mask_transform(mask_img)
        else:
            mask = torch.zeros((1024, 1024), dtype=torch.long)  # Fallback

        return image, mask
class SecondaryDatasetDamage(Dataset):
    def __init__(self, root_dir, mask_transform=None, include_building_mask=True):
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.include_building_mask = include_building_mask
        self.mask_transform = mask_transform

        self.building_mask_dir = os.path.join(root_dir, "building_masks") if include_building_mask else None

        self.resize = transforms.Resize((1024, 1024))
        self.to_tensor = transforms.ToTensor() #[0,1]

        self.normalize_7_channels = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.5],
            std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.25]
        )

        all_mask_files = os.listdir(self.mask_dir)
        self.ids = sorted(set(
            f.replace("_mask.png", "").replace("_mask.png", "")
            for f in all_mask_files if f.endswith(".png")
        ))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]

        # Rutes
        pre_img_path = os.path.join(self.image_dir, f"{image_id}_pre_disaster.png")
        post_img_path = os.path.join(self.image_dir, f"{image_id}_post_disaster.png")
        mask_img_path = os.path.join(self.mask_dir, f"{image_id}_mask.png")

        # load images ipl
        pre_img = Image.open(pre_img_path).convert("RGB")
        post_img = Image.open(post_img_path).convert("RGB")
        mask_img = Image.open(mask_img_path)

        # Transform
        pre_tensor = self.to_tensor(self.resize(pre_img))   # [3, H, W], to  [0,1]
        post_tensor = self.to_tensor(self.resize(post_img)) # [3, H, W], to  [0,1]

        # Concat pre post (6 chan)
        image_for_model = torch.cat([pre_tensor, post_tensor], dim=0) # [6, H, W]

        building_mask_original_binary = None

        if self.include_building_mask:
            building_path = os.path.join(self.building_mask_dir, f"{image_id}_building_mask.png")
            building_img = Image.open(building_path).convert("L")

            building_mask_original_binary = transforms.ToTensor()(self.resize(building_img)) # [1, H, W]
            building_mask_original_binary = (building_mask_original_binary > 0).float().squeeze(0) # [H, W] (0.0 o 1.0)

            building_tensor_for_model = transforms.ToTensor()(self.resize(building_img)) # [1, H, W]

            image_for_model = torch.cat([image_for_model, building_tensor_for_model], dim=0) # [7, H, W]

        image_for_model = self.normalize_7_channels(image_for_model)

        if self.mask_transform:
            mask_tensor = self.mask_transform(mask_img)
        else:
            mask_tensor = transforms.ToTensor()(self.resize(mask_img))
            mask_tensor = (mask_tensor >= 2).long().squeeze(0) #[H, W] long

        return image_for_model, mask_tensor, building_mask_original_binary


class AugmentedSecondaryCropDataset(Dataset):
    def __init__(self, image_dir, mask_dir, crop_size=(224, 224), min_ratio=0.1, max_attempts=40):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.crop_size = crop_size
        self.min_ratio = min_ratio
        self.max_attempts = max_attempts

        self.filenames = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith("_mask.png")
        ])

        ############### TYPE OF AUGMENTS ###############
        self.augment = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=2, p=0.5),
            A.RandomBrightnessContrast(p=0.16),
            A.GaussianBlur(p=0.11),
        ])
        
        ############### NEEDED SINCE THE ORIGINAL MODEL WAS NORMALIZED #############
        self.to_tensor = A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
            ),
            ToTensorV2(transpose_mask=True),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base_name = self.filenames[idx].replace("_mask.png", "")
        pre_path = os.path.join(self.image_dir, f"{base_name}_pre_disaster.png")
        post_path = os.path.join(self.image_dir, f"{base_name}_post_disaster.png")
        mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")

        # Load images
        pre = np.array(Image.open(pre_path).convert("RGB").resize((1024, 1024)))
        post = np.array(Image.open(post_path).convert("RGB").resize((1024, 1024)))
        mask = np.array(Image.open(mask_path).resize((1024, 1024), resample=Image.NEAREST))
        mask = (mask > 0).astype(np.uint8)

        image = np.concatenate([pre, post], axis=-1)  # shape: [H, W, 6]
        h, w, _ = image.shape
        ch, cw = self.crop_size

        for _ in range(self.max_attempts):
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)

            img_crop = image[top:top + ch, left:left + cw, :]
            mask_crop = mask[top:top + ch, left:left + cw]

            ratio = np.count_nonzero(mask_crop) / (ch * cw)

            if ratio >= self.min_ratio:
                aug = self.augment(image=img_crop, mask=mask_crop)
                final = self.to_tensor(image=aug["image"], mask=aug["mask"])
                return final["image"], final["mask"].long()

        # Fallback (last crop without checking ratio %)
        final = self.to_tensor(image=img_crop, mask=mask_crop)
        return final["image"], final["mask"].long()
class AugmentedSecondaryCropDatasetDamage(Dataset):
    def __init__(self, image_dir, mask_dir, building_mask_dir, crop_size=(224, 224), min_ratio=0.3, max_attempts=40):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.building_mask_dir = building_mask_dir
        self.crop_size = crop_size
        self.min_ratio = min_ratio
        self.max_attempts = max_attempts

        self.filenames = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith("_mask.png")
        ])

        self.augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.25),
            A.GaussianBlur(p=0.05),
        ])

        self.to_tensor_and_normalize = A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.5),
                std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.25)
            ),

            ToTensorV2(transpose_mask=True),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base_name = self.filenames[idx].replace("_mask.png", "")
        pre_path = os.path.join(self.image_dir, f"{base_name}_pre_disaster.png")
        post_path = os.path.join(self.image_dir, f"{base_name}_post_disaster.png")
        mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")
        build_mask_path = os.path.join(self.building_mask_dir, f"{base_name}_building_mask.png")

        #1024x1024
        pre = np.array(Image.open(pre_path).convert("RGB").resize((1024, 1024)))
        post = np.array(Image.open(post_path).convert("RGB").resize((1024, 1024)))
        mask = np.array(Image.open(mask_path).resize((1024, 1024), resample=Image.NEAREST))
        building = np.array(Image.open(build_mask_path).resize((1024, 1024), resample=Image.NEAREST))

        # (0 o 1)
        mask = (mask > 0.5).astype(np.uint8) # DMG mask
        building_mask_original_binary = (building > 127).astype(np.float32)

        # [H, W, 1]
        building_channel_for_model = (building > 127).astype(np.float32)[..., np.newaxis]

        # [H, W, 7]
        image_for_model = np.concatenate([pre, post, building_channel_for_model], axis=-1)

        h, w, _ = image_for_model.shape
        ch, cw = self.crop_size

        for _ in range(self.max_attempts):
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)

            img_crop = image_for_model[top:top + ch, left:left + cw, :]
            mask_crop = mask[top:top + ch, left:left + cw]
            building_mask_original_binary_crop = building_mask_original_binary[top:top + ch, left:left + cw]

            damage_in_building = (mask_crop == 1) * (building_mask_original_binary_crop == 1)
            total_building_pixels_in_crop = np.sum(building_mask_original_binary_crop)
            
            # no div cero
            if total_building_pixels_in_crop > 0:
                ratio = np.sum(damage_in_building) / total_building_pixels_in_crop
            else:
                ratio = 0.0

            if ratio >= self.min_ratio:
                augmented = self.augment(image=img_crop, mask=mask_crop)
                aug_image = augmented["image"]
                aug_mask = augmented["mask"]

                final_transformed = self.to_tensor_and_normalize(image=aug_image, mask=aug_mask)
        
                return final_transformed["image"], final_transformed["mask"].long(), torch.from_numpy(building_mask_original_binary_crop).float()

        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        img_crop = image_for_model[top:top + ch, left:left + cw, :]
        mask_crop = mask[top:top + ch, left:left + cw]
        building_mask_original_binary_crop = building_mask_original_binary[top:top + ch, left:left + cw]

        augmented = self.augment(image=img_crop, mask=mask_crop)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        final_transformed = self.to_tensor_and_normalize(image=aug_image, mask=aug_mask)
        return final_transformed["image"], final_transformed["mask"].long(), torch.from_numpy(building_mask_original_binary_crop).float()

