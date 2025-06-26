from orbital_response.ml_logic.model.imports import *
from orbital_response.ml_logic.model.config import * 
from orbital_response.ml_logic.model.model_architecture import *
from orbital_response.ml_logic.model.dataloaders import *

############### COMPUTE WEIGHT FOR THE LOSS (DUE TO INBALANCE)  ###############
def compute_pos_weight(mask_dirs):
    total_positives = 0
    total_negatives = 0

    for mask_dir in mask_dirs:
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

        for filename in tqdm(mask_files, desc=f"Calculating pos_weight at {mask_dir}"):
            mask = np.array(Image.open(os.path.join(mask_dir, filename)))
            mask_bin = (mask > 0).astype(np.uint8)

            total_positives += np.sum(mask_bin)
            total_negatives += mask_bin.size - np.sum(mask_bin)

    if total_positives == 0:
        raise ValueError("No positive pixels")

    return total_negatives / total_positives

mask_dir = [
    "../data/filtered/data_secondary/split/train/masks",
    ]

pos_weight_val = compute_pos_weight(mask_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
pos_weight = torch.tensor([pos_weight_val]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

############### THRESHOLD WILL DETERMINE THE % THAT WILL DETERMINE ONE CLASS  ###############
threshold_buildings=0.65

def binary_dice_score(preds, targets, threshold_buildings, smooth=1e-6):
    # logits → probas
    probs = torch.sigmoid(torch.clamp(preds, -20, 20))
    preds_bin = (probs > threshold_buildings).float()
    targets = targets.unsqueeze(1).float()

    intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()

def compute_loss_and_dice(y_pred, mask, criterion):
    loss = criterion(y_pred.squeeze(1), mask.float())
    with torch.no_grad():
        dice = binary_dice_score(y_pred, mask, threshold_buildings)
    return loss, dice


############### Early Stopper ###############
class EarlyStopping:
    def __init__(self, patience, verbose=True, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_metric, model):
        score = val_metric
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_model_wts = deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"Better model found (DICE: {score:.4f})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Epochs without improving: {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.early_stop = True


def model1_training():
    train_losses, train_dcs = [], []
    val_losses, val_dcs = [], []

    early_stopper = EarlyStopping(PATIENCE_1, verbose=True)

    for epoch in tqdm(range(EPOCHS_1), desc="Epochs"):
        model.train()
        train_running_loss = 0
        train_running_dc = 0

        for idx, (img, mask) in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img.float().to(device)                    # [B, 6, 224, 224]
            mask = mask.long().to(device)                   # [B, 224, 224]

            y_pred = model(img)                             # [B, 1, 224, 224]
            optimizer.zero_grad()

            loss, dice = compute_loss_and_dice(y_pred, mask, criterion)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            train_running_dc += dice.item()

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)

        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        ############### VALIDATION ###############
        model.eval()
        val_running_loss = 0
        val_running_dc = 0

        with torch.no_grad():
            for idx, (img, mask) in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img.float().to(device)
                mask = mask.long().to(device)

                y_pred = model(img)
                loss, dice = compute_loss_and_dice(y_pred, mask, criterion)

                val_running_loss += loss.item()
                val_running_dc += dice.item()

        val_loss = val_running_loss / (idx + 1)
        val_dc = val_running_dc / (idx + 1)

        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        ############### LOGGING ###############
        
        print("-" * 40)
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f} | DICE: {train_dc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | DICE: {val_dc:.4f}")
        print("-" * 40)

        ############## EARLY STOPPER ###############
        
        early_stopper(val_dc, model)  ###############################################
        
        if early_stopper.early_stop:
            print("🛑 Early stopped.")
            break

    model.load_state_dict(early_stopper.best_model_wts)
    torch.save(model.state_dict(), model_1_v)

###################### CREATE ALL MASKS FROM PREDICTIONS ##########################  
def generate_building_masks(base_dataset, model_building, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(base_dataset, batch_size=1, shuffle=False)

    model_building.eval()
    with torch.no_grad():
        for i, (img, _) in enumerate(tqdm(loader, desc=f"Generating {output_dir}")):
            img = img.to(device)
            pred = torch.sigmoid(model_building(img))
            bin_mask = (pred > 0.765).float().squeeze(0).squeeze(0)  # [H, W]

            mask_img = to_pil_image(bin_mask.cpu())
            image_id = base_dataset.ids[i]
            mask_img.save(os.path.join(output_dir, f"{image_id}_building_mask.png"))

def create_mask_files():
    splits = {
        "train": "../data/filtered/data_secondary/split/train",
        "val": "../data/filtered/data_secondary/split/val",
        "test": "../data/filtered/data_secondary/split/test"
    }

    model.load_state_dict(torch.load(model_1_v, map_location=device))
    model.eval()

    for split_name, path in splits.items():
        dataset = SecondaryDataset(root_dir=path, mask_transform=mask_transform)
        output_dir = os.path.join(path, "building_masks")
        generate_building_masks(dataset, model, device, output_dir)

if __name__ == "__main__":
    model1_training()
    create_mask_files()