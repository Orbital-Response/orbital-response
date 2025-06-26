from orbital_response.ml_logic.model.imports import *
from orbital_response.ml_logic.model.config import * 
from orbital_response.ml_logic.model.model_architecture import *
from orbital_response.ml_logic.model.dataloaders import *

############### COMPUTE WEIGHT FOR THE LOSS (DUE TO INBALANCE)  ###############
def compute_pos_weight(mask_dirs, building_mask_dirs, target_size=(1024, 1024)):
    total_pos = 0
    total_neg = 0

    for mdir in mask_dirs:
        for fname in tqdm(os.listdir(mdir), desc=f"Processing {mdir}"):
            if not (fname.endswith("_post_disaster_mask.png") or fname.endswith("_mask.png")):
                continue

            if fname.endswith("_post_disaster_mask.png"):
                base_name = fname.replace("_post_disaster_mask.png", "")
            else:
                base_name = fname.replace("_mask.png", "")

            mask_path = os.path.join(mdir, fname)

            # Buscar la building_mask correspondiente
            build_path = next(
                (os.path.join(bdir, f"{base_name}_building_mask.png") 
                 for bdir in building_mask_dirs 
                 if os.path.exists(os.path.join(bdir, f"{base_name}_building_mask.png"))),
                None
            )

            if build_path is None:
                continue

            mask = np.array(Image.open(mask_path).resize(target_size, resample=Image.NEAREST))
            building = np.array(Image.open(build_path).resize(target_size, resample=Image.NEAREST))

            mask_bin = (mask > 1).astype(np.uint8)
            building_bin = (building > 127).astype(np.uint8)

            valid_pixels = mask_bin * building_bin
            pos = np.sum(valid_pixels == 1)
            neg = np.sum(building_bin) - pos

            total_pos += pos
            total_neg += neg

    if total_pos == 0:
        total_pos = 1  # avoid division by zero

    return total_neg / total_pos

mask_dirs = ["../data/filtered/data_secondary/split/train/masks",]
building_mask_dirs = ["../data/filtered/data_secondary/split/train/building_masks",]
pos_weight_val = compute_pos_weight(mask_dirs, building_mask_dirs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetModelDestruction().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

pos_weight = torch.tensor([pos_weight_val]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


############### THRESHOLD WILL DETERMINE THE % THAT WILL DETERMINE ONE CLASS  ###############
threshold=0.5
def binary_dice_score(preds, targets, threshold=threshold, building_mask=None, smooth=1e-6):
    preds = torch.sigmoid(torch.clamp(preds, -20, 20))  # [B, 1, H, W]
    preds_bin = (preds > threshold).float().squeeze(1)  # [B, H, W]
    targets = targets.float()

    if building_mask is not None:
        building_mask = building_mask.float()
        preds_bin *= building_mask
        targets *= building_mask

    intersection = (preds_bin * targets).sum(dim=(1, 2))
    union = preds_bin.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()

def compute_loss_and_dice(preds, targets, building_mask, criterion, threshold=threshold, smooth=1e-6):
    preds = preds.squeeze(1)  # [B, H, W]
    targets = targets.float()
    building_mask = building_mask.float()

    # Aplica building mask para focalizar la pérdida
    masked_preds = preds * building_mask
    masked_targets = targets * building_mask

    # Evita division por 0 si no hay edificios
    if building_mask.sum() < 1:
        zero = torch.tensor(0.0, device=preds.device)
        return zero.clone().requires_grad_(), zero

    loss = criterion(masked_preds, masked_targets)

    with torch.no_grad():
        probs = torch.sigmoid(torch.clamp(preds, -20, 20))
        preds_bin = (probs > threshold).float()
        intersection = ((preds_bin * targets) * building_mask).sum(dim=(1, 2))
        union = ((preds_bin + targets) * building_mask).sum(dim=(1, 2))
        dice = ((2 * intersection + smooth) / (union + smooth)).mean()

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
        self.best_epoch = 0 

    def __call__(self, val_metric, model, epoch=None):
        score = val_metric
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_model_wts = deepcopy(model.state_dict())
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
            if self.verbose:
                print(f"Better model found (DICE: {score:.4f})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Epochs without improving: {self.counter}/{self.patience}")
                print(f"Old Best Score is {self.best_score:.4f} found at epoch {self.best_epoch}")
            if self.counter >= self.patience:
                self.early_stop = True


def model2_training():
    train_losses, train_dcs = [], []
    val_losses, val_dcs = [], []

    early_stopper = EarlyStopping(PATIENCE_2, verbose=True)

    for epoch in tqdm(range(EPOCHS_2), desc="Epochs"):
        model.train()
        train_running_loss = 0 
        train_running_dc = 0

        for idx, (img, mask, building_mask_original) in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img.float().to(device)  # [B, 7, H, W] 
            mask = mask.long().to(device)  # [B, H, W]  Ground Truth
        
            building_mask_original = building_mask_original.float().to(device) # [B, H, W] (0 o 1)

            optimizer.zero_grad()

            y_pred = model(img)

            loss, dice = compute_loss_and_dice(y_pred, mask, building_mask_original, criterion)

            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            train_running_dc += dice.item()

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)

        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        model.eval()
        val_running_loss = 0
        val_running_dc = 0

        with torch.no_grad():
            for idx, (img, mask, building_mask_original) in enumerate(tqdm(val_dataloader, position=0, leave=True)):

                img = img.float().to(device) # [B, 7, H, W]
                mask = mask.long().to(device) # [B, H, W] Ground Truth

                building_mask_original = building_mask_original.float().to(device) # [B, H, W]

                y_pred = model(img)

                loss, dice = compute_loss_and_dice(y_pred, mask, building_mask_original, criterion)

                val_running_loss += loss.item()
                val_running_dc += dice.item()

        val_loss = val_running_loss / (idx + 1)
        val_dc = val_running_dc / (idx + 1)

        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        # ------------- Logging and EARLY STOPPING -------------\
        print("-" * 40)
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f} | DICE: {train_dc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | DICE: {val_dc:.4f}")
        print("-" * 40)

        early_stopper(val_dc, model, epoch=epoch + 1)
        if early_stopper.early_stop:
             print("Early stopped.")
             print(f"Best model found at epoch {early_stopper.best_epoch} with DICE {early_stopper.best_score:.4f}")
             break
    
    model.load_state_dict(early_stopper.best_model_wts)
    torch.save(model.state_dict(), model_2_v)

if __name__ == "__main__":
    model2_training()

