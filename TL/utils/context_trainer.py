import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_one_epoch_context(model, train_loader, optimizer, criterion_cls, cka_weight, context_weight, device):
    """
    Trains the ViTSNN_Context model for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training Context Model")

    # CORRECTED: Unpacking the nested tuple from the dataloader
    for i, ((source_data, target_data), labels) in enumerate(progress_bar):
        source_data, target_data, labels = source_data.to(device), target_data.to(device), labels.to(device)

        optimizer.zero_grad()

        # --- Forward pass with the new model ---
        source_clf, target_clf, sps_cka_loss, block_cka_loss, context_loss = model(source_data, target_data)

        # --- Loss Calculation ---
        loss_cls_s = criterion_cls(source_clf, labels)
        loss_cls_t = criterion_cls(target_clf, labels)
        classification_loss = (loss_cls_s + loss_cls_t) / 2

        cka_loss = sps_cka_loss + block_cka_loss

        batch_loss = classification_loss + (cka_weight * cka_loss) + (context_weight * context_loss)

        # --- Backward pass and optimization ---
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * source_data.size(0)
        total_samples += source_data.size(0)

        progress_bar.set_postfix({
            'loss': f'{batch_loss.item():.4f}',
            'cls_loss': f'{classification_loss.item():.4f}',
            'cka_loss': f'{cka_loss.item():.4f}',
            'ctx_loss': f'{context_loss.item():.4f}'
        })

    avg_loss = total_loss / total_samples
    logging.info(f"Epoch Training Finished. Average Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate_context(model, test_loader, criterion_cls, device):
    """
    Evaluates the ViTSNN_Context model.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    progress_bar = tqdm(test_loader, desc="Evaluating")

    # The validation/test loader returns (data, labels)
    for i, (data, labels) in enumerate(progress_bar):
        data, labels = data.to(device), labels.to(device)

        outputs = model(data)
        loss = criterion_cls(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * data.size(0)

        accuracy = 100 * total_correct / total_samples
        progress_bar.set_postfix({'accuracy': f'{accuracy:.2f}%'})

    avg_loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples
    logging.info(f"Evaluation Finished. Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}")
    return accuracy, avg_loss
