import torch
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.clock_driven import functional
import os
from tqdm import tqdm


class BaseTrainer:
    """A base class for trainers, providing basic save/load functionalities."""

    def __init__(self, model, optimizer, device, max_epoch, cfg, loss_function=F.cross_entropy):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.max_epoch = max_epoch
        self.cfg = cfg
        self.loss_function = loss_function
        self.start_epoch = 0
        self.epoch = 0
        self.max_acc = 0
        # 移除混合精度训练的scaler

    def save_model(self):
        """Saves the model checkpoint to the predefined path if validation accuracy improves."""
        # 直接使用在 vit_tl.py 中定义的 model_path
        # 确保目录存在
        save_dir = os.path.dirname(self.model_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        torch.save(self.model.state_dict(), self.model_path)
        print(f'\nBest model saved to {self.model_path}')

    def load_model(self, path):
        """Loads a model checkpoint from the given path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f'Model loaded from {path}')


class ViTAlignmentTLTrainer(BaseTrainer):
    """A specialized trainer for the ViTSNN model to handle transfer learning with multiple CKA losses."""

    def __init__(self, model, optimizer, device, max_epoch, cfg, writer, model_path,
                 sps_cka_loss_weight, block_cka_loss_weight, attn_cka_loss_weight,
                 loss_function=F.cross_entropy):
        super().__init__(model, optimizer, device, max_epoch, cfg, loss_function)
        self.writer = writer
        self.model_path = model_path  # 使用在主脚本中定义的完整模型路径
        self.sps_cka_loss_weight = sps_cka_loss_weight
        self.block_cka_loss_weight = block_cka_loss_weight
        self.attn_cka_loss_weight = attn_cka_loss_weight

    def train(self, train_loader, val_data_loader):
        """Main training loop. Returns best training and validation accuracy."""
        best_train_acc = 0.0
        print(f"Starting training for {self.max_epoch} epochs...")

        for self.epoch in range(self.start_epoch, self.max_epoch):
            print(f"\n===== Epoch {self.epoch + 1}/{self.max_epoch} =====")

            # --- Training Phase ---
            print("--- Training Phase ---")
            train_loss, train_acc = self._train_one_epoch(train_loader)
            print(f"  Training Summary: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")
            if train_acc > best_train_acc:
                best_train_acc = train_acc

            # --- Validation Phase ---
            print("--- Validation Phase ---")
            val_acc = self._val_one_epoch(val_data_loader)
            print(f"  Validation Summary: Accuracy={val_acc:.4f}, Best Accuracy So Far={self.max_acc:.4f}")

            self.writer.add_scalar('Accuracy/Validation', val_acc, self.epoch)

        print("\nTraining finished.")
        return best_train_acc, self.max_acc

    def _train_one_epoch(self, train_loader):
        """Runs a single training epoch."""
        self.model.train()
        total_loss, total_s_correct, total_t_correct, total_samples = 0, 0, 0, 0
        total_s_clf_loss, total_t_clf_loss = 0, 0
        total_sps_loss, total_block_loss, total_attn_loss = 0, 0, 0

        # 使用简化的进度条，避免与详细信息冲突
        for batch_idx, data_batch in enumerate(train_loader):
            # Robust unpacking
            if len(data_batch) == 2:
                # This handles the case where the dataloader returns (data, label)
                (rgb_img, dvs_img), label = data_batch[0], data_batch[1]
            elif len(data_batch) == 3:
                # This handles the case where the dataloader returns (rgb, dvs, label)
                rgb_img, dvs_img, label = data_batch[0], data_batch[1], data_batch[2]
            else:
                raise ValueError(f"Unexpected data format from DataLoader. Expected 2 or 3 items, got {len(data_batch)}")
            source, target, label = rgb_img.to(self.device), dvs_img.to(self.device), label.to(self.device)

            # Expand RGB data to T steps
            if source.dim() == 4:
                source = source.unsqueeze(1).repeat(1, self.cfg.T, 1, 1, 1)

            self.optimizer.zero_grad()

            # 移除混合精度计算，使用常规训练
            s_output, t_output, sps_loss, block_loss, attn_loss = self.model(source, target)
            
            s_clf_loss = self.loss_function(s_output, label)
            t_clf_loss = self.loss_function(t_output, label)
            tl_loss = (self.sps_cka_loss_weight * sps_loss + self.block_cka_loss_weight * block_loss + self.attn_cka_loss_weight * attn_loss)
            loss = s_clf_loss + t_clf_loss + tl_loss

            loss.backward()
            self.optimizer.step()

            functional.reset_net(self.model)

            # 计算累计统计
            total_loss += loss.item()
            total_s_clf_loss += s_clf_loss.item()
            total_t_clf_loss += t_clf_loss.item()
            total_sps_loss += sps_loss.item()
            total_block_loss += block_loss.item()
            total_attn_loss += attn_loss.item()
            
            # 分别计算源域和目标域准确率
            s_correct = (s_output.argmax(1) == label).float().sum().item()
            t_correct = (t_output.argmax(1) == label).float().sum().item()
            total_s_correct += s_correct
            total_t_correct += t_correct
            total_samples += label.size(0)

            # 计算当前批次的准确率
            s_acc = s_correct / label.size(0)
            t_acc = t_correct / label.size(0)

            # 更新进度条（第一行）
            progress_bar_length = 50
            filled_length = int(progress_bar_length * (batch_idx + 1) // len(train_loader))
            bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
            progress = f"Training: |{bar}| {batch_idx + 1}/{len(train_loader)}"
            
            # 显示详细信息（第二行）
            detail_info = f"Detail: Total={loss.item():.4f} | S_CLF={s_clf_loss.item():.4f} | T_CLF={t_clf_loss.item():.4f} | SPS={sps_loss.item():.4f} | Attn={attn_loss.item():.4f}"
            if self.block_cka_loss_weight > 0:
                detail_info += f" | Block={block_loss.item():.4f}"
            detail_info += f" | S_Acc={s_acc:.3f} | T_Acc={t_acc:.3f}"
            
            # 使用光标控制实现两行动态更新
            print(f"\r\033[K{progress}\n\033[K{detail_info}\033[A", end="", flush=True)

        # 清除两行显示信息
        print("\r\033[K\n\033[K\033[A", end="", flush=True)
        
        num_batches = len(train_loader)
        train_acc = (total_s_correct + total_t_correct) / (total_samples * 2)
        
        # 使用全局步数记录每个epoch的平均指标
        global_step = self.epoch
        self.writer.add_scalar('Epoch_Loss/Total', total_loss / num_batches, global_step)
        self.writer.add_scalar('Epoch_Loss/Source_CLF', total_s_clf_loss / num_batches, global_step)
        self.writer.add_scalar('Epoch_Loss/Target_CLF', total_t_clf_loss / num_batches, global_step)
        self.writer.add_scalar('Epoch_Loss_CKA/SPS', total_sps_loss / num_batches, global_step)
        self.writer.add_scalar('Epoch_Loss_CKA/Block', total_block_loss / num_batches, global_step)
        self.writer.add_scalar('Epoch_Loss_CKA/Attention', total_attn_loss / num_batches, global_step)
        self.writer.add_scalar('Epoch_Accuracy/Train', train_acc, global_step)
        self.writer.add_scalar('Epoch_Accuracy/Source_Train', total_s_correct / total_samples, global_step)
        self.writer.add_scalar('Epoch_Accuracy/Target_Train', total_t_correct / total_samples, global_step)

        return total_loss / num_batches, train_acc

    def _val_one_epoch(self, val_data_loader):
        """Runs a single validation epoch."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, item in enumerate(val_data_loader):
                (rgb_img, dvs_img), label = item
                dvs_img, label = dvs_img.to(self.device), label.to(self.device)
                # Pass DVS data as the `source` argument for evaluation
                target_clf = self.model(source=dvs_img)
                predict = torch.argmax(target_clf, dim=1)
                batch_correct = (predict == label).sum().item()
                batch_acc = batch_correct / label.size(0)
                correct += batch_correct
                total += label.size(0)
                
                # 更新进度条（第一行）
                progress_bar_length = 50
                filled_length = int(progress_bar_length * (batch_idx + 1) // len(val_data_loader))
                bar = '█' * filled_length + '-' * (progress_bar_length - filled_length)
                progress = f"Validation: |{bar}| {batch_idx + 1}/{len(val_data_loader)}"
                
                # 显示详细信息（第二行）
                detail_info = f"Detail: Batch_Acc={batch_acc:.3f} | Total_Acc={correct/total:.3f} | Samples={total}"
                
                # 使用光标控制实现两行动态更新
                print(f"\r\033[K{progress}\n\033[K{detail_info}\033[A", end="", flush=True)
                
                functional.reset_net(self.model)
        
        # 清除两行显示信息
        print("\r\033[K\n\033[K\033[A", end="", flush=True)
        
        acc = correct / total
        if acc > self.max_acc:
            self.max_acc = acc
            self.save_model()
        return acc

    def test(self, test_data_loader):
        """Tests the model on the given data loader(s)."""
        if isinstance(test_data_loader, list):
            results = [self._test_epoch(loader) for loader in test_data_loader]
            losses, accs1, accs5 = map(list, zip(*results))
            return losses, accs1, accs5
        else:
            return self._test_epoch(test_data_loader)

    def _test_epoch(self, data_loader):
        """Runs a single test epoch on a single data loader."""
        self.model.eval()
        total_loss, correct1, correct5, total = 0, 0, 0, 0
        pbar = tqdm(data_loader, desc='Testing', ncols=120, position=0)
        with torch.no_grad():
            for item in pbar:
                (rgb_img, dvs_img), label = item
                dvs_img, label = dvs_img.to(self.device), label.to(self.device)
                # Pass DVS data as the `source` argument for evaluation
                output = self.model(source=dvs_img)
                loss = self.loss_function(output, label)
                total_loss += loss.item()

                _, pred = output.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(label.view(1, -1).expand_as(pred))

                correct1 += correct[0].reshape(-1).float().sum(0, keepdim=True).item()
                correct5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                total += label.size(0)
                functional.reset_net(self.model)

        avg_loss = total_loss / len(data_loader)
        acc1 = 100 * correct1 / total
        acc5 = 100 * correct5 / total
        return avg_loss, acc1, acc5
