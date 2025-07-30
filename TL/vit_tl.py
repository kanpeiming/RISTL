import torch
import argparse
import os
from utils.trainer import ViTAlignmentTLTrainer

from dataloader import *
from models.vit import ViTSNN
from torch.utils.tensorboard import SummaryWriter
from utils.common_utils import seed_all

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('--data_set', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'Caltech101', 'MNIST', 'ImageNet100',
                             'CINIC10_WO_CIFAR10', 'ImageNet2Caltech', 'Caltech51'],
                    help='the data set type.')
# 新增T参数，作为全局的时间步长(time steps)，是保证RGB和DVS数据时间维度一致的唯一来源
parser.add_argument('--T', type=int, default=4, help='SNN time steps.')
parser.add_argument('--batch_size', default=16, type=int, help='Batchsize')
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')  # 降低学习率防止梯度爆炸
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=200, type=int, help='Training epochs')

parser.add_argument('--seed', type=int, default=1000, help='seed for initializing training. ')
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, help='the ratio of used RGB training set. ')
parser.add_argument('--dvs_sample_ratio', type=float, default=1.0, help='the ratio of used dvs training set. ')
parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-GPU parallelism')


parser.add_argument('--log_dir', type=str, default='/home/user/kpm/results/RISTL/log_dir',
                    help='the path of tensorboard dir.')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/results/RISTL/checkpoints',
                    help='the path of checkpoint dir.')
parser.add_argument('--GPU_id', type=int, default=0, help='the id of used GPU.')
parser.add_argument('--num_classes', type=int, default=10, help='the number of data classes.')
parser.add_argument('--sps_weight', type=float, default=0.1, help='Weight for SPS CKA loss')  # 降低CKA损失权重
parser.add_argument('--block_weight', type=float, default=0.1, help='Weight for Block CKA loss')
parser.add_argument('--attn_weight', type=float, default=0.1, help='Weight for Attention CKA loss')

args = parser.parse_args()

device = torch.device(f"cuda:{args.GPU_id}")

log_name = f"{args.data_set}_sps{args.sps_weight}_block{args.block_weight}_attn{args.attn_weight}_lr{args.lr}_seed{args.seed}_T{args.T}"
writer = SummaryWriter(log_dir=os.path.join(args.log_dir, log_name))
checkpoint_path = args.checkpoint
model_path = os.path.join(checkpoint_path, f'{log_name}.pth')

# The SummaryWriter creates the log directory, so we only need to ensure the checkpoint directory exists.
os.makedirs(checkpoint_path, exist_ok=True)

print(log_name)
print(writer.log_dir)

if __name__ == "__main__":
    seed_all(args.seed)
    f = open(f"{args.data_set}_{args.seed}_{args.RGB_sample_ratio}_grid_result.txt", "a")

    # preparing data
    if args.data_set == 'CIFAR10':
        train_loader, dvs_test_loader = get_dvs_aligned_tl_cifar10(args.batch_size, args.RGB_sample_ratio,
                                                                   args.dvs_sample_ratio, T=args.T)
    elif args.data_set == 'CINIC10_WO_CIFAR10':
        train_loader, dvs_test_loader = get_tl_cinic10_wo_cifar10(args.batch_size, args.RGB_sample_ratio,
                                                                  args.dvs_sample_ratio)
    elif args.data_set == 'ImageNet2Caltech':
        train_loader, dvs_test_loader = get_tl_imagenet2caltech(args.batch_size, args.RGB_sample_ratio,
                                                                args.dvs_sample_ratio)
    elif args.data_set == 'Caltech51':
        train_loader, dvs_test_loader = get_tl_imagenet2caltech(args.batch_size, args.RGB_sample_ratio,
                                                                args.dvs_sample_ratio)
    elif args.data_set == 'Caltech101':
        train_loader, dvs_test_loader = get_tl_caltech101(args.batch_size, args.RGB_sample_ratio, args.dvs_sample_ratio)
    elif args.data_set == 'MNIST':
        train_loader, dvs_test_loader = get_tl_mnist(args.batch_size, args.RGB_sample_ratio, args.dvs_sample_ratio)
    elif args.data_set == 'ImageNet100':
        train_loader, dvs_val_loader, dvs_test_loader_list = get_tl_imagenet100(args.batch_size, args.RGB_sample_ratio,
                                                                                args.dvs_sample_ratio, args.seed,
                                                                                args.num_classes)

    print("--- Data Loading Summary ---")
    if args.data_set == 'CIFAR10':
        # The new DVSAlignedTLCIFAR10 dataset provides a single, unified training loader
        print(f"Aligned Training Set: {len(train_loader.dataset)} samples")
    else:
        print(f"Training Set: {len(train_loader.dataset)} samples")
    if args.data_set == 'ImageNet100':
        print(f"Validation Set: {len(dvs_val_loader.dataset)} samples")
        print(f"Test Sets: {[len(loader.dataset) for loader in dvs_test_loader_list]} samples")
    else:
        print(f"Validation Set: {len(dvs_test_loader.dataset)} samples")
        print(f"Test Set: {len(dvs_test_loader.dataset)} samples")
    print("--------------------------")

    # preparing model
    model = ViTSNN(cls_num=args.num_classes, img_size_h=128, img_size_w=128, patch_size=16, embed_dims=256,
                   num_heads=16, depths=2, T=args.T)
    if args.parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # preparing training set
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                    nesterov=False)
    else:
        raise Exception(f"The value of optim should in ['SGD', 'Adam'], and your input is {args.optim}")

    if args.data_set == 'CIFAR10':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif args.data_set == 'CINIC10_WO_CIFAR10':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.data_set == 'ImageNet2Caltech':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.data_set == 'Caltech101':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.data_set == 'MNIST':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.data_set == 'ImageNet100':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        raise Exception(
            f"The value of data_set should in ['CIFAR10', 'CINIC10_WO_CIFAR10', 'Caltech101', 'MNIST', 'ImageNet100'], and your input is {args.data_set}")

    # train
    trainer = ViTAlignmentTLTrainer(model=model, optimizer=optimizer, device=device, max_epoch=args.epoch, cfg=args, writer=writer, model_path=model_path,
                                    sps_cka_loss_weight=args.sps_weight, 
                                    block_cka_loss_weight=args.block_weight, 
                                    attn_cka_loss_weight=args.attn_weight, 
                                    loss_function=torch.nn.CrossEntropyLoss())
    if args.data_set == 'ImageNet100':
        best_train_acc, best_val_acc = trainer.train(train_loader, dvs_val_loader)
        test_loss, test_acc1, test_acc5 = trainer.test(dvs_test_loader_list)
    else:
        best_train_acc, best_val_acc = trainer.train(train_loader, dvs_test_loader)
        test_loss, test_acc1, test_acc5 = trainer.test(dvs_test_loader)

    if type(test_loss) is list:
        all_test_loss = test_loss
        test_loss = sum(test_loss) / len(test_loss)
        all_test_acc1 = test_acc1
        test_acc1 = sum(test_acc1) / len(test_acc1)
        all_test_acc5 = test_acc5
        test_acc5 = sum(test_acc5) / len(test_acc5)
        print(
            f'test_loss={test_loss:.5f} test_acc1={test_acc1:.3f} all_test_acc1={all_test_acc1} val_acc5={test_acc5:.4f} all_val_acc5={all_test_acc5}')
        writer.add_scalar(tag="test/accuracy1", scalar_value=test_acc1, global_step=0)
        writer.add_scalar(tag="test/accuracy5", scalar_value=test_acc5, global_step=0)
        writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)
        for test_id in range(len(all_test_loss)):
            writer.add_scalar(tag=f"test{test_id + 1}/accuracy1", scalar_value=all_test_acc1[test_id], global_step=0)
            writer.add_scalar(tag=f"test{test_id + 1}/accuracy5", scalar_value=all_test_acc5[test_id], global_step=0)
            writer.add_scalar(tag=f"test{test_id + 1}/loss", scalar_value=all_test_loss[test_id], global_step=0)
    else:
        print(f'test_loss={test_loss:.5f} test_acc1={test_acc1:.3f} val_acc5={test_acc5:.4f}')
        writer.add_scalar(tag="test/accuracy1", scalar_value=test_acc1, global_step=0)
        writer.add_scalar(tag="test/accuracy5", scalar_value=test_acc5, global_step=0)
        writer.add_scalar(tag="test/loss", scalar_value=test_loss, global_step=0)

        write_content = (
            f'seed: {args.seed}'
            f'sps_weight: {args.sps_weight}, block_weight: {args.block_weight}, attn_weight: {args.attn_weight}'
            f'best_train_acc: {best_train_acc:.4f}, best_val_acc: {best_val_acc:.4f}'
        )
        f.write(write_content)
        f.close()
