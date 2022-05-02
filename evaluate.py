# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
#
# from utils.dice_score import multiclass_dice_coeff, dice_coeff
# from utils.ac import accuracy, multiclass_accuracy, IOU, multiclass_IOU
#
# def evaluate(net, dataloader, device):
#     net.eval()
#     num_val_batches = len(dataloader)
#     IOU_score = 0
#     accuracy_test = 0
#     # iterate over the validation set
#     for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#         image, mask_true = batch['image'], batch['mask']
#         # move images and labels to correct device and type
#         image = image.to(device=device, dtype=torch.float32)
#         mask_true = mask_true.to(device=device, dtype=torch.long)
#         mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
#
#         with torch.no_grad():
#             # predict the mask
#             mask_pred = net(image)
#             if net.n_classes == 1:
#                 mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#                 accuracy_test += accuracy(mask_pred, mask_true)
#                 IOU_score +=IOU(mask_pred, mask_true, reduce_batch_first=False)
#                 # IOU_score +=IOU(mask_pred, mask_true)
#             else:
#                 mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#                 accuracy_test += multiclass_accuracy(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])
#                 IOU_score += multiclass_IOU(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
#                 # IOU_score += multiclass_IOU(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])
#             # if net.n_classes == 1:
#             #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#             #
#             #     dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
#             # else:
#             #     mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#             #
#             #     dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
#
#
#
#
#     net.train()
#
#     # Fixes a potential division by zero error
#     if num_val_batches == 0:
#         return accuracy_test, IOU_score
#     return accuracy_test / num_val_batches, IOU_score / num_val_batches
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches