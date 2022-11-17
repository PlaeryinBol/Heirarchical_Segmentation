import time
import os
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import config


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=len(config.CLASSES),
         top_level=config.TOP_LEVEL, middle_level=config.MIDDLE_LEVEL, bottom_level=config.BOTTOM_LEVEL):
    """Calculate different levels mIoU scores between predicted mask and gt mask."""
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_classes = []
        iou_top_level = []
        iou_middle_level = []
        iou_bottom_level = []
        # loop per pixel class
        for clas in range(0, n_classes):
            true_class = pred_mask == clas
            true_label = mask == clas
            # no exist label in this loop
            if true_label.long().sum().item() == 0:
                iou_per_classes.append(np.nan)
                iou_top_level.append(np.nan)
                iou_middle_level.append(np.nan)
                iou_bottom_level.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_classes.append(iou)
                if clas in top_level:
                    iou_top_level.append(iou)
                if clas in middle_level:
                    iou_middle_level.append(iou)
                if clas in bottom_level:
                    iou_bottom_level.append(iou)

        result_per_classes = np.nanmean(iou_per_classes)
        result_top_level = np.nanmean(iou_top_level)
        result_middle_level = np.nanmean(iou_middle_level)
        result_bottom_level = np.nanmean(iou_bottom_level)
        return result_per_classes, result_top_level, result_middle_level, result_bottom_level


def pixel_accuracy(output, mask):
    """Calculate pixel accuracy score between predicted mask and gt mask."""
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def get_lr(optimizer):
    """Get current optimiser learning rate."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(epochs, model, device, train_loader, val_loader, criterion, optimizer, scheduler, exp_name):
    """Training and validation stage, saving history."""
    if not os.path.exists('./models'):
        os.mkdir('./models')
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0

        model.train()
        # training loop
        for i, data in enumerate(tqdm(train_loader)):
            image_tiles, mask_tiles = data
            image = image_tiles.to(device)
            mask = mask_tiles.to(device)

            output = model(image)
            loss = criterion(output, mask)

            iou_score += mIoU(output, mask)[0]
            accuracy += pixel_accuracy(output, mask)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            scheduler.step()
            running_loss += loss.item()
        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image_tiles, mask_tiles = data
                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)

                    val_iou_score += mIoU(output, mask)[0]
                    test_accuracy += pixel_accuracy(output, mask)

                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculation mean for each batch
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                decrease += 1
                print('saving model...')
                torch.save(model, os.path.join(config.MODELS_DIR,
                                               '{}_{}_epoch_mIoU-{:.3f}.pt'.format(exp_name, e,
                                                                                   val_iou_score/len(val_loader))))

            if (test_loss/len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss/len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == config.NOT_IMPROVE:
                    print(f'Loss not decrease for {config.NOT_IMPROVE} times, Stop Training')
                    break

            # iou
            val_iou.append(val_iou_score/len(val_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy/len(val_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score/len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score/len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


def plot_loss(history):
    """Plot loss diagram."""
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_miou(history):
    """Plot mIoU diagram."""
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU',  marker='*')
    plt.title('mIoU score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_accuracy(history):
    """Plot plot_accuracy diagram."""
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy',  marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def predict_image_mask_miou(model, device, image, mask, mean=config.MEAN, std=config.STD):
    """Get different levels mIoU scores for single image."""
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        output = model(image)
        score, top_score, mid_score, bot_score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score, top_score, mid_score, bot_score


def predict_image_mask_pixel(model, device, image, mask, mean=config.MEAN, std=config.STD):
    """Get pixel accuracy score for single image."""
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


def miou_score(model, device, test_set):
    """Get different levels mIoU scores for all test set."""
    score_iou = []
    top_score_iou = []
    middle_score_iou = []
    bottom_score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        _, score, top_score, mid_score, bot_score = predict_image_mask_miou(model, device, img, mask)
        score_iou.append(score)
        top_score_iou.append(top_score)
        middle_score_iou.append(mid_score)
        bottom_score_iou.append(bot_score)
    mean_scores = (np.nanmean(score_iou), np.nanmean(top_score_iou),
                   np.nanmean(middle_score_iou), np.nanmean(bottom_score_iou))
    return mean_scores


def accuracy_score(model, device, test_set):
    """Get pixel accuracy score for all test set."""
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        _, acc = predict_image_mask_pixel(model, device, img, mask)
        accuracy.append(acc)
    return np.mean(accuracy)
