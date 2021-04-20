import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse, time, sys, os, cv2
import numpy as np
from dataloader import AlzhDataset
import tensorboard_logger as tb_logger
from PIL import Image
from utils import AverageMeter, accuracy, adjust_learning_rate
from network.resnet import SupConResNet
from network.custom import Custom_CNN, Linear_cls
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, classification_report
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default=[60,80],
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--class_type', type=str, default='AD_CN', choices=['AD_CN', 'MCI_CN', 'AD_MCI', '3class'])
    parser.add_argument('--pretrained_model', type=str, default='./save_models/SimCLR_pretrained.pth')
    parser.add_argument('--dataset_path', type=str, default='/data/tm/alzh/data_PGGAN')
    opt = parser.parse_args()

    opt.tb_path = './logs'
    opt.model_path = './save_models'
    opt.tb_folder = os.path.join(opt.tb_path, time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '-' + opt.class_type)
    opt.model_folder = os.path.join(opt.model_path, time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '-' + opt.class_type)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)
    return opt

def set_loader(opt):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    train_dataset = AlzhDataset(type=opt.class_type, root=os.path.join(opt.dataset_path, 'train'), transform=transform_train)
    valid_dataset = AlzhDataset(type=opt.class_type, root=os.path.join(opt.dataset_path, 'validation'), transform=transform_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, opt.batch_size, num_workers=0, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, opt.batch_size, num_workers=0, shuffle=False, drop_last=False)

    return train_loader, valid_loader

def set_model(opt):
    model = SupConResNet(name='resnet18')
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt, model2):
    model.train()
    model2.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (image, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            image = image.cuda()
            labels = labels.cuda()
        bsz = labels.shape[0]
        logits = model.encoder(image)
        logits = model2(logits)
        loss = criterion(logits, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()
    return losses.avg

def validation(val_loader, model, model2):
    model.eval()
    model2.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for i, (image, label) in enumerate(val_loader):
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
            bsz = label.shape[0]
            output = model.encoder(image)
            output = model2(output, softmax=True)

            acc1 = accuracy(output, label)
            top1.update(acc1[0].item(), bsz)
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test(opt, model, model2, root, transform=None):
    model.eval()
    model2.eval()

    if opt.class_type == 'AD_CN':
        type_list = ['AD_0', 'CN_1']
    elif opt.class_type == 'AD_MCI':
        type_list = ['AD_0', 'MCI_1']
    elif opt.class_type == 'MCI_CN':
        type_list = ['MCI_0', 'CN_1']
    elif opt.class_type == '3class':
        type_list = ['AD_0', 'MCI_1', 'CN_2']
    y_true = []
    y_pred = []
    y_pred_label = []
    output_list = []
    num_correct = 0
    num = 0
    with torch.no_grad():
        for types in type_list:
            correct = 0
            total = 0

            path = os.path.join(root, types.split('_')[0])
            for dirname in os.listdir(path):
                new_path = os.path.join(path, dirname)
                for i in range(len(os.listdir(new_path))):
                    img = Image.open(os.path.join(new_path, os.listdir(new_path)[i])).convert('RGB')
                    if transform is not None:
                        img = transform(img)
                    if i == 0:
                        img_concat = img.unsqueeze(0)
                    else:
                        img_concat = torch.cat([img_concat, img.unsqueeze(0)], dim=0)
                label = torch.empty(i + 1)
                class_type = int(types.split('_')[1])
                label.fill_(class_type)
                if torch.cuda.is_available():
                    img_concat = img_concat.cuda()
                    label = label.cuda()

                bsz = label.shape[0]

                output = model.encoder(img_concat)
                output = model2(output, softmax=True)

                acc1 = accuracy(output, label)
                if acc1[0].item() >= 50:
                    correct += 1
                total += 1
                num_correct += bsz * acc1[0].item() / 100
                num += bsz
                y_true = y_true + label.cpu().tolist()
                y_pred = y_pred + output[:, 0].tolist()
                y_pred_label = y_pred_label + torch.argmax(output.cpu(), dim=1).tolist()
            output_list.append([types.split('_')[0], total, correct])
        precision = precision_score(y_true, y_pred_label, pos_label=0)
        recall = recall_score(y_true, y_pred_label, pos_label=0)
        f1 = f1_score(y_true, y_pred_label, pos_label=0)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=0)
        auc_score = auc(fpr, tpr)
    return output_list, 100 * num_correct/num, [precision, recall, f1, auc_score]

def main():
    opt = parse_option()

    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, valid_loader = set_loader(opt)
    model, criterion = set_model(opt)
    if opt.pretrained_model is not '':
        checkpoint = torch.load(opt.pretrained_model)
        model.load_state_dict(checkpoint)
    if opt.class_type == 'AD_CN' or opt.class_type == 'AD_MCI' or opt.class_type == 'MCI_CN':
        model2 = Linear_cls(512, 2)
    else:
        model2 = Linear_cls(512, 3)
    model2 = model2.cuda()


    optimizer = torch.optim.SGD(list(model.parameters()) +list(model2.parameters()),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    valid_best = 0
    for epoch in range(1, opt.epochs + 1):
        lr = adjust_learning_rate(opt, optimizer, epoch)

        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt, model2)
        time2 = time.time()

        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        top1 = validation(valid_loader, model, model2)
        if top1 > valid_best:
            valid_best = top1
            torch.save(model.state_dict(), os.path.join(opt.model_folder, 'best_model.pth'))

        output_list, test_acc, metric = test(opt, model, model2, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]), root=os.path.join(opt.dataset_path, 'test'))
        print('test acc {:.2f} auc {:.2f} precision {:.2f} recall {:.2f} f1 score {:.2f}'.format(test_acc, metric[0], metric[1], metric[2], metric[3]))
        for test_list in output_list:
            print('{0}: {1}/{2}'.format(test_list[0], test_list[2], test_list[1]))
            logger.log_value(test_list[0], test_list[2]/test_list[1], epoch)
        logger.log_value('loss', loss, epoch)
        logger.log_value('valid acc', top1, epoch)
        logger.log_value('test/acc', test_acc, epoch)
        logger.log_value('test/auc', metric[0], epoch)
        logger.log_value('test/precision', metric[1], epoch)
        logger.log_value('test/recall', metric[2], epoch)
        logger.log_value('test/f1 score', metric[3], epoch)


if __name__ == '__main__':
    main()
