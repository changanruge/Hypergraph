import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc
import config

def topk_loss(s, ratio):
    """TopK正则化损失"""
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values
    res = -torch.log(s[:, -int(s.size(1)*ratio):] + config.EPS).mean() - torch.log(1 - s[:, :int(s.size(1)*ratio)] + config.EPS).mean()
    return res

def consist_loss(s):
    """一致性损失"""
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0], s.shape[0])
    D = torch.eye(s.shape[0]) * torch.sum(W, dim=1)
    L = D - W
    L = L.to(config.device)
    res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
    return res

def evaluation_metrics(pred, proba, label):
    """计算TP, TN, FN, FP"""
    TP, TN, FN, FP = 0, 0, 0, 0
    predc = pred.cpu().numpy()
    labelc = label.cpu().numpy()
    TP += ((predc == 1) & (labelc == 1)).sum().item()
    TN += ((predc == 0) & (labelc == 0)).sum().item()
    FN += ((predc == 0) & (labelc == 1)).sum().item()
    FP += ((predc == 1) & (labelc == 0)).sum().item()
    return TP, TN, FN, FP

def train(trainloader, valloader, model, optimizer, save_path):
    """
    训练函数 - 已修复数据泄露问题
    使用valloader进行模型选择，而不是testloader
    """
    min_loss = 1e10
    max_acc = 0
    patience_cnt = 0
    best_epoch = 0
    epochs = config.NUM_EPOCHS
    patience = config.PATIENCE

    t = time.time()
    model.train()

    for epoch in range(epochs):
        loss_train = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in trainloader:
            x_batch = x_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            model.train()

            out = model(x_batch)
            out = out.to(config.device)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, pred = torch.max(out, 1)
            total += y_batch.size(0)
            correct += (pred == y_batch).sum().item()

        acc_train = correct / total
        
        # 【修复数据泄露】使用验证集进行模型选择，而不是测试集
        acc_val, loss_val, _, sen_val, spe_val, _, _, _, _ = test(valloader, model, test=True)
        
        print('\r', 'Epoch: {:06d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'sen_val: {:.6f}'.format(sen_val),
              'spe_val: {:.6f}'.format(spe_val), 'time: {:.6f}s'.format(time.time() - t), flush=True, end='')

        if epoch < 10:
            continue
        
        # 【修复数据泄露】基于验证集性能选择最佳模型
        if acc_val > max_acc and abs(sen_val - spe_val) < 0.25:
            model_state = {'net': model.state_dict()}
            torch.save(model_state, os.path.join(save_path, '{}.pth'.format(epoch)))
            min_loss = loss_val
            max_acc = acc_val
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == patience:
            break

        # 清理旧模型
        files = [f for f in os.listdir(save_path) if f.endswith('.pth')]
        for f in files:
            if f.startswith('fold'):
                continue
            epoch_nb = int(f.split('.')[0])
            if epoch_nb != best_epoch:
                os.remove(os.path.join(save_path, f))
        torch.cuda.empty_cache()
    
    print('\nOptimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))
    return best_epoch

def test(loader, model, test=True):
    """测试函数"""
    model.eval()
    correct = 0.0
    loss_test = 0.0
    ACC, SEN, SPE, PPV, NPV, F1, AUC = 0, 0, 0, 0, 0, 0, 0
    TPS, TNS, FNS, FPS = 0, 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    total = 0
    labels_list = []
    probs_list = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            
            out = model(x_batch)
            out = out.to(config.device)
            labels_list.append(y_batch.cpu().numpy())
            
            if test:
                _, pred = torch.max(out, 1)
                total += y_batch.size(0)
                correct += (pred == y_batch).sum().item()
                proba = torch.sigmoid(out.detach()).cpu().numpy()[:]
                probs = proba[:, 1]
                probs_list.append(probs)
                TP, TN, FN, FP = evaluation_metrics(pred, proba, y_batch)
                TPS += TP
                TNS += TN
                FPS += FP
                FNS += FN
                loss_test += criterion(out, y_batch).item()
            else:
                _, pred = torch.max(out, 1)
                total += y_batch.size(0)
                correct += (pred == y_batch).sum().item()
                loss_test += criterion(out, y_batch).item()
        
        if test:
            labels_concat = np.concatenate(labels_list)
            probs_concat = np.concatenate(probs_list)
            fpr, tpr, threholds = roc_curve(labels_concat, probs_concat)
            
            ACC = (TPS + TNS) / (TPS + TNS + FPS + FNS) if (TPS + TNS + FPS + FNS) > 0 else 0
            SEN = TPS / (TPS + FNS) if (TPS + FNS) > 0 else 0
            SPE = TNS / (TNS + FPS) if (TNS + FPS) > 0 else 0
            PPV = TPS / (TPS + FPS) if (TPS + FPS) > 0 else 0
            NPV = TNS / (TNS + FNS) if (TNS + FNS) > 0 else 0
            F1 = 2 * (ACC * SEN) / (ACC + SEN) if (ACC + SEN) > 0 else 0
            AUC = auc(fpr, tpr)

    torch.cuda.empty_cache()
    return correct / total, loss_test, ACC, SEN, SPE, PPV, NPV, F1, AUC
