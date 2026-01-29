import os
import torch
import torch_scatter
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

import config
from data_utils import load_labels, match_data, CustomDataset
from hypergraph_model import HypergraphModel
from train_utils import train, test

def main():
    print("Torch version:", torch.__version__)
    print("Torch Scatter version:", torch_scatter.__version__)
    print(f"Device: {config.device}")
    print("=" * 60)
    print("超图神经网络 (HyperGraph Neural Network)")
    print("数据泄露已修复：使用验证集进行模型选择")
    print("=" * 60)

    # 加载标签并匹配数据
    labels = load_labels(config.LABEL_FILE)
    mat_files = []
    
    if not os.path.exists(config.DATA_DIR):
        print(f"错误：数据目录不存在: {config.DATA_DIR}")
        return
    
    for root, dirs, files in os.walk(config.DATA_DIR):
        for file in files:
            if file.lower().endswith('.mat'):
                mat_files.append(os.path.join(root, file))

    matched_data = match_data(mat_files, labels)
    matched_files = list(matched_data.keys())
    matched_labels = [matched_data[f]['label'] for f in matched_files]

    if len(matched_files) == 0:
        print("错误：没有成功匹配任何样本！")
        return

    print(f"成功匹配 {len(matched_files)} 个样本")

    # 10折交叉验证
    skf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    ACCS, SENS, SPES, PPVS, NPVS, F1S, AUCS = [], [], [], [], [], [], []
    results_list = []  # 修复pandas concat警告
    columns = ['fold', 'ACC', 'SEN', 'SPE', 'PPV', 'NPV', 'F1', 'AUC']

    for fold, (train_val_indices, test_indices) in enumerate(skf.split(matched_files, matched_labels)):
        model = HypergraphModel(
            seq_len=config.INDIM,
            enc_in=config.INDIM,
            num_class=config.NCLASS,
            d_model=config.D_MODEL,
            dropout=config.DROPOUT
        ).to(config.device)
        
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{config.N_FOLDS}")
        print(f"{'='*50}")
        
        train_files = [matched_files[i] for i in train_val_indices]
        train_labels_split = [matched_labels[i] for i in train_val_indices]
        train_files, val_files = train_test_split(
            train_files, test_size=1/9, random_state=config.RANDOM_SEED, stratify=train_labels_split
        )
        test_files = [matched_files[i] for i in test_indices]

        train_dataset = CustomDataset(train_files, matched_data)
        val_dataset = CustomDataset(val_files, matched_data)
        test_dataset = CustomDataset(test_files, matched_data)

        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        work_path = os.path.join(config.MODEL_SAVE_DIR, 'models_all_sites')
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        
        best_model = train(train_dataloader, val_dataloader, model, optimizer, work_path)

        checkpoint = torch.load(os.path.join(work_path, '{}.pth'.format(best_model)))
        model.load_state_dict(checkpoint['net'])
        test_acc, test_loss, ACC, SEN, SPE, PPV, NPV, F1, AUC = test(test_dataloader, model, test=True)

        # 修复pandas concat警告：使用列表append
        results_list.append([fold+1, ACC*100, SEN*100, SPE*100, PPV*100, NPV*100, F1*100, AUC*100])

        print(f'\nModel {fold + 1} fold test set results:')
        print(f'Loss: {test_loss:.6f}, Accuracy: {test_acc:.6f}')
        print(f'ACC: {ACC*100:.2f}%, SEN: {SEN*100:.2f}%, SPE: {SPE*100:.2f}%, PPV: {PPV*100:.2f}%, NPV: {NPV*100:.2f}%, F1: {F1*100:.2f}%, AUC: {AUC*100:.2f}%')

        ACCS.append(ACC), SENS.append(SEN), SPES.append(SPE), PPVS.append(PPV), NPVS.append(NPV), F1S.append(F1), AUCS.append(AUC)

        state = {'net': model.state_dict()}
        save_path = os.path.join(work_path, 'fold_{:d}_test_acc_{:.6f}_epoch_{:d}_.pth'.format(fold + 1, test_acc, best_model))
        torch.save(state, save_path)

    # 在循环结束后创建DataFrame（修复pandas警告）
    results_s = pd.DataFrame(results_list, columns=columns)
    
    # 打印最终结果
    print(f"\n{'='*80}")
    print("超图神经网络 - 10-Fold Cross Validation Results")
    print(f"{'='*80}")
    print(f'ACC: {np.mean(ACCS)*100:.2f}±{np.std(ACCS)*100:.2f}%')
    print(f'SEN: {np.mean(SENS)*100:.2f}±{np.std(SENS)*100:.2f}%')
    print(f'SPE: {np.mean(SPES)*100:.2f}±{np.std(SPES)*100:.2f}%')
    print(f'PPV: {np.mean(PPVS)*100:.2f}±{np.std(PPVS)*100:.2f}%')
    print(f'NPV: {np.mean(NPVS)*100:.2f}±{np.std(NPVS)*100:.2f}%')
    print(f'F1:  {np.mean(F1S)*100:.2f}±{np.std(F1S)*100:.2f}%')
    print(f'AUC: {np.mean(AUCS)*100:.2f}±{np.std(AUCS)*100:.2f}%')
    print(f"{'='*80}")
    
    # 保存结果
    with open('hypergraph_train_result.txt', 'w') as f:
        for index, row in results_s.iterrows():
            f.write(f"Fold {int(row['fold'])}: ACC={row['ACC']:.2f}, SEN={row['SEN']:.2f}, SPE={row['SPE']:.2f}, "
                    f"PPV={row['PPV']:.2f}, NPV={row['NPV']:.2f}, F1={row['F1']:.2f}, AUC={row['AUC']:.2f}\n")
    results_s.to_csv('./hypergraph_metrics.csv')
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
