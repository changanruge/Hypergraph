import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import config

def preprocess_fc(fc_matrix: np.ndarray) -> np.ndarray:
    """
    FC预处理：Fisher Z变换 + z-score归一化
    """
    fc = np.clip(fc_matrix, -0.999999, 0.999999)
    fc = np.arctanh(fc)
    col_mu = fc.mean(axis=0, keepdims=True)
    col_std = fc.std(axis=0, keepdims=True) + 1e-6
    fc = (fc - col_mu) / col_std
    mu = fc.mean()
    sigma = fc.std() + 1e-6
    fc = (fc - mu) / sigma
    return fc.astype(np.float32)


def load_labels(csv_file):
    """加载标签文件"""
    df = pd.read_csv(csv_file)
    labels = {}

    for _, row in df.iterrows():
        if 'subject' in df.columns and not pd.isna(row['subject']):
            file_id = int(row['subject'])
        elif 'FILE_ID' in df.columns and not pd.isna(row['FILE_ID']) and row['FILE_ID'] != 'no_filename':
            file_id = row['FILE_ID']
        else:
            continue

        dx_group = None
        if 'DX_GROUP' in df.columns and not pd.isna(row['DX_GROUP']):
            dx_group = row['DX_GROUP']
        elif 'DSM_IV_TR' in df.columns and not pd.isna(row['DSM_IV_TR']):
            dx_group = row['DSM_IV_TR']

        if dx_group is None:
            continue

        if dx_group == 1:
            label = 1
        elif dx_group == 2:
            label = 0
        else:
            continue

        site_id = row['SITE_ID'] if 'SITE_ID' in df.columns and not pd.isna(row['SITE_ID']) else 'unknown'

        labels[file_id] = {
            'label': label,
            'file_id': file_id,
            'sub_id': row['SUB_ID'] if 'SUB_ID' in df.columns else None,
            'site_id': site_id
        }

    return labels


def match_data(mat_files, labels):
    """匹配mat文件与标签"""
    matched_files = {}
    for h5_file in mat_files:
        filename = os.path.basename(h5_file)
        match = re.search(r'(\d{6,})', filename)
        if match:
            numeric_id_str = match.group(1)
            numeric_id_int = int(numeric_id_str)
            if numeric_id_int in labels:
                matched_files[h5_file] = labels[numeric_id_int]
                continue
            for label_key, label_info in labels.items():
                if isinstance(label_key, str):
                    label_key_str = str(label_key)
                    label_match = re.search(r'_(\d+)$', label_key_str)
                    if label_match:
                        label_numeric_int = int(label_match.group(1))
                        if numeric_id_int == label_numeric_int:
                            matched_files[h5_file] = label_info
                            break
                    elif numeric_id_str in label_key_str:
                        matched_files[h5_file] = label_info
                        break
    return matched_files


class CustomDataset(Dataset):
    """
    超图数据集，返回FC矩阵作为节点特征
    """
    def __init__(self, file_paths, label_map):
        self.file_paths = file_paths
        self.label_map = label_map

        # 预加载并预处理FC矩阵
        self.fc_matrices = []
        for path in self.file_paths:
            mat = sio.loadmat(path)
            if 'fc' in mat:
                fc_matrix = mat['fc']
            elif 'fc_matrix' in mat:
                fc_matrix = mat['fc_matrix']
            elif 'FC_Matrix' in mat:
                fc_matrix = mat['FC_Matrix']
            else:
                raise KeyError(f"未找到fc矩阵键: {path}")
            fc_matrix = preprocess_fc(fc_matrix)
            self.fc_matrices.append(fc_matrix)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fc_matrix = self.fc_matrices[idx]
        # 将FC矩阵作为节点特征 [seq_len, channels]
        x = torch.tensor(fc_matrix, dtype=torch.float32)
        y = torch.tensor(self.label_map[self.file_paths[idx]]['label'], dtype=torch.long)
        
        return x, y
