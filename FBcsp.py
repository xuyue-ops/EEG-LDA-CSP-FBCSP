import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP
from sklearn.feature_selection import SelectKBest,mutual_info_classif

# 读取数据
file_path = "data/A01T.gdf"
raw = mne.io.read_raw_gdf(file_path, preload=True)
raw.pick_types(eeg=True)


#提取事件
events,event_id = mne.events_from_annotations(raw)
event_id = dict(left=7,right=8)

#定义多个频段
frequency_bands = [
    (4,8),   # theta
    (8,12),  # mu节律，运动想象相关的主要频率
    (12,16),
    (16,20),
    (20,24),
    (24,28),
    (28,32)
]
print(f"定义了{len(frequency_bands)}个频段")
print(frequency_bands)

tmin = 0.5
tmax = 2.5
all_band_features = [] #创建一个空列表，用于存储多频段的特征
y_labels = None  # 初始化标签变量，后面会赋值真正的标签
raw.original = raw.copy() # 保存原始数据的副本，以便后续使用

for band in frequency_bands:
    print(f"正在处理频段: {band[0]}-{band[1]} Hz")
    # 复制原始数据，避免修改原始数据
    raw_band = raw.original.copy()
    # 滤波
    raw_band.filter(band[0], band[1], fir_design='firwin', skip_by_annotation='edge')
    # 分段
    epochs_band = mne.Epochs(raw_band, events, event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # 提取数据和标签
    X_band = epochs_band.get_data()
    y_band = epochs_band.events[:, -1]
    y_band = np.where(y_band == 7, 0, 1)  # 将标签转换为二分类
   
    if y_labels is None:
        y_labels = y_band  # 只需要设置一次标签
    csp = CSP(n_components=6, reg='ledoit_wolf', log=True)
    X_csp = csp.fit_transform(X_band, y_band)
    all_band_features.append(X_csp)  # 将当前频段的特征添加到列表中
    print(f"频段 {band[0]}-{band[1]} Hz 的CSP特征 shape: {X_csp.shape}")

# 将所有频段的特征进行拼接
X_fbcsp = np.hstack(all_band_features)
print(f"所有频段的特征拼接后的 shape: {X_fbcsp.shape}")
print(f"标签 shape: {y_labels.shape}")
print(f"样本数: {X_fbcsp.shape[0]}, 特征数: {X_fbcsp.shape[1]}")

#特征选择
selector = SelectKBest(mutual_info_classif, k=20)
X_selected = selector.fit_transform(X_fbcsp, y_labels)
print(f"特征选择后形状: {X_selected.shape}")

# ========== 9. 分类评估 ==========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 9.1 使用全部特征
clf_full = Pipeline([
    ('lda', LinearDiscriminantAnalysis())
])
scores_full = cross_val_score(clf_full, X_fbcsp, y_labels, cv=cv)

print(f"\n{'='*50}")
print("FBCSP结果（全部特征）")
print(f"{'='*50}")
print(f"准确率: {np.mean(scores_full):.4f} ± {np.std(scores_full):.4f}")
print(f"各折: {np.round(scores_full, 4)}")

# 9.2 使用特征选择后的特征
clf_selected = Pipeline([
    ('lda', LinearDiscriminantAnalysis())
])
scores_selected = cross_val_score(clf_selected, X_selected, y_labels, cv=cv)

print(f"\n{'='*50}")
print("FBCSP结果（特征选择后）")
print(f"{'='*50}")
print(f"准确率: {np.mean(scores_selected):.4f} ± {np.std(scores_selected):.4f}")
print(f"各折: {np.round(scores_selected, 4)}")

# ========== 10. 可选：对比原始单频段方法 ==========
print(f"\n{'='*50}")
print("对比：原始单频段CSP（8-30Hz）")
print(f"{'='*50}")
print(f"当前 CSP n_components: {CSP(n_components=6, reg='ledoit_wolf', log=True).n_components}")