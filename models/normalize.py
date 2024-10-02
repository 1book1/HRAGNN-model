import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from scipy.stats import mode

# 数据选择器
data_selector = VarianceThreshold(threshold=0.0)
features_selector = VarianceThreshold(threshold=0.000)
data = sio.loadmat('F:\deboke桌面\最终版代码\datas\GBM.mat')

def normalize(data):
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min)

def l2(data):
    data = data - data.mean(axis=0, keepdims=True)
    data = data / np.sqrt(data.var(axis=0, keepdims=True))
    return data

def nor_data(data):
    n_data = []
    for line in range(data.shape[1]):
        l_data = data[:, line]
        n_data.append(normalize(l_data))
    return np.array(n_data).T

def add_gaussian_noise(features, noise_level=0.01):
    noise = np.random.normal(0, noise_level, features.shape)
    return features + noise

def scale_features(features, scale_factor=1.2):
    return features * scale_factor

def shift_features(features, shift_value=0.1):
    return features + shift_value

def jitter_features(features, jitter_level=0.02):
    jitter = np.random.uniform(-jitter_level, jitter_level, features.shape)
    return features + jitter

def data_augmentation(features, labels):
    noisy_fea = add_gaussian_noise(features, noise_level=0.05)
    scaled_fea = scale_features(features, scale_factor=1.1)
    shifted_fea = shift_features(features, shift_value=0.05)
    jittered_fea = jitter_features(features, jitter_level=0.03)

    augmented_features = np.concatenate([features, noisy_fea, scaled_fea, shifted_fea, jittered_fea], axis=0)
    augmented_labels = np.concatenate([labels] * 5)

    return augmented_features, augmented_labels

def lasso_feature_selection(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = Lasso()
    params = {'alpha': np.logspace(-4, -2, 50)}
    grid_search = GridSearchCV(lasso, params, cv=5, scoring='neg_mean_squared_error')
    print(X_scaled.shape)
    print(y.shape)
    grid_search.fit(X_scaled, y)

    best_lasso = grid_search.best_estimator_
    mask = best_lasso.coef_ != 0
    selected_features = X[:, mask]

    print(f"Number of selected features: {selected_features.shape[1]}")

    y_pred_train = best_lasso.predict(X_scaled)
    mse_train = mean_squared_error(y, y_pred_train)
    print(f"Training MSE: {mse_train}")

    return selected_features, best_lasso

# 加载数据和标签
labels = data['GBM_clinicalMatrix'].squeeze()
# data_1 = np.array(data['BRCA_Gene_Expression']).T
# data_2 = np.array(data['BRCA_Methy_Expression']).T
# data_3 = np.array(data['BRCA_Mirna_Expression']).T

# labels = np.loadtxt('F:\deboke桌面\基于深度学习的癌症亚型分类\数据集\多组学1\Breast\BRCA_clinicalMatrix.txt')
# labels = np.array(labels).astype(np.float32)
data_1 = pd.read_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\GBM数据集\\'+'data_1.csv', header=None)
data_1 = np.array(data_1).astype(np.float32)
print(data_1.shape)
data_2 = pd.read_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\GBM数据集\\'+'data_2.csv', header=None)
data_2 = np.array(data_2).astype(np.float32)
data_3 = pd.read_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\GBM数据集\\'+'data_3.csv', header=None)
data_3 = np.array(data_3).astype(np.float32)

# 数据选择和归一化
# data_1 = features_selector.fit_transform(nor_data(data_selector.fit_transform(data_1)))
# data_2 = features_selector.fit_transform(nor_data(data_selector.fit_transform(data_2)))
# data_3 = features_selector.fit_transform(nor_data(data_selector.fit_transform(data_3)))
# data_1 = nor_data(data_1)
# data_2 = nor_data(data_2)
# data_3 = nor_data(data_3)

#l2
# data_1 = features_selector.fit_transform(l2(data_selector.fit_transform(data_1)))
# data_2 = features_selector.fit_transform(l2(data_selector.fit_transform(data_2)))
# data_3 = features_selector.fit_transform(l2(data_selector.fit_transform(data_3)))


# 使用Lasso进行特征选择
# data_1, best_lasso_1 = lasso_feature_selection(data_1, labels)
# data_2, best_lasso_2 = lasso_feature_selection(data_2, labels)
# data_3, best_lasso_3 = lasso_feature_selection(data_3, labels)

scarer = StandardScaler()
data_1 = scarer.fit_transform(data_1)
data_2 = scarer.fit_transform(data_2)
data_3 = scarer.fit_transform(data_3)

# 对训练集进行数据增强
# data_1tr_aug, labels_1tr_aug = data_augmentation(data_1tr, labels_tr)
# data_2tr_aug, labels_2tr_aug = data_augmentation(data_2tr, labels_tr)
# data_3tr_aug, labels_3tr_aug = data_augmentation(data_3tr, labels_tr)

# 训练集和测试集划分
data_1tr, data_1te, labels_tr, labels_te = train_test_split(data_1, labels, test_size=0.2, random_state=45)
data_2tr, data_2te, labels_tr, labels_te = train_test_split(data_2, labels, test_size=0.2, random_state=45)
data_3tr, data_3te, labels_tr, labels_te = train_test_split(data_3, labels, test_size=0.2, random_state=45)




pd.DataFrame(data_1tr).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\BRCA-Myself\\'+'1_tr_n.csv',index=None,header=None)
pd.DataFrame(data_1te).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\BRCA-Myself\\'+'1_te_n.csv',index=None,header=None)
pd.DataFrame(data_2tr).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\BRCA-Myself\\'+'2_tr_n.csv',index=None,header=None)
pd.DataFrame(data_2te).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\BRCA-Myself\\'+'2_te_n.csv',index=None,header=None)
pd.DataFrame(data_3tr).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\BRCA-Myself\\'+'3_tr_n.csv',index=None,header=None)
pd.DataFrame(data_3te).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\BRCA-Myself\\'+'3_te_n.csv',index=None,header=None)
pd.DataFrame(labels_tr).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\BRCA-Myself\\'+'labels_tr.csv',index=None,header=None)
pd.DataFrame(labels_te).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\BRCA-Myself\\'+'labels_te.csv',index=None,header=None)


# pd.DataFrame(data_1tr).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\GBM-Myself\\'+'1_tr_n.csv',index=None,header=None)
# pd.DataFrame(data_1te).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\GBM-Myself\\'+'1_te_n.csv',index=None,header=None)
# pd.DataFrame(data_2tr).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\GBM-Myself\\'+'2_tr_n.csv',index=None,header=None)
# pd.DataFrame(data_2te).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\GBM-Myself\\'+'2_te_n.csv',index=None,header=None)
# pd.DataFrame(data_3tr).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\GBM-Myself\\'+'3_tr_n.csv',index=None,header=None)
# pd.DataFrame(data_3te).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\GBM-Myself\\'+'3_te_n.csv',index=None,header=None)
# pd.DataFrame(labels_tr).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\GBM-Myself\\'+'labels_tr.csv',index=None,header=None)
# pd.DataFrame(labels_te).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\GBM-Myself\\'+'labels_te.csv',index=None,header=None)
