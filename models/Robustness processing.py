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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler


# 数据选择器
data_selector = VarianceThreshold(threshold=0.0)
features_selector = VarianceThreshold(threshold=0.000)
data = sio.loadmat('F:\deboke桌面\最终版代码\datas\GBM.mat')

# Normalized function
def normalize(data):
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min)
# Normalize each column
def nor_data(data):
    n_data = []
    for line in range(data.shape[1]):
        l_data = data[:, line]
        n_data.append(normalize(l_data))
    return np.array(n_data).T

# Data enhancement method
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


def number_to_excel_column(num):
    letters = ""
    while num > 0:
        num, remainder = divmod(num - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters



# 加载数据和标签
labels = data['GBM_clinicalMatrix'].squeeze()
data_1 = np.array(data['GBM_Gene_Expression']).T
data_2 = np.array(data['GBM_Methy_Expression']).T
data_3 = np.array(data['GBM_Mirna_Expression']).T
#
# data_1 = pd.read_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\BRCA数据集\IQR异常值\\'+'data1.csv', header=None)
# data_1 = np.array(data_1)
# data_2 = pd.read_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\BRCA数据集\IQR异常值\\'+'data2.csv', header=None)
# data_2 = np.array(data_2)
# data_3 = pd.read_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\BRCA数据集\IQR异常值\\'+'data3.csv', header=None)
# data_3 = np.array(data_3)



'''
#View basic information about the data set：
df = pd.DataFrame(data_1)
print(df.describe())

#Presentation data set：
#The K-means clustering algorithm was used to visualize the data
scarer = StandardScaler()
data = scarer.fit_transform(data_1)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
labels = kmeans.fit_predict(data)
cluster_centers = kmeans.cluster_centers_
plt.figure(figsize=(8, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i in range(n_clusters):
    cluster_points = data_1[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='black', s=100, label='Cluster Centers')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
'''

''''
#Clear zero and one feature
#Look at the first few lines of the dataset to make sure the data is loaded correctly
df = pd.DataFrame(data_1)
print(df.head())
print("---------------------------------------------------------------------")
print(df.info())
print("---------------------------------------------------------------------")
missing_values = df.isnull().sum()
print("Missing value statistics:\n", missing_values)
print("---------------------------------------------------------------------")
print(df)
print(df.shape)
zero_counts = df.eq(0).sum()
# Print out the result
zero = np.array(zero_counts)
labels = np.array(labels)
zero_list = []
list = []
j = 0
for i in range(df.shape[1]):
    if zero[i] >= 10:
        j= j+1
        zero_list.append(zero[i])
        list.append(i)
print("The number of 0 values in each column:")
print(zero_list)
print(list)
print("---------------------------------------------------------------------")
# 删除列：
# df = df.drop(df.columns[list], axis=1)
print(df.shape)
pd.DataFrame(df).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\GBM数据集\\'+'data_1.csv',index=None,header=None)
'''


'''
# Check for duplicate samples
df = pd.DataFrame(data)
duplicate_samples = df.duplicated()
# Output whether there are duplicate samples
if duplicate_samples.any():
    print("There are duplicate samples in the data set:")

    print("Duplicate line：")
    print(df[duplicate_samples])
else:
    print("There are no duplicate samples in the dataset:")


#The quartile identifies and modifies outliers'''

'''
# IQR
df = pd.DataFrame(data_3)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
threshold = 1.5
outliers = (df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))
print(outliers)
#Count the number of outliers
outlier_counts = outliers.sum()
print("Number of outliers per column:")
print(outlier_counts)
total_outliers = outliers.values.sum()
print("\nTotal number of outliers:", total_outliers)

# Modify outliers to boundary values
for column in df.columns:
    lower_bound = Q1[column] - threshold * IQR[column]
    upper_bound = Q3[column] + threshold * IQR[column]

    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
# 打印修改后的DataFrame
print("The processed data set：")
print(df)
pd.DataFrame(df).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\GBM数据集\\'+'data_3.csv',index=None,header=None)
'''

'''
#Find the correlation coefficient between the features
scarer = StandardScaler()
df = scarer.fit_transform(data)
df = pd.DataFrame(df)
print(df)

labelss = pd.DataFrame(labels)
df['Labels'] = labels

grouped = df.groupby('Labels')
for labels, group in grouped:
    print(f"Labels {labels}:")
    print(group)
    print()

correlations_with_target = df.corr()['Labels'].drop('Labels')
print(correlations_with_target)
print("---------------------------------------------------------------------")
threshold = 0.4
strong_correlations = correlations_with_target[abs(correlations_with_target) >= threshold]
print("Strongly correlated features：")
print(strong_correlations)
print("The number is：", len(strong_correlations))
# correlations_with_target = np.array(correlations_with_target)
# pd.DataFrame(correlations_with_target).to_csv('F:\deboke桌面\基于深度学习的癌症亚型分类\研究-代码\数据集训练\健壮性测试\BRCA600\\'+'data1相关性.csv',index=None,header=None)
'''












