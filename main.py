import numpy as np
import matplotlib.pyplot as plt

# 加载数据
def load_data(filename):
  data = np.loadtxt(filename, dtype=np.float32)
  return data

# 计算协方差矩阵
def covariance_matrix(data):
  mean = np.mean(data, axis=0)
  cov_matrix = np.cov(data, rowvar=False)
  return cov_matrix

# 计算协方差阵特征值和特征向量
def eigen_values_vectors(cov_matrix):
  eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
  return eigen_values, eigen_vectors

# K-L变换
def k_l_transform(data, eigen_vectors):
  transformed_data = np.dot(eigen_vectors.T, data.T).T
  return transformed_data

# 绘制散点图
def plot_scatter(data, labels, title):
  # 检查 data 的维度
  if data.ndim == 1:
    # data 是一维数组，绘制散点图
    plt.scatter(data, np.zeros(data.shape[0]), c=labels, cmap='viridis')
    plt.title(title)
    plt.yticks([])
  else:
    # data 是二维数组，绘制散点图
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    #plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    #plt.show()

# 分类
def classify(data, w,w0):
    classlabel = np.zeros(len(data))
    projections = np.dot(data, w)+w0  #g(x)=w.T*x+w0 >0 male=1 <0 female=0
    for i in range(len(data)):
        if projections[i]>0:
            classlabel[i]=1
    return classlabel

# 读取数据
female_data = load_data("FEMALE.txt")  # 加载女性数据
male_data = load_data("MALE.txt")      # 加载男性数据
all_data = np.vstack((female_data, male_data))  # 合并数据
labels = np.concatenate((np.zeros(len(female_data)), np.ones(len(male_data))))  # 创建标签数组 女性为0 男性为1

plt.subplot(2,2,1)
plt.scatter(all_data[:, 0], all_data[:, 1], c=labels)
plt.xlabel('height/cm')
plt.ylabel('weight/kg')
plt.title('initial')


# 1. 不考虑类别信息进行 PCA
cov_matrix = covariance_matrix(all_data)  # 计算协方差矩阵
eigen_values, eigen_vectors = eigen_values_vectors(cov_matrix)  # 计算特征值和特征向量
#print(eigen_values," ",eigen_vectors)
max_index = next(i for i, x in enumerate(eigen_values) if x == max(eigen_values))
w1=-eigen_vectors[:,max_index]
print("PCA w1:",w1)
transformed_data = k_l_transform(all_data, w1)  # 进行KL变换
#print(all_data[0],"*",eigen_vectors[max_index],"=",np.dot(eigen_vectors[max_index].T, all_data[0].T).T)

# 绘制 PCA 投影后的散点图
plt.subplot(2,2,2)
plot_scatter(transformed_data, labels, "PCA without class information")  # 绘制散点图

# 2. 类平均向量提取判别信息
# 计算类内散布矩阵
def within_class_scatter_matrix(data, labels):
    Sw = np.zeros((data.shape[1], data.shape[1]))
    mean_vector = np.zeros((data.shape[1], data.shape[1]))
    for i in range(2):
        class_data = data[labels == i]
        mean_vector[i] = np.mean(class_data, axis=0)
        Sw += np.cov(class_data, rowvar=False)
    return Sw,mean_vector

# 计算类间散布矩阵
def between_class_scatter_matrix(data, labels):
    Sb = np.zeros((data.shape[1], data.shape[1]))
    mean_vector = np.mean(data, axis=0)
    for i in range(2):
        class_data = data[labels == i]
        mean_class_vector = np.mean(class_data, axis=0)
        Sb += len(class_data) * np.outer(mean_class_vector - mean_vector, mean_class_vector - mean_vector)
    return Sb

Sw,mean_vector = within_class_scatter_matrix(all_data, labels)
Sb = between_class_scatter_matrix(all_data, labels)
# 计算判别方向
#Sw=[[3.5,1.5],[1.5,3.5]]
eigen_values, eigen_vectors = eigen_values_vectors(Sw)  # 计算特征值和特征向量  Sw
#eigen_values = [5,2]
#eigen_vectors = [[0.707,0.707],[0.707,-0.707]]
#print(eigen_values)
#print(eigen_vectors[:,0])
diag_matrix = np.diag(eigen_values)
B=np.dot(eigen_vectors,np.linalg.inv(diag_matrix**(0.5)))
#Sb=[[16,8],[8,4]]
Sb2=np.dot(np.dot(B.T,Sb),B)
eigen_values2, eigen_vectors2 = eigen_values_vectors(Sb2)
max_index2 = next(i for i, x in enumerate(eigen_values2) if x == max(eigen_values2))
w2=np.dot(B,eigen_vectors2[:,max_index2])*10
print("含类别信息 w2",w2)
transformed_data_discriminant = k_l_transform(all_data, w2)  # 进行KL变换

# 绘制判别分析投影后的散点图
plt.subplot(2,2,3)
plot_scatter(transformed_data_discriminant, labels, "Discriminant Analysis")  # 绘制散点图


# 绘制散点图
def plot_scatter_liner(data, labels, title, w=None,w0=None):
    if w is not None:
        # 使用 Fisher 判别方向对数据进行投影
        data = data.dot(w)+w0
    # 检查 data 的维度
    if data.ndim == 1:
        # data 是一维数组，绘制散点图
        plt.scatter(data, np.zeros(data.shape[0]), c=labels, cmap='viridis')
    else:
        # data 是二维数组，绘制散点图
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.yticks([])

# Fisher 判别
def fisher_discriminant(data, labels):
    Sw,mean_vector = within_class_scatter_matrix(data, labels)
    Sb = between_class_scatter_matrix(data, labels)
    eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    # 选择最大的特征值对应的特征向量
    w = eigen_vectors[:, np.argmax(eigen_values)]
    #w = np.dot(np.linalg.inv(Sw),(mean_vector[1]-mean_vector[0]))
    return w,mean_vector

# Fisher 判别
w,mean_vector = fisher_discriminant(all_data, labels)
print("Fisher w=",w)
w0 = -0.5 * np.dot(w,(mean_vector[0]+mean_vector[1]))
print("w0=",w0)
# 绘制 Fisher 判别投影后的散点图
plt.subplot(2,2,4)
plot_scatter_liner(all_data, labels, "Fisher Discriminant", w,w0)
plt.tight_layout()
plt.show()
# 分类结果
discriminant_labels = classify(all_data, w,w0)

# 计算准确率
accuracy = np.mean(discriminant_labels == labels)
print("Fisher discriminant accuracy:", accuracy)


w10 = -0.5 * np.dot(w1,(mean_vector[0]+mean_vector[1]))
w20 = -0.5 * np.dot(w2,(mean_vector[0]+mean_vector[1]))

# 分类结果
discriminant_labels1 = classify(all_data, w1,w10)
discriminant_labels2 = classify(all_data, w2,w20)
# 计算准确率
accuracy1 = np.mean(discriminant_labels1 == labels)
print("PCA accuracy:", accuracy1)

accuracy2 = np.mean(discriminant_labels2 == labels)
print("考虑类别信息 accuracy:", accuracy2)