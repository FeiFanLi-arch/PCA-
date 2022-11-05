import os.path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# 加载本地数据
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        image = Image.open(img_item_path)
        return image

    def __len__(self):
        return len(self.img_path)


# 加载数据集
def data():
    train_matrix = []
    test_matrix = []

    root_dir = "D:\\Python\\PCA\\dataset"
    train_dir = "train_set"
    test_dir = "test_set"
    train_dataset = MyData(root_dir, train_dir)
    test_dataset = MyData(root_dir, test_dir)
    for img01 in train_dataset:
        img_xn = img01
        img_xn = img_xn.resize((128, 128))
        gray_image = img_xn.convert('L')
        imgarray = list(gray_image.getdata())
        train_matrix.append(imgarray)
    for img02 in test_dataset:
        img_xn = img02
        img_xn = img_xn.resize((128, 128))
        gray_image = img_xn.convert('L')
        imgarray = list(gray_image.getdata())
        test_matrix.append(imgarray)

    train_matrix = np.array(train_matrix)
    test_matrix = np.array(test_matrix)
    return train_matrix, test_matrix, train_dataset


# 训练数据，PCA降维
def train_data(matrix):
    # 对样本进行中心化、去均值
    image_number, image_size = np.shape(matrix)
    mean_array = matrix.mean(axis=0)
    diff_matrix = matrix - mean_array
    diff_matrix = np.mat(diff_matrix).T
    # 求协方差矩阵
    cov = diff_matrix * diff_matrix.T
    # 求cov的特征值、特征向量
    lamda, u = np.linalg.eig(cov)
    u = list(u)
    # 取特征值大于1的，取巧：未严格按照特征值从大到小排
    for k in range(image_number):
        if lamda[k] < 1:
            u.pop(k)
    u = np.array(u)
    u = np.mat(u)
    p = u * diff_matrix
    return p, mean_array, u


# 识别
def test_data(p, mean_matrix, u, matrix):
    imagedata = matrix - mean_matrix
    imagedata = np.mat(imagedata).T
    test_image = u * imagedata  # 测试脸在特征向量下的数据

    # 计算距离
    distance = p - test_image
    distance = np.array(distance)
    distance = distance ** 2
    sum_distance = np.sum(distance, axis=0)
    new_sum_distance = sum_distance.argsort()
    train_index = []
    for j in range(2):
        train_index.append(new_sum_distance[j])
    train_index = np.array(train_index)
    return train_index


if __name__ == '__main__':
    train_image_matrix, test_image_matrix, train_raw_dataset = data()
    y, mean_image_matrix, w = train_data(train_image_matrix)
    index = test_data(y, mean_image_matrix, w, test_image_matrix)
    for i in range(2):
        img = train_raw_dataset[index[i]]
        img.show()
