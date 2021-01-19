import numpy as np
import torch
import torch.utils.data as data
import glob
import imageio


def is_image_file(filename):  # 定义一个判断是否是图片的函数
    return any(filename.endswith(extension) for extension in [".tif", '.png', '.jpg'])


class train_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256, flip=0, time_series=4, batch_size=1):
        super(train_dataset, self).__init__()
        self.src_list = np.array(sorted(glob.glob(data_path + 'src/' + '*.jpg')))
        self.lab_list = np.array(sorted(glob.glob(data_path + 'lab/' + '*.png')))
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip
        self.time_series = time_series
        self.index = 0
        self.batch_size = batch_size

    def data_iter_index(self, index=1000):
        batch_index = np.random.choice(len(self.src_list), index)
        x_batch = self.src_list[batch_index]
        y_batch = self.lab_list[batch_index]
        data_series = []
        label_series = []
        try:
            for i in range(index):
                data_series.append(imageio.imread(x_batch[i]) / 255.0)
                label_series.append(imageio.imread(y_batch[i]) - 1)
                self.index += 1

        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

        return data_iter

    def data_iter(self):
        data_series = []
        label_series = []
        try:
            for i in range(len(self.src_list)):
                data_series.append(imageio.imread(self.src_list[i]) / 255.0)
                label_series.append(imageio.imread(self.lab_list[i]) - 1)
                self.index += 1

        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

        return data_iter