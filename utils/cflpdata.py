from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 在这里实现你的数据加载逻辑，只返回数据
        sample = self.data_list[index]
        # 假设数据是一个元组 
        return sample