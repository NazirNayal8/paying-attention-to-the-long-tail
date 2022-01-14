import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageNetLTDataset(Dataset):
    
    train_data_info_path = 'data/ImageNet_LT_train.json'
    valid_data_info_path = 'data/ImageNet_LT_val.json'
    
    def __init__(self, mode='train', transforms=None):
        
        self.transforms = transforms
        self.data_info_path = self.train_data_info_path if mode == 'train' else self.valid_data_info_path
        
        with open(self.data_info_path, 'rb') as f:
            data_info = json.load(f)
            
        self.num_classes = data_info['num_classes']
        self.annotations = data_info['annotations']
        self.data_size = len(self.annotations)
        
        per_class_frequency = np.zeros(self.num_classes)
        
        for a in self.annotations:
            label = a['category_id']
            per_class_frequency[label] += 1
            
        self.per_class_frequency = per_class_frequency
        
        ordered_classes = [(i, val) for i, val in enumerate(self.per_class_frequency)]
        ordered_classes.sort(key=lambda x: -x[1])
        self.ordered_classes = np.array([i  for (i, val) in ordered_classes])
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        
        img_path = self.annotations[index]['fpath']
        label = self.annotations[index]['category_id']
        img = Image.open(img_path)
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, label
        