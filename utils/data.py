import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import idr_torch
from PIL import Image
import os


class ImageDataset(Dataset):
    def __init__(self, data_dir, mode='train', transforms=None):
        A_dir = os.path.join(data_dir, 'monet_jpg')
        B_dir = os.path.join(data_dir, 'photo_jpg')
        
        if mode == 'train':
            self.files_A = [os.path.join(A_dir, name) for name in sorted(os.listdir(A_dir))[:250]]
            self.files_B = [os.path.join(B_dir, name) for name in sorted(os.listdir(B_dir))[:250]]
        elif mode == 'test':
            self.files_A = [os.path.join(A_dir, name) for name in sorted(os.listdir(A_dir))[250:]]
            self.files_B = [os.path.join(B_dir, name) for name in sorted(os.listdir(B_dir))[250:301]]
        
        self.transforms = transforms
        
    def __len__(self):
        return len(self.files_A)
    
    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = self.files_B[index]
        
        img_A = Image.open(file_A)
        img_B = Image.open(file_B)
        
        if self.transforms is not None:
            img_A = self.transforms(img_A)
            img_B = self.transforms(img_B)
        
        return img_A, img_B


transforms_ = transforms.Compose([
   # transforms.Resize(int(256*1.12), Image.BICUBIC),
    #transforms.RandomCrop(256, 256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class ImageDatasetEval(Dataset):
    def __init__(self, data_dir, transforms=None):

        B_dir = os.path.join(data_dir, 'photo_jpg')   
        self.files_B = [os.path.join(B_dir, name) for name in sorted(os.listdir(B_dir))]
        
        self.transforms = transforms
        
    def __len__(self):
        return len(self.files_B)
    
    def __getitem__(self, index):

        file_B = self.files_B[index]
        img_B = Image.open(file_B)
        
        if self.transforms is not None:
            img_B = self.transforms(img_B)
        
        return img_B


transforms_train = transforms.Compose([
   # transforms.Resize(int(256*1.12), Image.BICUBIC),
    #transforms.RandomCrop(256, 256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


transforms_eval = transforms.Compose([
   # transforms.Resize(int(256*1.12), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_dataset(data_dir, batch_size=32, mode="train"):

    if mode == "train":
        dataset = ImageDataset(data_dir, mode='train', transforms=transforms_train)
        data_sampler = DistributedSampler(dataset, shuffle=True, num_replicas=idr_torch.size, rank=idr_torch.rank)
        
        loader = DataLoader(
            dataset,
            sampler=data_sampler,
            batch_size = batch_size,
            num_workers = 3,
            persistent_workers = True,
            pin_memory = True,
            prefetch_factor = 2,
            )

    elif mode == "eval":
        dataset = ImageDatasetEval(data_dir, transforms=transforms_eval)
        data_sampler = DistributedSampler(dataset, shuffle=False, num_replicas=idr_torch.size, rank=idr_torch.rank)

        loader = DataLoader(
            dataset,
            sampler=data_sampler,
            batch_size = batch_size,
            num_workers = 3,
            persistent_workers = True,
            pin_memory = True,
            prefetch_factor = 2,
            )

    elif mode == "test":
        dataset = ImageDataset(data_dir, mode='test', transforms=transforms_train)
        data_sampler = DistributedSampler(dataset, shuffle=False, num_replicas=idr_torch.size, rank=idr_torch.rank)

        loader = DataLoader(
            dataset,
            sampler=data_sampler,
            batch_size = batch_size,
            num_workers = 3,
            persistent_workers = True,
            pin_memory = True,
            prefetch_factor = 2,
            )

    return loader
