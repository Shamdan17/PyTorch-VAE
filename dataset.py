import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
# Add your custom dataset class here
class BEVDataset(Dataset):
    def __init__(self, 
                data_path: str,
                split: str, 
                transform: Optional[Callable] = None,
                zero_out_red_channel: bool = False,
                image_folder: str = 'bev'):
        """
        Args:
            data_path: path to the data folder containing a random number of folders with images inside a folder called image_folder
            split: train, val or test
            transform: optional transform to be applied on a sample
            image_folder: name of the folder containing the images
        """
        self.data_path = data_path
        self.split = split
        self.image_folder = image_folder
        self.transform = transform
        self.zero_out_red_channel = zero_out_red_channel
        
        self.main_folders = os.listdir(self.data_path)
        if split=="train":
            self.main_folders = self.main_folders[:-2]
        elif split=="val":
            self.main_folders = [self.main_folders[-2]]
        elif split=="test":
            self.main_folders = [self.main_folders[-1]]


        self.images = []
        for folder in self.main_folders:
            self.images += [os.path.join(self.data_path, folder, self.image_folder, x) for x in os.listdir(os.path.join(self.data_path, folder, self.image_folder))]
            print(f"Detected {len(self.images)} images in split {self.split}")
        # self.train_folder_sizes= [len(os.listdir(os.path.join(self.data_path, folder, self.image_folder))) for folder in self.train_folders]
        # self.val_folder_sizes = [len(os.listdir(os.path.join(self.data_path, folder, self.image_folder))) for folder in self.val_folders]
        # self.test_folder_sizes = [len(os.listdir(os.path.join(self.data_path, folder, self.image_folder))) for folder in self.test_folders]
    
    
    def __len__(self):
        return len(self.images)
        # pass
    
    def __getitem__(self, idx):
        img = default_loader(self.images[idx])

        if self.transform is not None:
            img = self.transform(img)
        
        if self.zero_out_red_channel:
            img[0, :, :] = 0
        
        return img, 0.0

    

# class OxfordPets(Dataset):
#     """
#     URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
#     """
#     def __init__(self, 
#                  data_path: str, 
#                  split: str,
#                  transform: Callable,
#         self.data_dir = Path(data_path) / "OxfordPets"        
#         self.transforms = transform
#         imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
#         self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
#     def __len__(self):
#         return len(self.imgs)
    
#     def __getitem__(self, idx):
#         img = default_loader(self.imgs[idx])
        
#         if self.transforms is not None:
#             img = self.transforms(img)
        
#         return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        crop_size: Union[int, Sequence[int]] = (192, 192),
        patch_size: Union[int, Sequence[int]] = (192, 192),
        random_crop_chance = 0.3,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_folder: str = 'bev',
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.crop_size = crop_size
        self.random_crop_chance = random_crop_chance
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_folder = image_folder

    # def setup(self, stage: Optional[str] = None) -> None:
    def setup(self, **kwargs) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )

#       =========================  BEV Dataset  =========================
    
        # TODO: Normalize possibly
        random_crop_transform = [transforms.RandomCrop(self.crop_size)]


        train_transforms = transforms.Compose([transforms.RandomApply(random_crop_transform, p=self.random_crop_chance),
                                              transforms.CenterCrop(self.crop_size),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.CenterCrop(self.crop_size),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = BEVDataset(
            self.data_dir,
            split='train',
            image_folder=self.image_folder,
            transform=train_transforms
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = BEVDataset(
            self.data_dir,
            split='val',
            image_folder=self.image_folder,
            transform=val_transforms,
        )
#       ===============================================================
      
# #       =========================  CelebA Dataset  =========================
    
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(148),
#                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(148),
#                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),])
        
#         self.train_dataset = MyCelebA(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#             download=False,
#         )
        
#         # Replace CelebA with your dataset
#         self.val_dataset = MyCelebA(
#             self.data_dir,
#             split='test',
#             transform=val_transforms,
#             download=False,
#         )
# #       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     