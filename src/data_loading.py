from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def get_data(dataset_path, dataset_type):
    train_dataset_path = dataset_path + dataset_type
    custom_transform = transforms.Compose(
                            [transforms.Resize((50,50)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=0.4301,std=0.2203)])

    train_ds = datasets.ImageFolder(root=train_dataset_path, transform=custom_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    classes = list(train_ds.class_to_idx.keys())
    return train_loader, classes