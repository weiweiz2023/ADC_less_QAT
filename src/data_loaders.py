import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import src.argsparser as argsparser
import os
args = argsparser.get_parser().parse_args()

# imagenet_data = torchvision.datasets.ImageNet('./imagenet')
# data_loader = torch.utils.data.DataLoader(imagenet_data,
#                                           batch_size=4,
#                                           shuffle=True,
#                                           num_workers=4)
#
import numpy as np
import torch


def get_loaders(dataset, batch_size, workers):


    

    if dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                   std=[0.2023, 0.1994, 0.2010])
         
        

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=True)   # Add this

    elif dataset == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data_100', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data_100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=True)

    elif dataset == 'tiny_imagenet':
        tiny_imagenet_normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root='./tiny-imagenet-200/train', transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),
                transforms.ToTensor(),
                tiny_imagenet_normalize,
            ])),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root='./tiny-imagenet-200/val', transform=transforms.Compose([
                transforms.ToTensor(),
                tiny_imagenet_normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=True)

    elif dataset == 'imagenet':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        imagenet_train = datasets.ImageNet('~/ImageNet/', split='train', transform=train_transforms)
        imagenet_val = datasets.ImageNet('~/ImageNet/', split='val', transform=val_transforms)
        train_loader = torch.utils.data.DataLoader(imagenet_train,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(imagenet_val,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=workers, pin_memory=True, drop_last=True)

    
     
    
    elif dataset == 'MNIST':
        
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./MNIST_Data', train=True, download=True,
                           transform=transforms.Compose([
                               
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(28, 4),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.1307,), (0.3081,)),
                           ])),
            batch_size=batch_size, shuffle=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./MNIST_Data', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=False)

    else:
        raise ValueError("Inncorect dataset selection, check paramaeters")

    return train_loader, val_loader

