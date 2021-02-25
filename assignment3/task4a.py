import pathlib
import matplotlib.pyplot as plt
import utils
import torchvision
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from task2 import create_plots


class transferLearningModel(nn.Module):

    def __init__(self):
        #Code from Listing 1 from assignment 
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512,10)

        for param in self.model.parameters(): #Freeze all params
            param.requires_grad = False
        for param in self.model.fc.parameters(): #Unfreeze last FC layer
            param.requires_grad = True
        for param in self.model.layer4.parameters(): #Unfreeze last 5 conv layers
            param.requires_grad = True


    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size, resize=True)
    model = transferLearningModel()#image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    trainer.train()
    create_plots(trainer, "task4a")

    #Get final test accuracy
    best_model = trainer.load_best_model()
    [_, test_accuracy] = compute_loss_and_accuracy(dataloaders[2], #This is the test set
                                 best_model, torch.nn.CrossEntropyLoss())
    print(test_accuracy)