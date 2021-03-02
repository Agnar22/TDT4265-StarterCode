from typing import Dict
import torch.nn as nn
import pathlib
import utils
import task2
import torch
import matplotlib.pyplot as plt
from trainer import Trainer
from typing import List


def create_conv(in_channels, num_filters, batch_norm, batch_norm_affine, kernel_size):
  return [
    nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=2),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(num_filters, affine=batch_norm_affine),
    nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=1, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, stride=2),
    nn.BatchNorm2d(num_filters, affine=batch_norm_affine)
  ] if batch_norm else [
    nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=2),
    nn.ReLU(inplace=True),
    nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=1, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, stride=2),
  ]

def create_linear(in_shape, nodes, dropout):
  return [
    nn.Linear(in_shape, nodes),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout),
  ]

class ExampleModel(nn.Module):

  def __init__(self,
               image_channels,
               num_classes,
               conv_layers: List[int],
               num_output_features: int,
               dense_layers: List[int],
               batch_norm: bool = False,
               batch_norm_affine: bool = False,
               dropout: float = 0,
               kernel_size=3):
    """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
    """
    super().__init__()
    feature_extractor_layers = [image_channels] + conv_layers
    self.num_classes = num_classes
    # Define the convolutional layers
    self.feature_extractor = nn.Sequential(
      *[
        sub_layer for layer in [
          create_conv(
            feature_extractor_layers[num_layer - 1],
            feature_extractor_layers[num_layer],
            batch_norm,
            batch_norm_affine,
            kernel_size
          ) for num_layer in range(1, len(feature_extractor_layers))]
        for sub_layer in layer
      ]
    )
    self.feature_extractor.apply(self.init_weights)
    # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
    self.num_output_features = num_output_features
    linear_layers = [num_output_features] + dense_layers
    # Initialize our last fully connected layer
    # Inputs all extracted features from the convolutional layers
    # Outputs num_classes predictions, 1 for each class.
    # There is no need for softmax activation function, as this is
    # included with nn.CrossEntropyLoss
    self.classifier = nn.Sequential(
      *[
        sub_layer for layer in [
          create_linear(
            linear_layers[num_layer - 1],
            linear_layers[num_layer],
            dropout
          ) for num_layer in range(1, len(linear_layers))
        ] for sub_layer in layer
      ],
      nn.Linear(64, num_classes),
    )
    self.classifier.apply(self.init_weights)

  def forward(self, x):
    """
    Performs a forward pass through the model
    Args:
        x: Input image, shape: [batch_size, 3, 32, 32]
    """
    inp = x
    inp = self.feature_extractor(inp)
    inp = inp.view(-1, self.num_output_features)
    out = self.classifier(inp)
    batch_size = x.shape[0]
    expected_shape = (batch_size, self.num_classes)
    assert out.shape == (batch_size, self.num_classes), \
      f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
    return out

  def init_weights(self, m):
    if type(m) in [nn.Linear, nn.Conv2d]:
      torch.nn.init.xavier_normal_(m.weight)


def create_combined_plots(trainers: Dict[str, Trainer], name: str):
  plot_path = pathlib.Path("plots")
  plot_path.mkdir(exist_ok=True)
  # Save plots and show them
  plt.figure(figsize=(20, 8))
  plt.subplot(1, 2, 1)
  plt.title("Cross Entropy Loss")
  for trainer_name, trainer in trainers.items():
    utils.plot_loss(trainer.train_history["loss"], label=f'{trainer_name} - training loss', npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label=f'{trainer_name} - validation loss')
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.title("Accuracy")
  for trainer_name, trainer in trainers.items():
    utils.plot_loss(trainer.validation_history["accuracy"], label=f'{trainer_name} - validation accuracy')
  plt.legend()
  plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
  plt.show()


if __name__ == "__main__":
  # Set the random generator seed (parameters, shuffling etc).
  # You can try to change this and check if you still get the same result!
  utils.set_seed(0)
  epochs = 10
  batch_size = 64
  dataloaders = task2.load_cifar10(batch_size, augmentation=True)
  # Assuming that by "the final train loss..." it means the epoch where the
  # model had the lowest validation loss.
  #
  # Task 3 a)
  # Worst model.
  learning_rate = 1e-2
  early_stop_count = 4
  worst_model = ExampleModel(
    image_channels=3,
    num_classes=10,
    conv_layers=[32, 64],
    num_output_features=64 * 8 * 8,
    dense_layers=[256, 128, 64],
    batch_norm=True,
    batch_norm_affine=False,
    dropout=0.0,
    kernel_size=5
  )
  worst_trainer = task2.Trainer(batch_size, learning_rate, early_stop_count, epochs, worst_model, dataloaders,adam=False)
  worst_trainer.train()
  worst_trainer.load_best_model()
  print(f'Train accuracy for worst model: {task2.compute_loss_and_accuracy(dataloaders[0], worst_trainer.model, worst_trainer.loss_criterion)[1]}')
  print(f'Validation accuracy for worst model: {task2.compute_loss_and_accuracy(dataloaders[1], worst_trainer.model, worst_trainer.loss_criterion)[1]}')
  print(f'Test accuracy for worst model: {task2.compute_loss_and_accuracy(dataloaders[2], worst_trainer.model, worst_trainer.loss_criterion)[1]}')

  # Best model.
  learning_rate = 1e-3
  early_stop_count = 4
  best_model = ExampleModel(
    image_channels=3,
    num_classes=10,
    conv_layers=[32, 64, 128],
    num_output_features=128*7*7,
    dense_layers=[256, 128, 128, 64],
    batch_norm=True,
    batch_norm_affine=False,
    dropout=0.1,
    kernel_size=3
  )
  best_trainer = task2.Trainer(batch_size, learning_rate, early_stop_count, epochs, best_model, dataloaders, adam=True)
  best_trainer.train()
  best_trainer.load_best_model()

  # Task 3 b)
  print(f'Train accuracy for best model: {task2.compute_loss_and_accuracy(dataloaders[0], best_trainer.model, best_trainer.loss_criterion)[1]}')
  print(f'Validation accuracy for best model: {task2.compute_loss_and_accuracy(dataloaders[1], best_trainer.model, best_trainer.loss_criterion)[1]}')
  print(f'Test accuracy for best model: {task2.compute_loss_and_accuracy(dataloaders[2], best_trainer.model, best_trainer.loss_criterion)[1]}')
  task2.create_plots(best_trainer, "task3b")

  # Task 3 d)
  learning_rate = 1e-3
  early_stop_count = 4
  best_model_no_batch = ExampleModel(
    image_channels=3,
    num_classes=10,
    conv_layers=[32, 64, 128],
    num_output_features=128*7*7,
    dense_layers=[256, 128, 128, 64],
    batch_norm=False,
    batch_norm_affine=False,
    dropout=0.1,
    kernel_size=3
  )
  best_trainer_no_batch = task2.Trainer(batch_size, learning_rate, early_stop_count, epochs, best_model_no_batch,
                                        dataloaders, adam=True)
  best_trainer_no_batch.train()
  create_combined_plots({'best_model': best_trainer, 'best_model_no_batch': best_trainer_no_batch}, "task3d")
  print(task2.compute_loss_and_accuracy(dataloaders[2], best_model_no_batch, best_trainer.loss_criterion))

  # Task 3 e)
  learning_rate = 5e-4
  early_stop_count = 4
  improved_best_model = ExampleModel(
    image_channels=3,
    num_classes=10,
    conv_layers=[32, 64, 128],
    num_output_features=128*7*7,
    dense_layers=[256, 128, 128, 64],
    batch_norm=True,
    batch_norm_affine=True,
    dropout=0.2,
    kernel_size=3
  )
  improved_best_trainer = task2.Trainer(batch_size, learning_rate, early_stop_count, epochs, improved_best_model, dataloaders, adam=True)
  improved_best_trainer.train()
  improved_best_trainer.load_best_model()
  task2.create_plots(improved_best_trainer, "task3e")
  print(f'Test accuracy for improved best model: {task2.compute_loss_and_accuracy(dataloaders[2], improved_best_trainer.model, improved_best_trainer.loss_criterion)[1]}')
