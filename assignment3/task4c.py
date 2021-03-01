import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
from task4b import torch_image_to_numpy

if __name__ == "__main__":
    image = Image.open("images/zebra.jpg")
    print("Image shape:", image.size)

    model = torchvision.models.resnet18(pretrained=True)
    # Resize, and normalize the image with the mean and standard deviation
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = image_transform(image)[None]

    indices_4c = [i for i in range(10)]
    model_reduced = torch.nn.Sequential(*list(model.children())[:-2]) #Drop last 2 ResNet Modules

    final_activations = model_reduced(image)

    # Plot the activations from the final conv layer
    
    fig, axs = plt.subplots(1,len(indices_4c))

    for i in range(len(indices_4c)):
        act_image = torch_image_to_numpy(final_activations[0,indices_4c[i],:,:])
        axs[i].imshow(act_image, cmap = 'gray')

    plt.savefig('task4c_plot.png')
    plt.show()