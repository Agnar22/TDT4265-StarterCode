
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np

def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

if __name__ == "__main__":
    image = Image.open("images/zebra.jpg")
    print("Image shape:", image.size)

    model = torchvision.models.resnet18(pretrained=True)

    print(model)
    first_conv_layer = model.conv1
    print("First conv layer weight shape:", first_conv_layer.weight.shape)
    print("First conv layer:", first_conv_layer)

    # Resize, and normalize the image with the mean and standard deviation
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = image_transform(image)[None]
    print("Image shape:", image.shape)

    activation = first_conv_layer(image)
    print("Activation shape:", activation.shape)

    indices = [14, 26, 32, 49, 52]

    fig, axs = plt.subplots(2,len(indices))


    for i in range(len(indices)):
        weight_image = torch_image_to_numpy(first_conv_layer.weight[indices[i],:,:,:])
        act_image = torch_image_to_numpy(activation[0,indices[i],:,:])

        axs[0,i].imshow(weight_image)
        axs[1,i].imshow(act_image, cmap = 'gray')

    plt.savefig('task4b_plot.png')
    plt.show()
