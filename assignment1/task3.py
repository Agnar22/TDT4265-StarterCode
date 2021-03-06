import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    pred = model.forward(X).argmax(axis=1)
    targets_dec = targets.argmax(axis=1)
    accuracy = sum(pred == targets_dec) / targets.shape[0]
    return accuracy

def get_importance(model: SoftmaxModel):
    # Plotting of softmax weights (Task 4b)
    image = None
    for img in range(10):
        weight = model.w[:-1,img]
        weight = weight - np.min(weight)
        weight = weight / np.max(weight)
        if image is None:
            image = weight.reshape((28, 28))
        else:
            image = np.hstack((image, weight.reshape((28, 28))))
    return image


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        pred = self.model.forward(X_batch)
        self.model.backward(X_batch, pred, Y_batch)
        self.model.w -= learning_rate*self.model.grad
        loss = cross_entropy_loss(Y_batch, pred)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 500
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)
    l2_lambdas = [0.0, 1.0]
    pixel_weights = []
    for l2_reg_lambda in l2_lambdas:
        model = SoftmaxModel(l2_reg_lambda=l2_reg_lambda)
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
        # You can finish the rest of task 4 below this point.
        pixel_weights.append(get_importance(model))
    plt.imsave(f'task4b_softmax_weight.png', np.array(pixel_weights).reshape((28*len(l2_lambdas), -1)), cmap="gray")

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1.0, 0.1, 0.01, 0.001]
    plt.ylim([0.82, .95])
    l2_norms = []
    for l2_reg_lambda in l2_lambdas:
        model = SoftmaxModel(l2_reg_lambda=l2_reg_lambda)
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
        utils.plot_loss(val_history_reg01["accuracy"], f'val_acc l2: {l2_reg_lambda}')
        l2_norms.append(np.sum(np.square(model.w)))
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4e - Plotting of the l2 norm for each weight
    plt.bar(np.flip(np.arange(len(l2_lambdas))), l2_norms, align='center')
    plt.xticks(np.flip(np.arange((len(l2_lambdas)))), l2_lambdas)
    plt.xlabel("l2 lambdas")
    plt.ylabel("L2 norm")
    plt.savefig("task4e_l2_reg_norms.png")
    plt.show()
