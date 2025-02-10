import tensorflow as tf
import numpy as np

class AdversarialFuzzer:
    def __init__(self, model, loss_function, epsilon=0.05, max_iterations=50, targeted=False, target_label=None):
        self.model = model
        self.loss_function = loss_function
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.targeted = targeted  # Flag for targeted attacks
        self.target_label = target_label # Specific target label for targeted attacks

    def generate_adversarial_example(self, input_data, true_label):
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
        input_data = tf.Variable(input_data)

        for i in range(self.max_iterations):
            with tf.GradientTape() as tape:
                predictions = self.model(input_data)
                if self.targeted:
                    # For targeted attacks, maximize the loss of the target label
                    loss = self.loss_function(self.target_label, predictions)
                else:  # Untargeted attack
                    loss = self.loss_function(true_label, predictions)

            gradients = tape.gradient(loss, input_data)
            signed_grad = tf.sign(gradients)

            if self.targeted:
                perturbation = -self.epsilon * signed_grad # Reverse direction for targeted attack
            else:
                perturbation = self.epsilon * signed_grad

            adversarial_example = input_data + perturbation
            adversarial_example = tf.clip_by_value(adversarial_example, 0, 1)  # Example for image data
            input_data.assign(adversarial_example)

        return adversarial_example.numpy()

    def fuzz(self, input_data_samples, true_labels, target_labels=None):
        adversarial_examples = []
        for i in range(len(input_data_samples)):
            if self.targeted:
                if target_labels is None:
                    raise ValueError("Target labels must be provided for targeted attacks.")
                adversarial_example = self.generate_adversarial_example(input_data_samples[i], true_labels[i], target_labels[i])
            else:
                adversarial_example = self.generate_adversarial_example(input_data_samples[i], true_labels[i])
            adversarial_examples.append(adversarial_example)
        return np.array(adversarial_examples)



# Example Usage (replace with your actual model and data)
# Assuming 'model' is your SafeTensor model, 'images' are your input data, and 'labels' are the corresponding labels

# Sample Data (replace with your actual data)
images = np.random.rand(10, 28, 28, 1) # Example image data (10 samples, 28x28 grayscale)
labels = np.random.randint(0, 10, 10) # Example labels (10 samples, 10 classes)
# One-hot encode labels (important for CategoricalCrossentropy)
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=10)


# Example Loss Function (adjust for your task)
loss_fn = tf.keras.losses.CategoricalCrossentropy()  # For multi-class classification

# Untargeted Attack
fuzzer_untargeted = AdversarialFuzzer(model, loss_fn)
adversarial_images_untargeted = fuzzer_untargeted.fuzz(images, labels_one_hot)

# Targeted Attack (Example: target each image to the next class)
target_labels = (labels + 1) % 10  # Example target labels (shift by one class)
target_labels_one_hot = tf.keras.utils.to_categorical(target_labels, num_classes=10)

fuzzer_targeted = AdversarialFuzzer(model, loss_fn, targeted=True, target_label=target_labels_one_hot) # Pass target labels here
adversarial_images_targeted = fuzzer_targeted.fuzz(images, labels_one_hot, target_labels_one_hot) # Pass target labels to fuzz

# ... Evaluate the model's performance on the adversarial examples
predictions_adv_untargeted = model.predict(adversarial_images_untargeted)
predictions_adv_targeted = model.predict(adversarial_images_targeted)

# ... compare predictions_adv with the true labels to measure the attack's success
