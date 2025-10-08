> NAMES: M K LOKESH KUMAR & HARISH SENTHILKUMARAN DHARANI

> REG NO: 22011103026 & 22011103017

> CLASS: CYBERSECURITY 4TH YEAR


# CAPSTONE PROJECT REVIEW 2: ADVERSARIAL ATTACKS DETECTION ON IMAGE CLASSIFYING MODEL

## Overall Project Architecture

- Multi-stage defensive pipeline to protect Convolutional Neural Network (CNN) from adversarial attacks
- Modular architecture for detecting, purifying and classifying input images

### Workflow

1. Input: Submit an image to the System via API Gateway
2. Detection Stage: Image is first analyzed by the Detection Model. The model is a binary classifier trained to distinguished the image as either "Clean" or "Adversarial"
3. Routing based on Condition
    - If the image is classified as "Clean", it skips the defensive purification step and is sent directly to final classification model
    - If the image is classified as "Adversarial", it is routed to the Purification Model
4. Purification Stage: Model processes the flagged image to remove/reduce the malicious perturbations to attempt to produce a Purified Image
5. Mitigation Stage: The Mitigation Model is a robust classifier specifically hardened against attacks, and receives the purified adversarial image to perform the final prediction
6. Output: Final classification result is returned

## Use of Adversarial Attack: Projected Gradient Descent (PGD)

### What is PGD?
- Type of White-Box attack
    - attacker has full knowledge of the target model's architecture, and parameters

- Tries to find the worst-case peturbation (slight offset to the image) for the input image that will cause a misclassification, while keeping the perturbation as small as possible, such that it is imperceptible to humans by simply looking

- It is an iterative model
    - takes multiple small steps in the direction that maximizes the model's loss

- Process of the PGD attack
    - starts with original image and adds a small amount of random noise
    - in each iteration, calculates gradient of the loss function with respect to the input image's pixels
        - gradient will point in the direction of the steepest increase in the model's error
    - take a small step in the SAME direction of the gradient, adding more noise to the image
    - to prevent the perturbation from exceeding the maximum allowed limit, which is determined by a budget, called ε (epsilon)
        - this is to ensure the image remains visully similar to the original

### Use of PGD in the project

1. For the Detection Dataset Generation (Offensive Step)
- Objective: To generate large dataset of adversarial examples needed to train the detection model
- Implementation: Launch the attack against the undefended baseline model
    - adding minimal amount of invisible noise needed to fool the model, causing it to misclassify the image (makes the model think of 7 as 2)
    - this image is saved and given the adversarial label "1" while original image is given clean label "0"
    - Epsilon is set to 0.09, which defines the maximum allowed perturbation. This value is moderate as it is sufficient to fool a basic model

- This is done to train the detection model to be able to identify what an attack looks like

2. For Adversarial Training of the Mitigation Model (Defensive Step)
- Objective: To harden the mitigation model by exposing it to strong attacks during training phase
- Implementation: Integrated directly into the training loop wrapping the mitigation model
    - For each training batch, adversarial examples are generated on-the-go
    - Model's weights are updated to correctly classify the purturbed images
    - Epsilon is a bit higher now (0.2) to allow for a much larger perturbation for a much stronger attack hardening the mitigation model to learn to correctly classify the images, despite perturbation

## Model Architecture Used: SimpleCNN

- use of a simple model like CNN is to ensure that any form of performance difference is due to training data and methods, and not the architectural damages

- Detailed Architecture
    - Input: 1 x 28 x 28 single channel image

    - CNN layer 1: 1 input channel with 32 output channels using 3x3 kernel
        - Output Feature Map Size - 32 (in number) x 26 x 26
        - Reason: Initial feature detector. Identifying basic, low-level patterns like edges, corners, gradients and simple textures. Each of the 32 feature maps highlights locations of one of the specific learned patterns

    - ReLU Activation Function (Rectified Linear Unit)
        - Reason: To introduce non-linearity to the model. Otherwise it would behave like a simple linear model. Allows network to learn more complex, non-linear relationships between input pixels and final output. Also helps prevent the Vanishing Gradient problem during training

    - CNN layer 2: 32 input channels with 64 output channels using 3x3 kernel
        - Output Feature Map Size - 64 (in number) x 24 x 24
        - Reason: Takes the simple patterns learned in the first layer and learns to combine them into more complex and meaningful shapes like learning curves. Acts as hierarchical feature aggregator

    - ReLU Activation Function (Rectified Linear Unit)

    - Dropout Layer - p = 0.25 for regularization
        - Reason: Powerful regularization technique used to prevent overfitting. During training, randomly drops out a certain percentage of neurons to ensure the model does not become too reliant on any single neuron to be active and forces model to learn more robust and redundant features

    - Flattening 64 x 24 x 24 to 1-D vector of size 9216

    - Fully Connected Layer: Transforming 9216-feature vector to 128 feature
        - Reason: High level reasoning. Receives flattened vector of complex features from convolutional layers and learns to weigh their importance. To learn final non-linear combinations of these features to determine which output class is most likely.
    - ReLU Activation Function (Rectified Linear Unit)
    - Dropout Layer - p = 0.5 for further regularization

    - Output Layer
        - For Baseline and Mitigation Models - 128 input features and 10 output features (digits 0-9)
        - For Detection Model - 128 input features and 2 output features ("clean" or "adversarial")
        - Softmax activation function to produce probabilities for each output class

## Training Process and Hyperparameters

1. Common Setup
- Data Preprocessing - Transforming all MNIST (train=True) images to convert them to PyTorch tensors and then normalize with a mean=0.1307 and standard deviation=0.3801
- Batch Size - 128 for all training processes
- Learning Rate - 0.001 used for all models

2. Basline Model Training
- Objective: To create standard, NON-robust classifier
- Dataset: Clean MNIST training set
- Process: Standard supervised training for 5 epochs
- Output: To ['baseline_model.pth'](../models/baseline_model.pth)

3. Detection Model Training
- Objective: Train Binary Classifier to detect adversarial examples
- Dataset Generation: Custom dataset built using combination of two sources
    - clean MNIST images used as "Clean" and labeled with 0
    - adversarial versions generated using PGD attack (eps=0.09) against baseline model and used as "adversarial" with label 1
- Process: The SimpleCNN is trained on the custom dataset for 10 epochs
- Output: To ['detection_model.pth'](../models/detection_model.pth)

4. Mitigation Model Training
- Objective: To build hardened classifier against PGD attacks
- Process: SimpleCNN trained for 20 epochs where in each set the model learns to classify correctly adversarial examples that are generated on-the-go with strong PGD configuration (eps=0.2)
- Output: To ['mitigation_model.pth'](../models/mitigation_model.pth)

## Model Testing and Performance Analysis

- Final validation conducted on the unseen MNIST test data (train=False) using very strong PGD attack (eps=0.3)

1. Evaluation of the Detection System
- test confirmed that the attack successfully fools the baseline model, as it classified the adversarial image of the digit "7" as a "2" instead
- the detection model correctly classifies the adversarial image as "Adversarial"

2. Baseline Model vs Mitigation Model
- Baseline Model Performance
    - On the clean image (True: 7): It predicted 7 (Correct).
    - On the adversarial image (True: 7): It predicted 2 (Failed).
- Mitigation Model Performance
    - On the adversarial image (True: 7): It predicted 7 (Correct and Robust).
    - On the clean image (True: 7): It predicted 2 (Incorrect due to accuracy-robustness tradeoff).
        - training a model to be highly resilient to adverserial attacks makes it slightly less accurate on clean, normal images
        - noticeable in MNIST due to natural variation in different handwritings causes the robust model to mistake these handwriting quirks for a malicious attack

- Accuracy-Robustness Trade-Off
    - A standard model, trained for maximum accuracy, learns a sharp and complex decision boundary to fit the training data precisely. This makes it highly accurate on clean data but brittle and vulnerable to attacks, as a small push can easily cross the boundary.
    - A robust model, trained adversarially, learns a much smoother and simpler decision boundary with a large buffer zone. This makes it excellent at resisting adversarial attacks, as the small push is no longer enough to cross the wide margin.
    - A clean but "atypical" data point (like an unusual but valid handwritten digit) might be misclassified because the new, smoother boundary no longer has the specific curve needed to include it. In essence, the model gives up a little bit of precision on these edge cases to gain a huge amount of stability against malicious attacks.

