# NN Visualizer

NN visualizer is an in-browser, interactive neural network visualizer. It allows you to build, configure, train, and visualize various neural network architectures directly in your web browser.

## Screenshot

* Training on MNIST
![training on mnist](assets/mnist.gif "training on mnist")

## Features

  * **Network building:**
      * Add different types of layers: Dense, Convolutional, Flatten, and Max Pooling.
      * Customize layer parameters such as the number of neurons, kernel size, stride, padding, and activation functions.
      * Easily reorder layers using drag and drop.
  * **Training and Visualization:**
      * Train your custom neural networks on pre-loaded datasets like MNIST and catvnoncat.
      * See real-time updates on training progress, including loss, accuracy, and time per batch.
      * Visualize the training loss and accuracy on a live-updating chart.
      * Visualize the activations of each layer for a sample input from the dataset.
      * Inspect the learned weights of convolutional filters on hover.
  * **Persistence:**
      * Save your network architecture to the browser's local storage to continue your work later.
      * Load previously saved networks.
      * Clear the current network or the saved network from storage.
  * **Web Worker-based Training (async):**
      * The training process runs in a background web worker to keep the UI responsive.
      * You can start, pause, resume, and stop the training at any time.

## Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/vark1/nn-visualizer.git
    cd nn-visualizer
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    ```

3.  **Run the development server:**

    ```bash
    npm run dev
    ```

    This will start the Vite development server, and you can access the visualizer in your browser at the provided URL (usually `http://localhost:5173`).