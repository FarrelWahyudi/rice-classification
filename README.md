# Rice Variety Classification using CNN

This repository features a comprehensive deep learning pipeline designed to classify five distinct varieties of rice using a Convolutional Neural Network (CNN). Built with **TensorFlow** and **Keras**, the project includes automated data preprocessing, GPU-accelerated training, and performance visualization.

## 🌾 Supported Varieties
The model is specifically trained to identify and distinguish between:
* **Arborio**
* **Basmati**
* **Ipsala**
* **Jasmine**
* **Karacadag**

## 🚀 Features
* **Automated Data Cleaning**: Scans dataset directories to identify and remove corrupted images or unsupported file formats (supports `.jpg`, `.jpeg`, `.png`, `.bmp`).
* **Optimized Data Pipeline**: Utilizes `tf.keras.utils.image_dataset_from_directory` for efficient memory management and batching.
* **GPU Acceleration**: Pre-configured for NVIDIA CUDA support to significantly reduce training time.
* **Inference Script**: Includes a testing section to verify model predictions on new images with confidence scores.

## 🛠️ Tech Stack
* **Frameworks**: TensorFlow 2.16.1, Keras 3.14.0
* **Image Processing**: OpenCV 4.11.0.86, Pillow (PIL)
* **Data & Math**: NumPy 1.26.4, Matplotlib

## 📋 Prerequisites
* Python 3.x
* NVIDIA Drivers & CUDA Toolkit (for GPU support)
* Recommended: Virtual environment (venv or conda)

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/rice-classification.git](https://github.com/your-username/rice-classification.git)
   cd rice-classification
   ```

2. **Install Dependencies**
    The following versions are required for compatibility:
    ```bash
    pip install --upgrade pip
    pip install "tensorflow[and-cuda]==2.16.1"
    pip install numpy==1.26.4 opencv-python==4.11.0.86 pillow matplotlib
    ```

## 📂 Project Structure
    ```
    ├── main.ipynb          # Full pipeline: Setup, Cleaning, Training, and Evaluation
    ├── data/               # Dataset root directory
    │   └── rice/           # Subfolders: Arborio, Basmati, Ipsala, Jasmine, Karacadag
    └── test/               # Directory for testing unseen images
        └── rice_test1.jpg  # Sample test image
    ```

## 🧪 Usage
1. Prepare Data: Organize your dataset into the data/rice/ directory, with each variety in its own named folder.

2. Configure GPU: The notebook automatically attempts to enable memory growth for available GPUs to prevent allocation errors.

3. Data Cleaning: Run the "Import and Setup" and "Data Preprocessing" cells to sanitize your image files.

4. Train Model: Execute the training blocks to build the CNN. The model uses a 250x250 input size.

5. Inference: Use the final cell in the notebook to pass a file path to img_path to see the prediction results and confidence levels.

## 📊 Evaluation
The notebook generates training history plots (accuracy and loss) and a confusion matrix to help evaluate the model's precision across all five rice categories.