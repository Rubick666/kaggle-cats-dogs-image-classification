# Cats and Dogs Image Classification (Kaggle Dataset)

This repository contains three deep learning projects for binary image classification (cats vs. dogs) using the [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data). Each project demonstrates a different approach to solving the problem with Keras and TensorFlow.

## Projects

### 1. Baseline CNN Model (`Cats_and_Dogs_1.ipynb`)
- Implements a basic Convolutional Neural Network (CNN) from scratch.
- Data is split into training, validation, and test sets.
- Model is trained and evaluated, and results are visualized.

### 2. CNN with Data Augmentation & Dropout (`Cats_and_Dogs_2.ipynb`)
- Expands on the baseline by adding data augmentation and dropout layers to reduce overfitting.
- Uses Keras' `ImageDataGenerator` for real-time data augmentation.
- Achieves improved generalization on the validation set.

### 3. Transfer Learning with VGG16 (`Cats_and_Dogs_TransferLearning.ipynb`)
- Uses a pretrained VGG16 model as a feature extractor and for fine-tuning.
- Demonstrates three approaches: feature extraction, fine-tuning, and end-to-end training.
- Achieves state-of-the-art results for this dataset.

## Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib

## How to Run
1. Download and extract the [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
2. Update the dataset paths in the notebooks as needed.
3. Open each notebook in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
4. Run all cells in order to train and evaluate the models.
5. (Optional) If running locally, install the dependencies:

    ```
    pip install -r requirements.txt
    ```

## Results
- The transfer learning approach achieves the highest accuracy.
- Training/validation accuracy and loss plots are included in each notebook.

## Folder Structure
cats-dogs-image-classification/
├── Cats_and_Dogs_1.ipynb
├── Cats_and_Dogs_2.ipynb
├── Cats_and_Dogs_TransferLearning.ipynb
├── requirements.txt
└── README.md

## Dataset
- [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

## Author
- [Amirfarhad](https://github.com/Rubick666)
