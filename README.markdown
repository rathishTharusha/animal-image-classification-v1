# Animal Image Classification

This project builds an image classification model to distinguish between dogs, cats, and snakes using ResNet50 with transfer learning in Keras. The model is trained on the Animal Image Classification Dataset from Kaggle, leveraging pre-trained weights from ImageNet. The implementation includes data preprocessing with augmentation, model training with callbacks, fine-tuning, evaluation with multiple metrics, and visualization of results, meeting the requirements for a professional-level machine learning project.

## Dataset

- **Source**: Animal Image Classification Dataset
- **Structure**: Images are organized in the `Animals` folder with subfolders `dogs`, `cats`, and `snakes`.
- **Size**: 2000 training images, 50 validation images, 50 test images across three classes.
- **Splitting**: The dataset is split into training (80%), validation (10%), and test (10%) sets using `ImageDataGenerator` with `validation_split` for train/validation and a manual split for the test set.

## Obtaining the Dataset

The dataset is hosted on Kaggle and can be downloaded using one of the following methods:

### Method 1: Manual Download

1. **Create a Kaggle Account**: Sign up or log in at kaggle.com.
2. **Access the Dataset**: Visit the Animal Image Classification Dataset page.
3. **Download**: Click the "Download" button to get the dataset as a ZIP file (e.g., `animal-image-classification-dataset.zip`).
4. **Extract**: Unzip the file to create an `Animals` folder with subfolders `dogs`, `cats`, and `snakes`.
5. **Place in Project**: Move the `Animals` folder to the root of your project directory (same level as `animal_classifier.ipynb`).

### Method 2: Kaggle API

1. **Install Kaggle API**:

   ```bash
   pip install kaggle
   ```
2. **Set Up API Key**:
   - Go to your Kaggle account: Profile &gt; Settings &gt; API &gt; Create New Token.
   - Download `kaggle.json` and place it in `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<YourUsername>\.kaggle\kaggle.json` (Windows).
   - Set permissions (Linux/macOS):

     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```
3. **Download Dataset**:

   ```bash
   kaggle datasets download -d borhanitrash/animal-image-classification-dataset
   ```
4. **Extract**:
   - Unzip the downloaded file (e.g., `animal-image-classification-dataset.zip`).
   - Move the `Animals` folder to your project directory.
5. **Verify Structure**: Ensure the `Animals` folder contains `dogs`, `cats`, and `snakes` subfolders.

**Note**: The dataset is not included in this repository due to its size. Follow the above steps to obtain it.

## Model

- **Architecture**:
  - **Base**: ResNet50 pre-trained on ImageNet, with top layers removed.
  - **Custom Layers**: GlobalAveragePooling2D, Dropout (0.5), Dense (128 units, ReLU), Dense (3 units, softmax).
- **Preprocessing**:
  - Images resized to 128x128 pixels.
  - Data augmentation: width/height shift (0.1), shear (0.1), zoom (0.2), horizontal flip, fill mode `nearest`.
- **Training**:
  - **Initial Training**: Frozen ResNet50 layers, Adam optimizer (learning rate 0.0001), categorical crossentropy loss, batch size 32, 25 epochs.
  - **Callbacks**: EarlyStopping (monitor `val_loss`, patience=5), ModelCheckpoint (save best model as `best_model.h5`), ReduceLROnPlateau (factor=0.5, patience=2).
  - **Fine-Tuning**: Unfreeze top 10 layers, recompile with Adam (learning rate 1e-6), train for 10 additional epochs.
- **Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix on the test set.

## Files

- `animal_classifier.ipynb`: Jupyter Notebook with the complete workflow (data loading, preprocessing, model building, training, fine-tuning, evaluation, and visualization).
- `final_model.h5`: Trained ResNet50 model file (if uploaded; for large files, see Notes).
- `training_history.csv`: CSV file with training and validation loss/accuracy per epoch.
- `training_plots.png`: Plots of training and validation loss/accuracy curves.
- `.gitignore`: Excludes large files (e.g., `Animals/` dataset, `__pycache__/`) to keep the repository clean.

## Requirements

To run the project, install the following dependencies:

```bash
pip install tensorflow keras matplotlib numpy pandas scikit-learn kaggle seaborn
```

## How to Run

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/rathishTharusha/animal-image-classification-v1.git
   cd animal-image-classification-v1
   ```
2. **Obtain the Dataset**:
   - Follow the instructions in Obtaining the Dataset to download and place the `Animals` folder in the project directory.
3. **Set Up the Test Set**:
   - Run the dataset splitting code in `animal_classifier.ipynb` to create a `test` directory (10% of images).
4. **Run the Notebook**:
   - Open `animal_classifier.ipynb` in Jupyter Notebook.
   - Execute all cells to preprocess data, train the model, fine-tune, and generate results.
5. **View Results**:
   - Check `training_history.csv` for metrics.
   - View `training_plots.png` for loss/accuracy curves.
   - See the notebook’s confusion matrix and sample predictions (2x5 grid of test images with true/predicted labels).

## Results

- **Test Accuracy**: \~94% based on validation accuracy of 0.9444 in epoch 16
- **Metrics**: 
              precision    recall  f1-score   support

        cats       0.97      0.93      0.95       150
        dogs       0.95      0.97      0.96       150
      snakes       0.97      0.99      0.98       150

    accuracy                           0.96       450
   macro avg       0.96      0.96      0.96       450
weighted avg       0.96      0.96      0.96       450
- **Visualizations**:
  ![Training Graph](image.png)
  ![Confusion Matrix](image-1.png)

## Notes

- **Model Format**: The model is saved as `final_model.h5` (HDF5 format). Note the warning in the notebook about the legacy HDF5 format; consider saving as `final_model.keras` for compatibility (`model.save('final_model.keras')`).
- **Large Files**: If `final_model.h5` exceeds GitHub’s 25MB limit, it may be hosted on Google Drive or use Git LFS. Check the repository for a link or instructions.
- **Reproducibility**: Random seeds (`np.random.seed(42)`, `tf.random.set_seed(42)`) ensure consistent results.
- **Class Imbalance**: If the dataset is imbalanced, class weights may be applied (check notebook for implementation).

## Future Improvements

- Fine-tune additional ResNet50 layers for improved accuracy.
- Experiment with other architectures like EfficientNet or MobileNet.
- Deploy the model as a web app using Flask or Streamlit.
- Add data cleaning to handle corrupted images or outliers.

## Citations

- Dataset: Animal Image Classification Dataset
- ResNet50: Keras Applications
- Kaggle API: Kaggle API Documentation