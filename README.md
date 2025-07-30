# Animal Image Classification

This project builds an image classification model to distinguish between dogs, cats, and snakes using ResNet50 with transfer learning in Keras. The model is trained on the Animal Image Classification Dataset from Kaggle, leveraging pre-trained weights from ImageNet to achieve robust performance. The project includes data preprocessing, model training with data augmentation, evaluation with multiple metrics, and visualizations, meeting the requirements for a professional-level machine learning project.

## Dataset
- **Source**: [Animal Image Classification Dataset](https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset)
- **Structure**: Images are organized in the `Animals` folder with subfolders `dogs`, `cats`, and `snakes`.
- **Splitting**: The dataset is split into training (80%), validation (10%), and test (10%) sets using `ImageDataGenerator` with `validation_split` for train/validation and a manual split for the test set to ensure unbiased evaluation.

## Model
- **Architecture**: 
  - Base: ResNet50 pre-trained on ImageNet, with the top layers removed.
  - Custom Layers: GlobalAveragePooling2D, Dense (128 units, ReLU), Dropout (0.5), Dense (3 units, softmax).
- **Training**:
  - Initial training with frozen ResNet50 layers using Adam optimizer and categorical crossentropy loss.
  - Fine-tuning by unfreezing the last 10 layers with a lower learning rate (1e-5).
  - Data augmentation: rotation, width/height shift, shear, zoom, horizontal flip.
  - Early stopping based on validation loss (patience=5).
- **Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix on the test set.

## Files
- `animal_classifier.ipynb`: Jupyter Notebook containing the complete workflow (data loading, preprocessing, model building, training, evaluation, and visualization).
- `animal_classifier.h5`: Trained ResNet50 model file (if uploaded; for large files, see [Notes](#notes)).
- `training_history.csv`: CSV file with training and validation loss/accuracy per epoch.
- `training_plots.png`: Plots of training and validation loss/accuracy curves.
- `.gitignore`: Excludes large files (e.g., `Animals/` dataset, `__pycache__/`) to keep the repository clean.

## Requirements
To run the project, install the following dependencies:
```bash
pip install tensorflow keras matplotlib numpy pandas scikit-learn
```

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rathishTharusha/animal-image-classification-v1.git
   cd animal-image-classification-v1
   ```
2. **Download the Dataset**:
   - Download from [Kaggle](https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset).
   - Place the dataset in an `Animals` folder with subfolders `dogs`, `cats`, and `snakes`.
3. **Set Up the Test Set**:
   - Run the dataset splitting code in `animal_classifier.ipynb` to create a `test` directory (10% of images).
4. **Run the Notebook**:
   - Open `animal_classifier.ipynb` in Jupyter Notebook.
   - Execute all cells to preprocess data, train the model, and generate results.
5. **View Results**:
   - Check `training_history.csv` for metrics.
   - View `training_plots.png` for loss/accuracy curves.
   - See the notebook’s confusion matrix and sample predictions.

## Results
- **Test Accuracy**: [Insert accuracy from your evaluation, e.g., 92%]
- **Metrics**: [Insert precision, recall, F1-score from classification report]
- **Visualizations**: 
  - Loss and accuracy plots in `training_plots.png`.
  - Confusion matrix and sample predictions in `animal_classifier.ipynb`.

## Notes
- **Large Files**: If `animal_classifier.h5` exceeds GitHub’s 25MB limit, it may be hosted on [Google Drive link] or use Git LFS. Check the repository for a link or instructions.
- **Reproducibility**: Random seeds (`np.random.seed(42)`, `tf.random.set_seed(42)`) ensure consistent results.
- **Class Imbalance**: If the dataset is imbalanced, class weights are used during training (see notebook).

## Future Improvements
- Fine-tune additional ResNet50 layers for better accuracy.
- Experiment with other architectures like EfficientNet or MobileNet.
- Deploy the model as a web app using Flask or Streamlit.
- Add data cleaning to handle corrupted images or outliers.

## Citations
- Dataset: [Animal Image Classification Dataset](https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset)
- ResNet50: [Keras Applications](https://keras.io/api/applications/resnet/#resnet50-function)
