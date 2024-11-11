# Brain Tumor Segmentation using Deep Learning

This project focuses on brain tumor segmentation using deep learning techniques, utilizing various pre-trained models and concluding with Atrous Convolution U-Net, significantly improving segmentation accuracy.

## Pre-trained Models Used

The following pre-trained models were utilized during the segmentation process:

1. **VGG16**: A convolutional neural network used for feature extraction and classification tasks.
2. **ResNet50**: A deep residual network that effectively handles the vanishing gradient problem.
3. **InceptionV3**: Optimizes performance using factorized convolutions and auxiliary classifiers.
4. **DenseNet121**: A densely connected network that improves information flow between layers.
5. **Atrous Convolution U-Net**: Concluded with Atrous Convolution U-Net, which uses dilated convolutions to enhance segmentation performance by improving contextual understanding.

## Jupyter Notebook

The project is implemented using Jupyter Notebook and PyTorch as the deep learning framework. The notebook includes steps for data preprocessing, model training, evaluation, and visualization of results.

### How to Run the Project

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/brain-tumor-segmentation.git
    cd brain-tumor-segmentation
    ```

2. Install the required Python libraries using `pip`:

    ```bash
    pip install torch torchvision numpy matplotlib
    ```

    Or, install libraries directly in Jupyter Notebook:

    ```python
    !pip install torch torchvision numpy matplotlib
    ```

3. Open the Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

4. Run the cells in `brain_tumor_segmentation_pytorch.ipynb` to execute the segmentation process.

## Results

The segmentation accuracy was evaluated using performance metrics such as accuracy, precision, recall, and the Dice coefficient. Atrous Convolution U-Net provided a significant boost in segmentation accuracy.

## Dataset

The dataset used for training and testing is a standard brain tumor segmentation dataset. You can download it [https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection](#) (link to the dataset or provide download instructions).

## Conclusion

By leveraging various pre-trained models and concluding with Atrous Convolution U-Net, the project achieved higher segmentation accuracy compared to traditional models.

