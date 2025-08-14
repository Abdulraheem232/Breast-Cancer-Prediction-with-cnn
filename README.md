
---

```markdown
# Breast-Cancer-Prediction-with-cnn

This repository implements a **Convolutional Neural Network (CNN)** for breast cancer MRI/ultrasound prediction.  
It was developed as part of the research project:  
**"Vision Transformer vs. CNN: A Comparative Study for MRI Scans."**

The main file is:
- **`Breast_cancer_detection_with_cnn.ipynb`** — a Jupyter Notebook implementing data loading, preprocessing, CNN training, and evaluation.

---

##  Dataset

This project uses the **[Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)** by *Arya Shah* (Kaggle).  

The dataset contains three classes of ultrasound images:
- **Benign**  
- **Malignant**  
- **Normal**  

### Download Instructions
1. Go to the dataset page:  
   [Breast Ultrasound Images Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
2. Download the dataset.
3. Extract it into a folder (for example `data/`), so that the structure looks like:
```

data/
├── benign/
├── malignant/
└── normal/

````
4. Update the dataset path in the notebook if needed.

---

##  Project Overview

The project evaluates CNN performance for breast cancer detection and serves as a baseline for comparison with Vision Transformer (ViT) models.

**Key objectives:**
- Preprocess ultrasound images for classification.
- Train a CNN on the dataset.
- Evaluate model performance with accuracy and loss.
- Provide a baseline for comparison with Vision Transformers.

---

##  Repository Contents

- `Breast_cancer_detection_with_cnn.ipynb`  
- Data loading and preprocessing  
- CNN architecture definition  
- Training and validation  
- Evaluation and performance metrics  

---

##  Getting Started

### Prerequisites

- Python 3.8+  
- Jupyter Notebook  
- Install dependencies:
```bash
pip install torch torchvision numpy matplotlib
````

### Running the Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/Abdulraheem232/Breast-Cancer-Prediction-with-cnn.git
   cd Breast-Cancer-Prediction-with-cnn
   ```

2. Download and prepare the dataset as described above.

3. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open `Breast_cancer_detection_with_cnn.ipynb` and run the cells step by step.

---

## Project Structure

```
Breast-Cancer-Prediction-with-cnn/
├── Breast_cancer_detection_with_cnn.ipynb   Notebook implementing CNN
├── data/                                    (User-provided dataset)
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── README.md
```

---

## Model Details

* **Architecture**: Custom CNN with convolutional, pooling, and fully connected layers.
* **Input**: Preprocessed grayscale ultrasound images (resized to `224×224`).
* **Output**: 3-class classification (benign, malignant, normal).
* **Evaluation**: Accuracy, loss, and visualization of results.

---

## Research Context

This CNN serves as a **baseline** for the research study:
**"Vision Transformer vs. CNN: A Comparative Study for MRI Scans."**

Next steps for researchers:

1. Implement a Vision Transformer (ViT) model for the same dataset.
2. Compare CNN and ViT results in terms of:

   * Accuracy
   * Training efficiency
   * Robustness to complex cases

---

## Citation

If you use this repository, please cite both this repo and the dataset:

* Arya Shah, *Breast Ultrasound Images Dataset*, Kaggle (2020).
  [Link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

---

## Contact

For questions or collaboration, please reach out via GitHub Issues or Discussions.

---

```

---

```
