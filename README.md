**Animal Classifier**
---
![alt text](http://url/to/img.png](https://github.com/movesen/animal-classification/blob/main/bird.png)
---
The repo contains:
- [train_CNN.py](https://github.com/movesen/animal-classification/blob/main/train_CNN.py) - Train CNN
- [streamlit_app.py](https://github.com/movesen/animal-classification/blob/main/streamlit_app.py) - Streamlit application backend
---
**Features**
- Custom Dataset: The code includes a custom dataset class that loads images from a directory structure organized by bird species and preprocesses them for training.
- Convolutional Neural Network: A CNN architecture is implemented using PyTorch, comprising convolutional layers, max-pooling layers, dropout layers, and fully connected layers.
- Training Pipeline: The code includes a training pipeline that trains the CNN model using the Adam optimizer and Cross Entropy Loss.
- Image Transformations: Input images undergo transformations such as resizing, normalization, and conversion to tensors before being fed into the CNN.
- Export Trained Model: Once trained, the model weights are saved to a file for future use.
---
**Requirements**
- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- tqdm
---

**Usage**
1. **Clone the repository:**
```
git clone https://github.com/yourusername/bird-species-classifier.git
```
3. **Install the required dependencies:**
```
pip install -r requirements.txt
```
4. **Prepare your dataset:** Organize your bird images into directories based on their species. Each directory should contain images of one bird species.
5. Adjust the dataset path and parameters in the code as needed.
6. **Run the training script:**
```
python train.py
```
7. Once training is complete, the trained model weights will be saved to a file specified in the code.
---

**License**
- This project is licensed under the MIT License.

