# DINOv3 - Vision Foundation Models for FiftyOne

This repository provides DINOv3 models for the FiftyOne model zoo, enabling powerful vision tasks without fine-tuning.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from dinov3 import download_model, load_model

# Download the DINOv3 ConvNeXt Small model
model_name = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
model_path = "./models/dinov3-convnext-small"

download_model(model_name, model_path)

# Load the model for different tasks
model = load_model(model_name, model_path, output_type="dense_features")
```

## 🎯 Supported Tasks

Based on the [Hugging Face documentation](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m), DINOv3 models can be used without fine-tuning for the following tasks:

### 1. Image Classification
- **k-NN classifiers on the class token**
- **Logistic regression classifiers on the class token**
- **Linear layers on class token + patch tokens**

```python
# Load model for classification
model = load_model(model_name, model_path, output_type="dense_features")

# Extract features
embeddings = model.predict(images)

# For dense features, you get both summary and spatial
if isinstance(embeddings, tuple):
    summary_embeddings, spatial_embeddings = embeddings
else:
    summary_embeddings = embeddings

# Use summary_embeddings for classification tasks
```

### 2. Image Retrieval
- **Nearest neighbor search using dense features**

```python
# Load model for retrieval
model = load_model(model_name, model_path, output_type="retrieval")

# Extract retrieval embeddings
embeddings = model.predict(images)

# Use for similarity search and retrieval
```

### 3. Geometric and Semantic 3D Keypoint Correspondences
- **Feature matching between images**

```python
# Load model for correspondence
model = load_model(model_name, model_path, output_type="correspondence")

# Extract correspondence features
correspondence_features = model.predict([img1, img2])

# Use for finding matching points between images
```

### 4. Depth Estimation
- **Monocular depth estimation using linear layers**

```python
# Load model for depth estimation
model = load_model(model_name, model_path, output_type="depth")

# Extract depth features
depth_features = model.predict(images)

# Process into depth maps
```

### 5. Semantic Segmentation
- **Unsupervised object discovery and segmentation**

```python
# Load model for segmentation
model = load_model(model_name, model_path, output_type="segmentation")

# Extract segmentation features
segmentation_features = model.predict(images)

# Process into segmentation masks
```

### 6. Attention Maps
- **Self-attention visualization**

```python
# Load model for attention maps
model = load_model(model_name, model_path, output_type="attention_maps")

# Extract attention features
attention_features = model.predict(images)

# Process into attention heatmaps
```

## 🏗️ Model Architecture

The DINOv3 ConvNeXt Small model:
- **Parameters**: 50M
- **Architecture**: ConvNeXt-Small
- **Patch Size**: 16x16
- **Input Resolution**: 224x224 (minimum), multiples of 16
- **Hidden Dimensions**: 768
- **Training**: Self-distillation from ViT-7B on LVD-1689M dataset

## 📊 Performance

According to the [Hugging Face model card](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m):

| Task | Performance |
|------|-------------|
| ImageNet-ReaL | 87.9% |
| ImageNet-R | 88.7% |
| ObjectNet | 73.7% |
| ADE20k | 52.6% |
| NYU Depth | 0.432 |

## 🔧 Configuration Options

### Output Types

- `"dense_features"`: Both summary and spatial features (default)
- `"attention_maps"`: Spatial attention features
- `"segmentation"`: Spatial features for segmentation
- `"depth"`: Spatial features for depth estimation
- `"correspondence"`: Both summary and spatial for matching
- `"retrieval"`: Summary features for retrieval

### Feature Format

- `"NCHW"`: Computer vision format [batch, channels, height, width]
- `"NLC"`: Transformer sequence format [batch, length, channels]

### Feature Layers

- `"last"`: Final transformer layer only
- `"intermediate"`: Selected intermediate layers [10, 20, 30, 40]
- `"all"`: All transformer layers

## 📝 Example: Complete Classification Pipeline

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from dinov3 import download_model, load_model

# 1. Download and load model
model_name = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
model_path = "./models/dinov3-convnext-small"

download_model(model_name, model_path)
model = load_model(model_name, model_path, output_type="dense_features")

# 2. Prepare training data
train_images = [...]  # Your training images
train_labels = [...]  # Your training labels

# 3. Extract features
train_embeddings = model.predict(train_images)

# Get summary features (class token)
if isinstance(train_embeddings, tuple):
    summary_embeddings = train_embeddings[0]
else:
    summary_embeddings = train_embeddings

# 4. Train classifier
X = np.array(summary_embeddings)
y = np.array(train_labels)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# 5. Make predictions
test_image = [...]  # Your test image
test_embedding = model.predict([test_image])

if isinstance(test_embedding, tuple):
    test_summary = test_embedding[0][0]
else:
    test_summary = test_embedding[0]

prediction = knn.predict([test_summary])
print(f"Predicted class: {prediction[0]}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_dinov3_convnext.py
```

This will test all supported tasks and demonstrate the model's capabilities.

## 🔗 References

- [DINOv3 Paper](https://arxiv.org/abs/2508.10104)
- [Hugging Face Model](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m)
- [Transformers Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/dinov3)
- [FiftyOne Model Zoo](https://docs.voxel51.com/model_zoo/remote.html)

## 📄 License

This project is licensed under the DINOv3 License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.