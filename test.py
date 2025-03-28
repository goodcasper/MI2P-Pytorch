from model import MultimodalDenseNet
import torch

# Create dummy input data
images = torch.randn(8, 3, 224, 224)         # A batch of images with size 224x224
input_ids = torch.randint(0, 30522, (8, 128))  # Simulated BERT token IDs
attention_mask = torch.ones_like(input_ids)   # Attention mask (typically 0 or 1)

# Initialize the model and run a forward pass
model = MultimodalDenseNet()
outputs = model(images, input_ids, attention_mask)

print(outputs.shape)  # Output classification results; default is 101 classes (e.g., Food101)