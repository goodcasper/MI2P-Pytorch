import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision.models import densenet121
from transformers import BertTokenizer

# Define a class called MI2P that inherits from nn.Module
class MI2P(nn.Module):
    def __init__(self, channel, textual_dim, attention_dim, h, w):  # textual_dim is always 768, attention_dim is always 128; channel, h, w are the channel, height, and width of the input feature map
        super(MI2P, self).__init__()

        self.dq = attention_dim
        self.dk = attention_dim
        self.dv = h * w
        self.query_conv = nn.Conv2d(channel, channel * self.dq, kernel_size=1, groups=channel)  # Input channel is `channel`, output is `channel*dq`
        # group=channel means the input channels are divided into `channel` groups, one group per channel. Each channel has `dq` kernels, so output channel is 128*dq.
        # This is also called depthwise separable convolution. Each input channel has its own kernel, which reduces the number of parameters and computation.
        # Depthwise convolution enforces each input channel to learn features independently, potentially increasing feature diversity and improving representation power.

        self.key_fc = nn.Linear(textual_dim, self.dk)  # Train a weight matrix to map textual features to attention space, forming the K matrix. Shape: nn.Linear(768,128)

        self.value_fc = nn.Linear(textual_dim, self.dv)  # Train a weight matrix to map textual features to visual space, forming the V matrix
        self.attention_dim = attention_dim

    def forward(self, visual_features, textual_features):
        batch_size, C, H, W = visual_features.size()  # Get shape of visual features

        L, d = textual_features.size(1), textual_features.size(2)  # Get shape of textual features

        # Generate queries via convolution and reshape
        queries = self.query_conv(visual_features)  # Output shape: (batch_size, c*dq, 28, 28). Each channel has its own dq kernels

        queries = F.avg_pool2d(queries, kernel_size=H)  # Output shape: (batch_size, c*dq, 1, 1). Apply average pooling over h*w feature map

        queries = queries.view(batch_size, C, self.dq, 1, 1)  # Shape: (batch_size, c, dq, 1, 1)

        queries = queries.squeeze(-1).squeeze(-1)  # Remove unnecessary dimensions -> (batch_size, c, dq)

        # Generate keys and values through linear layers
        keys = self.key_fc(textual_features)
        values = self.value_fc(textual_features)

        # Compute attention scores and apply softmax normalization
        attention_scores = torch.matmul(queries, keys.transpose(1, 2))  # Matrix multiplication between queries and transposed keys
        attention_scores = F.softmax(attention_scores / (self.attention_dim ** 0.5), dim=-1)  # Scale and apply softmax normalization
        # dim=-1 means softmax is applied on the last dimension (i.e., sequence_length)

        attended_textual_features = torch.matmul(attention_scores, values)  # Compute weighted sum of values using attention scores

        # attended_textual_features = attended_textual_features.permute(0, 2, 1) # permute is not necessary, but improves accuracy by 1.5% (not sure if it's just a coincidence)

        attended_textual_features = attended_textual_features.reshape(batch_size, C, H, W)

        # Add the weighted textual features to the original visual features
        # enhanced_visual_features = visual_features + attended_textual_features
        enhanced_visual_features = visual_features + attended_textual_features

        return enhanced_visual_features

# Define a class called MultimodalDenseNet that inherits from nn.Module
class MultimodalDenseNet(nn.Module):
    def __init__(self, pretrained=True, attention_dim=128):
        super(MultimodalDenseNet, self).__init__()
        # Load pre-trained DenseNet model
        self.visual_model = densenet121(pretrained=pretrained)
        # Load pre-trained BERT model
        self.textual_model = BertModel.from_pretrained('bert-base-uncased')

        # Initialize initial blocks from DenseNet model
        self.initial_conv = self.visual_model.features.conv0
        self.initial_norm = self.visual_model.features.norm0
        self.initial_relu = self.visual_model.features.relu0
        self.initial_pool = self.visual_model.features.pool0

        # Initialize DenseNet modules
        self.dense_block1 = self.visual_model.features.denseblock1
        self.transition1 = self.visual_model.features.transition1
        self.dense_block2 = self.visual_model.features.denseblock2
        self.transition2 = self.visual_model.features.transition2
        self.dense_block3 = self.visual_model.features.denseblock3
        self.transition3 = self.visual_model.features.transition3
        self.dense_block4 = self.visual_model.features.denseblock4
        self.final_bn = self.visual_model.features.norm5

        # Initialize three MI2P modules corresponding to three different DenseNet stages
        self.mi2p1 = MI2P(128, 768, attention_dim, 28, 28)  # Output channels of denseblock1 = 128
        self.mi2p2 = MI2P(256, 768, attention_dim, 14, 14)  # Output channels of denseblock2 = 256
        self.mi2p3 = MI2P(512, 768, attention_dim, 7, 7)    # Output channels of denseblock3 = 512
        self.mi2p4 = MI2P(1024, 768, attention_dim, 7, 7)   # Output channels of denseblock4 = 1024

        # Define classifier
        self.classifier = nn.Linear(1024, 101)

    def forward(self, images, input_ids, attention_mask):
        # Get textual features from BERT
        text_features = self.textual_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # Use the last hidden state as text features

        # Process visual input through initial DenseNet blocks
        visual_features = self.initial_relu(self.initial_norm(self.initial_conv(images)))
        visual_features = self.initial_pool(visual_features)

        # Pass through the first DenseNet block and fuse with text features
        visual_features = self.dense_block1(visual_features)
        visual_features = self.transition1(visual_features)
        visual_features = self.mi2p1(visual_features, text_features)

        # Pass through the second DenseNet block and fuse with text features
        visual_features = self.dense_block2(visual_features)
        visual_features = self.transition2(visual_features)
        visual_features = self.mi2p2(visual_features, text_features)

        # Pass through the third DenseNet block and fuse with text features
        visual_features = self.dense_block3(visual_features)
        visual_features = self.transition3(visual_features)
        visual_features = self.mi2p3(visual_features, text_features)

        # Pass through the fourth DenseNet block and final batch norm
        visual_features = self.dense_block4(visual_features)
        visual_features = self.mi2p4(visual_features, text_features)
        visual_features = self.final_bn(visual_features)

        # Apply adaptive average pooling and flatten features
        visual_features = F.adaptive_avg_pool2d(visual_features, (1, 1)).view(visual_features.size(0), -1)  # Adaptive average pooling to reduce feature map to a single value per channel, then flatten for classifier

        # Feed into classifier
        outputs = self.classifier(visual_features)

        return outputs
