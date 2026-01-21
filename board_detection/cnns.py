
import torch.nn as nn
import torch.nn.functional as F
import torch
class SpatialAttention(nn.Module):
    """Learn which spatial locations matter"""
    def __init__(self, channels, temperature=1.0):
        super().__init__()
        print("attention temp",temperature)
        self.temperature = temperature
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # 1 attention map
            #nn.Sigmoid()                       # Values 0-1
        )
    
    def forward(self, x):
        # x: [batch, 128, 10, 10]
        logits = self.conv(x)
        attention = torch.sigmoid(logits / self.temperature) 
        attended = x * attention      # Weighted features
        return attended, attention    # Return attention for visualization!

class EmptyNetMultiTask(nn.Module):
    def __init__(self, in_channels=3, attention_temp=0.5,N=4, embedding_dim=64):
        super().__init__()
        
        # Shared feature extractor (same as before)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Shared attention
        self.attention = SpatialAttention(128, attention_temp)
        
        # Shared embedding (features → compact representation)
        self.embed = nn.Sequential(
            nn.AdaptiveAvgPool2d((N, N)),
            nn.Flatten(),
            #nn.Dropout(0.4),
            nn.Linear(128 * N * N, embedding_dim),  # Shared 128-dim embedding
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # Task-specific heads
        self.head_empty = nn.Sequential(
            #nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1)
        )
        
        self.head_white = nn.Sequential(
            #nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1)
        )
        self.head_pattern = nn.Sequential(
            #nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1)
        )
    def forward(self, x, return_attention=False):
        features = self.features(x)
        attended, attention_map = self.attention(features)
        
        # Shared embedding
        embedding = self.embed(attended)
        emb_norm = F.normalize(embedding, p=2, dim=1)
        # Task outputs
        out_empty = self.head_empty(emb_norm)
        out_white = self.head_white(emb_norm)
        out_pattern = self.head_pattern(emb_norm)

        outputs = {
            "empty": out_empty,
            "white": out_white,
            "pattern":out_pattern
        }
        
        if return_attention:
            return outputs, attention_map
        return outputs
    
    def get_embedding(self, x):
        """Extract embedding for downstream tasks"""
        features = self.features(x)
        attended, _ = self.attention(features)
        embedding = self.embed(attended)
        # L2 normalize for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def compare(self, x1, x2):
        """Return similarity score between two patches"""
        emb1 = self.get_embedding(x1)
        emb2 = self.get_embedding(x2)
        # Cosine similarity (already normalized)
        similarity = (emb1 * emb2).sum(dim=1)
        return similarity  # Range: [-1, 1], higher = more similar


# cnn on a single task
class EmptyNetAttention(nn.Module):
    def __init__(self, in_channels=3,N=4,embedding_dim=64,attention_temp=1.0):
        super().__init__()
        self.N = N
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Spatial attention - learns WHERE to look
        self.attention = SpatialAttention(128,attention_temp)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((N, N)),  # Now pools ATTENDED features
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128*N*N, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.embedding_dim, 1)
        )
    
    def forward(self, x, return_attention=False):
        features = self.features(x)
        attended, attention_map = self.attention(features)
        output = self.classifier(attended)
        
        if return_attention:
            return output, attention_map  # For visualization
        return output