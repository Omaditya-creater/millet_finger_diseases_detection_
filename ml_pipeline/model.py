import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderBlock(nn.Module):
    """
    Vision Transformer Encoder Layer with Multi-Head Self-Attention (MHSA)
    and Feed-Forward Network (FFN).
    """
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class HybridEfficientNetViT(nn.Module):
    """
    Hybrid Deep Learning Architecture combining CNN local feature extraction 
    with Vision Transformer (ViT) spatial attention for Finger Millet Disease Detection.
    """
    def __init__(self, num_classes=5, in_channels=3, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        
        # 1. Local Convolutional Backbone (EfficientNet-style Stem & Stage Extractor)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False), # 112x112
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),         # 56x56
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),  # 28x28
            nn.BatchNorm2d(embed_dim),
            nn.SiLU()
        )
        
        # 2. Spatial Projection to Transformer Tokens
        # Feature map size: 28x28 = 784 spatial tokens
        self.seq_len = 28 * 28
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
        
        # 3. Vision Transformer Encoder Layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. Classification Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Save feature maps for Grad-CAM inspection
        self.last_conv_features = None
        self.last_conv_gradients = None

    def activations_hook(self, grad):
        self.last_conv_gradients = grad

    def forward(self, x):
        # Local CNN Feature Extraction
        feat_map = self.stem(x)  # (B, embed_dim, 28, 28)
        
        # Register hook on target convolutional feature map for Grad-CAM
        if feat_map.requires_grad:
            h = feat_map.register_hook(self.activations_hook)
        self.last_conv_features = feat_map
        
        B, C, H, W = feat_map.shape
        # Flatten spatial dimensions to sequence tokens: (B, H*W, C)
        tokens = feat_map.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
        # Prepend Class Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        
        # Add Positional Embeddings
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)
        
        # Transformer Multi-Head Self-Attention Pass
        for layer in self.transformer_layers:
            tokens = layer(tokens)
            
        tokens = self.norm(tokens)
        
        # Classification from CLS token
        cls_out = tokens[:, 0]
        out = self.head(cls_out)
        return out

def build_model(num_classes=5):
    """Factory function to build the hybrid model."""
    model = HybridEfficientNetViT(num_classes=num_classes)
    return model

if __name__ == "__main__":
    model = build_model()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Hybrid EfficientNet-ViT Output Shape: {out.shape} (Expected: [2, 5])")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
