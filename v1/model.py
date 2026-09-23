import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import config


class ProtoNet(nn.Module):
    """
    Prototypical Network for Few-Shot Learning.
    Supports ResNet-18, EfficientNet-B0, and ConvNeXt-Tiny backbones.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        use_pretrained: bool = True,
        embedding_dim: int = 256,
        use_classical: bool = False,
        classical_in_dim: int = 6110,  # Default dimension for HOG+LBP features
        classical_proj_dim: int = 128
    ):
        super(ProtoNet, self).__init__()
        
        if getattr(config, 'USE_COSINE_DISTANCE', False):
            self.temperature = nn.Parameter(torch.tensor(config.COSINE_TEMP_INIT))
        else:
            self.temperature = None
        self.backbone_name = backbone_name
        self.use_classical = use_classical
        
        # 1. Initialize CNN backbone
        if backbone_name == "resnet18":
            if use_pretrained:
                weights = models.ResNet18_Weights.DEFAULT
            else:
                weights = None
            resnet = models.resnet18(weights=weights)
            
            # Remove classification head (fc layer)
            self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
            cnn_feature_dim = 512
            
        elif backbone_name == "efficientnet_b0":
            if use_pretrained:
                weights = models.EfficientNet_B0_Weights.DEFAULT
            else:
                weights = None
            effnet = models.efficientnet_b0(weights=weights)
            
            # Keep feature extractor and average pool
            self.backbone = effnet.features
            self.avgpool = effnet.avgpool
            cnn_feature_dim = 1280
            
        elif backbone_name == "convnext_tiny":
            if use_pretrained:
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT
            else:
                weights = None
            convnext = models.convnext_tiny(weights=weights)
            
            self.backbone = convnext.features
            self.avgpool = convnext.avgpool
            cnn_feature_dim = 768
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # 2. Linear projection for deep CNN embeddings
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 3. Handle classical features projection
        self.final_embedding_dim = embedding_dim
        if self.use_classical:
            self.classical_projection = nn.Sequential(
                nn.Linear(classical_in_dim, classical_proj_dim),
                nn.BatchNorm1d(classical_proj_dim),
                nn.ReLU(),
                nn.Linear(classical_proj_dim, classical_proj_dim)
            )
            # Final feature dimension is concatenated CNN + Classical
            self.final_embedding_dim = embedding_dim + classical_proj_dim
            
        # 4. Stage 1 Pre-training Classifier Head
        self.classifier_head = nn.Linear(self.final_embedding_dim, config.NUM_CLASSES)

    def forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and project deep features from input images.
        """
        # Feature extraction
        x = self.backbone(x)
        
        if self.backbone_name in ["efficientnet_b0", "convnext_tiny"]:
            x = self.avgpool(x)
            
        # Flatten: (B, C, 1, 1) -> (B, C)
        x = torch.flatten(x, 1)
        
        # Project to embedding space
        return self.cnn_projection(x)

    def forward(self, x_images: torch.Tensor, x_classical: torch.Tensor = None, stage: int = 2) -> torch.Tensor:
        """
        Forward pass. Extracts embeddings. If stage=1, returns logits from the classifier head.
        If stage=2, returns the raw embeddings. Processes images in chunks to save VRAM.
        """
        # Sequential embedding chunking to prevent CUDA OOM on large 40-way batches
        chunk_size = 16
        cnn_embeddings_list = []
        for i in range(0, x_images.size(0), chunk_size):
            chunk = x_images[i:i+chunk_size]
            cnn_embeddings_list.append(self.forward_cnn(chunk))
            
        cnn_embeddings = torch.cat(cnn_embeddings_list, dim=0)
        
        if self.use_classical and x_classical is not None:
            classical_embeddings = self.classical_projection(x_classical)
            combined_embeddings = torch.cat([cnn_embeddings, classical_embeddings], dim=-1)
        else:
            combined_embeddings = cnn_embeddings
            
        if stage == 1:
            return self.classifier_head(combined_embeddings)
        return combined_embeddings


def compute_prototypes_and_logits(
    embeddings: torch.Tensor,
    n_way: int,
    k_shot: int,
    q_query: int,
    temperature: torch.Tensor = None
):
    """
    Computes class prototypes from support embeddings and calculates
    logits (scaled cosine similarity or negative squared Euclidean distances).
    
    Args:
        embeddings: Tensor of shape (N_way * (K_shot + Q_query), D)
        n_way: Number of classes in the episode
        k_shot: Number of support items per class
        q_query: Number of query items per class
        
    Returns:
        logits: Tensor of shape (N_way * Q_query, N_way)
        targets: Tensor of shape (N_way * Q_query)
    """
    D = embeddings.size(-1)
    
    # 1. Reshape embeddings into (N_way, K_shot + Q_query, D)
    embeddings = embeddings.view(n_way, k_shot + q_query, D)
    
    # 2. Split into support set and query set
    support_set = embeddings[:, :k_shot, :]  # Shape: (N_way, K_shot, D)
    query_set = embeddings[:, k_shot:, :]    # Shape: (N_way, Q_query, D)
    
    # 3. Compute class prototypes (mean of support set along K dimension)
    prototypes = support_set.mean(dim=1)    # Shape: (N_way, D)
    
    # 4. Flatten query set to (N_way * Q_query, D)
    queries = query_set.reshape(n_way * q_query, D)
    
    # 6. Generate target indices (which prototype it belongs to)
    targets = torch.arange(n_way, device=embeddings.device).repeat_interleave(q_query)
    
    if temperature is not None:
        # Cosine Similarity metric
        queries_norm = F.normalize(queries, p=2, dim=-1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        similarities = torch.matmul(queries_norm, prototypes_norm.t())
        logits = similarities * temperature
    else:
        # Negative Euclidean Distance metric
        q_expanded = queries.unsqueeze(1)
        p_expanded = prototypes.unsqueeze(0)
        distances = torch.sum((q_expanded - p_expanded) ** 2, dim=-1)
        logits = -distances
    
    return logits, targets


def compute_protonet_loss(logits: torch.Tensor, targets: torch.Tensor):
    """
    Computes Prototypical Network cross-entropy loss and accuracy.
    """
    loss = F.cross_entropy(logits, targets)
    
    # Calculate accuracy
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == targets).float().mean()
    
    return loss, acc


if __name__ == "__main__":
    # Test model forward pass
    net = ProtoNet(
        backbone_name="resnet18",
        use_pretrained=False,
        use_classical=True,
        classical_in_dim=100
    )
    
    # Simulate a batch of 5-way 5-shot 5-query: 5 * (5 + 5) = 50 samples
    dummy_imgs = torch.randn(50, 3, 224, 224)
    dummy_feats = torch.randn(50, 100)
    
    # Stage 1 test
    logits_stage1 = net(dummy_imgs, dummy_feats, stage=1)
    print(f"Stage 1 Logits shape: {logits_stage1.shape}") # Should be (50, 40)
    
    # Stage 2 test
    embs = net(dummy_imgs, dummy_feats, stage=2)
    print(f"Stage 2 Embeddings shape: {embs.shape}")  # Should be (50, 256 + 128 = 384)
    
    logits, tgts = compute_prototypes_and_logits(embs, n_way=5, k_shot=5, q_query=5)
    print(f"Logits shape: {logits.shape}")   # Should be (25, 5)
    print(f"Targets shape: {tgts.shape}")       # Should be (25,)
    
    loss, acc = compute_protonet_loss(logits, tgts)
    print(f"Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")
