import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

import config
from model import ProtoNet
from features import extract_classical_features

class FishClassifier:
    def __init__(self, model_path="checkpoints/best_model.pth", prototypes_path="checkpoints/prototypes.pth"):
        self.device = config.DEVICE
        
        # Load checkpoint to get config
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint["config"]
        
        # Initialize model
        self.model = ProtoNet(
            backbone_name=model_config["backbone"],
            use_pretrained=False,
            embedding_dim=model_config["embedding_dim"],
            use_classical=model_config["use_classical"],
            classical_in_dim=model_config["classical_in_dim"],
            classical_proj_dim=model_config["classical_proj_dim"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        # Load or tell user to generate prototypes
        if not os.path.exists(prototypes_path):
            raise FileNotFoundError(f"Prototypes not found at {prototypes_path}. You need to compute and save them once from the training set.")
            
        proto_data = torch.load(prototypes_path, map_location=self.device)
        self.prototypes = proto_data["prototypes"]  # Shape: (5, Embedding_Dim)
        self.class_names = proto_data["class_names"] # List of 5 class names
        
        # Inference Transforms
        self.transform = v2.Compose([
            v2.Resize(config.IMAGE_SIZE),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    @torch.no_grad()
    def predict(self, image_path: str):
        # 1. Load and Transform Image
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 2. Extract Classical Features if needed
        classical_feats = None
        if self.model.use_classical:
            cf = extract_classical_features(image)
            classical_feats = torch.tensor(cf, dtype=torch.float32).unsqueeze(0).to(self.device)
            
        # 3. Get Model Embedding (Stage 2)
        embedding = self.model(img_tensor, classical_feats, stage=2)
        
        # 4. Compare to Prototypes using Cosine Similarity
        # Normalize vectors for Cosine Similarity
        emb_norm = F.normalize(embedding, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        # Calculate similarities (1 x 5)
        # Cosine similarity is in range [-1, 1]
        similarities = torch.matmul(emb_norm, proto_norm.t())
        
        # Absolute Max Similarity to any prototype
        max_raw_sim = torch.max(similarities).item()
        all_similarities = dict(zip(self.class_names, similarities.squeeze(0).tolist()))
        
        # Apply temperature scaling if it was used during training
        if self.model.temperature is not None:
            logits = similarities * self.model.temperature
        else:
            logits = similarities
            
        # 5. Get probabilities and predicted class
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_idx = torch.argmax(probs).item()
        
        return self.class_names[pred_idx], probs[pred_idx].item(), dict(zip(self.class_names, probs.tolist())), max_raw_sim, all_similarities

if __name__ == "__main__":
    print("Inference module ready. To use:")
    print("classifier = FishClassifier()")
    print("pred_class, confidence, all_probs, max_sim, all_sims = classifier.predict('path/to/image.jpg')")
