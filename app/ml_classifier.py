import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Tuple
from .settings import settings

class AlgorithmClassifier:
    """CodeBERT + RandomForest algorithm classifier"""
    
    def __init__(self):
        self.tokenizer = None
        self.codebert_model = None
        self.rf_classifier = None
        self.label_map = None
        self.loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load CodeBERT and trained Random Forest model"""
        if self.loaded:
            return
            
        try:
            model_dir = settings.codebert_model_path
            
            # Load CodeBERT
            if os.path.exists(os.path.join(model_dir, 'codebert')):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    os.path.join(model_dir, 'codebert')
                )
                self.codebert_model = AutoModel.from_pretrained(
                    os.path.join(model_dir, 'codebert')
                )
            else:
                # Fallback to HuggingFace
                print("⚠ Loading CodeBERT from HuggingFace (first time may be slow)")
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                self.codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
            
            self.codebert_model.to(self.device)
            self.codebert_model.eval()
            
            # Load Random Forest classifier
            rf_path = os.path.join(model_dir, 'algorithm_classifier.pkl')
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.rf_classifier = pickle.load(f)
            else:
                print("⚠ Random Forest model not found, using fallback")
                self.rf_classifier = None
            
            # Load label map
            label_map_path = os.path.join(model_dir, 'label_map.pkl')
            if os.path.exists(label_map_path):
                with open(label_map_path, 'rb') as f:
                    self.label_map = pickle.load(f)
            else:
                # Default labels for your 12 patterns
                self.label_map = {
                    0: "Sorting",
                    1: "Searching",
                    2: "Recursion",
                    3: "Dynamic Programming",
                    4: "Greedy Algorithms"
                }
            
            self.loaded = True
            print("✓ CodeBERT + RandomForest model loaded successfully")
            print(f"  Device: {self.device}")
            print(f"  Labels: {list(self.label_map.values())}")
            
        except Exception as e:
            print(f"⚠ Could not load models: {e}")
            print("  Using fallback heuristic classifier")
    
    def get_code_embedding(self, code: str) -> np.ndarray:
        """Convert code to CodeBERT embedding"""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.codebert_model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            return embedding.cpu().numpy()
    
    def predict(self, code: str) -> Tuple[str, float]:
        """Predict algorithm label and confidence"""
        if not self.loaded or self.rf_classifier is None:
            return self._fallback_predict(code)
        
        try:
            # Get CodeBERT embedding
            embedding = self.get_code_embedding(code)
            
            # Predict with Random Forest
            pred_label = self.rf_classifier.predict([embedding])[0]
            confidence = float(np.max(self.rf_classifier.predict_proba([embedding])))
            
            # Get label name
            label_name = self.label_map.get(pred_label, "Unknown Algorithm")
            
            # Hybrid approach: if ML confidence is low, check keywords too
            if confidence < 0.60:
                fallback_label, fallback_conf = self._fallback_predict(code)
                # If fallback has specific detection and ML doesn't match, prefer fallback
                if fallback_conf > 0.50 and fallback_label not in ["General Algorithm", "Recursion"]:
                    # Boost confidence slightly for keyword match
                    return fallback_label, min(0.75, fallback_conf + 0.15)
            
            return label_name, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_predict(code)
    
    def _fallback_predict(self, code: str) -> Tuple[str, float]:
        """Enhanced keyword-based fallback with better pattern detection"""
        code_lower = code.lower()
        
        # Check for greedy algorithms (higher priority with more keywords)
        greedy_keywords = ["greedy", "activity_selection", "fractional_knapsack", 
                          "job_sequencing", "huffman", "interval"]
        greedy_count = sum(1 for keyword in greedy_keywords if keyword in code_lower)
        if greedy_count >= 2:
            return "Greedy Algorithms", 0.70
        
        # Check for sorting algorithms
        if any(word in code_lower for word in ["mergesort", "quicksort", "bubblesort", "merge_sort", "quick_sort"]):
            return "Sorting", 0.70
        
        # Check for searching algorithms
        if "binary" in code_lower and "search" in code_lower:
            return "Searching", 0.65
        
        # Check for dynamic programming
        dp_keywords = ["dp[", "memo", "memoization", "tabulation", "dynamic programming", "overlapping subproblem"]
        if any(keyword in code_lower for keyword in dp_keywords):
            return "Dynamic Programming", 0.65
        
        # Check for graph algorithms
        if any(word in code_lower for word in ["bfs", "dfs", "breadth", "depth", "graph", "adjacency"]):
            if "queue" in code_lower or "deque" in code_lower:
                return "bfs", 0.60
            elif "stack" in code_lower or code.count("def ") > 3:
                return "dfs", 0.60
        
        # Check for recursion pattern
        if code.count("def ") > 2:
            funcs = [line.split("def ")[1].split("(")[0] for line in code.split("\n") if "def " in line]
            for func in funcs:
                if func in code.split(f"def {func}")[1]:
                    return "Recursion", 0.55
        
        # Weak greedy signal
        if any(word in code_lower for word in ["greedy", "optimal", "choice"]):
            return "Greedy Algorithms", 0.50
        
        return "General Algorithm", 0.45

# Global classifier instance
classifier = AlgorithmClassifier()
