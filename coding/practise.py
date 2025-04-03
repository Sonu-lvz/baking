import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time

# Ensure computations are done on CPU for better compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IngredientWeightPredictor(nn.Module):
    def _init_(self, num_ingredients, num_units):
        super(IngredientWeightPredictor, self)._init_()
        self.ingredient_embedding = nn.Embedding(num_ingredients, 16)
        self.unit_embedding = nn.Embedding(num_units, 8)
        self.fc = nn.Sequential(
            nn.Linear(16 + 8 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, ingredient_idx, unit_idx, amount):
        ingredient_emb = self.ingredient_embedding(ingredient_idx)
        unit_emb = self.unit_embedding(unit_idx)
        x = torch.cat([ingredient_emb, unit_emb, amount.unsqueeze(1)], dim=1)
        return self.fc(x)

class CameraIngredientDetector:
    def _init_(self):
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device)
        self.model.eval()
        self.ingredient_classes = ['flour', 'sugar', 'salt', 'butter', 'milk', 'egg']
        self.unit_classes = ['cup', 'tbsp', 'tsp', 'piece']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Smaller image size for faster inference
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Reference object width in cm (adjust for better accuracy)
        self.ref_object_width = 2.0  
    
    def detect_ingredient(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blurred, 50, 100)  # Reduced thresholds for better edge detection
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pixels_per_cm = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pixels_per_cm = w / self.ref_object_width

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = self.transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self.model(img)

        _, pred = torch.max(outputs, 1)
        pred_class = self.ingredient_classes[pred.item() % len(self.ingredient_classes)]
        
        volume = 1.0  # Default volume estimation
        if pixels_per_cm:
            volume = (w * h) / (100 * pixels_per_cm**2) 
        
        return pred_class, volume

def create_synthetic_dataset():
    ingredients = ['flour', 'sugar', 'salt', 'butter', 'milk', 'egg']
    units = ['cup', 'tbsp', 'tsp', 'piece']
    
    densities = {'flour': 120, 'sugar': 200, 'salt': 300, 'butter': 240, 'milk': 240, 'egg': 50}
    unit_multipliers = {'cup': 1.0, 'tbsp': 0.0625, 'tsp': 0.0208, 'piece': 1.0}
    
    data = []
    for _ in range(500):  # Reduced dataset size for training efficiency
        ingredient = np.random.choice(ingredients)
        unit = np.random.choice(units)
        amount = np.random.uniform(0.1, 2.5)  # Lowered max amount for consistency
        weight = amount * densities[ingredient] * unit_multipliers.get(unit, 1.0)
        weight += np.random.normal(0, weight * 0.1)
        data.append({'ingredient': ingredient, 'unit': unit, 'amount': amount, 'weight': weight})
    return data, ingredients, units

def train_model(data, ingredients, units):
    ingredient_to_idx = {ing: i for i, ing in enumerate(ingredients)}
    unit_to_idx = {unit: i for i, unit in enumerate(units)}
    
    ingredient_indices = torch.tensor([ingredient_to_idx[d['ingredient']] for d in data], dtype=torch.long)
    unit_indices = torch.tensor([unit_to_idx[d['unit']] for d in data], dtype=torch.long)
    amounts = torch.tensor([d['amount'] for d in data], dtype=torch.float32)
    weights = torch.tensor([d['weight'] for d in data], dtype=torch.float32).view(-1, 1)
    
    model = IngredientWeightPredictor(len(ingredients), len(units)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):  # Reduced epochs for faster training
        optimizer.zero_grad()
        outputs = model(ingredient_indices.to(device), unit_indices.to(device), amounts.to(device))
        loss = criterion(outputs, weights.to(device))
        loss.backward()
        optimizer.step()
    
    return model, ingredient_to_idx, unit_to_idx

def main():
    print("Creating synthetic dataset...")
    data, ingredients, units = create_synthetic_dataset()
    
    print("Training model...")
    model, ingredient_to_idx, unit_to_idx = train_model(data, ingredients, units)
    
    print("Initializing camera detector...")
    detector = CameraIngredientDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    current_unit = "piece"
    last_detection_time = 0
    detection_interval = 2
    
    print("\nPrecision Baking AI Ready!")
    print("Controls: 1=cup, 2=tbsp, 3=tsp, 4=piece, q=Quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        current_time = time.time()
        if current_time - last_detection_time > detection_interval:
            detected_ingredient, estimated_volume = detector.detect_ingredient(frame)
            last_detection_time = current_time
            
            if detected_ingredient in ingredient_to_idx and current_unit in unit_to_idx:
                ingredient_idx = torch.tensor([ingredient_to_idx[detected_ingredient]], dtype=torch.long).to(device)
                unit_idx = torch.tensor([unit_to_idx[current_unit]], dtype=torch.long).to(device)
                amount_tensor = torch.tensor([estimated_volume], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    weight = model(ingredient_idx, unit_idx, amount_tensor).item()
                
                cv2.putText(frame, f"{detected_ingredient}: {weight:.2f}g", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Precision Baking AI', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()