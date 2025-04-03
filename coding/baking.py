
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import time

# Ingredient weight predictor model
class IngredientWeightPredictor(nn.Module):
    def __init__(self, num_ingredients):
        super(IngredientWeightPredictor, self).__init__()
        self.ingredient_embedding = nn.Embedding(num_ingredients, 16)
        self.fc = nn.Sequential(
            nn.Linear(16 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, ingredient_idx, amount):
        ingredient_emb = self.ingredient_embedding(ingredient_idx)
        x = torch.cat([ingredient_emb, amount.unsqueeze(1)], dim=1)
        return self.fc(x)

# Camera ingredient detector
class CameraIngredientDetector:
    def __init__(self):
        self.model = mobilenet_v2(weights='IMAGENET1K_V1')
        self.model.eval()
        self.ingredient_classes = ['flour', 'sugar', 'salt', 'butter', 'milk', 'egg']
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect_ingredient(self, frame):
        # Convert frame to PIL image for model
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred = torch.max(probabilities, 1)

        detected_ingredient = self.ingredient_classes[pred.item() % len(self.ingredient_classes)]
        presence_percentage = confidence.item() * 100  # Convert to percentage

        return detected_ingredient, presence_percentage

# Generate synthetic dataset
def create_synthetic_dataset():
    ingredients = ['flour', 'sugar', 'salt', 'butter', 'milk', 'egg']
    densities = {'flour': 120, 'sugar': 200, 'salt': 300, 'butter': 240, 'milk': 240, 'egg': 50}

    data = []
    for _ in range(1000):
        ingredient = np.random.choice(ingredients)
        volume = np.random.uniform(0.1, 4)  # Random volume
        weight = volume * densities[ingredient] + np.random.normal(0, 10)
        data.append({'ingredient': ingredient, 'volume': volume, 'weight': weight})
    return data, ingredients

# Train the model
def train_model(data, ingredients):
    ingredient_to_idx = {ing: i for i, ing in enumerate(ingredients)}
    ingredient_indices = torch.tensor([ingredient_to_idx[d['ingredient']] for d in data], dtype=torch.long)
    volumes = torch.tensor([d['volume'] for d in data], dtype=torch.float32)
    weights = torch.tensor([d['weight'] for d in data], dtype=torch.float32).view(-1, 1)

    model = IngredientWeightPredictor(len(ingredients))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(ingredient_indices, volumes)
        loss = criterion(outputs, weights)
        loss.backward()
        optimizer.step()

    return model, ingredient_to_idx

# Main function
def main():
    print("Creating synthetic dataset...")
    data, ingredients = create_synthetic_dataset()

    print("Training model...")
    model, ingredient_to_idx = train_model(data, ingredients)

    print("Initializing camera detector...")
    detector = CameraIngredientDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    last_detection_time = 0
    detection_interval = 2  # seconds

    print("\n🎯 Precision Baking AI Ready!")
    print("Press 'q' to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        current_time = time.time()
        if current_time - last_detection_time > detection_interval:
            detected_ingredient, presence_percentage = detector.detect_ingredient(frame)
            last_detection_time = current_time

            # Predict weight
            if detected_ingredient in ingredient_to_idx:
                ingredient_idx = torch.tensor([ingredient_to_idx[detected_ingredient]], dtype=torch.long)
                amount_tensor = torch.tensor([1.0], dtype=torch.float32)  # Assume 1 unit volume for display

                with torch.no_grad():
                    weight = model(ingredient_idx, amount_tensor).item()

                # Print detected information in terminal
                print(f"\n📌 Detected: {detected_ingredient}")
                print(f"   📏 Estimated Weight: {weight:.2f}g")
                print(f"   🎯 Presence: {presence_percentage:.2f}%\n")

                # Display information on frame
                cv2.putText(frame, f"Ingredient: {detected_ingredient}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Estimated Weight: {weight:.2f}g", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Presence: {presence_percentage:.2f}%", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Precision Baking AI', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
