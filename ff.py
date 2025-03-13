# # # ==========================================
# # # Fringe Order Prediction from RGB Values
# # # ==========================================

# # # Import necessary libraries
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from mpl_toolkits.mplot3d import Axes3D
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.svm import SVR
# # from xgboost import XGBRegressor
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # # ------------------------------------------
# # # 1. Generate Synthetic Data (If needed)
# # # ------------------------------------------
# # beam_h = 10  # Beam height in mm
# # beam_b = 6   # Beam thickness in mm
# # beam_I = (beam_b * beam_h**3) / 12  # Moment of Inertia in mm^4
# # bm = 600  # Bending Moment in N-mm
# # Fsigma = 12  # N/mm/fringe

# # # Compute bending stress
# # b_stress = (bm * (beam_h / 2) / beam_I)  # Bending stress in MPa

# # # Wavelengths from 390nm to 760nm
# # lambda_vals = np.linspace(390, 760, 371)  # nm

# # # Spectral Power Distribution (SPD)
# # SPD_fluorescent = (0.1 * np.exp(-((lambda_vals - 435) ** 2) / (2 * 10 ** 2)) +
# #                    0.2 * np.exp(-((lambda_vals - 545) ** 2) / (2 * 10 ** 2)) +
# #                    0.3 * np.exp(-((lambda_vals - 580) ** 2) / (2 * 15 ** 2)) +
# #                    0.4 * np.exp(-((lambda_vals - 610) ** 2) / (2 * 20 ** 2)))

# # # Spectral response functions for R, G, B
# # spectral_response_R = np.exp(-((lambda_vals - 650) ** 2) / (2 * 20 ** 2))
# # spectral_response_G = np.exp(-((lambda_vals - 547) ** 2) / (2 * 20 ** 2))
# # spectral_response_B = np.exp(-((lambda_vals - 460) ** 2) / (2 * 20 ** 2))

# # # Create synthetic stress field
# # rows = 1000
# # stress_max = b_stress * 10**6  # Convert MPa to Pascals
# # stress_map = np.linspace(0, stress_max, rows).reshape(-1, 1)

# # # Material properties
# # h = 6e-3  # Thickness of beam in meters
# # C = 4.55E-11  # Birefringence constant (m²/N)

# # # Initialize RGB and fringe order arrays
# # R = np.zeros_like(stress_map)
# # G = np.zeros_like(stress_map)
# # B = np.zeros_like(stress_map)
# # N = np.zeros_like(stress_map)

# # # Compute intensity integral for each RGB channel
# # for i, lam in enumerate(lambda_vals):
# #     phase = (2 * np.pi * h * C * stress_map) / (lam * 1e-9)
# #     N = phase / (2 * np.pi)  # Fringe order
# #     intensity = 0.5 * (1 - np.cos(phase)) / 300  # Dark field formula

# #     R += intensity * spectral_response_R[i] * SPD_fluorescent[i]
# #     G += intensity * spectral_response_G[i] * SPD_fluorescent[i]
# #     B += intensity * spectral_response_B[i] * SPD_fluorescent[i]

# # # Normalize RGB values
# # R /= np.max(R)
# # G /= np.max(G)
# # B /= np.max(B)

# # # Create dataset
# # dataset = pd.DataFrame({"R": R.flatten(), "G": G.flatten(), "B": B.flatten(), "Fringe_Order": N.flatten()})
# # dataset.to_csv("synthetic_rgb_fringe_data.csv", index=False)

# # print("Dataset generated and saved.")

# # # ------------------------------------------
# # # 2. Data Visualization
# # # ------------------------------------------

# # # Load the dataset
# # df = pd.read_csv("synthetic_rgb_fringe_data.csv")

# # # Pairwise scatter plot
# # sns.pairplot(df)
# # plt.show()

# # # 3D Scatter Plot
# # fig = plt.figure(figsize=(10, 7))
# # ax = fig.add_subplot(111, projection='3d')
# # sc = ax.scatter(df['R'], df['G'], df['B'], c=df['Fringe_Order'], cmap='viridis', alpha=0.5)
# # ax.set_xlabel('R')
# # ax.set_ylabel('G')
# # ax.set_zlabel('B')
# # ax.set_title("RGB Space vs Fringe Order")
# # plt.colorbar(sc)
# # plt.show()

# # # Correlation matrix
# # plt.figure(figsize=(8,6))
# # sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# # plt.title("Correlation Matrix")
# # plt.show()

# # # ------------------------------------------
# # # 3. Train/Test Split and Preprocessing
# # # ------------------------------------------

# # # Define features and target variable
# # X = df[['R', 'G', 'B']]
# # y = df['Fringe_Order']

# # # Split dataset into training (80%) and testing (20%)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Scale the input data
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)

# # # ------------------------------------------
# # # 4. Model Training
# # # ------------------------------------------

# # # Initialize models
# # models = {
# #     "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
# #     "Support Vector Machine": SVR(kernel='rbf'),
# #     "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
# # }

# # # Train and evaluate models
# # results = {}
# # for name, model in models.items():
# #     print(f"Training {name}...")
# #     model.fit(X_train_scaled, y_train)
# #     y_pred = model.predict(X_test_scaled)
    
# #     # Compute metrics
# #     mae = mean_absolute_error(y_test, y_pred)
# #     mse = mean_squared_error(y_test, y_pred)
# #     r2 = r2_score(y_test, y_pred)
    
# #     results[name] = {"MAE": mae, "MSE": mse, "R2": r2}
# #     print(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")

# # # ------------------------------------------
# # # 5. Model Comparison
# # # ------------------------------------------

# # # Convert results to DataFrame
# # results_df = pd.DataFrame(results).T
# # print("\nModel Performance:\n", results_df)

# # # Plot R2 scores
# # plt.figure(figsize=(8, 5))
# # sns.barplot(x=results_df.index, y=results_df["R2"], palette="viridis")
# # plt.title("R2 Score Comparison")
# # plt.ylabel("R2 Score")
# # plt.ylim(0, 1)
# # plt.show()

# # # ------------------------------------------
# # # 6. Predictions and Visualization
# # # ------------------------------------------

# # # Random Forest Predictions
# # rf_model = models["Random Forest"]
# # y_rf_pred = rf_model.predict(X_test_scaled)

# # # Scatter plot of true vs predicted values
# # plt.figure(figsize=(6, 6))
# # plt.scatter(y_test, y_rf_pred, alpha=0.5, color="blue")
# # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# # plt.xlabel("True Fringe Order")
# # plt.ylabel("Predicted Fringe Order")
# # plt.title("True vs Predicted Fringe Order (Random Forest)")
# # plt.show()

# # Import required libraries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import cv2
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# import xgboost as xgb
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# from torchvision.models import vit_b_16, resnet18
# from PIL import Image

# # -------------------------------
# # Data Generation (Synthetic RGB and Fringe Order)
# # -------------------------------

# # Beam properties
# beam_h, beam_b, bm, Fsigma = 10, 6, 600, 12
# beam_I = (beam_b * beam_h**3) / 12
# b_stress = (bm * (beam_h / 2) / beam_I)
# N_max = b_stress * (beam_b / Fsigma)

# # Wavelengths
# lambda_vals = np.linspace(390, 760, 371)

# # Light Source (SPD)
# SPD_fluorescent = (0.1 * np.exp(-((lambda_vals - 435) ** 2) / (2 * 10 ** 2)) +
#                    0.2 * np.exp(-((lambda_vals - 545) ** 2) / (2 * 10 ** 2)) +
#                    0.3 * np.exp(-((lambda_vals - 580) ** 2) / (2 * 15 ** 2)) +
#                    0.4 * np.exp(-((lambda_vals - 610) ** 2) / (2 * 20 ** 2)))

# # RGB Response
# spectral_response_R = np.exp(-((lambda_vals - 650) ** 2) / (2 * 20 ** 2))
# spectral_response_G = np.exp(-((lambda_vals - 547) ** 2) / (2 * 20 ** 2))
# spectral_response_B = np.exp(-((lambda_vals - 460) ** 2) / (2 * 20 ** 2))

# # Stress Field
# rows, cols = 256, 256
# stress_map = np.tile(np.linspace(0, b_stress * 10**6, rows).reshape(-1, 1), cols)

# # Material & Optical Properties
# h, C = 6e-3, 4.55E-11

# # Initialize
# R, G, B, N = np.zeros_like(stress_map), np.zeros_like(stress_map), np.zeros_like(stress_map), np.zeros_like(stress_map)

# # Compute RGB
# for i, lam in enumerate(lambda_vals):
#     phase = (2 * np.pi * h * C * stress_map) / (lam * 1e-9)
#     N = phase / (2 * np.pi)
#     intensity = 0.5 * (1 - np.cos(phase)) / 300
#     R += intensity * spectral_response_R[i] * SPD_fluorescent[i]
#     G += intensity * spectral_response_G[i] * SPD_fluorescent[i]
#     B += intensity * spectral_response_B[i] * SPD_fluorescent[i]

# # Normalize
# R /= np.max(R)
# G /= np.max(G)
# B /= np.max(B)

# # Save Image
# isochromatic_image = np.stack((R, G, B), axis=-1)
# plt.imsave("synthetic_fringe.png", isochromatic_image)

# # Convert to dataset
# df = pd.DataFrame({"R": R.flatten(), "G": G.flatten(), "B": B.flatten(), "Fringe_Order": N.flatten()})
# df.to_csv("synthetic_rgb_fringe_data.csv", index=False)

# print("Dataset and image saved.")

# # -------------------------------
# # Data Preprocessing
# # -------------------------------
# df = pd.read_csv("synthetic_rgb_fringe_data.csv")
# X = df[['R', 'G', 'B']]
# y = df['Fringe_Order']

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalize
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # -------------------------------
# # Train Random Forest, SVM, XGBoost
# # -------------------------------
# models = {
#     "Random Forest": RandomForestRegressor(n_estimators=100),
#     "SVM": SVR(),
#     "XGBoost": xgb.XGBRegressor()
# }

# for name, model in models.items():
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#     print(f"{name} Results:")
#     print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
#     print(f"MSE: {mean_squared_error(y_test, y_pred)}")
#     print(f"R² Score: {r2_score(y_test, y_pred)}\n")

# # -------------------------------
# # CNN & ViT for Image-Based Prediction
# # -------------------------------
# class FringeDataset(Dataset):
#     def __init__(self, img_path, target_values, transform=None):
#         self.image = Image.open(img_path)
#         self.target = torch.tensor(target_values, dtype=torch.float32).unsqueeze(0)
#         self.transform = transform
    
#     def __len__(self):
#         return 1
    
#     def __getitem__(self, idx):
#         image = self.image
#         if self.transform:
#             image = self.transform(image)
#         return image, self.target

# transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# dataset = FringeDataset("synthetic_fringe.png", [N_max], transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # CNN Model
# class CNNRegressor(nn.Module):
#     def __init__(self):
#         super(CNNRegressor, self).__init__()
#         self.cnn = resnet18(pretrained=True)
#         self.cnn.fc = nn.Linear(512, 1)
    
#     def forward(self, x):
#         return self.cnn(x)

# cnn_model = CNNRegressor()
# optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
# loss_fn = nn.MSELoss()

# # Training Loop
# for epoch in range(10):
#     for img, target in dataloader:
#         optimizer.zero_grad()
#         output = cnn_model(img)
#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# # ViT Model
# vit_model = vit_b_16(pretrained=True)
# vit_model.heads.head = nn.Linear(768, 1)
# optimizer = optim.Adam(vit_model.parameters(), lr=0.001)

# # Training Loop for ViT
# for epoch in range(10):
#     for img, target in dataloader:
#         optimizer.zero_grad()
#         output = vit_model(img)
#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# print("Training Complete.")
# Full Jupyter Notebook for Fringe Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Step 1: Synthetic Data Generation
# ================================
# Beam and material properties
beam_h = 10  # Beam height in mm
beam_b = 6   # Beam thickness in mm
beam_I = (beam_b * beam_h**3) / 12  # Moment of Inertia in mm^4
bm = 600  # Bending Moment in N-mm
Fsigma = 12  # N/mm/fringe

# Compute bending stress
b_stress = (bm * (beam_h / 2) / beam_I)  # Bending stress in MPa
N_max = b_stress * (beam_b / Fsigma)  # Fringe order

# Wavelengths from 390nm to 760nm
lambda_vals = np.linspace(390, 760, 371)  # nm

# Spectral Power Distribution (SPD) of light source (fluorescent)
SPD_fluorescent = (0.1 * np.exp(-((lambda_vals - 435) ** 2) / (2 * 10 ** 2)) +
                   0.2 * np.exp(-((lambda_vals - 545) ** 2) / (2 * 10 ** 2)) +
                   0.3 * np.exp(-((lambda_vals - 580) ** 2) / (2 * 15 ** 2)) +
                   0.4 * np.exp(-((lambda_vals - 610) ** 2) / (2 * 20 ** 2)))

# Spectral response functions for R, G, B channels
spectral_response_R = np.exp(-((lambda_vals - 650) ** 2) / (2 * 20 ** 2))
spectral_response_G = np.exp(-((lambda_vals - 547) ** 2) / (2 * 20 ** 2))
spectral_response_B = np.exp(-((lambda_vals - 460) ** 2) / (2 * 20 ** 2))

# Create synthetic stress field
rows, cols = 1000, 1
stress_max = b_stress * 10**6  # Convert MPa to Pascals
stress_map = np.tile(np.linspace(0, stress_max, rows).reshape(-1, 1), cols)

# Material and optical properties
h = 6e-3  # Thickness of beam in meters
C = 4.55E-11  # Birefringence constant (m²/N)

# Initialize RGB and fringe order arrays
R = np.zeros_like(stress_map)
G = np.zeros_like(stress_map)
B = np.zeros_like(stress_map)
N = np.zeros_like(stress_map)

# Compute intensity integral for each RGB channel
for i, lam in enumerate(lambda_vals):
    phase = (2 * np.pi * h * C * stress_map) / (lam * 1e-9)
    N = phase / (2 * np.pi)  # Fringe order
    intensity = 0.5 * (1 - np.cos(phase)) / 300  # Dark field formula
    R += intensity * spectral_response_R[i] * SPD_fluorescent[i]
    G += intensity * spectral_response_G[i] * SPD_fluorescent[i]
    B += intensity * spectral_response_B[i] * SPD_fluorescent[i]

# Normalize RGB channels
R /= np.max(R)
G /= np.max(G)
B /= np.max(B)

# Save dataset as CSV
dataset = pd.DataFrame({"R": R.flatten(), "G": G.flatten(), "B": B.flatten(), "Fringe_Order": N.flatten()})
dataset.to_csv("synthetic_rgb_fringe_data.csv", index=False)
print("Dataset saved as 'synthetic_rgb_fringe_data.csv'")

# Save RGB image
isochromatic_image = np.stack((R, G, B), axis=-1)
plt.imsave("synthetic_image.png", isochromatic_image)

# ================================
# Step 2: Load Dataset for Training
# ================================
df = pd.read_csv("synthetic_rgb_fringe_data.csv")
X = df[['R', 'G', 'B']].values
y = df['Fringe_Order'].values

# Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# Step 3: Train ML Models
# ================================
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R² Score: {r2_score(y_test, y_pred)}")
    print("="*40)

# Random Forest
print("Random Forest Results:")
evaluate_model(RandomForestRegressor(), X_train_scaled, X_test_scaled, y_train, y_test)

# SVM
print("SVM Results:")
evaluate_model(SVR(), X_train_scaled, X_test_scaled, y_train, y_test)

# XGBoost
print("XGBoost Results:")
evaluate_model(XGBRegressor(), X_train_scaled, X_test_scaled, y_train, y_test)

# ================================
# Step 4: Train CNN on Image
# ================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn.fc = nn.Linear(512, 1)  # Output a single value (fringe order)

    def forward(self, x):
        return self.cnn(x)

# Load image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :])  # Fix alpha channel issue
])

img = Image.open("synthetic_image.png").convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# Train CNN
cnn_model = CNN().to(device)
cnn_model.eval()
output = cnn_model(img)
print("CNN Output (Fringe Order Prediction):", output.item())

# ================================
# Step 5: Train Vision Transformer (ViT)
# ================================
vit_model = models.vit_b_16(weights="IMAGENET1K_V1")
vit_model.heads.head = nn.Linear(768, 1)  # Adjust for regression

# Transform Image for ViT
img = transform(img).unsqueeze(0).to(device)
output_vit = vit_model(img)
print("ViT Output (Fringe Order Prediction):", output_vit.item())
