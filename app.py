import streamlit as st
import torch
from torch import nn 
from PIL import Image
from torchvision import transforms

# Définir votre modèle (remplacez MyModel par votre modèle réel)
class ECGTransformer(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, output_dim, expanded_dim):
        super(ECGTransformer, self).__init__()
        self.linear_in = nn.Linear(input_dim, expanded_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=expanded_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(expanded_dim, output_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out

# Fonction pour charger le modèle
def load_model(model_path):
    model = ECGTransformer(input_dim, num_layers, num_heads, hidden_dim, output_dim, expanded_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Chemin vers le modèle
model_path = 'ecg_transformers_model.pth'
input_dim = 100  # Remplacez par la bonne valeur
num_layers = 2   # Remplacez par la bonne valeur
num_heads = 2    # Remplacez par la bonne valeur
hidden_dim = 256 # Remplacez par la bonne valeur
output_dim = 1   # Remplacez par la bonne valeur
expanded_dim = 8 # Remplacez par la bonne valeur

# Charger le modèle
model = load_model(model_path)

# Fonction de transformation des images
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# Fonction de prédiction
def predict(image):
    image = transform_image(image)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return 'Normal' if predicted.item() == 0 else 'Anormal'

st.title('Interface de Prédiction pour les Images ECG')
uploaded_file = st.file_uploader("Choisissez une image d'ECG...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image ECG téléchargée.', use_column_width=True)
    st.write("")
    st.write("Classification...")
    label = predict(image)
    st.write(f'Prédiction : {label}')
