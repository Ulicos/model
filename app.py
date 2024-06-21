import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Définir votre modèle (remplacez MyModel par votre modèle réel)
class ECGTransformer(torch.nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, output_dim, expanded_dim):
        super(ECGTransformer, self).__init__()
        self.linear_in = torch.nn.Linear(input_dim, expanded_dim)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=expanded_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(expanded_dim, output_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out

# Charger le modèle à partir de GitHub
model_url = "https://github.com/Ulicos/model/raw/main/ecg_transformers_model.pth"
input_dim = 224  # Ajustez la dimension d'entrée en fonction de votre modèle
num_layers = 2  # Ajustez le nombre de couches en fonction de votre modèle
num_heads = 2  # Ajustez le nombre de têtes en fonction de votre modèle
hidden_dim = 256  # Ajustez la dimension cachée en fonction de votre modèle
output_dim = 1  # Ajustez la dimension de sortie en fonction de votre modèle
expanded_dim = 8  # Ajustez la dimension étendue en fonction de votre modèle

# Télécharger le modèle
model_state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=torch.device('cpu'))

# Initialiser le modèle avec les paramètres téléchargés
model = ECGTransformer(input_dim, num_layers, num_heads, hidden_dim, output_dim, expanded_dim)
model.load_state_dict(model_state_dict)
model.eval()

# Fonction de transformation des images
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ajustez cette taille en fonction des exigences de votre modèle
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalisation basique, ajustez si nécessaire
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
