import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Charger le modèle
model_path = 'model_checkpoint.pth' 
model = MyModel()  # Remplacez par la définition de votre modèle
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  
model.eval()

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
