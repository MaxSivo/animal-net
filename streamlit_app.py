import streamlit as st
from PIL import Image
from PIL.ExifTags import TAGS
import torch
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
from datetime import datetime
import io

# # Connect to the SQLite database
# conn = sqlite3.connect('observations.db')
# c = conn.cursor()

# # Create a table to store observations if it doesn't exist
# c.execute('''CREATE TABLE IF NOT EXISTS observations
#              (id INTEGER PRIMARY KEY AUTOINCREMENT, image BLOB, species TEXT, date TEXT, location TEXT)''')
# conn.commit()


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 32)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the trained model
@st.cache_resource
def load_model():
    model = SimpleCNN()
    state_dict = torch.load('weights/bird_weights.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()


st.title('Animal Classification')
st.write('Upload an image, and the CNN will predict the species.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    def preprocess_image(image):
        """Resize, convert to tensor, and normalize the image."""
        transformation = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transformation(image).unsqueeze(0)
        return image

    processed_image = preprocess_image(image)

    with torch.no_grad():
        prediction = model(processed_image)
        _, predicted = torch.max(prediction.data, 1)

    # Mapping of labels
    labels = {0: 'Anseriformes',
            1: 'Sphenisciformes',
            2: 'Gaviiformes',
            3: 'Passeriformes',
            4: 'Pelecaniformes',
            5: 'Cuculiformes',
            6: 'Gruiformes',
            7: 'Charadriiformes',
            8: 'Columbiformes',
            9: 'Piciformes',
            10: 'Procellariiformes',
            11: 'Caprimulgiformes',
            12: 'Accipitriformes',
            13: 'Cathartiformes',
            14: 'Opisthocomiformes',
            15: 'Suliformes',
            16: 'Trogoniformes',
            17: 'Galliformes',
            18: 'Coraciiformes',
            19: 'Psittaciformes',
            20: 'Podicipediformes',
            21: 'Ciconiiformes',
            22: 'Bucerotiformes',
            23: 'Strigiformes',
            24: 'Falconiformes',
            25: 'Struthioniformes',
            26: 'Musophagiformes',
            27: 'Phoenicopteriformes',
            28: 'Coliiformes',
            29: 'Casuariiformes',
            30: 'Otidiformes',
            31: 'Galbuliformes'}

    species = labels[predicted.item()]

    st.write(f'Prediction: {species}')

#     # Extract metadata from the image
#     exif_data = image._getexif()
#     date = None
#     location = None

#     if exif_data:
#         for tag, value in exif_data.items():
#             tag_name = TAGS.get(tag, tag)
#             if tag_name == 'DateTimeOriginal':
#                 date = value
#             elif tag_name == 'GPSInfo':
#                 location = value

#     # If date or location is missing, prompt the user to input manually
#     if not date:
#         date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     if not location:
#         location = st.text_input("Location:", "")

#     if st.button("Save Observation"):
#         # Convert image to bytes
#         img_bytes = uploaded_file.getvalue()
#         # Insert observation into database
#         c.execute("INSERT INTO observations (image, species, date, location) VALUES (?, ?, ?, ?)",
#                   (img_bytes, str(species), str(date), str(location)))
#         conn.commit()

# show_observations = st.button("Show Observations")
# if show_observations:
#     st.subheader("Observations")
#     # Retrieve observations from the database
#     observations = c.execute("SELECT id, species, date, location, image FROM observations ORDER BY id ASC").fetchall()
#     for obs in observations:
#         obs_id, species, date, location, img_bytes = obs
#         # Display species, date, and location
#         st.write(f"Species: {species}")
#         st.write(f"Date: {date}")
#         st.write(f"Location: {location}")
#         # Display image
#         img = Image.open(io.BytesIO(img_bytes))
#         st.image(img, caption='Observation Image', use_column_width=True)
#         # Add delete button for each observation
#         delete_button = st.button(f"Delete Observation {obs_id}")
#         if delete_button:
#             try:
#                 print(f"Deleting observation with ID: {obs_id}")
#                 c.execute("DELETE FROM observations WHERE id=?", (obs_id,))
#                 conn.commit()
#                 st.write("Observation deleted successfully.")
#                 # Refresh observations after deletion
#                 st.experimental_rerun()
#             except Exception as e:
#                 st.error(f"Error deleting observation: {e}")
# conn.close()