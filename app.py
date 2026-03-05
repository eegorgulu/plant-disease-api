from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import timm

app = FastAPI()

# Modeli oluştur
model = timm.create_model("convnext_small", pretrained=False, num_classes=7)
model.load_state_dict(torch.load("bestmodel.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

CLASSES = [
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "mosaic_virus",
    "septoria",
    "yellow_leaf_curl"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()

    return {"prediction": CLASSES[pred]}