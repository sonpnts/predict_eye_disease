# from fastapi import FastAPI, File, UploadFile
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.efficientnet import preprocess_input
# import io
# from PIL import Image
# app = FastAPI()

# # Đường dẫn đến tệp mô hình
# model_path = 'efficientnetb3-EyeDisease-95.14.h5'

# # Tải mô hình
# model = load_model(model_path)

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     # Đọc và tiền xử lý hình ảnh
#     contents = await file.read()
#     img = Image.open(io.BytesIO(contents))
#     img = img.resize((224, 224))  # Thay đổi kích thước theo yêu cầu của mô hình
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     # Dự đoán
#     predictions = model.predict(img_array)

#     # In kết quả dự đoán
#     return {"prediction": predictions.tolist()}

from fastapi import FastAPI, File, UploadFile
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import io
from PIL import Image

app = FastAPI()

# URL của tệp mô hình trên Google Drive
model_url = 'https://drive.google.com/uc?id=1LXsCWLzM4BJnXWMwCGysZuTCABeBB9OH'

# Đường dẫn tạm thời để lưu tệp mô hình
model_path = 'efficientnetb3-EyeDisease-95.14.h5'

def download_model():
    gdown.download(model_url, model_path, quiet=False)

# Tải mô hình
download_model()
model = load_model(model_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Đọc và tiền xử lý hình ảnh
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = img.resize((224, 224))  # Thay đổi kích thước theo yêu cầu của mô hình
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Dự đoán
    predictions = model.predict(img_array)

    # In kết quả dự đoán
    return {"prediction": predictions.tolist()}
