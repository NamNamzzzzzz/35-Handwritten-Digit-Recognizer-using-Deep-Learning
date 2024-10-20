from google.colab import files
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

uploaded = files.upload()
image_path = next(iter(uploaded))

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

processed_image = preprocess_image(image_path)

prediction = model.predict(processed_image)
predicted_digit = np.argmax(prediction)

print(f"Chữ số được dự đoán là: {predicted_digit}")

plt.imshow(processed_image.reshape(28, 28), cmap='gray')
plt.title(f"Dự đoán: {predicted_digit}")
plt.axis('off')
plt.show()