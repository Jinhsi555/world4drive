from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import numpy

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-base')
model = AutoModel.from_pretrained('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-base')
processor.do_normalize = False
inputs = processor(images=[image], return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

np.save('data.npy', last_hidden_state)
