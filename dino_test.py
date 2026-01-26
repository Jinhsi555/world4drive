from transformers import AutoImageProcessor, AutoModel
from transformers.models.dinov2 import Dinov2Model
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from PIL import Image
import requests
import numpy
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-small')
model = AutoModel.from_pretrained('/vepfs-mlp2/c20250502/haoce/wlb/world4drive/checkpoints/dinov2-small')
# processor.do_normalize = False
inputs = processor(images=[image], return_tensors="pt")

def get_dino_features_with_scene_query(model: Dinov2Model, inputs: dict, scene_query: torch.Tensor):
    output_hidden_states = inputs.get("output_hidden_states", None)
    pixel_values = inputs.get("pixel_values", None)

    if output_hidden_states is None:
        output_hidden_states = model.config.output_hidden_states

    if pixel_values is None:
        raise ValueError("You have to specify pixel_values")

    embedding_output = model.embeddings(pixel_values, None)
    embedding_output = torch.cat([scene_query, embedding_output], dim=1)

    encoder_outputs: BaseModelOutput = model.encoder(
        embedding_output, None, output_hidden_states=output_hidden_states
    )
    sequence_output = encoder_outputs.last_hidden_state
    sequence_output = model.layernorm(sequence_output)
    pooled_output = sequence_output[:, 0, :]

    return BaseModelOutputWithPooling(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
    )

scene_query = torch.tensor([1.0]).expand(1, 2, 384)
outputs = get_dino_features_with_scene_query(model, inputs, scene_query)
last_hidden_states = outputs.last_hidden_state

# np.save('data.npy', last_hidden_state)
