#pip3 install torch diffusers transformers accelerate peft


from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.float16).to('cuda', use_auth_token=token)
pipeline.load_lora_weights('Amirkid/flux-dev-finetune', weight_name='lora.safetensors')
print("the model is loaded")

print("This is the testing image")
image = pipeline('Amir Kidwai surfing on the moon').images[0]
