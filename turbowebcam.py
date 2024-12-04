from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch, cv2
import numpy as np
from PIL import Image

prompt = "gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")
g_cuda = torch.Generator(device='cuda')
g_cuda.manual_seed(2147483647)
cv2.namedWindow("preview")
cv2.namedWindow("result")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    cv2.imshow("preview", frame)
    color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_converted).resize((512,512))
    g_cuda.manual_seed(2147483647)
    sdxl_result_image = pipe(prompt, image=pil_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0, generator=g_cuda).images[0]
    numpy_image=np.array(sdxl_result_image)
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", opencv_image)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
cv2.destroyWindow("result")
