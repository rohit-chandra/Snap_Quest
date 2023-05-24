from typing import Any
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

class ImageCaptionTool(BaseTool):
    # name of the tool
    name = "Image captioner"
    
    # The description is a natural language description of the tool the LLM uses to decide whether it needs to use it. 
    # Tool descriptions should be very explicit on what they do, when to use them, and when not to use them.
    description = " Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."
    
    def _run(self, img_path):
        
        image = Image.open(img_path).convert("RGB")
    
        model_name = "Salesforce/blip-image-captioning-large"
        
        # use cuda if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        processor = BlipProcessor.from_pretrained(model_name)
        
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # covert the image into representation that the model can understand
        # return_tensors="pt" ==> Return PyTorch torch.Tensor objects.
        inputs = processor(image, return_tensors="pt").to(device)
        
        output = model.generate(**inputs, max_length=20)
        
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    


class ObjectDetectionTool(BaseTool):
    name = "Object dectector"
    
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objectd. Each element in the list is in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score"
    
    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')
        
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # bbox
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            # add label 
            detections += ' {}'.format(model.config.id2label[int(label)])
            # add confidence score
            detections += ' {}\n'.format(float(score))
        
        return detections
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")