from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

 # 
def get_image_caption(image_path):
    """Generated a short caption for the input image
    source: https://huggingface.co/Salesforce/blip-image-captioning-large

    Args:
        image_path (str): the path to the image file
        
    Returns:
        str: A string representing the image caption
    """
    # read the image locally and covert to RGB
    image = Image.open(image_path).convert("RGB")
    
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


def detect_objects(image_path):
    """Detects objects in the input image and returns a list of detected objects
    source: https://huggingface.co/facebook/detr-resnet-50

    Args:
        image_path (str): the path to the image file
        
    Returns:
        list: A list of detected objects
    """
    image = Image.open(image_path).convert('RGB')
    
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

if __name__ == "__main__":
    image_path = "D:/Courses/Computer_vision_engineer/Ask_image_question/img1.jpg"
    caption = get_image_caption(image_path)
    print(caption)
    detections = detect_objects(image_path)
    print(detections)
    
    