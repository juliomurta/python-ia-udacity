import argparse
import json
import numpy as np
import torch
import torchvision

from torchvision import models
from torch import nn
from PIL import Image

def get_predict_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str, help="set image path")
    parser.add_argument("checkpoint_dir", type=str, help="set checkpoint path")
    parser.add_argument("--top_k", type=int, default=5, help="return top k most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="set category names path")
    parser.add_argument("--gpu", action='store_true', help="set GPU or not")
    
    return parser.parse_args()

def process_image(image):    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    new_size = 256    
    crop_size = 224
    np_image  = []
    
    with Image.open(image) as pil_image:        
        new_crop_size = (new_size - crop_size) * .5
        pil_image = pil_image.resize((new_size, new_size))        
        
        pil_image = pil_image.crop((new_crop_size, 
                                    new_crop_size,
                                    new_size - new_crop_size,
                                    new_size - new_crop_size))
        
        np_image = np.array(pil_image) / 255         
        np_image = (np_image - mean) / std
        
    return np_image.transpose(2,0,1)

def predict(image_path, model, device, topk):
    indexes = []
    label = []
    
    model = model.to(device)
    model.eval()
    
    image = process_image(image_path)    
    image = torch.from_numpy(np.array([image])).float() 
    image = image.to(device)
    
    output = model.forward(image)    
    ps = torch.exp(model(image)).data    
    probability = torch.topk(ps, topk)[0].tolist()[0] 
    classes = torch.topk(ps, topk)[1].tolist()[0]
            
    for index in range(len(model.class_to_idx.items())):
        indexes.append(list(model.class_to_idx.items())[index][0])
        
    for index in range(topk):
        label.append(indexes[classes[index]])
        
    return (probability, label)

def load_categories(filename):    
    with open(filename, 'r') as f:
        return json.load(f)

def load_model(checkpoint_file):    
    checkpoint = torch.load(checkpoint_file)
    
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)   
    model.classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_units']),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(checkpoint['hidden_units'], 102),
                                     nn.LogSoftmax(dim=1))
        
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
    
def main():
    input_args = get_predict_args()
    
    device = 'cpu'
    if input_args.gpu and torch.cuda.is_available():
        device = 'cuda'
        
    print(f"Device: {device}")    
        
    categories = load_categories(input_args.category_names)
    model = load_model(input_args.checkpoint_dir)
    
    probs, classes = predict(input_args.image_dir, model, device, input_args.top_k)
    
    labels = [categories[class_name] for class_name in classes]   
    flower_name = labels[np.argmax(probs)]
    
    print(f"Flower name: { flower_name }")
    print("Results: ")
    for label, prob in zip(labels, probs):
        print(f"{ label } : { prob }")
        
if __name__ == "__main__":    
    main()