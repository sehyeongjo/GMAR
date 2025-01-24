import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def save_overlay_image(input, output, gmar_rollout):

    original_image = Image.open(input).convert("RGB")
    original_image = original_image.resize((512, 512))
    original_array = np.array(original_image)

    gmar_resized = cv2.resize((gmar_rollout.cpu().numpy() * 255).astype(np.uint8), (512, 512))
    gmar_colored = cv2.applyColorMap(gmar_resized, cv2.COLORMAP_RAINBOW)
    gmar_overlay = cv2.addWeighted(original_array, 0.6, gmar_colored, 0.4, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    axes[1].imshow(gmar_overlay)
    axes[1].axis('off')
    axes[1].set_title('GMAR Overlay') 

    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"output saved to {output}")