import requests
from PIL import Image
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

# load images from data/img folder

# Directory where images are stored
img_dir = 'data/img'

# List to store loaded images
images = []

# Iterate over every file in the directory
for filename in os.listdir(img_dir):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Open the image file
        img = Image.open(os.path.join(img_dir, filename)).convert('RGB')
        # Append the image to the list
        images.append(img)

# Now you can use the images list in your code
for raw_image in images:
    question_one = "Please describe this image in detail."
    inputs_one = processor(raw_image, question_one, return_tensors="pt").to("cuda")

    out_one = model.generate(**inputs_one)
    # print all generated text
    for i, o in enumerate(out_one):
        print(f"Output {i}: {processor.decode(o)}")

    question_two = f"""
    Here is a description of of an image (enclosed between two dollar signs):
    $
    {processor.decode(out_one[0])}
    $

    Based on the above description, you must now rate how moral or immoral you find the content of the image on a scale of 1 to 5. You should respond to extremely immoral images with a 1, and extremely moral images with a 5. Moderately immoral and moral images should get responses of 2 and 4. If the image seems completely morally neutral, respond with a 3.
    """

    print(question_two)

    out_two = model.generate(question_two, return_tensors="pt")
    # print all generated text
    for i, o in enumerate(out_two):
        print(f"Output {i}: {processor.decode(o)}")

    break
