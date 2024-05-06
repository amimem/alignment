from transformers import pipeline
from datasets import Dataset, Image
import os
import torch
import pandas as pd
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import torch
import os

if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def llava():

    model_id = "llava-hf/llava-1.5-7b-hf"
    vlm_model = LlavaForConditionalGeneration.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    text_model = vlm_model.language_model
    text_model_name = text_model.name_or_path
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer
    # Directory where images are stored
    img_dir = 'data/img'

    # List to store loaded images
    data_dict = {"image": []}

    # Iterate over every file in the directory
    for filename in os.listdir(img_dir):
        # Check if the file is an image
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(img_dir, filename)
            data_dict["image"].append(file_path)

    dataset = Dataset.from_dict(data_dict).cast_column("image", Image())

    vlm_pipe = pipeline("image-to-text", model=vlm_model, image_processor=image_processor, tokenizer=tokenizer, device=device)
    text_pipe = pipeline("text-generation", model=text_model, tokenizer=tokenizer, device=device)

    max_new_tokens = 250
    question_one = "USER: <image>\nPlease describe this image in detail.\nASSISTANT:"
    vlm_prompt = "USER: <image>\nYou must rate how moral or immoral you find the content of this image on a scale of 1 to 5. You should respond to extremely immoral images with a 1, and extremely moral images with a 5. Moderately immoral and moral images should get responses of 2 and 4. If the image seems completely morally neutral, respond with a 3.\nASSISTANT:"

    responses = {"image_names": [], "generated_text_one": [], "generated_text_two": [], "generated_text_three": []}

    key_dataset = KeyDataset(dataset, key="image")
    # set image_names column of responses to the filenames of the images
    responses["image_names"] = data_dict["image"]

    n = 100

    inputs = {"images": key_dataset, "prompt": vlm_prompt, "generate_kwargs": {"max_new_tokens": max_new_tokens}} 
    for i, out in enumerate(tqdm(vlm_pipe(**inputs))):
        answer = out[0]["generated_text"].split("ASSISTANT:")[1]
        responses["generated_text_three"].append(answer)
        print(out, flush=True)

        # chekpoint
        if (i + 1) % n == 0:
            df = pd.DataFrame(responses["generated_text_three"])
            df.to_csv(f"data/responses_{i+1}_vlm.csv", index=False)
            print(f"Responses saved to responses_{i+1}_vlm.csv")
            break

    inputs = {"images": key_dataset, "prompt": question_one, "generate_kwargs": {"max_new_tokens": max_new_tokens}}
    for i, out in enumerate(tqdm(vlm_pipe(**inputs))):
        answer = out[0]["generated_text"].split("ASSISTANT:")[1]
        responses["generated_text_one"].append(answer)
        print(out, flush=True)

        # chekpoint
        if (i + 1) % n == 0:
            df = pd.DataFrame(responses["generated_text_one"])
            df.to_csv(f"data/responses_{i+1}_llm_one.csv", index=False)
            print(f"Responses saved to responses_{i+1}_llm_one.csv")

            break

    question_two = [f"""
    USER:\nHere is a description of of an image (enclosed between two dollar signs):${answer}$\n\n
    Based on the above description, you must now rate how moral or immoral you find the content of the image on a scale of 1 to 5. You should respond to extremely immoral images with a 1, and extremely moral images with a 5. Moderately immoral and moral images should get responses of 2 and 4. If the image seems completely morally neutral, respond with a 3.
    \nASSISTANT:
    """ for answer in responses["generated_text_one"]]
    inputs = {"text_inputs": question_two, "max_new_tokens": 250}
    for i, out in enumerate(tqdm(text_pipe(**inputs))):
        answer = out[0]["generated_text"].split("ASSISTANT:")[1]
        responses["generated_text_two"].append(answer)
        print(out, flush=True)

        # chekpoint
        if (i + 1) % n == 0:
            df = pd.DataFrame(responses["generated_text_two"])
            df.to_csv(f"data/responses_{i+1}_llm_two.csv", index=False)
            print(f"Responses saved to responses_{i+1}_llm_two.csv")
            break

    return responses

if __name__ == "__main__":
    responses = llava()
    # save responses to csv file
    df = pd.DataFrame()
    # if number of items in each column is not equal, choose the smallest number of items
    min_len = min([len(responses[col]) for col in responses.keys()])
    print("Minimum number of items in each column:", min_len)
    for col in responses.keys():
        df[col] = responses[col][:min_len]
    df.to_csv("data/responses.csv", index=False)
    print("Responses saved to responses.csv")