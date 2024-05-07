from transformers import pipeline, set_seed
from datasets import Dataset, Image
import os
import torch
import pandas as pd
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import torch
import os

set_seed(42)

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


    text_pipe = pipeline("text-generation", model=text_model, tokenizer=tokenizer, device=device)

    max_new_tokens = 250
    responses = {"image_names": [], "kosmos_decriptions": [], "llama_ratings": []}

    # load csv file with image names and kosmos descriptions
    df = pd.read_csv("data/results_kosmos-2-patch14-224.csv")

    n = 100

    question_two = [f"""
    USER:\nHere is a description of of an image (enclosed between two dollar signs):${answer}$\n\n
    Based on the above description, you must now rate how moral or immoral you find the content of the image on a scale of 1 to 5. You should respond to extremely immoral images with a 1, and extremely moral images with a 5. Moderately immoral and moral images should get responses of 2 and 4. If the image seems completely morally neutral, respond with a 3.
    \nASSISTANT:
    """ for answer in responses["kosmos_decriptions"]]
    inputs = {"text_inputs": question_two, "max_new_tokens": max_new_tokens}
    for i, out in enumerate(tqdm(text_pipe(**inputs))):
        answer = out[0]["generated_text"].split("ASSISTANT:")[1]
        responses["generated_text_two"].append(answer)

        # chekpoint
        if (i + 1) % n == 0:
            df = pd.DataFrame(responses["generated_text_two"])
            df.to_csv(f"data/responses_{i+1}_llm_two.csv", index=False)
            print(f"Responses saved to responses_{i+1}_llm_two.csv")

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
    df.to_csv("data/kosmos_to_llama.csv", index=False)
    print("Responses saved to data/kosmos_to_llama.csv")