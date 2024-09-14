import base64
import requests
import os
import csv
import pandas as pd
import re

# OpenAI API Key
api_key = ""
directory_path = ""
csv_output_file = ""
model = "gpt-4-turbo"
temperature = 0

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_description(image_path,
                          api_key,
                          model="gpt-4-turbo",
                          temperature=0):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe this image in detail"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": temperature
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    return response_data


# List all files and directories in the given directory
# List all files and directories in the given directory
entries = os.listdir(directory_path)

# Filter out directories, keep only files
files = [os.path.join(directory_path, entry) for entry in entries if
         os.path.isfile(os.path.join(directory_path, entry))]
all_responses = []

for i, image_path in enumerate(files):
    try:
        response = get_image_description(image_path=image_path,
                                         api_key=api_key,
                                         model=model,
                                         temperature=temperature)
        response['image_name'], _ = os.path.splitext(os.path.basename(image_path))
        all_responses.append(response)
    except Exception as e:
        print(f"Failed to read image {image_path}: {e}")
    if i % 50 == 0:
        print(i)

0  # Keys to write to the CSV file, including paths to nested data
keys_to_write = ['image_name', 'choices.message']


# Define a function to extract nested data
def extract_data(item):
    return [item['image_name'], item['choices'][0]['message']['content']]


# Extract data using list comprehension
extracted_data = [extract_data(item) for item in all_responses]
# Convert extracted data to DataFrame
df = pd.DataFrame(extracted_data, columns=['id', 'description'])

print("Stage Two - Description to moral rate")


def extract_number(text):
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    else:
        return 0


def moral_analysis(description, api_key, model="gpt-4-turbo", temperature=0):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Based the following description of an image: {description}, you must now rate how moral or "
                                f"immoral you find the image on a scale of 1 to 5. "
                                f"You should respond to extremely immoral with a 1, and extremely "
                                f"moral with a 5. Moderately immoral and moral description should get "
                                f"responses of 2 and 4. If the content seems completely morally neutral, "
                                f"respond with a 3. Do not add any explanation to the description."
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": temperature
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    return int(extract_number(response_data['choices'][0]['message']['content']))


def get_moral_rate(df):
    df['moral_assessment_GPT'] = df['description'].apply(moral_analysis, args=(api_key, model, temperature,))
    return df


df_moral_rates = get_moral_rate(df)

df_moral_rates.to_csv(csv_output_file)

print('Done')
