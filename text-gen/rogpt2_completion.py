'''
Text Generation
Run RoGPT-2 model for text completion task.
'''

import pandas as pd
from transformers import AutoModel, AutoTokenizer, TFAutoModel, AutoModelForCausalLM
import torch

df=pd.read_pickle('insert-dataset-path')

# Take the first 10 tokens from each human text to be given as input to the text generation model
df["AI_Input"] = df["HumanText"].str.split().str[:10].str.join(sep=" ")
df["AI_Input"] = df["AI_Input"].astype('string')
df = df.dropna()
mylist = df.AI_Input.to_list()
mylist = list(dict.fromkeys(mylist))

model = AutoModelForCausalLM.from_pretrained('readerbench/RoGPT2-base')

# Tokenize the input
tokenizer2 = AutoTokenizer.from_pretrained('readerbench/RoGPT2-base')
tokenized_inputs =[tokenizer2.encode(text) for text in mylist]

# Run the model
outputs = []
for input_ids in tokenized_inputs:
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    output = model.generate(input_ids, max_length=512, no_repeat_ngram_size=2, pad_token_id=tokenizer2.eos_token_id)
    generated_text = tokenizer2.decode(output[0])
    outputs.append(generated_text)

# Create new dataframe to store the AI generated text (optional)
ai_df = pd.DataFrame({'input': mylist, 'output': outputs})
ai_df["output"] = ai_df["output"].str.replace(r'<|endoftext|>', " ")
ai_df["output"] = ai_df["output"].str.replace(r'\|', " ")

# Save it in format text - label- description
output_frame= ai_df[["output"]]
output_frame['label'] = '0'
output_frame['description'] = 'This is AI-generated text'
output_frame.rename(columns = {'output':'text'}, inplace = True)
output_frame.to_csv('ai-text-rogpt2.csv',index = False, encoding = 'utf-8-sig')
