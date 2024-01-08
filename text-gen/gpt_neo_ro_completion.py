'''
Text Generation
Run GPT-NEO-RO for text completion task.
'''

import pandas as pd
from transformers import AutoModel, AutoTokenizer, TFAutoModel, AutoModelForCausalLM
import torch

# Load your input dataset (human text)
df=pd.read_pickle('insert-dataset-path')

# Take the first 10 tokens from each human text to be given as input to the text generation model
df["AI_Input"] = df["HumanText"].str.split().str[:10].str.join(sep=" ")
df["AI_Input"] = df["AI_Input"].astype('string')
df = df.dropna()
mylist = df.AI_Input.to_list()
mylist = list(dict.fromkeys(mylist))

model = AutoModelForCausalLM.from_pretrained('dumitrescustefan/gpt-neo-romanian-780m')
tokenizer2 = AutoTokenizer.from_pretrained('dumitrescustefan/gpt-neo-romanian-780m')
tokenized_inputs =[tokenizer2.encode(text) for text in mylist]

outputs = []
for input_ids in tokenized_inputs:
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    output = model.generate(input_ids, max_length=512, do_sample=True, no_repeat_ngram_size=2, top_k=50, top_p=0.9, early_stopping=True)
    generated_text = tokenizer2.decode(output[0])
    outputs.append(generated_text)

ai_df = pd.DataFrame({'input': mylist, 'output': outputs})

output_frame= ai_df[["output"]]
output_frame['label'] = '0'
output_frame['description'] = 'This is AI-generated text'
output_frame.rename(columns = {'output':'text'}, inplace = True)
output_frame.to_csv('ai-text-gpt_neo_ro.csv',index = False, encoding = 'utf-8-sig')
