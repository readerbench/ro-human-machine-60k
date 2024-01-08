'''
Text Generation
Run Flan-T5 model for paraphrasing task.
'''

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load your input dataset (human text)
df=pd.read_pickle('insert-dataset-path')

df['text'] = df["text"].astype('string')
df = df.dropna()
mylist = df.text.to_list()
mylist = list(dict.fromkeys(mylist))

model = AutoModelForSeq2SeqLM.from_pretrained("BlackKakapo/flan-t5-small-paraphrase-ro")
tokenizer = AutoTokenizer.from_pretrained("BlackKakapo/flan-t5-small-paraphrase-ro")
tokenized_inputs =[tokenizer.encode(text) for text in mylist]

outputs = []
for input_ids in tokenized_inputs:
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    output = model.generate(input_ids, max_length=512, no_repeat_ngram_size=2,pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0])
    outputs.append(generated_text)

ai_df = pd.DataFrame({'input': mylist, 'output': outputs})
ai_df["output"] = ai_df["output"].str.replace(r"<pad>", "")
ai_df['output'] = ai_df['output'].str.replace(r'<\/s>', " ")

output_frame= ai_df[["output"]]
output_frame['label'] = '0'
output_frame['description'] = 'This is AI-generated text'
output_frame.rename(columns = {'output':'text'}, inplace = True)
output_frame.to_csv('ai-text-flant5.csv',index = False, encoding = 'utf-8-sig')
