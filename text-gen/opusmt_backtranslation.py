'''
Text Generation - Backtranslation (Ro-Fr-Es-Ro)
Run Opus-MT model [1] for backtrasnlation task.

[1] https://github.com/UKPLab/EasyNMT

Pre-requisite: install easynmt
'''

import pandas as pd
from easynmt import EasyNMT
import tqdm

# Load input dataset (human text)
df=pd.read_pickle('insert-dataset-path')
df['text'] = df["text"].astype('string')
df = df.dropna()
mylist = df.text.to_list()
mylist = list(dict.fromkeys(mylist))

model = EasyNMT('opus-mt',max_loaded_models=2)

# RO -> FR -> ES -> RO (iteratively change the source and target languages)
src_lang='ro'
tgt_lang='fr'

# Set the desired max_length for each chunk
#max_length = 500

# Translate the texts
translated_texts = []
for input_text in mylist:
    # Split the input text into smaller chunks
    #chunks = [input_text[i:i + max_length] for i in range(0, len(input_text), max_length)]
    chunks = [input_text[i] for i in range(0, len(input_text))]

    # Translate each chunk and append the translations
    translated_chunks = []
    for chunk in chunks:
        # Translate the input chunk to English
        translated_text = model.translate(chunk,source_lang=src_lang, target_lang=tgt_lang)
        translated_chunks.append(translated_text)

    # Join the translated chunks into a single text
    translated_text = " ".join(translated_chunks)
    translated_texts.append(translated_text)

# Create a dataframe with the input and translated texts
ai_df = pd.DataFrame({'Input': mylist, 'Output': translated_texts})

# Save it in format text - label- description
output_frame= ai_df[["output"]]
output_frame['label'] = '0'
output_frame['description'] = 'This is AI-generated text'
output_frame.rename(columns = {'output':'text'}, inplace = True)
output_frame.to_csv('ai-text-opusMT.csv',index = False, encoding = 'utf-8-sig')
