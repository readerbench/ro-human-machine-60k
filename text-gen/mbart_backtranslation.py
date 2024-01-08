'''
Text Generation
Run mBART model [1] for backtranslation task.
Iterative translations: Ro-En-Ru-Ro.

[1] https://github.com/UKPLab/EasyNMT

Pre-requisite: install easynmt
'''

import pandas as pd
from easynmt import EasyNMT

# Load your input dataset (human text)
df=pd.read_pickle('insert-dataset-path')
df['text'] = df["text"].astype('string')
df = df.dropna()
mylist = df.text.to_list()
mylist = list(dict.fromkeys(mylist))

model = EasyNMT('mbart50_m2m')

max_length = 400

# ro -> en
src_lang='ro'
tgt_lang='en'

translated_texts_en = []
for input_text in mylist:
    # Split the input text into smaller chunks
    chunks = [input_text[i:i + max_length] for i in range(0, len(input_text), max_length)]
    #chunks = [input_text[i] for i in range(0, len(input_text))]

    # Translate each chunk and append the translations
    translated_chunks_en = []
    for chunk in chunks:
        # Translate the input chunk
        translated_text_en = model.translate(chunk,source_lang=src_lang, target_lang=tgt_lang)
        translated_chunks_en.append(translated_text_en)

    # Join the translated chunks into a single text
    translated_text_en = " ".join(translated_chunks_en)
    translated_texts_en.append(translated_text_en)

# Create a dataframe with the input and translated texts
ai_df = pd.DataFrame({'input': mylist, 'output': translated_texts_en})

# en -> ru
src_lang='en'
tgt_lang='ru'

# take previously generated translation and give it as input to next iteration
new_df = ai_df['output']
list_en = new_df.to_list()
list_en = list(dict.fromkeys(list_en))

translated_texts_ru = []
for input_text in list_en:
    chunks = [input_text[i:i + max_length] for i in range(0, len(input_text), max_length)]

    translated_chunk_ru = []
    for chunk in chunks:
        translated_text_ru = model.translate(chunk,source_lang=src_lang, target_lang=tgt_lang)
        translated_chunk_ru.append(translated_text_ru)

    translated_text_ru = " ".join(translated_chunk_ru)
    translated_texts_ru.append(translated_text_ru)

ai_df2 = pd.DataFrame({'input': list_en, 'output': translated_texts_ru})
ai_df2['output']=ai_df2['output'].str.join(' ')

# ru -> ro
src_lang='ru'
tgt_lang='ro'

# take previously generated translation and give it as input to next iteration
new_df = ai_df['output']
list_ru = new_df.to_list()
list_ru = list(dict.fromkeys(list_ru))

translated_texts_ro = []
for input_text in list_ru:
    chunks = [input_text[i:i + max_length] for i in range(0, len(input_text), max_length)]

    translated_chunk_ro = []
    for chunk in chunks:
        translated_text_ro = model.translate(chunk,source_lang=src_lang, target_lang=tgt_lang)
        translated_chunk_ro.append(translated_text_ro)

    translated_text_ro = " ".join(translated_chunk_ro)
    translated_texts_ro.append(translated_text_ro)

ai_df3 = pd.DataFrame({'input': list_ru, 'output': translated_texts_ro})
ai_df3['output']=ai_df3['output'].str.join(' ')

# Save it in format text - label- description
output_frame= ai_df3[["output"]]
output_frame['label'] = '0'
output_frame['description'] = 'This is AI-generated text'
output_frame.rename(columns = {'output':'text'}, inplace = True)
output_frame.to_csv('ai-text-mBART.csv',index = False, encoding = 'utf-8-sig')
