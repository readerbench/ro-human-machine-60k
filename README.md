# ro-human-machine-60k

Romanian multidomain human-machine dataset

| Domain   | Method        | Model          | Avg TTR | Aggregate |
|----------|---------------|----------------|---------|-----------|
| Books    | Human         | Human          | 0.7447  | 11,208    |
|          | Completion    | RoGPT2         | 0.6615  |           |
|          | Completion    | GPT-Neo-Ro      | 0.7011  |           |
|          | Completion    | davinci-003     | 0.6125  |           |
|          | Backtranslation| davinci-003     | 0.7652  |           |
|          | Paraphrasing  | Flan-T5         | 0.8708  |           |
|          | Backtranslation| Opus-MT         | 0.7581  |           |
|          | Backtranslation| mBART           | 0.7379  |           |
| News     | Human         | Human          | 0.6510  | 34,560    |
|          | Completion    | RoGPT2         | 0.6762  |           |
|          | Completion    | GPT-Neo-Ro      | 0.6867  |           |
|          | Completion    | davinci-003     | 0.6508  |           |
|          | Backtranslation| davinci-003     | 0.7798  |           |
|          | Paraphrasing  | Flan-T5         | 0.8389  |           |
|          | Backtranslation| Opus-MT         | 0.6589  |           |
|          | Backtranslation| mBART           | 0.7024  |           |
| Medical  | Human         | Human          | 0.6911  | 4,456     |
|          | Completion    | RoGPT2         | 0.6795  |           |
|          | Completion    | GPT-Neo-Ro      | 0.6893  |           |
|          | Completion    | davinci-003     | 0.6262  |           |
|          | Backtranslation| davinci-003     | 0.7510  |           |
|          | Paraphrasing  | Flan-T5         | 0.8503  |           |
|          | Backtranslation| Opus-MT         | 0.7490  |           |
|          | Backtranslation| mBART           | 0.7618  |           |
| Legal    | Human         | Human          | 0.7264  | 8,000     |
|          | Completion    | RoGPT2         | 0.6542  |           |
|          | Completion    | GPT-Neo-Ro      | 0.6880  |           |
|          | Completion    | davinci-003     | 0.5828  |           |
|          | Backtranslation| davinci-003     | 0.7987  |           |
|          | Paraphrasing  | Flan-T5         | 0.8418  |           |
|          | Backtranslation| Opus-MT         | 0.7231  |           |
|          | Backtranslation| mBART           | 0.7514  |           |
| RoCHI    | Human         | Human          | 0.6234  | 872       |
|          | Completion    | RoGPT2         | 0.6901  |           |
|          | Completion    | GPT-Neo-Ro      | 0.5460  |           |
|          | Completion    | davinci-003     | 0.5810  |           |
|          | Backtranslation| davinci-003     | 0.7514  |           |
|          | Paraphrasing  | Flan-T5         | 0.8356  |           |
|          | Backtranslation| Opus-MT         | 0.6032  |           |
|          | Backtranslation| mBART           | 0.7477  |           |
| Total    |               |                |           | 59,096    |

# detection of machine generated text

![MGT](https://github.com/readerbench/ro-mgt-detection/blob/main/method.png)

# paper

Cite this as: reference paper to be uploaded soon.
