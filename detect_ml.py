'''
ML-based Detector
Feature selection: extract ReaderBench Textual Complexity Indices (RBI) and use Kruskal-Wallis Test (Mean Ranks) to select top 100 RBIs; in parallel compute burstiness.
Input features: top 100 RBIs x burstiness score
Perform Binary Text Classification with XGBoost
SHAP Analysis 
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from rb import Document, Lang
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.similarity.vector_model_factory import (VectorModelType, create_vector_model)
from tqdm.notebook import tqdm_notebook as tqdm
import shap
from xgboost import XGBClassifier
from scipy.stats import mstats
import matplotlib.pyplot as plt

human_texts=pd.read_excel('domain1-human.xlsx')
machine_texts=pd.read_excel('domain1-machine.xlsx')

lang = Lang.RO

all_texts = pd.concat([human_texts, machine_texts]).reset_index(drop=True)
all_texts['label'] = np.concatenate([np.ones(len(human_texts)), np.zeros(len(machine_texts))])

def calculate_burstiness(text):
    words = text.split()
    if len(words) > 1:
        mean_word_length = np.mean([len(word) for word in words])
        std_deviation = np.std([len(word) for word in words])
        burstiness = std_deviation / mean_word_length
    else:
        burstiness = 0.0
    return burstiness
    
rb_indices_list = []

# Compute RBIs
for text, label in tqdm(zip(all_texts['text'], all_texts['label']), desc="Computing RB Indices", ncols=100):
    doc = Document(lang, text)
    model = create_vector_model(lang, VectorModelType.TRANSFORMER, "")
    model.encode(doc)
    cna_graph = CnaGraph(docs=doc, models=[model])
    compute_indices(doc=doc, cna_graph=cna_graph)
    result = {
        'label': int(label),
        **{
            str(index): float(value) if value is not None else None
            for index, value in doc.indices.items()
        }
    }
    rb_indices_list.append(result)

rb_indices_df = pd.DataFrame(rb_indices_list)

group_0 = rb_indices_df[rb_indices_df['label'] == 0]
group_1 = rb_indices_df[rb_indices_df['label'] == 1]

results = []

# Loop through each index column and perform Kruskal-Wallis test
for column in rb_indices_df.columns[1:]:  # exclude the first column (label/group)
    data_group_0 = group_0[column].to_numpy()
    data_group_1 = group_1[column].to_numpy()

    # Filter out None values
    data_group_0 = data_group_0[pd.notna(data_group_0)]
    data_group_1 = data_group_1[pd.notna(data_group_1)]

    if len(np.unique(data_group_0)) > 1 and len(np.unique(data_group_1)) > 1:
        stat, p_value = mstats.kruskalwallis(data_group_0, data_group_1)
        results.append((column, p_value, stat))

sorted_results = sorted(results, key=lambda x: x[1])[:100]

print("top 100 most significant indices:")
top_100_indices = [result[0] for result in sorted_results[:100]]
for i, result in enumerate(sorted_results[:100], start=1):
    print(f"{i}. Index: {result[0]}, Statistic: {result[2]}, p-value: {result[1]}")


combined_df = pd.concat([all_texts, rb_indices_df], axis=1)
combined_df['burstiness'] = combined_df['text'].apply(calculate_burstiness)

X_train, X_test, y_train, y_test = train_test_split(combined_df[top_100_indices + ['burstiness']], combined_df['label'], test_size=0.2, random_state=42, stratify=combined_df['label'])

y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 4, 5],
    'scale_pos_weight': [1, 3, 5, 7],
    'reg_alpha': [0.1, 0.01, 0.001],
    'reg_lambda': [0.1, 0.01, 0.001],
}

clf = XGBClassifier(seed=42)

grid_search = GridSearchCV(clf, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1_macro') # zero_division=1

grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best F1 Score: ", grid_search.best_score_)
print("\n ")

best_clf = grid_search.best_estimator_

best_clf.fit(X_train, y_train)

test_preds = best_clf.predict(X_test)
print(classification_report(y_test, test_preds))  # zero_division=1

def classify_text(df, feature_matrix):
    predicted_labels = best_clf.predict(feature_matrix)  # Use best_clf instead of clf
    probabilities = best_clf.predict_proba(feature_matrix)[:, 1]  # Use best_clf instead of clf
    true_labels = df['label'].tolist()
    descriptions = ['Human-generated' if label == 1 else 'Machine-generated' for label in true_labels]
    data = {'true_label': true_labels,
            'predicted_label': predicted_labels,
            'probability_score': probabilities,
            'description': descriptions}
    return pd.DataFrame(data)

output_df = classify_text(rb_indices_df.iloc[y_test.index], X_test)

output_df['text'] = all_texts.loc[rb_indices_df.iloc[y_test.index].index, 'text'].tolist()

human_count = output_df[output_df['true_label'] == 1]['true_label'].count()
machine_count = output_df[output_df['true_label'] == 0]['true_label'].count()
print("Number of human texts in the test set:", human_count)
print("Number of machine-generated texts in the test set:", machine_count)

category_counts = output_df.groupby(['true_label', 'description', 'predicted_label']).agg({
    'probability_score': 'mean',
    'text': 'count'
}).reset_index()

category_counts['true_label'] = category_counts['true_label'].astype(int)
category_counts.columns = ['true_label', 'description', 'predicted_label', 'probability_score_avg', 'count']
category_counts['misclassified'] = category_counts.apply(lambda row: row['count'] if row['true_label'] != row['predicted_label'] else 0, axis=1)

print(category_counts.to_string(index=False))

shap_explainer = shap.Explainer(best_clf, X_train)
shap_values = shap_explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=top_100_indices+ ['burstiness'], show=False)

plt.title("SHAP Summary Plot")
plt.show()

