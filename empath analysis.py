import pandas as pd
from empath import Empath

lexicon = Empath()

df = pd.read_csv('suicide_final_cleaned.csv')
categories = ['sadness', 'anger', 'fear']

def analyze_texts(texts):
    category_scores = {category: [] for category in categories}
    for text in texts:
        analysis = lexicon.analyze(text, categories=categories)
        for category in categories:
            category_scores[category].append(analysis[category])
    mean_scores = {category: sum(scores) / len(scores) for category, scores in category_scores.items()}
    return mean_scores


suicidal_texts = df[df['class'] == 'suicide']['cleaned_texts'].tolist()
non_suicidal_texts = df[df['class'] == 'non-suicide']['cleaned_texts'].tolist()

suicidal_scores = analyze_texts(suicidal_texts)
non_suicidal_scores = analyze_texts(non_suicidal_texts)

# Print mean scores for each category
print("Suicidal Texts Analysis:")
for category, score in suicidal_scores.items():
    print(f'Mean {category} score: {score:.4f}')

print("\nNon-Suicidal Texts Analysis:")
for category, score in non_suicidal_scores.items():
    print(f'Mean {category} score: {score:.4f}')
