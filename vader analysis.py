import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

df = pd.read_csv('suicide_final_cleaned.csv')

def analyze_texts_vader(texts):
    sentiment_scores = {'pos': [], 'neu': [], 'neg': []}
    for text in texts:
        vs = analyzer.polarity_scores(text)
        for key in sentiment_scores.keys():
            sentiment_scores[key].append(vs[key])
    mean_scores = {key: sum(scores) / len(scores) for key, scores in sentiment_scores.items()}
    return mean_scores

suicidal_texts = df[df['class'] == 'suicide']['cleaned_texts'].tolist()
non_suicidal_texts = df[df['class'] == 'non-suicide']['cleaned_texts'].tolist()

suicidal_scores_vader = analyze_texts_vader(suicidal_texts)
non_suicidal_scores_vader = analyze_texts_vader(non_suicidal_texts)

print("Suicidal Texts VADER Analysis:")
for category, score in suicidal_scores_vader.items():
    print(f'Mean {category} score: {score:.4f}')

print("\nNon-Suicidal Texts VADER Analysis:")
for category, score in non_suicidal_scores_vader.items():
    print(f'Mean {category} score: {score:.4f}')
