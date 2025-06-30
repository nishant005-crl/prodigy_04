import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from io import StringIO

# Simulated CSV data with quotes around tweets
data = StringIO("""Tweet
"I love the new iPhone! Best one yet."
"Terrible customer service at the bank today."
"The weather is nice and I feel great."
"I’m so disappointed with this product."
"Not bad, but could be better."
"Absolutely fantastic experience!"
"Worst update ever, it broke everything."
"This is just okay, nothing special."
"Thanks for the quick response!"
"I'm never buying from this company again."
""")

# Load the dataset
df = pd.read_csv(data, quotechar='"')

# Sentiment analysis function
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df['Sentiment'] = df['Tweet'].apply(get_sentiment)

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Sentiment', palette='Set2')
plt.title("Sentiment Distribution of Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.show()

# Print categorized sample tweets
print("\n✅ Positive Tweets:\n", df[df['Sentiment'] == 'Positive']['Tweet'].to_string(index=False))
print("\n❌ Negative Tweets:\n", df[df['Sentiment'] == 'Negative']['Tweet'].to_string(index=False))
print("\n😐 Neutral Tweets:\n", df[df['Sentiment'] == 'Neutral']['Tweet'].to_string(index=False))
