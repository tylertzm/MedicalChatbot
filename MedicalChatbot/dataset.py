import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import scipy.sparse as sp


# Step 1: Load and clean the dataset
df = pd.read_csv("/Users/zhen-meitan/Desktop/Personal/Uni/Projektstudium/BertModel/dataset.csv")

def clean_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

df['symptoms'] = df['symptoms'].apply(clean_text)
df['diagnosis'] = df['diagnosis'].apply(clean_text)

# Step 2: Encoding the labels
X = df[['symptoms', 'diagnosis']]
y = df['diseases']

# Step 3: Remove classes with fewer than 2 samples
class_counts = y.value_counts()
classes_to_remove = class_counts[class_counts < 2].index
df_filtered = df[~df['diseases'].isin(classes_to_remove)]
X_filtered = df_filtered[['symptoms', 'diagnosis']]
y_filtered = df_filtered['diseases']

# Re-encode labels after removing rare classes
label_encoder = LabelEncoder()
y_filtered = label_encoder.fit_transform(y_filtered)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_symptoms = vectorizer.fit_transform(X_train['symptoms'])
X_train_diagnosis = vectorizer.transform(X_train['diagnosis'])
X_test_symptoms = vectorizer.transform(X_test['symptoms'])
X_test_diagnosis = vectorizer.transform(X_test['diagnosis'])

X_train_combined = sp.hstack([X_train_symptoms, X_train_diagnosis])
X_test_combined = sp.hstack([X_test_symptoms, X_test_diagnosis])

# Step 5: Train a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_combined, y_train)

# Step 6: Make predictions and evaluate
y_pred = classifier.predict(X_test_combined)
print(classification_report(y_test, y_pred))

