import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # 添加confusion_matrix

# 1. data
df = pd.read_csv('/your/datapath.csv', encoding='utf-8')

# 2. Data preprocessing
df.dropna(subset=['微博正文'], inplace=True)
df['cut_text'] = df['微博正文'].apply(lambda x: ' '.join(jieba.cut(x)))

# 3. Feature Extraction （TF-IDF）
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cut_text'])
y = df['label']

# 4. Divide the training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. definition model
svm_model = SVC(kernel='linear', probability=True, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 6. Training and Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    # （FPR）
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()  
    fpr = fp / (fp + tn)         
    print(f"False Positive Rate: {fpr:.4f}\n")

print("SVM Performance:")
svm_model.fit(X_train, y_train)
evaluate_model(svm_model, X_test, y_test)

print("\nRandom Forest Performance:")
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, X_test, y_test)