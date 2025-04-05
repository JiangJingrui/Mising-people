import os
import pandas as pd
import numpy as np
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
from datasets import Dataset
import re
from pandarallel import pandarallel  
import psutil
import torch
import warnings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- pandarallel -------------------- #
pandarallel.initialize(nb_workers=psutil.cpu_count(logical=False))  #

#enviroment 
def setup_environment():
    
    os.environ['TRANSFORMERS_CACHE'] = '/your model path/bert-base-chinese'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"  

#Text cleaning function
def prepare_text(text):
   
    text = re.sub(re.compile(r'<(.*?)>', re.S), '', text)
    text = re.sub(r'\[([^\[\]]+)\]', r'\1', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'[\s\u3000]+', ' ', text)
    return text.strip()


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['label', '微博正文']].rename(columns={'微博正文': 'text'})

    df['text'] = df['text'].parallel_apply(prepare_text)  

    # class balance
    class_0 = df[df['label'] == 0]
    class_1 = df[df['label'] == 1]
    min_len = min(len(class_0), len(class_1))
    return pd.concat([class_0.sample(min_len, random_state=42),
                      class_1.sample(min_len, random_state=42)])


def main():
    # -------------------- config-------------------- #
    file_path = '/your data path/csv'  
    model_name = 'bert-base-chinese'  # model name
    max_seq_length = 128 
    batch_size = 32  
    epochs = 15  
    learning_rate = 2e-5  
    num_labels = 2  

    # -------------------- Data loading and preprocessing -------------------- #
    df = load_and_preprocess_data(file_path)
    train_df, eval_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42  
    )

    # -------------------- Initialize the model and tokenizer -------------------- #
    tokenizer = BertTokenizer.from_pretrained('/your model path/bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained(
        '/your model path/bert-base-chinese', num_labels=num_labels
    )

    # -------------------- data coding（HuggingFace Dataset） -------------------- #
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_tensors='pt'
        )
        tokenized['labels'] = torch.tensor(examples['label'], dtype=torch.long)
        return tokenized

    # Dataset
    train_dataset = Dataset.from_pandas(train_df).map(
        tokenize_function,
        batched=True,
        remove_columns=['text', 'label', '__index_level_0__'],  
        batch_size=32,
        num_proc=4
    )

    eval_dataset = Dataset.from_pandas(eval_df).map(
        tokenize_function,
        batched=True,
        remove_columns=['text', 'label', '__index_level_0__'],
        batch_size=32,
        num_proc=4
    )
    # -------------------- Print verification column names (optional) -------------------- #
    print("Train Dataset Columns:", train_dataset.column_names)  # ['input_ids', 'attention_mask', 'label']

    # -------------------- Define training parameters -------------------- #
    training_args = TrainingArguments(
        output_dir='./weibo_classifier',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",                     
        fp16=True
    )

    # -------------------- Define evaluation index calculation function -------------------- #
    def compute_metrics(pred):
      
        logits = pred.predictions
        labels = pred.label_ids
        preds = np.argmax(logits, axis=1)


        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='binary')
        precision = precision_score(labels, preds, average='binary')
        recall = recall_score(labels, preds, average='binary')


        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fpr': fpr  
        }

    # -------------------- Trainer-------------------- #
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    #training
    trainer.train()
    trainer.save_model('./weibo_classifier')  # Save the best model to the root directory
    tokenizer.save_pretrained('./weibo_classifier')

    # evaluation
    metrics = trainer.evaluate()
    # output
    print("\nbert-base-chinese Performance:")
    print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Precision: {metrics['eval_precision']:.4f}")
    print(f"Recall: {metrics['eval_recall']:.4f}")
    print(f"F1 Score: {metrics['eval_f1']:.4f}")
    print(f"False Positive Rate: {metrics['eval_fpr']:.4f}") 

if __name__ == "__main__":
    setup_environment()
    main()
    