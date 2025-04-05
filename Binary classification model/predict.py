from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVClassifier:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.eval()
        
    def _predict_batch(self, texts, batch_size=32):
        predictions = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch = texts[i:i+batch_size]
            try:
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
                
            except Exception as e:
                logger.error(f"Batch {i//batch_size} failed: {str(e)}")
                predictions.extend([-1] * len(batch))
        return predictions

    def process_csv(self, input_path, output_path, text_column="微博正文"):
        try:
            # Read data and process column order
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} rows")
            
            if text_column not in df.columns:
                raise ValueError(f"Missing text column: {text_column}")

            
            df = df.dropna(subset=[text_column])
            
            # predict
            predictions = self._predict_batch(df[text_column].tolist())
            
            
            if 'label' in df.columns:
                
                label_pos = df.columns.get_loc('label') + 1
                df.insert(label_pos, 'pre_label', predictions)
            else:
                
                df.insert(1, 'pre_label', predictions)
                logger.warning("Input CSV has no 'label' column")

            # Reorder columns
            # Ensure 'label' and 'pre_label' are at the front
            ordered_cols = []
            if 'label' in df.columns:
                ordered_cols += ['label', 'pre_label']
            else:
                ordered_cols += ['pre_label']
                
            ordered_cols += [col for col in df.columns if col not in {'label', 'pre_label'}]
            df = df[ordered_cols]

            # save predictions
            df.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path} with column order: {df.columns.tolist()}")

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

if __name__ == "__main__":
    MODEL_PATH = "/your model/path/weibo_classifier"
    INPUT_CSV = "/your input/dataset.csv"
    OUTPUT_CSV = "/your output/prediction.csv"
    
    classifier = CSVClassifier(MODEL_PATH)
    classifier.process_csv(INPUT_CSV, OUTPUT_CSV)