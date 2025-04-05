import pandas as pd
import os
from sklearn.model_selection import train_test_split

# DATASET
df = pd.read_csv('/your/datapath.csv')

# data preprocessing
def preprocess_text(text):
    
    return str(text).replace('"', '').strip()

#preprocessing
df['微博正文'] = df['微博正文'].apply(preprocess_text)


df = df.dropna(subset=['微博正文', 'label'])
df = df[df['微博正文'].ne('')]  


formatted_data = df.apply(lambda row: f"{row['微博正文']}\t{row['label']}", axis=1)

# Divide the dataset (8:1:1)
train_data, temp_data = train_test_split(formatted_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# saving data
def save_to_txt(data, file_path):
    
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)  
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')


base_dir = '/your_dir/path'  
save_to_txt(train_data, os.path.join(base_dir, 'train.txt'))
save_to_txt(val_data, os.path.join(base_dir, 'dev.txt'))    
save_to_txt(test_data, os.path.join(base_dir, 'test.txt'))  

print(f"数据已成功保存至 {base_dir} 目录：")
print(f"- 训练集样本数: {len(train_data)}")
print(f"- 验证集样本数: {len(val_data)}")
print(f"- 测试集样本数: {len(test_data)}")