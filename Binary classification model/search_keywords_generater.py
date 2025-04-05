import csv
import re
import os
from collections import defaultdict
from tqdm import tqdm

def read_blacklist(blacklist_path):
    """Read blacklist file (one keyword to be excluded per line)"""
    if not blacklist_path or not os.path.exists(blacklist_path):
        return set()
    with open(blacklist_path, 'r', encoding='utf-8') as f:
        return {line.strip().replace('#', '') for line in f}  

#hashtag extraction function
def extract_hashtags(text, blacklist):
    
    pattern = r'#((?:紧急|爱心|公益|寻人|救援|助力|希望|转发|寻找|扩散|求助|接力|平安)[\u4e00-\u9fa5]{1,6})#'
    matches = re.findall(pattern, text)
    
    # Filter the blacklist and remove duplicates
    return list({tag for tag in matches if tag not in blacklist})
    

def is_valid_hashtag(tag):
    
    exclusion_rules = [
        r'\d',                    
        r'[岁天年月]',            
        r'(失踪|遗体|溺亡|立案|水库|遇难|死亡)',
        r'.{7,}',                 
        r'(男童|女童|男孩|女孩|男子|女子)'
    ]
    return not any(re.search(rule, tag) for rule in exclusion_rules)

def main():
    # ================ config ================
    input_csv = "/your predict_true/dataset.csv/path"
    output_csv = "your/high_value_seeds.csv/path"
    blacklist_path = "/data1/jrjiang/workspace/blacklist.txt"
    rcp_threshold = 0.01  # RCP阈值
    min_occurrences = 2  # 最小出现次数（过滤低频词）
    # ========================================

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"输入文件不存在：{input_csv}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    #redad blacklist
    blacklist = read_blacklist(blacklist_path)
    print(f"已加载黑名单关键词数量：{len(blacklist)}")

    # keyword statistics
    keyword_stats = defaultdict(lambda: {'total': 0, 'hits': 0})

    with open(input_csv, 'r', encoding='utf-8-sig') as f_in:
        reader = csv.DictReader(f_in)
        
        
        required_columns = {'label', '微博正文'}
        if not required_columns.issubset(reader.fieldnames):
            missing = required_columns - set(reader.fieldnames)
            raise ValueError(f"CSV文件缺少必要列：{', '.join(missing)}")

      
        total_rows = sum(1 for _ in open(input_csv, 'r', encoding='utf-8-sig')) - 1
        reader = tqdm(reader, total=total_rows, desc="分析微博数据")

        for row in reader:
            text = row['微博正文'].strip()
            label = row['label']
            
            # extract hashtags
            tags = [tag for tag in extract_hashtags(text, blacklist) if is_valid_hashtag(tag)]
            
            # update statistics
            for tag in tags:
                keyword_stats[tag]['total'] += 1
                if label == '1':
                    keyword_stats[tag]['hits'] += 1

    # extract high-value seeds
    high_value_seeds = []
    for tag, stats in keyword_stats.items():
        # RCP
        if stats['total'] >= min_occurrences and stats['hits'] > 0:
            rcp = stats['hits'] / stats['total']
            if rcp >= rcp_threshold:
               
                formatted_tag = f"#{tag}#"
                high_value_seeds.append((formatted_tag, 'hashtag'))

    # sort and save results
    high_value_seeds.sort(key=lambda x: x[0])  
    with open(output_csv, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['seed', 'type'])
        writer.writerows(high_value_seeds)

    
    print("\n生成报告：")
    print(f"总发现标签数：{len(keyword_stats)}")
    print(f"通过率（RCP ≥ {rcp_threshold}）：{len(high_value_seeds)}")
    print(f"结果文件已保存至：{output_csv}")

if __name__ == '__main__':
    main()
    