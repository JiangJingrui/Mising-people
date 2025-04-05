import csv
import os

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 1. model和tokenizer加载
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/your model/path/glm-4-9b-chat'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
   ).eval() 
model = model.to(device)

# 2. Read and clean CSV data
def clean_text(text):
    text = re.sub(r"�+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff，。！？、；：]", " ", text)
    return text.strip()

def read_csv(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and row[0]:
                cleaned_text = clean_text(row[0])
                data.append({"text": cleaned_text})
    return data


# 3. JSON processing functions
def extract_json(response):
    try:
        code_block = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
        if code_block:
            return code_block.group(1)
        json_match = re.search(r'{.*}', response, re.DOTALL)
        return json_match.group() if json_match else None
    except Exception:
        return None

def parse_json(json_str):
    try:
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r'//.*?\n', '', json_str)  
        json_str = re.sub(r',\s*}', '}', json_str)  
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON解析失败: {str(e)}")
        return None

# 4.Structured transformation function（Zero-Shot版本）
def structure_text(original_text):
    """
    直接根据任务指令进行信息抽取（Zero-Shot），输出格式为：
    {
        "info": {
            "name": "姓名（无则null）",
            "gender": "性别（男/女/null）",
            "missing_age": 年龄（数字/null）,
            "missing_height": 身高厘米数（数字/null）,
            "missing_date": "日期（YYYY-MM-DD或YYYY）",
            "missing_location": "省+市",
            "missing_reason": "失踪原因",
            "description": "特征描述（包含详细地址）"
        }
    }
    """
    prompt = f"""请根据下面文本抽取关键信息，并生成严格JSON格式，要求输出内容满足以下结构：
{{
    "info": {{
        "name": "姓名（无则null）",
        "gender": "性别（男/女/null）",
        "missing_age": 年龄（数字/若无具体，则年龄失踪日期减去出生日期）,
        "missing_height": 身高厘米数（数字/null）,
        "missing_date": "日期（YYYY-MM-DD或YYYY）",
        "missing_location": "省",
        "missing_reason": "失踪原因",
        "description": "特征描述（包含详细地址）"
    }}
}}

文本：{original_text}
输出JSON："""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(inputs.input_ids)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=500,
        temperature=0.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_str = extract_json(full_response.split("输出JSON：")[-1])
    
    if not json_str:
        print("未提取到 JSON 字符串")
        return None
    
    result = parse_json(json_str)
    if result is None:
        print("解析失败: 返回结果为空")
        return None

    return {
        "text": original_text,
        "info": result.get("info", {})
    }

# 5.process the data
if __name__ == "__main__":
    raw_data = read_csv("/your Pending data/input path/")
    
    structured_data = []
    for idx, item in enumerate(raw_data):
        print(f"处理进度: {idx+1}/{len(raw_data)}")
        result = structure_text(item["text"])
        if result and result["info"]:
            info = result["info"]
            if info.get("missing_date"):
                date_str = str(info["missing_date"])
                if re.match(r"\d{8}", date_str):
                    info["missing_date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            structured_data.append(result)

     # save the structured data to JSON
    with open("/your output json/path/", "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功处理 {len(structured_data)}/{len(raw_data)} 条数据")