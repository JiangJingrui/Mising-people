import csv
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json5

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# 1. Load local model and tokenizer
model_path = "/your model /path/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Read and clean CSV data
def clean_text(text):
    """清洗异常字符和多余空格"""
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
        response = re.sub(r'</?\w+>', '', response)
        code_block = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
        if code_block:
            json_str = code_block.group(1)
        else:
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                return None
        return json_str.strip()
    except Exception as e:
        print(f"JSON提取异常: {str(e)}")
        return None

def parse_json(json_str):
    try:
        if not json_str.strip().endswith('}'):
            json_str = json_str.rstrip() + '}'
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r'//.*?\n', '', json_str)
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        json_str = re.sub(r'([{,]\s*)([A-Za-z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
        json_str = re.sub(r'\.\.\.', 'null', json_str)
        json_str = re.sub(r'(\w+)"t', r"\1't", json_str)

        return json5.loads(json_str)
    except Exception as e:
        print(f"json5解析失败: {str(e)}")
        print("待解析字符串:", json_str)
        return None

# 4. Structured transformation function（Zero-Shot版本）
def structure_text(original_text):
    """基于任务指令直接进行信息抽取（Zero-Shot）"""
    prompt = f"""请根据下面文本抽取关键信息，并生成严格JSON格式，包含以下结构：
{{
    "info": {{
        "name": "姓名（无则null）",
        "gender": "性别（男/女/null）",
        "missing_age": 年龄（数字/若无具体，则年龄失踪日期减去出生日期）,
        "missing_height": 身高厘米数（数字/null）,
        "missing_date": "日期（YYYY-MM-DD或YYYY）",
        "missing_location": "省",
        "missing_reason": "失踪原因"
        
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
        temperature=0.6,
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

# 5. process the data
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
