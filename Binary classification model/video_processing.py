import os
import pandas as pd
import whisper
from paddleocr import PaddleOCR
import subprocess

# path settings
csv_path = r"/your/text_only/path"
output_csv = r"/your output/path"
image_dir = r"/your/image/path"
video_dir = r"your/video/path"
temp_audio_path = "temp_audio0.wav"

# initialize models
ocr_model = PaddleOCR(use_angle_cls=True, lang="ch")
whisper_model = whisper.load_model("large-v2")

# get video duration using ffprobe
def get_video_duration(video_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-i", video_path, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return float(result.stdout.strip())
    except:
        return 0

# whisper 音频识别
def is_music(audio_path):
    # Using Whisper to identify audio and determine if it is pure music
    try:
        result = whisper_model.transcribe(audio_path, language="zh")
        return len(result["text"].strip()) == 0  #If the recognition result is empty, it is considered as music
    except:
        return True  

# Extract image text (supports multiple images)
def extract_image_text(image_id):
    # Find all image files that match the ID
    image_files = [f for f in os.listdir(image_dir) if f.startswith(image_id) and f.endswith((".jpg", ".png"))]
    
    if not image_files:
        return ""

    all_text = []
    for image_file in sorted(image_files):  # order by filename
        image_path = os.path.join(image_dir, image_file)
        try:
            ocr_result = ocr_model.ocr(image_path, cls=True)
            # extract text from OCR result
            text = "".join([line[1][0] for line in ocr_result[0]])
            all_text.append(text)
        except Exception as e:
            print(f"图片文字提取失败：{image_path}，错误：{e}")

    return "".join(all_text)  # Connect all image text


#Extract video audio and text
def extract_video_text(video_id):
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return "视频未找到"
    
    # Extract audio and determine if it is pure music
    try:
        video_duration = get_video_duration(video_path)
        audio_command = [
            "ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-f", "wav", temp_audio_path, "-y"
        ]
        subprocess.run(audio_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if is_music(temp_audio_path):
            return "music"
        
        # Transcribing speech
        result = whisper_model.transcribe(temp_audio_path, language="zh")
        return result["text"].strip()
    except Exception as e:
        print(f"视频语音提取失败：{video_path}，错误：{e}")
        return ""
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


#csv file processing
df = pd.read_csv(csv_path)


df["语音文字"] = ""
df["图片文字"] = ""

for index, row in df.iterrows():
    id = str(row["id"])
    video_url = row.get("微博视频url", "")
    image_url = row.get("微博图片url", "")

    # extract image text
    if pd.notna(image_url):
        df.at[index, "图片文字"] = extract_image_text(id)

    #   extract video text
    if pd.notna(video_url):
        df.at[index, "语音文字"] = extract_video_text(id)

# save results
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"处理完成，结果保存到 {output_csv}")    