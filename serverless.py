import base64
from moviepy import VideoFileClip, concatenate_videoclips
from openai import OpenAI
import requests
import cv2
import subprocess
import glob
import re
import os
import time
import math
import json
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pydub import AudioSegment
import runpod
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
STT_ENDPOINT = os.environ.get("STT_ENDPOINT")
TTS_ENDPOINT = os.environ.get("TTS_ENDPOINT")

# Language 목록
language = {"cmn": "표준 중국어", "en-us": "미국 영어", "ja": "일본어", "ko": "한국어", "vi": "베트남어"}

# format file to base64
def file_to_base64(file: str):
    with open(file, "rb") as f:
        audio_data = f.read()
        base64_data = base64.b64encode(audio_data).decode('utf-8')
    return base64_data

# format base64 to file:
def base64_to_file(base64_data, output_file):
    data = base64.b64decode(base64_data)
    print('base64_to_file', output_file)
    with open(output_file, "wb") as file:
        file.write(data)

# Faster Whisper (STT)
def stt_faster_whisper(audio_base64: str, api_key, endpoint_id):
    """
        audio_base64: 오디오 파일(.mp3, .wav),
        api_key: Runpod API KEY,
        endpoint_id: STT Endpoint
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "audio": audio_base64
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if "id" not in response_data:
        print(f"Error : {response_data}")
        return None
    
    job_id = response_data["id"]
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

    while True:
        status_response = requests.get(status_url, headers=headers)
        status_data = status_response.json()

        if status_data.get("status") == "COMPLETED":
            return status_data.get("output")
        elif status_data.get("status") in ["FAILED", "CANCELLED"]:
            print(f"Job Failed or Cancelled: {status_data}")
            return None
        time.sleep(3)

# Segments Video Split
def format_time(seconds):
    """
        초(int) -> timestamp
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def segment_split_video(origin_video, video_file: str, output_prefix, stt_segment, max_segment_duration=20):
    """
        video_file: 원본 비디오 파일명,
        output_prefix: 저장될 디렉토리명,
        stt_segment: STT 결과값,
        max_segment_duration: 자를 최대 시간
    """
    logger.info("비디오 분할 시작 함수")
    logger.info(f"origin_video: {origin_video}")
    logger.info(f"video_file: {video_file}")
    logger.info(f"output_prefix: {output_prefix}")

    os.makedirs(output_prefix, exist_ok=True) # 자른 파일들 저장될 디렉토리

    grouped_segments = []
    current_group = []
    current_duration = 0
    segments_data = []

    for segment in stt_segment:
        segment_duration = segment["end"] - segment["start"]
        if segment_duration > max_segment_duration:
            if current_group:
                grouped_segments.append(current_group)
                current_group = []
                current_duration = 0
            split_count = math.ceil(segment_duration / max_segment_duration)
            split_duration = segment_duration / split_count
            for i in range(split_count):
                start = segment["start"] + i * split_duration
                end = min(segment["start"] + (i + 1) * split_duration, segment["end"])
                split_segment = {
                    "start": start,
                    "end": end,
                    "text": segment["text"] if i == 0 else "[계속]"
                }
                grouped_segments.append([split_segment])
        elif current_duration + segment_duration > max_segment_duration:
            if current_group:
                grouped_segments.append(current_group)
            current_group = [segment]
            current_duration = segment_duration
        else:
            current_group.append(segment)
            current_duration += segment_duration
    
    if current_group:
        grouped_segments.append(current_group)

    for i, group in enumerate(grouped_segments):
        if not group:
            continue
        if i == 0:
            start_time = 0
        else:
            start_time = end_time

        # 중간에 잘리는 경우가 있어서, end에 0.5초 추가, 수정 가능..
        end_time = group[-1]['end'] 

        start_str = format_time(start_time)
        end_str = format_time(end_time)

        segments_data.append([start_time, end_time])

        output_file = os.path.join(output_prefix, f"{origin_video.split('.')[0]}_{i}.mp4")

        # ffmpeg segment video
        cmd = [
            "ffmpeg",
            "-i", video_file,
            "-ss", start_str,
            "-to", end_str,
            "-c:v", "libx264",
            "-c:a", "copy",
            output_file
        ]

        subprocess.run(cmd)
    
    return segments_data

def split_video_to_audio(output_prefix, origin_video):
    video_list = glob.glob(f"./{output_prefix}/*.mp4")
    for i in range(len(video_list)):
        # 파일 크기가 0인지 확인
        if os.path.getsize(video_list[i]) == 0:
            logger.warning(f"Skipping empty video file: {video_list[i]}")
            continue
            
        try:
            video = VideoFileClip(video_list[i])
            video.audio.write_audiofile(os.path.join(output_prefix, origin_video.split(".")[0] + "_" + str(i)) + ".mp3")
        except Exception as e:
            logger.error(f"Error processing video {video_list[i]}: {str(e)}")


def extract_text_by_segments(translation_result: list, segments_list: list):
    """
        stt_result: faster whisper 결과 segments,
        segments_list: 나눠진 start, end 리스트
    """
    seg_result = []
    print(f"segments_list : {segments_list}")
    print(f"translation_result : {translation_result}")
    for segment_range in segments_list:
        print(f"segment_range : {segment_range}")
        start_time, end_time = segment_range
        segment_texts = []
        for translation in translation_result:
            print(f"translation : {translation}")
            # segment가 시간 구간과 겹치는지 확인
            if translation["end"] > start_time and translation["start"] < end_time:
                segment_texts.append({'start': translation['start'], 'end': translation['end'], 'text': translation['text']})
        
        seg_result.append(segment_texts)

    result = []
    # segment의 이전 end, 다음 start 차이가 0.15 이내거나 같다면, segment를 붙여줌
    for seg in seg_result:
        new_seg = []
        i = 0
        while i < len(seg):
            current = seg[i]
            while i + 1 < len(seg) and (current['end'] == seg[i+1]['start'] or seg[i+1]['start'] - current['end'] <= 0.15):
                next_item = seg[i+1]
                current['end'] = next_item['end']
                current['text'] += ' ' + next_item['text']
                i += 1
            new_seg.append(current)
            i += 1
        result.append(new_seg)

    return result
    

# LLM 번역, 현재 Gemini -> OpenAI 수정 필요
def translate(language: str, text_list, api_key):
    """
        language: {"cmn": "표준 중국어", "en-us": "미국 영어", "ja": "일본어", "ko": "한국어", "vi": "베트남어"},
        texts: 나눠진 stt text 리스트
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt_ = f"""
    {text_list}
    
    이 리스트에서 각 딕셔너리의 'text' 키에 해당하는 문장을 {language}로 번역해줘. 
    번역 결과는 원본과 동일한 구조를 유지하되, 'text' 값만 {language}로 번역된 새로운 리스트로 반환해줘.
    즉, 'start'와 'end' 값은 그대로 유지하고, 'text' 값만 {language}로 번역해줘.
    
    예를 들어 첫 번째 항목은 다음과 같이 변환되어야 해:
    {{'end': {text_list[0]['end']}, 'start': {text_list[0]['start']}, 'text': '[여기에 번역문]'}}
    
    전체 리스트를 {language}로 번역할 수 있도록 Python 리스트 형식으로 반환해줘. 다른 구문들은 '\n' 포함해서 절대 붙이지 말고 리스트 형식으로 반환해줘.
    그리고, 첫 시작하는 문자는 반드시 '['이 되어야 해. {language}로 번역해줘
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates text while preserving JSON structure."},
            {"role": "user", "content": prompt_}
        ],
        top_p=0.95,
        temperature=0.2,
        max_tokens=8192,
    )

    response_text = response.choices[0].message.content
    if '\n' in response_text:
        response_text = response_text.replace('\n', '')
    
    response_text = response_text.strip('`"\n ')
    response_text = response_text.replace("'",'"')
    
    return json.loads(response_text)
        
# TTS Zonos
def tts_zonos(model_type, endpoint_id, api_key, language, text, reference_audio_path,
              speaking_rate=15.0, pitch_std=20.0, fmax=22050.0,
              emotion_happiness=0.3077, emotion_sadness=0.0256,
              emotion_disgust=0.0256, emotion_fear=0.0256,
              emotion_suprise=0.0256, emotion_anger=0.0256,
              emotion_other=0.2564, emotion_neutral=0.3077,
              vqscore_8=[0.78]*8, ctc_loss=0.0, dnsmos_ovrl=4.0, speaker_noised=False):
    """
        model_type: transformer, hybrid(Mamba),
        endpoint_id: TTS Endpoint,
        api_key: Runpod API KEY,
        language: {"cmn": "표준 중국어", "en-us": "미국 영어", "ja": "일본어", "ko": "한국어", "vi": "베트남어"},
        text: 번역된 텍스트,
        reference_audio_path: Voice cloning할 음성, 1분 정도의 참조할 음성 파일 위치,
        speaking_rate: 발화 속도,
        pitch_std: 감정 표현, 높을수록 감정적(0~400),
        fmax: 최대 오디오 주파수,
        emotion: 감정 수치(0.0~1.0),
        vqscore_8: 값이 높을수록 깨끗한 음성, 감정 표현이 높아짐(?),
        ctc_loss: 0 또는 1,
        dnsmos_ovrl: 감정 및 언어와 관계가 있다고 함.
        speaker_noised: 화자의 음성이 노이즈가 있는지 여부
    """
    reference_audio_path = file_to_base64(reference_audio_path)

    payload = {
        "input": {
            "model_type": model_type,
            "text": text,
            "reference_audio": reference_audio_path,
            "language": language,
            "speaking_rate": speaking_rate,
            "pitch_std": pitch_std,
            "fmax": fmax,
            "emotion_happiness": emotion_happiness,
            "emotion_sadness": emotion_sadness,
            "emotion_disgust": emotion_disgust,
            "emotion_fear": emotion_fear,
            "emotion_suprise": emotion_suprise,
            "emotion_anger": emotion_anger,
            "emotion_other": emotion_other,
            "emotion_neutral": emotion_neutral,
            "vqscore_8": vqscore_8,
            "ctc_loss": ctc_loss,
            "dnsmos_ovrl": dnsmos_ovrl,
            "speaker_noised": speaker_noised
        }
    }

    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if "id" not in response_data:
        print(f"Error : {response_data}")
        return None
    
    job_id = response_data["id"]
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

    while True:
        status_response = requests.get(status_url, headers=headers)
        status_data = status_response.json()

        if status_data.get("status") == "COMPLETED":
            return status_data.get("output")
        elif status_data.get("status") in ["FAILED", "CANCELLED"]:
            print(f"Job failed or cancelled: {status_data}")
            return None
        
        time.sleep(3)

# LatentSync
def latentsync(output_prefix, vast_instance_url, vast_instance_port, mode_):
    """
        output_prefix: 저장될 디렉토리,
        vast_instance_url: vast instance ip 주소
        vast_instance_port: vast instance flask api 포트
        mode_:
            - normal
            - pingpong: Audio가 길 경우 비디오 역재생으로 길이를 맞춤
            - loop_to_audio: Audio가 길 경우 비디오 처음부터 다시 재생하여 길이를 맞춤
    """
    video_files = glob.glob(f"./{output_prefix}/*.mp4")
    convert_audio_files = glob.glob(f"./{output_prefix}/combined_*.mp3")

    for i in range(len(video_files)):
        video_cap = cv2.VideoCapture(video_files[i])
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_data = file_to_base64(video_files[i])
        audio_data = file_to_base64(convert_audio_files[i])

        headers = {"Content-Type": "application/json"}
        payload = {
            "video": video_data,
            "audio": audio_data,
            "video_name": os.path.basename(video_files[i]),
            "custom_width": width,
            "custom_height": height,
            "force_rate": 25,
            "fps": 25.0,
            "lips_expression": 1.5,
            "inference_steps": 20,
            "mode": mode_
        }
        response = requests.post(f"http://{vast_instance_url}:{vast_instance_port}/latentsync", headers=headers, json=payload, timeout=1200)

        if response.status_code == 200:
            result = response.json()
            if result.get("success", True):
                os.makedirs(os.path.join(output_prefix, "latentsync"), exist_ok=True)
                output_filename = result["output"]["video_name"]
                output_data = result["output"]["video_data"]
                output_path = os.path.join(output_prefix, 'latentsync', output_filename)
                base64_to_file(output_data, output_path)
            else:
                print(f"Failed latentsync: {result.get('error', '...')}")
        else:
            print(f"API Error: {response.status_code}. {response.text}")

# Audio - Background Sound Split & loud Background Sound
def split_video_to_bg(output_prefix, split_audio_file, bg_model, device_, bg_dir, cnt):
    """
        split_audio_file: split했던 원본 Audio파일
        bg_model: htdemucs model
        device: cuda or cpu
        bg_dir: bg 저장될 임시 디렉토리 -> 임시처리 Serverless 환경에서 처리하기 위해 수정 필요
        cnt: split video count number
    """
    audio, sample_rate = torchaudio.load(split_audio_file)
    if audio.shape[0] == 1:
        audio = torch.cat([audio, audio], dim=0)
    
    audio = audio.unsqueeze(0)
    sources = apply_model(bg_model, audio.to(device_), device=device_)
    source_names = bg_model.sources

    for i, source_name in enumerate(source_names):
        source_audio = sources[0, i]
        torchaudio.save(
            f"{bg_dir}/{source_name}.mp3",
            source_audio.cpu(),
            sample_rate
        )

    vocals_idx = source_names.index("vocals") if "vocals" in source_names else None
    if vocals_idx is not None:
        bg = torch.zeros_like(sources[0, 0])
        for j, name in enumerate(glob.glob(f"{bg_dir}/*.mp3")):
            if "vocals.mp3" not in name:
                bg += sources[0, j]
        torchaudio.save(f"{output_prefix}/bg_{cnt}.mp3", bg.cpu(), sample_rate)
    [os.remove(file) for file in glob.glob(f"{bg_dir}/*.mp3")]

    cmd = [
        "ffmpeg",
        "-i", f"{output_prefix}/bg_{cnt}.mp3",
        "-af", "loudnorm=I=-23:LRA=7:TP=-2",
        f"{output_prefix}/bg_{cnt}_loud.mp3"
    ]
    subprocess.run(cmd)
    os.remove(f"{output_prefix}/bg_{cnt}.mp3")

# 별도 RunPod Serverless 엔드포인트에서 Demucs 실행하는 함수
def demucs_runpod_serverless(audio_file, api_key, endpoint_id, output_prefix, cnt):
    """
    RunPod Serverless GPU 인스턴스에서 Demucs 실행
    
    Args:
        audio_file: 분리할 오디오 파일 경로
        api_key: RunPod API 키
        endpoint_id: Demucs 서비스용 RunPod 엔드포인트 ID
        output_prefix: 출력 파일 저장 디렉토리
        cnt: 분할 비디오 카운트 번호
    """
    audio_base64 = file_to_base64(audio_file)
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "audio": audio_base64
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()
    
    if "id" not in response_data:
        print(f"오류: {response_data}")
        return None
    
    job_id = response_data["id"]
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    
    while True:
        status_response = requests.get(status_url, headers=headers)
        status_data = status_response.json()
        
        if status_data.get("status") == "COMPLETED":
            result = status_data.get("output")
            if "background_audio" in result:
                base64_to_file(result["background_audio"], f"{output_prefix}/bg_{cnt}.mp3")
                
                # 배경 음악 볼륨 정규화
                cmd = [
                    "ffmpeg",
                    "-i", f"{output_prefix}/bg_{cnt}.mp3",
                    "-af", "loudnorm=I=-23:LRA=7:TP=-2",
                    f"{output_prefix}/bg_{cnt}_loud.mp3"
                ]
                subprocess.run(cmd)
                os.remove(f"{output_prefix}/bg_{cnt}.mp3")
                
                return True
            return False
        elif status_data.get("status") in ["FAILED", "CANCELLED"]:
            print(f"작업 실패 또는 취소됨: {status_data}")
            return False
        
        time.sleep(3)

def group_combined_segment(divide_text, output_prefix, origin_video):
    """
        divide_text: 각 segment(dict)로 이루어진 list
    """
    # 침묵 구간 설정
    silence_duration = 0
    for group_idx, group in enumerate(divide_text):
        result_audio = AudioSegment.silent(duration=0)
        current_position = 0
        previous_end = 0
        for segment_idx, segment in enumerate(group):
            original_start = segment['start'] * 1000
            original_end = segment['end'] * 1000
            audio_path = os.path.join(output_prefix, f"convert_{origin_video[:-4]}_{group_idx}_{segment_idx}_loud.mp3")

            if os.path.exists(audio_path):
                audio = AudioSegment.from_file(audio_path)
                actual_duration = len(audio)
                original_duration = original_end - original_start

                if segment_idx == 0:
                    current_position = original_start
                else:
                    if original_start > previous_end:
                        silence_duration = original_start - previous_end
                        result_audio = result_audio + AudioSegment.silent(duration=silence_duration)
                        current_position += silence_duration
                
                if group_idx == 0 and segment_idx == 0:
                    if original_start != 0:
                        result_audio = result_audio + AudioSegment.silent(duration=original_start) + audio
                    else:
                        result_audio = result_audio + audio
                else:
                    if original_duration > actual_duration:
                        result_audio = result_audio + audio + AudioSegment.silent(duration=original_duration - actual_duration)
                    else:
                        result_audio = result_audio + audio

                # 배율 제외 테스트 후, 문제 생길시 추후 수정 예정
                silence_duration = 0
                current_position += actual_duration
                previous_end = current_position

            else:
                print(f"Audio 파일 - {audio_path} 없음")

        convert_output_path = os.path.join(output_prefix, f"res_convert_{group_idx}.mp3")
        result_audio.export(convert_output_path, format="mp3")

# Merge split video files
def merge_mp4_files(input_dir, output_file):
    """
        input_dir: split된 mp4 파일들 경로; output_prefix/latentsync
        output_dir: 합친 파일; output_prefix/latentsync/final.mp4
    """
    files = [x for x in os.listdir(input_dir) if x.endswith(".mp4")]

    # def extract_number(filename):
    #     """
    #         파일명에서 숫자 추출; convert_{filename}_{number}_{~~~~}.mp4 -> number
    #     """
    #     match = re.search(r"_(\d+)_", filename)
    #     if match:
    #         return int(match.group(1))
    #     return 0
    
    # files.sort(key=extract_number)

    clips = []
    for file in files:
        file_path = os.path.join(input_dir, file)
        clip = VideoFileClip(file_path)
        clips.append(clip)

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_file, codec="libx264")

    final_clip.close()
    for clip in clips:
        clip.close()

def dubbing_process(video_file, target_language="en-us", use_gpu_for_demucs=False, demucs_endpoint_id=None, vast_instance_url="", vast_instance_port=""):
    """
    비디오 더빙 프로세스를 실행합니다.
    
    Args:
        video_file: 원본 비디오 파일 경로
        target_language: 번역할 언어 코드 ("en-us", "ko", "ja", "cmn", "vi")
        use_gpu_for_demucs: GPU로 Demucs 실행 여부
        demucs_endpoint_id: Demucs용 별도 RunPod 엔드포인트 ID
    
    Returns:
        최종 비디오 파일 경로 및 처리 로그
    """
    logs = []
    logger.info("더빙 프로세스 시작")
    
    # 원본 파일명 설정
    origin_video = os.path.basename(video_file)
    
    # 작업 디렉토리 설정
    work_dir = os.path.dirname(os.path.abspath(video_file))
    os.chdir(work_dir)
    
    # 영상 자체를 25frame으로 강제 변환
    logger.info("25fps 변환 시작")
    fps25_cmd = [
        "ffmpeg",
        "-i", origin_video,
        "-filter:v", "fps=25",
        "-map", "0", "-c:a", "copy",
        f"{origin_video[:-4]}_25fps.mp4"
    ]
    subprocess.run(fps25_cmd)
    
    origin_video = origin_video[:-4] + "_25fps.mp4"
    output_prefix = origin_video.split(".")[0] + "_output"
    
    # STT하기 위한 오디오 파일 생성
    logger.info("오디오 추출 시작")
    video_ = VideoFileClip(origin_video)
    audio_filename = origin_video.split(".")[0] + ".mp3"
    video_.audio.write_audiofile(audio_filename)
    audio_base64 = file_to_base64(audio_filename)
    
    # Faster whisper (STT)
    logger.info("STT 시작")
    stt_result = stt_faster_whisper(audio_base64, RUNPOD_API_KEY, STT_ENDPOINT)
    logger.info("STT 완료")
    
    # Segment split Video
    logger.info("비디오 분할 시작")
    segments_list = segment_split_video(origin_video, video_file, output_prefix, stt_result["segments"], 20)
    logger.info("비디오 분할 완료")
    
    # Split video to audio
    logger.info("분할된 비디오에서 오디오 추출 시작")
    split_video_to_audio(output_prefix, origin_video)
    logger.info("분할된 비디오에서 오디오 추출 완료")
    
    # OpenAI translation
    logger.info(f"{language[target_language]}로 번역 시작")
    translation_result = translate(language[target_language], stt_result["segments"], OPENAI_API_KEY)
    print(f"translation_result : {translation_result}")
    logger.info("번역 완료")
    
    # Split text 
    logger.info("텍스트 분할 시작")
    divide_text = extract_text_by_segments(translation_result, segments_list)
    logger.info("텍스트 분할 완료")
    
    # Zonos (TTS)
    logger.info("TTS 시작")
    reference_audio = origin_video.split(".")[0] + ".mp3"
    cmd = [
        'ffmpeg', 
        '-i', origin_video[:-4] + '.mp3',
        '-af', "loudnorm=I=-14:LRA=7:TP=-2",
        origin_video[:-4] + "_loud.mp3"
    ]
    subprocess.run(cmd)
    
    for i in range(len(divide_text)):
        for j in range(len(divide_text[i])):
            logger.info(f"TTS 처리 중: 세그먼트 {i}, 문장 {j}")
            result = tts_zonos(
                model_type="transformer",
                endpoint_id=TTS_ENDPOINT,
                api_key=RUNPOD_API_KEY,
                language=target_language,
                text=divide_text[i][j]['text'],
                reference_audio_path=reference_audio,
                speaking_rate=15.0,
                pitch_std=60.0,
                fmax=24000.0,
                emotion_happiness=0.0256,
                emotion_sadness=0.0256,
                emotion_disgust=0.0256,
                emotion_fear=0.0256,
                emotion_suprise=0.0256,
                emotion_anger=0.0256,
                emotion_other=0.2564,
                emotion_neutral=1.0,
                vqscore_8=[0.78]*8,
                ctc_loss=1.0,
                dnsmos_ovrl=4.0,
                speaker_noised=False
            )
            
            if result and "audio" in result:
                base64_to_file(result["audio"], f"{output_prefix}/convert_{origin_video.split('.')[0]}_{i}_{j}.mp3")
            else:
                logger.info(f"TTS 저장 실패: 세그먼트 {i}, 문장 {j}")
    
    logger.info("TTS 완료")
    
    # 음성 파일 볼륨 조절
    logger.info("음성 파일 볼륨 정규화 시작")
    for i in glob.glob(f"./{output_prefix}/convert*.mp3"):
        cmd = [
            "ffmpeg",
            "-i", i,
            "-af", "loudnorm=I=-14:LRA=7:TP=-2",
            i[:-4] + "_loud" + ".mp3"
        ]
        subprocess.run(cmd)
        i = os.path.normpath(i)
        os.remove(i)
    logger.info("음성 파일 볼륨 정규화 완료")
    
    # TTS Combined with group
    logger.info("음성 파일 병합 시작")
    group_combined_segment(divide_text, output_prefix, origin_video)
    logger.info("음성 파일 병합 완료")
    
    # Background Sound Split
    logger.info("배경 사운드 분리 시작")
    os.makedirs(f"{output_prefix}/bg_temp", exist_ok=True)
    bg_dir = os.path.join(output_prefix, "bg_temp")
    
    if use_gpu_for_demucs:
        # GPU 모드 - 로컬에서 처리
        bg_model = get_model("htdemucs")
        bg_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for audio_idx, audio_file in enumerate(glob.glob(os.path.join(output_prefix, f"{origin_video[:-4]}_*.mp3"))):
            logger.info(f"로컬 GPU로 오디오 파일 {audio_idx} 처리 중...")
            split_video_to_bg(output_prefix, audio_file, bg_model, bg_device, bg_dir, audio_idx)
    else:
        # 별도의 Demucs RunPod 엔드포인트가 제공된 경우
        if demucs_endpoint_id:
            for audio_idx, audio_file in enumerate(glob.glob(os.path.join(output_prefix, f"{origin_video[:-4]}_*.mp3"))):
                logger.info(f"별도 RunPod Serverless GPU로 오디오 파일 {audio_idx} 처리 중...")
                demucs_runpod_serverless(audio_file, RUNPOD_API_KEY, demucs_endpoint_id, output_prefix, audio_idx)
        else:
            # 별도 엔드포인트가 없는 경우 로컬 CPU로 처리
            logger.info("별도 Demucs 엔드포인트가 없어 로컬 CPU로 처리합니다 (느릴 수 있음)")
            bg_model = get_model("htdemucs")
            bg_device = 'cpu'
            for audio_idx, audio_file in enumerate(glob.glob(os.path.join(output_prefix, f"{origin_video[:-4]}_*.mp3"))):
                logger.info(f"로컬 CPU로 오디오 파일 {audio_idx} 처리 중...")
                split_video_to_bg(audio_file, bg_model, bg_device, bg_dir, audio_idx)
    
    logger.info("배경 사운드 분리 완료")
    
    # 음성과 배경 사운드 혼합
    logger.info("음성과 배경 사운드 혼합 시작")
    voice_ = glob.glob(os.path.join(output_prefix, "res_convert_*.mp3"))
    bg_ = glob.glob(os.path.join(output_prefix, "bg*_loud.mp3"))
    for i in range(len(voice_)):
        temp_voice = AudioSegment.from_file(voice_[i])
        temp_bg = AudioSegment.from_file(bg_[i])
        combined = temp_voice.overlay(temp_bg)
        combined.export(f"{output_prefix}/combined_{i}.mp3", format="mp3")
    logger.info("음성과 배경 사운드 혼합 완료")
    
    # LatentSync
    logger.info("립싱크 시작")
    VAST_URL = vast_instance_url
    VAST_PORT = vast_instance_port
    latentsync(output_prefix, VAST_URL, VAST_PORT, "pingpong")
    logger.info("립싱크 완료")
    
    # Merge Convert Video
    logger.info("최종 비디오 병합 시작")
    INPUT_DIR = output_prefix + "/latentsync"
    OUTPUT_DIR = INPUT_DIR + "/final.mp4"
    merge_mp4_files(INPUT_DIR, OUTPUT_DIR)
    logger.info("최종 비디오 병합 완료")

    # libx265 인코딩
    logger.info("libx265 인코딩 시작")
    libx265_cmd = [
        "ffmpeg",
        "-i", OUTPUT_DIR,
        "-c:v", "libx265",
        "-crf", "35",
        "preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
        OUTPUT_DIR
    ]
    subprocess.run(libx265_cmd)
    logger.info("libx265 인코딩 완료")

    # 최종 비디오 파일이 존재하는지 확인
    if not os.path.exists(OUTPUT_DIR) or os.path.getsize(OUTPUT_DIR) == 0:
        logger.error("최종 비디오 파일이 생성되지 않았거나 크기가 0입니다")
        return {
            "error": "최종 비디오 처리에 실패했습니다",
            "logs": logs
        }
    
    # 최종 비디오 파일을 base64로 변환
    try:
        final_video_base64 = file_to_base64(OUTPUT_DIR)
        if not final_video_base64:
            logger.error("비디오 파일을 base64로 변환하는데 실패했습니다")
            return {
                "error": "비디오 인코딩에 실패했습니다",
                "logs": logs
            }
    except Exception as e:
        logger.error(f"비디오 변환 중 오류: {str(e)}")
        return {
            "error": f"비디오 변환 오류: {str(e)}",
            "logs": logs
        }
    
    print(f"os.path.exists(os.path.join(output_prefix, 'latentsync', 'final.mp4')) : {os.path.exists(os.path.join(output_prefix, 'latentsync', 'final.mp4'))}")
    print(f"os.path.join(output_prefix, 'latentsync', 'final.mp4') : {os.path.join(output_prefix, 'latentsync', 'final.mp4')}")
    print(f"os.path.getsize(os.path.join(output_prefix, 'latentsync', 'final.mp4')) : {os.path.getsize(os.path.join(output_prefix, 'latentsync', 'final.mp4'))}")
    print(f"len(final_video_base64) : {len(final_video_base64)}")
    test_ = AudioSegment.from_file(os.path.join(output_prefix, 'latentsync', 'final.mp4'))
    print(f"test_.duration_seconds : {test_.duration_seconds}")

    return {
        "video": final_video_base64,
        "logs": logs
    }

def handler(event):
    """
    RunPod Serverless 핸들러 함수
    
    Args:
        event: RunPod 이벤트 객체
        
    Returns:
        처리 결과 객체
    """


    logger.info("RunPod Serverless 핸들러 함수 시작")
    try:
        job_input = event["input"]
        
        # 임시 작업 디렉토리 생성
        temp_dir = "/tmp/dubbing_job"
        #temp_dir = os.path.join(os.getcwd(), "dubbing_job")
        os.makedirs(temp_dir, exist_ok=True)
        os.chdir(temp_dir)
        
        # 입력 비디오 저장
        if "video" not in job_input:
            return {"error": "입력 비디오가 없습니다."}
        
        video_base64 = job_input["video"]
        video_file = os.path.join(temp_dir, "input_video.mp4")
        base64_to_file(video_base64, video_file)
        
        # 타겟 언어 설정
        target_language = job_input.get("target_language", "en-us")
        if target_language not in language:
            return {"error": f"지원하지 않는 언어입니다. 지원 언어: {list(language.keys())}"}
        
        # Demucs 설정
        use_gpu_for_demucs = job_input.get("use_gpu_for_demucs", True)
        demucs_endpoint_id = job_input.get("demucs_endpoint_id", None)
        
        # Lipsync 사용 여부 (현재는 사용하지 않지만 나중을 위해 파라미터 저장)
        use_lipsync = job_input.get("use_lipsync", True)
        
        # BG 처리 여부 (현재는 사용하지 않지만 나중을 위해 파라미터 저장)
        use_bg_separation = job_input.get("use_bg_separation", True)

        # Vast AI Instance id, Port 추가
        vast_instance_url = job_input.get("vast_instance_url", "")
        vast_instance_port = job_input.get("vast_instance_port", "")
        
        # 더빙 프로세스 실행
        result = dubbing_process(
            video_file=video_file,
            target_language=target_language,
            use_gpu_for_demucs=use_gpu_for_demucs,
            demucs_endpoint_id=demucs_endpoint_id,
            vast_instance_url=vast_instance_url,
            vast_instance_port=vast_instance_port
        )
        
        # 입력 파라미터 정보 추가
        result["params"] = {
            "target_language": target_language,
            "use_lipsync": use_lipsync,
            "use_bg_separation": use_bg_separation
        }
        
        return result
        
    except Exception as e:
        import traceback
        error_message = str(e)
        stack_trace = traceback.format_exc()
        return {
            "error": error_message,
            "stack_trace": stack_trace
        }

# RunPod 서버리스 API에 핸들러 등록
runpod.serverless.start({"handler": handler})
