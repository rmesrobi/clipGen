import os
import json
import time
from openai import OpenAI
from pydub import AudioSegment
from IPython.display import display
import tiktoken 
gpt4_enc = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")
import cv2
import whisper

def write_to_json(response, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(response, outfile, ensure_ascii=True, indent=4)

def load_json(path):
    with open(path, 'r') as openfile:
        data = json.load(openfile)
    return data

def get_tokens(enc, text):
    return list(map(lambda x: enc.decode_single_token_bytes(x).decode('utf-8'), 
                  enc.encode(text)))

def remove_words_from_text(text, remove_words):
    filtered_text = text.lower()
    for rw in remove_words:
        filtered_text = filtered_text.replace(rw, '')
    return filtered_text

def filter_sentences(sentences):
    filtered_sentences = []
    for sen in sentences:
        if len(sen) <= 2:
            pass
        elif len(filtered_sentences) > 0 and sen == filtered_sentences[-1]:
            pass
        else:
            filtered_sentences.append(sen)
    return filtered_sentences

def update_prompt_template(prompt_template: str, **kwargs):
    prompt = prompt_template.format(**kwargs)
    return prompt

def create_chat_completions(client, model, messages, temperature):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"}
    )

    response = completion.choices[0].message.content
    tokens_count = {
    'prompt_tokens':completion.usage.prompt_tokens,
    'completion_tokens':completion.usage.completion_tokens,
    'total_tokens':completion.usage.total_tokens,
    }

    try:
        json_obj = json.loads(response)
        return json_obj, tokens_count

    except:
        return response, tokens_count

def parse_response_for_label(response, keywords):
    response_key = [k for k in response.keys() if 'label' in k.lower() in keywords][0]
    return response[response_key]

def label_the_topics(client, gpt_model, topics_dict, topic_message, system_message, filepath):
    for key, value in topics_dict.items():
        docs = value['documents']
        keywords = ', '.join(value['representation'])
        user_prompt = update_prompt_template(topic_message, documents=docs, keywords=keywords)
        messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {

                    "role": "user",
                    "content": user_prompt
                }
                ]  
        response, tokens = create_chat_completions(client, gpt_model, messages, 0.2)
        topics_dict[key]['label'] = parse_response_for_label(response, ['label', 'topic'])
        topics_dict[key]['tokens'] = tokens

    write_to_json(topics_dict, filepath)
    return topics_dict

def get_transcriptions(whisper_model, audio_path, prompt):
    audio = whisper.load_audio(audio_path)
    print('Opened the audio file')
    whisper_model = whisper.load_model(whisper_model, in_memory=True)
    print('Loaded the whisper model')
    response = whisper.transcribe(model=whisper_model, audio=audio, prompt=prompt, verbose=False)
    return response

def get_all_transcriptions(audio_paths, configs, prompt):
    WHISPER_MODEL = configs['openai']['whisper_model']
    responses = []
    for ap in audio_paths:
        response = get_transcriptions(WHISPER_MODEL, ap, prompt)
        responses.append(response)
    
    response_dict = {'transcriptions': responses}
    return response_dict