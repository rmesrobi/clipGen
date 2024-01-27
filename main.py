import os
import json
import string
import re
import logging
logging.getLogger('moviepy').setLevel(logging.ERROR)
from openai import OpenAI
import utils.set_configs as config
import prompts as prompt
import utils.helper_functions as help_func
import utils.transcript_object as data_packaging
import utils.topic_modeling as tm
import utils.video_processing as vid
import tqdm
import warnings
warnings.filterwarnings('ignore')

configs = config.parse_config()
openai_configs = configs['openai']
JSON_DIR = configs['general']['json_path']
AUDIO_DIR = configs['general']['audio_path']
TRANSCRIPTION_PATH = os.path.join(JSON_DIR, configs['openai']['transcription_filename'])
WHISPER_MODEL = openai_configs['whisper_model']
AUDIO_PATH = os.path.join(AUDIO_DIR, configs['audio']['source_audio_filename']) 
topics_filepath = os.path.join(JSON_DIR, configs['topic-model']['topics_path'])
OPEN_API_KEY = os.getenv('OPENAI_API_KEY')

# Establish the OpenAI API client
if OPEN_API_KEY is not None:
        client = OpenAI(api_key=OPEN_API_KEY)
else:
    print('Please set the OPENAI_API_KEY environment variable')

# Step 1: Get the transcription data
print('Beginning Step 1: Get the transcription data')
# Load the video file
source_video_obj = vid.VideoObject(configs)
source_video_obj.load_and_convert()
response = help_func.get_transcriptions(WHISPER_MODEL,AUDIO_PATH,prompt.WHISPER_PROMPT)
response_obj = {'transcription': response}
help_func.write_to_json(response_obj, TRANSCRIPTION_PATH)

# Step 2: Data Preprocessing
print('Beginning Step 2: Data Preprocessing')
data = help_func.load_json(TRANSCRIPTION_PATH)
# Preprocess text
transcript_obj = data_packaging.Transcript(configs, data['transcription'])
text = transcript_obj.text
segments = transcript_obj.segments
# Package the text into documents
docs = transcript_obj.tokenized_sentences

# Step 3: Get Topics
print('Beginning Step 3: Get Topics')
tm_obj = tm.TopicModel(configs, segments)
topics_dict_temp = tm_obj.get_topics(docs)

# Step 4: Label the Topics
print('Beginning Step 4: Label the Topics')
topics_dict_labels = help_func.label_the_topics(client, openai_configs['gpt_model'], topics_dict_temp, prompt.TOPIC_MESSAGE, prompt.SYSTEM_MESSAGE, topics_filepath)
    
# Step 5: Match segments to docs / labels
print('Beginning Step 5: Match Segments')
# Find the segments that most closely match the docs
topics_labels, topics_docs = tm_obj.get_topics_docs_and_labels(topics_dict_labels)
related_sentences = tm_obj.find_related_sentences()
topics_dict = tm_obj.get_topics_segments()
help_func.write_to_json(topics_dict, os.path.join(JSON_DIR, 'topics.json'))
# Optional - if you want to display topic data in video
tm_obj.get_segment_topics()
help_func.write_to_json(tm_obj.segments, os.path.join(JSON_DIR, 'segment_topics.json'))
package = tm_obj.package_segments()
help_func.write_to_json(package, os.path.join(JSON_DIR, 'package.json'))

# Step 6: Create the video clips
print('Beginning Step 6: Create the clips')
package = help_func.load_json(JSON_DIR + '/package.json')
package_objs = package['package']
vid_obj = vid.VideoObject(configs)
# Iterate over package and save video clips
vid_obj.create_all_videos(package_objs)