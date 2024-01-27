import os
import json
import numpy as np
import math
import utils.helper_functions as help_func
import cv2
import time
from moviepy.editor import VideoFileClip, AudioFileClip

class VideoObject:
    def __init__(self, configs):
        self.configs = configs
        self.video_configs = configs['video']
        self.source_audio_path = os.path.join(configs['general']['audio_path'], self.configs['audio']['source_audio_filename'])
        self.video_path = os.path.join(self.configs['general']['video_path'], self.video_configs['video_filename'])
        self.max_value = int(self.video_configs['max_value'])
        self.bar_color = self.video_configs['bar_color']
        self.posx = int(self.video_configs['posx'])
        self.posy = int(self.video_configs['posy'])
        self.font = self.video_configs['font']
        self.fourcc = self.video_configs['fourcc']
        self.display_topics = bool(self.video_configs['display_topics'])
        self.video_codec = self.video_configs['codec']
        self.audio_codec = self.configs['audio']['audio_codec']
        self.audio_fps = int(self.configs['audio']['fps'])
        self.metadata = self.get_video_metadata()

    def split_video_into_chunks(self) -> list:
        filename = self.video_configs['video_filename']
        chunk_length = int(self.video_configs['chunk_video_length'])
        video = VideoFileClip(self.video_path)
        total_length = int(video.duration)
        chunk_length_seconds = chunk_length * 60
        chunks = math.ceil(total_length / chunk_length_seconds)
        print(f'Splitting video into {chunks} chunks.')

        self.chunk_filenames = []
        for i in range(chunks):
            start_time = i * chunk_length_seconds
            end_time = min((i + 1) * chunk_length_seconds, total_length)
            chunk = video.subclip(start_time, end_time)
            # Remove mp4 from filename
            filename = filename.replace('.mp4','')
            chunk_filename = os.path.join(self.configs['general']['video_path'], f"{filename}_chunk_{i+1}.mp4")
            chunk.write_videofile(chunk_filename, codec=self.video_codec, audio_codec=self.audio_codec)
            self.chunk_filenames.append(chunk_filename)
        return self.chunk_filenames
    
    def create_audio_file(self, video_path, audio_path):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, fps = self.audio_fps)

    def create_audio_files(self) -> list:
        self.audio_filenames = []
        for file in self.chunk_filenames:
            audio_chunk_filename = file.replace('.mp4', '.mp3').replace(self.configs['general']['video_path'],'')
            if audio_chunk_filename[0] == '/':
                audio_chunk_filename = audio_chunk_filename[1:]
            audio_path = os.path.join(self.configs['general']['audio_path'], audio_chunk_filename)
            self.audio_filenames.append(audio_path)
            self.create_audio_file(file, audio_path)
        return self.audio_filenames
    
    def load_and_convert(self):
        video = VideoFileClip(self.video_path)
        video.audio.write_audiofile(self.source_audio_path, fps = self.audio_fps) #44_100 for best quality

    def get_video_metadata(self) -> dict:
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_size = (frame_width, frame_height)
        cap.release()
        return {'fps': fps,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'frame_size': frame_size}
    
    def create_frame_data(self, package_data) -> dict:
        segment_topic = package_data['segment_topic']
        start = package_data['start']
        end = package_data['end']
        clip_frames = int((end - start) * self.metadata['fps'])
        clip_change_dict = {k: [] for k in segment_topic.keys()}
        last_value = None
        for k, val in segment_topic.items():
            for v in val:
                if last_value is None:
                    # Apply a fixed value
                    last_value = max(v - 0.1, 0)
                change = np.linspace(last_value, v, int(clip_frames / len(val)))
                clip_change_dict[k].extend(list(change))
                last_value = v
        return clip_change_dict

    def create_video_topics(self, video_path, clip_change_dict) -> str: 
        topics = [k for k in clip_change_dict.keys()]
        total_frames = max([len(v) for k, v in clip_change_dict.items()])
        cap = cv2.VideoCapture(video_path)
        output_video_filename = video_path.replace('_temp.mp4', '.mp4')
        output_video = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc(*'MP4V'), self.metadata['fps'], self.metadata['frame_size'])
        frame_number = 0
        while cap.isOpened():
            while frame_number < total_frames:
                ret, frame = cap.read()
                if ret:
                    posx = self.posx
                    posy = self.posy
                    frame = cv2.putText(frame, "Discussion Topics:", (posx, posy), cv2.FONT_HERSHEY_DUPLEX, .30, (0, 255, 0), 0, cv2.LINE_AA)
                    posy += 12
                    for tp in topics:
                        try:
                            display_val = max(clip_change_dict[tp][frame_number], 0)
                        except:
                            pass
                        frame = cv2.putText(frame, tp, (posx, posy), cv2.FONT_HERSHEY_DUPLEX, .30, (0, 255, 0), 0, cv2.LINE_AA)
                        posy += 6
                        bar_width = int(self.max_value * display_val)
                        frame = cv2.rectangle(frame, (posx, posy), (bar_width, posy), (0, 255, 0), 4)
                        posy += 12

                    output_video.write(frame)
                    frame_number += 1
            output_video.release()
            cap.release()
        return output_video_filename

    def create_video(self, video_name, package_data):
        clip = VideoFileClip(self.video_path)
        clip_chunk = clip.subclip(package_data['start'], package_data['end'])
        # Save file
        filename = os.path.join(self.configs['general']['outputs'], video_name)
        filename = filename.replace('.mp4', '_temp.mp4')
        clip_chunk.write_videofile(filename, codec=self.video_codec)
        clip_chunk.audio.write_audiofile('temp.mp3')
        # If display topics
        frame_data = self.create_frame_data(package_data)
        output_video_filename = self.create_video_topics(filename, frame_data)
        output_clip = VideoFileClip(output_video_filename)
        audio_clip = AudioFileClip('temp.mp3')
        output_clip = output_clip.set_audio(audio_clip)

        if os.path.exists(output_video_filename):
            os.remove(output_video_filename)

        output_clip.write_videofile(output_video_filename, codec=self.video_codec, audio_codec = self.audio_codec)

        if os.path.exists(filename):
            # Delete the file
            os.remove(filename)
        
        if os.path.exists('temp.mp3'):
            # Delete the file
            os.remove('temp.mp3')

    def create_all_videos(self, package):
        labels = []
        for pack_obj in package:
            label = pack_obj['label'].strip().lower().replace(' ', '_')
            labels.append(label)
            video_name = label + '_' + str(labels.count(label)) + '.mp4'
            self.create_video(video_name, pack_obj)



            

