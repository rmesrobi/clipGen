import os
import json
import time
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import utils.helper_functions as help_func

class Transcript:
    def __init__(self, configs: dict, transcript_object: json):
        self.configs = configs
        self.remove_words_list = self.configs['transcript']['remove_words']
        self.processed_transcript_filepath = os.path.join(self.configs['general']['json_path'], self.configs['transcript']['transcript_path'])
        self.transcript_obj = self.preprocess_transcript_object(transcript_object)
        self.text = self.transcript_obj['text']
        self.segments = self.transcript_obj['segments']
        self.tokenize = bool(self.configs['transcript']['sentence_tokenizer'])
        self.tokenized_sentences = self.create_documents()
        self.mapped_segments_filepath = os.path.join(self.configs['general']['json_path'], self.configs['transcript']['mapped_segments_path'])

    def preprocess_transcript_object(self, transcript_object) -> dict:
        # Transcript
        text = transcript_object['text']
        processed_text = self.process_text(text)
        transcript_object['text'] = processed_text

        # Segments
        segments = transcript_object['segments']
        for idx, seg in enumerate(segments):
            seg_text = seg['text']
            processed_text = self.process_text(seg_text)
            segments[idx]['text'] = processed_text

        help_func.write_to_json(transcript_object, self.processed_transcript_filepath)
        return transcript_object
    
    def process_text(self, text) -> str:
        text = self.remove_words(text)
        text = self.remove_extra_spaces(text)
        # text = self.remove_punctuation_tokens(text)
        text = self.remove_consecutive_characters(text)
        return text

    def remove_extra_spaces(self, text_object) -> str:
        # Remove extra spaces
        text = text_object.replace('  ', ' ').strip()
        return text
    
    def remove_words(self, text_object) -> str:
        remove_words = self.remove_words_list.split(', ')
        pattern = r'\b(' + '|'.join(map(re.escape, remove_words)) + r')\b'
        filtered_text = re.sub(pattern, '', text_object, flags=re.IGNORECASE)
        return filtered_text

    def create_documents(self) -> list:
        if self.tokenize:
            sentence_tokens = sent_tokenize(self.text)
            processed_tokens = self.process_tokens(sentence_tokens)
            return processed_tokens
        else:
            return [seg['text'] for seg in self.segments]

    def process_tokens(self, sentence_tokens) -> list:
        # Remove tokens that are just '.'
        sentence_tokens = self.remove_punctuation_tokens(sentence_tokens)
        # Remove consecutive chars like , , 
        sentence_tokens = list(map(self.remove_consecutive_characters, sentence_tokens))
        # sentence_tokens = [sen for sen in sentence_tokens if len(sen) > 3]
        return sentence_tokens
    
    def remove_punctuation_tokens(self, sentence_tokens) -> list:
        return [s for s in sentence_tokens if s not in list(char for char in string.punctuation)]
    
    def remove_consecutive_characters(self, text):
        # Replace two or more of any punctuation or space with a single instance of that punctuation or space
        for punct in string.punctuation + ' ':
            regex_pattern = r'{0}+'.format(re.escape(punct))
            # Use a lambda function for replacement to avoid issues with special characters
            text = re.sub(regex_pattern, lambda match: match.group(0)[0], text)
        return text

    def map_new_segments(self, docs: list) -> dict:
        self.mapped_segments = {}
        for idx, doc in enumerate(docs):
            combined_segment = self.find_segment_start_end(doc)
            if combined_segment:
                combined_segment['doc'] = doc
                self.mapped_segments[idx] = combined_segment
            else:
                self.mapped_segments[idx] = None
        
        help_func.write_to_json(self.mapped_segments, self.mapped_segments_filepath)
        return self.mapped_segments

    def find_max_seg_match(self, doc) -> int:
        return_lst = []
        return_idxs = []
        for idx, seg in enumerate(self.segments):
            if (len(doc) >= len(seg) )and (seg['text'].strip().lower() in doc.strip().lower() or seg['text'].strip().lower() == doc.strip().lower()):
                return_lst.append(seg['text'])
                return_idxs.append(idx)
            elif doc.strip().lower() in seg['text'].strip().lower():
                return_lst.append(seg['text'])
                return_idxs.append(idx)

        if len(return_lst) > 0:
            return return_idxs[return_lst.index(max(return_lst))]
        else:
            return None

    def combine_segments(self, start_idx, end_idx) -> dict:
        return_segment = {'segments_text': [],
                            'start': None,
                            'end': None}
        starts = []
        ends = []
        for i in range(start_idx, end_idx + 1):
            try:
                data = self.segments[i]
                return_segment['segments_text'].append(data['text'])
                starts.append(data['start'])
                ends.append(data['end'])
            except:
                pass
            
        return_segment['start'] = min(starts)
        return_segment['end'] = max(ends)

        return return_segment
         
    def find_segment_start_end(self, doc: str) -> dict:
        search_text = doc.strip().lower()
        max_result = self.find_max_seg_match(search_text)
        if max_result is None:
            return None
        else:
            data = self.segments[max_result]
            data_text = data['text'].strip().lower()
            if len(data_text) <= len(search_text):
                result = re.search(data_text, search_text).span()
                match_first, match_last = result[0], result[-1]
                combined_text = data_text

                iterr = 1
                search_keys = []
                while not search_text in combined_text.strip().lower():
                    start_idx = None
                    end_idx = None
                    try:
                        if match_first > 0:
                            prior = self.segments[max_result - iterr]
                            prior_text = prior['text']
                            search_keys.append(max_result - iterr)
                            combined_text = prior_text + ' ' + combined_text
                    except:
                        pass
                    
                    
                    combined_text = combined_text.strip().lower().replace('  ', ' ')
                    start_idx = max_result - iterr
                    end_idx = max_result + iterr

                    try:
                        if match_last < len(doc):
                            # segment_end_idx = max_result + iterr
                            future = self.segments[max_result + iterr]
                            future_text = future['text']
                            search_keys.append(max_result + iterr)
                            combined_text += ' ' + future_text
                    except:
                        pass
                    
                    
                    combined_text = combined_text.strip().lower().replace('  ', ' ')
                    start_idx = max_result - iterr
                    end_idx = max_result + iterr

                    iterr += 1
                    # Failsafe
                    if iterr > 10:
                        break
        
                    if start_idx and end_idx:
                        return self.combine_segments(start_idx, end_idx)
                    else:
                        return {'segments_text': data['text'],
                                            'start': data['start'],
                                            'end': data['end']}
            else:
                return None