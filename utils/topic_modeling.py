import os
import json
import numpy as np
import utils.helper_functions as help_func
# Bert
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
# transformer_model = SentenceTransformer('all-mpnet-base-v2')
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

class TopicModel:
    def __init__(self, configs, segments):
        self.configs = configs
        self.display_topics = bool(self.configs['video']['display_topics'])
        self.topic_configs = configs['topic-model']
        self.number_topics = self.topic_configs['number_topics']
        self.min_clip_time = int(self.topic_configs['min_clip_time'])
        self.max_clip_time = int(self.topic_configs['max_clip_time'])
        self.max_cluster_size = int(self.topic_configs['max_cluster_size'])
        self.topics_filepath = os.path.join(self.configs['general']['json_path'], self.topic_configs['topics_path'])
        self.segments = segments
        self.vectorizer_model = CountVectorizer(stop_words = 'english')
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.umap_model = UMAP(n_components=15, min_dist=0.0, metric='cosine', angular_rp_forest=True, low_memory=False, random_state=42)
        self.hdbscan_model = HDBSCAN(min_cluster_size=15, max_cluster_size=self.max_cluster_size, prediction_data=True)
        self.representation_model = KeyBERTInspired()
        
    def create_topic_model(self):
        topic_model = BERTopic(nr_topics=self.number_topics,
                       min_topic_size=10,
                       n_gram_range=(1,3),
                       ctfidf_model= self.ctfidf_model,
                       umap_model=self.umap_model,
                       hdbscan_model=self.hdbscan_model,
                       vectorizer_model= self.vectorizer_model,
                       representation_model= self.representation_model,
                      )
        return topic_model

    def create_topic_dict(self) -> dict:
        topics_dict = {}
        for idx, row in self.topic_info.iterrows():
            topic_number = idx + 1
            topics_dict[topic_number] = {'count': row['Count'],
                                        'name': row['Name'],
                                        'representation': row['Representation'],
                                        'documents': row['Representative_Docs'],
                                        'label': ''}
            
        return topics_dict
    
    def get_topics(self, docs: list) -> dict:
        print('Creating topic model')
        self.topic_model  = self.create_topic_model()
        print('Fit and transform')
        topics, ini_probs = self.topic_model.fit_transform(docs)
        print('Getting topic info')
        self.topic_info = self.topic_model.get_topic_info()
        print('Packaging topics into dict')
        self.topics_dict = self.create_topic_dict()
        return self.topics_dict
    
    def get_topics_docs_and_labels(self, topics_dict_labels: dict) -> list:
        self.topics_dict = topics_dict_labels
        self.topics_docs = []
        self.topics_labels = []
        for k, v in topics_dict_labels.items():
            self.topics_docs.extend(v['documents'])
            self.topics_labels.append(v['label'])
        return self.topics_labels, self.topics_docs
    
    def find_related_sentences(self) -> dict:
        docs = [sen['text'] for sen in self.segments]
        sentence_embeddings = transformer_model.encode(docs)
        topic_embeddings = transformer_model.encode(self.topics_docs)

        self.related_sentences = {}
        for i, topic in enumerate(self.topics_docs):
            similarity_scores = []
            for sdx, sentence_embedding in enumerate(sentence_embeddings):
                similarity = cosine_similarity([topic_embeddings[i]], [sentence_embedding])[0][0]
                similarity_scores.append((sdx, float(similarity)))

            self.related_sentences[topic] = similarity_scores
        return self.related_sentences
    
    def highest_score_in_time_range(self, scores, time_tuples, indexes):
        if not scores or not time_tuples or len(scores) != len(time_tuples) or len(scores) != len(indexes):
            return 0, []
        
        max_score = 0
        max_score_indexes = []
        n = len(scores)
        for i in range(n):
            current_score = 0
            current_indexes = []
            for j in range(i, n):
                start_time, end_time = time_tuples[i][0], time_tuples[j][1]
                current_score += scores[j]
                current_indexes.append(indexes[j])
                time_diff = end_time - start_time
                if self.min_clip_time <= time_diff <= self.max_clip_time and current_score / max(1, len(current_indexes)) > max_score / max(1, len(max_score_indexes)):
                    max_score = current_score
                    max_score_indexes = list(current_indexes)
                elif time_diff > self.max_clip_time:
                    break
        return max_score, max_score_indexes
    
    def get_topics_segments(self):

        for topic_key, topic_data in self.topics_dict.items():
            self.topics_dict[topic_key]['max_scores'] = []
            self.topics_dict[topic_key]['max_scores_idx'] = []
            topics_documents = topic_data['documents']

            for td in topics_documents:
                rss = self.related_sentences[td]
                idxs = [x[0] for x in rss]
                scores = [round(x[1],2) for x in rss]
                time_tuples = [(self.segments[i]['start'], self.segments[i]['end']) for i in idxs]

                max_score, max_score_indexes = self.highest_score_in_time_range(scores, time_tuples, idxs)
                self.topics_dict[topic_key]['max_scores'].append(max_score)
                self.topics_dict[topic_key]['max_scores_idx'].append(max_score_indexes)

        return self.topics_dict
    
    def get_segment_topics(self):
        for idx, seg in enumerate(self.segments):
            self.segments[idx]['topics'] = {l: [] for l in self.topics_labels}

        for topic_key, topic_data in self.topics_dict.items():
            topics_documents = topic_data['documents']
            label = topic_data['label']
            for td in topics_documents:
                rss = self.related_sentences[td]
                for seg_index, seg_score in rss:
                    self.segments[seg_index]['topics'][label].append(seg_score)
    
    def package_segments(self):
        self.packaged_segments = []
        for topic_key, topic_data in self.topics_dict.items():
            documents = topic_data['documents']
            for ddx, doc in enumerate(documents):
                packaged_segment = {'label': topic_data['label'],
                                    'doc': doc,
                                    'score': topic_data['max_scores'][ddx],
                                    'segment_loc': topic_data['max_scores_idx'][ddx],
                                    }
                start = None
                end = None
                texts = []
                for i in packaged_segment['segment_loc']:
                    seg_data = self.segments[i]
                    if start is None or seg_data['start'] < start:
                        start = seg_data['start']
                    if end is None or seg_data['end'] > end:
                        end = seg_data['end']
                    texts.append(seg_data['text'])

                # Complete the sentence
                texts_str_temp = ' '.join(texts)
                add_count = 1
                while texts_str_temp[-1] != '.':
                    try:
                        add_seg_data = self.segments[max(idxs) + add_count]
                        end = add_seg_data['end']
                        texts.append(add_seg_data['text'])
                    except:
                        break

                packaged_segment['text'] = ' '.join(texts)
                packaged_segment['start'] = start
                packaged_segment['end'] = end
                
                self.packaged_segments.append(packaged_segment)
        
        if self.display_topics:
            self.update_package_with_seg_data()

        return {'package': self.packaged_segments}
                
    def update_package_with_seg_data(self):
        for idx, pack in enumerate(self.packaged_segments):
            segment_loc = pack['segment_loc']
            self.packaged_segments[idx]['segment_topic'] = {}
            for loc in segment_loc:
                seg_data = self.segments[loc]
                seg_topics = seg_data['topics']
                seg_output = {k: np.mean(v) for k, v in seg_topics.items()}
                for k, v in seg_output.items():
                    if k not in self.packaged_segments[idx]['segment_topic']:
                        self.packaged_segments[idx]['segment_topic'][k] = []
                    self.packaged_segments[idx]['segment_topic'][k].append(v)