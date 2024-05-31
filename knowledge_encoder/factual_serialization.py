import copy
import datetime
import os
import json
import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from spacy.tokens import Span
"""
===================environmental setting=================
# Basic Setup (One time activity)

# 1. Clone the DYGIE++ repository from: https://github.com/dwadden/dygiepp. This repositiory is managed by Wadden et al., authors of the paper Entity, Relation, and Event Extraction with Contextualized Span Representations (https://www.aclweb.org/anthology/D19-1585.pdf).

# git clone https://github.com/dwadden/dygiepp.git

# 2. Navigate to the root of repo in your system and use the following commands to setup the conda environment:

# conda create --name dygiepp python=3.7
# conda activate dygiepp
# cd dygiepp
# pip install -r requirements.txt
# conda develop .   # Adds DyGIE to your PYTHONPATH

# Running Inference on Radiology Reports

# 3. Activate the conda environment:

# conda activate dygiepp

"""


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            print("MyEncoder-datetime.datetime")
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Span):
            return str(obj)
        else:
            return super(MyEncoder, self).default(obj)


class RadGraphNER:
    # change the data architecture
    def __init__(self, corpus=None, ann_path=None, model_path=None, cuda='1', is_get_output=True, is_mimic=False):
        """

        Args:
            corpus: dict: {id1: s1, id2: s2, id3: s3, ...}. if corpus is None, temp_dygie_output.json should be at current path
            model_path: the official checkpoint for radgraph
            cuda: the id for gpu
            is_get_input: Whether to convert to the format processed by radgraph
        """
        self.model_path = model_path
        self.cuda = cuda
        # user defined
        self.input_path = "temp_dygie_input.json"
        self.output_path = 'temp_dygie_output.json'
        if is_get_output:
            # get_dygie_input
            if is_mimic:
                self.get_mimic_temp_dygie_input(ann_path)
            else:
                self.get_corpus_temp_dygie_input(corpus)
            # extract entities and relationships using RadGraph
            self.extract_triplets()

    def get_mimic_temp_dygie_input(self, ann_path):
        # note that only the training corpus can be used.
        ann = json.load(open(ann_path))
        print("initialization the input data")
        del ann['val']
        del ann['test']
        with open(self.input_path, encoding='utf8', mode='w') as f:
            for split, value in ann.items():
                print(f"preprocessing the {split} data...")
                subject_study = []
                for item in tqdm(value):
                    subject, study = str(item['subject_id']), str(item['study_id'])
                    cur_subject_study = subject + '_' + study
                    if cur_subject_study not in subject_study:
                        subject_study.append(cur_subject_study)
                        sen = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                     item['report'])
                        input_item = {
                            'doc_key': cur_subject_study,
                            "sentences": [sen.strip().split()]
                        }
                        f.write(json.dumps(input_item, cls=MyEncoder))
                        f.write('\n')

    def get_corpus_temp_dygie_input(self, corpus):
        # note that only the training corpus can be used.
        with open(self.input_path, encoding='utf8', mode='w') as f:
            for item_id, value in corpus.items():
                input_item = {
                    'doc_key': item_id,
                    "sentences": [value.strip().split()]
                }
                f.write(json.dumps(input_item, cls=MyEncoder))
                f.write('\n')

    def extract_triplets(self):
        print("extract output files using radgraph.")
        os.system(f"allennlp predict {self.model_path} {self.input_path} \
                    --predictor dygie --include-package dygie \
                    --use-dataset-reader \
                    --output-file {self.output_path} \
                    --silent")

    def preprocess_mimic_radgraph_output(self):
        # =====================build the serialized string=============#
        final_dict = {}
        useless_findings = useless_core_findings_new()
        # negative_list = ['no ', 'not ', 'free of', 'negative', 'without', 'clear of']  # delete unremarkable
        print("obtain the triples for each report.")
        with open(self.output_path, 'r') as f:
            for line in tqdm(f):
                data_item = json.loads(line)
                n = data_item['predicted_ner'][0]
                # r = data_item['predicted_relations'][0]
                s = data_item['sentences'][0]
                if len(n) == 0:
                    # print(len(n), " ".join(s))
                    continue

                doc_key = data_item['doc_key']
                # if doc_key in ['10000935_50578979', '10116310_50782200', '17277688_55816958', ]:
                #     print()
                n = preprocessing_entities(n, s, doc_key)
                dict_entity = {'text': ' '.join(s)}

                # ====Remove some useless entities and relationships====
                # initialized the variables
                dot_index = [index for index, token in enumerate(s) if token in ['.', '?', '!']]
                if len(dot_index) != 0:
                    if dot_index[0] != 0:
                        dot_index = [0, *dot_index]
                    if dot_index[-1] != len(s) - 1:
                        dot_index = [*dot_index, len(s)]
                    else:
                        dot_index[-1] += 1
                else:
                    dot_index = [0, len(s)]  # the last index + 1
                core_findings = []
                dot_s_idx, dot_e_idx, pre_sen_idx = -1, -1, -1
                cur_core_findings, previous_node_modified = [], False
                # cur_core_findings: words of each sentence
                # core_findings: finding of each sentence
                for idx, ent_item in enumerate(n):
                    start_idx, end_idx, ent_label = ent_item[0], ent_item[1], ent_item[2].strip()
                    cur_ent = " ".join(s[start_idx:end_idx + 1]).strip('"').strip("'").strip()
                    # delete unrelated entities
                    if cur_ent in list(',:;!()*&-_?'):
                        continue

                    sen_idx = bisect.bisect_left(dot_index, start_idx)
                    if sen_idx != pre_sen_idx:
                        if len(cur_core_findings) != 0:
                            if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                                pass
                            else:
                                core_findings.append(' '.join(cur_core_findings))
                        cur_core_findings, previous_node_modified = [], False
                        if start_idx == dot_index[sen_idx]:
                            dot_s_idx = dot_index[sen_idx]
                            dot_e_idx = dot_index[sen_idx] + 1 if sen_idx == len(dot_index) - 1 else dot_index[
                                sen_idx + 1]
                            pre_sen_idx = sen_idx + 1
                        else:
                            dot_s_idx = dot_index[sen_idx - 1]
                            dot_e_idx = dot_index[sen_idx]
                            pre_sen_idx = sen_idx

                    if start_idx <= dot_e_idx < end_idx:
                        print(doc_key, "error!!", cur_ent)
                        cur_ent = cur_ent.split('.')[0].strip()
                    if "DA" in ent_label and not previous_node_modified:
                        cur_core_findings = ['no', *cur_core_findings]
                        previous_node_modified = True
                    elif "U" in ent_label and not previous_node_modified:
                        cur_core_findings = ["maybe", *cur_core_findings]
                        previous_node_modified = True
                    cur_core_findings.append(cur_ent)  # add word

                if len(cur_core_findings) != 0:
                    if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                        pass
                    else:
                        core_findings.append(' '.join(cur_core_findings))
                dict_entity['core_findings'] = core_findings
                final_dict.update({
                    doc_key: dict_entity
                })

        # =================== save files ======================#
        # with open(ent_path, 'w') as outfile:
        #     json.dump(final_dict, outfile, indent=2)
        return final_dict

    def preprocess_corpus_radgraph_output(self):
        # =====================build the serialized string=============#
        final_dict = {}
        useless_findings = useless_core_findings_new()
        # negative_list = ['no ', 'not ', 'free of', 'negative', 'without', 'clear of']  # delete unremarkable
        print("obtain the triples for each report.")
        with open(self.output_path, 'r') as f:
            for line in tqdm(f):
                data_item = json.loads(line)
                n = data_item['predicted_ner'][0]
                # r = data_item['predicted_relations'][0]
                s = data_item['sentences'][0]
                if len(n) == 0:
                    # print(len(n), " ".join(s))
                    continue

                doc_key = data_item['doc_key']
                # if doc_key in ['10000935_50578979', '10116310_50782200', '17277688_55816958', ]:
                #     print()
                n = preprocessing_entities(n, s, doc_key)
                dict_entity = {'text': ' '.join(s)}

                # ====Remove some useless entities and relationships====
                # initialized the variables
                dot_index = [index for index, token in enumerate(s) if token in ['.', '?', '!']]
                if len(dot_index) != 0:
                    if dot_index[0] != 0:
                        dot_index = [0, *dot_index]
                    if dot_index[-1] != len(s) - 1:
                        dot_index = [*dot_index, len(s)]
                    else:
                        dot_index[-1] += 1
                else:
                    dot_index = [0, len(s)]  # the last index + 1
                core_findings = []
                core_findings_index = []
                cur_ent_index_list = []
                dot_s_idx, dot_e_idx, pre_sen_idx = -1, -1, -1
                cur_core_findings, previous_node_modified = [], False
                # cur_core_findings: words of each sentence
                # core_findings: finding of each sentence
                for idx, ent_item in enumerate(n):
                    start_idx, end_idx, ent_label = ent_item[0], ent_item[1], ent_item[2].strip()
                    cur_ent = " ".join(s[start_idx:end_idx + 1]).strip('"').strip("'").strip()
                    # delete unrelated entities
                    if cur_ent in list(',:;!()*&-_?'):
                        continue
                    cur_ent_index = list(range(start_idx, end_idx+1))
                    sen_idx = bisect.bisect_left(dot_index, start_idx)
                    if sen_idx != pre_sen_idx:
                        if len(cur_core_findings) != 0:
                            if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                                pass
                            else:
                                core_findings.append(' '.join(cur_core_findings))
                                core_findings_index.append(cur_ent_index_list)
                        cur_core_findings, previous_node_modified = [], False
                        cur_ent_index_list = []
                        if start_idx == dot_index[sen_idx]:
                            dot_s_idx = dot_index[sen_idx]
                            dot_e_idx = dot_index[sen_idx] + 1 if sen_idx == len(dot_index) - 1 else dot_index[
                                sen_idx + 1]
                            pre_sen_idx = sen_idx + 1
                        else:
                            dot_s_idx = dot_index[sen_idx - 1]
                            dot_e_idx = dot_index[sen_idx]
                            pre_sen_idx = sen_idx

                    if start_idx <= dot_e_idx < end_idx:
                        print(doc_key, "error!!", cur_ent)
                        temp = cur_ent.split('.')[0].strip()
                        _idx = cur_ent.find(temp)
                        cur_ent_index = cur_ent_index[_idx: (_idx + len(temp))]
                        cur_ent = temp
                    if "DA" in ent_label and not previous_node_modified:
                        cur_core_findings = ['no', *cur_core_findings]
                        previous_node_modified = True
                    elif "U" in ent_label and not previous_node_modified:
                        cur_core_findings = ["maybe", *cur_core_findings]
                        previous_node_modified = True
                    cur_core_findings.append(cur_ent)  # add word
                    cur_ent_index_list.extend(cur_ent_index)

                if len(cur_core_findings) != 0:
                    if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                        pass
                    else:
                        core_findings.append(' '.join(cur_core_findings))
                        core_findings_index.append(cur_ent_index_list)
                dict_entity['report'] = s
                dict_entity['core_findings'] = core_findings
                dict_entity['core_findings_index'] = core_findings_index
                final_dict.update({
                    doc_key: dict_entity
                })

        # =================== save files ======================#
        # with open(ent_path, 'w') as outfile:
        #     json.dump(final_dict, outfile, indent=2)
        return final_dict


def preprocessing_entities(n, s, doc_key):
    new_n = []
    head_end_idx = -1
    for idx, item in enumerate(n, start=1):
        start_idx, end_idx, ent_label = item[0], item[1], item[2].strip()
        if start_idx > end_idx:
            continue
        elif start_idx <= head_end_idx:
            ori_s_idx, ori_e_idx = new_n[-1][0], new_n[-1][1]
            cur_best_str = ' '.join(s[ori_s_idx: (ori_e_idx + 1)])
            cur_str = ' '.join(s[start_idx: (end_idx + 1)])
            if ' .' in cur_best_str:
                if ' .' not in cur_str:
                    new_n.pop(-1)
                    new_n.append(item)
                    head_end_idx = end_idx
                    print(f"{doc_key} drop entities1: {cur_str} | {cur_best_str}")
                else:
                    print(f"{doc_key} drop entities2: {cur_best_str} | {cur_str}")
            else:
                if ' .' not in cur_str and ori_e_idx - ori_s_idx < (end_idx - start_idx):
                    new_n.pop(-1)
                    new_n.append(item)
                    head_end_idx = end_idx
                    print(f"{doc_key} drop entities3: {cur_str} | {cur_best_str}")
                else:
                    print(f"{doc_key} drop entities4: {cur_best_str} | {cur_str}")
            continue
        else:
            new_n.append(item)
            head_end_idx = end_idx
    return new_n


def useless_core_findings_new():
    result = {'down', 'surgery', 'port', 'wire', 'board', 'It', 'it', 'Body', 'upper',
              'fullness', 'line', 'anterior', 'support', 'Status', 'New', 'young',
              'rightward', 'Apices', 'leftward', 'hardware', 'resident', 'system',
              'level', 'Hardware', 'location', 'misdirected', 'external', 'Rotated', 'disease',
              'loops', 'course', 'off', 'new', 'opposite', 'lead', 'apices',
              'aspiration', 'midline', 'monitoring', 'bilaterally', 'size',
              'Position', 'overall', 'wires', 'standard', 'status',
              'anteriorly', 'right', 'patient', 'overlying', 'feeding', 'interval',
              'Otherwise', 'nodes', 'Multiple', 'rotation', 'findings',
              'habitus', 'positioning', 'Accessed', 'reading', 'placement', 'read',
              'levels', 'position', 'semi', 'otherwise', 'similar', 'curve',
              'postoperative', 'bases', 'surveillance', 'sited', 'medially',
              'region', 'They', 'mL', 'under', 'Left', 'limited', 'reposition',
              'repositioning', 'limitation', 'These', 'left', 'R', 'rotated', 'This',
              'post', 'Markedly', 'positioned', 'Post', }
    return result


def get_mimic_cxr_annotations(ann_path, ent_data, file_name):
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for split, value in ann_data.items():
        print(f"current preprocessing the {split}....")
        new_ann_data[split] = []
        for item in tqdm(value):
            try:
                doc_key = str(item['subject_id']) + '_' + str(item['study_id'])
                sample_core_finding = ent_data[doc_key]
                core_findings = sample_core_finding['core_findings']
                report = sample_core_finding['text']
            except:
                core_findings = []
                report = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                item['report'])
            sample_item = {
                'id': item['id'],
                "study_id": item['study_id'],
                'subject_id': item['subject_id'],
                "report": report,
                'image_path': item['image_path'],
                'core_findings': core_findings,
            }

            new_ann_data[split].append(sample_item)
    with open(file_name, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def get_mimic_cxr_annotations_temp(ann_path, ent_data, file_name):
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for split, value in ann_data.items():
        print(f"current preprocessing the {split}....")
        new_ann_data[split] = []
        for item in tqdm(value):
            try:
                doc_key = str(item['subject_id']) + '_' + str(item['study_id'])
                sample_core_finding = ent_data[doc_key]
                core_findings = sample_core_finding['core_findings']
                report = sample_core_finding['text']
            except:
                core_findings = []
                report = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                item['report'])
            sample_item = {
                'id': item['id'],
                "study_id": item['study_id'],
                'subject_id': item['subject_id'],
                "report": report,
                'image_path': item['image_path'],
                'core_findings': core_findings,
            }

            new_ann_data[split].append(sample_item)
    with open(file_name, 'w') as f:
        json.dump(new_ann_data, f, indent=2)




if __name__ == '__main__':
    # radgraph  from official checkpoint
    # it can be download in physionet.org, search radgraph
    radgraph_model_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'

    # obtain the mimic_cxr_annotation_sen.json, which does not include similar historical cases
    # note that "mimic_cxr_annotation_sen_best_reports_keywords_20.json" includes top20 similar historical cases
    # for each sample, which is obtained after performing pretrain_inference task in "main_finetune.py"
    root = '/media/miao/data/Dataset/MIMIC-CXR'
    ann_path = os.path.join(root, 'annotation.json')
    sen_ann_path = 'mimic_cxr_annotation_sen.json'
    # extract mimic-cxr factual serialization
    radgraph = RadGraphNER(ann_path=ann_path, is_get_output=True, is_mimic=True, model_path=radgraph_model_path, cuda='0')
    factual_serialization = radgraph.preprocess_mimic_radgraph_output()
    get_mimic_cxr_annotations(ann_path, factual_serialization, sen_ann_path)

    # extract factual serialization for some text
    hyps = ["patient is status post median sternotomy and cabg . the lungs are clear without focal consolidation . no pleural effusion or pneumothorax is seen . the cardiac and mediastinal silhouettes are unremarkable . no pulmonary edema is seen .",
            "___ year old woman with cirrhosis . patient is status post median sternotomy and cabg . the lungs are clear without focal consolidation . ",
    ]
    corpus = {i: item for i, item in enumerate(hyps[-1:])}
    radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_model_path, cuda='0')
    factual_serialization = radgraph.preprocess_corpus_radgraph_output()
    print(factual_serialization)

    # get_plot_cases_factual_serialization()
