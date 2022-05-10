import json
import os
import os.path
import re

from PIL import Image
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

import config
import utils
import jsonlines
import collections
import pickle
import scipy.io

preloaded_vocab = None


def get_ve_loader(mode):
    """ Returns a data loader for the desired split """
    # split = dataset_ve.SNLIVEDataset(
    #     utils.path_for(train=train, val=val, test=test, question=True),
    #     utils.path_for(train=train, val=val, test=test, answer=True),
    #     config.preprocessed_trainval_path if not test else config.preprocessed_test_path,
    #     answerable_only=train,
    #     dummy_answers=test,
    # )

    snli_dataset = SNLIVEDatasetBERT(split=mode, features_dir="/mnt/data/rpn_features_proposals_npy/", bbox_dir="/mnt/data/rpn_bbox_mat/",
                                     snli_file_path="/mnt/data/SNLI-VE/data/", max_hypothesis_len=56, vocab_size=10000)

    train = True if mode == "train" else False
    # print("Shuffle: ", train)
    loader = torch.utils.data.DataLoader(
        snli_dataset,
        batch_size=config.batch_size,
        shuffle=train,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_ve_fn,
    )
    return loader


def collate_ve_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


class SNLIVEDatasetHDF5(data.Dataset):
    def __init__(self, split, features_hdf5, imgid2idx_file, snli_file_path, max_hypothesis_len, vocab=None, vocab_size=10000):
        super().__init__()

        if split == "val":
            split_file = os.path.join(
                snli_file_path, 'snli_ve_{split}.jsonl'.format(split="dev"))
        else:
            split_file = os.path.join(
                snli_file_path, 'snli_ve_{split}.jsonl'.format(split=split))

        self.label_map = {"contradiction": 0, "neutral": 1, "entailment": 2}
        self.idx2labels = {}  # maps idx to labels
        self.idx2hypotheses = {}  # maps idx to preprocessed hypothesis
        self.idx_map = {}  # maps idx to image id

        self.get_hypotheses(split_file)

        self.vocab_size = vocab_size
        if vocab is None:
            self.vocab = self.construct_vocab()
        else:
            self.vocab = vocab

        self.max_hypothesis_len = max_hypothesis_len

        # self.features_dir = features_dir
        # self.feature_file = None

        self.features_hdf5 = h5py.File(features_hdf5)

        with open(imgid2idx_file, 'rb') as f:
            self.imgid2idx = pickle.load(f)

        # self.bbox_dir = bbox_dir
        self.feature_file = None

    def preprocess_hypothesis(self, hypothesis):
        hypothesis = re.sub(r'[^\w\s]', '', hypothesis.lower().strip())
        return hypothesis

    def tokenize(self, hypothesis):
        return hypothesis.split()

    def get_hypotheses(self, path):
        """
        Extracts and preprocesses hypotheses from jsonl file. Extracts and saves labels from jsonl file
        :param path: path to jsonl file
        :return: None
        """
        idx = 0
        with jsonlines.open(path) as fp:
            for line in fp:
                try:
                    image_id = line['pairID']
                    # print(image_id)
                    image_id = image_id[0: image_id.index(".jpg")]
                    hypothesis = self.preprocess_hypothesis(line['sentence2'])
                    label = torch.zeros(3)
                    label[self.label_map[line['gold_label']]] = 1
                    self.idx2hypotheses[idx] = hypothesis
                    self.idx2labels[idx] = label
                    self.idx_map[idx] = image_id
                    idx += 1
                except Exception as e:
                    print(e, image_id)
                    continue

    def construct_vocab(self):
        tokens = []
        for idx, hypothesis in self.idx2hypotheses.items():
            tokens.extend(self.tokenize(hypothesis))

        counter = collections.Counter(tokens)
        vocab_words = counter.most_common(self.vocab_size)

        vocab = {}
        idx = 0
        for token, count in vocab_words:
            vocab[token] = idx
            idx += 1

        return vocab

    def encode_hypothesis(self, hypothesis):
        tokens = self.tokenize(hypothesis)
        hyp_vec = torch.zeros(self.max_hypothesis_len)

        #### CHANGE - Clipping tokens to max_hypothesis_len ####
        tokens = tokens[:self.max_hypothesis_len]
        ################
        for i, token in enumerate(tokens):
            hyp_vec[i] = self.vocab.get(token, len(self.vocab))

        return hyp_vec, len(tokens)

    def __len__(self):
        return len(self.idx2hypotheses)

    def __getitem__(self, idx):
        image_id = self.idx_map[idx]
        hypothesis, q_length = self.encode_hypothesis(self.idx2hypotheses[idx])
        label = self.idx2labels[idx]

        # features = torch.from_numpy(
        #     np.load(os.path.join(self.features_dir, image_id+".npy")))
        # bb_boxes = torch.from_numpy(scipy.io.loadmat(os.path.join(
        #     self.bbox_dir, image_id+".mat"))['cur_bbxes'].astype(np.float16))
        feature_index = self.imgid2idx[image_id]

        features = torch.tensor(
            self.features_hdf5['img_features'][feature_index])
        bb_boxes = torch.tensor((self.features_hdf5['bboxes'][feature_index]))

        # Take only first config.output_size bounding boxes and features
        bb_boxes = bb_boxes[:config.output_size]
        features = features[:config.output_size]

        # Convert features to (4096, 1, 100)
        features = torch.unsqueeze(features, dim=0)
        features = torch.permute(features, (2, 0, 1))

        # Convert bb_boxes to [xmin, ymin, xmax, ymax]
        # bb_boxes[:, [0, 1, 2, 3]] = bb_boxes[:, [1, 0, 3, 2]]

        # Convert bb_boxes to [4, 100] shape
        bb_boxes = torch.permute(bb_boxes, (1, 0))

        # Convert question to int type
        hypothesis = hypothesis.long()

        # Convert visual features and bounding boxes to float
        features = features.float()
        bb_boxes = bb_boxes.float()

        # return {"features": features, "bboxes": bb_boxes, "hypothesis": hypothesis, "label": label}
        # v, q, a, b, item, q_length
        return features, hypothesis, label, bb_boxes, idx, q_length

    @property
    def num_tokens(self):
        return self.vocab_size + 1  # add 1 for <unknown> token at index 0


class SNLIVEDatasetLSTM(data.Dataset):
    def __init__(self, split, features_dir, bbox_dir, snli_file_path, max_hypothesis_len, vocab=None, vocab_size=10000):
        super().__init__()

        if split == "val":
            split_file = os.path.join(
                snli_file_path, 'snli_ve_{split}.jsonl'.format(split="dev"))
        else:
            split_file = os.path.join(
                snli_file_path, 'snli_ve_{split}.jsonl'.format(split=split))

        self.label_map = {"contradiction": 0, "neutral": 1, "entailment": 2}
        self.idx2labels = {}  # maps idx to labels
        self.idx2hypotheses = {}  # maps idx to preprocessed hypothesis
        self.idx_map = {}  # maps idx to image id

        self.get_hypotheses(split_file)

        self.vocab_size = vocab_size
        if vocab is None:
            self.vocab = self.construct_vocab()
        else:
            self.vocab = vocab

        self.max_hypothesis_len = max_hypothesis_len

        self.features_dir = features_dir
        self.feature_file = None

        self.bbox_dir = bbox_dir

    def preprocess_hypothesis(self, hypothesis):
        hypothesis = re.sub(r'[^\w\s]', '', hypothesis.lower().strip())
        return hypothesis

    def tokenize(self, hypothesis):
        return hypothesis.split()

    def get_hypotheses(self, path):
        """
        Extracts and preprocesses hypotheses from jsonl file. Extracts and saves labels from jsonl file
        :param path: path to jsonl file
        :return: None
        """
        idx = 0
        with jsonlines.open(path) as fp:
            for line in fp:
                try:
                    image_id = line['pairID']
                    # print(image_id)
                    image_id = image_id[0: image_id.index(".jpg")]
                    hypothesis = self.preprocess_hypothesis(line['sentence2'])
                    label = torch.zeros(3)
                    label[self.label_map[line['gold_label']]] = 1
                    self.idx2hypotheses[idx] = hypothesis
                    self.idx2labels[idx] = label
                    self.idx_map[idx] = image_id
                    idx += 1
                except Exception as e:
                    print(e, image_id)
                    continue

    def construct_vocab(self):
        tokens = []
        for idx, hypothesis in self.idx2hypotheses.items():
            tokens.extend(self.tokenize(hypothesis))

        counter = collections.Counter(tokens)
        vocab_words = counter.most_common(self.vocab_size)

        vocab = {}
        idx = 0
        for token, count in vocab_words:
            vocab[token] = idx
            idx += 1

        return vocab

    def encode_hypothesis(self, hypothesis):
        tokens = self.tokenize(hypothesis)
        hyp_vec = torch.zeros(self.max_hypothesis_len)

        #### CHANGE - Clipping tokens to max_hypothesis_len ####
        tokens = tokens[:self.max_hypothesis_len]
        ################
        for i, token in enumerate(tokens):
            hyp_vec[i] = self.vocab.get(token, len(self.vocab))

        return hyp_vec, len(tokens)

    def __len__(self):
        return len(self.idx2hypotheses)

    def __getitem__(self, idx):
        image_id = self.idx_map[idx]
        hypothesis, q_length = self.encode_hypothesis(self.idx2hypotheses[idx])
        label = self.idx2labels[idx]

        features = torch.from_numpy(
            np.load(os.path.join(self.features_dir, image_id+".npy")))
        # CHANGED AFTER RPN BBOXES
        bb_boxes = torch.from_numpy(scipy.io.loadmat(os.path.join(
            self.bbox_dir, image_id+".mat"))['bboxes'].astype(np.float16))

        # Take only first 50 bounding boxes and features
        bb_boxes = bb_boxes[:config.output_size]
        features = features[:config.output_size]

        # Convert features to (4096, 1, 100)
        features = torch.unsqueeze(features, dim=0)
        features = torch.permute(features, (2, 0, 1))

        # Convert bb_boxes to [xmin, ymin, xmax, ymax]
        # bb_boxes[:, [0, 1, 2, 3]] = bb_boxes[:, [1, 0, 3, 2]]

        # Convert bb_boxes to [4, 100] shape
        bb_boxes = torch.permute(bb_boxes, (1, 0))

        # Convert question to int type
        hypothesis = hypothesis.long()

        # Convert visual features and bounding boxes to float
        features = features.float()
        bb_boxes = bb_boxes.float()

        # return {"features": features, "bboxes": bb_boxes, "hypothesis": hypothesis, "label": label}
        # v, q, a, b, item, q_length
        return features, hypothesis, label, bb_boxes, idx, q_length

    @property
    def num_tokens(self):
        return self.vocab_size + 1  # add 1 for <unknown> token at index 0


class SNLIVEDatasetBERT(data.Dataset):
    def __init__(self, split, features_dir, bbox_dir, snli_file_path, max_hypothesis_len, vocab=None, vocab_size=10000):
        super().__init__()

        if split == "val":
            split_file = os.path.join(
                snli_file_path, 'snli_ve_{split}.jsonl'.format(split="dev"))
        else:
            split_file = os.path.join(
                snli_file_path, 'snli_ve_{split}.jsonl'.format(split=split))

        self.label_map = {"contradiction": 0, "neutral": 1, "entailment": 2}
        self.idx2labels = {}  # maps idx to labels
        self.idx2hypotheses = {}  # maps idx to preprocessed hypothesis
        self.idx_map = {}  # maps idx to image id

        self.get_hypotheses(split_file)

        self.vocab_size = vocab_size
        if vocab is None:
            self.vocab = self.construct_vocab()
        else:
            self.vocab = vocab

        self.max_hypothesis_len = max_hypothesis_len

        self.features_dir = features_dir
        self.feature_file = None

        self.bbox_dir = bbox_dir

    def preprocess_hypothesis(self, hypothesis):
        hypothesis = re.sub(r'[^\w\s]', '', hypothesis.lower().strip())
        return hypothesis

    def tokenize(self, hypothesis):
        return hypothesis.split()

    def get_hypotheses(self, path):
        """
        Extracts and preprocesses hypotheses from jsonl file. Extracts and saves labels from jsonl file
        :param path: path to jsonl file
        :return: None
        """
        idx = 0
        with jsonlines.open(path) as fp:
            for line in fp:
                try:
                    image_id = line['pairID']
                    # print(image_id)
                    image_id = image_id[0: image_id.index(".jpg")]
                    hypothesis = self.preprocess_hypothesis(line['sentence2'])
                    label = torch.zeros(3)
                    label[self.label_map[line['gold_label']]] = 1
                    self.idx2hypotheses[idx] = hypothesis
                    self.idx2labels[idx] = label
                    self.idx_map[idx] = image_id
                    idx += 1
                except Exception as e:
                    print(e, image_id)
                    continue

    def construct_vocab(self):
        tokens = []
        for idx, hypothesis in self.idx2hypotheses.items():
            tokens.extend(self.tokenize(hypothesis))

        counter = collections.Counter(tokens)
        vocab_words = counter.most_common(self.vocab_size)

        vocab = {}
        idx = 0
        for token, count in vocab_words:
            vocab[token] = idx
            idx += 1

        return vocab

    def encode_hypothesis(self, hypothesis):
        tokens = self.tokenize(hypothesis)
        hyp_vec = torch.zeros(self.max_hypothesis_len)

        #### CHANGE - Clipping tokens to max_hypothesis_len ####
        tokens = tokens[:self.max_hypothesis_len]
        ################
        for i, token in enumerate(tokens):
            hyp_vec[i] = self.vocab.get(token, len(self.vocab))

        return hyp_vec, len(tokens)

    def __len__(self):
        return len(self.idx2hypotheses)

    def __getitem__(self, idx):
        image_id = self.idx_map[idx]
        hypothesis, q_length = self.encode_hypothesis(self.idx2hypotheses[idx])
        # FOR BERT
        hypothesis_text = self.idx2hypotheses[idx]

        label = self.idx2labels[idx]

        features = torch.from_numpy(
            np.load(os.path.join(self.features_dir, image_id+".npy")))
        # CHANGED AFTER RPN BBOXES
        bb_boxes = torch.from_numpy(scipy.io.loadmat(os.path.join(
            self.bbox_dir, image_id+".mat"))['bboxes'].astype(np.float16))

        # Take only first 50 bounding boxes and features
        bb_boxes = bb_boxes[:config.output_size]
        features = features[:config.output_size]

        # Convert features to (4096, 1, 100)
        features = torch.unsqueeze(features, dim=0)
        features = torch.permute(features, (2, 0, 1))

        # Convert bb_boxes to [xmin, ymin, xmax, ymax]
        # bb_boxes[:, [0, 1, 2, 3]] = bb_boxes[:, [1, 0, 3, 2]]

        # Convert bb_boxes to [4, 100] shape
        bb_boxes = torch.permute(bb_boxes, (1, 0))

        # Convert question to int type
        hypothesis = hypothesis.long()

        # Convert visual features and bounding boxes to float
        features = features.float()
        bb_boxes = bb_boxes.float()

        # return {"features": features, "bboxes": bb_boxes, "hypothesis": hypothesis, "label": label}
        # v, q, a, b, item, q_length
        # return features, hypothesis, label, bb_boxes, idx, q_length
        return features, hypothesis_text, label, bb_boxes, idx, q_length

    @property
    def num_tokens(self):
        return self.vocab_size + 1  # add 1 for <unknown> token at index 0
