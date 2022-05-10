import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

import config
import counting

from transformer_encoder import TransformerEncodeLayer

def to_one_hot(scores):
        """ Turn a bunch of non-negative scalar values into a one-hot encoding.
        E.g. with self.objects = 3, 0 -> [1 0 0 0], 2.75 -> [0 0 0.25 0.75].
        """
        # sanity check, I don't think this ever does anything (it certainly shouldn't)
        num_objects = 10
        scores = scores.clamp(min=0, max=num_objects) # max objects
        # compute only on the support
        i = scores.long().data
        f = scores.frac()
        # target_l is the one-hot if the score is rounded down
        # target_r is the one-hot if the score is rounded up
        target_l = scores.data.new(i.size(0), num_objects + 1).fill_(0)
        target_r = scores.data.new(i.size(0), num_objects + 1).fill_(0)

        target_l.scatter_(dim=1, index=i.clamp(max=num_objects), value=1)
        target_r.scatter_(dim=1, index=(
            i + 1).clamp(max=num_objects), value=1)
        # interpolate between these with the fractional part of the score
        return (1 - f) * Variable(target_l) + f * Variable(target_r)

class Net(nn.Module):
    """ Based on ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2
        objects = 10

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=glimpses,
            drop=0.5,
        )

        # MODIFIED TRANSFORMER WITH ATTENTION SCORES
        self.te = TransformerEncodeLayer(dim=1024, ff_dim=512, num_head=1)

        # PYTORCH TRANSFORMER ENCODER
        self.encoder_layer_pytorch = nn.TransformerEncoderLayer(
            d_model=1024, nhead=1)
        self.te_pytorch = nn.TransformerEncoder(
            self.encoder_layer_pytorch, num_layers=1)

        # self.classifier = Classifier(
        #     in_features=(glimpses * vision_features, question_features),
        #     mid_features=1024,
        #     out_features=config.max_answers,
        #     count_features=objects + 1,
        #     drop=0.5,
        # )

        self.classifier = nn.Linear(
            in_features=1035, out_features=config.max_answers)

        self.bert_to_1024 = nn.Linear(in_features=768, out_features=1024)
        # self.classifier_no_count = nn.Linear(
        #     in_features=1035, out_features=config.max_answers)

        self.tokenizer = torch.hub.load(
            'huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self.model_bert = torch.hub.load(
            'huggingface/pytorch-transformers', 'model', 'bert-base-cased').cuda()

        self.classifier_no_count = nn.Linear(
            in_features=1024, out_features=config.max_answers)

        self.counter = counting.Counter(objects)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, co, q, q_len):
        # q = self.text(q, list(q_len.data))
        indexed_tokens = self.tokenizer.batch_encode_plus(
            q, return_tensors='pt', add_special_tokens=True, padding=True)
        for key, val in indexed_tokens.items():
            indexed_tokens[key] = indexed_tokens[key].cuda()
        # indexed_tokens = indexed_tokens.cuda()
        # tokens_tensor = torch.tensor([indexed_tokens])
        # indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        self.model_bert.eval()
        with torch.no_grad():
            q = self.model_bert(**indexed_tokens)[0]

        v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v)

        v_sq = torch.squeeze(v, dim=2)
        v_per = torch.permute(v_sq, (0, 2, 1))
        # UNCOMMENT for LSTM
        # q_mean = torch.unsqueeze(torch.mean(q, dim=1), dim=1) 
        q_mean = torch.mean(q, dim=1)
        q_mean = torch.unsqueeze(self.bert_to_1024(q_mean), dim=1)
        
        te_input = torch.cat((q_mean, v_per), dim=1)

        te_out = self.te(te_input, None)

        # UNCOMMENT FOR COUNT MODULE
        te_enc, te_a1 = te_out[0], te_out[1]

        # COMMENT FOR COUNT MODULE
        # te_enc = te_out

        te_enc = torch.mean(te_enc, dim=1)
        # a = self.attention(v, q)
        # v = apply_attention(v, a)

        # this is where the counting component is used
        # pick out the first attention map
        # a1 = a[:, 0, :, :].contiguous().view(a.size(0), -1)

        # UNCOMMENT FOR **OUR** COUNT MODULE
        # te_a1 = torch.squeeze(te_a1, dim=1)[:, 0, 1:]

        # give it and the bounding boxes to the component
        # count = self.counter(b, a1)

        # UNCOMMENT FOR COUNT MODULE
        # pre_count = torch.mul(co, te_a1)
        # count     = to_one_hot(torch.sum(pre_count, dim=1, keepdim=True))

        # answer = self.classifier(v, q, count)

        # UNCOMMENT FOR COUNT MODULE
        answer = self.classifier_no_count(te_enc)

        # COMMENT FOR COUNT MODULE
        # answer = self.classifier_no_count(te_enc)

        return answer


class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.relu(x + y)


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, count_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.fusion = Fusion()
        self.lin11 = nn.Linear(in_features[0], mid_features)
        self.lin12 = nn.Linear(in_features[1], mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.lin_c = nn.Linear(count_features, mid_features)
        self.bn = nn.BatchNorm1d(mid_features)
        self.bn2 = nn.BatchNorm1d(mid_features)

    def forward(self, x, y, c):
        x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
        x = x + self.bn2(self.relu(self.lin_c(c)))
        x = self.lin2(self.drop(self.bn(x)))
        return x


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(
            embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.GRU(input_size=embedding_features,
                           hidden_size=lstm_features,
                           num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(3, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        # packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        out, h = self.lstm(tanhed)
        # return h.squeeze(0)
        return out


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        # let self.lin take care of bias
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = Fusion()

    def forward(self, v, q):
        q_in = q
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.fusion(v, q)
        x = self.x_conv(self.drop(x))
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input.
        The attention map has to have the same size in all dimensions except dim=1.
    """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, c, -1)
    attention = attention.view(n, glimpses, -1)
    s = input.size(2)

    # apply a softmax to each attention map separately
    # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
    # so that each glimpse is normalized separately
    attention = attention.view(n * glimpses, -1)
    attention = F.softmax(attention, dim=1)

    # apply the weighting by creating a new dim to tile both tensors over
    target_size = [n, glimpses, c, s]
    input = input.view(n, 1, c, s).expand(*target_size)
    attention = attention.view(n, glimpses, 1, s).expand(*target_size)
    weighted = input * attention
    # sum over only the spatial dimension
    weighted_mean = weighted.sum(dim=3, keepdim=True)
    # the shape at this point is (n, glimpses, c, 1)
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_sizes = feature_map.size()[2:]
    tiled = feature_vector.view(
        n, c, *([1] * len(spatial_sizes))).expand(n, c, *spatial_sizes)
    return tiled
