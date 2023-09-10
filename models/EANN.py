from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import BertModel
from .Adapter import *
import math
import copy

class ReverseLayerF(Function):
    #@staticmethod
    def forward(self, x):
        self.lambd = 1
        return x.view_as(x)

    #@staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x):
    return ReverseLayerF().apply(x)

class CNN_Fusion(nn.Module):
    def __init__(self, args, W):
        super(CNN_Fusion, self).__init__()
        self.emb_type = args.emb_type
        if self.emb_type == 'bert':
            self.bert = BertModel.from_pretrained(args.bert,output_hidden_states=True).requires_grad_(False)
        self.event_num = args.event_num
        self.class_num = args.class_num
        self.hidden_size = args.hidden_dim
        # self.lstm_size = args.embed_dim
        # self.social_size = 19

        # TEXT RNN
        self.embed = nn.Embedding(args.vocab_size, args.embed_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))
        # self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        # self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        # self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        ### TEXT CNN
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, args.embed_dim)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)

        # self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        # hidden_size = args.hidden_dim
        vgg_19 = torchvision.models.vgg19(pretrained=False)
        vgg_19.load_state_dict(torch.load('./models/vgg19.pth'))
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        # self.image_fc2 = nn.Linear(512, self.hidden_size)
        # self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        # self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        ###social context
        # self.social = nn.Linear(self.social_size, self.hidden_size)

        ##ATTENTION
        # self.attention_layer = nn.Linear(self.hidden_size, emb_dim)

        ## Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size,self.class_num))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(2 * self.hidden_size, self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def forward(self, text, image, text_mask=None):
        ### IMAGE #####
        image = self.vgg(image)  # [N, 1000]
        image = F.leaky_relu(self.image_fc1(image))

        ##########CNN##################
        if text_mask is None:
            text_mask = torch.ones_like(text)

        if self.emb_type == 'bert':
            output = self.bert(text, attention_mask = text_mask)
            # print(vars(output).keys())
            text = output['last_hidden_state']
        elif self.emb_type == 'w2v':
            text = self.embed(text)  # [batch,seq_len,embed_dim]
            text = text * text_mask.unsqueeze(2).expand_as(text)  # [batch,seq_len]->[batch,seq_len,embed_dim]
        text = text.unsqueeze(1)  # [batch,1,seq_len,embed_dim]
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text] #[batch,20,...]->[batch,20]
        text = torch.cat(text, 1)  # [batch,80]
        text = F.leaky_relu(self.fc1(text))  # [batch,hidden_dim]
        text_image = torch.cat((text, image), 1)  # [batch,2*hidden_dim]

        ### Fake or real
        class_output = self.class_classifier(text_image)  # [batch,2]
        ## Domain Event label
        reverse_feature = grad_reverse(text_image)
        domain_output = self.domain_classifier(reverse_feature)  # [batch,10]

        # ### Multimodal
        # text_reverse_feature = grad_reverse(text)
        # image_reverse_feature = grad_reverse(image)
        # text_output = self.modal_classifier(text_reverse_feature)
        # image_output = self.modal_classifier(image_reverse_feature
        return class_output, domain_output

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_dim,num_attention_heads,attention_dropout_prob=0.1):
        super(BertSelfAttention, self).__init__()
        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_dim / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_dim,hidden_dropout_prob):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_dim,hidden_dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, hidden_dim, intermediate_size,hidden_act,hidden_dropout_prob=0.1):
        super(Intermediate, self).__init__()
        self.dense1 = nn.Linear(hidden_dim, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

        self.dense2 = nn.Linear(intermediate_size, hidden_dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class Transformer(nn.Module):
    def __init__(self,hidden_dim,num_attention_heads,intermediate_size,adapter=None,hidden_act="gelu",dropout_prob=0.1):
        super(Transformer, self).__init__()
        self.self = BertSelfAttention(hidden_dim=hidden_dim,num_attention_heads=num_attention_heads,attention_dropout_prob=dropout_prob)
        self.intermediate = Intermediate(hidden_dim=hidden_dim, intermediate_size=intermediate_size,
                                         hidden_act=hidden_act, hidden_dropout_prob=dropout_prob)
        self.norm_layer1 = BertLayerNorm(hidden_dim=hidden_dim)
        self.norm_layer2 = BertLayerNorm(hidden_dim=hidden_dim)
        self.adapter1 = adapter
        self.adapter2 = copy.deepcopy(adapter)

    def forward(self,input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        self_output = self.self(input_tensor)
        attention_output = self.adapter1(self_output)
        attention_output = self.norm_layer1(attention_output+input_tensor)
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.adapter2(intermediate_output)
        layer_out = self.norm_layer2(intermediate_output+attention_output)

        return layer_out.squeeze(1)



class Adapter_EANN(nn.Module):
    def __init__(self, args, W,adapter):

        super(Adapter_EANN, self).__init__()

        self.emb_type = args.emb_type
        if self.emb_type == 'bert':
            self.bert = BertModel.from_pretrained(args.bert).requires_grad_(False)
        self.event_num = args.event_num
        self.class_num = args.class_num
        self.hidden_size = args.hidden_dim
        # self.lstm_size = args.embed_dim
        # self.social_size = 19

        # TEXT RNN
        self.embed = nn.Embedding(args.vocab_size, args.embed_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))
        # self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        # self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        # self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        ### TEXT CNN
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, args.embed_dim)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)
        # self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        vgg_19 = torchvision.models.vgg19(pretrained=False)
        vgg_19.load_state_dict(torch.load('./models/vgg19.pth'))
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        # self.image_fc2 = nn.Linear(512, self.hidden_size)
        # self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        # self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        ###social context
        # self.social = nn.Linear(self.social_size, self.hidden_size)

        ##ATTENTION
        # self.attention_layer = nn.Linear(self.hidden_size, emb_dim)
        self.transformer = Transformer(hidden_dim=64,num_attention_heads=4,intermediate_size=128,adapter=adapter)

        ## Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, self.class_num))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        # self.class_classifier.add_module('c_fc2', nn.Linear(self.hidden_size, 2))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(self.hidden_size))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(2 * self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def forward(self, text, image, mask):
        ### IMAGE #####
        image = self.vgg(image)  # [N, 1000]
        image = F.leaky_relu(self.image_fc1(image))

        ##########CNN##################
        if self.emb_type == 'bert':
            text = self.bert(text, attention_mask = mask)[0]
        elif self.emb_type == 'w2v':
            text = self.embed(text)  # [batch,seq_len,embed_dim]
            text = text * mask.unsqueeze(2).expand_as(text)  # [batch,seq_len]->[batch,seq_len,embed_dim]
        text = text.unsqueeze(1)  # [batch_size,1,seq_len,embed_dim]
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)  # [batch,80]
        text = F.leaky_relu(self.fc1(text))  # [batch,hidden_dim]
        text_image = torch.cat((text, image), 1)  # [batch,2*hidden_dim]

        # plug-in module
        text_image = self.transformer(text_image)

        ### Fake or real
        class_output = self.class_classifier(text_image)  # [batch,2]
        ## Domain (which Event )
        reverse_feature = grad_reverse(text_image)
        domain_output = self.domain_classifier(reverse_feature)  # [batch,10]

        # ### Multimodal
        # text_reverse_feature = grad_reverse(text)
        # image_reverse_feature = grad_reverse(image)
        # text_output = self.modal_classifier(text_reverse_feature)
        # image_output = self.modal_classifier(image_reverse_feature
        return class_output, domain_output