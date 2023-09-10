import torch
import torch.nn as nn
import math
import torchvision
from transformers import BertModel,RobertaModel
from .DCT_module import Frequency_extractor
import copy

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


class CoAttention(nn.Module):
    def __init__(self,hidden_dim,num_attention_heads,attention_dropout_prob=0.1):
        super(CoAttention, self).__init__()
        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim,num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(
            hidden_dim / num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(hidden_dim, self.all_head_size)
        self.key1 = nn.Linear(hidden_dim, self.all_head_size)
        self.value1 = nn.Linear(hidden_dim, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)
        self.dropout1 = nn.Dropout(attention_dropout_prob)

        self.query2 = nn.Linear(hidden_dim, self.all_head_size)
        self.key2 = nn.Linear(hidden_dim, self.all_head_size)
        self.value2 = nn.Linear(hidden_dim, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)
        self.dropout2 = nn.Dropout(attention_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor1,
        input_tensor2,

    ):

        # for vision input.
        #[batch_size,seq_length,hidden_dim]->[batch_size,seq_length,all_head_size]
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)
        #[batch_size,seq_length,all_head_size]->[batch_size,num_attention_heads,seq_length,attention_head_size]
        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        # [batch_size,seq_length,hidden_dim]->[batch_size,seq_length,all_head_size]
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)
        # [batch_size,seq_length,all_head_size]->[batch_size,num_attention_heads,seq_length,attention_head_size]
        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        #matmul:[batch_size,num_attention_heads,seq_length,seq_length]
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        #[batch_size,num_attention_heads,seq_length,seq_length]->[batch_size,num_attention_heads,seq_length,attention_head_size]
        #permute:->[batch_size,seq_length,num_attention_heads,attention_head_size]
        #view:->[batch_size,seq_length,all_head_size]
        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        return context_layer1, context_layer2


class CoAttentionOutput(nn.Module):
    def __init__(self, hidden_dim,hidden_dropout_prob):
        super(CoAttentionOutput, self).__init__()

        self.dense1 = nn.Linear(hidden_dim,hidden_dim)
        self.dropout1 = nn.Dropout(hidden_dropout_prob)

        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states1, hidden_states2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        return context_state1, context_state2

class CoAttentionLayerNorm(nn.Module):
    def __init__(self,hidden_dim):
        super(CoAttentionLayerNorm, self).__init__()
        self.LayerNorm1 = BertLayerNorm(hidden_dim, eps=1e-12)
        self.LayerNorm2 = BertLayerNorm(hidden_dim, eps=1e-12)
    def forward(self, hidden_states1, input_tensor1, hidden_states2,input_tensor2):
        hidden_states1 = self.LayerNorm1(hidden_states1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(hidden_states2 + input_tensor2)
        return hidden_states1,hidden_states2


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


class CoConnectionBlock(nn.Module):
    def __init__(self,hidden_dim,num_attention_heads,intermediate_size,adapter=None,hidden_act="gelu",dropout_prob=0.1):
        super(CoConnectionBlock, self).__init__()
        self.CoAttention = CoAttention(hidden_dim=hidden_dim,num_attention_heads=num_attention_heads,attention_dropout_prob=dropout_prob)
        self.CoAttentionOutput = CoAttentionOutput(hidden_dim=hidden_dim,hidden_dropout_prob=dropout_prob)
        self.CoAttentionLayerNorm = CoAttentionLayerNorm(hidden_dim=hidden_dim)

        self.intermediate = Intermediate(hidden_dim=hidden_dim, intermediate_size=intermediate_size,hidden_act=hidden_act,hidden_dropout_prob=dropout_prob)
        self.norm_layer1 = BertLayerNorm(hidden_dim=hidden_dim)
        self.norm_layer2 = BertLayerNorm(hidden_dim=hidden_dim)
        self.co_embed = nn.Linear(2*hidden_dim,hidden_dim)

        self.adapter1 = adapter
        self.adapter2 = copy.deepcopy(adapter)
        self.adapter3 = copy.deepcopy(adapter)
        self.adapter4 = copy.deepcopy(adapter)

        # self.adapter_norm_layer1 = BertLayerNorm(hidden_dim=hidden_dim)
        # self.adapter_norm_layer2 = BertLayerNorm(hidden_dim=hidden_dim)
        # self.adapter_norm_layer3 = BertLayerNorm(hidden_dim=hidden_dim)
        # self.adapter_norm_layer4 = BertLayerNorm(hidden_dim=hidden_dim)

    def forward(self,input_tensor1,input_tensor2):
        #input_tensor [B,1,256]
        input_tensor1 = input_tensor1.unsqueeze(1)
        input_tensor2 = input_tensor2.unsqueeze(1)
        coatt_output1, coatt_output2 = self.CoAttention(input_tensor1,input_tensor2)
        # coatt_output1, coatt_output2 = self.CoAttentionOutput(coatt_output1, coatt_output2)
        # norm_layer before adapter
        # coatt_output1 = self.adapter_norm_layer1(coatt_output1+input_tensor2)
        # coatt_output2 = self.adapter_norm_layer2(coatt_output2+input_tensor1)

        if self.adapter1:
            coatt_output1 = self.adapter1(coatt_output1)
            coatt_output2 = self.adapter2(coatt_output2)

        attention_output1, attention_output2 = self.CoAttentionLayerNorm(coatt_output2, input_tensor1, coatt_output1, input_tensor2)
        intermediate_output1 = self.intermediate(attention_output1)
        intermediate_output2 = self.intermediate(attention_output2)
        # norm_layer before adapter
        # intermediate_output1 = self.adapter_norm_layer3(intermediate_output1+attention_output1)
        # intermediate_output2 = self.adapter_norm_layer4(intermediate_output2+attention_output2)

        if self.adapter3:
            intermediate_output1 = self.adapter3(intermediate_output1)
            intermediate_output2 = self.adapter4(intermediate_output2)


        layer_output1 = self.norm_layer1(intermediate_output1+attention_output1)
        layer_output2 = self.norm_layer2(intermediate_output2+attention_output2)
        layer_concat = torch.cat((layer_output1,layer_output2),dim=2).squeeze(1)
        layer_out = self.co_embed(layer_concat)

        return layer_out


class MCAN(nn.Module):
    def __init__(self,bert_ckpt,args,adapter=None,hidden_dim=256):
        super(MCAN, self).__init__()

        # CNN_pretrain_model
        self.image_feature_extractor = torchvision.models.vgg19(pretrained=False)
        self.image_feature_extractor.load_state_dict(torch.load('./models/vgg19.pth'), strict=False)
        for param in self.image_feature_extractor.parameters():
            param.requires_grad = False
        in_features = self.image_feature_extractor.classifier[3].in_features
        # self.image_feature_extractor.classifier[3] = nn.Linear(in_features, 256)
        self.s_fc = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(inplace=True))

        del self.image_feature_extractor.classifier[6]
        del self.image_feature_extractor.classifier[5]
        del self.image_feature_extractor.classifier[4]
        del self.image_feature_extractor.classifier[3]

        # DCT(Discrete Cosine Transform) model
        self.image_frequency_extractor = Frequency_extractor(args)
        for param in self.image_frequency_extractor.parameters():
            param.requires_grad_(False)
        self.f_fc = nn.Sequential(nn.Linear(64,256),nn.ReLU(inplace=True))
        #bert pretrain_model
        self.text_feature_extractor = BertModel.from_pretrained(bert_ckpt, output_hidden_states=False).requires_grad_(False)
        self.t_fc = nn.Sequential(nn.Linear(args.max_len * args.bert_emb_dim, hidden_dim), nn.ReLU(inplace=True))
        #CoConnectionlayer
        self.coconnectlayer = nn.ModuleList()
        for _ in range(4):
            self.coconnectlayer.append(copy.deepcopy(CoConnectionBlock(hidden_dim=hidden_dim,
                                                         num_attention_heads=4,
                                                         intermediate_size=512,
                                                         adapter=adapter)))

        self.p_fc = nn.Sequential(nn.Linear(hidden_dim, 35), nn.ReLU(inplace=True),
                                  nn.Linear(35, 2))

    def forward(self,text,image,text_mask=None):
        #extract image feature [B,256]
        image_feature_middle = self.image_feature_extractor(image)
        image_feature = self.s_fc(image_feature_middle)
        #extract dct frequency feature [B,256]
        freq_feature_middle = self.image_frequency_extractor(image)
        freq_feature = self.f_fc(freq_feature_middle)
        # text_feature [B,Seq,hidden_dim]
        text_feature = self.text_feature_extractor(text)['last_hidden_state']
        # extract text feature [B,256]
        text_feature = self.t_fc(text_feature.flatten(1))
        layer_out1 = self.coconnectlayer[0](image_feature, freq_feature)
        layer_out2 = self.coconnectlayer[1](layer_out1, freq_feature)
        layer_out3 = self.coconnectlayer[2](layer_out2, text_feature)
        layer_out4 = self.coconnectlayer[3](layer_out3, text_feature)
        result = self.p_fc(layer_out4)

        return result






