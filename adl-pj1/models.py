import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F
from torchvision.models import vgg16_bn
import numpy as np

class SelfEmbedding(nn.Embedding):
    def __init__():
        super(SelfEmbedding, self).__init__()
    
    def similarity(embed_vect):
        if len(embed_vect.size()) == 1:
            embed_vect = embed_vect.unsqueeze(0)
        return torch.cdist(embed_vect, self.weight)

class SelfVQA(nn.Module):
    def __init__(self, vocab_size,
                 output_size,
                 embedding_dim,
                 hidden_dim,
                 n_layers,
                 MAXLEN,
                 lstm_drop=0.5,
                 bidirectional=False):
        super(SelfVQA, self).__init__()
        self.vgg = vgg16_bn(pretrained=True, progress=True).features
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.img_linear = nn.Linear(512*5*5, 512)
        self.img_dropout = nn.Dropout(0.5)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid_value = 2 if bidirectional else 1
        self.embedding = nn.SelfEmbedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=lstm_drop,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.txt_linear = nn.Linear(self.bid_value*hidden_dim, 512)
        self.txt_dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1024, output_size)

    def forward(self, input_ids, images):
        imgs_embed = self.avg_pool(self.vgg(images)).view(images.size(0), -1)
        imgs_embed = self.img_dropout(F.relu(self.img_linear(imgs_embed)))
        questions_vects = self.embedding(input_ids)
        packed_embed = nn.utils.rnn.pack_padded_sequence(
            questions_vects, [input_ids.size(1)]*input_ids.size(0), batch_first=True
        )

        h0 = torch.zeros((self.n_layers*self.bid_value,
                          input_ids.size(0),
                          self.hidden_dim)).to(input_ids.device)
        c0 = torch.zeros((self.n_layers*self.bid_value,
                          input_ids.size(0),
                          self.hidden_dim)).to(input_ids.device)

        questions_embed, (hidden, cn) = self.lstm(packed_embed, (h0, c0))
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.txt_dropout(F.relu(self.txt_linear(hidden)))
        embed_vetcs = torch.cat((imgs_embed, hidden), dim=1)
        logits = self.classifier(embed_vetcs)
        log_logits = F.log_softmax(logits)
        return log_logits

class BertForSentiment(nn.Module):

    def __init__(
        self, vocab_file, num_classes,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_dropout_prob=0.1,
        vocab_size=None, contain_cls=True
    ):
        super(BertForSentiment, self).__init__()
        self.config = BertConfig(
            hidden_size=hidden_size,
            num_hiddel_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob
        )
        self.bert = BertModel.from_pretrained(vocab_file)
        if vocab_size:
            self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.contain_cls = contain_cls
        if self.contain_cls:
            self.classifier = nn.Linear(2*hidden_size, num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)
    
    def pool_method(self, inputs, method, dim=0):
        if method == 'max':
            return torch.max(inputs, dim=dim).values
        elif method == 'min':
            return torch.min(inputs, dim=dim).values
        elif method == 'avg':
            return torch.mean(inputs, dim=dim)
        else:
            return lambda x: x

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, method='cls'):
        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False
        ).last_hidden_state
        
        if method == 'cls':
            outputs = outputs[:, 0, :]
        else:
            non_zeros = torch.nonzero(input_ids)
            boundaries = torch.cat(
                (torch.nonzero((non_zeros[:-1, 0] != non_zeros[1:, 0]).int()).view(-1), 
                torch.tensor([-1]).to(input_ids.device)), dim=0
            )
            non_zero_indices = non_zeros[boundaries, 1]+1
            
            pool_list = []
            for i in range(outputs.size(0)):
                pool_embed = self.pool_method(outputs[i, 1:non_zero_indices[i], :], method, 0)
                pool_list.append(pool_embed)
            if self.contain_cls:
                outputs = torch.cat((torch.stack(pool_list, dim=0), outputs[:, 0, :]), dim=1)
            else:
                outputs = torch.stack(pool_list, dim=0)

        pooled_output = self.dropout(outputs)
        logits = self.classifier(pooled_output)
        log_logits = F.log_softmax(logits)
        return log_logits

class BertForRE(nn.Module):

    def __init__(
        self, vocab_file, num_classes,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_dropout_prob=0.1,
        fcc_uotput_size=128,
        vocab_size=None
    ):
        super(BertForRE, self).__init__()
        self.config = BertConfig(
            hidden_size=hidden_size,
            num_hiddel_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob
        )
        self.bert = BertModel.from_pretrained(vocab_file)
        if vocab_size:
            self.bert.resize_token_embeddings(vocab_size)
        self.entity_linear = nn.Linear(2*hidden_size, fcc_uotput_size)
        self.char_linear = nn.Linear(2*hidden_size, fcc_uotput_size)
        self.cls_linear = nn.Linear(hidden_size, fcc_uotput_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(3*fcc_uotput_size, num_classes)

    def forward(self, input_ids, attention_mask, firsts, seconds, token_type_ids=None, labels=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False
        ).last_hidden_state
        
        en_embdedlist = []
        char_embedlist = []
        for i in range(bert_outputs.size(0)):
            max_first = torch.max(bert_outputs[i, firsts[i, 0]+2:firsts[i, 1]+1, :], dim=0).values
            max_second = torch.max(bert_outputs[i, seconds[i, 0]+2:seconds[i, 1]+1, :], dim=0).values
            temp_en_embeds = torch.cat([max_first, max_second], dim=0)
            temp_char_embeds = torch.cat([
                bert_outputs[i, firsts[i, 0], :],
                bert_outputs[i, seconds[i, 0], :]
            ], dim=0)
            en_embdedlist.append(temp_en_embeds)
            char_embedlist.append(temp_char_embeds)
        en_embeds = torch.stack(en_embdedlist, dim=0)
        char_embeds = torch.stack(char_embedlist, dim=0)
        
        en_embeds = self.entity_linear(en_embeds)
        char_embeds = self.char_linear(char_embeds)
        cls_embeds = self.cls_linear(bert_outputs[:, 0, :])
        
        embed_output = torch.cat([cls_embeds, en_embeds, char_embeds], dim=1)
        embed_output = F.relu(embed_output)
        pooled_output = self.dropout(embed_output)
        logits = self.classifier(pooled_output)
        log_logits = F.log_softmax(logits)
        return log_logits