import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model.base_model import Base_Model
from model.losses import NtXentLoss
import numpy as np 
# from utils.torch_helper import move_to_device, squeeze_dim

class PQ_head(nn.Module):
    """
    The pq implementaion refers to SPQ(https://github.com/youngkyunJang/SPQ).
    """
    def __init__(self, N_words, N_books, L_word, gumbel_temp, dist_metric, sample_method = "gumbel_softmax"):
        # N_words = K; number of codewords
        # N_books = M; number of codebooks
        # L_word = Dimension of the codebwords

        super(PQ_head, self).__init__()

        self.C = nn.Parameter(torch.randn(N_words, N_books * L_word))

        self.N_books = N_books
        self.L_word = L_word
        self.gumbel_temp = gumbel_temp
        self.dist_metric = dist_metric

        self.sample_method = sample_method

    def sample_gumbel(self, shape, eps = 1e-20):
        U = torch.rand(shape)
        U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self,logits, temperature = 1):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim = -1)
    
    def gumbel_softmax(self, logits, temperature = 1, hard = True):
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y
        raise NotImplementedError()


    def defined_sim(self, x, c):
        if self.dist_metric == "euclidean":
            diff = x.unsqueeze(1) - c.unsqueeze(0)
            return  - torch.sum(diff * diff, - 1)
        else:
            raise NotImplementedError



    def forward(self, X):
        # input shape: (bsz, N_books * L_word)
        x = torch.split(X, self.L_word, dim = 1) # tuple; ele of the tuple has the shape like (bath_size, L_word)。 N_books element
        c = torch.split(self.C, self.L_word, dim = 1) # tuple; ele of the tuple has the shape like (N_words, L_word)。 N_books elements.
        prob_list = []
        for i in range(self.N_books):
            logits = self.defined_sim(x[i], c[i])
            
            if self.sample_method == "softmax":
                soft_prob = F.softmax(logits / self.gumbel_temp, dim = 1)
                prob_list.append(soft_prob)
            elif self.sample_method == "gumbel_softmax":
                logits = F.softmax(logits, dim = 1)
                logits = torch.log(logits + 1e-9)
                soft_prob = self.gumbel_softmax(logits, self.gumbel_temp, False)
                prob_list.append(soft_prob)

            if i == 0:
                Q = torch.mm(soft_prob, c[i])
            else:
                Q = torch.cat((Q, torch.mm(soft_prob, c[i])), dim = 1)
        return Q, prob_list 


class MICPQ(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def define_parameters(self):
        bert_config = AutoConfig.from_pretrained('bert-base-uncased')
        bert_config.attention_probs_dropout_prob = self.hparams.dropout_rate 
        bert_config.hidden_dropout_prob = self.hparams.dropout_rate
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased', config = bert_config) 
        for param in self.bert_model.parameters():
            param.requires_grad = False 
        
        assert self.hparams.encode_length == self.hparams.N_books * int(np.log2(self.hparams.N_words))
        
        self.pq_head = PQ_head(N_words = self.hparams.N_words, N_books = self.hparams.N_books, 
                                L_word = self.hparams.L_word, gumbel_temp = self.hparams.gumbel_temperature,
                                dist_metric = self.hparams.dist_metric, sample_method = self.hparams.sample_method)

        self.pro_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, int(self.hparams.L_word * self.hparams.N_books)),
        )
        
        self.criterion = NtXentLoss(self.hparams.temperature)
        
    def get_embeddings(self, inputs, pooling="mean"):
        bert_output = self.bert_model(**inputs)[0] # (batch_size, sequence_length, hidden_size) Sequence of hidden-states at the output of the last layer of the model.
        
        if pooling == 'mean':
            attention_mask = inputs['attention_mask'].unsqueeze(-1) # (batch_size, sequence_length, 1)
            output = torch.sum(bert_output*attention_mask, dim = 1) / torch.sum(attention_mask, dim = 1)
        elif pooling == 'cls':
            output = bert_output[:,0]
        else:
            raise NotImplementedError()
        return output


    def compute_prob_loss(self, prob0, prob1):
        def get_entropy(probs):
                q_ent = -(probs.mean(0) * torch.log(probs.mean(0) + 1e-12)).sum()
                return q_ent
        
        def get_cond_entropy(probs):
            q_cond_ent = - (probs * torch.log(probs + 1e-12)).sum(1).mean()
            return q_cond_ent
        
        q_ent0 = get_entropy(prob0)
        q_cond_ent0 = get_cond_entropy(prob0)

        q_ent1 = get_entropy(prob1)
        q_cond_ent1 = get_cond_entropy(prob1)

        wim_0 = q_ent0 - self.hparams.cond_ent_weight * q_cond_ent0
        wim_1 = q_ent1 - self.hparams.cond_ent_weight * q_cond_ent1

        wim = (wim_0 + wim_1) / 2.
        prob_loss = - wim
        return prob_loss 
    

    # def encode_quantize(self, inputs):
    #     embd = self.pro_layer(self.get_embeddings(inputs, pooling=self.hparams.pooler_type))
    #     repr_tuple = torch.split(embd, self.hparams.L_word, dim = 1)
    #     Q, prob_list = self.pq_head(embd)
    #     assign_list = []
    #     for prob in prob_list:
    #         assign = torch.argmax(prob, dim = 1)
    #         assign_list.append(assign)

    #     return repr_tuple, assign_list
        

    def forward(self, inputs):
        embd_0 = self.pro_layer(self.get_embeddings(inputs, pooling=self.hparams.pooler_type))
        embd_1 = self.pro_layer(self.get_embeddings(inputs, pooling=self.hparams.pooler_type))

        Q_0, prob0_list = self.pq_head(embd_0)
        Q_1, prob1_list = self.pq_head(embd_1)

        codebooks_losses = []
        for i in range(len(prob0_list)):
            prob0 = prob0_list[i]
            prob1 = prob1_list[i]
            codebooks_losses.append(self.compute_prob_loss(prob0, prob1))
        
        prob_loss = sum(codebooks_losses)
        contra_loss = self.criterion(Q_0, Q_1)

        loss = self.hparams.code_weight * contra_loss + self.hparams.prob_weight * prob_loss

        return {'loss': loss, 'contra_loss': contra_loss, 'prob_loss': prob_loss}


    def encode_continuous(self, target_inputs):
        embd = self.pro_layer(self.get_embeddings(target_inputs, pooling=self.hparams.pooler_type))
        return embd 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr = self.hparams.lr)
        return optimizer


    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument("-t", "--temperature", default = 0.3, type = float,
                            help = "Temperature [%(default)d]",)
        parser.add_argument("--gumbel_temperature", default = 5.0, type = float,
                            help = "gumbel_temperature [%(default)d]",)
        parser.add_argument("--dropout_rate", default = 0.3, type = float,
                            help = "Dropout rate [%(default)d]",)  
        parser.add_argument("--pooler_type", default="cls", type = str)   

        parser.add_argument("--dist_metric", default="euclidean", type = str)
        parser.add_argument("--code_weight", default = 1.0, type = float)
        parser.add_argument("--prob_weight", default = 0.1, type = float)
        parser.add_argument("--cond_ent_weight", default=0.1, type = float)           

        parser.add_argument("--L_word", default=24, type = int) 
        parser.add_argument('--N_books', default = 4, type = int )
        parser.add_argument('--N_words', default=16, type = int  )

        parser.add_argument("--sample_method", default = "gumbel_softmax", type = str)          

        return parser

