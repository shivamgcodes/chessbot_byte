import torch
from chessbot_byte.utils import NUM_ACTIONS
from chessbot_byte.tokenizer import SEQUENCE_LENGTH
class parent_config:
    dtype = torch.float32 #[torch.float16, torch.float32, torch.float64, torch.bfloat16, ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_return_buckets = 128

class data_config(parent_config):

    batch_size = 256 # repo
    shuffle = True
    number_of_files = 3 #max is 2148
    filename = f'/home/shivam/Desktop/chessbot_byte/data/train/action_value-@{number_of_files}_data.bag'
    #num_return_buckets = 128 # in the repo
    miniDataSet = False

    if(miniDataSet):
        count = 100
    
class model_config(parent_config):
    d_model = 64
    dim_feedforward = d_model*4
    nhead = 4
    dropout = 0.0
    activation = 'relu'
    layer_norm_eps = 1
    batch_first= True
    norm_first= False
    bias = False
    num_experts = 8
    num_experts_per_tok = 2

    expert_bias   = True
    expert_dropout_factor  = 'dont_matter'
    expert_activation = 'dont_matter'

    emb_init_scale = 0.02
    NUM_ACTIONS = NUM_ACTIONS
    SEQUENCE_LENGTH=SEQUENCE_LENGTH
    use_sinosoidal  = False
    max_timescale = 'dont_matter' 

    decoder_layernorm = True
    #output_size = 128 #num_return_buckets

    encoder_layers = 4
class train_config(parent_config):

    num_epochs = 10 #'figure tf out'
    max_grad_norm = 1.0
    learning_rate = 1e-4
