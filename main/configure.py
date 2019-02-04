train_data_path = "../data/ccg/train.data"
dev_data_path = "../data/ccg/dev.data"
test_data_path = "../data/ccg/test.data"

train_ori_data_path = "../data/ccg/train.stagged"
dev_ori_data_path = "../data/ccg/dev.stagged"
test_ori_data_path = "../data/ccg/test.stagged"

cat_path = "../data/categories"

word_idx_file_path = "../data/word.idx"

word_embedding_dim = 100
supertag_embedding_dim = 100

train_batch_size = 32
dev_batch_size = 64
test_batch_size = 500

l2_lambda = 0.000001
learning_rate = 0.001
epochs = 100
encoder_hidden_dim = 200
num_layers_decode = 1
word_size_max = 1

dropout = 0.0

path_embed_method = "lstm" # cnn or lstm or bi-lstm

unknown_word = "<unk>"
PAD = "<PAD>"
GO = "<GO>"
EOS = "<EOS>"
deal_unknown_words = True

rare_tag = "<rare>"

seq_max_len = 11

decoder_type = "greedy" # greedy, beam
beam_width = 4
attention = True
num_layers = 1 # 1 or 2

# the following are for the graph encoding method
weight_decay = 0.0000
sample_size_per_layer = 4
sample_layer_size = 4
hidden_layer_dim = 100
feature_max_len = 1
feature_encode_type = "uni"
# graph_encode_method = "max-pooling" # "lstm" or "max-pooling"
graph_encode_direction = "bi" # "single" or "bi"
concat = True

encoder = "gated_gcn" # "gated_gcn" "gcn"  "seq"

lstm_in_gcn = "none" # before, after, none
