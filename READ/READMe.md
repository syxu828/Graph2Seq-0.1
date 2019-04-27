# Graph2Seq
Graph2Seq is a simple code for building a graph-encoder and sequence-decoder for NLP and other AI/ML/DL tasks. 

# How To Run The Codes
To train your graph-to-sequence model, you need:

(1) Prepare your train/dev/test data which the form of:

    each line is a json object whose keys are "seq", "g_ids", "g_id_features", "g_adj":
    "seq" is a text which is supposed to be the output of the decoder
    "g_ids" is a mapping from the node ID to its ID in the graph
    "g_id_features" is a mapping from the node ID to its text features
    "g_adj" is a mapping from the node ID to its adjacent nodes (represented as thier IDs)

    See data/no_cycle/train.data as examples.


(2) Modify some hyper-parameters according to your task in the main/configure.py

(3) Train the model by running the following code
    "python run_model.py train -sample_size_per_layer=100 -sample_layer_size=5 -hidden_layer_dim=50 -epochs=300"
    The model that performs the best on the dev data will be saved in the dir "saved_model"

(4) Test the model by running the following code
    "python run_model.py test -sample_size_per_layer=100 -sample_layer_size=5 -hidden_layer_dim=50 -epochs=300"
    The prediction result will be saved in saved_model/prediction.txt

(5) You can run the pretrained mode on SDP-100 task by directly running the test code (see step 4). The test acc is 97.7.

# Env Setting
(1) Python 3.5
(2) Tensorflow 1.8.0

# How To Cite The Codes
Please cite our work if you like or are using our codes for your projects!

Kun Xu, Lingfei Wu, Zhiguo Wang, Yansong Feng, Michael Witbrock, and Vadim Sheinin (first and second authors contributed equally), "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks", arXiv preprint arXiv:1804.00823.

@article{xu2018graph2seq, <br/>
  title={Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks}, <br/>
  author={Xu, Kun and Wu, Lingfei and Wang, Zhiguo and Feng, Yansong and Witbrock, Michael and Sheinin, Vadim}, <br/>
  journal={arXiv preprint arXiv:1804.00823}, <br/>
  year={2018} <br/>
} <br/>

------------------------------------------------------
Contributors: Kun Xu, Lingfei Wu <br/>

