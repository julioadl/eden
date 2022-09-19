import numpy as np

import torch
from transformers import *

class embeddings:
    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer.from_pretrained('bert-base')
        self._model = model.from_pretrained('bert-base', output_hidden_states = True).cuda(0)
        self._model.eval()
        
    def tokenize(self, sentence):
        marked_sentence = '<s> ' + sentence + ' </s>'
        tokenized_text = self._tokenizer.tokenize(marked_sentence)
        return tokenized_text
    
    def get_embeddings(self, sentence):
        tokenized_text = self.tokenize(sentence)
        indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [1]*len(tokenized_text)
        
        #Convert to tensor
        tokens_tensor = torch.tensor([indexed_tokens]).cuda(0)
        segment_tensors = torch.tensor([segment_ids]).cuda(0)
        
        with torch.no_grad():
            encoded_layers = self._model(tokens_tensor, segment_tensors)
            
        return encoded_layers[-1][0:12]
    
    def sentence2vec(self, sentence):
        '''
        Returns concatenated hidden dimensions
        '''
        encoded_layers = self.get_embeddings(sentence)
        token_embeddings = []
        tokenized_text = self.tokenize(sentence)
        #What is is batch? The number of sentences passed
        batch_i = 0
        for token_i in range(len(tokenized_text)):
            hidden_layers = []
            for layer_i in range(len(encoded_layers)):
                vec = encoded_layers[layer_i][batch_i][token_i]
                hidden_layers.append(list(vec.cpu().detach().numpy()))
                
            token_embeddings.append(hidden_layers)
            
        #Concatenate embeddings
        token_vecs_concat = []
        for token in token_embeddings:
            concat_embeddings = np.concatenate(token[-4:], axis=0)
            token_vecs_concat.append(list(concat_embeddings))
            return token_vecs_concat#.ravel().tolist()