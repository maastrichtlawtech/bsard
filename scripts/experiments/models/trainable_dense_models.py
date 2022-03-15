import os
import json
from tqdm import tqdm
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type, Union, Optional, Set

import math
import numpy as np
import torch
from torch import nn, Tensor as T
from transformers import AutoTokenizer, AutoModel



class CSequential(nn.Sequential):
    """C(ustom)Sequential is an inherited class from nn.Sequential capable of handling multiple inputs. 
    See https://github.com/pytorch/pytorch/issues/19808 for more details.
    """
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class BiEncoder(nn.Module):
    def __init__(self, 
                 is_siamese: bool, #whether to train a Siamese (query and document are projected into the same embedding space) or a Dual-tower (query and document are projected into different embedding spaces) BiEncoder model.
                 q_model_name_or_path: str, #model name (Hugging Face) or path for the query encoder.
                 d_model_name_or_path: Optional[str] = None, #model name (Hugging Face) or path for the document encoder.
                 truncation: bool = False,  #whether to truncate the inputs to a length of 'chunk_size'.
                 max_input_len: Optional[int] = None, #size of the maximum sequence size when truncation is set to True.
                 chunk_size: Optional[int] = None, #size of the truncated input if 'truncation' is True OR size of the split chunks if 'truncation' set to False.
                 window_size: int = 10, #size of the overlapping window between the chunks (only used when 'truncation' is False).
                 pooling_mode: str = 'cls', #strategy to extract a fixed-sized passage embedding out of a variable number of word embeddings (default is set to cls; can also be mean or max).
                 score_fn: str = 'dot', #scoring function between passage embeddings (efault is set to dot product; can also be set to cosine similarity).
                 scale: float = 20.0 #output of scoring function is multiplied by scale value.
        ):
        super(BiEncoder, self).__init__()
        assert score_fn in ['cos', 'dot'], f"Unknown scoring function: {score_fn}"
        self.config = self._get_config_dict(locals())
        self.is_siamese = is_siamese
        self.score_fn = score_fn
        self.scale = scale
        self.q_encoder = PassageEncoder(model_name_or_path=q_model_name_or_path, 
                                        truncation=truncation,
                                        max_input_len=max_input_len,
                                        chunk_size=chunk_size,
                                        window_size=window_size,
                                        pooling_mode=pooling_mode)
        if is_siamese:
            self.d_encoder = self.q_encoder
        else:
            d_model_name_or_path = d_model_name_or_path if d_model_name_or_path is not None else q_model_name_or_path
            self.d_encoder = PassageEncoder(model_name_or_path=d_model_name_or_path, 
                                            truncation=truncation,
                                            max_input_len=max_input_len,
                                            chunk_size=chunk_size,
                                            window_size=window_size,
                                            pooling_mode=pooling_mode)

    def forward(self, q_input_ids: T, q_attention_masks: T, d_input_ids: T, d_attention_masks: T) -> Tuple[T, T]:
        q_pooled_out = self.q_encoder(q_input_ids, q_attention_masks)
        d_pooled_out = self.d_encoder(d_input_ids, d_attention_masks)
        scores = self.get_scores(q_pooled_out, d_pooled_out) * self.scale
        return scores

    def collate_batch(self, batch: List[Tuple[str, str]]) -> Tuple[T, T, T, T]:
        """The collate_fn function gets called with a list of return values from your Datasest.__getitem__(), and should return stacked tensors.
        In this case, the batch is a list of tuples: [(query, pos_document), ...].
        """
        queries, documents = map(list, zip(*batch)) # Unzip a list of tuples into individual lists: [query, query, ...], [document, document, ...]
        q_input_ids, q_attention_masks = self.q_encoder.encoder.word_encoder.tokenize(queries)
        d_input_ids, d_attention_masks = self.d_encoder.encoder.word_encoder.tokenize(documents)
        return {'q_input_ids':q_input_ids, 'q_attention_masks':q_attention_masks, 'd_input_ids':d_input_ids, 'd_attention_masks':d_attention_masks}

    def get_scores(self, a: T, b: T) -> T:
        if self.score_fn == 'cos':
            a = torch.nn.functional.normalize(a, p=2, dim=1)
            b = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a, b.transpose(0, 1))
    
    def save(self, output_path: str):
        if self.is_siamese:
            # If the model is a Siamese BiEncoder, save only the query encoder.
            self.q_encoder.save(os.path.join(output_path, 'p_encoder'))
        else:
            # If the model is a Dual-tower BiEncoder, save each encoder individually.
            for _, encoder_name in enumerate(self._modules):
                encoder = self._modules[encoder_name]
                encoder.save(os.path.join(output_path, encoder_name))

        # Save the global model configuration.
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f'{type(self).__name__}_config.json'), 'w') as fOut:
            json.dump(self.config, fOut, indent=2)
    
    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'BiEncoder_config.json')) as fIn:
            config = json.load(fIn)
            del config['q_start_checkpoint']
            del config['d_start_checkpoint']
        if config['is_siamese'] == True:
            return BiEncoder(q_model_name_or_path=os.path.join(input_path, 'p_encoder', 'WordEncoder'), **config)
        else:
            return BiEncoder(q_model_name_or_path=os.path.join(input_path, 'q_encoder', 'WordEncoder'),
                             d_model_name_or_path=os.path.join(input_path, 'd_encoder', 'WordEncoder'), **config)

    def _get_config_dict(self, items):
        del items['self']
        del items['__class__']
        items['q_start_checkpoint'] = items.pop('q_model_name_or_path')
        items['d_start_checkpoint'] = items.pop('d_model_name_or_path')
        return items


class PassageEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, truncation: bool, max_input_len: Optional[int], chunk_size: Optional[int], window_size: int, pooling_mode: str):
        super(PassageEncoder, self).__init__()
        self.config = self._get_config_dict(locals())
        self.encoder = CSequential(OrderedDict([
          ('word_encoder', WordEncoder(model_name_or_path, truncation, max_input_len, chunk_size, window_size)),
          ('word_pooler', WordPooler(pooling_mode))
        ]))

    def forward(self, input_ids: T, attention_masks: T):
        return self.encoder(input_ids, attention_masks)

    def encode(self, texts: Union[str, List[str]], device: str, batch_size: int, show_progress_bar: bool = False):
        self.eval()
        self.to(device)
        if isinstance(texts, str):
            texts = [texts]
        
        # Sort texts by length to get batches of similar lengths.
        length_sorted_idx = np.argsort([-len(t) for t in texts])
        texts_sorted = [texts[idx] for idx in length_sorted_idx]

        all_embeddings = []
        for start_idx in tqdm(range(0, len(texts), batch_size), desc="Docs:", disable=not show_progress_bar):
            texts_batch = texts_sorted[start_idx:start_idx+batch_size]

            # Tokenize.
            input_ids, attention_masks = self.encoder.word_encoder.tokenize(texts_batch)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)

            # Encode.
            with torch.no_grad():
                embeddings = self.forward(input_ids, attention_masks)
                embeddings = embeddings.detach().cpu()
                all_embeddings.extend(embeddings)

        # Sort the embeddings back in the original order of the input texts and convert to output tensor.
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def save(self, output_path: str):
        # Save the model configuration.
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f'{type(self).__name__}_config.json'), 'w') as fOut:
            json.dump(self.config, fOut, indent=2)

        # Save the model state.
        torch.save(self.state_dict(), os.path.join(output_path, f'{type(self).__name__}_model_state.bin'))
        
        # Also save each module of the encoder individually.
        for idx, module in enumerate(self.encoder):
            module_path = os.path.join(output_path, type(module).__name__)
            os.makedirs(module_path, exist_ok=True)
            module.save(module_path)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'PassageEncoder_config.json')) as fIn:
            config = json.load(fIn)
            del config['model_start_checkpoint']
        return PassageEncoder(model_name_or_path=os.path.join(input_path, 'WordEncoder'), **config)

    def _get_config_dict(self, items):
        del items['self']
        del items['__class__']
        items['model_start_checkpoint'] = items.pop('model_name_or_path')
        return items


class WordEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, truncation: bool, max_input_len: Optional[int], chunk_size: Optional[int], window_size: int): 
        super(WordEncoder, self).__init__()
        self.config = self._get_config_dict(locals())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.truncation = truncation
        self.max_input_len = max_input_len
        if chunk_size is None:
            chunk_size = min(self.encoder.config.max_position_embeddings, self.tokenizer.model_max_length)
        self.chunk_size = chunk_size
        self.window_size = window_size

    def tokenize(self, texts: List[str]) -> Tuple[T, T]:
        """
        Args:
            texts: a list of queries/questions or passages/documents to tokenize.
        Returns:
            input_ids: 3D tensor of size [batch_size x chunk_size x splits_size].
            attention masks: 3D tensor of size [batch_size x chunk_size x splits_size].
        """
        tokenized = self.tokenizer(texts, padding=True, truncation=self.truncation, max_length=self.max_input_len, return_tensors="pt") #Truncation to 'max_length' AND padding to max sequence in batch (see https://huggingface.co/transformers/preprocessing.html)
        input_ids, attention_masks =  self._transform_inputs(tokenized ['input_ids'], tokenized ['attention_mask'])
        return input_ids, attention_masks

    def forward(self, *inputs: Tuple[T, T]) -> Tuple[T, T, T]:
        """
        Args:
            input_ids: 3D tensor of size [batch_size x chunk_size x splits_size].
            attention masks: 3D tensor of size [batch_size x chunk_size x splits_size].
        Returns:
            token_embeddings: 4D tensor of size [batch_size x chunk_size x splits_size x dim].
            attention masks: 3D tensor of size [batch_size x chunk_size x splits_size].
        """
        input_ids, attention_masks = inputs
        token_embeddings = []
        for i in range(input_ids.shape[2]):
            out = self.encoder(input_ids=input_ids[:,:,i], 
                               attention_mask=attention_masks[:,:,i], 
                               return_dict=False)
            token_embeddings.append(out[0])
        token_embeddings = torch.stack(token_embeddings, dim=2)
        return token_embeddings, attention_masks

    def _transform_inputs(self, input_ids: T, attention_masks: T) -> Tuple[T,T]:
        """Transform the tokenized inputs returned by HF tokenizer.
        Args:
            input_ids: 2D tensor of size [batch_size x batch_max_seq_len].
            attention_masks: 2D tensor of size [batch_size x batch_max_seq_len].
        Returns:
            input_ids: 3D tensor of size [batch_size x chunk_size x splits_size].
            attention masks: 3D tensor of size [batch_size x chunk_size x splits_size].
        """
        batch_size, batch_max_seq_len = input_ids.shape

        if batch_max_seq_len <= self.chunk_size:
            # If the max sequence length from the current batch is smaller than the defined chunk size, simply return the tensors with a dimension of size one as the z-axis.
            return input_ids.unsqueeze(2), attention_masks.unsqueeze(2)
        else:
            # Remove first column from 2D tensor (corresponding to the CLS tokens of the long sequences).
            input_ids = input_ids[:, 1:] #T[batch_size x batch_max_seq_len-1]
            attention_masks = attention_masks[:, 1:] #T[batch_size x batch_max_seq_len-1]
            batch_max_seq_len -= 1
            chunk_size  = self.chunk_size - 1

            # Pad 2D tensor so that the 'batch_seq_len' is a multiple of 'chunk_size' (otherwise unfold ignore remaining tokens).
            num_windows = math.floor((batch_max_seq_len - chunk_size)/(chunk_size - self.window_size))
            num_repeated_tokens = num_windows * self.window_size
            batch_seq_len = math.ceil((batch_max_seq_len + num_repeated_tokens)/chunk_size) * chunk_size
            input_ids = torch.nn.functional.pad(input=input_ids, pad=(0, batch_seq_len - batch_max_seq_len), mode='constant', value=self.tokenizer.pad_token_id)
            attention_masks = torch.nn.functional.pad(input=attention_masks, pad=(0, batch_seq_len - batch_max_seq_len), mode='constant', value=0)

            # Split tensor along y-axis (i.e., along the 'batch_max_seq_len' dimension) with overlapping of 'window_size'
            # and create a new 3D tensor of size [batch_size x chunk_size-1 x batch_max_seq_len/chunk_size].
            input_ids = input_ids.unfold(dimension=1, size=chunk_size, step=chunk_size-self.window_size).permute(0,2,1)
            attention_masks = attention_masks.unfold(dimension=1, size=chunk_size, step=chunk_size-self.window_size).permute(0,2,1)
            splits_size = input_ids.size(2)

            # Add CLS token ids (with attention masks of 1) in the beginning of all split sequences.
            cls_input_ids = torch.full((batch_size, 1, splits_size), self.tokenizer.cls_token_id)
            input_ids = torch.cat([cls_input_ids, input_ids], axis=1)

            # Add attention masks for the new CLS token ids (set to 1 only if the corresponding sequence is not PAD tokens only).
            cls_attention_masks = torch.zeros(batch_size, 1, splits_size)
            cls_attention_masks[input_ids[:,1,:].unsqueeze(1) != self.tokenizer.pad_token_id] = 1
            attention_masks = torch.cat([cls_attention_masks, attention_masks], axis=1)
            return input_ids, attention_masks

    def save(self, output_path: str):
        with open(os.path.join(output_path, f'{type(self).__name__}_config.json'), 'w') as fOut:
            json.dump(self.config, fOut, indent=2)
        self.encoder.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    def _get_config_dict(self, items):
        del items['self']
        del items['__class__']
        items['model_start_checkpoint'] = items.pop('model_name_or_path')
        return items


class WordPooler(nn.Module):
    def __init__(self, pooling_mode: str):
        super(WordPooler, self).__init__()
        assert pooling_mode in ['mean', 'max', 'cls'], f"Unknown pooling mode: {pooling_mode}"
        self.pooling_mode = pooling_mode

    def forward(self, *inputs: Tuple[T, T]) -> T:
        """
        Args:
            token_embeddings: 4D tensor of size [batch_size x chunk_size x splits_size x embedding_dim]
            attention_masks: 3D tensor of size [batch_size x chunk_size x splits_size].
        Returns:
            passage_embeddings: 2D tensor of size [batch_size x embedding_dim]
        """
        token_embeddings, attention_masks = inputs
        batch_size, chunk_size, splits_size, embedding_dim = token_embeddings.shape
        if self.pooling_mode == 'cls':
            # Distill a global representation of the passage by averaging the CLS token embeddings of the split sequences.
            return self.pool(token_embeddings=token_embeddings[:,0,:,:],  #get all CLS token embeddings in the batch.
                        attention_masks=attention_masks[:,0,:],  #get the attention masks of all CLS tokens in the batch.
                        pooling_mode='mean')
        else:
            # Distill a global representation of the passage by mean/max pooling over all token embeddings (excluding the CLS tokens) of a given sequence.
            return self.pool(token_embeddings=token_embeddings[:,1:,:,:].permute(0,2,1,3).reshape(batch_size, -1, embedding_dim), #exclude all CLS token embeddings and reshape the batch tensor into a 3D matrice of size [batch_size x batch_max_seq_len x dim].
                        attention_masks=attention_masks[:,1:,:].permute(0,2,1).reshape(batch_size, -1), #exclude attention masks corresponding to the CLS token embeddings and reshape the batch tensor into a 2D matrice of size [batch_size x batch_max_seq_len].
                        pooling_mode=self.pooling_mode)

    def pool(self, token_embeddings: T, attention_masks: T, pooling_mode: str = 'mean') -> T:
        """
        Args:
            token_embeddings: 3D tensor of size [batch_size x seq_len x embedding_dim].
            attention_masks: 2D tensor of size [batch_size x seq_len].
            pooling_mode: type of pooling to perform among ['mean', 'max'].
        Returns:
            passage_vectors: 2D tensor of size [batch_size x embedding_dim].
        """
        if pooling_mode == 'max':
            # Set all values of the [PAD] embeddings to large negative values (so that they are never considered as maximum for a channel).
            attention_masks_expanded = attention_masks.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[attention_masks_expanded == 0] = -1e9
            # Compute the maxima along the 'seq_length' dimension (-> Tensor[batch_size, embedding_dim]).
            passage_vectors = torch.max(token_embeddings, dim=1).values
        elif pooling_mode == 'mean':
            # Set all values of the [PAD] embeddings to zeros (so that they are not taken into account in the sum for a channel).
            attention_masks_expanded = attention_masks.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[attention_masks_expanded == 0] = 0.0
            # Compute the means by first summing along the 'seq_length' dimension (-> Tensor[batch_size, embedding_dim]).
            sum_embeddings = torch.sum(token_embeddings, dim=1)
            # Then, divide all values of a passage vector by the original passage length.
            sum_mask = attention_masks_expanded.sum(dim=1) # -> Tensor[batch_size, embedding_dim] where each value is the length of the corresponding passage.
            sum_mask = torch.clamp(sum_mask, min=1e-9) # Make sure not to have zeros by lower bounding all elements to 1e-9.
            passage_vectors = sum_embeddings / sum_mask # Divide each dimension by the sequence length.
        return passage_vectors

    def save(self, output_path: str):
        config = {'pooling_mode': self.pooling_mode}
        with open(os.path.join(output_path, f'{type(self).__name__}_config.json'), 'w') as fOut:
            json.dump(config, fOut, indent=2)
