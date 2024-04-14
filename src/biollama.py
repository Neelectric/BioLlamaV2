### A refactored and cleaned version of biollama.py in BioLlamaV1
### Written by Neel Rajani

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaRMSNorm
from transformers.modeling_utils import load_state_dict
import torch
from time import time
from typing import List
import types
import math
from src.retrieval import load_db


def attach_retriever(self, retriever):
    return

# Custom forward pass that stores current input ids as model attribute for later access
def new_forward(self, *args, **kwargs):
    if "input_ids" in kwargs:
        self.input_ids_biollama = kwargs["input_ids"]
        output = self.old_forward(*args, **kwargs)
    else:
        raise Exception("input_ids or labels not found in kwargs")
    return output

def load_RETRO_weights(self, model_path, RETRO_layer_ids):
    return

def retrieve_neighbours(self):
    if self.biollama.retrieved_chunk_storage == None: # If this is first decoding step, retrieve neighbours and store them
        E_no_continuations = medcpt_FAISS_retrieval(
            H_list_decoded[0:l-1], # we do not retrieve for the last chunk, following RETRO
            db_name="RCT200ktrain",
            retrieval_text_mode="input_segmentation",
            chunk_length=self.biollama.chunk_length,
            verbose=False,
            query_tokenizer=self.biollama.query_tokenizer, # passed as a pre-loaded object to save time
            query_model=self.biollama.query_model, # passed as a pre-loaded object to save time
            rerank_tokenizer=self.biollama.rerank_tokenizer, # passed as a pre-loaded object to save time
            rerank_model=self.biollama.rerank_model, # passed as a pre-loaded object to save time
            top_k=top_k, # normally retrieve top 2, following RETRO
            k=k,
            db_faiss=self.biollama.db_faiss, # passed as a pre-loaded object to save time
            db_json=self.biollama.db_json, # passed as a pre-loaded object to save time
        )    
        self.biollama.retrieved_chunk_storage = E_no_continuations
    elif ((n-31) % 32 == 0) or (len(Hplus_list) > len(self.biollama.retrieved_chunk_storage)): # if this is not first decoding step, but we've generated enough to consider a new chunk, retrieve new neighbours and update store
        E_no_continuations = medcpt_FAISS_retrieval(
            H_list_decoded[0:l-1], # we do not retrieve for the last chunk, following RETRO
            db_name="RCT200ktrain",
            retrieval_text_mode="input_segmentation",
            chunk_length=self.biollama.chunk_length,
            verbose=False,
            query_tokenizer=self.biollama.query_tokenizer, # passed as a pre-loaded object to save time
            query_model=self.biollama.query_model, # passed as a pre-loaded object to save time
            rerank_tokenizer=self.biollama.rerank_tokenizer, # passed as a pre-loaded object to save time
            rerank_model=self.biollama.rerank_model, # passed as a pre-loaded object to save time
            top_k=top_k, # retrieve top 2, following RETRO
            k=k,
            db_faiss=self.biollama.db_faiss, # passed as a pre-loaded object to save time
            db_json=self.biollama.db_json, # passed as a pre-loaded object to save time
        )    
        self.biollama.retrieved_chunk_storage = E_no_continuations
    else: # otherwise, we do not need retrieval (as it would just retrieve the same as we already have stored)
        E_no_continuations = self.biollama.retrieved_chunk_storage
    return

def ca(self, Hplus, E_no_continuations) -> torch.Tensor:
    temp = torch.Tensor([])
    return temp

def cca_attn(self, hidden_states):
    input_ids = self.biollama.model.input_ids_biollama[0]
    print("entered cca_attn!")
    print(f"input_ids have length: {len(input_ids)}")
    n = len(input_ids)
    m = self.biollama.neighbour_length
    l = math.ceil(n/m)
    top_k = 1
    k = 1
    H_list = []
    H_list_decoded = []
    for i in range(l): # Splitting the input_ids into chunks of size m
        if (i+1)*m < n:
            H_chunk = input_ids[i*m : (i+1)*m]
            H_chunk_decoded = self.biollama.tokenizer.decode(H_chunk, skip_special_tokens=True)
            H_list.append(H_chunk)
            H_list_decoded.append(H_chunk_decoded)
        else:
            H_chunk = input_ids[i*m:]
            H_chunk_decoded = self.biollama.tokenizer.decode(H_chunk, skip_special_tokens=True)
            H_list.append(H_chunk)
            H_list_decoded.append(H_chunk_decoded)
    
    Hplus_list = []
    num_spliced_chunks = (n-(m-1)) // m 
    for i in range(m-1, num_spliced_chunks * m, m): # note: this for loop iterates differently than the one above
        Hplus_chunk = hidden_states[:,i:i+m:,:]
        Hplus_list.append(Hplus_chunk)

    E_no_continuations = self.retrieve()
    ca_list = torch.Tensor([])
    for i in range(len(Hplus_list)): # for these spliced chunks in Hplus_list, calculate cross attentions with neighbours
        Hplus_ca = ca(self, Hplus_list[i], E_no_continuations[i])
        if ca_list == torch.Tensor([]):
            ca_list = Hplus_ca
        else:
            ca_list = torch.cat((ca_list, Hplus_ca), dim=1)
    
    # concatenate together, following RETRO
    last_tokens_offset = (m-1) + num_spliced_chunks*m
    prefix_and_ca_tensors = torch.cat((hidden_states[:,0:m-1], ca_list), dim=1)
    output = torch.cat((prefix_and_ca_tensors, hidden_states[:,last_tokens_offset:]), dim=1)
    return output


# Custom forward pass for RETRO layers, adapts the HF transformers implementaiton with insights from intermediate decoding blogpost
def RETRO_layer_forward(self, *args, **kwargs):
    hidden_states = args[0]
    attention_mask = kwargs["attention_mask"]  # should be torch.FloatTensor with  `(batch_size, 1,query_sequence_length, key_sequence_length)`
    position_ids = kwargs["position_ids"]
    past_key_value = kwargs["past_key_value"]
    output_attentions = kwargs["output_attentions"]
    use_cache = kwargs["use_cache"]
    input_ids = self.biollama.model.input_ids_biollama

    # Self-Attention (with RMSNorm and residual connection)
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states = hidden_states,
        attention_mask = attention_mask,
        position_ids = position_ids,
        past_key_value = past_key_value,
        output_attentions = output_attentions,
        use_cache = use_cache
    )
    # if (hidden_states.device != residual.device): residual = residual.to(hidden_states.device)
    hidden_states = hidden_states + residual

    # Chunked Cross-Attention (with RMSNorm and residual connection)
    residual = hidden_states
    hidden_states = self.pre_cca_layernorm(hidden_states)
    hidden_states = self.cca_attn(hidden_states = hidden_states)
    hidden_states = hidden_states + residual

    # Multi-Layer Perceptrion (with RMSNorm and residual connection)
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = hidden_states + residual
    outputs = (hidden_states,)
    return outputs

# Adds RMSNorm and LlamaSdpaAttention modules to given layer
def RETRO_fit_layer(layer, layer_id, biollama, torch_dtype):
    config = biollama.model.config
    layer.biollama = biollama
    layer.cca_attn = LlamaSdpaAttention(config = config, layer_idx = layer_id).to(device = biollama.device, dtype = torch_dtype)
    layer.cca_attn.forward = cca_attn.__get__(layer.cca_attn)
    layer.retrieve = retrieve_neighbours.__get__(layer)
    layer.cca_attn.biollama = biollama
    layer.pre_cca_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device = biollama.device, dtype = torch_dtype) # Still need to ensure if this actually trains?
    layer.forward = RETRO_layer_forward.__get__(layer)
    return

# Switches specified decoder layers to be a RETRO layer
def RETRO_fit(biollama, RETRO_layer_ids, torch_dtype):
    for id, layer in enumerate(biollama.model.model.layers):
        if id in RETRO_layer_ids:
            RETRO_fit_layer(layer, id, biollama, torch_dtype) 
    return

class BioLlama():
    """
    The fundamental BioLlama class that holds the modified Llama-2 transformers object and all necessary retrieval components
    """
    def __init__(
            self, 
            model_path: str,
            torch_dtype: torch.dtype,
            RETRO_layer_ids: List[int],
            training: bool,
            retriever_name: str,
            db_name: str,
            neighbour_length: int,
            ):
        
        # Model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = "../hf_cache/")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                          cache_dir = "../hf_cache/", 
                                                        #   device_map = "auto", 
                                                          device_map = "cuda:0",
                                                          torch_dtype = torch_dtype)
        self.model.generation_config.use_cache = False # Editing the generation config to use
        self.model.generation_config.do_sample = False # GenerationMode.GREEDY_SEARCH in the hopes
        self.model.generation_config.top_k = None # that this makes generation deterministic

        # Add RETRO modules and load respective weights if not training
        RETRO_fit(self, RETRO_layer_ids = RETRO_layer_ids, torch_dtype = torch_dtype)
        if not training: load_RETRO_weights(self, model_path = model_path, RETRO_layer_ids = RETRO_layer_ids)

        # save original .forward() call and then replace .forward() with new customized implementation
        self.model.old_forward = self.model.forward
        self.model.forward = new_forward.__get__(self.model)

        # Prepare all retrieval components, ie. retriever, retrieval corpus and retrieved chunk storage
        attach_retriever(self, retriever = retriever_name)
        self.db_faiss, self.db_json = load_db(db_name = db_name, 
                                              retriever_name = retriever_name, 
                                              neighbour_length = neighbour_length)
        self.neighbour_length = neighbour_length
        self.neighbour_storage = None
        return
    # encode the queries (use the [CLS] last hidden states as the representations)
    
    def generate(self, prompt, max_new_tokens = 50):
        padding = False # keeping this for now, may need to change for batch generation
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=padding)
        self.model.input_ids_biollama = inputs["input_ids"] # Storing the input_ids to access them again later
        self.model.prompt_biollama = prompt # Same for pure prompt
        self.model.generation_config.max_new_tokens = max_new_tokens
        time_before = time()
        generated_tokens = self.model.generate(inputs.input_ids.to(self.device), 
                                               **self.model.generation_config.to_dict())
        time_after = time()
        time_taken = time_after - time_before
        num_new_tokens = len(generated_tokens[0]) - len(inputs.input_ids[0])
        output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,)[0]
        return (output, num_new_tokens, time_taken)
    
    

