### A refactored and cleaned version of biollama.py in BioLlamaV1
### Written by Neel Rajani

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaRMSNorm, apply_rotary_pos_emb, repeat_kv
from transformers.modeling_utils import load_state_dict
import torch
from time import time
from typing import List
import types
import math
from src.retrieval import load_db, retrieve


def attach_retriever(self, retriever):
    if retriever == "medcpt":
        query_model = "ncbi/MedCPT-Query-Encoder"
        rerank_model = "ncbi/MedCPT-Cross-Encoder"
    self.query_tokenizer = AutoTokenizer.from_pretrained(query_model)
    self.query_model = AutoModel.from_pretrained(query_model, device_map = "cuda:0")
    self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model)
    self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model, device_map = "cuda:0")
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

def retrieve_neighbours(self, queries, top_k, k):
    biollama = self.biollama
    E_no_continuations = retrieve(queries = queries,
            neighbour_length = 32,
            query_tokenizer = biollama.query_tokenizer,
            query_model = biollama.query_model,
            rerank_tokenizer = biollama.rerank_tokenizer,
            rerank_model = biollama.rerank_model,
            top_k = 1,
            k = 5,
            db_faiss = biollama.db_faiss, 
            db_json = biollama.db_json)
    biollama.neighbour_storage = E_no_continuations
    return E_no_continuations

def ca(self, hidden_states, e) -> torch.Tensor:
    # Tokenize and embed the neighbour E. Could be packaged into a function?
    if type(e) == list: e = e[0] 
    self.biollama.tokenizer.padding_side = 'left'  # This line and the following can be necessary to prevent padding bugs
    self.biollama.tokenizer.pad_token = self.biollama.tokenizer.eos_token
    e_encoded = self.biollama.tokenizer(e, return_tensors="pt", max_length=32, padding="max_length").to(self.q_proj.weight.device) # If E comes out shorter than 32 we pad (though it shouldn't)
    e_input_ids = e_encoded.input_ids
    e_input_ids = e_input_ids[:,0:32] # in case e is longer than 32 tokens, we truncate
    embed_tokens = self.biollama.model.base_model.embed_tokens
    e_encoded_and_embedded = embed_tokens(e_input_ids)
    # if e_encoded_and_embedded.device != self.q_proj.weight.device: # sometimes it complains about tensors not being on same device
    #     e_encoded_and_embedded = e_encoded_and_embedded.to(self.q_proj.weight.device)

    # Perform cross-attention, with queries from hidden_states and keys/values from neighbours
    bsz, q_len, _ = hidden_states.size()
    position_ids = torch.arange(hidden_states.shape[-2], dtype=torch.long, device=hidden_states.device).unsqueeze(0)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(e_encoded_and_embedded)
    value_states = self.v_proj(e_encoded_and_embedded)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Finally, perform scaled dot-product attention and return
    attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and q_len > 1,)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output

def cca_attn(self, hidden_states):
    input_ids = self.biollama.model.input_ids_biollama[0]
    # print("entered cca_attn!")
    # print(f"input_ids have length: {len(input_ids)}")
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
    E_no_continuations = self.biollama.neighbour_storage
    if (E_no_continuations == None) or (n-31 % 32 == 0) or (len(Hplus_list) > len(E_no_continuations)):
        E_no_continuations = self.biollama.retrieve(queries = H_list_decoded[0:l-1],
                                        top_k = top_k,
                                        k = k)
    ca_list = torch.Tensor([])
    for i in range(len(Hplus_list)): # for these spliced chunks in Hplus_list, calculate cross attentions with neighbours
        Hplus_ca = ca(self, Hplus_list[i], E_no_continuations[i])
        if ca_list.shape == torch.Size([0]):
            ca_list = Hplus_ca
        else:
            ca_list = torch.cat((ca_list, Hplus_ca), dim=1)
    
    # concatenate together, following RETRO
    prefix_offset = m-1
    suffix_offset = (m-1) + num_spliced_chunks*m
    prefix = hidden_states[:,0:prefix_offset]
    suffix = hidden_states[:,suffix_offset:]
    output = torch.cat((prefix, ca_list, suffix), dim=1)
    return output


# Custom forward pass for RETRO layers, adapts the HF transformers implementaiton with insights from intermediate decoding blogpost
def RETRO_layer_forward(self, *args, **kwargs):
    hidden_states = args[0]
    attention_mask = kwargs["attention_mask"]  # should be torch.FloatTensor with  `(batch_size, 1,query_sequence_length, key_sequence_length)`
    position_ids = kwargs["position_ids"]
    past_key_value = kwargs["past_key_value"]
    output_attentions = kwargs["output_attentions"]
    use_cache = kwargs["use_cache"]

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
    biollama.retrieve = retrieve_neighbours.__get__(layer)
    layer.cca_attn.biollama = biollama
    layer.pre_cca_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device = biollama.device, dtype = torch_dtype) # Still need to ensure if this actually trains?
    layer.forward = RETRO_layer_forward.__get__(layer)
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

        # Add RETRO modules to specified layers
        for id, layer in enumerate(self.model.model.layers):
            if id in RETRO_layer_ids:
                RETRO_fit_layer(layer, id, self, torch_dtype) 

        # Load respective weights if not training
        if not training: load_RETRO_weights(self, model_path = model_path, RETRO_layer_ids = RETRO_layer_ids)

        # Save original .forward() call and then replace .forward() with new customized implementation
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
    
    

