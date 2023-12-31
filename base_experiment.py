import logging
logging.basicConfig(level='ERROR')

import numpy as np
import pandas as pd
import os
import torch
import zlib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate


class BaseExperimentArgs:
    def __init__(self,
                 N=100,
                 batch_size=10,
                 seq_len=256, 
                 top_k=40,
                 top_p=1.0, 
                 checkpoint="bigcode/starcoderbase-1b", 
                 device='cpu'):
        self.N = N
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.top_k = top_k
        self.top_p = top_p
        self.checkpoint = checkpoint
        self.model_name = os.path.split(checkpoint)[1]
        self.device = device

class BaseExperiment:
    def __init__(self, filelog, filetable, args : BaseExperimentArgs):
        self.args = args
        self.filelog= filelog
        self.filetable = filetable
        self.model_ppl = f"{self.args.model_name}_PPL"

        df = pd.DataFrame(columns=['text', self.model_ppl, 'ZLIB'])
        df.to_csv(self.filetable, mode='a', index=False)

        self.perplexity = evaluate.load("perplexity", module_type="metric")
                
    def setup(self):
        with open(self.filelog, 'a') as f:
            f.write(f"using device: {self.args.device}\n")
            f.write("Loading Models...\n\n")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.checkpoint,
        )
            
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.checkpoint
        ).to(self.args.device)
        self.model.eval()
        with open(self.filelog, 'a') as f:
            f.write("Model loading is done!\n\n")

        self.samples = []
        self.scores = {
            self.model_ppl: [],
            "ZLIB": []
        }

    def generate_sequences(self, prompts):
        input_len = 1
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.args.device)
        outputs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.args.device),
            attention_mask=inputs['attention_mask'].to(self.args.device),
            max_length=input_len+self.args.seq_len,
            do_sample=True,
            top_k=self.args.top_k,
            top_p=self.args.top_p)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
    def run(self):
        with open(self.filelog, 'a') as f:
            f.write("Start experiment...\n\n")
        num_batches = int(np.ceil(self.args.N / self.args.batch_size))
        
        with tqdm(total=self.args.N) as pbar:
            for i in range(num_batches):
                prompts = ["<|endoftext|>"] * self.args.batch_size
                generated_sequences = self.generate_sequences(prompts)

                ppl = self.perplexity.compute(
                    predictions=generated_sequences,
                    model_id=self.args.checkpoint,
                )['perplexities']
                zlib_entropy = [len(zlib.compress(bytes(seq, 'utf-8'))) for seq in generated_sequences]

                self.scores[self.model_ppl].extend(ppl)
                self.scores["ZLIB"].extend(zlib_entropy)

                data = {
                    'text': generated_sequences,
                    self.model_ppl: ppl,
                    'ZLIB': zlib_entropy
                }
                df = pd.DataFrame(data)
                df.to_csv(self.filetable, mode='a', header=False, index=False)
        
                pbar.update(self.args.batch_size)
                
        with open(self.filelog, 'a') as f:
            f.write("Experiment done!\n\n")