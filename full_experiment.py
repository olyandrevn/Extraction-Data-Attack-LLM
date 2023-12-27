import logging
logging.basicConfig(level='ERROR')

import pandas as pd
import os
import torch
import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FullExperimentArgs:
    def __init__(self,
                 base_filetable,
                 checkpoint="bigcode/starcoderbase-3b"):
        self.base_filetable = base_filetable
        self.checkpoint = checkpoint
        self.model_name = os.path.split(checkpoint)[1]

class FullExperiment:
    def __init__(self, filelog, filetable, args : FullExperimentArgs):
        self.args = args
        self.filelog = filelog
        self.filetable = filetable
        self.model_ppl = f"{self.args.model_name}_PPL"

        self.perplexity = evaluate.load("perplexity", module_type="metric")
                
    def setup(self):
        with open(self.filelog, 'a') as f:
            f.write(f"using device: {device}\n")
        
    def run(self):
        df = pd.read_csv(self.args.base_filetable)
        generated_sequences = df['text'].tolist()

        with open(self.filelog, 'a') as f:
            f.write("Start perplexity calculation...\n\n")
        ppl = self.perplexity.compute(
            predictions=generated_sequences,
            model_id=self.args.checkpoint,
            batch_size=8,
        )['perplexities']
        with open(self.filelog, 'a') as f:
            f.write("Perplexity calculated!\n\n")

        df[self.model_ppl] = ppl
        df.to_csv(self.filetable, mode='w', index=False)
        with open(self.filelog, 'a') as f:
            f.write("Table saved!\n\n")
