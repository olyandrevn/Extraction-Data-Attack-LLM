<h1 align="center">Training Data Extraction Attack Task</h1>
<p>
</p>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


## Overview

It has been shown that pre-trained large language models (LM) can [memorize chunks of the training data](https://arxiv.org/pdf/2012.07805.pdf).

Your task is to compose a notebook that conducts such a data extraction attack on OSS LLM pre-trained on code (e.g. starcoder 1b, 3b)
 
It should include
* A way to check, for a given chunk of code, how likely it is that it was used for training this particular model
* Compare (plot) likelihoods assigned by a model to a chunks of code to an entropy of  zlib compressor and likelihood associated by another model
* Manually inspect the findings to validate if they actually are part of the training data

  
## Install

```sh
git clone https://github.com/olyandrevn/Extraction-Data-Attack-LLM.git
```
OR

Copy [notebook in Google Colab](https://colab.research.google.com/github/olyandrevn/Extraction-Data-Attack-LLM/blob/main/ExtractingTrainingDataTask.ipynb) and run all cells there (on GPU).

## Usage

```sh
python3 run_base_experiment.py
python3 run_full_experiment.py
```

OR

Copy [notebook in Google Colab](https://colab.research.google.com/github/olyandrevn/Extraction-Data-Attack-LLM/blob/main/ExtractingTrainingDataTask.ipynb) and run all cells there (on GPU).

## Author

👤 **Olga Kolomyttseva**

* Github: [@olyandrevn](https://github.com/olyandrevn)
* LinkedIn: [@olyandrevn](https://linkedin.com/in/olyandrevn)
