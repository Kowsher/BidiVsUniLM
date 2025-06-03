# 🔍 FlowNIB: Understanding Bidirectional Language Models through the Information Bottleneck

This is the implementation for the paper "`How Bidirectionality Helps Language Models Learn Better via Dynamic Bottleneck Estimation`".



<div align="center">

![](https://img.shields.io/github/last-commit/Kowsher/BidiVsUniLM?color=green)
![](https://img.shields.io/github/stars/Kowsher/BidiVsUniLM?color=yellow)
![](https://img.shields.io/github/forks/Kowsher/BidiVsUniLM?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<div align="center">

**[<a href="https://arxiv.org/abs/2506.00859">Paper</a>]**
**[<a href="http://github.com/Kowsher/BidiVsUniLM">Code</a>]**


</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
>
>> 🧰 Feel free to explore, open issues, or contribute!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@article{kowsher2025bidirectionalityhelpslanguagemodels,
      title={How Bidirectionality Helps Language Models Learn Better via Dynamic Bottleneck Estimation}, 
      author={Md Kowsher and Nusrat Jahan Prottasha and Shiyun Xu and Shetu Mohanto and Chen Chen and Niloofar Yousefi and Ozlem Garibay},
      year={2025},
      eprint={2506.00859},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.00859}, 
}
```

## Introduction
Bidirectional language models consistently outperform unidirectional ones on natural language understanding tasks, but the theoretical reasons behind this advantage have remained unclear.

**FlowNIB** provides a principled explanation using the **Information Bottleneck (IB)** framework. We propose a dynamic and scalable method for estimating mutual information during training, overcoming the computational and design limitations of classical IB methods.

## 🌟 Key Contributions

- 📊 **FlowNIB Framework**: A novel approach to estimate mutual information on-the-fly during model training.
- 🧠 **Theoretical Insights**: We prove that bidirectional models retain more mutual information and exhibit higher effective dimensionality.
- 🧰 **Practical Tooling**: FlowNIB enables in-depth analysis of information flow and compression in language models.
- 📈 **Empirical Validation**: Extensive experiments across multiple models and tasks confirm our theoretical claims.

## 📎 What's Inside

- 📁 Code for FlowNIB training and mutual information estimation  
- 📄 Theoretical results and proofs  
- 🧪 Experimental scripts
---

## Requirements
Use python 3.11 from MiniConda
```bash
conda create -n flownib_env python=3.11
conda activate flownib_env
```
To install all dependencies:
```bash
pip install -r requirements.txt
```

## Quick Demos

To get started, follow these simple steps:

1. **Import the necessary modules:**

    ```python
      from transformers import AutoModelForCausalLM, AutoTokenizer
      import RoCoFT
      from modeling import CLMSequenceClassification # For casual language models
      from modeling import MLMSequenceClassification # For masked language models
    ```

2. **Load a pre-trained model and apply PEFT:**

    ```python
      tokenizer = AutoTokenizer.from_pretrained("<model_name>", use_fast=False)
      # Use MLMSequenceClassification for masked language models
      model = CLMSequenceClassification.from_pretrained("<model_name>", num_labels=<number_of_labels>).to('cuda')
      RoCoFT.PEFT(model, method='row', rank=3) 
    ```

3. **For column**

    ```python

    ```
4. **In order to choose row or column using pruning technique**

    ```python

    ```


