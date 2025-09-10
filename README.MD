# CRAVE: LLM-based Conversational Recommendation Agents with Collaborative Verbalized Experience

These codes are associated with the following paper:

>LLM-based Conversational Recommendation Agents with Collaborative Verbalized Experience    
>**Yaochen Zhu**, Harald Steck, Dawen Liang, Yinhan He, Nathan Kallus, Jundong Li    
>EMNLP 2025 Findings 

## Dataset

The Reddit-v2 dataset introduced in the paper can be in the update section of this [repo](https://github.com/yaochenzhu/CRAG)

## How to run the codes:

### 1. Fintune the collaborative retriever network    
```python fintune_stella.py```   
```python inference_stella.py```


### 2. Gain verbalized experiences for the COT agent    
```python exp_gain_cot.py```


### 3. Gain verbalized experiences for the DCA system    
```python exp_gain_dca.py```


### 4. Evaluate COT agent with experience    
```python evaluate_cot.py```


### 5. Evaluate DCA system with experience   
```python evaluate_dca.py```


## 🌟 Citation
If you find this work is helpful to your research, please consider citing our paper:
```
@inproceedings{zhu2025collaborative,
  title={Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems},
  author={Zhu, Yaochen and Steck, Harald and Liang, Dawen and Yinhan, He and Kallus, Nathan and Li, Jundong},
  booktitle={EMNLP Findings},
  year={2025}
}
```
**Thanks for your interest in our work!**
