# CRAVE: LLM-based Conversational Recommendation Agents with Collaborative Verbalized Experience

These codes are associated with the following paper:

>LLM-based Conversational Recommendation Agents with Collaborative Verbalized Experience    
>EMNLP 2025 Findings 

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
