import time
import numpy as np

from copy import deepcopy
import nflx_copilot as ncp
import openai

def get_response(index, text, prompt, model, temperature, max_tokens, results, EXSTING):
    try:
        content = prompt.format(**text)
        if content in EXSTING:
            result = deepcopy(EXSTING[content])
            result['index'] = index
            print("Found in EXSTING!")
        else:
            resp = openai.ChatCompletion.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": content},
                ]
            )
            result = {'index': index, 'prompt': content, 'resp': resp}
            EXSTING[content] = result
        results.append(result)
    except Exception as e:
        if e == KeyboardInterrupt:
            raise e
        print(e)
        time.sleep(2)
        results.append({'index': index, 'prompt': content,
                       'resp': "API Failed"})