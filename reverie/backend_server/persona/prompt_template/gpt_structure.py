"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import requests
import openai
import time

from reverie.backend_server.utils import *

openai.api_key = openai_api_key


def temp_sleep(seconds=0.1):
    time.sleep(seconds)


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """
    temp_sleep()

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion["choices"][0]["message"]["content"]

    except:
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """
    temp_sleep()
    try:
        response = openai.Completion.create(
            model=gpt_parameter["engine"],
            prompt=prompt,
            temperature=gpt_parameter["temperature"],
            max_tokens=gpt_parameter["max_tokens"],
            top_p=gpt_parameter["top_p"],
            frequency_penalty=gpt_parameter["frequency_penalty"],
            presence_penalty=gpt_parameter["presence_penalty"],
            stream=gpt_parameter["stream"],
            stop=gpt_parameter["stop"], )
        return response.choices[0].text
    except:
        print("TOKEN LIMIT EXCEEDED")
        return "TOKEN LIMIT EXCEEDED"


def ollama_api_request(prompt, api_parameter):
    """ 通过 api 请求生成响应 """
    temp_sleep()
    try:
        data = {
            "model": api_parameter["engine"],
            "prompt": prompt,
            "options": {
                "temperature": api_parameter["temperature"],
                "num_predict": api_parameter["max_tokens"],
                "top_p": api_parameter["top_p"],
                "frequency_penalty": api_parameter["frequency_penalty"],
                "presence_penalty": api_parameter["presence_penalty"],
            },
            "stream": api_parameter["stream"],
        }
        response = requests.post(api_parameter["url"], json=data).json()
        text = response["response"]
        return text
    except (Exception,):
        err_msg = "api请求响应过程异常"
        print(err_msg)
        return err_msg


def generate_prompt(curr_input, prompt_lib_file):
    """
    Takes in the current input (e.g. comment that you want to classifiy) and
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this
    function replaces this substr with the actual curr_input to produce the
    final promopt that will be sent to the GPT3 server.
    ARGS:
      curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                  INPUT, THIS CAN BE A LIST.)
      prompt_lib_file: the path to the promopt file.
    RETURNS:
      a str prompt that will be sent to OpenAI's GPT server.
    """
    if isinstance(curr_input, str):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
    """ 生成 llm 回答的关键函数 """
    if verbose:
        print(prompt)

    for i in range(repeat):
        curr_gpt_response = ollama_api_request(prompt, gpt_parameter)
        # 判断生成的响应是否有效
        if func_validate(curr_gpt_response, prompt=prompt):
            # 对响应进行处理
            return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose:
            print("---- repeat count: ", i, curr_gpt_response)
            print(curr_gpt_response)
            print("~~~~")
    return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"
    return openai.Embedding.create(
        input=[text], model=model)['data'][0]['embedding']


if __name__ == '__main__':
    gpt_parameter = {"engine": "qwen2.5:7b", "max_tokens": 50,
                     "temperature": 0, "top_p": 1, "stream": False,
                     "frequency_penalty": 0, "presence_penalty": 0,
                     "stop": ['"'],
                     "url": "http://localhost:11434/api/generate",
                     }
    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "../prompt_template/v2/agent_chat_v1.txt"
    prompt = generate_prompt(curr_input, prompt_lib_file)


    def __func_validate(gpt_response, prompt=""):
        if len(gpt_response.strip()) <= 1:
            return False
        if len(gpt_response.strip().split(" ")) > 1:
            return False
        return True


    def __func_clean_up(gpt_response, prompt=""):
        cleaned_response = gpt_response.strip()
        return cleaned_response


    output = safe_generate_response(prompt,
                                    gpt_parameter,
                                    5,
                                    "rest",
                                    __func_validate,
                                    __func_clean_up,
                                    True)

    print(output)
