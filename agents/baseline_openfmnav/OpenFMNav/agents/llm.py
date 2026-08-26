import os
import re
import openai
import retry
import requests
from vl_prompt.p_manager import extract_integer_answer, extract_scores, extract_objects, \
    get_frontier_prompt, get_candidate_prompt, get_grouping_prompt, get_discover_prompt

_DUMMY_LLM = os.environ.get("OPENFMNAV_DUMMY_LLM", "").strip() == "1"
api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not api_key and not _DUMMY_LLM:
    # should be replaced by your API key
    with open("./apikey.txt") as f:
        # key order：temp key, long-term, mine
        keys = f.read().split("\n")
        openai.api_key = keys[0] # NOTE: change key before running


def _dummy_frontier_reply(prompt_type, message):
    n = len(re.findall(r"^- Description \d+:", message or "", re.MULTILINE))
    if prompt_type == "deterministic":
        return "Answer: 0"
    scores = [1.0] + [0.0] * max(n - 1, 0)
    return "Answer: [{}]".format(", ".join(str(x) for x in scores))


class LLM():
    def __init__(self, goal_name, prompt_type):
        self.api_name = "gpt-4.1-mini"
        # self.api_name = "gpt-3.5-turbo"
        self.goal_name = goal_name
        self.prompt_type = prompt_type
        self.history = []
    
    def inference_once(self, system_prompt, message):
        if _DUMMY_LLM:
            return _dummy_frontier_reply(self.prompt_type, message)
        if message:
            msg = {
                "role": "user",
                "content": message
            }
            self.history = system_prompt + [msg]
            try:
                chat = openai.ChatCompletion.create(
                    model=self.api_name, messages=self.history,
                    temperature=0
                )
                print(f"=====> llm inference success: {self.api_name}")
            except Exception as e:
                print(f"=====> llm inference error: {e}")
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=self.history,
                    temperature=0
                )
        reply = chat.choices[0].message.content
        return reply
    
    def discover_objects(self, img, objects):
        if _DUMMY_LLM:
            return []
        payload = get_discover_prompt(img, objects)
        print(f"=====> DiscoverVLM inference: {payload['model']}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        c = extract_objects(reply)
        print(f"=====> discovered objects: {c}")
        return c
        
    def inference_accumulate(self, message):
        if message:
            self.history.append(
                {"role": "user", "content": message},
            )
            try:
                chat = openai.ChatCompletion.create(
                    model=self.api_name, messages=self.messages
                )
                print(f"=====> llm inference success: {self.api_name}")
            except Exception as e:
                print(f"=====> gpt-4-turbo error: {e}")
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=self.history,
                    temperature=0
                )
            
        reply = chat.choices[0].message.content
        # print(reply)
        self.history.append({"role": "assistant", "content": reply})
        return reply
    
    def choose_frontier(self, message):
        system_prompt = get_frontier_prompt(self.prompt_type)
        reply = self.inference_once(system_prompt, message)
        if self.prompt_type == "deterministic":
            answer = extract_integer_answer(reply)
        elif self.prompt_type == "scoring":
            answer, scores = extract_scores(reply)
        
        return answer, reply
        
    def imagine_candidate(self, instr):
        system_prompt = get_candidate_prompt(candidate_type="open")
        reply = self.inference_once(system_prompt, instr)
        c = extract_objects(reply)
        return c
    
    def group_candidate(self, clist, nlist):
        system_prompt = get_grouping_prompt()
        message = f"Current object list: {clist}\n\nNew object list: {nlist}"
        reply = self.inference_once(system_prompt, message)
        c = extract_objects(reply) # newly discovered object list
        new_c = []
        for ob in c:
            if ("room" in ob) or ("wall" in ob) or ("floor" in ob) or ("ceiling" in ob):
                continue
            new_c.append(ob)
        return new_c