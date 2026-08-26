from openai import OpenAI
import base64
import cv2
import os
import time

# --- rt_ovn author addition (begin): deterministic dummy path for the probe ---
# When BMN_DUMMY_LLM=1 the GPT-4o calls in this file are stubbed out and return
# fixed canned answers so probe runs are reproducible without an OpenAI API key
# (mirrors the OPENFMNAV_DUMMY_LLM pattern used in OpenFMNav).
_DUMMY_LLM = os.environ.get("BMN_DUMMY_LLM", "").strip() == "1"


def _dummy_refinement_reply():
    # "Answer: yes" keeps every detection that GroundingDINO/YOLO produced.
    return "Reasoning: dummy\nAnswer: yes"


def _dummy_choise_reply(color_string_list):
    # Deterministically pick the first candidate color.
    if not color_string_list:
        return ""
    first = color_string_list[0]
    return str(first)
# --- rt_ovn author addition (end) ---


class OpenAI_API:
    def __init__(self):
        # rt_ovn author addition: upstream hard-codes the placeholders
        # "your api" / "you url" and always passes base_url, so a plain OpenAI
        # key cannot work. Read both from the environment instead, and only pass
        # base_url when set — the SDK already defaults to api.openai.com, and the
        # upstream authors' custom value was for a third-party relay.
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        # An *empty* OPENAI_BASE_URL env var (e.g. `OPENAI_BASE_URL=` in a
        # docker-compose environment block) is read internally by the OpenAI
        # SDK and used verbatim, producing protocol-less request URLs and
        # httpx.UnsupportedProtocol on every call. Remove it so the SDK falls
        # back to https://api.openai.com/v1.
        if not self.base_url:
            os.environ.pop("OPENAI_BASE_URL", None)
        # Skip client construction in dummy mode so no key is needed just to
        # instantiate the policy.
        if _DUMMY_LLM:
            self.gpt_client = None
        else:
            client_kwargs = {"base_url": self.base_url} if self.base_url else {}
            self.gpt_client = OpenAI(api_key=self.api_key, **client_kwargs)

    # --- rt_ovn author addition (begin): retry transient API failures ---
    # Upstream has no error handling around GPT-4o calls, so a single transient
    # openai.APIConnectionError kills the whole agent process during long
    # parallel evals. Retry with backoff (same spirit as OpenFMNav's
    # sleep-20s-and-retry loops in agents/llm.py and main.py).
    def _chat_with_retry(self, **kwargs):
        n_attempts = 8
        for attempt in range(n_attempts):
            try:
                return self.gpt_client.chat.completions.create(**kwargs)
            except Exception as e:
                if attempt == n_attempts - 1:
                    raise
                wait_s = min(20, 2 ** attempt)
                print(
                    f"[rt_ovn] OpenAI call failed ({type(e).__name__}: {e}); "
                    f"retry {attempt + 1}/{n_attempts - 1} in {wait_s}s...",
                    flush=True,
                )
                time.sleep(wait_s)
    # --- rt_ovn author addition (end) ---

    def detection_refinement(self, image, object):
        # rt_ovn author addition: dummy short-circuit.
        if _DUMMY_LLM:
            return _dummy_refinement_reply()
        base64_image = self.encode_image_from_array(image)
        if object.lower().startswith("couch"):
            # prompt =f"The couch must have at least three seat for person,the chair for one person is not couch. is there a/an {object} in the bbox of the given image? Please answer yes or no. "
            prompt =f''' onsider the reasonableness of the {object}'s appearance in the environment of the given iamge. is there a/an {object} in the contour line/bbox of the given image? if there is another couch in image is bigger than the couch in bbox, please answer also no. Please answer yes or no.
            1. The couch must have at least three seat for person,the chair for one person is not couch.
            2. The couch should in living room or bedroom, not in other rooms like kitchen or bathroom.
            3. pelase give your reasoning process and the answer.
            4. the output formulation is: 
            Reasoning: (the reasoning of the answer)
            Answer: yes or no
            '''
        elif object.lower().startswith("tv"):
            prompt = f'''onsider the reasonableness of the {object}'s appearance in the environment of the given iamge. is there a/an {object} in the contour line/bbox of the given image? Please answer yes or no. 
            1. Be careful not to mistake the picture frames and black Windows on the walls for televisions.
            2. The tv screen shuold be black, not white or other colors.
            3. There is no tv on a door or in toilet or in kitchen!
            4. pelase give your reasoning process and the answer.
            5. the output formulation is: 
            Reasoning: (the reasoning of the answer)
            Answer: yes or no'''
        elif object.lower().startswith("chair"):
            prompt = f'''onsider the reasonableness of the {object}'s appearance in the environment of the given iamge. is there a/an {object} in the contour line/bbox of the given image? Please answer yes or no. Be careful not to mistake the couch for chair
            1. please attention: the chair only have one seat, but the couch have seats over one.
            2. only output yes or no.
            3. pelase give your reasoning process and the answer.
            4. the output formulation is: 
            Reasoning: (the reasoning of the answer)
            Answer: yes or no'''
        elif "bed" in object.lower():
            prompt = f'''onsider the reasonableness of the {object}'s appearance in the environment of the given iamge. is there a/an {object} in the contour line/bbox of the given image? Please answer yes or no.
            1. Please attention: the bed can only be in bedroom, not in other rooms like living room or bathroom and so on. If the environment is living room, there is no bed.
            2. please attention the environment of the image, if you are not sure, please answer no.
            3. If the detected object is on the edge of the image and you are not sure, please answer No.
            4. pelase give your reasoning process and the answer.
            5. the output formulation is following: 
            Reasoning: (the reasoning of the answer)
            Answer: yes or no'''        
        else:
            prompt = f'''onsider the reasonableness of the {object}'s appearance in the environment of the given iamge. is there a/an {object} in the contour line/bbox of the given image?
            1. pelase give your reasoning process and the answer.
            2. the output formulation is: 
            Reasoning: (the reasoning of the answer)
            Answer: yes or no'''
        
        # rt_ovn author addition: retry wrapper (was a bare .create call).
        completion = self._chat_with_retry(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )
        answer = completion.choices[0].message.content
        return answer
    
    def detection_choise(self, image, object, color_string_list):
        # rt_ovn author addition: dummy short-circuit.
        if _DUMMY_LLM:
            return _dummy_choise_reply(color_string_list)
        base64_image = self.encode_image_from_array(image)
        if "couch" in object:
            prompt = f'''There are {len(color_string_list)} {object} in the image with {color_string_list} color contour line/bbox. you should choose only one object that best matches the feature of {object}. Please give the color of contour line/bbox of the chosen object. Only output one color in {color_string_list} of the contour line/bbox
            1. If there are some couches in this image, select the target with the most seats.
            2. Only output the color of the contour line/bbox.'''
        if "chair" in object:
            prompt = f'''There are {len(color_string_list)} {object} in the image with {color_string_list} color contour line/bbox. you should choose only one object that best matches the feature of {object}. Please give the color of contour line/bbox of the chosen object. Only output one color in {color_string_list} of the contour line/bbox
            1. please choose the chair which only has one seat, when there are some couches in the image. 
            2. Only output the color of the contour line/bbox.'''
        else:
            prompt = f"There are {len(color_string_list)} {object} in the image with {color_string_list} color contour line/bbox. you should choose only one object that best matches the feature of {object}. Please give the color of contour line/bbox of the chosen object. Only output one color in {color_string_list} of the contour line/bbox"
        # rt_ovn author addition: retry wrapper (was a bare .create call).
        completion = self._chat_with_retry(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )
        print("prompt: ",prompt)
        answer = completion.choices[0].message.content
        return answer
    
    
    def encode_image_from_array(self, image):
        # Convert RGB image to BGR before encoding
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Convert BGR image to JPEG format byte stream
        _, buffer = cv2.imencode(".jpg", image_bgr)
        # Convert byte stream to Base64 string
        return base64.b64encode(buffer).decode("utf-8")