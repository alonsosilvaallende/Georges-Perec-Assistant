import re
from threading import Thread
from typing import List
import torch
import solara
from unicodedata import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation import LogitsProcessor
from typing_extensions import TypedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_id, cache_dir="/big_storage/llms/hf_models/"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

def response_generator(user_input, logits_processor=[], enable_thinking=False):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_input}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs = dict(
        model_inputs,
        streamer=streamer,
        logits_processor=logits_processor,
        max_new_tokens=4 * 1024,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
        top_k=50,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for chunk in streamer:
        if tokenizer.eos_token in chunk or tokenizer.pad_token in chunk:
            chunk = chunk.split(tokenizer.eos_token)[0]
            chunk = chunk.split(tokenizer.pad_token)[0]
        yield chunk
    thread.join()

list_of_vowels = ["a", "e", "i", "o", "u"]
tokens_per_vowel = dict()
for vowel in list_of_vowels:
    tokens_containing_a_given_vowel = []
    for token_id in range(tokenizer.vocab_size):
        if (
            vowel in tokenizer.decode(token_id)
            or vowel.upper() in tokenizer.decode(token_id)
            or normalize('NFC', f"{vowel}\u0300") in tokenizer.decode(token_id)
            or normalize('NFC', f"{vowel}\u0301") in tokenizer.decode(token_id)
            or normalize('NFC', f"{vowel}\u0302") in tokenizer.decode(token_id)
            or normalize('NFC', f"{vowel}\u0303") in tokenizer.decode(token_id)
            or normalize('NFC', f"{vowel}\u0308") in tokenizer.decode(token_id)
        ):
            tokens_containing_a_given_vowel.append(token_id)
    tokens_per_vowel[vowel] = tokens_containing_a_given_vowel

class GeorgePerecLogitsProcessor(LogitsProcessor):
    def __init__(self, forbidden_tokens: List[int]):
        self.forbidden_tokens = forbidden_tokens

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores_processed = scores.clone()
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        forbidden_tokens = torch.tensor(self.forbidden_tokens, device=scores.device)
        forbidden_tokens_mask = torch.isin(vocab_tensor, forbidden_tokens)
        scores_processed = torch.where(forbidden_tokens_mask, -torch.inf, scores)

        return scores_processed


def add_chunk_to_ai_message(chunk: str):
    messages.value = [
        *messages.value[:-1],
        {
            "role": "assistant",
            "content": messages.value[-1]["content"] + chunk,
        },
    ]

class MessageDict(TypedDict):
    role: str
    content: str

messages: solara.Reactive[List[MessageDict]] = solara.reactive([])
enable_thinking_options = [True, False]
enable_thinking = solara.reactive(False)
vowels = ["a", "e", "i", "o", "u", "None"]
vowel = solara.reactive("e")
@solara.component
def Page():
    solara.lab.theme.themes.light.primary = "#0000ff"
    solara.lab.theme.themes.light.secondary = "#0000ff"
    solara.lab.theme.themes.dark.primary = "#0000ff"
    solara.lab.theme.themes.dark.secondary = "#0000ff"
    title = "Georges Perec"
    with solara.Head():
        solara.Title(f"{title}")
    with solara.Column(align="center"):
        with solara.Sidebar():
            solara.Markdown("# G⎵org⎵s P⎵r⎵c")
            solara.Markdown("## Forcing a language model not to use a vowel")
            solara.Markdown("Select a forbidden vowel:")
            solara.ToggleButtonsSingle(value=vowel, values=vowels)
            solara.Markdown("Enable thinking:")
            solara.ToggleButtonsSingle(value=enable_thinking, values=enable_thinking_options)
            if vowel.value == "None":
                logits_processor = []
            else:
                logits_processor = [
                    GeorgePerecLogitsProcessor(
                        forbidden_tokens=tokens_per_vowel[vowel.value],
                    )
                ]
        user_message_count = len([m for m in messages.value if m["role"] == "user"])
        def send(message):
            messages.value = [*messages.value, {"role": "user", "content": message}]
        def response(message):
            messages.value = [*messages.value, {"role": "assistant", "content": ""}]
            for chunk in response_generator(message, logits_processor=logits_processor, enable_thinking=enable_thinking.value):
                add_chunk_to_ai_message(chunk)
        def result():
            if messages.value != []:
                response(messages.value[-1]["content"])
        result = solara.lab.use_task(result, dependencies=[user_message_count])
        with solara.lab.ChatBox(style={"position": "fixed", "overflow-y": "scroll","scrollbar-width": "none", "-ms-overflow-style": "none", "top": "0", "bottom": "10rem", "width": "60%"}):
            for item in messages.value:
                with solara.lab.ChatMessage(
                    user=item["role"] == "user",
                    name="User" if item["role"] == "user" else "Assistant",
                    avatar_background_color="#33cccc" if item["role"] == "assistant" else "#ff991f",
                    border_radius="20px",
                    style="background-color:darkgrey!important;" if solara.lab.theme.dark_effective else "background-color:lightgrey!important;"
                ):
                    solara.Markdown(item["content"])
        solara.lab.ChatInput(send_callback=send, style={"position": "fixed", "bottom": "3rem", "width": "60%"})
