"""
Deploy model with gradio, then input questions directly (without using the "### Question ... ### Answer")
"""

import logging
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import AutoPeftModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from transformers import pipeline
import gradio as gr
from transformers import BitsAndBytesConfig

from LLM_finetuner.question_answer_utils import format_question_answer
from LLM_finetuner.utils import load_model_for_inference


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

model_checkpoints_dir = sys.argv[1]

model, tokenizer = load_model_for_inference(model_checkpoints_dir)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, num_return_sequences=2, num_beams=4, do_sample=True, max_new_tokens=50)

print("Deploying model")
# server.close() # to reuse port
# server = gr.Interface(fn=lambda question: get_answer(question, remove_query_from_response=True), inputs="text", outputs="text", flagging_dir="geometry_translation/flagged_inputs")
# server.launch(server_port=7862) # no copy paste in vscode jupyter, but in browser

# # del server
# try:
#     # if previous server still running
#     server.close()
# except:
#     pass

interface_kwargs = gr.pipelines.load_from_pipeline(pipe)
old_fn = interface_kwargs.pop("fn")
def new_fn(question):
    question = format_question_answer(question, "")
    # return question
    logger.info(f"Submitting question {question}")
    response = old_fn(question)
    logger.info(f"Got response {response}")
    # return response
    return response[len(question):]
interface_kwargs["fn"] = new_fn
server = gr.Interface(**interface_kwargs)#, flagging_dir="geometry_translation/flagged_inputs")
running_app = server.launch(server_port=7869)
