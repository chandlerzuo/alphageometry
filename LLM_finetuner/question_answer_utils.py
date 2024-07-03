# # from trl.extras.dataset_formatting import conversations_formatting_function

# # functions related to the question-answer style dataset

# from typing import Literal
# from transformers import AutoTokenizer


# def format_question_answer(question, answer):
#     return f"### Question: {question} ### Answer: {answer}"

# question_template = "### Question: "
# response_template = " ### Answer: "
# assert response_template.endswith(" ") # needs to be removed for generation to avoid leading underscores ____

# def extract_question_prompt(question_answer_text):
#     """
#     Given f"### Question: {question} ### Answer: {answer}", 
#     returns
#     f"### Question: {question} ### Answer: "
#     """
#     assert question_answer_text.startswith(question_template)
#     answer_start_idx = question_answer_text.find(response_template)
#     return question_answer_text[:answer_start_idx+len(response_template)-1] # -1 to avoid leading underscores

# def extract_answer(question_answer_text):
#     """
#     Given f"### Question: {question} ### Answer: {answer}", 
#     returns
#     f"{answer}"
#     """
    
#     answer_start_idx = question_answer_text.rfind(response_template)
#     return question_answer_text[answer_start_idx+len(response_template):]

# def extract_question_answer(question_answer_text):
#     assert question_answer_text.startswith(question_template)
#     answer_start_idx = question_answer_text.rfind(response_template)
#     return question_answer_text[len(question_template):answer_start_idx], \
#         question_answer_text[answer_start_idx+len(response_template):]
        

# # adapted from: from trl.extras.dataset_formatting import conversations_formatting_function
# # by adding kwargs
# def conversations_formatting_function_with_kwargs(tokenizer: AutoTokenizer, messages_field: Literal["messages", "conversations"], **kwargs):
#     r"""
#     return a callable function that takes in a "messages" dataset and returns a formatted dataset, based on the tokenizer
#     apply chat template to the dataset
#     """

#     def format_dataset(examples):
#         if isinstance(examples[messages_field][0], list):
#             output_texts = []
#             for i in range(len(examples[messages_field])):
#                 output_texts.append(tokenizer.apply_chat_template(examples[messages_field][i], tokenize=False, **kwargs))
#             return output_texts
#         else:
#             return tokenizer.apply_chat_template(examples[messages_field], tokenize=False, **kwargs)

#     return format_dataset

# def get_question_answer_to_chat_formatter(tokenizer, text_column, include_task_desc=True, add_generation_prompt=False, **kwargs):
#     # add_generation_prompt: when True, omits the answer, but adds a generation prompt, so the answer can be generated
#     conversation_formatter = conversations_formatting_function_with_kwargs(
#         tokenizer, "conversations", add_generation_prompt=add_generation_prompt, **kwargs
#     )
#     system_message = f"""\
# You are an expert in translating geometry problems from natural language into formal language.
# It is crucial to form syntactically valid statements in the formal language that correspond 
# exactly to its natural language description.
# You should only output the formal language description of the question, not the solution or anything else.\
#     """
    
#     def convert_question_answer_to_chat_format(batch):
#         # given single or batch of f"### Question: {question} ### Answer: {answer}", converts to a chat
        
#         batch = batch[text_column] if text_column is not None else batch
#         if isinstance(batch, list):
#             # batch: must support batching for trl, dataset.map(.., batched=True)
#             return [convert_question_answer_to_chat_format_single(ex) for ex in batch]
#         else:
#             return convert_question_answer_to_chat_format_single(batch)
        
#     def convert_question_answer_to_chat_format_single(row):
#         # given f"### Question: {question} ### Answer: {answer}", converts to a chat
#         nl_statement, fl_statement = extract_question_answer(row)
#         chat = []
#         if include_task_desc:
#             chat.append({
#                 "role": "system",
#                 "content": system_message
#             })
#         chat.append({
#             "role": "user",
#             "content": nl_statement  
#         })
#         if not add_generation_prompt:
#             chat.append({
#                 "role": "assistant",
#                 "content": fl_statement
#             })
#         return conversation_formatter({"conversations": chat})
    
#     return convert_question_answer_to_chat_format

    
# if __name__ == "__main__":
#     assert extract_question_prompt(format_question_answer("MyQuestion 111", "MyAnswer 222")) == "### Question: MyQuestion 111 ### Answer: "
#     assert extract_answer(format_question_answer("MyQuestionc111", "MyAnswer 222")) == "MyAnswer 222"
#     assert extract_question_answer(format_question_answer("MyQuestion 111", "MyAnswer 222")) == ("MyQuestion 111", "MyAnswer 222")
    
    
#     model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
#     # model_name_or_path = "meta-llama/Llama-2-7b-hf"
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_eos_token=True)
#     # inputs = "Hello world"
#     # inputs = [
#     #     {"role": "system", "content": "Hello from system"},
#     #     # must be ordered as user-assistant alternating sequence
#     #     {"role": "user", "content": "Translate this"},
#     #     {"role": "assistant", "content": "Translated"},
#     # ]
#     # print(conversation_formatter({"conversations": inputs}))
    
#     print(tokenizer.chat_template)
#     print()
    
#     conversation_formatter = get_question_answer_to_chat_formatter(tokenizer, text_column="text")
#     formatted_conversation = conversation_formatter({"text": format_question_answer("MyQuestion 111", "MyAnswer 222")})
#     print(formatted_conversation)
#     assert formatted_conversation.find("MyQuestion 111") < formatted_conversation.find("MyAnswer 222")