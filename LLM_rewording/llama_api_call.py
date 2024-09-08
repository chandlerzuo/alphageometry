from openai import OpenAI


def as_dict(**kwargs):
    return kwargs


def get_completion(conv):
    # see https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/Prompt_Engineering_with_Llama_3.ipynb
    extra_args = as_dict(
        model="tgi",
        # max_tokens=out_num_tokens,
        temperature=0.6,
        top_p=0.9,
    )
    if isinstance(conv, str):
        completion = client.completions.create(
            prompt=conv,
            **extra_args,
        )
        assert len(completion.choices) == 1, f"got {completion.choices}"
        return completion.choices[0].text
    else:
        completion = client.chat.completions.create(
            messages=conv,  # not conv[:-1]
            **extra_args,
        )
        assert len(completion.choices) == 1, f"got {completion.choices}"
        return completion.choices[0].message.content


# %%
endpoint = "http://i205:3000/v1"
# endpoint = "http://localhost:3000/v1"
client = OpenAI(
    base_url=endpoint,
    api_key="_",
)
# check if reachable
response = client.completions.create(
    model="tgi",
    prompt="Hello",
    max_tokens=20,
)
print("response", response)

response2 = get_completion("Hello")
print("response2", response2)