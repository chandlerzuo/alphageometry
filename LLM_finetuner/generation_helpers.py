import wandb
from transformers.trainer_callback import TrainerControl, TrainerState, TrainerCallback
from transformers import TrainingArguments, GenerationConfig
import functools
import contextlib

@contextlib.contextmanager
def set_padding_side_cm(tokenizer, padding_side):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    yield
    tokenizer.padding_side = old_padding_side

def model_generate_on_batch(batch, model, tokenizer):
    with set_padding_side_cm(tokenizer, "left"):
        encoded_inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
        encoded_inputs = encoded_inputs.to(model.device)
        gen_config = GenerationConfig(
            max_new_tokens=70, num_beams=2,
            pad_token_id=tokenizer.pad_token_id,
        )
        encoded_generations = model.generate(**encoded_inputs, generation_config=gen_config)
        return {"generated": tokenizer.batch_decode(encoded_generations, skip_special_tokens=True)}

class MakeModelGenerations(TrainerCallback):
    """
    Need to assign .gen_steps to TrainingArguments
    """
    def __init__(self, gen_dataset, prefix=""):
        super().__init__()
        self.gen_dataset = gen_dataset
        self.prefix = prefix
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, tokenizer, **kwargs):
        if not state.is_world_process_zero:
            # for now only generate on main process
            return
        print(f"Step: {state.global_step}, {model.device}")
        if (args.gen_steps == -1) or (state.global_step % args.gen_steps != 0):
            return
        
        # generations = self.gen_dataset.map(
        #     # model_generate_on_batch, 
        #     functools.partial(model_generate_on_batch, model=model, tokenizer=tokenizer),#, device=args.device), 
        #     batched=True, batch_size=args.per_device_eval_batch_size,
        #     remove_columns=set(self.gen_dataset.column_names) - {"labels"}
        # )
        generations = self.gen_dataset.map(
            # model_generate_on_batch, 
            functools.partial(model_generate_on_batch, model=model, tokenizer=tokenizer),#, device=args.device), 
            load_from_cache_file=False, keep_in_memory=True,
            batched=True, batch_size=args.per_device_eval_batch_size
        )
        #%%
        # generations = generations.select(range(min(2, len(generations))))
        df = generations.to_pandas()
        df["question"] = df["text"]
        df["gt_answer"] = df.apply(lambda x: x["labels"][len(x["text"]):], axis=1)
        df["pred_answer"] = df.apply(lambda x: x["generated"][len(x["text"]):], axis=1)
        df.drop(columns=set(df.columns) - {"question", "gt_answer", "pred_answer"}, inplace=True)
        df["epoch"] = state.epoch
        # logger.info(generations.to_pandas())
        # todo
        import pandas as pd
        with pd.option_context("display.max_colwidth", None), pd.option_context("display.max_columns", None):
            print(df)
        if wandb.run is not None:
            wandb_key = self.prefix + "generated_text"
            # wandb_key = f"eval_generated_text_{state.epoch}" # UI does not handle many tables with similar key very well
            wandb.log({wandb_key: wandb.Table(dataframe=df)}) # wandb only displays last table instead of using a slider as of now
            
        # return df
            