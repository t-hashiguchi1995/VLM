## Fine-tune LLaVa for document parsing (PDF -> JSON)
# このノートブックでは、ドキュメントAIのユースケースに向けて[LLaVa](https://huggingface.co/docs/transformers/main/en/model_doc/llava)モデルをファインチューニングします。LLaVaは、執筆時点で最も優れたオープンソースのマルチモーダルモデルの1つです。

# ## 必要なライブラリのインポート
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import json

# ## 変数の定義

# %%
MAX_LENGTH = 384
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REPO_ID = "thashiguchi/llava-finetuning-demo"
WANDB_PROJECT = "LLaVa"
WANDB_NAME = "llava-demo-cord"

# %% [markdown]
# ## データセットの読み込み

# %%
dataset = load_dataset("naver-clova-ix/cord-v2")

# %% [markdown]
# ## プロセッサとモデルの読み込み

# %%
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

# %% [markdown]
# ## LoRAの設定

# %%
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# %% [markdown]
# ## データの前処理関数

# %%

from torch.utils.data import Dataset
from typing import Any, Dict
import random

from datasets import Dataset

def preprocess_function(examples):
    images = examples["image"]
    texts = [f"USER: <image>\nExtract JSON.\nASSISTANT: {json.loads(gt)['gt_parse']}" for gt in examples["ground_truth"]]
    
    inputs = processor(text=texts, images=images, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["labels"][inputs["labels"] == processor.tokenizer.pad_token_id] = -100
    
    return inputs

def load_and_prepare_dataset(dataset_name_or_path, split):
    dataset = load_dataset(dataset_name_or_path, split=split)
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )
    return processed_dataset

train_dataset = load_and_prepare_dataset("naver-clova-ix/cord-v2", "train")
eval_dataset = load_and_prepare_dataset("naver-clova-ix/cord-v2", "validation")

def data_collator(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features])
    }
def formatting_func(example):
    ground_truth = example['ground_truth']
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]  # リストの場合、最初の要素を使用
    try:
        gt_parse = json.loads(ground_truth)['gt_parse']
    except json.JSONDecodeError:
        gt_parse = "Error: Unable to parse JSON"
    except KeyError:
        gt_parse = "Error: 'gt_parse' key not found in JSON"
    
    return f"USER: <image>\nExtract JSON.\nASSISTANT: {gt_parse}"

print("Train dataset example:")
print(dataset["train"][0])
print("\nValidation dataset example:")
print(dataset["validation"][0])

# %% [markdown]
# ## トレーニングの設定

# %%
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    gradient_accumulation_steps=4,
)

# %% [markdown]
# ## トレーナーの設定

# %%
# トレーナーの設定を修正
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,  # カスタムデータコレーターを使用
    tokenizer=processor.tokenizer,
    formatting_func=formatting_func,
    max_seq_length=MAX_LENGTH,
)

# %% [markdown]
# ## トレーニングの実行

# %%
trainer.train()

# %% [markdown]
# ## モデルの保存

# %%
trainer.save_model(REPO_ID)

# %% [markdown]
# ## 推論

# %%
# テスト用の画像を読み込む
test_example = dataset["test"][0]
test_image = test_example["image"]

# 入力の準備
prompt = f"USER: <image>\nExtract JSON.\nASSISTANT:"
inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to("cuda")

# 生成
generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

# デコード
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)

# %% [markdown]
# これで、transformers、trl、peftを使用してLLaVaモデルをファインチューニングし、推論を行うコードが完成しました。
