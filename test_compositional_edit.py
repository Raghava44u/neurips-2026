import os
import torch
import sys
from statistics import mean
from datetime import datetime

from easyeditor import (
    MultimodalTrainer,
    CompositionalCaptionDataset,
    CompositionalDataset_RAG_70,
    CompositionalDataset_RAG_50,
    CompositionalDataset,
    MENDMultimodalTrainingHparams,
    SERACMultimodalTrainingHparams,
    FTMultimodalHparams,
    OURSMultimodalHparams,
    WISEMultimodalHyperParams,
    LORAMultimodalHparams
)

############################################################
# DATASET PATHS
############################################################

train_comp_final_json_path = "datasets/CCKEB_train.json"
eval_comp_final_json_path = "datasets/CCKEB_eval.json"

hop = 1


############################################################
# FT BASELINE
############################################################

def test_LLaVA_FT_comp():

    hparams = FTMultimodalHparams.from_hparams(
        "hparams/FT/llava_compositional_edit.yaml"
    )

    eval_ds = CompositionalDataset(
        eval_comp_final_json_path, config=hparams, hop=hop
    )

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_compositional_ft(
        log=True,
        gap_num=gap_num,
        test_num=50
    )

def test_LLaVA_OURS_robustness():
    hparams = OURSMultimodalHparams.from_hparams("hparams/OURS/llava.yaml")
    
    # 1. Create a dummy dataset object just to satisfy the trainer's initialization
    # We use the existing training path just as a placeholder
    temp_ds = CompositionalCaptionDataset("datasets/CCKEB_train.json", config=hparams)

    import json
    with open("cckeb_like_dataset.json", 'r') as f:
        custom_data = json.load(f)
    
    # 2. Pass temp_ds here
    trainer = MultimodalTrainer(config=hparams, train_set=temp_ds, val_set=temp_ds)
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    results = {"original": 0, "easy": 0, "medium": 0, "hard": 0}
    total = len(custom_data)

    print(f"--- Starting Robustness Eval on {total} samples ---")
    
    for item in custom_data:
        versions = {
            "original": item['compositional_query'],
            "easy": item['corrupted_versions']['easy'],
            "medium": item['corrupted_versions']['medium'],
            "hard": item['corrupted_versions']['hard']
        }
        
        for level, query in versions.items():
            # Use the trainer's internal inference logic
            prediction = trainer.edit_model_inference(model, tokenizer, item['image'], query)
            if item['compositional_answer'].lower() in prediction.lower():
                results[level] += 1

    for level, count in results.items():
        print(f"{level.upper()} Accuracy: {(count/total)*100:.2f}%")
        
def train_LLaVA_OURS_scratch():
    # Load the same Hparams
    hparams = OURSMultimodalHparams.from_hparams(
        "hparams/OURS/llava.yaml"
    )

    # USE THE TRAINING DATASET PATH
    train_ds = CompositionalCaptionDataset(
        train_comp_final_json_path, config=hparams, hop=hop
    )

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=train_ds # Using train as val for scratch reproduction
    )

    # CALL THE TRAIN METHOD
    # This will use the max_iters (50000) from your yaml
    trainer.run()
    
def test_MiniGPT4_FT_comp():

    hparams = FTMultimodalHparams.from_hparams(
        "hparams/FT/minigpt4_compositional_edit.yaml"
    )

    eval_ds = CompositionalDataset(
        eval_comp_final_json_path, config=hparams, hop=hop
    )

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_compositional_ft(
        log=True,
        gap_num=gap_num,
        test_num=50
    )


############################################################
# LoRA BASELINE
############################################################

def test_LLaVA_one_lora_comp():

    hparams = LORAMultimodalHparams.from_hparams(
        "hparams/LORA/llava_compositional_one_lora.yaml"
    )

    eval_ds = CompositionalDataset(
        eval_comp_final_json_path, config=hparams, hop=hop
    )

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_compositional_ft(
        log=True,
        gap_num=gap_num,
        test_num=50
    )


def test_MiniGPT4_one_lora_comp():

    hparams = LORAMultimodalHparams.from_hparams(
        "hparams/LORA/minigpt4_compositional_one_lora.yaml"
    )

    eval_ds = CompositionalDataset(
        eval_comp_final_json_path, config=hparams, hop=hop
    )

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_compositional_ft(
        log=True,
        gap_num=gap_num,
        test_num=50
    )


############################################################
# SERAC
############################################################

def test_LLaVA_SERAC_comp():

    hparams = SERACMultimodalTrainingHparams.from_hparams(
        "hparams/SERAC/llava.yaml"
    )

    eval_ds = CompositionalCaptionDataset(
        eval_comp_final_json_path, config=hparams, hop=hop
    )

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_multi_gpus(
        log=True,
        gap_num=gap_num,
        test_num=50,
        comp=True,
        training=False,
        gpus=[0]
    )


############################################################
# OURS (MemEIC)
############################################################

def test_LLaVA_OURS_comp():

    hparams = OURSMultimodalHparams.from_hparams(
        "hparams/OURS/llava.yaml"
    )

    eval_ds = CompositionalCaptionDataset(
        eval_comp_final_json_path, config=hparams, hop=hop
    )

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_multi_gpus(
        log=True,
        gap_num=gap_num,
        test_num=50,
        comp=True,
        training=False,
        gpus=[0]
    )


############################################################
# MAIN
############################################################


def test_MiniGPT4_OURS_comp():

    hparams = OURSMultimodalHparams.from_hparams(
        "hparams/OURS/minigpt4.yaml"
    )

    eval_ds = CompositionalCaptionDataset(
        eval_comp_final_json_path, config=hparams, hop=hop
    )

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_multi_gpus(
        log=True,
        gap_num=gap_num,
        test_num=50,
        comp=True,
        training=False,
        gpus=[0]
    )

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nAvailable functions:\n")

        for name in list(globals().keys()):
            if name.startswith("test_") or name.startswith("train_"):
                print(name)

        print("\nExample:")
        print("python test_compositional_edit.py test_LLaVA_OURS_comp\n")

        sys.exit()

    function_name = sys.argv[1]

    if function_name not in globals():
        print(f"\nError: function '{function_name}' not found.\n")
        sys.exit()

    for gap_num in [0, 10, 20, 50, 100]:
        globals()[function_name]()

def test_MiniGPT4_OURS_comp():

    hparams = OURSMultimodalHparams.from_hparams(
        "hparams/OURS/minigpt4.yaml"
    )

    eval_ds = CompositionalCaptionDataset(
        eval_comp_final_json_path, config=hparams, hop=hop
    )

    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_multi_gpus(
        log=True,
        gap_num=gap_num,
        test_num=50,
        comp=True,
        training=False,
        gpus=[0]
    )
