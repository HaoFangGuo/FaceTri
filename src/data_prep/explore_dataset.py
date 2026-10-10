import pandas as pd
import os

# ===== Configuration =====
ATTR_FILE = "data/CelebAMask-HQ-attribute-anno.txt"
IMG_DIR = "CelebA-HQ-img"

RANDOM_SEED = 42
N_SAMPLES_PER_CLASS = 50

OUTPUT_FILE = "data/benchmark/benchmark.csv"

df = pd.read_csv(
    ATTR_FILE,
    sep=r"\s+",
    skiprows=1,
    engine="python"
)

TARGET_ATTRS = [
    "Straight_Hair",
    "Wavy_Hair",
    "Receding_Hairline",
    "Big_Lips",
    "Big_Nose",
    "Bags_Under_Eyes",
    "Double_Chin",
    "5_o_Clock_Shadow",
    "Sideburns"
]

# 只保留这9个属性
df = df[TARGET_ATTRS]

samples = []

for attr in df.columns:
    yes = df[df[attr] == 1].sample(
        N_SAMPLES_PER_CLASS,
        random_state=RANDOM_SEED
    )

    no = df[df[attr] == -1].sample(
        N_SAMPLES_PER_CLASS,
        random_state=RANDOM_SEED
    )

    yes = yes.copy()
    no = no.copy()

    yes["attribute"] = attr
    no["attribute"] = attr

    yes["label"] = "Yes"
    no["label"] = "No"

    samples.append(yes)
    samples.append(no)

# 前面读取数据、抽样……

benchmark = pd.concat(samples)
PROMPTS = {
    "Straight_Hair": "Carefully examine the facial image.\n\nQuestion: Does this person have straight hair?\n\nAnswer with only one word: Yes or No.",
    "Wavy_Hair": "Carefully examine the facial image.\n\nQuestion: Does this person have wavy hair?\n\nAnswer with only one word: Yes or No.",
    "Receding_Hairline": "Carefully examine the facial image.\n\nQuestion: Does this person have a receding hairline?\n\nAnswer with only one word: Yes or No.",
    "Big_Lips": "Carefully examine the facial image.\n\nQuestion: Does this person have big lips?\n\nAnswer with only one word: Yes or No.",
    "Big_Nose": "Carefully examine the facial image.\n\nQuestion: Does this person have a big nose?\n\nAnswer with only one word: Yes or No.",
    "Bags_Under_Eyes": "Carefully examine the facial image.\n\nQuestion: Does this person have bags under the eyes?\n\nAnswer with only one word: Yes or No.",
    "Double_Chin": "Carefully examine the facial image.\n\nQuestion: Does this person have a double chin?\n\nAnswer with only one word: Yes or No.",
    "5_o_Clock_Shadow": "Carefully examine the facial image.\n\nQuestion: Does this person have a 5 o'clock shadow?\n\nAnswer with only one word: Yes or No.",
    "Sideburns": "Carefully examine the facial image.\n\nQuestion: Does this person have sideburns?\n\nAnswer with only one word: Yes or No."
}
benchmark = benchmark.reset_index().rename(columns={"index": "image_name"})

benchmark["prompt"] = benchmark["attribute"].map(PROMPTS)

# ⭐ 只保留最终需要的四列
benchmark = benchmark[
    ["image_name", "attribute", "label", "prompt"]
]

print(benchmark.head())
benchmark.insert(0, "id", range(1, len(benchmark) + 1))
benchmark.to_csv("data/benchmark/benchmark.csv", index=False)
