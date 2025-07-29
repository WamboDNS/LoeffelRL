import os
from dotenv import load_dotenv
from typing import TypedDict
import csv
import random
import art
from art.local import LocalBackend
import weave
from pydantic import BaseModel
import openai
import re
import Levenshtein
import asyncio

SYSTEM_PROMPT = """
You are an expert in transforming standard German into German Spoon Language or Löffelsprache.
Given a German sentence, you will transform it into German Spoon Language.
Follow these strict rules:

Let x be any of the following vowels or vowel pairs:
{ei, ie, au, eu, äu, a, e, i, o, u}
For each occurrence of x (here a variable), replace it with xlewx.
Example: a → alewa, ei → eilewei
Always match vowel pairs first, before checking for single vowels.
After a replacement, continue from the end of the replaced text — do not reprocess inside the result.
Preserve casing:
If the original x begins with an uppercase letter, only the first letter of the xlewx replacement is uppercase.
Example: A → Alewa, Ei → Eilewei, Au → Aulewau
Example words:
Hallo -> Halewallolewo
Eier -> Eileweielewer
Do not apply transformations recursively.
Return only the converted sentence, wrapped in <spoon> ... </spoon> tags.
Do not explain your transformation.
"""

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

class SentencePair(BaseModel):
    german: str
    spoon: str
    
def load_data(file_path: str) -> list[SentencePair]:
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        return [SentencePair(german=row[0], spoon=row[1]) for row in reader]

def draw_sample(data: list[SentencePair]) -> SentencePair:
    return random.choice(data)

@weave.op
@art.retry(exceptions=(openai.LengthFinishReasonError,))
async def rollout(model: art.Model, pair: SentencePair) -> art.Trajectory:
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ],
        metadata={
            "notebook-id": "SpoonRL",
        },
        reward=0,
    )
    trajectory.messages_and_choices.append({
        "role": "user",
        "content": pair.german,
    })
    messages = trajectory.messages()
    try:
        client = model.openai_client()
        chat_completion = await client.chat.completions.create(
            model=model.get_model_name(),
            messages=messages,
            max_tokens=2048,
        )
    except openai.LengthFinishReasonError as e:
        raise e
    except Exception as e:
        print("Caught exception generating chat comopletion")
        print(e)
        global failing_trajectory
        failing_trajectory = trajectory
        raise e
    
    choice = chat_completion.choices[0]
    content = choice.message.content
    
    format_reward = 0
    match = re.search(r"<spoon>(.*?)</spoon>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
        format_reward = 1
    else:
        match = content
    dist = Levenshtein.distance(match, pair.spoon)
    max_len = max(len(match), len(pair.spoon), 1)
    spoon_reward = 1.0 - dist / max_len  
    
    reward = spoon_reward * 0.8 + format_reward * 0.2
    trajectory.reward = reward
    return trajectory    

async def main():
    data = load_data("data/german_spoon.csv")
    random.seed(42)
    backend = LocalBackend()
    model = art.TrainableModel(
        name="001-german-spoon",
        project="SpoonRL",
        base_model="wambosec/Qwen2.5-7B-Instruct-spoon-language-SFT",
    )
    await model.register(backend)
    print("initializing weave")
    weave.init(model.project, settings={"print_call_link": False})
    
    for i in range(400):
        pair = random.choice(data)
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(model, pair) for _ in range(10))
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=10,
        )
        await model.delete_checkpoints()
        await model.train(
            train_groups,
            config=art.TrainConfig(
                learning_rate=1e-5,
            ),
        )
    
if __name__ == "__main__":
    asyncio.run(main())