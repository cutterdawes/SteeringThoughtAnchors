import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "A": "Qwen/Qwen3-0.6B",
    "B": "Qwen/Qwen3-4B"
}

prompt_A = f"""
In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole
"""
prompt_B = f"""
In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole
"""
cot_A = f"""
You are solving a mathematical problem.
Keep the CoT explanations concise but clear. Ensure diversity in reasoning.

The **CoT should NOT state the final answer explicitly**. Instead, reason through the problem naturally and logically.
After reasoning, **put your final answer inside \\boxed{{}}**.

### **STRICT AND MANDATORY FORMAT (Follow exactly, no deviations are allowed):**
Examples of VALID expected output:
- \\boxed{{120000}}
- \\boxed{{90}}

Problem:
Sean is practicing for his role in a theater production. He has to memorize his lines for two scenes and the lyrics to one solo song. His solo song has 54 lines in the lyrics. The first scene has twice the number of lines, but only a third of them are his lines. The second scene has six more lines than the song, and four-fifths of them are his. How many lines does Sean have to memorize?

<think>
First, let’s calculate the number of lines in each part.
The solo song has 54 lines inside.
The first scene has twice the number of lines as the song: 2 * 54 = 108 lines. Only a third of them belong to Sean: 108 / 3 = 36 lines.
The second scene has six more lines than the song: 54 + 6 = 60 lines. Four-fifths are Sean’s lines: 60 * 4 / 5 = 48 lines.

Adding them together: 36 + 48 + 54 = 138 lines.
</think>
"""


cot_B = f"""
You are solving a mathematical problem.
Keep the CoT explanations concise but clear. Ensure diversity in reasoning.

The **CoT should NOT state the final answer explicitly**. Instead, reason through the problem naturally and logically.
After reasoning, **put your final answer inside \\boxed{{}}**.

### **STRICT AND MANDATORY FORMAT (Follow exactly, no deviations are allowed):**
Examples of VALID expected output:
- \\boxed{{120000}}
- \\boxed{{90}}

Problem:
Sean is practicing for his role in a theater production. He has to memorize his lines for two scenes and the lyrics to one solo song. His solo song has 54 lines in the lyrics. The first scene has twice the number of lines, but only a third of them are his lines. The second scene has six more lines than the song, and four-fifths of them are his. How many lines does Sean have to memorize?

<think>
First, let's calculate the number of lines in each part.
The solo song has 54 lines inside. 
The first scene has twice the number of lines as the song: 2 * 54 = 108 lines. Only a third of them are Sean's lines: 108 / 3 = 36 lines. 
The second scene has six more lines than the song: 54 + 6 = 60 lines. Four-fifths are Sean's lines: 60 * 4 / 5 = 48 lines. 

Adding them together: 36 + 48 + 54 = 138 lines.
</think>
"""