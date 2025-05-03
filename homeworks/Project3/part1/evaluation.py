# Don't edit this cell
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to(device)

# Run inference
from typing import List
def generate_batch_responses(input_ids, attention_mask, model, tokenizer, max_new_tokens=64) -> List[str]:
    # your code begins

    # your code ends

# Don't edit this cell.
from tqdm import tqdm
predictions = []
references = [example["response"].strip() for example in eval_dataset]

for batch in tqdm(eval_dataloader):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    batch_outputs = generate_batch_responses(
        input_ids, attention_mask, model, tokenizer
    )

    predictions.extend(batch_outputs)

# Don't edit this cell.
# the meaning of these metrics:
# bleu and rouge: https://avinashselvam.medium.com/llm-evaluation-metrics-bleu-rogue-and-meteor-explained-a5d2b129e87f
# bertscore: https://medium.com/@abonia/bertscore-explained-in-5-minutes-0b98553bfb71

import evaluate  # 🤗 evaluate library
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# Don't edit this cell.
bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
rouge_result = rouge.compute(predictions=predictions, references=references)
bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
print(f"BLEU: {bleu_result['bleu'] * 100:.2f}")
print(f"ROUGE-L: {rouge_result['rougeL'] * 100:.2f}")
print(f"BERTScore (F1): {sum(bertscore_result['f1']) / len(bertscore_result['f1']) * 100:.2f}")