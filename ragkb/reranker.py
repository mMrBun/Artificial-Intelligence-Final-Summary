from .model_manager import ModelManager


class Reranker:
    def __init__(self, models: ModelManager):
        self.models = models

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        import torch

        with self.models.lock, torch.inference_mode():
            tokenizer, model = self.models.reranker()
            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            yes, no = tokenizer.convert_tokens_to_ids(["yes", "no"])
            result = []
            for start in range(0, len(documents), 2):
                texts = [
                    f"<Instruct>: Given a knowledge-base question, retrieve passages that answer the question\n<Query>: {query}\n<Document>: {doc}"
                    for doc in documents[start : start + 2]
                ]
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=2048 - len(prefix_ids) - len(suffix_ids),
                    return_attention_mask=False,
                )
                encoded["input_ids"] = [
                    prefix_ids + ids + suffix_ids for ids in encoded["input_ids"]
                ]
                batch = tokenizer.pad(encoded, padding=True, return_tensors="pt").to(model.device)
                logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                scores = torch.softmax(logits[:, [no, yes]].float(), dim=-1)[:, 1]
                result.extend(scores.cpu().tolist())
            return result
