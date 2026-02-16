import re

import transformers
import torch


def split_chunk_into_words(chunk: str) -> list[tuple[str, int, int]]:
    """
    Returns list of (word, start_char, end_char)
    """
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+(?:'\w+)?|[^\w\s]", chunk)]


def prepare_sample_for_model(
    text: str,
    topic: str,
    tokenizer: transformers.AutoTokenizer,
    include_topic_description: bool = True,
    topic_description: str | None = None,
    max_length: int = 512,
):
    topic_text = f"{topic}" if not include_topic_description else f"{topic} - {topic_description}"

    tokenizer_output = tokenizer(
        topic_text,
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    offsets = tokenizer_output["offset_mapping"].squeeze(0).tolist()
    input_ids = tokenizer_output["input_ids"].squeeze(0)
    attention_mask = tokenizer_output["attention_mask"].squeeze(0)
    token_type_ids = tokenizer_output["token_type_ids"].squeeze(0)

    if len(input_ids) > max_length:
        return None
    
    # Check if token_type_ids already has 0 for topic and 1 for text, if not, create them manually
    if not (token_type_ids == 0).all() or not (token_type_ids == 1).all():
        sep_index = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()
        token_type_ids = torch.cat([
            torch.zeros(sep_index + 1, dtype=torch.long),  # +1 to include SEP token in topic segment
            torch.ones(len(input_ids) - (sep_index + 1), dtype=torch.long),
        ])

    sample = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offsets,
    }
    return sample


def extract_spans(predictions: list[int], offset_mapping: list[tuple[int, int]], gap_tolerance: int=1):
    result = []
    start_char, end_char = None, None
    gap_count = 0

    for pred, (offset_start, offset_end) in zip(predictions, offset_mapping[-len(predictions)-1:-1], strict=True):
        if pred == 1:
            if start_char is None:
                start_char = offset_start
                end_char = offset_end
                gap_count = 0
            else:
                if gap_count > 0:
                    end_char = offset_end
                    gap_count = 0
                else:
                    end_char = offset_end
        else:
            if start_char is not None:
                gap_count += 1
                if gap_count > gap_tolerance:
                    result.append((start_char, end_char))
                    start_char, end_char = None, None
                    gap_count = 0

    if start_char is not None:
        result.append((start_char, end_char))

    return result
    

if __name__ == "__main__":
    chunk = "Topic name - This is! a sample chunk of text."
    words = split_chunk_into_words(chunk)
    print(words)