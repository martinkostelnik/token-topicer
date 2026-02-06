import re

import transformers


def split_chunk_into_words(chunk: str) -> list[str]:
    pattern = r"\w+(?:'\w+)?|[^\w\s]"
    words = re.findall(pattern, chunk, re.UNICODE)
    return words


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
        max_length=512,
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