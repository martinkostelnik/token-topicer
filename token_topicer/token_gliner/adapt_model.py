import argparse
import transformers


def parse_args():
    parser = argparse.ArgumentParser(description="Adapt a language model for Gliner by adding a special token and resizing model embeddings.")

    parser.add_argument("--lm", type=str, required=True, help="The name or path of the pre-trained language model to adapt.")
    parser.add_argument("--output", type=str, required=True, help="The path to save the adapted model.")

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.lm)
    added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": ["[ENT]"]})
    print(f"Added {added_tokens} tokens to the tokenizer.")

    model = transformers.AutoModelForCausalLM.from_pretrained(args.lm)
    model.resize_token_embeddings(len(tokenizer))

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

if __name__ == "__main__":
    main()
