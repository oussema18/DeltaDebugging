from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codeT5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codeT5-small")
input_ids = tokenizer(
    "def get_vid_from_url(url):\n        \"\"\"Extracts video ID from URL.\n        \"\"\"\n        return match1(url, r'youtu\\.be/([^?/]+)') or \\\n          match1(url, r'youtube\\.com/embed/([^/?]+)') or \\\n          match1(url, r'youtube\\.com/v/([^/?]+)') or \\\n          match1(url, r'youtube\\.com/watch/([^/?]+)') or \\\n          parse_query_param(url, 'v') or \\\n          parse_query_param(parse_query_param(url, 'u'), 'v')",
    return_tensors="pt",
).input_ids
decoder_input_ids = tokenizer(
    "<pad>", add_special_tokens=False, return_tensors="pt"
).input_ids
outputs = model(
    input_ids=input_ids, decoder_input_ids=decoder_input_ids, output_attentions=True
)
attentions = outputs.encoder_attentions
last_layer_attentions = attentions[
    -1
]  # (1, num_heads, sequence_length, sequence_length)

# To get a single attention score per token, average across the attention heads
attention_scores = last_layer_attentions.mean(dim=1).squeeze(0)

# Now let's sort the tokens by their attention score and take the top N.
# Note that the scores are a matrix of (sequence_length, sequence_length)
# We are interested in the attention each token gives to the [CLS] token (or equivalent in T5)
cls_attentions = attention_scores[:, 0]

# Sort the tokens by attention score in descending order and take the indices
top_attention_indices = cls_attentions.argsort(descending=True)

# Decode the top tokens
top_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

# Take the top N tokens
N = 10  # or the size of your reduced token set
top_N_tokens = [top_tokens[i] for i in top_attention_indices[:N]]

print(top_N_tokens)
