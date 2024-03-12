from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codeT5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codeT5-base")

# Encode the input text
input_text = "def get_vid_from_url(url):\n        \"\"\"Extracts video ID from URL.\n        \"\"\"\n        return match1(url, r'youtu\\.be/([^?/]+)') or \\\n          match1(url, r'youtube\\.com/embed/([^/?]+)') or \\\n          match1(url, r'youtube\\.com/v/([^/?]+)') or \\\n          match1(url, r'youtube\\.com/watch/([^/?]+)') or \\\n          parse_query_param(url, 'v') or \\\n          parse_query_param(parse_query_param(url, 'u'), 'v')"
input_dict = tokenizer(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512,
)

# T5 uses a different token for decoding, often you can just set it to pad_token to initiate the output
decoder_input_ids = tokenizer("<pad>", return_tensors="pt").input_ids

# Run the model and get the attention scores
outputs = model(
    input_ids=input_dict["input_ids"],
    attention_mask=input_dict["attention_mask"],
    decoder_input_ids=decoder_input_ids,
    output_attentions=True,
)
attentions = outputs.attentions

# Attentions is a tuple where each item represents a layer.
# Let's assume you want to look at the last layer's attentions
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
top_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].squeeze().tolist())

# Take the top N tokens
N = 10  # or the size of your reduced token set
top_N_tokens = [top_tokens[i] for i in top_attention_indices[:N]]

print(top_N_tokens)
