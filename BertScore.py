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
print(outputs.encoder_attentions)
