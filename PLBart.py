from transformers import PLBartForConditionalGeneration, PLBartTokenizer


def replace_line_breaks(code):
    # Replace line breaks with "NEW_LINE_INDENT"
    code_with_new_line_indent = code.replace("\n", "NEW_LINE_INDENT")

    # Handle the indentation for the first line
    if code_with_new_line_indent.startswith("NEW_LINE_INDENT"):
        code_with_new_line_indent = code_with_new_line_indent[len("NEW_LINE_INDENT") :]

    return code_with_new_line_indent


tokenizer = PLBartTokenizer.from_pretrained(
    "uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX"
)

example_python_phrase = """"vidfromurl("Extracts video IDURL."""
inputs = tokenizer(example_python_phrase, return_tensors="pt")
model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-python-en_XX")

translated_tokens = model.generate(
    **inputs, decoder_start_token_id=tokenizer.lang_code_to_id["__en_XX__"]
)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
