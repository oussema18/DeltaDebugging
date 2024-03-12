from transformers import AutoTokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codeT5-base")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codeT5-base")

# Encode the input text
input_text = "your code snippet here"
input_dict = tokenizer(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512,
)

# For T5, decoder_input_ids are often created by shifting the input_ids to the right and adding the pad token at the beginning
decoder_start_token_id = model.config.decoder_start_token_id
input_dict["decoder_input_ids"] = model._shift_right(input_dict["input_ids"])

# Run the model and get the attention scores
outputs = model(**input_dict, output_attentions=True)

# Check if 'attentions' is in the outputs
if "attentions" in outputs:
    attentions = outputs.attentions
else:
    raise ValueError(
        "No attentions were found. Ensure that `output_attentions=True` is set."
    )

# If attentions are found, you can proceed to process them
