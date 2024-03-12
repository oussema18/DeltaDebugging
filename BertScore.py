from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
input_ids = tokenizer(
    "summarize: studies have shown that owning a dog is good for you ",
    return_tensors="pt",
).input_ids
decoder_input_ids = tokenizer(
    "<pad>", add_special_tokens=False, return_tensors="pt"
).input_ids
model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
outputs = model(
    input_ids=input_ids, decoder_input_ids=decoder_input_ids, output_attentions=True
)
print(outputs.encoder_attentions)
