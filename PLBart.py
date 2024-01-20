from transformers import PLBartForConditionalGeneration, PLBartTokenizer

tokenizer = PLBartTokenizer.from_pretrained(
    "uclanlp/plbart-base", src_lang="python", tgt_lang="en_XX"
)
example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
inputs = tokenizer(example_python_phrase, return_tensors="pt")
model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-base")
translated_tokens = model.generate(**inputs)
print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
