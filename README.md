# Delta Debugging Algorithm for CodeBERT
based on Sivand Repo : https://github.com/mdrafiqulrabin/SIVAND/tree/5d3101f3c35b7572a3680ae899e813f4de67eb6f
## CodeBERT Model 
Some changes were applied to confirm the codeBERT Model prediction task :
   1. load_model_M() function in **helper.py** file :
      
       >def load_model_M(model_path=""):  
    	&nbsp; &nbsp;  model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")  
    	return model
   2. prediction_with_M() function in **helper.py**file :  
      	>def prediction_with_M(model, code):  
	    &nbsp; &nbsp; pred, score, loss = None, None, 0  
	    &nbsp; &nbsp; tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")  
	    &nbsp; &nbsp; fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)  
	    &nbsp; &nbsp; predictions = fill_mask(code)  
	    &nbsp; &nbsp; pred, score = find_max_score_token(predictions)  
	    return pred, score, loss  
## Usage Example

### Setting the input 
when running the codeBERT Model with this input code:  
> CODE = for i in range(enumerate(j)) : print(< mask >)

```

model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

CODE = " for i in range(enumerate(j)) : print(<mask>) "
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
outputs = fill_mask(CODE)

```

we get in fact the token **'i'** as the token with highest score :
> outputs = [  
>  &nbsp; &nbsp;{'score': 0.952255368232727,     'token': 118,   'token_str': 'i',  'sequence': 'for i in range(enumerate(j)) : print(i)'},  
>  &nbsp; &nbsp;{'score': 0.027622802183032036,  'token': 267,   'token_str': 'j',  'sequence': 'for i in range(enumerate(j)) : print(j)'},  
>  &nbsp; &nbsp;{'score': 0.0017873893957585096, 'token': 100,   'token_str': 'I',  'sequence': 'for i in range(enumerate(j)) : print(I)'},  
>  &nbsp; &nbsp;{'score': 0.0016388560179620981, 'token': 33850, 'token_str': 'jj', 'sequence': 'for i in range(enumerate(j)) : print(jj)'},  
>  &nbsp; &nbsp;{'score': 0.0008160002762451768, 'token': 330,   'token_str': 'k',  'sequence': 'for i in range(enumerate(j)) : print(k)'}  
> ]  

now we want to know which tokens can be removed so the prediction won't change and still be the token **i**.


in the MyDD.py file, you would set the method_name and the method_body variables as the hidden token (mask) and the code without the token. So for this example, we want the model to predict the token **'in'**. so we hide with the word < mask > (_for i in range(enumerate(j)): print(< mask >)_) and pass it as the  method body and i as the hidden token for the method_name :  
    ```
    method_name, method_body = "i", "for i in range(enumerate(j)) : print(<mask>)"  
    ```

### Understanding the output
in the console, the minimal tokens required to preserve the original prediction for the input code
```
dd: done
The 1-minimal prediction-preserving input is [(2, 'i'), (17, '('), (18, '<'), (19, 'mask'), (20, '>')]
Removing any element will make the prediction go away.
```
this will mean that the minimal tokens are 
> i(< mask >

### Results correctness
to verify the correctness of the outputs we would pass the **i(< mask >** code to the codeBERT Model and see if the prediction is still i : 
```

model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

CODE = "i(<mask>"
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

outputs = fill_mask(CODE)
```

Indeed the model will predict the token **i** with the highest score for **CODE = "i(< mask >"**:  
>outputs = [  
>  &nbsp; &nbsp;{'score': 0.22026327252388,'token': 118, 'token_str': 'i', 'sequence': 'i(i'},  
>  &nbsp; &nbsp;{'score': 0.027704259380698204,'token': 4839, 'token_str': ' )', 'sequence': 'i( )'},  
>  &nbsp; &nbsp;{'score': 0.025121279060840607, 'token': 1178, 'token_str': 'x', 'sequence': 'i(x'},  
>  &nbsp; &nbsp;{'score': 0.02374928630888462, 'token': 428, 'token_str': 'b', 'sequence': 'i(b'},  
>  &nbsp; &nbsp;{'score': 0.01792656071484089, 'token': 506, 'token_str': 'f', 'sequence': 'i(f'}  
>]


