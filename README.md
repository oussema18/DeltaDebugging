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
in the MyDD.py file, you would set the method_name and the method_body variables as the hidden token (mask) and the code without the hidden token  
    >method_name, method_body = "i", "for i in range(enumerate(j)) : print(<mask>)"  







