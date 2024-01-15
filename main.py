import DD
import helper as hp
import json
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from treeSitter import extract_variables_names, get_function_name, modify_function_name

nltk.download("punkt")
###############################################################

g_model = None
g_original_method_name = None
g_predicted_method_name = None
g_all_data = []
g_cnt_dict = {}
g_cnt_pass = [0, 0, 0]

###############################################################


class MyDD(DD.DD):
    def __init__(self):
        DD.DD.__init__(self)

    def _test(self, _deltas):
        if not _deltas:
            return self.PASS

        try:
            g_cnt_pass[0] = g_cnt_pass[0] + 1
            _code = hp.deltas_to_code(_deltas)
            if hp.is_parsable(_code):
                g_cnt_pass[1] = g_cnt_pass[1] + 1
                _predict, _score, _loss = hp.prediction_with_M(g_model, _code)
                _time = hp.get_current_time()
                if _predict == g_predicted_method_name:
                    g_cnt_pass[2] = g_cnt_pass[2] + 1
                    _data = hp.get_json_data(
                        _time, _score, _loss, _code, _deltas[:], g_cnt_pass
                    )
                    g_all_data.append("{}".format(_data))
                    return self.FAIL
        except Exception:
            pass

        return self.PASS


if __name__ == "__main__":
    with tf.device("/device:GPU:0"):
        g_model = hp.load_model_M()
        assert g_model is not None
        # Read the JSONL file
        jsonl_file_path = "python_test_0.jsonl"
        i = 0
        with open(jsonl_file_path, "r") as file:
            for line in file:
                if i <= 1000:
                    print(
                        "======================Function execution i = ",
                        i,
                        "======================",
                    )
                    # Parse the JSON data in each line
                    data = json.loads(line)
                    # Extract the code from the desired
                    code = hp.remove_comments(data["code"])
                    summary = data["docstring"]
                    # get method_name and method_body
                    method_name = summary
                    method_body = code
                    g_cnt_dict[method_name] = g_cnt_dict.get(method_name, 0) + 1
                    # check predicted method_name
                    g_original_method_name = method_name
                    predict, score, loss = hp.prediction_with_M(g_model, method_body)
                    g_predicted_method_name = predict
                    # create deltas by char/token
                    deltas = []
                    if hp.g_deltas_type == "token":
                        deltas = hp.get_token_deltas(method_body)
                    else:
                        deltas = hp.get_char_deltas(method_body)
                    mydd = MyDD()

                    reduced_tokens = mydd.ddmin(deltas)

                    program = hp.deltas_to_code(reduced_tokens)
                    data = [
                        method_body,
                        summary,
                        predict,
                        program,
                        reduced_tokens,
                        hp.calculate_BLEU_score_strings(summary, predict),
                        hp.calculate_cosine_similarity(summary, reduced_tokens),
                        hp.calculate_cosine_similarity(
                            hp.deltas_to_code1(
                                hp.get_token_deltas(get_function_name(code))
                            ),
                            reduced_tokens,
                        ),
                        hp.calculate_cosine_similarity(
                            extract_variables_names(code), reduced_tokens
                        ),
                        hp.calculate_rouge_score(summary, predict),
                    ]

                    # Save the data to Excel
                    hp.save_to_excel(data, "codeT5+_Without_Comments.xlsx")
                    print("-----------------------------------------------------------")
                    print("====================== ORIGINAL CODE ======================")
                    print(method_body)
                    print()
                    print("=================== ORIGINAL DOCSTRING ===================")
                    print(summary)
                    print("=================== ORIGINAL PREDICTION ===================")
                    print(predict)
                    print()
                    print("====================== REDUCED CODE =======================")
                    print("\033[30;103m", program, "\033[0m")
                    print("======================= BLUE SCORE ========================")
                    print(
                        "BLEU between Commments and Reduced code : ",
                        hp.calculate_BLEU_score_strings(summary, predict),
                    )
                    print(
                        "Cosine between Prediction and Comments :",
                        hp.calculate_cosine_similarity(summary, reduced_tokens),
                    )
                    print(reduced_tokens)
                    print("-----------------------------------------------------------")
                    # print("Removing any element will make the prediction go away.")
                    g_all_data.append("\nMinimal simplified code:\n{}".format(program))
                i = i + 1
