import DD
import helper as hp
import json
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

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
            # c [(0, 'for'), (1, ' '), (2, 'i'), (3, ' '), (4, '<'), (5, 'mask'), (6, '>'), (7, ' ')]
            g_cnt_pass[0] = g_cnt_pass[0] + 1
            _code = hp.deltas_to_code(_deltas)
            if hp.is_parsable(_code):
                g_cnt_pass[1] = g_cnt_pass[1] + 1
                _predict, _score, _loss = hp.prediction_with_M(g_model, _code)
                _time = hp.get_current_time()
                """ print(
                    "time = {}, predict = {}, score = {}, loss = {}".format(
                        _time, _predict, _score, _loss
                    )
                )"""
                # _predict = ','
                # g_predicted_method_name = ','
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


def remove_comments(input_string):
    # Define a regular expression pattern to match comments between triple double-quotes
    pattern = r'"""(.*?)"""'

    # Use re.sub to replace the matched pattern with an empty string
    cleaned_string = re.sub(pattern, "", input_string, flags=re.DOTALL)

    return cleaned_string


def count(c, remaining_tokens):
    c_tokens = [token for _, token in c]
    for token in c_tokens:
        if token in remaining_tokens:
            remaining_tokens[token] += 1
        else:
            remaining_tokens[token] = 1
    # Sort tokens by their occurrences
    sorted_tokens = sorted(remaining_tokens.items(), key=lambda x: x[1], reverse=True)

    # Print the output as a table
    print("| Token                 | Occurrence |")
    print("|-----------------------|------------|")
    for token, occurrence in sorted_tokens:
        if token == "\n":
            token = "\\n"
        print(f"| {token.ljust(20)} | {str(occurrence).ljust(10)} |")
    return remaining_tokens


def format_tokens(tokens):
    # Group the elements by their positions
    groups = {}
    for pos, value in tokens:
        groups.setdefault(pos, []).append(value)

    # Merge consecutive elements
    merged = []
    for key in sorted(groups.keys()):
        if groups[key] != [" "]:
            merged.append("".join(groups[key]))
    return merged


def calculate_cosine_similarity(reference, candidate):
    # Tokenize the texts
    tokenizer = nltk.word_tokenize
    reference_tokens = tokenizer(reference.lower())
    candidate_tokens = format_tokens(candidate)

    # Join the token lists back into strings for CountVectorizer
    reference_str = " ".join(reference_tokens)
    candidate_str = " ".join(candidate_tokens)

    # Create CountVectorizer and fit the reference and candidate texts
    vectorizer = CountVectorizer().fit([reference_str, candidate_str])

    # Transform the texts to their vector representations
    reference_vector = vectorizer.transform([reference_str])
    candidate_vector = vectorizer.transform([candidate_str])

    # Compute cosine similarity between the vectors
    similarity_score = cosine_similarity(reference_vector, candidate_vector)[0, 0]
    return similarity_score


def calculate_BLEU_score(comment, pred_tokens):
    # Sample reference and candidate sentences
    reference_tokens = nltk.word_tokenize(comment)
    candidate_tokens = format_tokens(pred_tokens)
    print("reference tokens : ", reference_tokens)
    print("candidate tokens : ", candidate_tokens)
    # Computing BLEU score with smoothing
    bleu_score = nltk.translate.bleu_score.sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1,
    )
    return bleu_score


def calculate_BLEU_score_strings(reference, candidate):
    # Tokenizing the sentences
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())

    # Computing BLEU score with smoothing
    bleu_score = nltk.translate.bleu_score.sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1,
    )
    return bleu_score


def save_to_excel(data):
    columns = [
        "Original Code",
        "Comments",
        "Original Prediction",
        "Reduced Code",
        "Reduced Code Tokens",
        "BLEU Score Comments to Prediction",
        "Cosine Score Comments to Reduced Code",
    ]

    try:
        # Load existing data if the file exists
        df = pd.read_excel("metricsAfterRemovingComments.xlsx")
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame(columns=columns)

    # Append new data to the DataFrame
    df = df._append(pd.DataFrame([data], columns=columns), ignore_index=True)

    # Save the updated DataFrame to Excel
    df.to_excel("metricsAfterRemovingComments.xlsx", index=False)


if __name__ == "__main__":
    with tf.device("/device:GPU:0"):
        g_model = hp.load_model_M()
        assert g_model is not None
        # Read the JSONL file
        jsonl_file_path = "python_test_0.jsonl"
        i = 0
        remaining_tokens = {}
        deleted_tokens = {}
        with open(jsonl_file_path, "r") as file:
            # Iterate over each line in the file
            line = file.readline()
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
                    code = remove_comments(data["code"])
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
                        calculate_BLEU_score_strings(summary, predict),
                        calculate_cosine_similarity(summary, reduced_tokens),
                    ]

                    # Save the data to Excel
                    save_to_excel(data)
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
                        calculate_BLEU_score_strings(summary, predict),
                    )
                    print(
                        "Cosine between Prediction and Comments :",
                        calculate_cosine_similarity(summary, reduced_tokens),
                    )
                    print("-----------------------------------------------------------")
                    # print("Removing any element will make the prediction go away.")
                    g_all_data.append("\nMinimal simplified code:\n{}".format(program))
                i = i + 1
