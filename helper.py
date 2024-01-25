import re
import pandas as pd
import subprocess
import javalang
from datetime import datetime
import json
import nltk
from rouge_score import rouge_scorer

from argparse import ArgumentParser
from dd_model import Model
from config import Config
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
from transformers import RobertaTokenizer, AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PLBartForConditionalGeneration, PLBartTokenizer

###############################################################

g_deltas_types = ["token", "char"]
g_simp_file = "data/tmp/sm_test.java"
JAR_LOAD_JAVA_METHOD = "others/LoadJavaMethod/target/jar/LoadJavaMethod.jar"

# TODO - update file_path and delta_type
g_test_file = "data/selected_file/mn_c2x/c2x_jl_test_correct_prediction_samefile.txt"
g_deltas_type = g_deltas_types[0]
device = "cuda"  # for GPU usage or "cpu" for CPU usage
###############################################################


def replace_line_breaks(code):
    # Replace line breaks with "NEW_LINE_INDENT"
    code_with_new_line_indent = code.replace("\n", "NEW_LINE_INDENT")

    # Handle the indentation for the first line
    if code_with_new_line_indent.startswith("NEW_LINE_INDENT"):
        code_with_new_line_indent = code_with_new_line_indent[len("NEW_LINE_INDENT") :]

    return code_with_new_line_indent


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


def deltas_to_code1(d):
    return " ".join([c[1] for c in d])


def calculate_rouge_score(reference, candidate):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores["rougeL"][2]


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


def save_to_excel(data, filename):
    columns = [
        "Original Code",
        "Comments",
        "Original Prediction",
        "Reduced Code",
        "Reduced Code Tokens",
        "BLEU Score Comments to Prediction",
        "Cosine Score Comments to Reduced Code",
        "Cosine Score Function Name to Reduced Code",
        "Cosine Score Variables Name to Reduced Code",
        "ROUGER Score",
    ]

    try:
        # Load existing data if the file exists
        df = pd.read_excel(filename)
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame(columns=columns)

    # Append new data to the DataFrame
    df = df._append(pd.DataFrame([data], columns=columns), ignore_index=True)

    # Save the updated DataFrame to Excel
    df.to_excel(filename, index=False)


def get_file_list():
    file_list = []
    try:
        df = pd.read_csv(g_test_file)
        file_list = df["path"].tolist()[:1000]
    except Exception:
        pass
    return file_list


def get_current_time():
    return str(datetime.now())


def get_char_deltas(program):
    data = list(program)  # ['a',...,'z']
    deltas = list(zip(range(len(data)), data))  # [('a',0), ..., ('z',n)]
    return deltas


def get_token_deltas(program):
    token, tokens = "", []
    for c in program:
        if not c.isalpha():
            if not (
                (token == "NEW" and c == "_") or (token == "NEW_LINE" and c == "_")
            ):
                tokens.append(token)
                tokens.append(c)
                token = ""
            else:
                token = token + c
        else:
            token = token + c
    tokens.append(token)
    tokens = [token for token in tokens if len(token) != 0]
    deltas = list(zip(range(len(tokens)), tokens))
    return deltas


def deltas_to_code(d):
    return "".join([c[1] for c in d])


def is_parsable(code):
    return True


def get_json_data(time, score, loss, code, tokens=None, n_pass=None):
    score, loss = str(round(float(score), 4)), str(round(float(loss), 4))
    data = {"time": time, "score": score, "loss": loss, "code": code}
    if tokens:
        data["n_tokens"] = len(tokens)
    if n_pass:
        data["n_pass"] = n_pass
    j_data = json.dumps(data)
    return j_data


###############################################################


def load_model_M(model_path=""):
    return PLBartForConditionalGeneration.from_pretrained(
        "uclanlp/plbart-python-en_XX"
    ).to(device)


def prediction_with_M(model, code):
    pred, score, loss = None, 0, 0
    tokenizer = PLBartTokenizer.from_pretrained(
        "uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX"
    )
    inputs = tokenizer(replace_line_breaks(code), return_tensors="pt").to(device)
    translated_tokens = model.generate(
        **inputs, decoder_start_token_id=tokenizer.lang_code_to_id["__en_XX__"]
    ).to(device)
    pred = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return pred, score, loss


def find_max_score_token(tokens_data):
    max_score = float("-inf")
    max_token_str = None

    for token_data in tokens_data:
        if "score" in token_data and "token_str" in token_data:
            score = token_data["score"]
            token_str = token_data["token_str"]

            if score > max_score:
                max_score = score
                max_token_str = token_str

    return max_token_str, max_score


###############################################################


def load_method(file_path):
    try:
        # Example: extract name and body from method of JAVA program.
        cmd = ["java", "-jar", JAR_LOAD_JAVA_METHOD, file_path]
        contents = subprocess.check_output(cmd, encoding="utf-8", close_fds=True)
        contents = contents.split()
        method_name = contents[0]
        method_body = " ".join(contents[1:])
        return method_name, method_body
    except Exception:
        return "", ""


def store_method(sm_file, method_body):
    with open(sm_file, "w") as f:
        f.write(method_body + "\n")


def save_simplified_code(all_methods, output_file):
    open(output_file, "w").close()
    with open(output_file, "a") as f:
        for jCode in all_methods:
            print(jCode)
            f.write(jCode + "\n")
        f.write("\n")
