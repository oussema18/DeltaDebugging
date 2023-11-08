import pandas as pd
import subprocess
import javalang
from datetime import datetime
import json
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from dd_model import Model
from config import Config
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline

###############################################################

g_deltas_types = ["token", "char"]
g_simp_file = "data/tmp/sm_test.java"
JAR_LOAD_JAVA_METHOD = "others/LoadJavaMethod/target/jar/LoadJavaMethod.jar"

# TODO - update file_path and delta_type
g_test_file = "data/selected_file/mn_c2x/c2x_jl_test_correct_prediction_samefile.txt"
g_deltas_type = g_deltas_types[0]

###############################################################


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
            tokens.append(token)
            tokens.append(c)
            token = ""
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
    model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    return model


def prediction_with_M(model, code):
    pred, score, loss = None, None, 0
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    predictions = fill_mask(code)
    pred, score = find_max_score_token(predictions)
    print(pred)
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
