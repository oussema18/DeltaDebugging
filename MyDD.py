import DD
import helper as hp

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
                print(
                    "time = {}, predict = {}, score = {}, loss = {}".format(
                        _time, _predict, _score, _loss
                    )
                )
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


"""
def delta_debugging(deltas, model, pred):
    minimal_deltas=[]
    for i in range(len(deltas)):
        
    return None
"""

if __name__ == "__main__":
    g_model = hp.load_model_M()
    assert g_model is not None

    # get method_name and method_body
    method_name, method_body = "i", "for i in range(enumerate(j)) : print(<mask>)"
    assert (len(method_name) > 0) and (len(method_body) > 0)
    g_cnt_dict[method_name] = g_cnt_dict.get(method_name, 0) + 1
    g_all_data.append("method_name = {}".format(method_name))
    g_all_data.append("method_body = {}".format(method_body))

    # check predicted method_name
    g_original_method_name = method_name
    predict, score, loss = hp.prediction_with_M(g_model, method_body)
    g_predicted_method_name = predict
    g_all_data.append("predict, score, loss = {}, {}, {}".format(predict, score, loss))

    # create deltas by char/token
    deltas = []
    if hp.g_deltas_type == "token":
        deltas = hp.get_token_deltas(method_body)
    else:
        deltas = hp.get_char_deltas(method_body)

    # run ddmin to simplify program

    mydd = MyDD()
    print("Simplifying prediction-preserving input...")
    g_all_data.append("\nTrace of simplified code(s):")
    c = mydd.ddmin(deltas)
    print("The 1-minimal prediction-preserving input is", c)
    print("Removing any element will make the prediction go away.")
    program = hp.deltas_to_code(c)
    g_all_data.append("\nMinimal simplified code:\n{}".format(program))
    print(program)
