import random
import string
from tree_sitter_languages import get_language, get_parser
import json


# Function to extract and modify function names in the parsed tree
def modify_function_name(code, new_function_name):
    language = get_language("python")
    parser = get_parser("python")
    tree = parser.parse(bytes(code, "utf8"))
    node = tree.root_node.children[0]

    if node.type == "function_definition":
        function_name_node = node.children[1]
        function_name = code[
            function_name_node.start_byte : function_name_node.end_byte
        ]

        # Modify the function name
        modified_code = (
            code[: function_name_node.start_byte]
            + new_function_name
            + code[function_name_node.end_byte :]
        )

        return modified_code


def get_function_name(code):
    language = get_language("python")
    parser = get_parser("python")
    tree = parser.parse(bytes(code, "utf8"))
    node = tree.root_node.children[0]
    if node.type == "function_definition":
        function_name_node = node.children[1]
        function_name = code[
            function_name_node.start_byte : function_name_node.end_byte
        ]
        return function_name


def extract_variables_names(code):
    parser = get_parser("python")
    tree = parser.parse(bytes(code, "utf8"))
    # Get the root node of the syntax tree
    root_node = tree.root_node
    return " ".join([c for c in list(set(extract_variables(root_node, code)))])


# Function to extract variable names
def extract_variables(node, code):
    if (node.type == "identifier") and (
        node.parent.type == "assignment"
        or node.parent.type == "parameters"
        or node.parent.type == "for_statement"
        or node.parent.type == "default_parameter"
        or node.parent.type == "argument_list"
    ):
        return [code[node.start_byte : node.end_byte]]
    variable_nodes = []
    for child in node.children:
        variable_nodes.extend(extract_variables(child, code))
    return variable_nodes


print(
    extract_variables_names(
        """def sina_xml_to_url_list(xml_data):
    
    rawurl = []
    dom = parseString(xml_data)
    for node in dom.getElementsByTagName('durl'):
        url = node.getElementsByTagName('url')[0]
        rawurl.append(url.childNodes[0].data)
    return rawurl"""
    )
)
