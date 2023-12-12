from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk


def calculate_cosine_similarity(reference, candidate):
    # Tokenize the texts
    tokenizer = nltk.word_tokenize
    reference_tokens = tokenizer(reference.lower())
    candidate_tokens = tokenizer(candidate.lower())

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


reference = """get _ vid _ from _ url"""
candidate = """ vid from _ url ( url )
    com / embed / ' )"""
print(calculate_cosine_similarity(reference, candidate))
print(calculate_BLEU_score_strings(reference, candidate))
