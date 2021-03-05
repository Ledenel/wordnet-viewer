import os
if __name__ == "__main__" or not os.path.exists(os.path.expanduser("~/nltk_data/corpora/wordnet")):
    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("wordnet")
    nltk.download("omw")