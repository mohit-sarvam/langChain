from langchain.document_loaders import TextLoader


def get_documentation():
    docs = TextLoader("/Users/mohit/Documents/sarvam/langChain/docs/documentation.txt")
    return docs.load()
