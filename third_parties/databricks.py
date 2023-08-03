from langchain.document_loaders import TextLoader


def get_documentation(name: str):
    docs = TextLoader("/Users/mohit/Documents/sarvam/langChain/docs/documentation.txt")
    content = []
    for doc in docs.load():
        content = doc.page_content
    return content
