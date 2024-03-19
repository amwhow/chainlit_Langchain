from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path='test.json',
    jq_schema='.messages[].content',
    text_content=False)

data = loader.load()

print("data: ", data)