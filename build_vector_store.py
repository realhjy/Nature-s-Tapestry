import os
from langchain_community.document_loaders import TextLoader, WikipediaLoader, ArxivLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def build_store():

    all_documents = []

    # 加载本地知识文件
    if os.path.exists('knowledge.txt'):
        loader_local = TextLoader('knowledge.txt', encoding='utf-8')
        all_documents.extend(loader_local.load())
        print(f"成功加载本地知识库: knowledge.txt")

    # 从维基百科加载科普知识
    try:
        print("正在从维基百科加载")
        wiki_queries = [
            "ESG",
            "生态系统",
            "生物多样性",
            "碳中和",
            "关键种",
            "生物多样性热点地区"
        ]
        for query in wiki_queries:
            try:
                print(f"正在加载维基词条: {query}")
                docs_wiki = WikipediaLoader(query=query, lang="zh", load_max_docs=1).load()
                all_documents.extend(docs_wiki)
            except Exception as e:
                print(f"加载词条“{query}”失败，已跳过。错误: {e}")
        print("维基百科专业知识加载成功！")
    except Exception as e:
        print(f"维基百科加载失败，已跳过。错误: {e}")

    # 从ArXiv加载最新的学术论文摘要
    try:
        print("正在从ArXiv学术库加载")
        arxiv_queries = [
            "ecology",
            '"environmental justice" AND "policy"',
            '"biodiversity" AND "remote sensing"',
            '"carbon footprint" AND "supply chain"'
        ]
        for query in arxiv_queries:
            try:
                print(f"正在加载ArXiv论文: {query}")
                docs_arxiv = ArxivLoader(query=query, load_max_docs=3, sort_by="relevance").load()
                all_documents.extend(docs_arxiv)
            except Exception as e:
                print(f"加载论文“{query}”失败，已跳过。错误: {e}")
        print("ArXiv学术论文加载成功！")
    except Exception as e:
        print(f"ArXiv加载失败，已跳过。错误: {e}")

    if not all_documents:
        print("错误：没有找到任何知识文件，请确保knowledge.txt存在。")
        return

    # 将所有来源的知识合并后，进行文本分割
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(all_documents)
    print(f"所有知识源被合并并分割成 {len(docs)} 个知识块")

    # 初始化Embedding模型
    print("正在初始化Embedding模型")
    model_name = "BAAI/bge-large-zh-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Embedding模型初始化成功")

    # 创建并保存最终的向量数据库
    print("正在创建FAISS索引")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")

    print("专业向量知识库构建完成!")


if __name__ == "__main__":
    build_store()
