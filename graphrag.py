"""
Graph-RAG with LLMGraphTransformer & LCEL (Azure OpenAI Edition)
===============================================================
Azure OpenAI を用いた Graph-RAG 実装。
最新の LangChain API の変更に対応し、ImportError や TypeError を解消した安定版を目指しています。

主な機能:
- `LLMGraphTransformer` によるドキュメントからのグラフ抽出
- `Neo4jGraph` へのグラフデータの格納 (langchain-neo4j パッケージ利用)
- `PGVector` へのチャンクデータの格納とベクトル検索
- `ParentDocumentRetriever` の利用 (オプション、InMemoryStoreを使用)
- `LCEL` (LangChain Expression Language) を用いた柔軟なチェーン構築
- Azure OpenAI の利用を前提とした設定

必要なライブラリ (pip install ...):
- langchain, langchain-openai, langchain-community, langchain-postgres
- langchain-experimental, langchain-neo4j (Neo4jGraph のため)
- neo4j, tiktoken, python-dotenv, "psycopg[binary]"

.env ファイルの例:
------------------------------------
# --- Azure OpenAI ---
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_API_VERSION=2024-04-01-preview # 例: 2023-05-15, 2024-02-15-preview など
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini # ご自身のデプロイ名
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small # ご自身のデプロイ名

# --- Neo4j Aura (またはローカル) ---
NEO4J_URI=neo4j+s://your-aura-instance.databases.neo4j.io # 例: neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PW=your_neo4j_pw

# --- PGVector (PostgreSQL) ---
PG_CONN=postgresql+psycopg://postgres:your_pw@localhost:5432/graph_rag_db # ご自身の接続文字列
------------------------------------

実行方法:
python graph_rag_script_name.py  # input.txt をスクリプトと同じ階層に配置
                                # (graph_rag_script_name.py はこのファイル名に置き換えてください)
"""
from __future__ import annotations

import os
import traceback # エラー詳細表示用
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# LangChain / Azure OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Neo4jGraph (langchain-neo4j パッケージから)
try:
    from langchain_neo4j import Neo4jGraph
    print("Info: langchain_neo4j.Neo4jGraph をインポートしました。")
except ImportError:
    print("Error: langchain_neo4j.Neo4jGraph のインポートに失敗しました。")
    print("       'pip install -U langchain-neo4j' を実行してください。")
    Neo4jGraph = None # 後続処理でエラーチェックできるようにする

from langchain_community.vectorstores.pgvector import PGVector

# GraphDocument のインポート (最新のパスを優先)
try:
    from langchain_core.graph_document import GraphDocument
except ImportError:
    try:
        from langchain_community.graphs.graph_document import GraphDocument
    except ImportError:
        # 古いバージョンや予期せぬ配置へのフォールバック (型チェックは無視)
        from langchain_community.graphs import GraphDocument # type: ignore
        print("Warning: GraphDocument を langchain_community.graphs からインポートしました。パスが古い可能性があります。")

# Neo4jGraph を直接操作する GraphRetriever のインポート試行
GRAPH_RETRIEVER_CLASS = None
try:
    from langchain_community.retrievers.graph import GraphRetriever as Neo4jGraphRetriever
    GRAPH_RETRIEVER_CLASS = Neo4jGraphRetriever
    print("Info: langchain_community.retrievers.graph.GraphRetriever を Neo4jGraphRetriever としてインポートしました。")
except ImportError:
    print("Warning: langchain_community.retrievers.graph.GraphRetriever が見つかりませんでした。Graphベースの検索は限定的になります。")

# ParentDocumentRetriever と InMemoryStore のインポート試行
HAS_PARENT_DOCUMENT_RETRIEVER = False
ParentDocumentRetriever = None
InMemoryStore = None
try:
    # langchain パッケージの retriever と storage を優先的に試す
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore
    HAS_PARENT_DOCUMENT_RETRIEVER = True
    print("Info: langchain.retrievers.ParentDocumentRetriever と langchain.storage.InMemoryStore をインポートしました。")
except ImportError:
    try:
        # langchain_community からのフォールバック
        from langchain_community.retrievers import ParentDocumentRetriever
        from langchain_community.storage import InMemoryStore
        HAS_PARENT_DOCUMENT_RETRIEVER = True
        print("Info: langchain_community.retrievers.ParentDocumentRetriever と langchain_community.storage.InMemoryStore をインポートしました。")
    except ImportError:
        print("Warning: ParentDocumentRetriever または InMemoryStore が見つかりませんでした。標準のベクトルリトリーバーを使用します。")


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableConfig # 必要に応じて型ヒントで使用
)
from langchain_core.documents import Document

# --- 環境変数の読み込み ---
load_dotenv()

# Azure OpenAI
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini") # デフォルト値
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small") # デフォルト値

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j") # デフォルト値
NEO4J_PW = os.getenv("NEO4J_PW")

# PGVector
PG_CONN = os.getenv("PG_CONN")

# 環境変数の必須チェック
required_env_vars = {
    "AZURE_OPENAI_ENDPOINT": AZURE_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_VERSION,
    "PG_CONN": PG_CONN,
}
# Neo4jGraphが正常にインポートされた場合のみ、Neo4j関連の環境変数を必須とする
if Neo4jGraph is not None:
    required_env_vars["NEO4J_URI"] = NEO4J_URI
    required_env_vars["NEO4J_PW"] = NEO4J_PW

missing_vars = [key for key, value in required_env_vars.items() if value is None]
if missing_vars:
    raise ValueError(f"以下の環境変数が設定されていません: {', '.join(missing_vars)}. .envファイルを確認してください。")

# --- 0. ドキュメント読み込み ---
DOC_PATH_STR = "input.txt"
DOC_PATH = Path(DOC_PATH_STR)
if not DOC_PATH.is_file():
    raise FileNotFoundError(
        f"入力ファイル '{DOC_PATH_STR}' が見つかりません。"
        f"スクリプトと同じ階層に配置してください。"
    )
try:
    raw_text = DOC_PATH.read_text(encoding="utf-8")
    print(f"Info: 入力ファイル '{DOC_PATH_STR}' を読み込みました ({len(raw_text)}文字)。")
except Exception as e:
    raise IOError(f"ファイル '{DOC_PATH_STR}' の読み込みに失敗しました: {e}")


# --- 1. チャンク分割 (SemanticChunker) ---
print("Info: 埋め込みモデル (Embeddings) を初期化しています...")
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_key=AZURE_KEY,
    api_version=AZURE_VERSION,
    azure_deployment=AZURE_EMBED_DEPLOYMENT,
)

print("Info: テキストをチャンクに分割しています (SemanticChunker)...")
# SemanticChunkerのパラメータは、テキストの性質や実験によって調整してください。
# breakpoint_threshold_type はチャンク分割の閾値の決定方法を指定します。
text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
documents: List[Document] = []
if raw_text.strip(): # 空でないテキストの場合のみ分割
    documents = text_splitter.create_documents([raw_text])
    print(f"Info: {len(documents)} 個のチャンクが作成されました。")
else:
    print("Warning: 入力テキストが空または空白のみのため、チャンク分割をスキップします。")

# --- 2. LLMGraphTransformer で GraphDocument 化 ---
print("Info: チャットモデル (グラフ抽出用 LLM) を初期化しています...")
llm_graph_extraction = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_key=AZURE_KEY,
    api_version=AZURE_VERSION,
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    temperature=0, # グラフ抽出では決定性を高めるため temperature は低めが一般的
)

graph_documents: List[GraphDocument] = []
if documents: # チャンクが存在する場合のみグラフ抽出を実行
    print("Info: LLMGraphTransformer を使用してチャンクからグラフドキュメントを抽出しています...")
    # LLMGraphTransformer は Document オブジェクトのリストを受け取ります
    # allowed_nodes や allowed_relationships で抽出するエンティティや関係のタイプを制限可能
    transformer = LLMGraphTransformer(llm=llm_graph_extraction)
    try:
        graph_documents = transformer.convert_to_graph_documents(documents)
        print(f"Info: {len(graph_documents)} 個のグラフドキュメントが抽出されました。")
    except Exception as e:
        print(f"Error: LLMGraphTransformer でのグラフドキュメント抽出に失敗しました: {e}")
        print(traceback.format_exc())
else:
    print("Warning: 分割されたチャンクが空のため、グラフドキュメントの抽出をスキップします。")


# --- 3. Neo4j にロード ---
graph_db: Optional[Neo4jGraph] = None
if Neo4jGraph and graph_documents: # Neo4jGraphが利用可能で、グラフドキュメントが存在する場合
    print(f"Info: Neo4j ({NEO4J_URI}) に接続し、グラフドキュメントをロードしています...")
    try:
        graph_db = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PW)
        graph_db.add_graph_documents(graph_documents, include_source=True, base_entity_label=True)
        # base_entity_label=True は、全てのノードに共通のラベル (例: `__Entity__`) を追加します。
        # インデックス作成の推奨:
        # LLMGraphTransformerが生成するノードのプロパティ (通常 'id') やラベルにインデックスを作成するとパフォーマンスが向上します。
        # 例: CREATE INDEX entity_id IF NOT EXISTS FOR (n:__Entity__) ON (n.id)
        #     CREATE INDEX document_id IF NOT EXISTS FOR (n:Document) ON (n.id) (もしDocumentノードもあれば)
        print("Info: Neo4j にグラフドキュメントをロードしました。")
        print("Info: パフォーマンス向上のため、Neo4j Browser でノードの 'id' プロパティ等にインデックスを作成することを推奨します。")
    except Exception as e:
        print(f"Error: Neo4j への接続またはデータロードに失敗しました: {e}")
        print(traceback.format_exc())
        graph_db = None # エラー発生時は None に戻す
elif not Neo4jGraph:
    print("Warning: Neo4jGraph がインポートされていないため、Neo4jへのロードをスキップします。")
elif not graph_documents:
    print("Warning: 抽出されたグラフドキュメントが空のため、Neo4jへのロードをスキップします。")


# --- 4. PGVector にチャンク保存 ---
vector_store: Optional[PGVector] = None
if documents: # チャンクが存在する場合のみPGVectorに保存
    print(f"Info: PGVector ({PG_CONN}) に接続し、チャンクを保存しています...")
    try:
        # チャンクにユニークなIDを付与 (ParentDocumentRetrieverでdoc_idとして使用するため)
        # また、元のテキストソース情報などもメタデータに含めると良い
        for i, doc in enumerate(documents):
            doc.metadata["doc_id"] = f"chunk_{DOC_PATH_STR}_{i}" # ファイル名を含めてよりユニークに
            doc.metadata["source_file"] = DOC_PATH_STR

        vector_store = PGVector.from_documents(
            documents=documents, # LLMGraphTransformer に渡した元のチャンク (Document オブジェクト)
            embedding=embeddings,
            connection_string=PG_CONN,
            collection_name="graph_rag_chunks_collection_v2", # コレクション名を明示的に指定
            # pre_delete_collection=True # 既存のコレクションを削除して再作成する場合
        )
        print("Info: PGVector にチャンクを保存しました。")
    except Exception as e:
        print(f"Error: PGVector への接続またはデータ保存に失敗しました: {e}")
        print(traceback.format_exc())
else:
    print("Warning: 分割されたチャンクが空のため、PGVectorへの保存をスキップします。")


# --- 5. Retriever 構築 ---
# デフォルトのリトリーバーは、何も返さないダミーのRunnableLambda
# これにより、一部のコンポーネントが失敗しても、チェーン自体は実行試行可能になる
graph_retriever_runnable = RunnableLambda(lambda x: [])
vector_retriever_runnable = RunnableLambda(lambda x: [])

# Graph Retriever (Neo4j用)
if GRAPH_RETRIEVER_CLASS and graph_db:
    try:
        # Neo4jGraph インスタンスを直接使用する GraphRetriever
        # search_type="cypher" の場合、Cypherクエリを生成してグラフを検索します。
        # k は返す結果の数を制御します。
        graph_retriever_instance = GRAPH_RETRIEVER_CLASS(
            graph=graph_db,
            k=5, # 取得するグラフ要素の数 (例: 5つの関連情報)
            search_type="cypher" # または "vector" など、リトリーバーの実装に依存
                                 # "cypher" の場合、質問に基づいてCypherクエリを生成する
        )
        graph_retriever_runnable = graph_retriever_instance # Retriever自体がRunnable
        print("Info: Neo4jベースの GraphRetriever を構築しました。")
    except Exception as e:
        print(f"Warning: Neo4jベースのGraphRetrieverの構築に失敗しました: {e}. Graphベースの検索は限定的になります。")
        print(traceback.format_exc())
else:
    if not graph_db:
        print("Warning: Neo4jグラフデータベースが初期化されていないため、Neo4jベースのGraphRetrieverは構築されません。")
    elif not GRAPH_RETRIEVER_CLASS:
        print("Warning: Neo4jGraphを直接扱うGraphRetrieverクラスが見つからなかったため、Graphベースの検索は限定的になります。")


# Vector Retriever (PGVector用)
if vector_store:
    if HAS_PARENT_DOCUMENT_RETRIEVER and ParentDocumentRetriever and InMemoryStore:
        try:
            # 親ドキュメントストア (この例ではチャンク自体を親としてメモリに格納)
            byte_store = InMemoryStore()
            # 親ドキュメント (チャンク) をストアに登録
            # ParentDocumentRetriever は、子ドキュメントのメタデータに含まれる id_key を使って
            # byte_store から親ドキュメントを取得します。
            # ここでは、PGVectorに保存したチャンクの 'doc_id' メタデータを使用します。
            parent_doc_tuples = []
            valid_documents_for_pdr = [doc for doc in documents if "doc_id" in doc.metadata]
            if valid_documents_for_pdr:
                parent_doc_tuples = [(doc.metadata["doc_id"], doc) for doc in valid_documents_for_pdr]
                byte_store.mset(parent_doc_tuples)

                vector_retriever_instance = ParentDocumentRetriever(
                    vectorstore=vector_store, # 子ドキュメント(チャンク)が格納されたベクトルストア
                    byte_store=byte_store,   # 親ドキュメントが格納されたストア
                    id_key="doc_id",        # 子ドキュメントのメタデータ内で親IDを示すキー
                    search_kwargs={"k": 5},  # 最終的に返す親ドキュメントの数
                )
                vector_retriever_runnable = vector_retriever_instance
                print("Info: ParentDocumentRetriever を構築しました。")
            else:
                print("Warning: ParentDocumentRetriever のための有効なドキュメント (doc_id付き) がありません。標準リトリーバーを使用します。")
                vector_retriever_instance = vector_store.as_retriever(search_kwargs={"k": 5})
                vector_retriever_runnable = vector_retriever_instance
                print("Info: 標準ベクトルリトリーバー (フォールバック) を構築しました。")

        except Exception as e:
            print(f"Warning: ParentDocumentRetriever の構築に失敗しました: {e}. 標準ベクトルリトリーバーを使用します。")
            print(traceback.format_exc())
            vector_retriever_instance = vector_store.as_retriever(search_kwargs={"k": 5})
            vector_retriever_runnable = vector_retriever_instance
            print("Info: 標準ベクトルリトリーバー (フォールバック) を構築しました。")
    else:
        vector_retriever_instance = vector_store.as_retriever(search_kwargs={"k": 5}) # 標準リトリーバー
        vector_retriever_runnable = vector_retriever_instance
        print("Info: 標準ベクトルリトリーバーを構築しました。")
else:
    print("Warning: VectorStoreが初期化されていないため、Vector Retrieverは構築されません。")

# --- 6. LCEL チェイン定義 ---

def format_graph_results(graph_data: List[Dict[str, Any]] | Any) -> str:
    """
    Neo4j GraphRetriever (search_type="cypher") の結果を整形する関数。
    結果は辞書のリストや、単一の辞書、あるいは予期せぬ形式の場合もある。
    """
    if not graph_data:
        return "グラフ情報は取得されませんでした。"
    
    if isinstance(graph_data, list):
        if not graph_data: # 空リストの場合
             return "整形対象のグラフ情報がありません (空リスト)。"
        # 辞書のリストの場合、各辞書を文字列化して結合
        formatted_lines = [str(item) for item in graph_data]
        return "\n".join(formatted_lines)
    elif isinstance(graph_data, dict):
        # 単一の辞書の場合
        return str(graph_data)
    else:
        # その他の予期せぬ形式の場合
        return f"予期しない形式のグラフ情報: {str(graph_data)}"


def format_document_results(docs: List[Document]) -> str:
    """ドキュメントリストを整形"""
    if not docs:
        return "ドキュメントは取得されませんでした。"
    # metadataから 'doc_id' と 'source_file' を安全に取得
    return "\n---\n".join(
        [f"出典ファイル: {doc.metadata.get('source_file', 'N/A')}, "
         f"チャンクID: {doc.metadata.get('doc_id', 'N/A')}\n"
         f"内容: {doc.page_content}"
         for doc in docs]
    )

# RunnableParallel でグラフ検索とベクトル検索を並列実行し、結果をマージ
retrieve_context = RunnableParallel(
    {
        "graph_context": graph_retriever_runnable | RunnableLambda(format_graph_results),
        "document_context": vector_retriever_runnable | RunnableLambda(format_document_results),
        "question": RunnablePassthrough() # 質問をそのまま後段に渡す
    }
)

# プロンプトテンプレートの定義
prompt_template_str = """あなたは提供された情報に基づいて質問に答える専門家AIです。

以下のコンテキスト情報を利用して、質問に回答してください。

## グラフコンテキスト
{graph_context}

## ドキュメントコンテキスト
{document_context}

---
上記の情報のみを根拠として、以下の質問に対して、日本語で網羅的かつ正確に回答してください。
必要に応じて、関連する情報源（ドキュメントの出典やグラフのエンティティなど）を指摘してください。

質問: {question}
回答:
"""
prompt = PromptTemplate.from_template(prompt_template_str)

# LLM (回答生成用)
print("Info: チャットモデル (回答生成用 LLM) を初期化しています...")
llm_answer_generation = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_key=AZURE_KEY,
    api_version=AZURE_VERSION,
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    temperature=0.1, # 回答生成では少し創造性を持たせても良いが、基本は低め
)

# LCEL チェーンの構築
chain = (
    retrieve_context # 質問を受け取り、グラフとドキュメントのコンテキストを取得
    | prompt # 取得したコンテキストと質問をプロンプトに整形
    | llm_answer_generation # 整形されたプロンプトをLLMに渡し、回答を生成
    | StrOutputParser() # LLMの出力を文字列としてパース
)

# --- 7. 対話ループ ---
if __name__ == "__main__":
    print("\n--- Graph-RAG LCEL (Azure OpenAI) デモ ---")
    print("質問を入力してください ('exit', 'quit', 'q' のいずれかで終了します)。")
    print("-----------------------------------------------------------------------")
    print("注意: Neo4jからの検索結果 (グラフコンテキスト) の形式は、")
    print("      使用するGraphRetrieverの実装やCypherクエリによって異なります。")
    print("      format_graph_results 関数内の整形ロジックは、")
    print("      実際のデータ構造に合わせて調整が必要になる場合があります。")
    print("-----------------------------------------------------------------------")

    while True:
        try:
            question = input("\n質問> ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("デモを終了します。")
                break
            if not question:
                continue

            print("\n--- 回答生成中 ---")
            # chain.invoke には質問文字列を渡す
            response = chain.invoke(question)
            print("\n--- 回答 ---")
            print(response)

        except KeyboardInterrupt:
            print("\nデモを中断しました。")
            break
        except Exception as e:
            print(f"\n[エラー発生]")
            print(f"エラータイプ: {type(e).__name__}")
            print(f"エラーメッセージ: {e}")
            print("スタックトレース:", traceback.format_exc()) # デバッグ用にスタックトレースを表示
            print("環境変数、APIキー、Neo4j/PGVectorの接続、LangChainのバージョン等を確認してください。")

