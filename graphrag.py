"""
Graph-RAG with LLMGraphTransformer & LCEL (Google Gemini Edition)
=================================================================
Google Gemini を用いた Graph-RAG 実装。
Azure OpenAI 版をベースに、Gemini (langchain-google-genai) を利用するように変更。

主な機能:
- `LLMGraphTransformer` によるドキュメントからのグラフ抽出
- `Neo4jGraph` へのグラフデータの格納 (langchain-neo4j パッケージ利用)
- `PGVector` へのチャンクデータの格納とベクトル検索
- チャンク間の関連性 (:NEXT) をグラフで表現
- `ParentDocumentRetriever` の利用 (オプション、InMemoryStoreを使用)
- `LCEL` (LangChain Expression Language) を用いた柔軟なチェーン構築
- Google Gemini の利用を前提とした設定

必要なライブラリ (pip install ...):
- langchain, langchain-google-genai, langchain-community, langchain-postgres
- langchain-experimental, langchain-neo4j (Neo4jGraph のため)
- neo4j, python-dotenv, "psycopg[binary]"

.env ファイルの例:
------------------------------------
# --- Google Gemini ---
GOOGLE_API_KEY="your_google_api_key"

# --- Neo4j Aura (またはローカル) ---
NEO4J_URI=neo4j+s://your-aura-instance.databases.neo4j.io # 例: neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PW=your_neo4j_pw

# --- PGVector (PostgreSQL) ---
PG_CONN=postgresql+psycopg://postgres:your_pw@localhost:5432/graph_rag_db # ご自身の接続文字列
------------------------------------

実行方法:
python graphrag.py  # input.txt をスクリプトと同じ階層に配置
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from dotenv import load_dotenv

# LangChain / Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Neo4jGraph (langchain-neo4j パッケージから)
try:
    from langchain_neo4j import Neo4jGraph
    print("Info: langchain_neo4j.Neo4jGraph をインポートしました。")
except ImportError:
    print("Error: langchain_neo4j.Neo4jGraph のインポートに失敗しました。")
    print("       'pip install -U langchain-neo4j' を実行してください。")
    Neo4jGraph = None

# PGVector
try:
    from langchain_community.vectorstores.pgvector import PGVector
except ImportError:
    try:
        from langchain_postgres import PGVector
    except ImportError:
        print("Error: PGVector のインポートに失敗しました。")
        PGVector = None

# GraphDocument のインポート (最新のパスを優先)
try:
    from langchain_core.graph_document import GraphDocument
except ImportError:
    try:
        from langchain_community.graphs.graph_document import GraphDocument
    except ImportError:
        try:
            from langchain_experimental.graph_transformers.base import GraphDocument
        except ImportError:
            print("Error: GraphDocument をインポートできませんでした。")
            GraphDocument = None

# Neo4j QAチェーン
GraphCypherQAChain = None
try:
    from langchain_neo4j import GraphCypherQAChain
    print("Info: langchain_neo4j.GraphCypherQAChain をインポートしました。")
except ImportError:
    try:
        from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
        print("Info: langchain_community.chains.graph_qa.cypher.GraphCypherQAChain をインポートしました。")
    except ImportError:
        print("Warning: GraphCypherQAChain が見つかりませんでした。Graphベースの検索は限定的になります。")

# ParentDocumentRetriever と InMemoryStore のインポート試行
HAS_PARENT_DOCUMENT_RETRIEVER = False
ParentDocumentRetriever = None
InMemoryStore = None
try:
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore
    HAS_PARENT_DOCUMENT_RETRIEVER = True
    print("Info: langchain.retrievers.ParentDocumentRetriever と langchain.storage.InMemoryStore をインポートしました。")
except ImportError:
    try:
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
    RunnableConfig
)
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 環境変数の読み込み ---
load_dotenv()

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PW = os.getenv("NEO4J_PW")

# PGVector
PG_CONN = os.getenv("PG_CONN")

# 環境変数の必須チェック
required_env_vars = {
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "PG_CONN": PG_CONN,
}
# Neo4jGraphが正常にインポートされた場合のみ、Neo4j関連の環境変数を必須とする
if Neo4jGraph is not None:
    required_env_vars["NEO4J_URI"] = NEO4J_URI
    required_env_vars["NEO4J_PW"] = NEO4J_PW

missing_vars = [key for key, value in required_env_vars.items() if value is None]
if missing_vars:
    raise ValueError(f"以下の環境変数が設定されていません: {', '.join(missing_vars)}. .envファイルを確認してください。")

# --- グローバル変数定義 ---
graph_db: Optional[Neo4jGraph] = None
vector_store: Optional[PGVector] = None

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
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

print("Info: テキストをチャンクに分割しています (RecursiveCharacterTextSplitter)...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len
)
documents: List[Document] = []
if raw_text.strip():
    documents = text_splitter.create_documents([raw_text])
    print(f"Info: {len(documents)} 個のチャンクが作成されました。")
else:
    print("Warning: 入力テキストが空または空白のみのため、チャンク分割をスキップします。")

# --- 2. LLMGraphTransformer で GraphDocument 化 ---
print("Info: チャットモデル (グラフ抽出用 LLM) を初期化しています...")
llm_graph_extraction = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

graph_documents: List[GraphDocument] = []
if documents:
    print("Info: LLMGraphTransformer を使用してチャンクからグラフドキュメントを抽出しています...")
    transformer = LLMGraphTransformer(llm=llm_graph_extraction)
    try:
        graph_documents = transformer.convert_to_graph_documents(documents)
        print(f"Info: {len(graph_documents)} 個のグラフドキュメントが抽出されました。")
        # デバッグ: 最初のグラフドキュメントの内容を出力
        if graph_documents:
            print("--- Debug: First graph document ---")
            print(graph_documents[0])
            print("------------------------------------")
    except Exception as e:
        print(f"Error: LLMGraphTransformer でのグラフドキュメント抽出に失敗しました: {e}")
        print(traceback.format_exc())
else:
    print("Warning: 分割されたチャンクが空のため、グラフドキュメントの抽出をスキップします。")

# --- 3. PGVector にチャンク保存 ---
if PGVector and documents:
    print(f"Info: PGVector ({PG_CONN}) に接続し、チャンクを保存しています...")
    try:
        # doc_id を先に付与
        for i, doc in enumerate(documents):
            doc.metadata["doc_id"] = f"chunk_{DOC_PATH_STR}_{i}"
            doc.metadata["source_file"] = DOC_PATH_STR

        vector_store = PGVector.from_documents(
            documents=documents,
            embedding=embeddings,
            connection_string=PG_CONN,
            collection_name="graph_rag_chunks_collection_v2",
            pre_delete_collection=True
        )
        print("Info: PGVector にチャンクを保存しました。")
    except Exception as e:
        print(f"Error: PGVector への接続またはデータ保存に失敗しました: {e}")
        print(traceback.format_exc())
        vector_store = None
elif not PGVector:
    print("Warning: PGVector がインポートされていないため、PGVectorへの保存をスキップします。")
elif not documents:
    print("Warning: 分割されたチャンクが空のため、PGVectorへの保存をスキップします。")

# --- 4. Neo4j にロードとチャンク関連性のグラフ化 ---
if Neo4jGraph and graph_documents:
    print(f"Info: Neo4j ({NEO4J_URI}) に接続し、グラフドキュメントをロードしています...")
    try:
        graph_db = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PW
        )
        # 既存のグラフをクリア (オプション、開発時に便利)
        # graph_db.query("MATCH (n) DETACH DELETE n")
        
        # エンティティとリレーションをロード
        graph_db.add_graph_documents(
            graph_documents,
            include_source=True,
            baseEntityLabel=True
        )

        # チャンクノードの情報を更新
        print("Info: チャンクノード情報を更新しています...")
        # add_graph_documentsが作成するソースノードはidがpage_contentになる
        # これをキーにして、事前に付与したdoc_idと紐付ける
        for i, doc in enumerate(documents):
            # page_contentが長すぎるとパラメータとして扱えない場合があるので注意
            # ここでは簡単のためそのまま利用
            source_id = doc.page_content
            chunk_id = doc.metadata["doc_id"]
            chunk_name = f"Chunk {i}: {source_id[:30]}..."
            
            graph_db.query(
                """
                MATCH (d:Document {id: $source_id})
                SET d:Chunk, d.chunkId = $chunk_id, d.text = $text, d.name = $name
                REMOVE d.id
                """,
                params={
                    "source_id": source_id,
                    "chunk_id": chunk_id,
                    "text": source_id,
                    "name": chunk_name
                }
            )
        
        # チャンク間の :NEXT リレーションを作成
        print("Info: チャンク間の順序関係 (:NEXT) を作成しています...")
        if len(documents) > 1:
            for i in range(len(documents) - 1):
                graph_db.query(
                    """
                    MATCH (c1:Chunk {chunkId: $chunk_id_1})
                    MATCH (c2:Chunk {chunkId: $chunk_id_2})
                    MERGE (c1)-[:NEXT]->(c2)
                    """,
                    params={
                        "chunk_id_1": documents[i].metadata["doc_id"],
                        "chunk_id_2": documents[i+1].metadata["doc_id"]
                    }
                )

        graph_db.refresh_schema()
        print("Info: Neo4j にグラフをロードし、チャンク関連性を追加しました。")
        print("Info: パフォーマンス向上のため、Neo4j Browser で :Chunk(chunkId) やノードの 'id' プロパティにインデックスを作成することを推奨します。")

    except Exception as e:
        print(f"Error: Neo4j への接続またはデータロードに失敗しました: {e}")
        print(traceback.format_exc())
        graph_db = None
elif not Neo4jGraph:
    print("Warning: Neo4jGraph がインポートされていないため、Neo4jへのロードをスキップします。")
elif not graph_documents:
    print("Warning: 抽出されたグラフドキュメントが空のため、Neo4jへのロードをスキップします。")


# --- 5. Retriever 構築 ---
graph_retriever_runnable = RunnableLambda(lambda x: "")
vector_retriever_runnable = RunnableLambda(lambda x: [])

# Graph Retriever (Neo4j用)
def create_graph_retriever(graph_db_instance, llm):
    """グラフデータベース用のリトリーバーを作成"""
    if not graph_db_instance:
        return RunnableLambda(lambda x: "Graph DB not available")

    if GraphCypherQAChain:
        try:
            qa_chain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph_db_instance,
                verbose=True,
                return_intermediate_steps=False,
                allow_dangerous_requests=True
            )
            
            def retriever_func(query):
                try:
                    result = qa_chain.invoke({"query": query})
                    if isinstance(result, dict):
                        return result.get("result", "")
                    return str(result)
                except Exception as e:
                    print(f"Warning: グラフクエリ実行エラー: {e}")
                    return ""
            
            return RunnableLambda(retriever_func)
        except Exception as e:
            print(f"Warning: GraphCypherQAChain作成エラー: {e}")
    
    # フォールバック: シンプルなCypherクエリ実行
    def simple_graph_retriever(query):
        try:
            # :NEXTリレーションをたどるクエリの例を追加
            cypher_query = f"""
            MATCH (n)
            WHERE toLower(n.id) CONTAINS toLower($query) OR toLower(n.chunkId) CONTAINS toLower($query)
            // 関連するチャンクをNEXTリレーションでたどる
            OPTIONAL MATCH (n)-[:NEXT*..2]->(related_chunk)
            RETURN n, related_chunk
            LIMIT 10
            """
            results = graph_db_instance.query(cypher_query, params={"query": query})
            return str(results) if results else ""
        except Exception as e:
            print(f"Warning: グラフクエリ実行エラー: {e}")
            return ""
    
    return RunnableLambda(simple_graph_retriever)

# LLM (回答生成用)
print("Info: チャットモデル (回答生成用 LLM) を初期化しています...")
llm_answer_generation = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1
)

if graph_db:
    graph_retriever_runnable = create_graph_retriever(graph_db, llm_answer_generation)
    print("Info: Neo4jベースの GraphRetriever を構築しました。")
else:
    print("Warning: Neo4jグラフデータベースが初期化されていないため、GraphRetrieverは構築されません。")

# Vector Retriever (PGVector用)
if vector_store:
    if HAS_PARENT_DOCUMENT_RETRIEVER and ParentDocumentRetriever and InMemoryStore:
        try:
            byte_store = InMemoryStore()
            # doc_idがmetadataにあることを確認
            valid_documents_for_pdr = [doc for doc in documents if "doc_id" in doc.metadata]
            if valid_documents_for_pdr:
                parent_doc_tuples = [(doc.metadata["doc_id"], doc) for doc in valid_documents_for_pdr]
                byte_store.mset(parent_doc_tuples)

                vector_retriever_instance = ParentDocumentRetriever(
                    vectorstore=vector_store,
                    byte_store=byte_store,
                    id_key="doc_id",
                    search_kwargs={"k": 5},
                    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400)
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
        vector_retriever_instance = vector_store.as_retriever(search_kwargs={"k": 5})
        vector_retriever_runnable = vector_retriever_instance
        print("Info: 標準ベクトルリトリーバーを構築しました。")
else:
    print("Warning: VectorStoreが初期化されていないため、Vector Retrieverは構築されません。")

# --- 6. LCEL チェイン定義 ---

def format_graph_results(graph_data: Union[List[Dict[str, Any]], Any]) -> str:
    """
    Neo4j GraphRetriever の結果を整形する関数。
    """
    if not graph_data:
        return "グラフ情報は取得されませんでした。"
    
    if isinstance(graph_data, str):
        return graph_data
    
    if isinstance(graph_data, list):
        if not graph_data:
            return "整形対象のグラフ情報がありません (空リスト)。"
        formatted_lines = [str(item) for item in graph_data]
        return "\n".join(formatted_lines)
    elif isinstance(graph_data, dict):
        return str(graph_data)
    else:
        return f"予期しない形式のグラフ情報: {str(graph_data)}"

def format_document_results(docs: List[Document]) -> str:
    """ドキュメントリストを整形"""
    if not docs:
        return "ドキュメントは取得されませんでした。"
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
        "question": RunnablePassthrough()
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
必要に応じて、関連する情報源（ドキュメントの出典やグラフのエンティティ、チャンク間の関連性など）を指摘してください。

質問: {question}
回答:
"""
prompt = PromptTemplate.from_template(prompt_template_str)

# LCEL チェーンの構築
chain = (
    retrieve_context
    | prompt
    | llm_answer_generation
    | StrOutputParser()
)

# --- 7. 対話ループ ---
# デバッグのため、ここで一旦終了
print("\n--- Debug: Processing complete. Exiting. ---")
exit()

if __name__ == "__main__":
    print("\n--- Graph-RAG LCEL (Google Gemini) デモ ---")
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
            print("スタックトレース:", traceback.format_exc())
            print("環境変数、APIキー、Neo4j/PGVectorの接続、LangChainのバージョン等を確認してください。")
