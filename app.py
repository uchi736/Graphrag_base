import streamlit as st
import os
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Streamlit UI 設定 (最初に呼び出す) ---
st.set_page_config(page_title="📄 Chat with Document (Graph-RAG)", layout="wide")

from dotenv import load_dotenv

# LangChain / Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Neo4jGraph (langchain-neo4j パッケージから)
try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    st.error("langchain_neo4j がインストールされていません。`pip install langchain-neo4j` を実行してください。")
    Neo4jGraph = None

# PGVector
try:
    from langchain_community.vectorstores.pgvector import PGVector
except ImportError:
    try:
        from langchain_postgres import PGVector
    except ImportError:
        st.error("PGVector がインストールされていません。`pip install langchain-community` または `pip install langchain-postgres` を実行してください。")
        PGVector = None

# Core LangChain components
# GraphDocument のインポート (最新のパスを優先し、フォールバックを用意)
try:
    from langchain_core.graph_document import GraphDocument
except ImportError:
    try:
        from langchain_community.graphs.graph_document import GraphDocument
    except ImportError:
        try:
            from langchain_experimental.graph_transformers.base import GraphDocument
        except ImportError:
            st.error("GraphDocument をインポートできませんでした。langchain-coreまたはlangchain-communityのバージョンを確認してください。")
            GraphDocument = None

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

# Neo4j専用のQAチェーン (新しいバージョンの場合)
try:
    from langchain_neo4j import GraphCypherQAChain
except ImportError:
    try:
        from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    except ImportError:
        GraphCypherQAChain = None

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

st.title("📄 Chat with Document (Graph-RAG & Gemini)")

# --- セッションステートの初期化 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# --- Helper Functions ---

def format_graph_results(graph_data: List[Dict[str, Any]] | Any) -> str:
    """グラフデータを文字列にフォーマット"""
    if not graph_data:
        return "グラフ情報は取得されませんでした。"
    
    if isinstance(graph_data, str):
        return graph_data
    
    if isinstance(graph_data, list):
        if not graph_data:
            return "整形対象のグラフ情報がありません (空リスト)。"
        formatted_results = []
        for item in graph_data:
            if isinstance(item, dict):
                formatted_results.append(str(item))
            else:
                formatted_results.append(str(item))
        return "\n".join(formatted_results)
    
    return str(graph_data)

def format_document_results(docs: List[Document]) -> str:
    """ドキュメントリストを文字列にフォーマット"""
    if not docs:
        return "ドキュメントは取得されませんでした。"
    
    formatted_docs = []
    for doc in docs:
        source_file = doc.metadata.get('source_file', 'N/A')
        chunk_id = doc.metadata.get('doc_id', 'N/A')
        content = doc.page_content
        
        formatted_doc = f"出典ファイル: {source_file}, チャンクID: {chunk_id}\n内容: {content}"
        formatted_docs.append(formatted_doc)
    
    return "\n---\n".join(formatted_docs)

def create_graph_retriever(graph_db, llm):
    """グラフデータベース用のリトリーバーを作成"""
    if GraphCypherQAChain:
        try:
            # 新しいバージョンのlangchain-neo4jの場合
            qa_chain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph_db,
                verbose=True,
                return_intermediate_steps=False,
                allow_dangerous_requests=True  # 必要に応じて
            )
            
            def retriever_func(query):
                try:
                    result = qa_chain.invoke({"query": query})
                    if isinstance(result, dict):
                        return result.get("result", "")
                    return str(result)
                except Exception as e:
                    st.warning(f"グラフクエリ実行エラー: {e}")
                    return ""
            
            return RunnableLambda(retriever_func)
        except Exception as e:
            st.warning(f"GraphCypherQAChain作成エラー: {e}")
    
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
            results = graph_db.query(cypher_query, params={"query": query})
            return str(results) if results else ""
        except Exception as e:
            st.warning(f"グラフクエリ実行エラー: {e}")
            return ""
    
    return RunnableLambda(simple_graph_retriever)

@st.cache_resource(show_spinner="ドキュメントを処理中...")
def build_rag_chain(_raw_text, _file_name):
    """テキストコンテンツからRAGチェーンを構築する"""
    try:
        # 1. Embeddings と LLM の初期化
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GOOGLE_API_KEY
        )
        
        llm_graph_extraction = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
        
        llm_answer_generation = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )

        # 2. チャンク分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        documents = text_splitter.create_documents([_raw_text])
        
        # メタデータを追加
        for i, doc in enumerate(documents):
            doc.metadata["doc_id"] = f"chunk_{_file_name}_{i}"
            doc.metadata["source_file"] = _file_name

        # 3. グラフ抽出とNeo4jへのロード
        graph_retriever_runnable = RunnableLambda(lambda x: "")
        
        if Neo4jGraph and documents:
            transformer = LLMGraphTransformer(llm=llm_graph_extraction)
            try:
                # グラフドキュメントに変換
                graph_documents = transformer.convert_to_graph_documents(documents)
                
                # Neo4jに接続
                graph_db = Neo4jGraph(
                    url=NEO4J_URI, 
                    username=NEO4J_USER, 
                    password=NEO4J_PW,
                    refresh_schema=False  # スキーマの自動更新を無効化
                )
                
                # グラフドキュメントを追加
                graph_db.add_graph_documents(
                    graph_documents,
                    baseEntityLabel=True,
                    include_source=True
                )

                # チャンクノードの情報を更新
                st.sidebar.info("チャンクノード情報を更新中...")
                for i, doc in enumerate(documents):
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
                st.sidebar.info("チャンク間の関連を作成中...")
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
                
                # スキーマを手動で更新
                graph_db.refresh_schema()
                
                # グラフリトリーバーを作成
                graph_retriever_runnable = create_graph_retriever(graph_db, llm_answer_generation)
                
                st.sidebar.success("Neo4jにグラフとチャンク関連性をロードしました。")
            except Exception as e:
                st.sidebar.error(f"Neo4j処理エラー: {e}")
                st.sidebar.error(traceback.format_exc())

        # 4. PGVectorへの保存とRetriever構築
        vector_retriever_runnable = RunnableLambda(lambda x: [])
        
        if PGVector and documents:
            try:
                # コレクション名をサニタイズ
                collection_name = f"graph_rag_{_file_name.replace('.', '_').replace(' ', '_').lower()}"
                
                vector_store = PGVector.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    connection_string=PG_CONN,
                    collection_name=collection_name,
                    pre_delete_collection=True,
                )
                
                vector_retriever_runnable = vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )
                
                st.sidebar.success("PGVectorにチャンクを保存しました。")
            except Exception as e:
                st.sidebar.error(f"PGVector処理エラー: {e}")
                st.sidebar.error(traceback.format_exc())

        # 5. LCEL チェーン定義
        retrieve_context = RunnableParallel({
            "graph_context": graph_retriever_runnable | RunnableLambda(format_graph_results),
            "document_context": vector_retriever_runnable | RunnableLambda(format_document_results),
            "question": RunnablePassthrough()
        })

        prompt_template_str = """あなたは提供された情報に基づいて質問に答える専門家AIです。
以下のコンテキスト情報を利用して、質問に回答してください。

## グラフコンテキスト
{graph_context}

## ドキュメントコンテキスト
{document_context}

---
上記の情報のみを根拠として、以下の質問に対して、日本語で網羅的かつ正確に回答してください。
情報が見つからない場合は、「提供された情報からは回答できません」と答えてください。

質問: {question}
回答:"""
        
        prompt = PromptTemplate.from_template(prompt_template_str)

        chain = (
            retrieve_context 
            | prompt 
            | llm_answer_generation 
            | StrOutputParser()
        )
        
        return chain
        
    except Exception as e:
        st.error(f"RAGチェーン構築中にエラーが発生しました: {e}")
        st.error(traceback.format_exc())
        return None

# --- サイドバー ---
with st.sidebar:
    st.header("設定")
    
    # 環境変数のチェック
    st.subheader("環境変数の状態")
    env_vars = {
        "GOOGLE_API_KEY": bool(GOOGLE_API_KEY),
        "NEO4J_URI": bool(NEO4J_URI),
        "NEO4J_USER": bool(NEO4J_USER),
        "NEO4J_PW": bool(NEO4J_PW),
        "PG_CONN": bool(PG_CONN)
    }
    
    for var, is_set in env_vars.items():
        if is_set:
            st.success(f"✅ {var}")
        else:
            st.error(f"❌ {var}")
    
    # パッケージのチェック
    st.subheader("必要なパッケージ")
    packages = {
        "langchain-neo4j": Neo4jGraph is not None,
        "PGVector": PGVector is not None,
        "GraphDocument": GraphDocument is not None
    }
    
    for package, is_available in packages.items():
        if is_available:
            st.success(f"✅ {package}")
        else:
            st.error(f"❌ {package}")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "テキストファイル (.txt) をアップロードしてください", 
        type=["txt"]
    )

    if uploaded_file is not None:
        # ファイルがアップロードされたら処理を開始
        if st.session_state.processed_file != uploaded_file.name:
            st.session_state.processed_file = uploaded_file.name
            
            # ファイル内容を読み込む
            try:
                raw_text = uploaded_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                st.warning("ファイルのエンコーディングに問題がある可能性があります。")
            
            # 環境変数のチェック
            if not all([GOOGLE_API_KEY, NEO4J_URI, NEO4J_PW, PG_CONN]):
                st.error(".envファイルに必要な設定がされていません。")
            else:
                # RAGチェーンを構築
                with st.spinner("ドキュメントを処理中..."):
                    chain = build_rag_chain(raw_text, uploaded_file.name)
                    if chain:
                        st.session_state.chain = chain
                        st.success(f"`{uploaded_file.name}` の準備ができました！")
                        # 準備ができたらチャット履歴をクリア
                        st.session_state.messages = []
                    else:
                        st.error("RAGチェーンの構築に失敗しました。")

# --- メインチャット画面 ---

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの新しい入力を待つ
if prompt := st.chat_input("ドキュメントについて質問してください..."):
    if st.session_state.chain is None:
        st.warning("まずサイドバーからファイルをアップロードして処理を完了させてください。")
    else:
        # ユーザーのメッセージを履歴に追加して表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # アシスタントの回答を生成
        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                try:
                    response = st.session_state.chain.invoke(prompt)
                    st.markdown(response)
                    # アシスタントの回答を履歴に追加
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"回答の生成中にエラーが発生しました: {e}\n\n{traceback.format_exc()}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
