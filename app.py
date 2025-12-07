import streamlit as st
import google.generativeai as genai
from PIL import Image
import base64
import io
import requests
import chromadb
import pdfplumber
import os
# 🔽 ここが重要
import google.generativeai as genai
from google.generativeai import types




# =============================================
# CONFIG
# =============================================

# ガイドラインPDFを入れているフォルダID
GUIDELINE_FOLDER_ID = "1lIXLRzoOt0otLlyyananeohZ9YSHzgrc"

# 将来用：NG / OKバナー画像フォルダ（今は中身がなくてもOK）
NG_BANNER_FOLDER_ID = "1RRiThTvPari0nH18AbOXXUAVfcHjlREF"  
OK_BANNER_FOLDER_ID = "1gEvD4G2N33S2V-JWB04e_Ke1gTfiKm-f"  

# Gemini / Drive 共通で使うAPIキー
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("環境変数 GOOGLE_API_KEY が設定されていません")
    st.stop()

#embed_client = Client(api_key=API_KEY)変更した

# ★ 変更: クライアントを一元化し、APIキーを渡すか環境変数に依存させる
genai.configure(api_key=API_KEY)



# Gemini 初期化
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-pro")



#embed_model = genai.GenerativeModel("models/gemini-embedding-001")

import requests
import numpy as np
import io
import base64
from PIL import Image

# ... img_to_b64 関数はそのまま使用 ...

# 修正後のコードで、APIの直接呼び出しを止め、新しいSDKのclientを使う場合

from google.generativeai import types



from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

# モデル読み込み（CLIP）
clip_model = SentenceTransformer("clip-ViT-B-32")

def get_image_embedding(image: Image.Image):
    # PIL → ベクトル
    embedding = clip_model.encode(image, convert_to_numpy=True)
    return embedding







# ChromaDB（ガイドライン用に加えてOKバナーとNGバナーを提示する）
client = chromadb.Client()
guideline_collection = client.get_or_create_collection("guidelines")
ok_banner_collection = client.get_or_create_collection("ok_banners")
ng_banner_collection = client.get_or_create_collection("ng_banners")


# =============================================
# 共通ユーティリティ
# =============================================

def img_to_b64(img: Image.Image) -> str:
    """PIL画像 → base64(PNG)"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


import os

LOCAL_OK_DIR = "./OK"
LOCAL_NG_DIR = "./NG"


def index_local_images(folder_path, collection):
    """ローカルフォルダの画像をすべて読み込んで embedding → ChromaDB に登録"""

    files = os.listdir(folder_path)

    for filename in files:
        file_path = os.path.join(folder_path, filename)

        # PNG/JPG以外はスキップ
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            print("スキップ（画像でない）:", filename)
            continue

        # 画像読み込み
        try:
            pil_img = Image.open(file_path).convert("RGB")
        except Exception as e:
            print("画像読み込み失敗:", filename, e)
            continue

        # embedding取得
        try:
            emb = get_image_embedding(pil_img)
        except Exception as e:
            print("Embedding失敗:", filename, e)
            continue

        # ChromaDBへ登録
        collection.add(
            ids=[filename],
            embeddings=[emb],
            metadatas=[{"filename": filename}],
            documents=[f"local image: {filename}"],
        )

        print("登録完了:", filename)


def index_ok_banner_images_local():
    index_local_images(LOCAL_OK_DIR, ok_banner_collection)
    st.success("ローカル OK バナーのインデックス化が完了！")


def index_ng_banner_images_local():
    index_local_images(LOCAL_NG_DIR, ng_banner_collection)
    st.success("ローカル NG バナーのインデックス化が完了！")




# =============================================
# Google Drive: PDF 周り
# =============================================

def list_pdfs(folder_id: str):
    """Driveフォルダ内のPDF一覧を取得"""
    url = (
        "https://www.googleapis.com/drive/v3/files"
        f"?q='{folder_id}'+in+parents+and+mimeType='application/pdf'"
        "&fields=files(id,name,mimeType,size)"
        f"&key={API_KEY}"
    )
    res = requests.get(url).json()
    return res.get("files", [])


def download_pdf(file_id: str) -> io.BytesIO | None:
    """
    Drive の PDF や Google Docs を確実に PDF として取得する安全版。
    """

    # ① まずファイル情報を取得（mimeType を知る必要がある）
    meta_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=id,name,mimeType&key={API_KEY}"
    meta = requests.get(meta_url).json()
    mime = meta.get("mimeType", "")

    # ----------------------------------------------------------------------
    # ケース① 本物の PDF（application/pdf）
    # ----------------------------------------------------------------------
    if mime == "application/pdf":
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={API_KEY}"
        res = requests.get(url)

        if res.content.startswith(b"%PDF"):
            return io.BytesIO(res.content)

        print(f"⚠ PDF と表示されているがダウンロード内容が PDF でない: {meta.get('name')}")
        return None



def extract_pdf_pages(pdf_bytes: io.BytesIO):
    """
    pdfplumberでページごとにテキスト抽出
    return: List[(page_num, text)]
    """
    pages = []
    with pdfplumber.open(pdf_bytes) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((i + 1, text))
    return pages


def load_and_index_guidelines():
    """
    Drive上のガイドラインPDFを読み込み、
    ページ単位で ChromaDB に格納する
    """
    files = list_pdfs(GUIDELINE_FOLDER_ID)

    if not files:
        st.warning("Drive からガイドラインPDFが取得できませんでした。フォルダIDと共有設定を確認してください。")
        return

    indexed_count = 0
    with st.spinner("ガイドラインPDFを読み込んでインデックス化しています…"):
        for f in files:
            file_id = f["id"]
            file_name = f["name"]

            # PDF取得
            pdf_bytes = download_pdf(file_id)

            # ★ Noneなら絶対に読み込まない（今回のエラー原因はここ）
            if pdf_bytes is None:
                print(f"⚠ スキップ（PDF取得不可）: {file_name}")
                continue

            # PDFページ抽出（ここも try/except 内部で安全）
            pages = extract_pdf_pages(pdf_bytes)

            # ページが空ならスキップ（壊れPDF対策）
            if not pages:
                print(f"⚠ スキップ（PDFページ抽出失敗）: {file_name}")
                continue

            # ChromaDB に投入
            for page_num, text in pages:
                if not text.strip():
                    continue

                doc_id = f"{file_id}_{page_num}"

                guideline_collection.add(
                    documents=[text],
                    ids=[doc_id],
                    metadatas=[{
                        "file_name": file_name,
                        "page": page_num
                    }]
                )
                indexed_count += 1

    st.success(f"ガイドラインのインデックス化が完了しました（{indexed_count} ページ）")


# =============================================
# RAG: ガイドライン検索 & プロンプト構築
# =============================================

def rag_search(user_input: str):
    """
    ユーザーの質問に最も関連するガイドラインのページを1件返す
    戻り値: dict or None
    """
    if guideline_collection.count() == 0:
        return None

    result = guideline_collection.query(
        query_texts=[user_input],
        n_results=1
    )

    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    if not docs or not docs[0]:
        return None

    text = docs[0][0]
    meta = metas[0][0] if metas and metas[0] else {}

    return {
        "text": text,
        "file_name": meta.get("file_name", "不明なファイル"),
        "page": meta.get("page", "不明なページ")
    }


def build_prompt(user_input: str, rag_result: dict | None) -> str:
    """
    ガイドラインのRAG結果 + ユーザーの質問から
    Geminiに渡す最終プロンプトを構築
    """
    if rag_result is None:
        # ガイドラインがヒットしなかった場合：一般ルールで回答
        return f"""
あなたは広告審査官です。

ガイドライン検索結果がヒットしなかったため、
一般的な広告審査ルール（誇張表現・不当表示・薬機法・最上級表現・視認性・文字可読性など）に基づいて回答してください。

【ユーザーの質問】
{user_input}

300字以内で簡潔に、
・問題点
・根拠
・改善案
を日本語でわかりやすく説明してください。
"""

    text = rag_result["text"]
    file_name = rag_result["file_name"]
    page = rag_result["page"]

    return f"""
あなたは広告審査官です。

以下は、広告審査に必ず参照すべき自社ガイドラインから抽出した該当箇所です。

【参照ガイドライン】
- ファイル名: {file_name}
- ページ番号: {page}

【ガイドライン抜粋】
{text}

【ユーザーの質問】
{user_input}

上記のガイドライン内容に必ず基づいて、
以下の観点から300字以内で回答してください。

- このバナー or 表現がガイドライン的にOKかNGか
- 問題がある場合、その根拠（どのような考え方・条項に抵触しうるか）
- 具体的な改善案（NG表現→OK表現への書き換え、レイアウト/配色/フォントの改善など）

ガイドラインのどの考え方・表現を根拠にしているかも、
回答文中で自然な形で触れてください。
"""


# =============================================
# 将来用：NG/OKバナー類似画像検索の枠（まだDriveに画像なし）
# =============================================

def list_images(folder_id: str):
    """Driveフォルダ内の画像一覧を取得（将来拡張用）"""
    if not folder_id:
        return []

    url = (
        "https://www.googleapis.com/drive/v3/files"
        f"?q='{folder_id}'+in+parents+and+(mimeType contains 'image/')"
        "&fields=files(id,name,mimeType,size)"
        f"&key={API_KEY}"
    )
    res = requests.get(url).json()
    return res.get("files", [])


#ダウンロードする関数---------------------------------


def download_drive_image(file_id):
    """
    Google Drive の file_id から、必ず画像として取得する安全版。
    alt=media が HTML を返す画像（Google Photos扱い）にも対応。
    """
    # 1) メタデータ取得
    meta = requests.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=name,mimeType&key={API_KEY}"
    ).json()
    name = meta.get("name", "")
    mime = meta.get("mimeType", "")

    # 2) まず alt=media で試す
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={API_KEY}"
    r = requests.get(url)

    # ★ HTMLだったら export で取得（Google Photos 画像対策）
    if r.content.startswith(b"<html") or r.status_code != 200:
        print(f"⚠ alt=media が HTML: {name} → export に切り替え")

        export_url = (
            f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
            f"?mimeType=image/jpeg&key={API_KEY}"
        )
        r = requests.get(export_url)

    # ★ export もダメならエラー扱い
    if r.content.startswith(b"<html") or len(r.content) < 500:
        print(f"❌ 画像を取得できません: {name}")
        print("HEAD:", r.content[:200])
        return None

    # 3) PIL で画像読み込み
    try:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print("❌ Image decode failed:", name, e)
        print("HEAD:", r.content[:200])
        return None

#ダウンロードする関数---------------------------------


#index OKNGバナーの追加して、類似バナーをいれるようにしたーーーーーーーーーーーーーーーーーーーーーーーーーーーー
def index_ok_banner_images():
    ok_files = list_images(OK_BANNER_FOLDER_ID)

    if not ok_files:
        st.warning("OKバナー画像フォルダに画像がありません")
        return

    with st.spinner("OKバナー画像をembedding化しています…"):
        for f in ok_files:
            file_id = f["id"]
            name = f["name"]

            # ★ ここを安全版の画像取得に差し替え
            pil_img = download_drive_image(file_id)
            if pil_img is None:
                print("⚠ スキップ（画像取得不可）:", name)
                continue

            emb = get_image_embedding(pil_img)

            ok_banner_collection.add(
                ids=[file_id],
                embeddings=[emb],
                metadatas=[{"file_id": file_id, "name": name}],
                documents=[f"OK Banner: {name}"]
            )

    st.success("OKバナー embedding 登録が完了しました！")


def index_ng_banner_images():
    ng_files = list_images(NG_BANNER_FOLDER_ID)

    if not ng_files:
        st.warning("NGバナー画像フォルダに画像がありません")
        return

    with st.spinner("NGバナー画像をembedding化しています…"):
        for f in ng_files:
            file_id = f["id"]
            name = f["name"]

            # ★ ここを安全版の画像取得に差し替え
            pil_img = download_drive_image(file_id)
            if pil_img is None:
                print("⚠ スキップ（画像取得不可）:", name)
                continue

            emb = get_image_embedding(pil_img)

            ng_banner_collection.add(
                ids=[file_id],
                embeddings=[emb],
                metadatas=[{"file_id": file_id, "name": name}],
                documents=[f"NG Banner: {name}"]
            )

    st.success("NGバナー embedding 登録が完了しました！")


def find_similar_ok_banners(upload_img: Image.Image, top_k=1):
    if ok_banner_collection.count() == 0:
        return None

    emb = get_image_embedding(upload_img)
    result = ok_banner_collection.query(
        query_embeddings=[emb],
        n_results=top_k
    )
    return result


def find_similar_ng_banners(upload_img: Image.Image, top_k=1):
    if ng_banner_collection.count() == 0:
        return None

    emb = get_image_embedding(upload_img)
    result = ng_banner_collection.query(
        query_embeddings=[emb],
        n_results=top_k
    )
    return result

#index OKNGバナーの追加して、類似バナーをいれるようにしたーーーーーーーーーーーーーーーーーーーーーーーーーーーー




# =============================================
# Streamlit UI
# =============================================

st.set_page_config(page_title="広告バナー AI 審査チャット", layout="wide")
st.title("📘 広告バナー AI 審査チャット（ガイドラインRAG + 画像）")

# 初回ロード時にガイドラインPDFをインデックス化
if "guideline_loaded" not in st.session_state:
    load_and_index_guidelines()
    st.session_state.guideline_loaded = True

# チャット状態管理
if "messages" not in st.session_state:
    st.session_state.messages = []

if "image" not in st.session_state:
    st.session_state.image = None



# サイドバー：画像 & オプション
with st.sidebar:
    st.header("📸 バナー画像アップロード")
    upload = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])

    if upload:
        img = Image.open(upload)
        st.session_state.image = img
        st.image(img, caption="アップロード画像", width=280)

    st.markdown("---")
    show_guideline_chunk = st.checkbox("参照したガイドライン原文を表示する", value=True)

    st.markdown("---")

    #ここに再インデックスを追加する
    if st.button("OKバナー画像を再インデックス化"):
        index_ok_banner_images()

    if st.button("NGバナー画像を再インデックス化"):
        index_ng_banner_images()

    #if st.button("OKバナー画像を再インデックス化（ローカル）"):
        #index_ok_banner_images_local()

    #if st.button("NGバナー画像を再インデックス化（ローカル）"):
        #index_ng_banner_images_local()





# 過去のチャットログ表示
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# チャット入力
user_input = st.chat_input("質問やバナーについての相談内容を入力してください…")

if user_input:
    # ユーザーの発話を表示 & 保存
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ガイドラインRAG検索
    rag_result = rag_search(user_input)

    # Geminiに渡すプロンプト生成
    prompt = build_prompt(user_input, rag_result)

    parts = [{"text": prompt}]

    # 画像があれば一緒に送る（Gemini Vision）
    if st.session_state.image:
        parts.append({
            "mime_type": "image/png",
            "data": img_to_b64(st.session_state.image)
        })

    # Gemini 呼び出し
    response = gemini_model.generate_content(parts)


    ai_reply = response.text

    # 回答表示
    st.chat_message("assistant").write(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

    # 参照したガイドライン原文も必要なら表示
    if show_guideline_chunk and rag_result is not None:
        with st.expander("🔍 この回答の根拠になったガイドライン原文を見る"):
            st.write(f"**ファイル名:** {rag_result['file_name']}")
            st.write(f"**ページ番号:** {rag_result['page']}")
            st.markdown("---")
            st.text(rag_result["text"])
    

    # 類似 OK バナー
    if st.session_state.image:
        sim_ok = find_similar_ok_banners(st.session_state.image, top_k=1)
        if sim_ok and "metadatas" in sim_ok:
            meta = sim_ok["metadatas"][0][0]
            ok_name = meta["name"]
            ok_file_id = meta["file_id"]
            ok_url = f"https://drive.google.com/uc?id={ok_file_id}"

            with st.expander("✅ この画像に近い OK バナー"):
                st.image(ok_url, caption=f"最も類似した OK バナー: {ok_name}", width=320)

        # 類似 NG バナー
        sim_ng = find_similar_ng_banners(st.session_state.image, top_k=1)
        if sim_ng and "metadatas" in sim_ng:
            meta = sim_ng["metadatas"][0][0]
            ng_name = meta["name"]
            ng_file_id = meta["file_id"]
            ng_url = f"https://drive.google.com/uc?id={ng_file_id}"

            with st.expander("⚠ この画像に近い NG バナー"):
                st.image(ng_url, caption=f"最も類似した NG バナー: {ng_name}", width=320)


