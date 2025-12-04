import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import base64
import os

# ================================
# Gemini API キー設定
# ================================
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


# ================================
# 画像をbase64へ変換
# ================================
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ================================
# Streamlit UI設定
# ================================
st.set_page_config(page_title="広告バナーAIチャット", layout="wide")
st.title("💬 広告バナー AIデザインチャット（ローカル版）")

# チャット状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


# ================================
# サイドバー：画像アップロード
# ================================
with st.sidebar:
    st.header("📸 バナー画像")
    uploaded = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.session_state.uploaded_image = img
        st.image(img, caption="アップロード画像", use_column_width=True)

        st.info("この画像を見ながらAIと会話できます。")


# ================================
# 過去チャットメッセージ表示
# ================================
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ================================
# ユーザー入力（チャット）
# ================================
user_input = st.chat_input("質問をどうぞ（例: このデザインどう？）")

if user_input:
    # まずユーザー側の表示
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Geminiに送る parts を作成
    parts = [{"text": user_input}]

    # 画像があれば一緒に送る
    if st.session_state.uploaded_image:
        img_b64 = image_to_base64(st.session_state.uploaded_image)
        parts.append({
            "mime_type": "image/png",
            "data": img_b64
        })

    # Gemini呼び出し（対話型）
    response = model.generate_content(parts)

    ai_reply = response.text

    # AI側の表示
    st.chat_message("assistant").write(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
