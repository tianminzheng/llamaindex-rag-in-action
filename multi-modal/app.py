import json
import os
import time

import streamlit as st
from llama_index.core import SimpleDirectoryReader
from pdf2image import convert_from_bytes
from PIL import Image

from constants import DATA_EXTRACT_STR
from datahelper import extract_data
from db import load_data_as_dataframe, store_result
from utils import generate_unique_path

if "all_data" not in st.session_state:
    st.session_state["all_data"] = load_data_as_dataframe()

st.title("多模态内容解析器")
st.markdown(("多模态内容解析器允许您上传您的图像/PDF文件，并从中提取数据。"))

setup_tab, upload_tab, all_data_tab = st.tabs(["设置", "上传/提取数据", "持久化数据"])

with setup_tab:
    st.subheader("LLM设置")
    api_key = st.text_input("在这里输入您的OpenAI API密钥。", type="password")
    llm_name = st.selectbox("Which LLM?", ["gpt-4-turbo"])
    model_temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, step=0.1)
    data_extract_str = st.text_area("用于提取数据的查询", value=DATA_EXTRACT_STR)

with upload_tab:

    st.subheader("提取信息")
    st.markdown("上传一张餐厅广告的图片或PDF。")
    uploaded_file = st.file_uploader(
        "上传图片 (PNG, JPG, JPEG)或PDF文件:",
        type=["png", "jpg", "jpeg", "pdf"],
    )

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            # Convert a pdf to image and only use the first page.
            image = convert_from_bytes(uploaded_file.read())[0]
        else:
            image = Image.open(uploaded_file).convert("RGB")

    show_image = st.info("展示已上传文件")
    if show_image and uploaded_file:
        st.image(image)

    # Extract data
    if st.button("提取信息"):
        if not uploaded_file:
            st.warning("请上传文件")
        elif not api_key:
            st.warning(
                "在提取信息之前，您必须在设置标签页中设置一个OpenAI API密钥。"
            )
        else:
            st.session_state["data"] = {}
            with st.spinner("提取中..."):
                image.save("temp.png")
                image_documents = SimpleDirectoryReader(input_files=["temp.png"]).load_data()
                try:
                    response = extract_data(
                        image_documents,
                        data_extract_str,
                        llm_name,
                        model_temperature,
                        api_key,
                    )
                except Exception as e:
                    raise e
                finally:
                    os.remove("temp.png")
            st.session_state["data"].update(response)

    if "data" in st.session_state and st.session_state["data"]:
        # Edit result
        edit_data = st.info("编辑数据")
        if edit_data:
            st.markdown("双击值列的单元格以编辑数据。")
            updated_data = st.data_editor(st.session_state["data"])
            st.session_state["data"] = updated_data
        else:
            st.markdown("提取数据")
            st.json(st.session_state["data"])

        # Confirm and save result
        confirm = st.checkbox("确实结果是否正确")
        if confirm:
            if st.button("插入数据？"):
                with st.spinner("插入数据..."):
                    save_path = generate_unique_path(uploaded_file.name)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(save_path)
                    payload = json.dumps(st.session_state["data"], indent=4)


                    # 存储数据库
                    store_result(payload, save_path.as_posix())

                st.session_state["all_data"] = load_data_as_dataframe()
                st.session_state["data"] = {}

                # Notify user that he has successfully inserted the data.
                container = st.empty()
                container.success("插入数据库成功")
                time.sleep(1)
                container.empty()

                st.rerun()


with all_data_tab:
    # Show all data in the database (for verification purpose)
    st.subheader("已保存数据")
    st.markdown("双击一个单元格以查看整个值。")
    st.dataframe(st.session_state["all_data"], use_container_width=True)
