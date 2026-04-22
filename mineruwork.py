# adapting code from https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B
import io
import asyncio
import os
import pandas as pd
import fitz
from PIL import Image
import glob
from mineru_vl_utils import MinerUClient
import json

def run_extraction():
    cwd = os.getcwd()
    data_dir = f"{cwd}/data/pdfs"
    output_dir = f"{cwd}/data/output_jsons"
    os.makedirs(output_dir, exist_ok=True)
    client = MinerUClient(
        backend="http-client",
        server_url="http://127.0.0.1:8000" #/v1",
        #model_name="opendatalab/MinerU2.5-2509-1.2B"
    )
    pdf_files = glob.glob(f"{data_dir}/*.pdf")
    #interests = [254, 316, 363, 1071, 1182]
    #pdf_files = [f"{data_dir}/{idx}.pdf" for idx in interests]
    print(f"Found {len(pdf_files)} files. Starting sequential processing...")
    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        fname = fname.split(".")[0]
        if os.path.getsize(pdf_path) == 0:
            print(f"!! Skipping empty file: {fname}")
            continue
        try:
            doc = fitz.open(pdf_path)
            doc_data = []
            for page_num in range(len(doc)):
                print(f"[{fname}] Page {page_num + 1}/{len(doc)}...")
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)) # Reduced resolution to save RAM
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                result_blocks = client.two_step_extract(img)
                for block in result_blocks:
                    block.pop('bbox', None) 
                doc_data.append({
                    "page": page_num + 1,
                    "content": result_blocks
                })
            with open(os.path.join(output_dir, f"{fname}.json"), "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            doc.close()
            print(f"\nCompleted: {fname}")
        except Exception as e:
            print(f"\nFailed {fname}: {e}")

if __name__ == "__main__":
    run_extraction()