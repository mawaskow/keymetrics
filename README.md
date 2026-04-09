# Key Metric Analyser

1. prepping.ipynb
- Processes the downloaded policy metadata sheet from Odoo
- Pulls the PDFs from the PDF/data link for each policy
- Note: Some policies A) cannot successfully download the file from the link, or B) download corrupted files
- Identifies policies whose titles contain certain keywords (this is useful to start building our annotation guidelines)

2. mineruwork.py
- Accesses local mineru model hosted via vllm
- Extracts text/formatting from each page of each PDF and saves to json

3. post_mineru.ipynb
- Figures out how to construct dataset from new jsons and metadata sheet

4. iaa.ipynb
- Compares annotations of doccano dataset
