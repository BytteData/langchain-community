import pandas as pd
from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

class BytteLoader(BaseLoader):
    """Loader for Bytte-AI African Language Corpora."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        # Specifically designed for your BBC-Igbo-Pidgin CSV structure
        df = pd.read_csv(self.file_path)
        for _, row in df.iterrows():
            yield Document(
                page_content=row['content'], # Or your text column name
                metadata={"source": self.file_path, "language": row.get('language', 'igbo')}
            )
