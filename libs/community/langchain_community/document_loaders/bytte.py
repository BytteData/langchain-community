"""
Bytte-AI African Language Corpora Loader
Improved by bytte AI
"""
import pandas as pd
from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader


class BytteLoader(BaseLoader):
    """
    Loader for Bytte-AI African Language Corpora.
    
    This loader is specifically designed to load African language datasets
    from CSV files with Bytte-AI format structure.
    """
    
    def __init__(self, file_path: str):
        """
        Initializes the BytteLoader with the specified file path.
        
        Args:
            file_path (str): The path to the CSV file containing the African
                language corpus data. The file should contain 'content' and
                optionally 'language' columns.
        
        Raises:
            ValueError: If file_path is empty or None.
        """
        # Input validation - improved by bytte AI
        if not file_path:
            raise ValueError("file_path cannot be empty")
        
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily loads documents from the CSV file.
        
        Specifically designed for Bytte-AI BBC-Igbo-Pidgin CSV structure.
        Each row in the CSV is converted to a Document object with the text
        content and metadata including source file and language.
        
        Returns:
            Iterator[Document]: An iterator yielding Document objects, each
                containing the text content and associated metadata.
        
        Raises:
            FileNotFoundError: If the specified file_path does not exist.
            pd.errors.EmptyDataError: If the CSV file is empty.
            KeyError: If required columns are missing from the CSV.
        
        Yields:
            Document: A document with page_content from the 'content' column
                and metadata including source file path and language.
        """
        # Load CSV file - improved by bytte AI
        df = pd.read_csv(self.file_path)
        
        # Validate required columns - improved by bytte AI
        if 'content' not in df.columns:
            raise KeyError(f"Required column 'content' not found in {self.file_path}")
        
        # Iterate through rows and yield documents - improved by bytte AI
        for _, row in df.iterrows():
            yield Document(
                page_content=row['content'],  # Main text content
                metadata={
                    "source": self.file_path,
                    "language": row.get('language', 'igbo')  # Default to 'igbo' if not specified
                }
            )
