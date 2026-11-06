from fastmcp import FastMCP

from example_server.rag.fda_rag import search_fda_source


def register(mcp: FastMCP):

    @mcp.tool()
    def search_opioid_label_updates(
        query: str,
        top_k: int = 3,
    ) -> dict:
        """Search FDA opioid label updates PDF.

        Returns relevant passages, similarity scores,
        and FDA PDF page references.
        """
        return search_fda_source(
            query=query,
            source_type="pdf",
            top_k=top_k,
        )
