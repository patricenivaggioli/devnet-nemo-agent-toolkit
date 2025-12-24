# Adding a Retrieval Tool

After basic tool calling has been demonstrated, adding context retrieval tools to our agent is a reasonable next step.

In this section we will equip our agent with a tool that is capable of performing retrieval of additional context to answer questions about some new consumer products that the backbone model likely doesn't have pretrained knowledge of. It will use a vector store that stores details about products. We can create this agent using LlamaIndex to demonstrate the framework-agnostic capability of the library.

## Defining the Retrieval Tool

Just like with section 2 above, we will define our new tool by writing to a new source file for this agent: `llama_index_rag_tool.py`. This tool using Llama Index to chunk, embed, index, and retrieve ranked results from the source text when called.

Note: In a real‑world scenario, it is not recommended to upsert records at query time due to latency. However, the simplistic approach below is adequate for this demo.

```bash
cd ~/work/nemo-agent-toolkit-clone/
cat > retail_sales_agent/src/retail_sales_agent/llama_index_rag_tool.py <<'EOF'
import logging
import os

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LlamaIndexRAGConfig(FunctionBaseConfig, name="llama_index_rag"):

    llm_name: LLMRef = Field(description="The name of the LLM to use for the RAG engine.")
    embedder_name: EmbedderRef = Field(description="The name of the embedder to use for the RAG engine.")
    data_dir: str = Field(description="The directory containing the data to use for the RAG engine.")
    description: str = Field(description="A description of the knowledge included in the RAG system.")
    collection_name: str = Field(default="context", description="The name of the collection to use for the RAG engine.")


def _walk_directory(root: str):
    for root, dirs, files in os.walk(root):
        for file_name in files:
            yield os.path.join(root, file_name)


@register_function(config_type=LlamaIndexRAGConfig, framework_wrappers=[LLMFrameworkEnum.LLAMA_INDEX])
async def llama_index_rag_tool(config: LlamaIndexRAGConfig, builder: Builder):
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import StorageContext
    from llama_index.core import VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)
    embedder = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)

    Settings.embed_model = embedder
    Settings.llm = llm

    files = list(_walk_directory(config.data_dir))
    docs = SimpleDirectoryReader(input_files=files).load_data()
    logger.info("Loaded %s documents from %s", len(docs), config.data_dir)

    parser = SentenceSplitter(
        chunk_size=400,
        chunk_overlap=20,
        separator=" ",
    )
    nodes = parser.get_nodes_from_documents(docs)

    index = VectorStoreIndex(nodes)

    query_engine = index.as_query_engine(similarity_top_k=3, )

    async def _arun(inputs: str) -> str:
        """
        Search product catalog for information about tablets, laptops, and smartphones
        Args:
            inputs: user query about product specifications
        """
        try:
            response = query_engine.query(inputs)
            return str(response.response)

        except Exception as e:
            logger.error("RAG query failed: %s", e)
            return f"Sorry, I couldn't retrieve information about that product. Error: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description=config.description)
EOF
```

Then we will register it...

```bash
cat >> retail_sales_agent/src/retail_sales_agent/register.py <<'EOF'

from . import llama_index_rag_tool
EOF
```

## Retrieval Tool Workflow Configuration File

We need a new workflow configuration file which incorporates this new tool.

The key additions are:
* Introduction of an Embedder (`nvidia/nv-embedqa-e5-v5`)
* Addition of an instantiated `llama_index_rag` tool which processes files in the `data/rag` directory
* A custom RAG agent which interfaces with the RAG tool, providing a natural language frontend to the tool.
* Adding the custom RAG agent to the list of available tools to our original agent.

> **Note:** _The only impactful change to the top-level agent was the addition of the new RAG agent. All other changes to the configuration were for enabling the RAG agent._

```bash
cd ~/work/nemo-agent-toolkit-clone/
cat > retail_sales_agent/configs/config_rag.yml <<'EOF'
llms:
  azure_llm:
    _type: azure_openai
    azure_endpoint: ${AZURE_OPENAI_ENDPOINT}
    azure_deployment: ${AZURE_OPENAI_DEPLOYMENT}
    api_key: ${AZURE_OPENAI_API_KEY}
    api_version: ${AZURE_OPENAI_API_VERSION}
    temperature: 0.0

embedders:
  azure_embedder:
    _type: azure_openai
    azure_endpoint: ${AZURE_OPENAI_ENDPOINT}
    azure_deployment: ${AZURE_OPENAI_EMBEDDING_DEPLOYMENT}
    api_key: ${AZURE_OPENAI_API_KEY}
    api_version: ${AZURE_OPENAI_API_VERSION}
    truncate: END

functions:
  total_product_sales_data:
    _type: get_total_product_sales_data
    data_path: data/retail_sales_data.csv
  sales_per_day:
    _type: get_sales_per_day
    data_path: data/retail_sales_data.csv
  detect_outliers:
    _type: detect_outliers_iqr
    data_path: data/retail_sales_data.csv

  product_catalog_rag:
    _type: llama_index_rag
    llm_name: azure_llm
    embedder_name: azure_embedder
    collection_name: product_catalog_rag
    data_dir: data/rag/
    description: "Search product catalog for TabZen tablet, AeroBook laptop, NovaPhone specifications"

workflow:
  _type: react_agent
  tool_names:
    - total_product_sales_data
    - sales_per_day
    - detect_outliers
    - product_catalog_rag
  llm_name: azure_llm
  max_history: 10
  max_iterations: 15
  description: "A helpful assistant that can answer questions about the retail sales CSV data"
  verbose: true
EOF
```

## Running the Workflow

We can now test the RAG-enabled workflow with the following command:

```bash
nat run --config_file=retail_sales_agent/configs/config_rag.yml \
    --input "What is the Ark S12 Ultra tablet and what are its specifications?"
```

Note the significance of what we've achieved in just a few lines of code: a reasoning agent was brought up with tool calls that allow it the execute predefined python functions to achieve what an LLM alone cannot. Additionally, we've incorporated context retrieval RAG into the same workflow so that the agent can access domain-specific or real time data sources that it's backbone LLM has never seen during training.