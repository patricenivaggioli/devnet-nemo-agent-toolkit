# Human In The Loop (HITL)

## Adding a Custom Agent

Besides using inbuilt agents in the workflows, we can also create custom agents using LangGraph or any other framework and bring them into a workflow. We demonstrate this by swapping out the ReAct agent used by the data visualization expert for a custom agent that has human-in-the-loop capability. The agent will ask the user whether they would like a summary of graph content.

This exemplifies how complete agent workflows can be wrapped and used as tools by other agents, enabling complex multi-agent orchestration.

### Human-in-the-Loop (HITL) Approval Tool

The following steps define the approval tool and its registration.

```bash
cd ~/nemo-agent-toolkit/
cat > retail_sales_agent/src/retail_sales_agent/hitl_approval_tool.py <<'EOF'
import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.interactive import HumanPromptText
from nat.data_models.interactive import InteractionResponse

logger = logging.getLogger(__name__)


class HITLApprovalFnConfig(FunctionBaseConfig, name="hitl_approval_tool"):
    """
    This function is used to get the user's response to the prompt.
    It will return True if the user responds with 'yes', otherwise False.
    """

    prompt: str = Field(..., description="The prompt to use for the HITL function")


@register_function(config_type=HITLApprovalFnConfig)
async def hitl_approval_function(config: HITLApprovalFnConfig, builder: Builder):

    import re

    prompt = f"{config.prompt} Please confirm if you would like to proceed. Respond with 'yes' or 'no'."

    async def _arun(unused: str = "") -> bool:

        nat_context = Context.get()
        user_input_manager = nat_context.user_interaction_manager

        human_prompt_text = HumanPromptText(text=prompt, required=True, placeholder="<your response here>")
        response: InteractionResponse = await user_input_manager.prompt_user_input(human_prompt_text)
        response_str = response.content.text.lower()  # type: ignore
        selected_option = re.search(r'\b(yes)\b', response_str)

        if selected_option:
            return True
        return False

    yield FunctionInfo.from_fn(_arun,
                               description=("This function will be used to get the user's response to the prompt"))
EOF
```

```bash
cd ~/nemo-agent-toolkit/
cat >> retail_sales_agent/src/retail_sales_agent/register.py <<'EOF'

from . import hitl_approval_tool
EOF
```

### Graph Summarizer Tool

The following two steps define the graph summarizer tool and its registration.

```bash
cd ~/nemo-agent-toolkit/
cat > retail_sales_agent/src/retail_sales_agent/graph_summarizer_tool.py <<'EOF'
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class GraphSummarizerConfig(FunctionBaseConfig, name="graph_summarizer"):
    """Analyze and summarize chart data."""
    llm_name: LLMRef = Field(description="The name of the LLM to use for the graph summarizer.")


@register_function(config_type=GraphSummarizerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def graph_summarizer_function(config: GraphSummarizerConfig, builder: Builder):
    """Analyze chart data and provide natural language summaries."""
    import base64

    from langchain_core.messages import HumanMessage

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _graph_summarizer(image_path: str) -> str:
        """
        Analyze chart data and provide insights and summaries.

        Args:
            image_path: The path to the image to analyze

        Returns:
            String containing analysis and insights
        """

        def encode_image(image_path: str) -> str:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        base64_image = encode_image(image_path)

        # Create a multimodal message with text and image
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Please summarize the key insights from this graph in natural language."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        )

        # Invoke the LLM with the multimodal message
        response = await llm.ainvoke([message])
        return response.content

    yield FunctionInfo.from_fn(
        _graph_summarizer,
        description=("This tool can be used to summarize the key insights from a graph in natural language. "
                     "It takes in the path to an image and returns a summary of the key insights from the graph."))
EOF
```

```bash
cd ~/nemo-agent-toolkit/
cat >> retail_sales_agent/src/retail_sales_agent/register.py <<'EOF'

from . import graph_summarizer_tool
EOF
```

### Custom Data Visualization Agent With HITL Approval

The following two steps define the custom agent and its registration

```bash
cd ~/nemo-agent-toolkit/
cat > retail_sales_agent/src/retail_sales_agent/data_visualization_agent.py <<'EOF'
import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class DataVisualizationAgentConfig(FunctionBaseConfig, name="data_visualization_agent"):
    """
    NeMo Agent toolkit function config for data visualization.
    """
    llm_name: LLMRef = Field(description="The name of the LLM to use")
    tool_names: list[FunctionRef] = Field(description="The names of the tools to use")
    description: str = Field(description="The description of the agent.")
    prompt: str = Field(description="The prompt to use for the agent.")
    graph_summarizer_fn: FunctionRef = Field(description="The function to use for the graph summarizer.")
    hitl_approval_fn: FunctionRef = Field(description="The function to use for the hitl approval.")
    max_retries: int = Field(default=3, description="The maximum number of retries for the agent.")


@register_function(config_type=DataVisualizationAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def data_visualization_agent_function(config: DataVisualizationAgentConfig, builder: Builder):
    from langchain_core.messages import AIMessage
    from langchain_core.messages import BaseMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langchain_core.messages import ToolMessage
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode
    from pydantic import BaseModel

    class AgentState(BaseModel):
        retry_count: int = 0
        messages: list[BaseMessage]
        approved: bool = True

    tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm_n_tools = llm.bind_tools(tools)

    hitl_approval_fn: Function = await builder.get_function(config.hitl_approval_fn)
    graph_summarizer_fn: Function = await builder.get_function(config.graph_summarizer_fn)

    async def conditional_edge(state: AgentState):
        try:
            logger.debug("Starting the Tool Calling Conditional Edge")
            messages = state.messages
            last_message = messages[-1]
            logger.info("Last message type: %s", type(last_message))
            logger.info("Has tool_calls: %s", hasattr(last_message, 'tool_calls'))
            if hasattr(last_message, 'tool_calls'):
                logger.info("Tool calls: %s", last_message.tool_calls)

            if (hasattr(last_message, 'tool_calls') and last_message.tool_calls and len(last_message.tool_calls) > 0):
                logger.info("Routing to tools - found non-empty tool calls")
                return "tools"
            logger.info("Routing to check_hitl_approval - no tool calls to execute")
            return "check_hitl_approval"
        except Exception as ex:
            logger.error("Error in conditional_edge: %s", ex)
            if hasattr(state, 'retry_count') and state.retry_count >= config.max_retries:
                logger.warning("Max retries reached, returning without meaningful output")
                return "__end__"
            state.retry_count = getattr(state, 'retry_count', 0) + 1
            logger.warning(
                "Error in the conditional edge: %s, retrying %d times out of %d",
                ex,
                state.retry_count,
                config.max_retries,
            )
            return "data_visualization_agent"

    def approval_conditional_edge(state: AgentState):
        """Route to summarizer if user approved, otherwise end"""
        logger.info("Approval conditional edge: %s", state.approved)
        if hasattr(state, 'approved') and not state.approved:
            return "__end__"
        return "summarize"

    def data_visualization_agent(state: AgentState):
        sys_msg = SystemMessage(content=config.prompt)
        messages = state.messages

        if messages and isinstance(messages[-1], ToolMessage):
            last_tool_msg = messages[-1]
            logger.info("Processing tool result: %s", last_tool_msg.content)
            summary_content = f"I've successfully created the visualization. {last_tool_msg.content}"
            return {"messages": [AIMessage(content=summary_content)]}
        logger.info("Normal agent operation - generating response for: %s", messages[-1] if messages else 'no messages')
        return {"messages": [llm_n_tools.invoke([sys_msg] + state.messages)]}

    async def check_hitl_approval(state: AgentState):
        messages = state.messages
        last_message = messages[-1]
        logger.info("Checking hitl approval: %s", state.approved)
        logger.info("Last message type: %s", type(last_message))
        selected_option = await hitl_approval_fn.acall_invoke()
        if selected_option:
            return {"approved": True}
        return {"approved": False}

    async def summarize_graph(state: AgentState):
        """Summarize the graph using the graph summarizer function"""
        image_path = None
        for msg in state.messages:
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                import re
                img_ext = r'[a-zA-Z0-9_.-]+\.(?:png|jpg|jpeg|gif|svg)'
                pattern = rf'saved to ({img_ext})|({img_ext})'
                match = re.search(pattern, content)
                if match:
                    image_path = match.group(1) or match.group(2)
                    break

        if not image_path:
            image_path = "sales_trend.png"

        logger.info("Extracted image path for summarization: %s", image_path)
        response = await graph_summarizer_fn.ainvoke(image_path)
        return {"messages": [response]}

    try:
        logger.debug("Building and compiling the Agent Graph")
        builder_graph = StateGraph(AgentState)

        builder_graph.add_node("data_visualization_agent", data_visualization_agent)
        builder_graph.add_node("tools", ToolNode(tools))
        builder_graph.add_node("check_hitl_approval", check_hitl_approval)
        builder_graph.add_node("summarize", summarize_graph)

        builder_graph.add_conditional_edges("data_visualization_agent", conditional_edge)

        builder_graph.set_entry_point("data_visualization_agent")
        builder_graph.add_edge("tools", "data_visualization_agent")

        builder_graph.add_conditional_edges("check_hitl_approval", approval_conditional_edge)

        builder_graph.add_edge("summarize", "__end__")

        agent_executor = builder_graph.compile()

        logger.info("Data Visualization Agent Graph built and compiled successfully")

    except Exception as ex:
        logger.error("Failed to build Data Visualization Agent Graph: %s", ex)
        raise

    async def _arun(user_query: str) -> str:
        """
        Visualize data based on user query.

        Args:
            user_query (str): User query to visualize data

        Returns:
            str: Visualization conclusion from the LLM agent
        """
        input_message = f"User query: {user_query}."
        response = await agent_executor.ainvoke({"messages": [HumanMessage(content=input_message)]})

        return response

    try:
        yield FunctionInfo.from_fn(_arun, description=config.description)

    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up retail_sales_agent workflow.")
EOF
```

```bash
cd ~/nemo-agent-toolkit/
cat >> retail_sales_agent/src/retail_sales_agent/register.py <<'EOF'

from . import data_visualization_agent
EOF
```

## Custom Agent Workflow Configuration File

Next, we define the workflow configuration file for this custom agent.

The high-level changes include:
- switching from a ReAct agent to the custom agent with HITL
- adding additional tools (HITL, graph summarization)
- adding an OpenAI LLM for image summarization

```bash
cd ~/nemo-agent-toolkit/
cat > retail_sales_agent/configs/config_multi_agent_hitl.yml <<'EOF'
llms:
  azure_llm:
    _type: azure_openai
    azure_endpoint: ${AZURE_OPENAI_ENDPOINT}
    azure_deployment: ${AZURE_OPENAI_DEPLOYMENT}
    api_key: ${AZURE_OPENAI_API_KEY}
    api_version: ${AZURE_OPENAI_API_VERSION}
    temperature: 0.0

  summarizer_llm:
    _type: azure_openai
    azure_endpoint: ${AZURE_OPENAI_ENDPOINT}
    azure_deployment: ${AZURE_OPENAI_VISION_DEPLOYMENT}
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

  data_analysis_agent:
    _type: tool_calling_agent
    tool_names:
      - total_product_sales_data
      - sales_per_day
      - detect_outliers
    llm_name: azure_llm
    max_history: 10
    max_iterations: 15
    description: |
      A helpful assistant that can answer questions about the retail sales CSV data.
      Use the tools to answer the questions.
      Input is a single string.
    verbose: false

  plot_sales_trend_for_stores:
    _type: plot_sales_trend_for_stores
    data_path: data/retail_sales_data.csv
  plot_and_compare_revenue_across_stores:
    _type: plot_and_compare_revenue_across_stores
    data_path: data/retail_sales_data.csv
  plot_average_daily_revenue:
    _type: plot_average_daily_revenue
    data_path: data/retail_sales_data.csv

  hitl_approval_tool:
    _type: hitl_approval_tool
    prompt: |
      Do you want to summarize the created graph content?
  graph_summarizer:
    _type: graph_summarizer
    llm_name: summarizer_llm

  data_visualization_agent:
    _type: data_visualization_agent
    llm_name: azure_llm
    tool_names:
      - plot_sales_trend_for_stores
      - plot_and_compare_revenue_across_stores
      - plot_average_daily_revenue
    graph_summarizer_fn: graph_summarizer
    hitl_approval_fn: hitl_approval_tool
    prompt: |
      You are a data visualization expert.
      Your task is to create plots and visualizations based on user requests.
      Use available tools to analyze data and generate plots.
    description: |
      This is a data visualization agent that should be called if the user asks for a visualization or plot of the data.
      It has access to the following tools:
      - plot_sales_trend_for_stores: This tool can be used to plot the sales trend for a specific store or all stores.
      - plot_and_compare_revenue_across_stores: This tool can be used to plot and compare the revenue trends across stores. Use this tool only if the user asks for a comparison of revenue trends across stores.
      - plot_average_daily_revenue: This tool can be used to plot the average daily revenue for stores and products.
      The agent will use the available tools to analyze data and generate plots.
      The agent will also use the graph_summarizer tool to summarize the graph data.
      The agent will also use the hitl_approval_tool to ask the user whether they would like a summary of the graph data.

  product_catalog_rag:
    _type: llama_index_rag
    llm_name: azure_llm
    embedder_name: azure_embedder
    collection_name: product_catalog_rag
    data_dir: data/rag/
    description: "Search product catalog for TabZen tablet, AeroBook laptop, NovaPhone specifications"

  rag_agent:
    _type: react_agent
    llm_name: azure_llm
    tool_names:
      - product_catalog_rag
    max_history: 3
    max_iterations: 5
    max_retries: 2
    retry_parsing_errors: true
    description: |
      An assistant that can answer questions about products.
      Use product_catalog_rag to answer questions about products.
      Do not make up information.
    verbose: true


workflow:
  _type: react_agent
  tool_names:
    - data_analysis_agent
    - data_visualization_agent
    - rag_agent
  llm_name: summarizer_llm
  verbose: true
  handle_parsing_errors: true
  max_retries: 2
  system_prompt: |
    Answer the following questions as best you can. You may communicate and collaborate with various experts to answer the questions:

    {tools}

    If the user responds "no" to a request to continue, you should end the conversation.

    You may respond in one of two formats.
    Use the following format exactly to communicate with an expert:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action (if there is no required input, include "Action Input: None")
    Observation: wait for the expert to respond, do not assume the expert's response

    ... (this Thought/Action/Action Input/Observation can repeat N times.)
    Use the following format once you have the final answer:

    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
EOF
```

## Running the Workflow

```bash
nat run --config_file retail_sales_agent/configs/config_multi_agent_hitl.yml \
    --input "Plot average daily revenue"
```

You should have the following output, please reply `yes` to proceed with the graph analysis:

```console
2026-01-04 19:28:16 - INFO     - nat.cli.commands.start:192 - Starting NAT from config file: 'retail_sales_agent/configs/config_multi_agent_hitl.yml'
2026-01-04 19:28:19 - INFO     - retail_sales_agent.llama_index_rag_tool:48 - Loaded 1 documents from data/rag/
2026-01-04 19:28:20 - INFO     - retail_sales_agent.data_visualization_agent:152 - Data Visualization Agent Graph built and compiled successfully

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 12
Number of Function Groups: 0
Number of LLMs: 2
Number of Embedders: 1
Number of Memory: 0
Number of Object Stores: 0
Number of Retrievers: 0
Number of TTC Strategies: 0
Number of Authentication Providers: 0

2026-01-04 19:28:21 - INFO     - nat.agent.react_agent.agent:183 - 
------------------------------
[AGENT]
Agent input: Plot average daily revenue
Agent's thoughts: 
Thought: The user wants to visualize the average daily revenue for stores and products. I will use the data_visualization_agent to generate the plot.

Action: data_visualization_agent
Action Input: {"user_query": "Plot average daily revenue"}
------------------------------
2026-01-04 19:28:21 - INFO     - retail_sales_agent.data_visualization_agent:98 - Normal agent operation - generating response for: content='User query: Plot average daily revenue.' additional_kwargs={} response_metadata={}
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:58 - Last message type: <class 'langchain_core.messages.ai.AIMessage'>
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:59 - Has tool_calls: True
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:61 - Tool calls: [{'name': 'plot_average_daily_revenue', 'args': {'arg': 'average_daily_revenue'}, 'id': 'call_Z35Nf4TkcjBq5K7bqkIi7Tub', 'type': 'tool_call'}]
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:64 - Routing to tools - found non-empty tool calls
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:95 - Processing tool result: Average daily revenue plot saved to average_daily_revenue.png
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:58 - Last message type: <class 'langchain_core.messages.ai.AIMessage'>
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:59 - Has tool_calls: True
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:61 - Tool calls: []
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:66 - Routing to check_hitl_approval - no tool calls to execute
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:104 - Checking hitl approval: True
2026-01-04 19:28:23 - INFO     - retail_sales_agent.data_visualization_agent:105 - Last message type: <class 'langchain_core.messages.ai.AIMessage'>
Do you want to summarize the created graph content?
 Please confirm if you would like to proceed. Respond with 'yes' or 'no'.:
```

When you reply `yes` the workflow continue with the graph analysis and you should have the following final output:

```console
2026-01-04 19:31:11 - INFO     - retail_sales_agent.data_visualization_agent:84 - Approval conditional edge: True
2026-01-04 19:31:11 - INFO     - retail_sales_agent.data_visualization_agent:128 - Extracted image path for summarization: average_daily_revenue.png
2026-01-04 19:31:18 - INFO     - nat.agent.base:221 - 
------------------------------
[AGENT]
Calling tools: data_visualization_agent
Tool's input: {'user_query': 'Plot average daily revenue'}
Tool's response: 
[{'messages': ['This graph compares the average daily revenue generated by three product categories (Laptop, Phone, and Tablet) across two stores (S001 and S002). \n\nKey insights:\n1. **Phones** generate the highest revenue in both stores, with S001 slightly outperforming S002 in this category.\n2. **Laptops** are the second-highest revenue generator, with similar performance across both stores.\n3. **Tablets** contribute the least revenue in both stores, with S001 and S002 showing comparable results for this product category.\n4. Overall, S001 and S002 have similar revenue patterns across all product categories.'], 'approved': True}]
------------------------------
2026-01-04 19:31:20 - INFO     - nat.agent.react_agent.agent:207 - 
------------------------------
[AGENT]
Agent input: Plot average daily revenue
Agent's thoughts: 
Final Answer: The graph compares the average daily revenue generated by three product categories (Laptop, Phone, and Tablet) across two stores (S001 and S002). Key insights include:

1. **Phones** generate the highest revenue in both stores, with S001 slightly outperforming S002 in this category.
2. **Laptops** are the second-highest revenue generator, with similar performance across both stores.
3. **Tablets** contribute the least revenue in both stores, with S001 and S002 showing comparable results for this product category.
4. Overall, S001 and S002 have similar revenue patterns across all product categories.
------------------------------
2026-01-04 19:31:20 - INFO     - nat.front_ends.console.console_front_end_plugin:102 - --------------------------------------------------
Workflow Result:
['The graph compares the average daily revenue generated by three product categories (Laptop, Phone, and Tablet) across two stores (S001 and S002). Key insights include:\n\n1. **Phones** generate the highest revenue in both stores, with S001 slightly outperforming S002 in this category.\n2. **Laptops** are the second-highest revenue generator, with similar performance across both stores.\n3. **Tablets** contribute the least revenue in both stores, with S001 and S002 showing comparable results for this product category.\n4. Overall, S001 and S002 have similar revenue patterns across all product categories.']
--------------------------------------------------
Cleaning up retail_sales_agent workflow.
```
