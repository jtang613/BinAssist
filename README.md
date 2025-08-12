# BinAssist
Author: **Jason Tang**

_A comprehensive LLM-powered Binary Ninja plugin for intelligent binary analysis and reverse engineering._

## Description

BinAssist is an advanced LLM plugin designed to enhance binary analysis and reverse engineering workflows through intelligent automation. It leverages local and remote LLM capabilities to provide context-aware assistance throughout the binary analysis process. It supports fully agentic reverse engineering through its MCP client and MCP servers like [BinAssistMCP](https://github.com/jtang613/BinAssistMCP)

The plugin supports any OpenAI v1-compatible or Anthropic API, making it compatible with popular LLM providers including Ollama, LM Studio, Open-WebUI, OpenAI, and others. Recommended models include Claude Sonnet, GPT-OSS, DeepSeek and LLaMA-based coder models for optimal performance.

## Core Features

### Explain Tab
- **Function Analysis**: Comprehensive analysis of functions at all IL levels (LLIL, MLIL, HLIL)
- **Instruction Analysis**: Detailed explanations of individual instructions and their purpose
- **Context-Aware**: Stores responses at a function level, allowing you to easily keep track
- **Edit Responses**: Tweak the response as needed and save it for later

### Query Tab 
- **Interactive LLM Chat**: Direct conversation interface with the LLM
- **Agentic Assistant**: Multi-step autonomous analysis using agent frameworks via MCP
- **Binary Context**: Automatically includes relevant binary information in queries
- **Flexible Prompting**: Support for custom queries and analysis requests
- **Streaming Responses**: Real-time response generation with cancellation support

### Actions Tab
- **Intelligent Suggestions**: LLM-powered recommendations for improving binary analysis
- **Four Action Types**:
  - **Rename Function**: Suggest semantically meaningful function names
  - **Rename Variable**: Propose descriptive variable names based on usage
  - **Retype Variable**: Recommend appropriate data types for variables
  - **Auto Create Struct**: Generate structure definitions from data patterns
- **Confidence Scoring**: Each suggestion includes confidence metrics (0.0-1.0)
- **Selective Application**: Choose which suggestions to apply via interactive UI
- **Status Tracking**: Real-time feedback on action application success/failure
- **Tool-Based Architecture**: Uses native LLM tool calling for precise suggestions

### Advanced Capabilities
- **Function Calling**: LLM can directly interact with Binary Ninja API to navigate and modify analysis
- **MCP Integration**: Model Context Protocol support for extensible tool integration  
- **RLHF Dataset Generation**: Collect interaction data to enable model fine-tuning
- **RAG Augmentation**: Enhance queries with relevant documentation and context
- **Async Processing**: Non-blocking UI with background LLM processing
- **Streaming Support**: Real-time response generation with proper cancellation

### Configuration & Settings
- **Flexible API Configuration**: Support for multiple LLM providers and endpoints
- **Model Selection**: Choose from available models for different analysis tasks
- **Token Limits**: Configurable maximum tokens for cost and performance optimization
- **Database Management**: Built-in RLHF and RAG database configuration
- **Thread Safety**: Proper async handling for Binary Ninja's threading requirements

## Future Roadmap

### Planned Features
- **Model Fine-tuning**: Leverage collected RLHF data for specialized model training
- **Advanced RAG**: Enhanced document retrieval and code understanding with Graph-RAG
- **Collaborative Features**: Share analysis insights and suggestions across teams

### Research Areas
- **Domain-Specific Models**: Fine-tuned models for specific binary types
- **Code Generation**: Automated exploit development and patching suggestions
- **Vulnerability Detection**: AI-powered security analysis and reporting
- **Performance Optimization**: Enhanced suggestion accuracy and speed

## Quick Start Guide

### 1. Installation
```bash
# Install dependencies from the plugin directory
pip install -r requirements.txt
```

### 2. Configuration
1. Open **Settings → BinAssist** in Binary Ninja
2. Configure your LLM provider:
   - **API Host**: Point to your preferred API endpoint (e.g., `http://localhost:11434` for Ollama)
   - **API Key**: Set authentication key if required
   - **Model**: Choose your preferred model (e.g., `gpt-oss:20b`)
3. Set database paths for RLHF and RAG features
4. Adjust token limits based on your needs

### 3. Usage
1. Load a binary in Binary Ninja
2. Click the **🤖 robot icon** in the sidebar to open BinAssist
3. Navigate between tabs based on your analysis needs:
   - **Explain**: Analyze functions and instructions
   - **Query**: Interactive chat with the LLM  
   - **Actions**: Get intelligent improvement suggestions

### 4. Workflow Examples

**Function Analysis:**
1. Navigate to a function in Binary Ninja
2. Switch to the **Explain** tab
3. Click **"Explain Function"** for comprehensive analysis

**Getting Suggestions:**  
1. Go to the **Actions** tab
2. Select desired action types (Rename Function, etc.)
3. Click **"Analyse"** to get LLM-powered suggestions
4. Review confidence scores and apply selected actions

**Interactive Queries:**
1. Use the **Query** tab for free-form questions
2. Ask about specific functions, algorithms, or analysis techniques
3. The LLM has full context of your current binary

## Screenshot
![Screenshot](https://raw.githubusercontent.com/jtang613/BinAssist/refs/heads/master/res/screenshot1.png)
![Screenshots](/res/screenshots.gif)

## Homepage
https://github.com/jtang613/BinAssist


## Technical Architecture

### Design Patterns
- **Model-View-Controller (MVC)**: Clean separation of concerns across all tabs
- **Service-Oriented Architecture**: Modular services for different analysis tasks  
- **Observer Pattern**: Qt signal-slot communication for responsive UI
- **Async/Await**: Non-blocking operations with proper thread management

### Key Components

**Controllers:**
- `ExplainController`: Manages function and instruction analysis
- `QueryController`: Handles interactive LLM conversations
- `ActionsController`: Coordinates intelligent suggestion generation

**Services:**
- `BinaryContextService`: Extracts and formats binary information
- `ActionsService`: Validates and applies code improvements
- `RLHFService`: Manages training data collection
- `RAGService`: Handles document retrieval and context enhancement

**Threading:**
- `ExplainLLMThread`: Background processing for explanations
- `QueryLLMThread`: Async chat processing with streaming
- `ActionsLLMThread`: LLM tool calling for suggestion generation

**Tools & Integration:**
- **MCP Tools**: Native LLM tool calling for precise actions
- **Binary Ninja API**: Direct integration with analysis engine
- **OpenAI Compatibility**: Works with any OpenAI v1-compatible endpoint

## Installation & Compatibility

### System Requirements
- **Binary Ninja**: Version 4000 or higher
- **Python**: 3.8+ with Binary Ninja's Python environment
- **Memory**: 4GB+ recommended for local LLM usage
- **Storage**: ~100MB for plugin + model-dependent storage

### Platform Support

**Linux** (Primary Development Platform)
- Fully tested and supported
- Recommended for production use

**Windows** 
- Should work but less tested
- Submit issues for Windows-specific problems

**macOS**
- Should work but less tested  
- Submit issues for macOS-specific problems

### LLM Provider Setup

**Local LLMs (Recommended):**
```bash
# Ollama setup
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
ollama serve  # Runs on http://localhost:11434
```

**Other Compatible Providers:**
- **LM Studio**: Desktop GUI for local models
- **Ollama**: Advanced local LLM interface CLI
- **Open-WebUI**: Advanced local LLM interface GUI
- **OpenAI API**: Cloud-based (requires API key)
- **Anthropic API**: Cloud-based (requires API key)
- **OpenRouter**: Access to multiple models via API

### Installation Steps

1. **Install Dependencies**:
   ```bash
   cd /path/to/BinAssist2
   pip install -r requirements.txt
   ```

2. **Configure Binary Ninja**:
   - Install as a plugin in Binary Ninja's plugin directory
   - Or run directly from development directory

3. **Set up LLM Provider**:
   - Configure your preferred provider (see LLM Provider Setup above)
   - Update BinAssist settings with correct API endpoints

## Dependencies

**Required Python Packages:**
```
anthropic              # LLM API communication
openai                 # LLM API communication
pysqlite3              # Database operations
markdown               # Documentation formatting  
whoosh                 # RAG document store
mcp                    # MCP client
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

- **Bug Reports**: Submit detailed issues with reproduction steps
- **Feature Requests**: Propose new functionality with clear use cases  
- **Code Contributions**: Follow existing patterns

## Support & Community

- **GitHub Issues**: Primary support channel for bugs and features

## License

This plugin is released under the **MIT License** - see LICENSE file for details.

**Minimum Binary Ninja Version**: 4000

**Metadata Version**: 2
