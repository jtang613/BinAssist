# BinAssist (v0.1.0)
Author: **Jason Tang**

_A plugin that provides LLM helpers to explain code and assist in RE._

## Description:

This is a LLM plugin aimed at enabling the use of local LLM's (ollama, text-generation-webui, lm-studio, etc) for assisting with binary exploration and reverse engineering. It supports any OpenAI v1-compatible API. Recommended models are LLaMA-based models such as llama3.1:8b, but others should work as well.

Current features include:
* Explain the current function - Works at all IL levels.
* Explain the current instruction - Works at all IL levels.
* General query - Query the LLM directly from the UI.
* Propose actions - Provide a list of proposed actions to apply.
* Function calling - Allow agent to call functions to navigate the binary, rename functions and variables.
* RLHF dataset generation - To enable model fine tuning.
* RAG augmentation - Supports adding contextual documents to refine query effectiveness.
* Settings to modify API host, key, model name and max tokens.

Future Roadmap:
* Agentic assistant - Use Autogen or similar framework for self-guided binary RE.
* Model fine tuning - Leverage the RLHF dataset to fine tune the model.

## Screenshot
![Screenshot](https://github.com/jtang613/BinAssist/res/screenshots.gif)

## Homepage
https://github.com/jtang613/BinAssist

## Installation Instructions

### Linux

An OpenAI compatible API is required. For local LLM support, use Ollama, LMStudio, Open-WebUI, Text-Generation-WebUI, etc.

### Windows

Untested but should work. Submit an Issue or Pull Request for support.

### Darwin

Untested but should work. Submit an Issue or Pull Request for support.

## Minimum Version

This plugin requires the following minimum version of Binary Ninja:

* 3164



## Required Dependencies

The following dependencies are required for this plugin:

 * pip - openai, pysqlite3, markdown, httpx, chromadb, sentence-transformers


## License

This plugin is released under a MIT license.
## Metadata Version

2
