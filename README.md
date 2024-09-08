# BinAssist
Author: **Jason Tang**

_A local LLM Assistant to aid in binary RE and exploration._

## Description:
This is a LLM plugin aimed at enabling the use of local LLM's (ollama, text-generation-webui, lm-studio, etc) for assisting with binary exploration and reverse engineering. It supports any OpenAI v1-compatible API. Recommended models are LLaMA-based models such as llama3.1:8b, but others should work as well.

Current features include:
* Explain the current function - works at all IL levels.
* Explain the current instruction - disassembly and LLIL.
* General query - query the LLM directly from the UI.
* Propose actions - Provide a list of proposed actions to apply.
* RLHF dataset generation - to enable model fine tuning.
* RAG augmentation - Supports adding contextual documents to refine query effectiveness.
* Settings to modify API host, key, model name and max tokens.

Future Roadmap:
* Function calling - Allow agent to call functions to navigate the binary, rename functions and variables.
* Agentic assistant - Use Autogen or similar framework for self-guided binary RE.

## Screenshot
![Screenshot](res/screenshot.png)

## License

This plugin is released under an [MIT license](./LICENSE).
