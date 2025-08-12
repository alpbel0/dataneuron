# 🧠 DataNeuron

**Intelligent Document AI Agent with Multi-Step Reasoning**

DataNeuron is an advanced AI agent that transforms your documents into intelligent conversation partners. Unlike basic RAG systems, DataNeuron employs multi-step reasoning, tool-based architecture, and semantic intelligence to provide comprehensive document analysis and insights.

## ✨ Key Features

### 🤖 Multi-Step Reasoning AI Agent
- **Complex query decomposition** into manageable steps
- **Tool chain planning** with sequential execution
- **Context-aware processing** where each tool builds on previous results
- **Smart commentary** throughout the analysis process

### 📄 Hybrid Document Processing
- **Multi-format support**: PDF, TXT, DOCX
- **Smart routing**: Small documents use full-text search, large documents use vector search
- **Anti-hallucination measures** with source attribution
- **Document persistence** with session management

### 🛠️ Extensible Tool Ecosystem
- **Document tools**: Read, search, summarize, compare documents
- **Web research tools**: Search, verify information, get industry benchmarks
- **Analysis tools**: Extract insights, generate risk assessments
- **Semantic tools**: Query expansion, domain detection

### 🧠 Semantic Intelligence
- **Query expansion**: "fesih" → "sona erme, iptal, terminate, sonlandırma"
- **Domain-aware processing**: Legal, financial, technical terminology
- **Multilingual support**: Turkish ↔ English semantic mapping
- **Context recognition**: Document type detection and optimization

### 🎨 Professional User Interface
- **ChatGPT-style interface** with real-time tool tracking
- **Session management** with document continuity options
- **Sidebar document manager** for file organization
- **Smart error handling** with user-friendly messages

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/dataneuron.git
cd dataneuron
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the application**
```bash
python main.py
```

The application will start on `http://localhost:8501`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# ChromaDB Configuration
CHROMADB_PATH=./data/chroma_db
CHROMADB_COLLECTION=dataneuron_docs

# Web Search (Optional)
SERPAPI_KEY=your_serpapi_key_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/dataneuron.log

# Streamlit
STREAMLIT_PORT=8501
```

### Model Configuration

DataNeuron supports multiple OpenAI models:
- `gpt-3.5-turbo` (default, cost-effective)
- `gpt-4` (enhanced reasoning, higher cost)
- `gpt-4-turbo` (latest features)

## 💡 Usage Examples

### Basic Document Analysis
```
User: "Bu PDF'yi özetle"
DataNeuron: "Belgenizi analiz ediyorum..." 
→ read_full_document() → comprehensive summary with insights
```

### Complex Multi-Document Analysis
```
User: "Bu 3 şirketin mali tablolarını karşılaştır ve sektör ortalamasıyla kıyasla"
DataNeuron: Multi-step execution:
1. read_full_document() × 3
2. compare_documents()
3. search_web() for industry benchmarks
4. synthesize_multi_tool_results()
→ Comprehensive comparative analysis
```

### Legal Document Review
```
User: "Bu sözleşmede riskli maddeler var mı?"
DataNeuron: 
1. analyze_document_domain() → "Legal contract detected"
2. search_in_document("ceza, sorumluluk, fesih") → Semantic expansion
3. search_web() for industry standards
4. generate_risk_assessment()
→ Risk analysis with recommendations
```

## 🏗️ Architecture

### Core Components

- **LLM Agent** (`core/llm_agent.py`): Multi-step reasoning engine
- **Tool Manager** (`core/tool_manager.py`): Tool orchestration and execution
- **Document Processor** (`core/document_processor.py`): Multi-format file reading
- **Session Manager** (`core/session_manager.py`): Document persistence
- **Vector Store** (`core/vector_store.py`): ChromaDB integration

### Tool Categories

1. **Document Tools**: Core document operations
2. **Web Tools**: External research capabilities
3. **Analysis Tools**: Advanced analytics and synthesis
4. **Semantic Tools**: Intelligence enhancement

### Data Flow

```
User Query → LLM Agent → Tool Chain Planning → Sequential Execution → Result Synthesis → Smart Commentary → User Response
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Core functionality tests
pytest tests/test_document_processor.py -v
pytest tests/test_tool_manager.py -v
pytest tests/test_session_manager.py -v

# Integration tests
pytest tests/integration_tests.py -v
```

### Test Coverage
```bash
pytest --cov=dataneuron tests/
```

## 📊 Performance

### Optimization Features
- **Smart document routing**: Avoids unnecessary chunking for small documents
- **Caching**: Repeated queries use cached results
- **Parallel processing**: Multiple tools can run concurrently where possible
- **Resource management**: Memory-efficient processing for large documents

### Benchmarks
- **Small documents** (≤10 pages): ~2-5 seconds response time
- **Large documents** (>10 pages): ~5-15 seconds for initial processing
- **Multi-tool chains**: ~10-30 seconds depending on complexity
- **Web integration**: +3-8 seconds for external research

## 🔒 Security & Privacy

### Data Protection
- **Local processing**: Documents processed locally
- **No data retention**: No user data stored on external servers
- **API key encryption**: Secure credential management
- **Session isolation**: User data isolated between sessions

### Best Practices
- Store API keys in environment variables
- Regularly rotate API keys
- Use HTTPS in production deployments
- Monitor API usage and costs

## 🛣️ Roadmap

### V1.0 (Current)
- ✅ Multi-step reasoning engine
- ✅ Hybrid document processing
- ✅ Basic tool ecosystem
- ✅ Session management
- ✅ Semantic intelligence

### V1.5 (Coming Soon)
- 🔄 Advanced semantic expansion
- 🔄 Multi-language document support
- 🔄 Performance optimizations
- 🔄 Enhanced web research
- 🔄 Custom domain terminologies

### V2.0 (Future)
- 🚀 Image generation tools (DALL-E integration)
- 🚀 Presentation creation tools
- 🚀 Excel/CSV analysis tools
- 🚀 Email automation tools
- 🚀 Calendar integration tools

### Enterprise Features (Future)
- 🏢 Multi-tenant architecture
- 🏢 Role-based access control
- 🏢 Audit logging
- 🏢 Custom model fine-tuning
- 🏢 On-premise deployment

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes
5. Run tests: `pytest`
6. Submit a pull request

### Code Standards
- **PEP 8** compliance
- **Type hints** for all functions
- **Docstrings** for all public methods
- **Unit tests** for new features
- **Integration tests** for new workflows

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: Check this README and code comments
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join GitHub Discussions for questions

### Common Issues

**API Key Errors**
```
Error: OpenAI API key not found
Solution: Ensure OPENAI_API_KEY is set in your .env file
```

**Memory Issues with Large Documents**
```
Error: Memory allocation failed
Solution: Process documents in smaller chunks or increase system memory
```

**ChromaDB Persistence Issues**
```
Error: ChromaDB collection not found
Solution: Check CHROMADB_PATH permissions and restart application
```

## 🙏 Acknowledgments

- **OpenAI** for GPT models and embeddings API
- **ChromaDB** for vector storage capabilities
- **Streamlit** for the user interface framework
- **LangChain** for text processing utilities

## 📈 Unique Value Proposition

### What DataNeuron IS:
- **Multi-Step Reasoning AI Agent** → Complex query decomposition and systematic analysis
- **Context-Aware Tool Orchestrator** → Tools work together with shared context
- **Intelligent Document Consultant** → Understands, analyzes, and recommends
- **Semantic-Aware AI Partner** → Grasps language nuances and domain expertise
- **Extensible Intelligence Platform** → Continuously growing capabilities

### What DataNeuron is NOT:
- Basic PDF reader/search tool
- Static Q&A system
- Embedding-dependent solution
- Single-purpose utility
- Simple RAG implementation

**DataNeuron transforms your documents from static files into intelligent conversation partners that understand context, provide insights, and reason through complex queries just like a human expert would.**

---

*Built with ❤️ for intelligent document analysis*