Bike Chatbot – Powered by Mistral 7B

This is a Streamlit-based chatbot for buying, selling, and getting information about bikes.
It uses the Mistral 7B Instruct v0.2 language model in quantized .gguf format for efficient local inference.

🚀 Setup Instructions
1. Clone Repository
git clone https://github.com/your-username/ChatBot-Bikes.git
cd ChatBot-Bikes

2. Create Virtual Environment
python -m venv .venv
source .venv/Scripts/activate   # Git Bash (Windows)
# OR
.venv\Scripts\activate          # PowerShell (Windows)

3. Install Requirements
pip install -r requirements.txt

📦 Model Used

We use the Mistral 7B Instruct v0.2 (GGUF) model:

mistral-7b-instruct-v0.2.Q3_K_M.gguf


Family: Mistral

Size: 7B parameters

Type: Instruction-tuned (for Q&A/chat)

Format: GGUF (optimized for llama-cpp-python)

Quantization: Q3_K_M (3-bit, memory-efficient)

🔗 Download Model

Since the model is too large for GitHub, please download it manually:

👉 Hugging Face – Mistral 7B GGUF Models
