# Bedrock Model Evaluator

A Python web application for evaluating and comparing different Amazon Bedrock models using Prompt Management.

## Features

- **Create prompts** directly in Bedrock Prompt Management
- List and select prompts from Bedrock Prompt Management
- **Dynamically loads all enabled models** from your AWS account
- Compare responses across multiple Bedrock models
- Support for prompt variables with auto-detection
- Clean, responsive UI
- Real-time model evaluation

## Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- AWS credentials configured (via AWS CLI or environment variables)
- Bedrock models enabled in your AWS account

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials:
```bash
aws configure
```

3. Copy `.env.example` to `.env` and adjust settings:
```bash
cp .env.example .env
```

4. Run the application:
```bash
python app.py
```

5. Open your browser to `http://localhost:5000`

## Usage

### Creating a Prompt

1. Click "+ Create New" button
2. Enter prompt name, description, and text (use `{{variable}}` syntax for variables)
3. Select a default model
4. Add variables manually or use "Auto-Detect" to extract them from your prompt
5. Click "Create Prompt"

### Evaluating Models

1. Select a prompt from your Bedrock Prompt Management library
2. (Optional) Add variables in JSON format if your prompt uses them
3. Select one or more models to compare (all enabled models in your account are shown)
4. Click "Evaluate Models" to see responses from each model

## Project Structure

```
.
├── app.py                      # Flask application entry point
├── services/
│   └── bedrock_service.py      # Bedrock API integration
├── templates/
│   └── index.html              # Web UI
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Supported Models

The application automatically detects and displays all text generation models enabled in your AWS Bedrock account, including:

- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku, Claude 3 Opus
- **Amazon**: Titan Text models
- **Meta**: Llama models
- **Cohere**: Command models
- **AI21 Labs**: Jurassic models
- **Mistral AI**: Mistral models

Models must be enabled in your AWS account and support on-demand inference.

## AWS Permissions Required

Your AWS credentials need the following permissions:
- `bedrock:ListFoundationModels` - List available models
- `bedrock:InvokeModel` - Invoke models for evaluation
- `bedrock-agent:ListPrompts` - List prompts
- `bedrock-agent:GetPrompt` - Get prompt details
- `bedrock-agent:CreatePrompt` - Create new prompts
