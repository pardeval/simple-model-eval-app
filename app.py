from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from services.bedrock_service import BedrockService
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

bedrock_service = BedrockService(region=os.getenv('AWS_REGION', 'us-east-2'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    try:
        prompts = bedrock_service.list_prompts()
        return jsonify({'success': True, 'prompts': prompts})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    models = bedrock_service.get_available_models()
    return jsonify({'success': True, 'models': models})

@app.route('/models')
def models_page():
    return render_template('models.html')

@app.route('/api/all-models', methods=['GET'])
def get_all_models():
    models = bedrock_service.get_all_text_models()
    return jsonify({'success': True, 'models': models})

""" @app.route('/api/prompts', methods=['POST'])
def create_prompt():
    try:
        data = request.json
        prompt = bedrock_service.create_prompt(
            name=data.get('name'),
            description=data.get('description', ''),
            prompt_text=data.get('prompt_text'),
            model_id=data.get('model_id'),
            variables=data.get('variables', [])
        )
        return jsonify({'success': True, 'prompt': prompt})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500 """

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json
        print(f"Received evaluation request: {data}")
        
        prompt_id = data.get('prompt_id')
        model_ids = data.get('model_ids', [])
        variables = data.get('variables', {})
        
        print(f"Evaluating prompt {prompt_id} with models {model_ids} and variables {variables}")
        
        results = bedrock_service.evaluate_prompt_across_models(
            prompt_id, model_ids, variables
        )
        
        print(f"Evaluation results: {results}")
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/prompts', methods=['POST'])
def create_prompt():
    try:
        data = request.json
        name = data.get('name')
        description = data.get('description', '')
        prompt_text = data.get('prompt_text')
        model_id = data.get('model_id')
        variables = data.get('variables', [])
        
        result = bedrock_service.create_prompt(
            name, description, prompt_text, model_id, variables
        )
        return jsonify({'success': True, 'prompt': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
