from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from services.bedrock_service import BedrockService
from output_evaluations.evaluation_service import EvaluationService
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

bedrock_service = BedrockService(region=os.getenv('AWS_REGION', 'us-east-2'))
evaluation_service = EvaluationService(region=os.getenv('AWS_REGION', 'us-east-2'))

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

@app.route('/api/prompts/<prompt_id>', methods=['GET'])
def get_prompt(prompt_id):
    try:
        prompt = bedrock_service.get_prompt(prompt_id)
        return jsonify({'success': True, 'prompt': prompt})
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

@app.route('/evaluations')
def evaluations_page():
    return render_template('evaluations.html')

@app.route('/api/evaluate-response', methods=['POST'])
def evaluate_response():
    try:
        data = request.json
        response_text = data.get('response_text')
        reference_text = data.get('reference_text')
        metrics = data.get('metrics', ['length', 'readability', 'toxicity', 'coherence'])
        
        results = evaluation_service.evaluate_response(
            response_text, reference_text, metrics
        )
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/evaluate-with-judge', methods=['POST'])
def evaluate_with_judge():
    try:
        data = request.json
        response_text = data.get('response_text')
        criteria = data.get('criteria', 'overall quality')
        judge_model = data.get('judge_model', 'us.anthropic.claude-3-haiku-20240307-v1:0')
        
        results = evaluation_service.evaluate_with_bedrock(
            response_text, criteria, judge_model
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/compare-responses', methods=['POST'])
def compare_responses():
    try:
        data = request.json
        responses = data.get('responses', [])
        metrics = data.get('metrics', ['length', 'readability', 'toxicity', 'coherence'])
        
        print(f"Comparing {len(responses)} responses")
        for resp in responses:
            print(f"Model: {resp.get('model_id')}, Usage: {resp.get('usage')}")
        
        results = evaluation_service.compare_responses(responses, metrics)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"Error in compare_responses: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model-pricing', methods=['GET'])
def get_model_pricing():
    """Debug endpoint to check MODEL_PRICING"""
    from output_evaluations.evaluation_service import MODEL_PRICING
    return jsonify({
        'success': True,
        'pricing_count': len(MODEL_PRICING),
        'sample_models': list(MODEL_PRICING.keys())[:10],
        'full_pricing': MODEL_PRICING
    })

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
