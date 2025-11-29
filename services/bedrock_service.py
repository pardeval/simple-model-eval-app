import boto3
import json
import time
from typing import List, Dict, Any

class BedrockService:
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.bedrock_client = boto3.client('bedrock', region_name=region)
        self.bedrock_agent_client = boto3.client('bedrock-agent', region_name=region)
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts from Bedrock Prompt Management"""
        try:
            response = self.bedrock_agent_client.list_prompts()
            prompts = []
            for prompt in response.get('promptSummaries', []):
                prompts.append({
                    'id': prompt.get('id'),
                    'name': prompt.get('name'),
                    'description': prompt.get('description', ''),
                    'version': prompt.get('version', 'DRAFT')
                })
            return prompts
        except Exception as e:
            print(f"Error listing prompts: {e}")
            return []
    
    def get_prompt(self, prompt_id: str, version: str = 'DRAFT') -> Dict[str, Any]:
        """Get a specific prompt by ID"""
        try:
            response = self.bedrock_agent_client.get_prompt(
                promptIdentifier=prompt_id,
                promptVersion=version
            )
            return response
        except Exception as e:
            print(f"Error getting prompt: {e}")
            return {}
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available inference profiles and models from AWS Bedrock"""
        try:
            models = []
            model_to_profile = {}  # Map model IDs to their inference profiles
            
            # First, get all inference profiles (preferred method)
            try:
                ip_response = self.bedrock_client.list_inference_profiles()
                for profile in ip_response.get('inferenceProfileSummaries', []):
                    profile_id = profile.get('inferenceProfileId')
                    profile_name = profile.get('inferenceProfileName', profile_id)
                    profile_type = profile.get('type', '')
                    status = profile.get('status', '')
                    
                    # Only include ACTIVE profiles
                    if status == 'ACTIVE':
                        # Get the underlying model info
                        profile_models = profile.get('models', [])
                        if profile_models:
                            underlying_model_id = profile_models[0].get('modelId', '')
                            
                            # Track which models have profiles
                            if underlying_model_id:
                                model_to_profile[underlying_model_id] = profile_id
                            
                            # Create friendly display name
                            # Remove region prefix and format nicely
                            display_name = profile_name
                            #USERNOTES: THIS DOES NOT MAKE SENSE
                            #if profile_name.startswith('us.'):
                                #display_name = profile_name.replace('us.', '', 1)
                            #display_name = display_name.replace('.', ' ').replace('-', ' ').title()
                            
                            # Determine provider from profile_name
                            provider = ''
                            if 'anthropic' in profile_name.lower():
                                provider = 'Anthropic - '
                            elif 'amazon' in profile_name.lower():
                                provider = 'Amazon - '
                            elif 'meta' in profile_name.lower():
                                provider = 'Meta - '
                            elif 'cohere' in profile_name.lower():
                                provider = 'Cohere - '
                            elif 'ai21' in profile_name.lower():
                                provider = 'AI21 Labs - '
                            elif 'mistral' in profile_name.lower():
                                provider = 'Mistral AI - '
                            
                            models.append({
                                'id': profile_id,  # Use profile ID for invocation
                                'name': f"{provider} {display_name}",
                                'provider': provider,
                                'type': 'inference_profile',
                                'underlying_model': underlying_model_id
                            })
                
                print(f"Found {len(models)} inference profiles")
            except Exception as e:
                print(f"Could not list inference profiles: {e}")
            
            # Then, get foundation models that support ON_DEMAND (not requiring profiles)
            try:
                fm_response = self.bedrock_client.list_foundation_models()
                for model in fm_response.get('modelSummaries', []):
                    model_id = model.get('modelId')
                    model_name = model.get('modelName', model_id)
                    inference_types = model.get('inferenceTypesSupported', [])
                    output_modalities = model.get('outputModalities', [])
                    
                    # Only add if it's a text model with ON_DEMAND and doesn't have a profile
                    if ('ON_DEMAND' in inference_types and 
                        'TEXT' in output_modalities and 
                        model_id not in model_to_profile):
                        
                        provider = model.get('providerName', '')
                        display_name = f"{provider} - {model_name}" if provider else model_name
                        
                        models.append({
                            'id': model_id,
                            'name': display_name,
                            'provider': provider,
                            'type': 'foundation_model'
                        })
                
                print(f"Total models available: {len(models)}")
            except Exception as e:
                print(f"Could not list foundation models: {e}")
            
            # Sort by provider and name
            models.sort(key=lambda x: (x.get('provider', ''), x.get('name', '')))
            
            return models
        except Exception as e:
            print(f"Error listing models: {e}")
            # Fallback to inference profile IDs if available
            return [
                {'id': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0', 'name': 'Anthropic - Claude 3.5 Sonnet v2', 'provider': 'Anthropic'},
                {'id': 'us.anthropic.claude-3-haiku-20240307-v1:0', 'name': 'Anthropic - Claude 3 Haiku', 'provider': 'Anthropic'},
            ]
    

    def invoke_model(self, model_id: str, prompt: str) -> Dict[str, Any]:
        """Invoke a specific model with a prompt (supports both model IDs and inference profile IDs)"""
        try:
            # Build request body based on model provider
            if 'anthropic' in model_id.lower() or 'claude' in model_id.lower():
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                })
            elif 'nova' in model_id.lower():
                # Amazon Nova models use converse API format
                body = json.dumps({
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {
                        "max_new_tokens": 2000,
                        "temperature": 0.7
                    }
                })
            elif 'titan' in model_id.lower():
                body = json.dumps({
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 2000,
                        "temperature": 0.7
                    }
                })
            elif 'llama' in model_id.lower():
                body = json.dumps({
                    "prompt": prompt,
                    "max_gen_len": 2000,
                    "temperature": 0.7
                })
            elif 'cohere' in model_id.lower():
                body = json.dumps({
                    "prompt": prompt,
                    "max_tokens": 2000,
                    "temperature": 0.7
                })
            elif 'ai21' in model_id.lower():
                body = json.dumps({
                    "prompt": prompt,
                    "maxTokens": 2000,
                    "temperature": 0.7
                })
            elif 'mistral' in model_id.lower():
                body = json.dumps({
                    "prompt": prompt,
                    "max_tokens": 2000,
                    "temperature": 0.7
                })
            else:
                # Generic fallback
                body = json.dumps({
                    "prompt": prompt,
                    "max_tokens": 2000,
                    "temperature": 0.7
                })
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract latency from response metadata if available
            latency_ms = None
            if 'ResponseMetadata' in response:
                headers = response['ResponseMetadata'].get('HTTPHeaders', {})
                # Bedrock returns latency in x-amzn-bedrock-invocation-latency header
                latency_header = headers.get('x-amzn-bedrock-invocation-latency')
                if latency_header:
                    try:
                        latency_ms = int(latency_header)
                    except (ValueError, TypeError):
                        pass
            
            # Parse response based on model provider
            if 'anthropic' in model_id.lower() or 'claude' in model_id.lower():
                result = {
                    'text': response_body['content'][0]['text'],
                    'usage': response_body.get('usage', {})
                }
                if latency_ms is not None:
                    result['latency_ms'] = latency_ms
                return result
            elif 'nova' in model_id.lower():
                # Amazon Nova response format
                output = response_body.get('output', {})
                message = output.get('message', {})
                content = message.get('content', [])
                text = content[0].get('text', '') if content else ''
                result = {
                    'text': text,
                    'usage': response_body.get('usage', {})
                }
                if latency_ms is not None:
                    result['latency_ms'] = latency_ms
                return result
            elif 'titan' in model_id.lower():
                result = {
                    'text': response_body['results'][0]['outputText'],
                    'usage': {'inputTokens': response_body.get('inputTextTokenCount', 0)}
                }
                if latency_ms is not None:
                    result['latency_ms'] = latency_ms
                return result
            elif 'llama' in model_id.lower():
                result = {
                    'text': response_body.get('generation', ''),
                    'usage': response_body.get('prompt_token_count', {})
                }
                if latency_ms is not None:
                    result['latency_ms'] = latency_ms
                return result
            elif 'cohere' in model_id.lower():
                result = {
                    'text': response_body.get('generations', [{}])[0].get('text', ''),
                    'usage': {}
                }
                if latency_ms is not None:
                    result['latency_ms'] = latency_ms
                return result
            elif 'ai21' in model_id.lower():
                result = {
                    'text': response_body.get('completions', [{}])[0].get('data', {}).get('text', ''),
                    'usage': {}
                }
                if latency_ms is not None:
                    result['latency_ms'] = latency_ms
                return result
            elif 'mistral' in model_id.lower():
                result = {
                    'text': response_body.get('outputs', [{}])[0].get('text', ''),
                    'usage': {}
                }
                if latency_ms is not None:
                    result['latency_ms'] = latency_ms
                return result
            else:
                # Generic fallback - try common response fields
                text = (response_body.get('completion') or 
                       response_body.get('text') or 
                       response_body.get('generated_text') or
                       str(response_body))
                result = {
                    'text': text,
                    'usage': {}
                }
                if latency_ms is not None:
                    result['latency_ms'] = latency_ms
                return result
        except Exception as e:
            return {'error': str(e)}
    
    def evaluate_prompt_across_models(
        self, 
        prompt_id: str, 
        model_ids: List[str],
        variables: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate a prompt across multiple models"""
        prompt_data = self.get_prompt(prompt_id)
        
        if not prompt_data:
            return [{'error': 'Prompt not found'}]
        
        # Extract prompt text and apply variables
        prompt_text = self._extract_prompt_text(prompt_data, variables)
        
        results = []
        for model_id in model_ids:
            result = self.invoke_model(model_id, prompt_text)
            result_dict = {
                'model_id': model_id,
                'response': result.get('text', ''),
                'error': result.get('error'),
                'usage': result.get('usage', {})
            }
            # Include latency if available
            if 'latency_ms' in result:
                result_dict['latency_ms'] = result['latency_ms']
            results.append(result_dict)
        
        return results
    
    def _extract_prompt_text(self, prompt_data: Dict, variables: Dict[str, str] = None) -> str:
        """Extract and format prompt text with variables"""
        variants = prompt_data.get('variants', [])
        if not variants:
            return ""
        
        template_config = variants[0].get('templateConfiguration', {})
        text_config = template_config.get('text', {})
        prompt_text = text_config.get('text', '')
        
        if variables:
            for key, value in variables.items():
                prompt_text = prompt_text.replace(f"{{{{{key}}}}}", value)
        
        return prompt_text
    
    def get_all_text_models(self) -> List[Dict[str, Any]]:
        """Get all text generation models (enabled and disabled) from AWS Bedrock"""
        try:
            response = self.bedrock_client.list_foundation_models()
            
            models = []
            for model in response.get('modelSummaries', []):
                output_modalities = model.get('outputModalities', [])
                
                if 'TEXT' in output_modalities:
                    inference_types = model.get('inferenceTypesSupported', [])
                    provider = model.get('providerName', '')
                    model_name = model.get('modelName', model.get('modelId'))
                    
                    models.append({
                        'id': model.get('modelId'),
                        'arn': model.get('modelArn', ''),
                        'name': model_name,
                        'provider': provider,
                        'status': 'Enabled' if 'ON_DEMAND' in inference_types or 'INFERENCE_PROFILE' in inference_types else 'Disabled',
                        'inference_types': inference_types,
                        'input_modalities': model.get('inputModalities', []),
                        'output_modalities': output_modalities
                    })
            
            models.sort(key=lambda x: (x.get('provider', ''), x.get('name', '')))
            return models
        except Exception as e:
            print(f"Error listing all models: {e}")
            return []
    
    def create_prompt(
        self,
        name: str,
        description: str,
        prompt_text: str,
        model_id: str,
        variables: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Create a new prompt in Bedrock Prompt Management"""
        try:
            # Build input variables configuration
            input_variables = []
            if variables:
                for var in variables:
                    input_variables.append({
                        'name': var['name']
                    })
            
            # Create the prompt
            response = self.bedrock_agent_client.create_prompt(
                name=name,
                description=description,
                variants=[
                    {
                        'name': 'default',
                        'templateType': 'TEXT',
                        'templateConfiguration': {
                            'text': {
                                'text': prompt_text,
                                'inputVariables': input_variables
                            }
                        },
                        'modelId': model_id,
                        'inferenceConfiguration': {
                            'text': {
                                'maxTokens': 2000,
                                'temperature': 0.7
                            }
                        }
                    }
                ]
            )
            
            return {
                'id': response.get('id'),
                'name': response.get('name'),
                'version': response.get('version'),
                'arn': response.get('arn')
            }
        except Exception as e:
            raise Exception(f"Error creating prompt: {str(e)}")
