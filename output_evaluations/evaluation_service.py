import boto3
import json
from typing import Dict, List, Any, Optional
import re


def build_model_pricing() -> Dict[str, Dict[str, float]]:
    """
    Dynamically build MODEL_PRICING from AWS Bedrock and Pricing APIs
    
    Returns:
        Dictionary mapping model IDs to their input/output pricing per 1000 tokens
    """
    pricing_dict = {}
    
    try:
        bedrock_client = boto3.client('bedrock', region_name='us-east-1')
        pricing_client = boto3.client('pricing', region_name='us-east-1')
        
        # Get all foundation models
        models_response = bedrock_client.list_foundation_models()
        models = models_response.get('modelSummaries', [])
        
        # Get unique providers
        providers = set()
        model_to_provider = {}
        
        for model in models:
            model_id = model.get('modelId')
            provider_name = model.get('providerName', '')
            if model_id and provider_name:
                providers.add(provider_name)
                model_to_provider[model_id] = provider_name
        
        print(f"Found {len(providers)} unique providers: {providers}")
        
        # Fetch pricing for each provider
        for provider in providers:
            try:
                filters = [
                    {'Type': 'TERM_MATCH', 'Field': 'serviceCode', 'Value': 'AmazonBedrock'},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': 'US East (N. Virginia)'}
                ]
                
                # Add provider filter if available
                if provider:
                    filters.append({'Type': 'TERM_MATCH', 'Field': 'modelProvider', 'Value': provider})
                
                response = pricing_client.get_products(
                    ServiceCode='AmazonBedrock',
                    Filters=filters,
                    MaxResults=100
                )
                
                for price_item in response.get('PriceList', []):
                    price_data = json.loads(price_item)
                    product = price_data.get('product', {})
                    attributes = product.get('attributes', {})
                    
                    model_id = attributes.get('modelId', '')
                    usage_type = attributes.get('usagetype', '')
                    
                    if not model_id:
                        continue
                    
                    # Get pricing from terms
                    terms = price_data.get('terms', {}).get('OnDemand', {})
                    for term_key, term_value in terms.items():
                        price_dimensions = term_value.get('priceDimensions', {})
                        for dim_key, dim_value in price_dimensions.items():
                            price_per_unit = float(dim_value.get('pricePerUnit', {}).get('USD', 0))
                            
                            # Initialize model pricing if not exists
                            if model_id not in pricing_dict:
                                pricing_dict[model_id] = {'input': 0, 'output': 0}
                            
                            # Determine if input or output based on usage type
                            if 'Input' in usage_type or 'input' in usage_type.lower():
                                pricing_dict[model_id]['input'] = price_per_unit
                            elif 'Output' in usage_type or 'output' in usage_type.lower():
                                pricing_dict[model_id]['output'] = price_per_unit
                
                print(f"Fetched pricing for provider: {provider}")
                
            except Exception as e:
                print(f"Error fetching pricing for provider {provider}: {e}")
        
        print(f"Built pricing for {len(pricing_dict)} models")
        
    except Exception as e:
        print(f"Error building model pricing: {e}")
        # Return fallback pricing
        return {
            'us.anthropic.claude-3-5-sonnet-20241022-v2:0': {'input': 0.003, 'output': 0.015},
            'us.anthropic.claude-3-haiku-20240307-v1:0': {'input': 0.00025, 'output': 0.00125},
            'us.anthropic.claude-3-opus-20240229-v1:0': {'input': 0.015, 'output': 0.075},
            'anthropic.claude-3-5-sonnet-20241022-v2:0': {'input': 0.003, 'output': 0.015},
            'anthropic.claude-3-haiku-20240307-v1:0': {'input': 0.00025, 'output': 0.00125},
            'anthropic.claude-3-opus-20240229-v1:0': {'input': 0.015, 'output': 0.075},
            'us.amazon.nova-pro-v1:0': {'input': 0.0008, 'output': 0.0032},
            'us.amazon.nova-lite-v1:0': {'input': 0.00006, 'output': 0.00024},
            'us.amazon.nova-micro-v1:0': {'input': 0.000035, 'output': 0.00014},
        }
    
    return pricing_dict if pricing_dict else {
        'us.anthropic.claude-3-5-sonnet-20241022-v2:0': {'input': 0.003, 'output': 0.015},
        'us.anthropic.claude-3-haiku-20240307-v1:0': {'input': 0.00025, 'output': 0.00125},
        'us.anthropic.claude-3-opus-20240229-v1:0': {'input': 0.015, 'output': 0.075},
        'anthropic.claude-3-5-sonnet-20241022-v2:0': {'input': 0.003, 'output': 0.015},
        'anthropic.claude-3-haiku-20240307-v1:0': {'input': 0.00025, 'output': 0.00125},
        'anthropic.claude-3-opus-20240229-v1:0': {'input': 0.015, 'output': 0.075},
        'us.amazon.nova-pro-v1:0': {'input': 0.0008, 'output': 0.0032},
        'us.amazon.nova-lite-v1:0': {'input': 0.00006, 'output': 0.00024},
        'us.amazon.nova-micro-v1:0': {'input': 0.000035, 'output': 0.00014},
    }


# Build pricing once at module load
MODEL_PRICING = build_model_pricing()


class EvaluationService:
    """Service for evaluating model outputs using operational metrics"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    def _evaluate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> Dict:
        """Calculate the cost of a model invocation based on token usage."""
        print(f"Calculating cost for model: {model_id}, input: {input_tokens}, output: {output_tokens}")
        print(f"Available models in pricing: {list(MODEL_PRICING.keys())[:5]}...")
        
        if model_id not in MODEL_PRICING:
            print(f"WARNING: Pricing not available for {model_id}")
            return {"error": f"Pricing not available for {model_id}"}
        
        pricing = MODEL_PRICING[model_id]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        print(f"Calculated costs - Input: ${input_cost:.8f}, Output: ${output_cost:.8f}, Total: ${total_cost:.8f}")
        
        try:
            response = self.cloudwatch.put_metric_data(
                Namespace='llm_custom_operational_metrics',  # A logical container for your metrics
                MetricData=[
                    {
                        'MetricName': 'TotalCost',  # The name of your custom metric
                        'Value': total_cost,  # The value of the metric
                        'Dimensions': [  # Optional: Add dimensions for more granular analysis
                            {
                                'Name': 'Model',
                                'Value': model_id
                            }
                        ]
                    }
                ]
            )
            print("Custom metric published")
        except Exception as e:
            print(f"Error publishing CloudWatch metric: {e}")
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": f"${input_cost:.8f}",
            "output_cost_usd": f"${output_cost:.8f}",
            "total_cost_usd": f"${total_cost:.8f}"
        }
    
    def evaluate_response(
        self,
        response_text: str,
        reference_text: Optional[str] = None,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model response using multiple metrics
        
        Args:
            response_text: The model's response to evaluate
            reference_text: Optional reference/ground truth text
            metrics: List of metrics to compute (default: all)
        
        Returns:
            Dictionary with evaluation results
        """
        if metrics is None:
            metrics = ['length', 'readability', 'toxicity', 'coherence']
        
        results = {}
        
        if 'length' in metrics:
            results['length'] = self._evaluate_length(response_text)
        
        if 'readability' in metrics:
            results['readability'] = self._evaluate_readability(response_text)
        
        if 'toxicity' in metrics:
            results['toxicity'] = self._evaluate_toxicity(response_text)
        
        if 'coherence' in metrics:
            results['coherence'] = self._evaluate_coherence(response_text)
        
        if 'similarity' in metrics and reference_text:
            results['similarity'] = self._evaluate_similarity(response_text, reference_text)
        
        return results
    
    def _evaluate_length(self, text: str) -> Dict[str, Any]:
        """Evaluate text length metrics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def _evaluate_readability(self, text: str) -> Dict[str, Any]:
        """Evaluate readability using Flesch Reading Ease"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {'flesch_reading_ease': 0, 'grade_level': 'N/A'}
        
        # Count syllables (simplified)
        syllable_count = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease formula
        fre = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllable_count / len(words))
        fre = max(0, min(100, fre))  # Clamp between 0-100
        
        # Determine grade level
        if fre >= 90:
            grade = '5th grade'
        elif fre >= 80:
            grade = '6th grade'
        elif fre >= 70:
            grade = '7th grade'
        elif fre >= 60:
            grade = '8th-9th grade'
        elif fre >= 50:
            grade = '10th-12th grade'
        elif fre >= 30:
            grade = 'College'
        else:
            grade = 'College graduate'
        
        return {
            'flesch_reading_ease': round(fre, 2),
            'grade_level': grade,
            'interpretation': self._interpret_fre(fre)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Every word has at least one syllable
        if syllable_count == 0:
            syllable_count = 1
        
        return syllable_count
    
    def _interpret_fre(self, score: float) -> str:
        """Interpret Flesch Reading Ease score"""
        if score >= 90:
            return 'Very Easy'
        elif score >= 80:
            return 'Easy'
        elif score >= 70:
            return 'Fairly Easy'
        elif score >= 60:
            return 'Standard'
        elif score >= 50:
            return 'Fairly Difficult'
        elif score >= 30:
            return 'Difficult'
        else:
            return 'Very Difficult'
    
    def _evaluate_toxicity(self, text: str) -> Dict[str, Any]:
        """Evaluate toxicity using keyword-based heuristics"""
        # Simplified toxicity detection
        toxic_keywords = [
            'hate', 'kill', 'stupid', 'idiot', 'dumb', 'awful', 
            'terrible', 'horrible', 'disgusting', 'pathetic'
        ]
        
        text_lower = text.lower()
        toxic_count = sum(1 for keyword in toxic_keywords if keyword in text_lower)
        
        # Calculate toxicity score (0-1)
        words = text.split()
        toxicity_score = min(1.0, toxic_count / max(len(words) / 10, 1))
        
        return {
            'toxicity_score': round(toxicity_score, 3),
            'is_toxic': toxicity_score > 0.3,
            'toxic_keywords_found': toxic_count,
            'severity': 'high' if toxicity_score > 0.5 else 'medium' if toxicity_score > 0.3 else 'low'
        }
    
    def _evaluate_coherence(self, text: str) -> Dict[str, Any]:
        """Evaluate text coherence using basic heuristics"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {
                'coherence_score': 1.0,
                'is_coherent': True,
                'note': 'Single sentence - coherence not applicable'
            }
        
        # Check for transition words
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'thus', 'hence', 'meanwhile', 'nevertheless',
            'also', 'then', 'next', 'finally', 'first', 'second', 'third'
        ]
        
        text_lower = text.lower()
        transition_count = sum(1 for word in transition_words if word in text_lower)
        
        # Calculate coherence score
        coherence_score = min(1.0, 0.5 + (transition_count / len(sentences)) * 0.5)
        
        return {
            'coherence_score': round(coherence_score, 3),
            'is_coherent': coherence_score > 0.6,
            'transition_words_found': transition_count,
            'quality': 'high' if coherence_score > 0.8 else 'medium' if coherence_score > 0.6 else 'low'
        }
    
    def _evaluate_similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """Evaluate similarity between two texts using Jaccard similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return {'jaccard_similarity': 0.0, 'is_similar': False}
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union)
        
        return {
            'jaccard_similarity': round(jaccard, 3),
            'is_similar': jaccard > 0.5,
            'common_words': len(intersection),
            'total_unique_words': len(union)
        }
    
    def evaluate_with_bedrock(
        self,
        response_text: str,
        evaluation_criteria: str,
        model_id: str = 'us.anthropic.claude-3-haiku-20240307-v1:0'
    ) -> Dict[str, Any]:
        """
        Use Bedrock model as a judge to evaluate response quality
        
        Args:
            response_text: The response to evaluate
            evaluation_criteria: What to evaluate (e.g., "accuracy", "helpfulness")
            model_id: Model to use as judge
        
        Returns:
            Evaluation results from the judge model
        """
        prompt = f"""You are an expert evaluator. Please evaluate the following response based on {evaluation_criteria}.

Response to evaluate:
{response_text}

Provide your evaluation in the following JSON format:
{{
    "score": <number from 1-10>,
    "reasoning": "<your reasoning>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}"""
        
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            evaluation_text = response_body['content'][0]['text']
            
            # Try to parse JSON from response
            try:
                # Extract JSON from markdown code blocks if present
                json_match = re.search(r'```json\s*(.*?)\s*```', evaluation_text, re.DOTALL)
                if json_match:
                    evaluation_json = json.loads(json_match.group(1))
                else:
                    evaluation_json = json.loads(evaluation_text)
                
                return {
                    'success': True,
                    'evaluation': evaluation_json,
                    'raw_response': evaluation_text
                }
            except json.JSONDecodeError:
                return {
                    'success': False,
                    'error': 'Could not parse evaluation as JSON',
                    'raw_response': evaluation_text
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def compare_responses(
        self,
        responses: List[Dict[str, str]],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple model responses
        
        Args:
            responses: List of dicts with 'model_id', 'response', and optional 'usage' keys
            metrics: Metrics to use for comparison
        
        Returns:
            Comparison results including cost analysis
        """
        results = []
        
        for resp in responses:
            model_id = resp.get('model_id', 'unknown')
            response_text = resp.get('response', '')
            usage = resp.get('usage', {})
            
            evaluation = self.evaluate_response(response_text, metrics=metrics)
            evaluation['model_id'] = model_id
            
            # Add cost calculation if usage data is available
            if usage:
                # Support both camelCase and snake_case keys
                input_tokens = usage.get('inputTokens') or usage.get('input_tokens', 0)
                output_tokens = usage.get('outputTokens') or usage.get('output_tokens', 0)
                
                if input_tokens > 0 or output_tokens > 0:
                    cost_data = self._evaluate_cost(model_id, input_tokens, output_tokens)
                    evaluation['cost'] = cost_data
                else:
                    print(f"No token usage data for {model_id}: {usage}")
            
            results.append(evaluation)
        
        # Calculate rankings
        rankings = self._calculate_rankings(results)
        
        return {
            'individual_results': results,
            'rankings': rankings,
            'summary': self._generate_comparison_summary(results)
        }
    
    def _calculate_rankings(self, results: List[Dict]) -> Dict[str, List[str]]:
        """Calculate rankings for each metric"""
        rankings = {}
        
        # Rank by word count
        sorted_by_length = sorted(results, key=lambda x: x.get('length', {}).get('word_count', 0), reverse=True)
        rankings['longest_response'] = [r['model_id'] for r in sorted_by_length]
        
        # Rank by readability
        sorted_by_readability = sorted(
            results, 
            key=lambda x: x.get('readability', {}).get('flesch_reading_ease', 0), 
            reverse=True
        )
        rankings['most_readable'] = [r['model_id'] for r in sorted_by_readability]
        
        # Rank by coherence
        sorted_by_coherence = sorted(
            results,
            key=lambda x: x.get('coherence', {}).get('coherence_score', 0),
            reverse=True
        )
        rankings['most_coherent'] = [r['model_id'] for r in sorted_by_coherence]
        
        return rankings
    
    def _generate_comparison_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for comparison"""
        if not results:
            return {}
        
        word_counts = [r.get('length', {}).get('word_count', 0) for r in results]
        readability_scores = [r.get('readability', {}).get('flesch_reading_ease', 0) for r in results]
        
        return {
            'total_responses': len(results),
            'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'avg_readability': sum(readability_scores) / len(readability_scores) if readability_scores else 0,
            'word_count_range': {
                'min': min(word_counts) if word_counts else 0,
                'max': max(word_counts) if word_counts else 0
            }
        }
