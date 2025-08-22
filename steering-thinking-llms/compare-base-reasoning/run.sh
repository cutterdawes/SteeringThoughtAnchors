# Distill models
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --skip_viz --model deepseek/deepseek-r1-distill-llama-8b # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --skip_viz --model deepseek/deepseek-r1-distill-llama-70b # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --skip_viz --model deepseek/deepseek-r1-distill-qwen-1.5b # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --skip_viz --model deepseek/deepseek-r1-distill-qwen-14b # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --skip_viz --model deepseek/deepseek-r1-distill-qwen-32b # Available in OpenRouter
# Instruct models
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --skip_viz --model meta-llama/llama-3.1-8b-instruct # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --skip_viz --model meta-llama/llama-3.3-70b-instruct # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --skip_viz --model Qwen/Qwen2.5-1.5B-Instruct # Not available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --skip_viz --model Qwen/Qwen2.5-14B-Instruct # Not available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --skip_viz --model Qwen/Qwen2.5-32B-Instruct # Not available in OpenRouter
# Make visualization, any model will do
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --compute_from_json --model Qwen/Qwen2.5-32B-Instruct