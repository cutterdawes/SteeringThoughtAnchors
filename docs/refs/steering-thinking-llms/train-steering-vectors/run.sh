python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --save_every 1 --max_tokens 1000 --n_samples 500 --seed 42 --batch_size 64
python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --save_every 1 --max_tokens 1000 --n_samples 500 --seed 42 --batch_size 32
python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --save_every 1 --max_tokens 1000 --n_samples 500 --seed 42 --batch_size 16

python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --save_every 1 --n_samples 500 --seed 42 --batch_size 16
python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --save_every 1 --n_samples 500 --seed 42 --batch_size 4
python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --save_every 1 --n_samples 500 --seed 42 --batch_size 2
