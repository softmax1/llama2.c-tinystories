from transformers import pipeline

pipe = pipeline("text-generation", "./out/huggingface/softmax0-15m-hf")
print(pipe("Once upon a time, Alice", max_new_tokens=256))


# README warning: converting PyTorch llama2.c -> HuggingFace model works
# BUT generated tokens (and logprobs) don't align -- prboably some
# very subtle detail in matmul that I'll never fix.
# Just accept this limitation, and sanity check that models are
# "close enough" via ppl, completion examples, etc.

# Examples: softmax0 15M 2023-12 (self-trained)

# llama2.c: 1+1+1+1+1+1= | . She was so happy to see her friend, so she ran to her ...
# HuggingFace: 1+1+1+1+1+1= | . Lily was so happy and proud of herself for being brave.

# llama2.c: Once upon a time | , Alice was walking in the park. She saw a big, red ...
# HuggingFace: Once upon a time | , Alice was walking through the forest. Suddenly, ...
