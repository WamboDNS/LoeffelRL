from vllm import LLM, SamplingParams

def main():
    # 1. Define model path or name (local or Hugging Face)
    model_path = "wambosec/Qwen2.5-7B-Instruct-spoon-language-SFT"  # Replace with your model

    # 2. Load the model
    print("Loading model...")
    llm = LLM(model=model_path)

    # 3. Define a simple prompt and sampling parameters
    prompt = "What is the capital of France?"
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=32)

    # 4. Run inference
    print("Running inference...")
    outputs = llm.generate([prompt], sampling_params)

    # 5. Print result
    print("Output:")
    print(outputs[0].outputs[0].text.strip())

if __name__ == "__main__":
    main()
