from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Path to your downloaded model (DeepSeek)
MODEL_PATH = "./my-python-model-final"

print("Loading model... this might take a minute...")

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    model.to("cpu")

model.eval()

# FastAPI app
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

def build_prompt_deepseek(user_input: str) -> str:
    """
    Constructs the prompt with strict system instructions for DeepSeek.
    DeepSeek Coder typically follows this format:
    {System Prompt}
    ### Instruction:
    {User Input}
    ### Response:
    """
    
    # Strict System Prompt
    system_prompt = (
        "You are a specialized Python programming assistant. "
        "Your ONLY purpose is to answer questions strictly related to the Python programming language, "
        "its libraries (like Pandas, NumPy, Django, FastAPI), and debugging. "
        "\n\nIMPORTANT RULE: If the user asks about anything unrelated to Python "
        "(such as general knowledge, history, other languages like Java/C++, cooking, or weather), "
        "you MUST refuse and reply with exactly: 'I am designed to answer Python-related questions only. I do not have data for that query.'"
    )
    
    return f"{system_prompt}\n### Instruction:\n{user_input}\n### Response:"

@app.post("/chat")
def chat(req: ChatRequest):
    # 1. Build the formatted prompt for DeepSeek
    prompt = build_prompt_deepseek(req.message)

    # 2. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2, # Low temperature keeps it focused and less likely to hallucinate
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. Decode
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 5. Clean up response
    # DeepSeek output usually contains the prompt followed by "### Response:"
    # We split by that marker to get only the new text.
    if "### Response:" in decoded_output:
        reply = decoded_output.split("### Response:")[-1].strip()
    else:
        # Fallback: if the model messed up the tag, try to return the raw output
        # (removing the system prompt if possible to keep it clean)
        reply = decoded_output.replace(prompt, "").strip()

    return {"reply": reply}