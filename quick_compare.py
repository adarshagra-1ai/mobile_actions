"""
quick_compare.py — interactive prompt comparison
Run: python quick_compare.py
Type a prompt, see response + time from each loaded model.
Type 'quit' to exit.
"""

import re, os, time, torch
from datetime import datetime
from pathlib import Path

BASE_MODEL   = "google/functiongemma-270m-it"
ADAPTER_PATH = os.path.expanduser("~/ADARSH/Project/mobile_actions/model")
GGUF_PATH    = Path("models/functiongemma-270m-it.Q8_0.gguf")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

CURRENT_TIME = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

TOOLS_TEXT = (
    "<start_function_declaration>declaration:turn_on_flashlight"
    "{description:<escape>Turns the flashlight on.<escape>,"
    "parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
    "<start_function_declaration>declaration:turn_off_flashlight"
    "{description:<escape>Turns the flashlight off.<escape>,"
    "parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
    "<start_function_declaration>declaration:open_wifi_settings"
    "{description:<escape>Opens WiFi settings.<escape>,"
    "parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
    "<start_function_declaration>declaration:create_calendar_event"
    "{description:<escape>Creates a calendar event.<escape>,"
    "parameters:{properties:{"
    "datetime:{description:<escape>Date and time in YYYY-MM-DDTHH:MM:SS format<escape>,type:<escape>STRING<escape>},"
    "title:{description:<escape>Title of the event<escape>,type:<escape>STRING<escape>}},"
    "required:[<escape>datetime<escape>,<escape>title<escape>],"
    "type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
    "<start_function_declaration>declaration:show_map"
    "{description:<escape>Shows a map for a location.<escape>,"
    "parameters:{properties:{query:{description:<escape>Location or address<escape>,type:<escape>STRING<escape>}},"
    "required:[<escape>query<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
    "<start_function_declaration>declaration:send_email"
    "{description:<escape>Sends an email.<escape>,"
    "parameters:{properties:{"
    "body:{description:<escape>Body of the email<escape>,type:<escape>STRING<escape>},"
    "subject:{description:<escape>Subject line<escape>,type:<escape>STRING<escape>},"
    "to:{description:<escape>Recipient email address<escape>,type:<escape>STRING<escape>}},"
    "required:[<escape>to<escape>,<escape>subject<escape>,<escape>body<escape>],"
    "type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
    "<start_function_declaration>declaration:create_contact"
    "{description:<escape>Creates a new contact.<escape>,"
    "parameters:{properties:{"
    "email:{description:<escape>Email address<escape>,type:<escape>STRING<escape>},"
    "first_name:{description:<escape>First name<escape>,type:<escape>STRING<escape>},"
    "last_name:{description:<escape>Last name<escape>,type:<escape>STRING<escape>},"
    "phone_number:{description:<escape>Phone number<escape>,type:<escape>STRING<escape>}},"
    "required:[<escape>first_name<escape>,<escape>last_name<escape>],"
    "type:<escape>OBJECT<escape>}}<end_function_declaration>"
)

def build_prompt(user_prompt, add_bos=True):
    bos = "<bos>" if add_bos else ""
    return (
        f"{bos}<start_of_turn>developer\n"
        f"Current date and time given in YYYY-MM-DDTHH:MM:SS format: {CURRENT_TIME}. "
        f"You are a model that can do function calling with the following functions\n"
        f"{TOOLS_TEXT}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def parse(raw):
    """Return a clean human-readable result from model raw output."""
    fn_match = re.search(r'call:([a-zA-Z_]+)', raw)
    if not fn_match:
        return "(no function detected)"
    fn = fn_match.group(1)
    args = re.findall(r'(\w+):<escape>(.*?)<escape>', raw)
    # Filter out hallucinated/None args
    real_args = [(k, v) for k, v in args if v and v != "None"]
    lines = [fn]
    for k, v in real_args:
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)

# ── Load models ───────────────────────────────────────────────────────────────

models = {}

print("Loading Float32 LoRA...", flush=True)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import login
    if HF_TOKEN:
        login(HF_TOKEN, add_to_git_credential=False)
    _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    _base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="cpu", dtype=torch.float32, attn_implementation="eager")
    _lora = PeftModel.from_pretrained(_base, ADAPTER_PATH)
    _lora.eval()

    def lora_run(prompt):
        inputs = _tok(build_prompt(prompt), return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = _lora.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=200, do_sample=False,
                pad_token_id=0, eos_token_id=[1, 50])
        elapsed = time.perf_counter() - t0
        raw = _tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        return parse(raw), elapsed

    models["Float32 LoRA"] = lora_run
    print("  Float32 LoRA ready.")
except Exception as e:
    print(f"  Float32 LoRA FAILED: {e}")

print("Loading GGUF Q8_0...", flush=True)
try:
    from llama_cpp import Llama
    _gguf = Llama(model_path=str(GGUF_PATH), n_ctx=2048,
                  n_threads=os.cpu_count(), verbose=False)

    def gguf_run(prompt):
        t0 = time.perf_counter()
        out = _gguf(
            build_prompt(prompt, add_bos=False),
            max_tokens=200, temperature=0.0, echo=False,
            stop=["<end_function_call>", "<start_function_response>",
                  "<end_of_turn>", "\n<start_of_turn>"])
        elapsed = time.perf_counter() - t0
        raw = out["choices"][0]["text"]
        return parse(raw), elapsed

    models["GGUF Q8_0"] = gguf_run
    print("  GGUF Q8_0 ready.")
except Exception as e:
    print(f"  GGUF Q8_0 FAILED: {e}")

print()
if not models:
    print("No models loaded. Exiting.")
    exit(1)

loaded = ", ".join(models)
print(f"Models loaded: {loaded}")
print("Type a prompt and press Enter. Type 'quit' to exit.\n")

# ── Interactive loop ──────────────────────────────────────────────────────────

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        print("Bye.")
        break

    print()
    for name, run_fn in models.items():
        result, elapsed = run_fn(user_input)
        print(f"[{name}]  {elapsed:.2f}s")
        print(result)
        print()