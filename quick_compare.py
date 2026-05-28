"""
quick_compare.py — interactive prompt comparison across 3 paths:
  • Float32 LoRA (PyTorch + PEFT)  -> shows function name + args
  • GGUF Q8_0   (llama.cpp)        -> shows function name + args
  • LiteRT Q8   (Google LiteRT-LM) -> shows natural-language response

Run: python quick_compare.py
"""

import re, os, time, torch
from datetime import datetime
from pathlib import Path

# ── Load .env first ───────────────────────────────────────────────────────────

ENV_PATH = Path(os.path.expanduser("~/ADARSH/Project/mobile_actions/.env"))

def load_env(env_path):
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

load_env(ENV_PATH)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL   = "google/functiongemma-270m-it"
ADAPTER_PATH = os.path.expanduser("~/ADARSH/Project/mobile_actions/model")
GGUF_PATH    = Path(os.path.expanduser("~/ADARSH/Project/mobile_actions/models/functiongemma-270m-it.Q8_0.gguf"))
LITERT_PATH  = Path(os.path.expanduser("~/ADARSH/Project/mobile_actions/models/mobile_actions_q8_ekv1024.litertlm"))
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

CURRENT_TIME = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# ── Tools in native FunctionGemma serialization (for paths 1 and 2) ───────────

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

# ── Tools as Python functions (for LiteRT) ────────────────────────────────────
# LiteRT reads docstrings + type hints to build its tool schema.

def turn_on_flashlight() -> str:
    """Turns the flashlight on."""
    return "Flashlight turned on."

def turn_off_flashlight() -> str:
    """Turns the flashlight off."""
    return "Flashlight turned off."

def open_wifi_settings() -> str:
    """Opens WiFi settings."""
    return "WiFi settings opened."

def create_calendar_event(datetime: str, title: str) -> str:
    """Creates a calendar event.

    Args:
        datetime: Date and time in YYYY-MM-DDTHH:MM:SS format.
        title: Title of the event.
    """
    return f"Event '{title}' created at {datetime}."

def show_map(query: str) -> str:
    """Shows a map for a location.

    Args:
        query: Location or address.
    """
    return f"Showing map for: {query}."

def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email.

    Args:
        to: Recipient email address.
        subject: Subject line.
        body: Body of the email.
    """
    return f"Email sent to {to}."

def create_contact(first_name: str, last_name: str,
                   phone_number: str = "", email: str = "") -> str:
    """Creates a new contact.

    Args:
        first_name: First name.
        last_name: Last name.
        phone_number: Phone number.
        email: Email address.
    """
    return f"Contact {first_name} {last_name} created."

LITERT_TOOLS = [
    turn_on_flashlight,
    turn_off_flashlight,
    open_wifi_settings,
    create_calendar_event,
    show_map,
    send_email,
    create_contact,
]

# ── Shared helpers ────────────────────────────────────────────────────────────

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

def parse_functiongemma(raw):
    """Parse FunctionGemma's call:name{key:<escape>val<escape>} format."""
    fn_match = re.search(r'call:([a-zA-Z_]+)', raw)
    if not fn_match:
        return "(no function detected)"
    fn = fn_match.group(1)
    args = re.findall(r'(\w+):<escape>(.*?)<escape>', raw)
    real_args = [(k, v) for k, v in args if v and v != "None"]
    lines = [fn]
    for k, v in real_args:
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)

# ── Pre-flight checks ─────────────────────────────────────────────────────────

print("=" * 60)
print("  Pre-flight checks")
print("=" * 60)

if HF_TOKEN:
    print("  [OK]   HF_TOKEN loaded from .env")
else:
    print(f"  [FAIL] HF_TOKEN not found in {ENV_PATH}")

adapter_file = Path(ADAPTER_PATH) / "adapter_model.safetensors"
if adapter_file.exists():
    size_mb = adapter_file.stat().st_size / (1024 * 1024)
    print(f"  [OK]   LoRA adapter found ({size_mb:.1f} MB)")
else:
    print(f"  [FAIL] LoRA adapter missing at {adapter_file}")

if GGUF_PATH.exists():
    size_mb = GGUF_PATH.stat().st_size / (1024 * 1024)
    print(f"  [OK]   GGUF Q8_0 file found ({size_mb:.1f} MB)")
else:
    print(f"  [FAIL] GGUF file missing at {GGUF_PATH}")

if LITERT_PATH.exists():
    size_mb = LITERT_PATH.stat().st_size / (1024 * 1024)
    print(f"  [OK]   LiteRT Q8 file found ({size_mb:.1f} MB)")
else:
    print(f"  [FAIL] LiteRT file missing at {LITERT_PATH}")

for pkg, imp in [
    ("transformers",     "transformers"),
    ("peft",             "peft"),
    ("llama-cpp-python", "llama_cpp"),
    ("litert-lm",        "litert_lm"),
]:
    try:
        __import__(imp)
        print(f"  [OK]   package: {pkg}")
    except ImportError:
        print(f"  [FAIL] package: {pkg} — run: pip install {pkg}")

print("=" * 60)
print()

# ── Load models ───────────────────────────────────────────────────────────────

models = {}

# PATH 1 — Float32 LoRA
print("Loading Float32 LoRA...", flush=True)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import login
    if HF_TOKEN:
        login(HF_TOKEN, add_to_git_credential=False)
    _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    _base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="cpu", dtype=torch.float32,
        attn_implementation="eager")
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
        raw = _tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=False)
        return parse_functiongemma(raw), elapsed

    models["Float32 LoRA"] = lora_run
    print("  Float32 LoRA ready.\n")
except Exception as e:
    print(f"  Float32 LoRA FAILED: {e}\n")

# PATH 2 — GGUF Q8_0
print("Loading GGUF Q8_0...", flush=True)
try:
    from llama_cpp import Llama
    _gguf = Llama(
        model_path=str(GGUF_PATH),
        n_ctx=2048,
        n_threads=os.cpu_count(),
        verbose=False)

    def gguf_run(prompt):
        t0 = time.perf_counter()
        out = _gguf(
            build_prompt(prompt, add_bos=False),
            max_tokens=200, temperature=0.0, echo=False,
            stop=["<end_function_call>", "<start_function_response>",
                  "<end_of_turn>", "\n<start_of_turn>"])
        elapsed = time.perf_counter() - t0
        raw = out["choices"][0]["text"]
        return parse_functiongemma(raw), elapsed

    models["GGUF Q8_0"] = gguf_run
    print("  GGUF Q8_0 ready.\n")
except Exception as e:
    print(f"  GGUF Q8_0 FAILED: {e}\n")

# PATH 3 — LiteRT Q8
# LiteRT executes tools internally and only streams the final natural-language
# answer back. We return that text — you can read it and verify whether the
# right action was taken.
print("Loading LiteRT Q8...", flush=True)
try:
    import litert_lm
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    _litert_engine = litert_lm.Engine(
        str(LITERT_PATH),
        backend=litert_lm.Backend.CPU(),
    )
    _litert_conv = _litert_engine.create_conversation(
        messages=[litert_lm.Message.system(
            f"Current date and time given in YYYY-MM-DDTHH:MM:SS format: {CURRENT_TIME}. "
            "You are a mobile actions agent."
        )],
        tools=LITERT_TOOLS,
    )

    def litert_run(prompt):
        t0 = time.perf_counter()
        chunks = []
        for chunk in _litert_conv.send_message_async(prompt):
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    chunks.append(item.get("text", ""))
        elapsed = time.perf_counter() - t0
        full_text = "".join(chunks).strip()
        return full_text if full_text else "(no response)", elapsed

    models["LiteRT Q8"] = litert_run
    print("  LiteRT Q8 ready.\n")
except Exception as e:
    print(f"  LiteRT Q8 FAILED: {e}\n")

# ── Interactive loop ──────────────────────────────────────────────────────────

if not models:
    print("No models loaded. Exiting.")
    exit(1)

print("=" * 60)
print(f"  Models loaded: {', '.join(models)}")
print("  Type a prompt and press Enter. Type 'quit' to exit.")
print("=" * 60)
print()

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
        try:
            result, elapsed = run_fn(user_input)
            print(f"[{name}]  {elapsed:.2f}s")
            print(result)
        except Exception as e:
            print(f"[{name}]  ERROR: {e}")
        print()