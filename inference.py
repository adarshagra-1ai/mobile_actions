import re
import os
import torch
from datetime import datetime
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

def load_env(env_path):
    if not os.path.exists(env_path):
        raise FileNotFoundError(f".env file not found at: {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

load_env("/home/adarshagrahari/ADARSH/Project/mobile_actions/.env")

BASE_MODEL   = "google/functiongemma-270m-it"
ADAPTER_PATH = "/home/adarshagrahari/ADARSH/Project/mobile_actions/model"
HF_TOKEN     = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file.")

TOOLS_TEXT = """<start_function_declaration>declaration:turn_on_flashlight{description:<escape>Turns the flashlight on.<escape>,parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>
<start_function_declaration>declaration:turn_off_flashlight{description:<escape>Turns the flashlight off.<escape>,parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>
<start_function_declaration>declaration:open_wifi_settings{description:<escape>Opens WiFi settings.<escape>,parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>
<start_function_declaration>declaration:create_calendar_event{description:<escape>Creates a calendar event.<escape>,parameters:{properties:{datetime:{description:<escape>Date and time in YYYY-MM-DDTHH:MM:SS format<escape>,type:<escape>STRING<escape>},title:{description:<escape>Title of the event<escape>,type:<escape>STRING<escape>}},required:[<escape>datetime<escape>,<escape>title<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>
<start_function_declaration>declaration:show_map{description:<escape>Shows a map for a location.<escape>,parameters:{properties:{query:{description:<escape>Location or address<escape>,type:<escape>STRING<escape>}},required:[<escape>query<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>
<start_function_declaration>declaration:send_email{description:<escape>Sends an email.<escape>,parameters:{properties:{body:{description:<escape>Body of the email<escape>,type:<escape>STRING<escape>},subject:{description:<escape>Subject line<escape>,type:<escape>STRING<escape>},to:{description:<escape>Recipient email address<escape>,type:<escape>STRING<escape>}},required:[<escape>to<escape>,<escape>subject<escape>,<escape>body<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>
<start_function_declaration>declaration:create_contact{description:<escape>Creates a new contact.<escape>,parameters:{properties:{email:{description:<escape>Email address<escape>,type:<escape>STRING<escape>},first_name:{description:<escape>First name<escape>,type:<escape>STRING<escape>},last_name:{description:<escape>Last name<escape>,type:<escape>STRING<escape>},phone_number:{description:<escape>Phone number<escape>,type:<escape>STRING<escape>}},required:[<escape>first_name<escape>,<escape>last_name<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>"""

def load_model():
    print("\nLogging in to Hugging Face...")
    login(HF_TOKEN)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading base model (this takes 30-60 seconds on CPU)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    print("Model ready.\n")
    return model, tokenizer

def build_prompt(user_prompt, current_time):
    return (
        f"<bos><start_of_turn>developer\n"
        f"Current date and time given in YYYY-MM-DDTHH:MM:SS format: {current_time}. "
        f"You are a model that can do function calling with the following functions\n"
        f"{TOOLS_TEXT}<end_of_turn>\n"
        f"<start_of_turn>user\n"
        f"{user_prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def run(model, tokenizer, user_prompt):
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    prompt = build_prompt(user_prompt, current_time)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=[1, 50],
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)

    functions = re.findall(r'call:([a-zA-Z_]+)\{', raw)
    calls     = re.findall(r'call:[a-zA-Z_]+\{[^}]*\}', raw)

    return functions, calls

def format_output(functions, calls):
    if not functions:
        return "  No function call detected. Try rephrasing your command."

    lines = []
    for i, (fn, call) in enumerate(zip(functions, calls), 1):
        args_str = re.search(r'\{(.*)\}', call, re.DOTALL)
        args_raw = args_str.group(1) if args_str else ""
        pairs = re.findall(r'(\w+):<escape>(.*?)<escape>', args_raw)

        lines.append(f"  Action {i}: {fn}")
        if pairs:
            for key, val in pairs:
                lines.append(f"    {key}: {val}")
        else:
            lines.append(f"    (no arguments)")

    return "\n".join(lines)

def main():
    print("=" * 55)
    print("  FunctionGemma 270M — Mobile Actions Agent")
    print("  Fine-tuned on Mobile Actions | 100% accuracy")
    print("=" * 55)
    print("\nSupported actions:")
    print("  - Turn on/off flashlight")
    print("  - Open WiFi settings")
    print("  - Create calendar events")
    print("  - Show map / navigate")
    print("  - Send email")
    print("  - Create contacts")
    print("\nType 'quit' or 'exit' to stop.")
    print("Type 'help' to see example commands.\n")

    model, tokenizer = load_model()

    examples = [
        "Turn on the flashlight",
        "Schedule a meeting tomorrow at 3pm titled Team Sync",
        "Add John Smith to contacts, phone 9876543210",
        "Navigate to the nearest hospital",
        "Send email to alice@example.com about the project update",
        "Turn on flashlight and open wifi settings",
    ]

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye.")
                break

            if user_input.lower() == "help":
                print("\nExample commands:")
                for ex in examples:
                    print(f"  • {ex}")
                print()
                continue

            print("Thinking...")
            functions, calls = run(model, tokenizer, user_input)

            print("\nResult:")
            print(format_output(functions, calls))
            print()

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"Error: {e}\n")
            continue

if __name__ == "__main__":
    main()
