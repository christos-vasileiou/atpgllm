import re
import json

def revert_qwen2_5_template(chat_string):
    """
    Parses a Qwen 2.5 Instruct string back into a list of messages,
    handling tool calls, tool responses, and system prompt stripping.
    """
    # 1. Split by the standard ChatML end token
    # We filter empty segments (often caused by the final trailing newline)
    segments = [s.strip() for s in chat_string.split('<|im_end|>') if s.strip()]
    
    messages = []
    
    # Regex to extract Role and Content from a block: <|im_start|>role\nContent
    # We use DOTALL so (.) matches newlines
    block_pattern = re.compile(r'<\|im_start\|>(.*?)\n(.*)', re.DOTALL)
    
    # Regex to find tool calls inside assistant messages
    tool_call_pattern = re.compile(r'<tool_call>\n(.*?)\n</tool_call>', re.DOTALL)
    
    # Regex to find tool responses inside "user" messages
    tool_response_pattern = re.compile(r'<tool_response>\n(.*?)\n</tool_response>', re.DOTALL)
    
    # The static boilerplate text Qwen adds when tools are enabled
    # We use the start of it to identify where to cut.
    tool_system_marker = "\n\n# Tools\n\nYou may call one or more functions"
    
    for segment in segments:
        match = block_pattern.search(segment)
        if not match:
            continue
            
        role = match.group(1).strip()
        content = match.group(2) # Don't strip yet, content might need precise parsing
        
        # --- Handle SYSTEM ---
        if role == 'system':
            # Check if the tool definition boilerplate is present
            if tool_system_marker in content:
                # Keep only the text BEFORE the marker
                clean_content = content.split(tool_system_marker)[0]
                messages.append({"role": "system", "content": clean_content})
            else:
                messages.append({"role": "system", "content": content})
        
        # --- Handle ASSISTANT (Text + Tool Calls) ---
        elif role == 'assistant':
            tool_calls = []
            
            # Find all XML tool calls
            found_calls = tool_call_pattern.findall(content)
            for json_str in found_calls:
                try:
                    call_data = json.loads(json_str)
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": call_data["name"],
                            "arguments": json.dumps(call_data["arguments"]) # Standardize as string
                        }
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse tool json: {json_str}")
            
            # Remove the tool call XML from the content to get the "text" part
            text_content = tool_call_pattern.sub('', content).strip()
            
            msg_obj = {"role": "assistant"}
            if text_content:
                msg_obj["content"] = text_content
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            
            messages.append(msg_obj)
        
        # --- Handle USER (might be actual User OR Tool Outputs) ---
        elif role == 'user':
            # Check if this is actually a disguised tool response block
            tool_responses = list(tool_response_pattern.finditer(content))
            
            if tool_responses:
                # This block is a sequence of tool outputs
                for tr in tool_responses:
                    messages.append({
                        "role": "tool",
                        # We don't have the tool_call_id in the string, 
                        # so we recover the content. 
                        # (Real Qwen context tracking requires mapping IDs externally)
                        "content": tr.group(1).strip()
                    })
            else:
                # It's just a normal user
                messages.append({"role": "user", "content": content.strip()})
    
    return messages
