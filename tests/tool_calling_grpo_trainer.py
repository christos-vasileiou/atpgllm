from trl import GRPOTrainer
import torch
import copy
from accelerate.utils import gather_object
from trl.data_utils import is_conversational

class ToolCallingGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass with custom tool calling support.
    """
    
    def __init__(self, *args, tool_functions: list = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_functions = {tool_function.__name__: tool_function for tool_function in tool_functions} or {}
    
    def _generate(self, prompts: list):
        """
        Override the generation function to implement custom tool calling.
        
        This method is called by _generate_and_score_completions and must return:
        - prompt_ids: list of lists of token IDs for prompts
        - completion_ids: list of lists of token IDs for completions
        - tool_mask: list of lists (1 for model tokens, 0 for tool result tokens) or None
        - completions: list of decoded completions (strings or message dicts)
        - total_completion_tokens: total tokens across all completions (for DAPO loss)
        - logprobs: list of lists of log probabilities (or None if not using vLLM IS)
        - extra_fields: dict of extra fields to pass to reward functions
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)
        
        # Step 1: Initial generation (reuse parent's single-turn generation)
        prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)
        
        # Step 2: Decode completions
        if is_conversational({"prompt": prompts[0]}):
            contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            completions = [[{"role": "assistant", "content": content}] for content in contents]
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Step 3: Implement your tool calling loop here
        # Parse tool calls from completions, execute tools, regenerate if needed
        tool_mask = self._custom_tool_call_loop(
            prompts, 
            prompt_ids, 
            completion_ids, 
            completions, 
            logprobs
        )
        
        # Step 4: Compute metrics (copied from parent)
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        if tool_mask is not None:
            completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
        else:
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_completion_tokens = agg_completion_lengths.sum()
        
        # Log metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (agg_prompt_lengths.sum() + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())
        
        # Check for truncated sequences
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        
        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
        )
    
    def _custom_tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs):
        """
        Implement your custom tool calling logic here.
        
        Returns:
            tool_mask: list of lists where 1 = model token, 0 = tool result token
                       Return None if no tool calling was performed
        """
        tool_mask = [[1] * len(ids) for ids in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        
        # Example: Parse tool calls from completions
        for idx, completion in enumerate(completions):
            tool_call = self._parse_tool_call(completion)
            if tool_call:
                # Execute the tool
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})
                
                if tool_name in self.tool_functions:
                    try:
                        result = self.tool_functions[tool_name](**tool_args)
                    except Exception as e:
                        result = {"error": str(e)}
                    
                    # Here you would:
                    # 1. Append tool result to the prompt
                    # 2. Regenerate completion
                    # 3. Update completion_ids, completions, tool_mask
                    # 4. Potentially loop if more tool calls are made
        
        return tool_mask if any(0 in mask for mask in tool_mask) else None
    
    def _parse_tool_call(self, completion):
        """
        Parse a tool call from model completion.
        Override this for your specific tool call format.
        """
        import re
        import json
        
        # Example: Parse <tool_call>{"name": ..., "arguments": ...}</tool_call>
        if isinstance(completion, list):
            # Conversational format
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = completion
        
        match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None