import json
import logging
import os
from typing import List, Dict
from tqdm import tqdm

from experiment import benchmark as benchmarklib
from llm_toolkit import models, prompt_builder, prompts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
current_dir = os.path.dirname(os.path.abspath(__file__))

class DocumentationPromptBuilder(prompt_builder.PromptBuilder):
    def __init__(self, model: models.LLM, target_lib_name: str):
        super().__init__(model)
        self.target_lib_name = target_lib_name

    def build(self, function_name: str, definition: str, signature: str) -> prompts.Prompt:
        system_prompt_path = os.path.join(current_dir, 'prompts', 'template_xml', 'insight_system_prompt.txt')
        user_prompt_path = os.path.join(current_dir, 'prompts', 'template_xml', 'insight_user_prompt.txt')

        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            system_prompt = file.read()

        with open(user_prompt_path, 'r', encoding='utf-8') as file:
            instruction_template = file.read()

        system_prompt_formatted = system_prompt.replace("{target_lib_name}", self.target_lib_name)
        instruction_formatted = instruction_template.replace("{function_name}", function_name)
        instruction_formatted = instruction_formatted.replace("{definition}", definition)
        instruction_formatted = instruction_formatted.replace("{signature}", signature)

        prompt = self._model.prompt_type()()
        prompt.add_priming(system_prompt_formatted)
        prompt.add_problem(instruction_formatted)

        return prompt

def load_function_definitions(library_name: str) -> Dict:
    """Load function definitions from JSON file."""
    json_file = os.path.join("test_parsing", library_name, "function_definitions.json")
    if not os.path.exists(json_file):
        logging.error(f"Function definitions file not found: {json_file}")
        return {}
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading function definitions: {e}")
        return {}

def save_function_definitions(library_name: str, data: Dict):
    """Save function definitions to JSON file."""
    json_file = os.path.join("test_parsing", library_name, "function_definitions.json")
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving function definitions: {e}")

def analyze_functions(model: models.LLM, target_lib_name: str, output_dir: str):
    """
    Analyze functions and update their definitions with summaries.
    """
    prompt_builder = DocumentationPromptBuilder(model, target_lib_name)
    
    # Load function definitions
    function_defs = load_function_definitions(target_lib_name)
    if not function_defs:
        logging.error("No function definitions found")
        return
    
    # Process each function
    for function_name, function_info in tqdm(function_defs.items(), desc="Analyzing Functions"):
        try:
            # Get function definition and signature
            definition = function_info.get('definition', '')
            signature = function_info.get('signature', '')
            
            if not definition or not signature:
                logging.warning(f"Missing definition or signature for function: {function_name}")
                continue
            
            # Generate analysis
            prompt = prompt_builder.build(function_name, definition, signature)
            model.query_llm(prompt, output_dir)
            
            # Read the response from the rawoutput file
            response_file = os.path.join(output_dir, "01.rawoutput")
            if not os.path.exists(response_file):
                logging.error(f"Response file not found for {function_name}")
                continue
                
            with open(response_file, "r", encoding='utf-8') as f:
                summary = f.read()
            
            # Update function info with summary
            function_info['summary'] = summary
            
            # Save updated definitions after each function
            save_function_definitions(target_lib_name, function_defs)
            
            # Remove the rawoutput file
            os.remove(response_file)
            
            logging.info(f"Successfully analyzed function: {function_name}")
            
        except Exception as e:
            logging.error(f"Failed to analyze function {function_name}: {e}")
            continue

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze functions and generate insights')
    parser.add_argument('--library', '-l', required=True, help='Library name')
    parser.add_argument('--model', '-m', default='gpt-4o-mini', 
                      choices=['gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo'],
                      help='LLM model to use (default: gpt-4o-mini)')
    args = parser.parse_args()
    
    target_lib_name = args.library
    output_dir = os.path.join("test_parsing", target_lib_name, "trace_file", "function_insights")
    os.makedirs(output_dir, exist_ok=True)
    
    model = models.LLM.setup(
        ai_binary=None,
        name=args.model
    )
    
    analyze_functions(model, target_lib_name, output_dir)

if __name__ == '__main__':
    main()