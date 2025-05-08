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

    def build(self, code_file: str, api_used: List[str], unit_test_code: str) -> prompts.Prompt:
        system_prompt_path = os.path.join(current_dir, 'prompts', 'template_xml', 'system_prompt.txt')
        user_prompt_path = os.path.join(current_dir, 'prompts', 'template_xml', 'user_prompt.txt')

        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            system_prompt = file.read()

        with open(user_prompt_path, 'r', encoding='utf-8') as file:
            instruction_template = file.read()

        system_prompt_formatted = system_prompt.replace("{target_lib_name}", self.target_lib_name)
        instruction_formatted = instruction_template.replace("{unit_test_code}", unit_test_code)
        instruction_formatted = instruction_formatted.replace("{api_used}", json.dumps(api_used, indent=2))

        prompt = self._model.prompt_type()()
        prompt.add_priming(system_prompt_formatted)
        prompt.add_problem(instruction_formatted)

        return prompt

def get_code_files(library_name: str) -> List[str]:
    """Get all code files from the extracted_code directory."""
    code_dir = os.path.join("test_parsing", library_name, "trace_file", "extracted_code")
    if not os.path.exists(code_dir):
        logging.error(f"Code directory not found: {code_dir}")
        return []
    
    return [f for f in os.listdir(code_dir) if os.path.isfile(os.path.join(code_dir, f))]

def load_api_mapping(library_name: str) -> Dict[str, List[str]]:
    """Load API mapping from file_api_mapping.json."""
    mapping_file = os.path.join("test_parsing", library_name, "file_api_mapping.json")
    if not os.path.exists(mapping_file):
        logging.error(f"API mapping file not found: {mapping_file}")
        return {}
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading API mapping: {e}")
        return {}

def read_code_file(library_name: str, filename: str) -> str:
    """Read content of a code file."""
    file_path = os.path.join("test_parsing", library_name, "trace_file", "extracted_code", filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading code file {filename}: {e}")
        return ""

def documentation_engineer(model: models.LLM, target_lib_name: str, output_dir: str):
    """
    Generate function documentation for each code file.
    """
    prompt_builder = DocumentationPromptBuilder(model, target_lib_name)
    
    # Get all code files
    code_files = get_code_files(target_lib_name)
    if not code_files:
        logging.error("No code files found")
        return
    
    # Load API mapping
    api_mapping = load_api_mapping(target_lib_name)
    if not api_mapping:
        logging.error("No API mapping found")
        return
    
    # Process each code file
    for code_file in tqdm(code_files, desc="Generating Documentation"):
        try:
            # Get APIs used in this file
            api_used = api_mapping.get(code_file, [])
            if not api_used:
                logging.warning(f"No APIs found for file: {code_file}")
                continue
            
            # Read code file content
            unit_test_code = read_code_file(target_lib_name, code_file)
            if not unit_test_code:
                logging.warning(f"Empty code file: {code_file}")
                continue
            
            # Generate documentation
            prompt = prompt_builder.build(code_file, api_used, unit_test_code)
            model.query_llm(prompt, output_dir)
            
            # Read the response from the rawoutput file
            response_file = os.path.join(output_dir, "01.rawoutput")
            if not os.path.exists(response_file):
                logging.error(f"Response file not found for {code_file}")
                continue
                
            with open(response_file, "r", encoding="utf-8") as f:
                response = f.read()
            
            # Save documentation with original filename
            output_path = os.path.join(output_dir, f"{code_file}")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response)
            
            # Remove the rawoutput file
            os.remove(response_file)
            
            logging.info(f"Successfully generated documentation for: {code_file}")
            
        except Exception as e:
            logging.error(f"Failed to process file {code_file}: {e}")
            continue

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate documentation for code files')
    parser.add_argument('--library', '-l', required=True, help='Library name')
    parser.add_argument('--model', '-m', default='gpt-4o-mini', 
                      choices=['gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo'],
                      help='LLM model to use (default: gpt-4o-mini)')
    args = parser.parse_args()
    
    target_lib_name = args.library
    output_dir = os.path.join("test_parsing", target_lib_name, "trace_file", "function_deps")
    os.makedirs(output_dir, exist_ok=True)
    
    model = models.LLM.setup(
        ai_binary=None,
        name=args.model
    )
    
    documentation_engineer(model, target_lib_name, output_dir)

if __name__ == '__main__':
    main()