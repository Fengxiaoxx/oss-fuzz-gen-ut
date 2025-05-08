#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from clang.cindex import Cursor, TokenKind
from clang import cindex
from collections import Counter
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_functions_from_txt(txt_file_path):
    """
    Load function list from a txt file
    
    Args:
        txt_file_path: Path to txt file with one function name per line
        
    Returns:
        Set of function names
    """
    if not os.path.exists(txt_file_path):
        logger.error(f"Function list file does not exist: {txt_file_path}")
        return set()
    
    try:
        with open(txt_file_path, 'r') as f:
            # Read each line, strip whitespace, and filter out empty lines
            function_names = set(line.strip() for line in f if line.strip())
        
        logger.info(f"Loaded {len(function_names)} functions from {txt_file_path}")
        return function_names
    
    except Exception as e:
        logger.error(f"Error loading function list file: {e}")
        return set()

def get_source_files(library_name):
    """
    Get list of extracted source code files
    
    Args:
        library_name: Name of the library
        
    Returns:
        List of source code file paths
    """
    extract_dir = os.path.join("test_parsing", library_name, "trace_file", "extracted_code")
    
    if not os.path.exists(extract_dir):
        logger.error(f"Extracted code directory does not exist: {extract_dir}")
        return []
    
    # Get all files in the directory
    files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if os.path.isfile(os.path.join(extract_dir, f))]
    logger.info(f"Found {len(files)} source files in {extract_dir}")
    return files

def parse_exec_code(language: str, file_path: str) -> Cursor:
    """
    Parse source code file and return parsing cursor
    
    Args:
        language: Programming language ('c' or 'c++')
        file_path: Path to source code file
        
    Returns:
        Parsed cursor object
    """
    # Set compilation arguments
    language = language.lower()
    if language == 'c':
        compile_args = ['-x', 'c']
    elif language in ['c++', 'cpp']:
        compile_args = ['-x', 'c++']
    else:
        logging.error(f"Unsupported language: {language}")
        raise ValueError(f"Unsupported language: {language}")

    # Check if file exists
    if not os.path.exists(file_path):
        logging.error(f"Source file {file_path} does not exist")
        raise FileNotFoundError(f"Source file {file_path} does not exist")

    # Create index object
    index = cindex.Index.create()

    try:
        # Parse source file
        tu = index.parse(
            file_path,
            args=compile_args,
            options=cindex.TranslationUnit.PARSE_INCOMPLETE
        )
    except cindex.TranslationUnitLoadError as e:
        logging.error(f"Failed to parse source file {file_path}: {e}")
        raise RuntimeError(f"Failed to parse source file {file_path}: {e}")
    finally:
        del index  # Release index object resources

    # Return parsed cursor
    return tu.cursor

def traveler_cursor(cursor: Cursor, target_function_list: set):
    """
    Traverse cursor to find target function calls
    
    Args:
        cursor: Parsing cursor
        target_function_list: Set of target function names
        
    Returns:
        Set of called target functions
    """
    target_function_call = set()
    for child in cursor.get_tokens():
        if child.kind == TokenKind.IDENTIFIER and child.spelling in target_function_list:
            target_function_call.add(child.spelling)
    return target_function_call

def guess_language(file_path):
    """
    Guess programming language based on file extension
    
    Args:
        file_path: File path
        
    Returns:
        Guessed language type
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.c']:
        return 'c'
    elif ext in ['.cpp', '.cc', '.cxx', '.hpp', '.h']:
        return 'c++'
    else:
        # Default to C
        return 'c'

def analyze_file(file_path, target_functions):
    """
    Analyze a single file for function calls
    
    Args:
        file_path: Source file path
        target_functions: Set of target functions
        
    Returns:
        Set of called target functions
    """
    language = guess_language(file_path)
    
    try:
        cursor = parse_exec_code(language, file_path)
        function_calls = traveler_cursor(cursor, target_functions)
        return function_calls
    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")
        return set()

def save_called_functions(library_name, called_functions):
    """
    Save called functions to a new file
    
    Args:
        library_name: Name of the library
        called_functions: Set of called function names
    """
    output_dir = os.path.join("test_parsing", library_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "function_list.txt")
    
    try:
        with open(output_file, 'w') as f:
            for func_name in sorted(called_functions):
                f.write(f"{func_name}\n")
        
        logger.info(f"Saved {len(called_functions)} called functions to {output_file}")
    except Exception as e:
        logger.error(f"Error saving called functions to file: {e}")

def save_file_api_mapping(library_name, file_api_mapping):
    """
    Save file to API mapping to a JSON file
    
    Args:
        library_name: Name of the library
        file_api_mapping: Dictionary mapping file names to their called APIs
    """
    output_dir = os.path.join("test_parsing", library_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "file_api_mapping.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(file_api_mapping, f, indent=2)
        
        logger.info(f"Saved file to API mapping to {output_file}")
    except Exception as e:
        logger.error(f"Error saving file to API mapping: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Find which functions in the list are actually called')
    parser.add_argument('--library', '-l', required=True, help='Library name')
    parser.add_argument('--functions-file', '-f', required=True, help='Path to txt file with function names, one per line')
    args = parser.parse_args()
    
    library_name = args.library
    functions_file = args.functions_file
    
    # 1. Load function list
    target_functions = load_functions_from_txt(functions_file)
    if not target_functions:
        logger.error("No functions found, exiting")
        sys.exit(1)
    
    # 2. Get all source files
    source_files = get_source_files(library_name)
    if not source_files:
        logger.error("No source files found, exiting")
        sys.exit(1)
    
    # 3. Track which functions are called
    called_functions = set()
    file_api_mapping = {}  # Dictionary to store file to API mapping
    
    # 4. Analyze each file
    for file_path in source_files:
        file_name = os.path.basename(file_path)
        logger.info(f"Analyzing file: {file_name}")
        
        file_calls = analyze_file(file_path, target_functions)
        
        if file_calls:
            # Add to the set of called functions
            called_functions.update(file_calls)
            # Add to file to API mapping
            file_api_mapping[file_name] = list(file_calls)
            logger.info(f"Found {len(file_calls)} function calls in {file_name}")
        else:
            logger.info(f"No function calls found in {file_name}")
    
    # 5. Save called functions to file
    save_called_functions(library_name, called_functions)
    
    # 6. Save file to API mapping
    save_file_api_mapping(library_name, file_api_mapping)
    
    # 7. Print results
    print("\n===== Function Call Analysis Results =====")
    print(f"Library: {library_name}")
    print(f"Total target functions: {len(target_functions)}")
    print(f"Total files analyzed: {len(source_files)}")
    print(f"Functions called: {len(called_functions)} ({len(called_functions)/len(target_functions)*100:.2f}%)")
    print(f"Functions not called: {len(target_functions) - len(called_functions)} ({(len(target_functions) - len(called_functions))/len(target_functions)*100:.2f}%)")
    print(f"Results saved to: test_parsing/{library_name}/function_list.txt")
    print(f"File to API mapping saved to: test_parsing/{library_name}/file_api_mapping.json")

if __name__ == "__main__":
    main() 