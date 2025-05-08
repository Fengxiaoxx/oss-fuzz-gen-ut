from clang.cindex import CompilationDatabase
from clang import cindex
from tqdm import tqdm
import logging
import os
import json
import argparse
import sys
from typing import Tuple, List, Dict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_compile_commands(path):
    comp_db = CompilationDatabase.fromDirectory(path)
    all_compile_commands = comp_db.getAllCompileCommands()
    return all_compile_commands


def process_compile_args(cmd):
    """
    处理编译参数列表，移除与解析无关的选项，并将优化级别设置为 -O0。

    参数：
        cmd (clang.cindex.CompileCommand): 包含编译参数的编译命令。

    返回：
        list: 处理后的编译参数列表。
    """
    # 移除第一个参数（通常是编译器名称）
    compile_args = list(cmd.arguments)[1:]

    # 移除 '-c' 参数
    compile_args = [arg for i, arg in enumerate(compile_args)
                    if arg != '-c' and (i == 0 or compile_args[i - 1] != '-c')]

    # 移除 '-o' 及其后续参数
    cleaned_args = []
    skip_next = False
    for arg in compile_args:
        if skip_next:
            skip_next = False
            continue
        if arg == '-o':
            skip_next = True  # 跳过 '-o' 的下一个参数
        else:
            cleaned_args.append(arg)

    # 查找并移除现有的优化参数（如 '-O1', '-O2', '-O3', '-Os', '-Ofast'）
    optimization_flags = ['-O0', '-O1', '-O2', '-O3', '-Os', '-Ofast']
    cleaned_args = [arg for arg in cleaned_args if arg not in optimization_flags]

    # 添加 '-O0' 参数以禁用优化
    cleaned_args.append('-O0')

    return cleaned_args

def extract_lines(filename: str, start_line: int, end_line: int) -> str:
    # 检查 start_line 和 end_line 是否为整数且大于 0
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        raise TypeError("start_line 和 end_line 必须是整数")
    if start_line <= 0 or end_line <= 0:
        raise ValueError("start_line 和 end_line 必须大于 0")
    if start_line > end_line:
        raise ValueError("start_line 不能大于 end_line")

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        # 确保 end_line 不超过文件总行数
        end_line = min(end_line, len(lines))
        selected_lines = lines[start_line - 1:end_line]
        return ''.join(selected_lines)

def get_function_signature(cursor: cindex.Cursor) -> str:
    file_path_decl = cursor.location.file.name
    start_line = cursor.extent.start.line
    end_line = cursor.extent.end.line

    raw_comment = cursor.raw_comment
    decl = extract_lines(file_path_decl, start_line, end_line)

    if raw_comment:
        function_declaration = raw_comment + "\n" + decl
    else:
        function_declaration = decl
    return function_declaration

def parse_source_file_and_get_cursor(cmd: cindex.CompileCommand) -> Tuple[cindex.Cursor, str, List[str]]:
    src_file = cmd.filename  # 获取源文件路径

    # 检查文件是否存在
    if not os.path.exists(src_file):
        logging.error(f"Source file {src_file} does not exist.")
        raise FileNotFoundError(f"Source file {src_file} does not exist.")

    # 处理编译参数
    compile_args = process_compile_args(cmd)
    # 创建索引对象
    index = cindex.Index.create()

    try:
        # 解析源文件，生成 TranslationUnit
        logging.info(f"Parsing source file: {src_file}")
        tu = index.parse(
            src_file,
            args=compile_args,
            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
        logging.info(f"Successfully parsed source file: {src_file}")
    except cindex.TranslationUnitLoadError as e:
        logging.error(f"Failed to parse source file {src_file}: {e}")
        raise RuntimeError(f"Failed to parse source file {src_file}: {e}")
    finally:
        # 释放索引对象资源
        del index

    return tu.cursor, src_file, compile_args


def get_function_set(function_txt: str):
    """
    从文本文件中读取函数名列表
    
    参数:
        function_txt: 包含函数名列表的文件路径
        
    返回:
        set: 函数名集合
    """
    function_set = set()
    if not os.path.exists(function_txt):
        logger.error(f"Function list file does not exist: {function_txt}")
        return function_set
        
    try:
        with open(function_txt, 'r') as f:
            for line in f:
                function_name = line.strip()
                if function_name:
                    function_set.add(function_name)
        logger.info(f"Loaded {len(function_set)} functions from {function_txt}")
        return function_set
    except Exception as e:
        logger.error(f"Error loading function list file: {e}")
        return function_set
    
def get_relative_path(absolute_path: str, library_name: str) -> str:
    """将绝对路径转换为相对于target_lib/{库名}的相对路径"""
    base_dir = os.path.join("target_lib", library_name)
    try:
        return os.path.relpath(absolute_path, base_dir)
    except ValueError:
        return absolute_path

def traverse_cursor(cursor: cindex.Cursor, src_file: str, function_set: set, library_name: str) -> Dict[str, dict]:
    """遍历游标，找到目标函数的定义和声明"""
    function_defs = {}
    functions_remaining = set(function_set)
    function_kind = {cindex.CursorKind.FUNCTION_DECL, cindex.CursorKind.CXX_METHOD, 
                    cindex.CursorKind.FUNCTION_TEMPLATE, cindex.CursorKind.CONVERSION_FUNCTION}
    
    try:
        with open(src_file, 'r') as f:
            file_lines = f.readlines()
    except Exception as e:
        logger.error(f"无法读取文件 {src_file}: {e}")
        return function_defs
    
    for child in cursor.walk_preorder():
        # 获取函数定义
        if child.kind in function_kind and child.is_definition() and child.spelling in function_set and child.location.file and child.location.file.name == src_file:
            extent = child.extent
            start_line = extent.start.line
            end_line = extent.end.line
            
            if 1 <= start_line <= len(file_lines) and start_line <= end_line <= len(file_lines):
                function_lines = file_lines[start_line-1:end_line]
                function_def = ''.join(function_lines)
                
                if len(function_def.strip()) > 0:
                    # 查找对应的函数声明
                    for decl in cursor.walk_preorder():
                        if (decl.kind == cindex.CursorKind.FUNCTION_DECL and 
                            decl.spelling == child.spelling and 
                            not decl.is_definition()):
                            function_signature = get_function_signature(decl)
                            header = get_relative_path(decl.location.file.name, library_name)
                            break
                    else:
                        # 如果没有找到声明，使用定义的信息
                        function_signature = get_function_signature(child)
                        header = get_relative_path(child.location.file.name, library_name)
                    
                    function_defs[child.spelling] = {
                        'definition': function_def,
                        'signature': function_signature,
                        'header': header
                    }
                    functions_remaining.discard(child.spelling)
                    logger.info(f"Found function definition for {child.spelling}")
                else:
                    logger.warning(f"Empty definition for {child.spelling} at lines {start_line}-{end_line}")
            else:
                logger.warning(f"Invalid line range for {child.spelling}: {start_line}-{end_line}, file has {len(file_lines)} lines")
    
    if functions_remaining:
        logger.info(f"Could not find definitions for {len(functions_remaining)} functions in {src_file}")
        
    return function_defs

def save_function_defs_to_json(library_name: str, function_defs: Dict[str, dict]):
    """将函数定义保存到JSON文件"""
    output_dir = os.path.join("test_parsing", library_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "function_definitions.json")
    with open(output_file, 'w') as f:
        json.dump(function_defs, f, indent=2)
    
    logger.info(f"Saved {len(function_defs)} function definitions to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract function definitions from source code')
    parser.add_argument('--library', '-l', required=True, help='Library name')
    parser.add_argument('--compile-commands-dir', '-c', required=True, help='Directory containing compile_commands.json')
    parser.add_argument('--functions-file', '-f', help='Path to function list txt file')
    args = parser.parse_args()
    
    # 构建函数列表文件路径
    function_txt = args.functions_file or os.path.join("test_parsing", args.library, "function_list.txt")
    
    # 加载编译命令和函数列表
    compile_commands = load_compile_commands(args.compile_commands_dir)
    function_set = get_function_set(function_txt)
    
    if not function_set:
        logger.error("No functions to extract, exiting")
        return
    
    # 提取函数定义
    all_function_defs = {}
    
    for cmd in tqdm(compile_commands, desc="Processing compile commands"):
        try:
            root_cursor, src_file, _ = parse_source_file_and_get_cursor(cmd)
            function_defs = traverse_cursor(root_cursor, src_file, function_set, args.library)
            all_function_defs.update(function_defs)
            
            if len(all_function_defs) == len(function_set):
                logger.info("Found definitions for all functions, stopping search")
                break
        except Exception as e:
            logger.error(f"Error processing file {cmd.filename}: {e}")
    
    # 保存结果
    save_function_defs_to_json(args.library, all_function_defs)
    
    # 打印结果
    print(f"\n===== Function Definition Extraction Results =====")
    print(f"Library: {args.library}")
    print(f"Total target functions: {len(function_set)}")
    print(f"Functions found: {len(all_function_defs)} ({len(all_function_defs)/len(function_set)*100:.2f}%)")
    if function_set - set(all_function_defs.keys()):
        print(f"Functions not found: {', '.join(function_set - set(all_function_defs.keys()))}")
    print(f"Results saved to: test_parsing/{args.library}/function_definitions.json")

if __name__ == "__main__":
    main()

