import subprocess
import json
import sys
import os
import argparse
import re
import logging
import concurrent.futures
from enum import Enum
from threading import Lock
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TestFramework(Enum):
    GTEST = "gtest"
    # 可以在这里添加其他测试框架
    # BOOST = "boost"
    # CATCH2 = "catch2"
    # UNITY = "unity"

class GTest:
    """Class for parsing Google Test framework test suites."""
    
    def __init__(self, library_name: str):
        """Initialize the GTest parser.
        
        Args:
            library_name: Name of the library being tested
        """
        self.library_name = library_name
        self.output_dir = os.path.join("test_parsing", library_name)
        self.output_file = os.path.join(self.output_dir, "test_cases.json")
        self.test_cases_file = os.path.join(self.output_dir, "test_cases.txt")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def parse_tests(self, executable_path: str):
        """Parse test suites and tests from a gtest executable.
        
        Args:
            executable_path: Path to the compiled gtest executable
            
        Returns:
            A dictionary where keys are test suite names and values are lists of test names
        """
        try:
            # Run the gtest executable with --gtest_list_tests
            logger.info(f"Running {executable_path} to get test list...")
            result = subprocess.run(
                [executable_path, '--gtest_list_tests'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output
            suites = {}
            current_suite = None
            
            for line in result.stdout.splitlines():
                if not line.startswith(' '):
                    # This is a test suite
                    current_suite = line.strip().rstrip('.')
                    suites[current_suite] = []
                else:
                    # This is a test name
                    test_name = line.strip().split()[0]
                    suites[current_suite].append(test_name)
            
            logger.info(f"Found {len(suites)} test suites")
            return suites
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running gtest executable: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing gtest output: {e}")
            return {}

    def save_to_json(self, suites):
        """Save test suites to a JSON file.
        
        Args:
            suites: Dictionary of test suites and tests
        """
        try:
            with open(self.output_file, 'w') as f:
                json.dump(suites, f, indent=2)
            logger.info(f"Successfully saved test suites to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving tests to JSON: {e}")
    
    def process_test_cases(self):
        """Process test cases from the JSON file and save to a text file.
        
        This method reads the JSON file, processes test suite names and test cases,
        and saves the processed test cases to a text file.
        """
        try:
            # Check if JSON file exists
            if not os.path.exists(self.output_file):
                logger.error(f"Error: JSON file {self.output_file} does not exist.")
                return False
            
            # Read JSON file
            logger.info(f"Reading test cases from {self.output_file}...")
            with open(self.output_file, 'r') as f:
                suites = json.load(f)
            
            # Process test cases
            test_cases_set = set()
            
            for suite_name, tests in suites.items():
                # Process suite name: take only the first part before any space
                clean_suite_name = suite_name.split()[0]
                
                # Process suite names that end with "/digit" or "/digit."
                import re
                if re.search(r'/\d+\.?$', clean_suite_name):
                    # Replace only the digit with "*", keeping the slash and dot (if any)
                    if clean_suite_name.endswith('.'):
                        # For names ending with "/digit."
                        clean_suite_name = re.sub(r'/\d+\.$', '/*.', clean_suite_name)
                    else:
                        # For names ending with just "/digit"
                        clean_suite_name = re.sub(r'/\d+$', '/*', clean_suite_name)
                
                for test in tests:
                    # Process test name: take only the first part before any space
                    clean_test = test.split()[0]
                    
                    # Combine suite name and test name
                    if clean_suite_name.endswith('.'):
                        full_test_case = f"{clean_suite_name}{clean_test}"
                    else:
                        full_test_case = f"{clean_suite_name}.{clean_test}"
                    
                    # Process test cases that end with "/digit"
                    if re.search(r'/\d+$', full_test_case):
                        # Replace only the digit with "*", keeping the slash
                        base_test_case = re.sub(r'/\d+$', '/*', full_test_case)
                        test_cases_set.add(base_test_case)
                    else:
                        test_cases_set.add(full_test_case)
            
            # Save to text file
            with open(self.test_cases_file, 'w') as f:
                for test_case in sorted(test_cases_set):
                    f.write(f"{test_case}\n")
            
            logger.info(f"Successfully processed and saved {len(test_cases_set)} test cases to {self.test_cases_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing test cases: {e}")
            return False
    
    def run_single_test(self, executable_path, test_case, profraw_dir, i, total):
        """Run a single test case and generate coverage data.
        
        Args:
            executable_path: Path to the test executable
            test_case: Test case to run
            profraw_dir: Directory to save profraw files
            i: Test index
            total: Total number of tests
            
        Returns:
            Tuple of (test_case, success, profraw_file, error_message)
        """
        # Create a safe filename for the profraw file
        safe_filename = test_case.replace('/', '_').replace('.', '_').replace('*', 'ALL')
        profraw_file = os.path.join(profraw_dir, f"{safe_filename}.profraw")
        
        # Set environment variables
        env = os.environ.copy()
        env["LLVM_PROFILE_FILE"] = profraw_file
        
        try:
            # Run the test
            logger.info(f"[{i}/{total}] Running test: {test_case}")
            result = subprocess.run(
                [executable_path, f"--gtest_filter={test_case}"],
                env=env,
                capture_output=True,
                text=True
            )
            
            # Check if the test passed
            if result.returncode == 0:
                if os.path.exists(profraw_file):
                    logger.info(f"✅ Test passed: {test_case}")
                    return (test_case, True, profraw_file, "")
                else:
                    logger.warning(f"⚠️ Test {test_case} passed but no coverage data was generated")
                    return (test_case, False, None, "No coverage data generated")
            else:
                logger.error(f"❌ Test failed: {test_case} (return code: {result.returncode})")
                # Remove the profraw file if the test failed
                if os.path.exists(profraw_file):
                    os.remove(profraw_file)
                return (test_case, False, None, f"Test failed with return code {result.returncode}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error running test {test_case}: {error_msg}")
            # Remove the profraw file if an error occurred
            if os.path.exists(profraw_file):
                os.remove(profraw_file)
            return (test_case, False, None, error_msg)
    
    def run_test_cases(self, executable_path: str, num_threads=None):
        """Run test cases from the processed text file and generate coverage data.
        
        Args:
            executable_path: Path to the compiled gtest executable
            num_threads: Number of threads to use (default: number of CPU cores)
            
        Returns:
            Number of successfully executed test cases
        """
        try:
            # Check if the test cases file exists
            if not os.path.exists(self.test_cases_file):
                logger.error(f"Error: Test cases file {self.test_cases_file} not found.")
                logger.error("Please run with --process flag first to generate the test cases file.")
                return 0
            
            # Create directory for profraw files
            profraw_dir = os.path.join(self.output_dir, "trace_file", "profraw")
            os.makedirs(profraw_dir, exist_ok=True)
            
            # Read test cases from file
            with open(self.test_cases_file, 'r') as f:
                test_cases = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Found {len(test_cases)} test cases to run.")
            
            # Set default number of threads if not specified
            if num_threads is None:
                import multiprocessing
                num_threads = multiprocessing.cpu_count()
            
            logger.info(f"Using {num_threads} threads to run tests.")
            
            # Initialize counters with thread-safe lock
            successful_tests = 0
            failed_tests = 0
            results_lock = Lock()
            
            # Run tests in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit all tasks
                future_to_test = {
                    executor.submit(self.run_single_test, executable_path, test_case, profraw_dir, i+1, len(test_cases)): test_case
                    for i, test_case in enumerate(test_cases)
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_test):
                    test_case, success, profraw_file, error_msg = future.result()
                    with results_lock:
                        if success:
                            successful_tests += 1
                        else:
                            failed_tests += 1
            
            logger.info("\n---- Test Execution Summary ----")
            logger.info(f"Total test cases: {len(test_cases)}")
            logger.info(f"Successfully executed: {successful_tests}")
            logger.info(f"Failed: {failed_tests}")
            logger.info(f"Coverage data files generated: {successful_tests}")
            logger.info(f"Coverage data directory: {profraw_dir}")
            
            return successful_tests
            
        except Exception as e:
            logger.error(f"Error running test cases: {e}")
            return 0

def get_test_parser_class(framework: TestFramework):
    """Get the appropriate test parser class based on the framework.
    
    Args:
        framework: The test framework to get parser for
        
    Returns:
        The appropriate test parser class
    """
    if framework == TestFramework.GTEST:
        return GTest
    else:
        raise ValueError(f"Unsupported test framework: {framework}")

# 添加公共函数用于将profraw文件转换为lcov格式
def convert_profraw_to_lcov(library_name, executable_path, num_threads=None):
    """Convert profraw files to lcov format for a given library.
    
    Args:
        library_name: Name of the library
        executable_path: Path to the executable used to generate profraw files
        num_threads: Number of threads to use (default: number of CPU cores)
        
    Returns:
        Tuple of (success_count, total_count) indicating how many files were processed successfully
    """
    # 设置目录路径
    profraw_dir = os.path.join("test_parsing", library_name, "trace_file", "profraw")
    profdata_dir = os.path.join("test_parsing", library_name, "trace_file", "profdata")
    lcov_dir = os.path.join("test_parsing", library_name, "trace_file", "lcov")
    
    # 检查profraw目录是否存在
    if not os.path.exists(profraw_dir):
        logger.error(f"Error: profraw directory not found: {profraw_dir}")
        return 0, 0
    
    # 创建必要的目录
    os.makedirs(profdata_dir, exist_ok=True)
    os.makedirs(lcov_dir, exist_ok=True)
    
    # 获取所有profraw文件
    profraw_files = [f for f in os.listdir(profraw_dir) if f.endswith('.profraw')]
    if not profraw_files:
        logger.error(f"No profraw files found in {profraw_dir}")
        return 0, 0
    
    # 设置默认线程数
    if num_threads is None:
        import multiprocessing
        num_threads = multiprocessing.cpu_count()
    
    logger.info(f"Found {len(profraw_files)} profraw files to process")
    logger.info(f"Using {num_threads} threads for conversion")
    
    # 用于多线程安全计数的锁和计数器
    success_count = 0
    results_lock = Lock()
    
    # 定义处理单个文件的函数
    def process_file(args):
        i, profraw_file = args
        logger.info(f"[{i}/{len(profraw_files)}] Processing {profraw_file}")
        
        # 设置文件路径
        profraw_path = os.path.join(profraw_dir, profraw_file)
        base_name = os.path.splitext(profraw_file)[0]
        profdata_path = os.path.join(profdata_dir, f"{base_name}.profdata")
        lcov_path = os.path.join(lcov_dir, f"{base_name}.lcov")
        
        try:
            # 步骤1: 将profraw文件转换为profdata
            logger.info(f"Converting {profraw_file} to profdata...")
            result = subprocess.run(
                ["llvm-profdata", "merge", "-sparse", profraw_path, "-o", profdata_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # 步骤2: 将profdata文件转换为lcov格式
            logger.info(f"Converting {base_name}.profdata to lcov format...")
            with open(lcov_path, 'w') as lcov_file:
                result = subprocess.run(
                    ["llvm-cov", "export", executable_path, "-instr-profile", profdata_path, "--format=lcov"],
                    stdout=lcov_file,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
            
            logger.info(f"✅ Successfully converted {profraw_file} to lcov format: {lcov_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error processing {profraw_file}: {e}")
            logger.error(f"Command stderr: {e.stderr}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Unexpected error processing {profraw_file}: {e}")
            return False
    
    # 使用线程池并行处理文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_file, (i+1, profraw_file)): profraw_file
            for i, profraw_file in enumerate(profraw_files)
        }
        
        # 处理结果
        for future in concurrent.futures.as_completed(future_to_file):
            profraw_file = future_to_file[future]
            try:
                success = future.result()
                with results_lock:
                    if success:
                        success_count += 1
            except Exception as e:
                logger.error(f"Error in thread processing {profraw_file}: {e}")
    
    logger.info("\n---- Conversion Summary ----")
    logger.info(f"Total profraw files: {len(profraw_files)}")
    logger.info(f"Successfully converted: {success_count}")
    logger.info(f"Failed: {len(profraw_files) - success_count}")
    logger.info(f"profdata directory: {profdata_dir}")
    logger.info(f"lcov directory: {lcov_dir}")
    
    return success_count, len(profraw_files)

def extract_frequent_code(library_name, min_execution=1, ut_path=None, language='c'):
    """从lcov文件中提取执行频次大于阈值的代码。
    
    Args:
        library_name: 库名称
        min_execution: 最小执行频次，默认为1
        ut_path: 单元测试代码路径，只提取该路径下的单元测试代码，默认为None，不过滤
        language: 目标语言，决定输出文件扩展名，默认为'c'
        
    Returns:
        提取的文件数量
    """
    # 设置lcov文件夹路径
    lcov_dir = os.path.join("test_parsing", library_name, "trace_file", "lcov")
    output_dir = os.path.join("test_parsing", library_name, "trace_file", "extracted_code")
    
    # 确定文件扩展名
    ext_map = {
        'c': '.c',
        'cpp': '.cpp',
        'c++': '.cpp',
        'python': '.py',
        'py': '.py',
        'java': '.java',
        'js': '.js',
        'go': '.go',
        'rust': '.rs'
    }
    file_ext = ext_map.get(language.lower(), '.txt')
    
    # 检查lcov目录是否存在
    if not os.path.exists(lcov_dir):
        logger.error(f"Error: lcov directory not found: {lcov_dir}")
        return 0
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有lcov文件
    lcov_files = [f for f in os.listdir(lcov_dir) if f.endswith('.lcov')]
    if not lcov_files:
        logger.error(f"No lcov files found in {lcov_dir}")
        return 0
    
    logger.info(f"Found {len(lcov_files)} lcov files to process")
    logger.info(f"Output file extension: {file_ext}")
    if ut_path:
        logger.info(f"Unit test path: {ut_path}")
        ut_path = os.path.abspath(ut_path)
    
    # 记录处理的文件计数
    processed_lcov_files = 0
    
    # 处理每个lcov文件
    for i, lcov_file in enumerate(lcov_files, 1):
        logger.info(f"[{i}/{len(lcov_files)}] Processing {lcov_file}")
        lcov_path = os.path.join(lcov_dir, lcov_file)
        
        # 确定输出文件名和扩展名
        base_name = os.path.splitext(lcov_file)[0]
        output_file = os.path.join(output_dir, f"{base_name}{file_ext}")
        
        # 用于存储所有提取的代码行
        all_executed_lines = []
        
        # 解析lcov文件
        current_file = None
        skip_until_end = False
        
        try:
            with open(lcov_path, 'r', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    
                    if line.startswith('TN:'):
                        continue  # 忽略测试名称
                        
                    elif line.startswith('SF:'):
                        # 新的源文件开始
                        file_path = line[3:].strip()
                        current_file = os.path.abspath(file_path)
                        
                        # 检查文件是否在指定单元测试路径下
                        if ut_path and not current_file.startswith(ut_path):
                            skip_until_end = True
                        else:
                            skip_until_end = False
                            
                    elif line.startswith('DA:'):
                        if current_file and not skip_until_end:
                            parts = line[3:].split(',')
                            if len(parts) == 2:
                                try:
                                    line_num = int(parts[0])
                                    exec_count = int(parts[1])
                                    if exec_count >= min_execution:
                                        # 尝试读取该行代码
                                        try:
                                            if os.path.exists(current_file) and os.access(current_file, os.R_OK):
                                                with open(current_file, 'r', errors='ignore') as src:
                                                    lines = src.readlines()
                                                    if 1 <= line_num <= len(lines):
                                                        code_line = lines[line_num - 1].rstrip()
                                                        # 只添加代码，不加注释
                                                        if not code_line.strip().startswith('//') and not code_line.strip().startswith('/*'):
                                                            all_executed_lines.append(f"{code_line}\n")
                                        except Exception as e:
                                            logger.warning(f"Error reading source file {current_file}: {e}")
                                except ValueError:
                                    continue
                    
                    elif line == 'end_of_record' or line == 'END_OF_RECORD':
                        current_file = None
                        skip_until_end = False
                        
        except Exception as e:
            logger.warning(f"Error processing lcov file {lcov_file}: {e}")
            continue
        
        # 如果提取到了代码行，写入输出文件
        if all_executed_lines:
            try:
                with open(output_file, 'w') as out:
                    # 不添加注释，直接写入代码
                    for line in all_executed_lines:
                        out.write(line)
                
                processed_lcov_files += 1
                logger.info(f"Generated code file: {output_file} with {len(all_executed_lines)} code lines")
            except Exception as e:
                logger.warning(f"Error writing to {output_file}: {e}")
        else:
            logger.warning(f"No code was extracted from {lcov_file}")
    
    # 打印总结信息
    logger.info(f"提取完成: 处理了 {processed_lcov_files}/{len(lcov_files)} 个lcov文件")
    logger.info(f"输出目录: {output_dir}")
    
    return processed_lcov_files

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Parse test suites from a test executable')
    parser.add_argument('--executable', '-e', help='Path to the test executable')
    parser.add_argument('--framework', '-f', choices=[f.value for f in TestFramework], 
                        help='Test framework to use')
    parser.add_argument('--library', '-l', required=True, help='Name of the library')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    parser.add_argument('--process', '-p', action='store_true', help='Process test cases after parsing')
    parser.add_argument('--run', '-r', action='store_true', help='Run test cases and generate coverage data')
    parser.add_argument('--threads', '-t', type=int, help='Number of threads to use for operations')
    parser.add_argument('--convert', '-c', action='store_true', help='Convert profraw files to lcov format')
    parser.add_argument('--extract', '-x', action='store_true', help='Extract frequently executed code')
    parser.add_argument('--ut-path', '-u', help='Path to unit test code, only extract code from this path')
    parser.add_argument('--min-exec', type=int, default=1, help='Minimum execution count for code extraction (default: 1)')
    parser.add_argument('--language', default='c', help='Target language for code extraction (default: c)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 如果指定了提取频繁执行的代码
        if args.extract:
            result = extract_frequent_code(args.library, args.min_exec, args.ut_path, args.language)
            if isinstance(result, tuple):
                total_extracted, file_issues = result
            return
            
        # 如果指定了转换profraw文件
        if args.convert:
            # 检查是否提供了可执行文件
            if not args.executable:
                logger.error("Error: --executable is required when using --convert")
                sys.exit(1)
            convert_profraw_to_lcov(args.library, args.executable, args.threads)
            return
        
        # 对于解析、处理和运行测试，需要验证必需参数
        if not args.executable:
            logger.error("Error: --executable is required for parsing, processing or running tests")
            sys.exit(1)
            
        if not args.framework:
            logger.error("Error: --framework is required for parsing, processing or running tests")
            sys.exit(1)
            
        framework = TestFramework(args.framework)
        parser_class = get_test_parser_class(framework)
        test_parser = parser_class(args.library)
        
        # 解析所有测试用例
        suites = test_parser.parse_tests(args.executable)
        
        if suites:
            if args.output:
                test_parser.output_file = args.output
            
            # 保存到JSON
            test_parser.save_to_json(suites)
            
            # 如果指定了处理测试用例
            if args.process:
                test_parser.process_test_cases()
                
            # 如果指定了运行测试用例
            if args.run:
                test_parser.run_test_cases(args.executable, args.threads)
        else:
            logger.error("No test suites found or error occurred")
            sys.exit(1)
            
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Supported frameworks: {[f.value for f in TestFramework]}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
'''
# 步骤1: 解析测试用例
python test_parser.py --library $LIBRARY --executable $EXECUTABLE --framework $FRAMEWORK

# 步骤2: 处理测试用例
python test_parser.py --library $LIBRARY --executable $EXECUTABLE --framework $FRAMEWORK --process

# 步骤3: 运行测试用例并生成覆盖率数据
python test_parser.py --library $LIBRARY --executable $EXECUTABLE --framework $FRAMEWORK --run

# 步骤4: 转换profraw文件到lcov格式
python test_parser.py --library $LIBRARY --executable $EXECUTABLE --convert

# 步骤5: 提取频繁执行的代码
python test_parser.py --library $LIBRARY --extract --ut-path $UT_PATH --language $LANGUAGE
'''