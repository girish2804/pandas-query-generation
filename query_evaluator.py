import os
import pandas as pd
from groq import Groq
import re
import warnings
import json
import argparse
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

class PandasQueryEvaluator:
    def __init__(self, api_key: str, model: str = "deepseek-r1-distill-llama-70b"):
        """Initialize the query evaluator with Groq API settings"""
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)
        self.df = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_filename = f"query_evaluation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_filename}")
        
    def load_data(self, filepath: str):
        """Load and prepare the household power consumption dataset"""
        try:
            self.logger.info(f"Loading data from {filepath}")
            self.df = pd.read_csv(filepath,
                                sep=';',
                                parse_dates={'datetime': ['Date', 'Time']},
                                infer_datetime_format=True,
                                na_values=['?'],
                                low_memory=False)
            self.df = self.df.dropna()
            self.df['Global_active_power'] = self.df['Global_active_power'].astype(float)
            self.df = self.df.set_index('datetime')
            self.logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False
            
    def load_queries_from_file(self, filepath: str) -> List[str]:
        """Load queries from a text file (one query per line)"""
        try:
            self.logger.info(f"Loading queries from {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            self.logger.info(f"Loaded {len(queries)} queries from file")
            return queries
        except Exception as e:
            self.logger.error(f"Error loading queries from file: {str(e)}")
            return []
            
    def get_default_queries(self) -> List[str]:
        """Return default set of queries"""
        return [
            'What was the average active power consumption in March 2007?',
            'What hour of the day had the highest power usage on Christmas 2006?',
            'Compare energy usage (Global_active_power) on weekdays vs weekends.',
            'Find days where energy consumption exceeded 5 kWh.',
            'Plot the energy usage trend for the first week of January 2007.',
            'Find the average voltage for each day of the first week of February 2007.',
            'What is the correlation between global active power and sub-metering values?'
        ]
        
    def get_few_shot_prompt(self) -> str:
        """Return the few-shot prompt template"""
        return """You are a pandas code generator for energy dataset analysis. The dataset has datetime index and columns: Global_active_power, Voltage, Sub_metering_1, Sub_metering_2, Sub_metering_3.
Examples:
Query: Show first 3 rows
Steps: Use head() function with parameter 3
Code: df.head(3)
Query: Max voltage in January 2007
Steps: Filter for 2007-01, then find max of Voltage column
Code: df.loc['2007-01']['Voltage'].max()
Query: Average power by month in 2007
Steps: Filter for 2007, group by month, calculate mean of Global_active_power
Code: df.loc['2007'].groupby(df.loc['2007'].index.month)['Global_active_power'].mean()
IMPORTANT: You must follow this EXACT format for your response:
Steps: [your brief reasoning here]
Code: [your pandas code here]
Do NOT provide alternative solutions. Do NOT add explanations after the code. Do NOT use markdown formatting.
Query: {user_query}"""

    def query_gen(self, prompt: str, nl_query: str) -> str:
        """Generate pandas code using Groq API"""
        try:
            final_prompt = prompt.format(user_query=nl_query)
            self.logger.info(f"Generating code for query: {nl_query}")
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": final_prompt,
                    }
                ],
                temperature=0.6,
                model=self.model,
                stream=False,
                max_completion_tokens=2048,
                top_p=0.95
            )
            out = chat_completion.choices[0].message.content
            self.logger.info("Code generation successful")
            return out
        except Exception as e:
            self.logger.error(f"Error generating code: {str(e)}")
            return ""

    def extract_code_regex(self, llm_response: str) -> Optional[str]:
        """Use regex to find code after 'Code:' """
        pattern = r'Code:\s*(.+?)(?:\n|$)'
        match = re.search(pattern, llm_response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def evaluate_query(self, code_str: str) -> Tuple[Any, Optional[str]]:
        """Safely evaluate pandas code and return result"""
        try:
            # Create a safe namespace for eval
            namespace = {'df': self.df, 'pd': pd}
            result = eval(code_str, {"__builtins__": {}}, namespace)
            return result, None
        except Exception as e:
            return None, str(e)

    def format_result(self, result: Any, error: Optional[str] = None) -> str:
        """Format result for display and logging"""
        if error:
            return f"Error: {error}"
        
        if isinstance(result, pd.DataFrame):
            return f"DataFrame Result:\n{result.to_string()}"
        elif isinstance(result, pd.Series):
            return f"Series Result:\n{result.to_string()}"
        elif hasattr(result, '__iter__') and not isinstance(result, str):
            items = '\n'.join([f"  {item}" for item in result])
            return f"Iterable Result:\n{items}"
        else:
            return f"Result: {result}"

    def process_queries(self, queries: List[str]) -> List[dict]:
        """Process all queries and return results"""
        results = []
        prompt = self.get_few_shot_prompt()
        
        self.logger.info(f"Processing {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            self.logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
            
            # Generate code
            llm_output = self.query_gen(prompt, query)
            
            # Extract code
            code = self.extract_code_regex(llm_output)
            
            # Evaluate code
            if code:
                result, error = self.evaluate_query(code)
                formatted_result = self.format_result(result, error)
                
                query_result = {
                    'query_id': i + 1,
                    'query': query,
                    'llm_output': llm_output,
                    'extracted_code': code,
                    'result': formatted_result,
                    'success': error is None,
                    'error': error
                }
                
                if error:
                    self.logger.error(f"Query {i+1} failed: {error}")
                else:
                    self.logger.info(f"Query {i+1} executed successfully")
            else:
                query_result = {
                    'query_id': i + 1,
                    'query': query,
                    'llm_output': llm_output,
                    'extracted_code': None,
                    'result': "Code extraction failed",
                    'success': False,
                    'error': "Code extraction failed"
                }
                self.logger.error(f"Query {i+1}: Code extraction failed")
            
            results.append(query_result)
            
        return results

    def save_results(self, results: List[dict], output_file: str):
        """Save results to a JSON file"""
        try:
            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{output_file}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Results saved to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return None

    def print_summary(self, results: List[dict]):
        """Print a summary of results"""
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"Total queries: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {successful/total*100:.1f}%")
        
        # Show failed queries
        failed = [r for r in results if not r['success']]
        if failed:
            print(f"\nFailed queries:")
            for r in failed:
                print(f"  {r['query_id']}. {r['query']}")
                print(f"     Error: {r['error']}")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate pandas queries using LLM')
    parser.add_argument('--data-file', default='household_power_consumption.csv',
                       help='Path to the household power consumption CSV file')
    parser.add_argument('--query-file', default=None,
                       help='Path to file containing queries (one per line)')
    parser.add_argument('--output-file', default='query_results',
                       help='Base name for output files')
    parser.add_argument('--api-key', default=None,
                       help='Groq API key (can also be set via GROQ_API_KEY env var)')
    parser.add_argument('--model', default='deepseek-r1-distill-llama-70b',
                       help='Groq model to use')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('GROQ_API_KEY')
    if not api_key:
        print("Error: Please provide API key via --api-key or GROQ_API_KEY environment variable")
        return
    
    # Initialize evaluator
    evaluator = PandasQueryEvaluator(api_key, args.model)
    
    # Load data
    if not evaluator.load_data(args.data_file):
        print(f"Error: Could not load data from {args.data_file}")
        return
    
    # Load queries
    if args.query_file:
        queries = evaluator.load_queries_from_file(args.query_file)
        if not queries:
            print(f"Error: Could not load queries from {args.query_file}")
            return
    else:
        queries = evaluator.get_default_queries()
        print("Using default queries")
    
    # Process queries
    results = evaluator.process_queries(queries)
    
    # Save results
    output_file = evaluator.save_results(results, args.output_file)
    if output_file:
        print(f"Results saved to {output_file}")
    
    # Print summary
    evaluator.print_summary(results)

if __name__ == "__main__":
    main()