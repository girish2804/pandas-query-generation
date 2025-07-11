# Pandas Query Evaluator

A tool that uses Large Language Models to generate pandas queries from natural language and evaluates their execution on energy consumption datasets.

## Setup

1. **Extract the dataset**:
```bash
unzip "individual+household+electric+power+consumption.zip"
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set API key**:
```bash
export GROQ_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage
```bash
python query_evaluator.py --data-file household_power_consumption.txt
```

### With Custom Queries
```bash
python query_evaluator.py \
    --data-file household_power_consumption.txt \
    --query-file my_queries.txt \
    --output-file my_results
```

## Input Files

### Query File Format
Create a text file with one query per line:
```
What was the average active power consumption in March 2007?
What hour of the day had the highest power usage on Christmas 2006?
# Comments start with # and are ignored
Find days where energy consumption exceeded 5 kWh.
```

## Output Files

- **Results**: `query_results.json` - Complete execution results
- **Logs**: `query_evaluation.log` - Detailed execution logs

## Default Queries

If no query file is provided, the tool uses these default queries:
1. What was the average active power consumption in March 2007?
2. What hour of the day had the highest power usage on Christmas 2006?
3. Compare energy usage (Global_active_power) on weekdays vs weekends.
4. Find days where energy consumption exceeded 5 kWh.
5. Plot the energy usage trend for the first week of January 2007.
6. Find the average voltage for each day of the first week of February 2007.
7. What is the correlation between global active power and sub-metering values?
