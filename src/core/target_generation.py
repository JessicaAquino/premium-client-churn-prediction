import duckdb
import config.logger_config as log

@log.process_log
def target_generation(name :str, output: str="csv"):
    print("Hi!")