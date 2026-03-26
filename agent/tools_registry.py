from tools.load_data import load_data
from tools.correlation import correlation_analysis
from tools.regression import regression_analysis
from tools.search_literature import search_literature
from tools.generate_report import generate_report

TOOLS = {
    "load_data": load_data,
    "correlation_analysis": correlation_analysis,
    "regression_analysis": regression_analysis,
    "search_literature": search_literature,
    "generate_report": generate_report,
}