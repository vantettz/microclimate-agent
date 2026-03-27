from tools.load_data import load_data
from tools.correlation import correlation_analysis
from tools.regression import regression_analysis
from tools.generate_report import generate_report
from tools.vision_analysis import vision_analysis
from tools.noise_analysis import noise_analysis

TOOLS = {
    "load_data": load_data,
    "correlation_analysis": correlation_analysis,
    "regression_analysis": regression_analysis,
    "generate_report": generate_report,
    "vision_analysis": vision_analysis,
    "noise_analysis": noise_analysis,
}