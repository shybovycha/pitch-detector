import sys
from FileAnalyzer import FileAnalyzer

if __name__ == '__main__':
    analyzer = FileAnalyzer(sys.argv[1])
    analyzer.analyze()