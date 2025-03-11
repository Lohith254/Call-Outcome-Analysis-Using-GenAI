import unittest
from sentiment_analysis import analyze_sentiment

class TestSentimentAnalysis(unittest.TestCase):
    def test_positive_sentiment(self):
        text_data = {"test1.txt": "Thank you for your help! The issue is resolved."}
        result = analyze_sentiment(text_data)
        self.assertEqual(result["test1.txt"], "Positive")
    
    def test_negative_sentiment(self):
        text_data = {"test2.txt": "This is frustrating. No one is helping me."}
        result = analyze_sentiment(text_data)
        self.assertEqual(result["test2.txt"], "Negative")
    
    def test_neutral_sentiment(self):
        text_data = {"test3.txt": "I need to check my balance."}
        result = analyze_sentiment(text_data)
        self.assertEqual(result["test3.txt"], "Neutral")
    
if __name__ == "__main__":
    unittest.main()